import os
import asyncio
import pandas as pd
import time
from tqdm import tqdm
# Lazy-import heavy modules when needed to avoid unnecessary model loading
from config_manager import get_config
from concurrent.futures import ProcessPoolExecutor, as_completed
from logger import get_logger, log_time
logger = get_logger("tiktok_feature_extractor")

class TikTokFeatureExtractor:
    """Main controller for TikTok video feature extraction with GPT4O integration."""
    def __init__(self, config=None, lightweight: bool = False):
        """
        Initialize the feature extractor with configuration
        
        Args:
            config: Configuration manager instance or None to use default
        """
        self.logger = get_logger("TikTokFeatureExtractor")
        self.config = config or get_config()
        self.lightweight = bool(lightweight)

        if not self.lightweight:
            # Initialize processors with their respective configurations
            from audio_processor import AudioProcessor  # defer import
            from video_processor import VideoProcessor  # defer import
            from frame_analyzer import FrameAnalyzer    # defer import
            from multimodal_extractor import MultimodalExtractor  # defer import

            self.audio_processor = AudioProcessor(self.config.get_audio_config())
            self.video_processor = VideoProcessor(self.config.get_visual_config())
            self.frame_analyzer = FrameAnalyzer(self.config.get_multimodal_config())

            # Get complete multimodal config including the new control options
            multimodal_config = self.config.get_multimodal_config()
            self.multimodal_extractor = MultimodalExtractor({
                "yolo": self.config.get_model_config("yolo"),
                "blip": self.config.get_model_config("blip"),
                "clip": self.config.get_model_config("clip"),
                "gpt4o": self.config.get_model_config("gpt4o"),
                "embedding_analyzer": self.config.get_embedding_analyzer_config(),
                # Add the new control options
                "enable_batch_image_upload": multimodal_config.get("enable_batch_image_upload", True),
                "enable_embedding_analysis": multimodal_config.get("enable_embedding_analysis", True),
                "use_embedding_analyzer": multimodal_config.get("use_embedding_analyzer", True),
                "representative_frames": multimodal_config.get("representative_frames", 10),
                "batch_size": multimodal_config.get("batch_size", 16),
            })
        else:
            # Defer heavy initializations
            self.audio_processor = None
            self.video_processor = None
            self.frame_analyzer = None
            self.multimodal_extractor = None
        
        # Get processing configuration
        self.processing_config = self.config.get_processing_config()
        self.output_config = self.config.get_output_config()

    async def analyze_video(self, video_obj):
        """Asynchronously analyze a TikTok video object.

        - If a local file exists (downloaded video), run the full feature extraction pipeline.
        - Otherwise, return a lightweight metadata-only analysis so the batch flow can proceed.
        """
        try:
            download_path = getattr(video_obj, 'download_path', '') or ''
            creator_username = getattr(video_obj, 'creator_username', 'unknown_creator')
            video_id = getattr(video_obj, 'video_id', 'unknown_video')

            # Prefer running the full pipeline when a local file is available
            if (not self.lightweight) and isinstance(download_path, str) and download_path and os.path.exists(download_path):
                # Choose an output directory for this video
                base_output_dir = self.output_config.get('base_output_dir', 'data/analysis')
                video_output_dir = os.path.join(base_output_dir, creator_username, video_id)
                os.makedirs(video_output_dir, exist_ok=True)

                # Collect API metadata to enrich GPT prompts and final outputs
                try:
                    description = getattr(video_obj, 'video_description', '') or getattr(video_obj, 'desc', '') or ''
                    hashtags = getattr(video_obj, 'hashtags', [])
                    if isinstance(hashtags, list):
                        try:
                            hashtags = [getattr(h, 'name', str(h)) for h in hashtags]
                        except Exception:
                            hashtags = [str(h) for h in hashtags]
                    music_title = getattr(video_obj, 'music_title', '')
                    music_author = getattr(video_obj, 'music_author', '')
                    is_ad = bool(getattr(video_obj, 'is_ad', False) or getattr(video_obj, 'isAd', False))
                    is_commerce = bool(getattr(video_obj, 'is_commerce', False) or getattr(video_obj, 'isCommerce', False))
                    ad_authorization = bool(getattr(video_obj, 'ad_authorization', False) or getattr(video_obj, 'adAuthorization', False))
                    likes = int(getattr(video_obj, 'likes', 0) or 0)
                    comments = int(getattr(video_obj, 'comments', 0) or 0)
                    shares = int(getattr(video_obj, 'shares', 0) or 0)
                    views = int(getattr(video_obj, 'views', 0) or 0)
                    duration = int(getattr(video_obj, 'duration', 0) or 0)
                    created_time = getattr(video_obj, 'created_time', None)
                    if hasattr(created_time, 'isoformat'):
                        created_time = created_time.isoformat()
                    api_metadata = {
                        'creator_username': creator_username,
                        'video_id': video_id,
                        'video_description': description,
                        'hashtags': hashtags,
                        'music_title': music_title,
                        'music_author': music_author,
                        'is_ad': is_ad,
                        'is_commerce': is_commerce,
                        'ad_authorization': ad_authorization,
                        'likes': likes,
                        'comments': comments,
                        'shares': shares,
                        'views': views,
                        'duration': duration,
                        'created_time': created_time,
                    }
                except Exception:
                    api_metadata = {
                        'creator_username': creator_username,
                        'video_id': video_id,
                    }

                loop = asyncio.get_running_loop()
                features = await loop.run_in_executor(
                    None, self.extract_video_features, download_path, video_output_dir, api_metadata
                )

                return {
                    'mode': 'full-pipeline',
                    'features': features,
                    'analysis_metadata': {
                        'creator_username': creator_username,
                        'video_id': video_id,
                        'used_local_file': True,
                    }
                }

            # Fallback: metadata-only analysis (no local video file)
            description = getattr(video_obj, 'video_description', '') or getattr(video_obj, 'desc', '') or ''
            hashtags = getattr(video_obj, 'hashtags', [])
            if isinstance(hashtags, list):
                try:
                    hashtags = [getattr(h, 'name', str(h)) for h in hashtags]
                except Exception:
                    hashtags = [str(h) for h in hashtags]

            # Extract ad/commercial/music fields directly for prompt/CSV enrichment
            music_title = getattr(video_obj, 'music_title', '')
            music_author = getattr(video_obj, 'music_author', '')
            is_ad = bool(getattr(video_obj, 'is_ad', False)) or bool(getattr(video_obj, 'isAd', False))
            is_commerce = bool(getattr(video_obj, 'is_commerce', False)) or bool(getattr(video_obj, 'isCommerce', False))
            ad_authorization = bool(getattr(video_obj, 'ad_authorization', False)) or bool(getattr(video_obj, 'adAuthorization', False))
            likes = int(getattr(video_obj, 'likes', 0) or 0)
            comments = int(getattr(video_obj, 'comments', 0) or 0)
            shares = int(getattr(video_obj, 'shares', 0) or 0)
            views = int(getattr(video_obj, 'views', 0) or 0)
            duration = int(getattr(video_obj, 'duration', 0) or 0)
            created_time = getattr(video_obj, 'created_time', None)
            if hasattr(created_time, 'isoformat'):
                created_time = created_time.isoformat()

            return {
                'mode': 'metadata-only',
                'features': {
                    'video_description': description,
                    'primary_category': 'Unknown',
                    'secondary_category': 'Unknown',
                    'tertiary_category': 'Unknown',
                    'hashtags': hashtags,
                    'creator_username': creator_username,
                    'video_id': video_id,
                    'music_title': music_title,
                    'music_author': music_author,
                    'is_ad': is_ad,
                    'is_commerce': is_commerce,
                    'ad_authorization': ad_authorization,
                    'likes': likes,
                    'comments': comments,
                    'shares': shares,
                    'views': views,
                    'duration': duration,
                    'created_time': created_time
                },
                'analysis_metadata': {
                    'creator_username': creator_username,
                    'video_id': video_id,
                    'used_local_file': False,
                }
            }

        except Exception as e:
            self.logger.error(f"analyze_video failed for video: {getattr(video_obj, 'video_id', 'unknown')} - {e}")
            return {
                'mode': 'error',
                'features': {},
                'analysis_metadata': {
                    'error': str(e),
                    'creator_username': getattr(video_obj, 'creator_username', 'unknown_creator'),
                    'video_id': getattr(video_obj, 'video_id', 'unknown_video'),
                }
            }

    @log_time("extract_video_features")
    def extract_video_features(self, video_path, output_dir, api_metadata: dict = None):
        """Extract comprehensive features from a single video with GPT4O analysis."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        features = {
            'video_name': video_name,
            'video_path': video_path,
            'duration': 0,
            'width': 0,
            'height': 0,
            'frame_rate': 0,
            'file_size': 0,
            'has_speech': False,
            'speech_text': '',
            'keyframe_count': 0,
            'representative_frame_count': 0,
            'video_description': '',
            'primary_category': '',
            'secondary_category': '',
            'tertiary_category': '',
            'processing_time': 0,
            'step_timings': {}
        }
        step_timings = {}
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            total_start_time = time.time()
            self.logger.info(f"Processing video: {video_path}")
            
            t0 = time.time()
            # 1. Extract video metadata (always needed for basic info)
            metadata = self.video_processor.get_video_metadata(video_path)
            step_timings['get_video_metadata'] = time.time() - t0
            features.update(metadata)
            
            t0 = time.time()
            # 2. Process audio
            audio_results = self.audio_processor.process_audio_for_video(video_path, output_dir)
            step_timings['process_audio_for_video'] = time.time() - t0
            
            # Clean audio results for CSV
            cleaned_audio_results = self._clean_audio_results_for_csv(audio_results)
            features.update(cleaned_audio_results)

            # 2.5. Extract speech_segments for smart frame extraction
            speech_segments = None
            if isinstance(audio_results, dict):
                speech_segments = (
                    audio_results.get('structured_data', {})
                    .get('speech_analysis', {})
                    .get('segments', [])
                )

            t0 = time.time()
            # 3. Extract keyframes (only if visual processing is enabled)
            if self.config.is_feature_enabled('visual'):
                self.logger.info("Keyframe extraction started")
                keyframes_dir, keyframe_count = self.video_processor.extract_keyframes(video_path, output_dir, speech_segments=speech_segments)
                features['keyframe_count'] = keyframe_count
            else:
                self.logger.info(f"Visual processing disabled for {video_name}")
                keyframes_dir = output_dir
                keyframe_count = 0
                features['keyframe_count'] = 0
            step_timings['extract_keyframes'] = time.time() - t0

            if keyframe_count > 0:
                t0 = time.time()
                # 4. Filter similar keyframes
                self.logger.info("Frame analysis started")
                filtered_frames = self.frame_analyzer.filter_similar_keyframes(keyframes_dir, video_name)
                step_timings['filter_similar_keyframes'] = time.time() - t0
                
                t0 = time.time()
                # 5. Select representative frames
                representative_frames = self.frame_analyzer.get_representative_frames(keyframes_dir, video_name)
                features['representative_frame_count'] = len(representative_frames)
                step_timings['get_representative_frames'] = time.time() - t0
                
                t0 = time.time()
                # 6. Save representative frames
                new_representative_frames = self.frame_analyzer.save_representative_frames(representative_frames, output_dir, video_name)
                self.frame_analyzer.save_representative_timestamps(output_dir, video_name)
                step_timings['save_representative_frames'] = time.time() - t0

                t0 = time.time()
                # 8. Extract GPT4O features (only if enabled)
                multimodal_config = self.config.get_multimodal_config()
                use_gpt4o = multimodal_config.get('use_gpt4o', False)
                
                if use_gpt4o:
                    self.logger.info("Multimodal feature extraction started (GPT4O enabled)")
                    # Prepare speech rate analysis data for GPT4O
                    speech_rate_data = {}
                    if 'speech_rate_analysis' in audio_results:
                        speech_rate_data = audio_results['speech_rate_analysis']
                    
                    gpt4o_results = self.multimodal_extractor.extract_gpt4o_features(
                        new_representative_frames, 
                        video_name, 
                        audio_transcript=audio_results.get('speech_text', ''),
                        speech_rate_analysis=speech_rate_data,
                        api_metadata=api_metadata or {}
                    )
                    step_timings['extract_gpt4o_features'] = time.time() - t0
                
                    # Update features with cleaned GPT4O results
                    if gpt4o_results:
                        cleaned_gpt4o_results = self._clean_gpt4o_results_for_csv(gpt4o_results)
                        features.update(cleaned_gpt4o_results)
                else:
                    self.logger.info("GPT4O analysis disabled - skipping multimodal feature extraction")
                    step_timings['extract_gpt4o_features'] = 0.0
                    # Set default values for features that would come from GPT4O
                    features.update({
                        'video_description': '',
                        'primary_category': 'Unknown',
                        'secondary_category': 'Unknown',
                        'tertiary_category': 'Unknown'
                    })
            
            # After feature extraction, build final aggregated outputs (API + GPT)
            try:
                final_output = {}
                api_md = api_metadata or {}
                # Core identity
                final_output['creator_username'] = api_md.get('creator_username', '')
                final_output['video_id'] = api_md.get('video_id', '')
                # Source/API info
                final_output['source_video_description'] = api_md.get('video_description', '')
                final_output['hashtags'] = api_md.get('hashtags', [])
                final_output['music_title'] = api_md.get('music_title', '')
                final_output['music_author'] = api_md.get('music_author', '')
                final_output['is_ad'] = bool(api_md.get('is_ad', False))
                final_output['is_commerce'] = bool(api_md.get('is_commerce', False))
                final_output['ad_authorization'] = bool(api_md.get('ad_authorization', False))
                final_output['likes'] = int(api_md.get('likes', 0) or 0)
                final_output['comments'] = int(api_md.get('comments', 0) or 0)
                final_output['shares'] = int(api_md.get('shares', 0) or 0)
                final_output['views'] = int(api_md.get('views', 0) or 0)
                final_output['duration'] = int(api_md.get('duration', 0) or 0)
                final_output['created_time'] = api_md.get('created_time', '')
                # Audio basics
                final_output['speech_text'] = features.get('speech_text', '')
                final_output['speech_confidence'] = features.get('speech_confidence', 0.0)
                final_output['speech_rate_wpm'] = features.get('speech_rate_wpm', 0)
                final_output['speech_rate_category'] = features.get('speech_rate_category', 'unknown')
                # AI analysis
                final_output['ai_video_description'] = features.get('video_description', '')
                final_output['primary_category'] = features.get('primary_category', 'Unknown')
                final_output['secondary_category'] = features.get('secondary_category', 'Unknown')
                final_output['tertiary_category'] = features.get('tertiary_category', 'Unknown')
                # Creator analysis
                for key in [
                    'creator_gender','creator_age','creator_physical_appearance','creator_body_type',
                    'creator_hair_color','creator_hair_type','creator_persona','creator_charisma_authenticity',
                    'creator_economic_status']:
                    final_output[key] = features.get(key, '')
                # Content analysis
                for key in [
                    'content_niche_specialty','content_format','content_overall_tone','content_energy_levels',
                    'content_visual_presentation','content_product_presentation','content_video_quality',
                    'content_video_scene','content_video_background','content_pacing','content_speech_rate',
                    'content_face_visibility','content_tiktok_effects']:
                    final_output[key] = features.get(key, '')
                # Business analysis
                for key in [
                    'business_commercial_intent','business_product_placement','business_target_audience',
                    'business_target_audience_fit','business_engagement_drivers','business_trend_adoption',
                    'business_branding_integration','business_call_to_actions','business_has_discount',
                    'business_has_limited_time_discount','business_has_exclusive_discount']:
                    final_output[key] = features.get(key, '')
                # Product analysis
                for key in [
                    'product_key_products','product_brands_mentioned','product_selling_points',
                    'product_content_creative_ideas','product_emotional_value']:
                    final_output[key] = features.get(key, '')
                # User analysis
                for key in ['user_demographics','user_preferences','user_pain_points','user_goals']:
                    final_output[key] = features.get(key, '')
                # Performance analysis
                for key in [
                    'performance_content_style','performance_engagement_potential','performance_trend_alignment',
                    'performance_compliance_issues']:
                    final_output[key] = features.get(key, '')
                # Optimization suggestions
                for key in [
                    'optimization_content_suggestions','optimization_product_placement_suggestions',
                    'optimization_content_creative_suggestions','optimization_emotional_value_suggestions',
                    'optimization_compliance_suggestions']:
                    final_output[key] = features.get(key, '')

                # Persist final outputs
                try:
                    import json as _json
                    import csv as _csv
                    final_json_path = os.path.join(output_dir, 'final_analysis.json')
                    with open(final_json_path, 'w', encoding='utf-8') as f:
                        _json.dump(final_output, f, ensure_ascii=False, indent=2)

                    # Write CSV with same fields
                    final_csv_path = os.path.join(output_dir, 'final_analysis.csv')
                    # Normalize list fields for CSV
                    csv_row = {}
                    for k, v in final_output.items():
                        if isinstance(v, list):
                            csv_row[k] = ','.join([str(x) for x in v])
                        else:
                            csv_row[k] = v
                    import pandas as _pd
                    _pd.DataFrame([csv_row]).to_csv(final_csv_path, index=False, encoding='utf-8')

                    # Cleanup: remove all non-final artifacts within this video's output directory
                    try:
                        import shutil
                        keep_names = {'final_analysis.json', 'final_analysis.csv'}
                        for name in os.listdir(output_dir):
                            if name in keep_names:
                                continue
                            path = os.path.join(output_dir, name)
                            try:
                                if os.path.isdir(path):
                                    shutil.rmtree(path, ignore_errors=True)
                                else:
                                    os.remove(path)
                            except Exception:
                                # Best-effort cleanup
                                pass
                    except Exception:
                        # Best-effort cleanup
                        pass

                    self.logger.info(f"Final analysis saved: {final_json_path}, {final_csv_path}")
                except Exception as _e:
                    self.logger.warning(f"Failed to save final per-video outputs: {_e}")
            except Exception as _e:
                self.logger.warning(f"Final aggregation failed: {_e}")

            # Calculate total time
            total_time = time.time() - total_start_time
            features['processing_time'] = total_time
            
            # Clean step timings for CSV - convert to string summary
            step_timings_summary = []
            for step, time_taken in step_timings.items():
                if time_taken > 0:
                    step_timings_summary.append(f"{step}: {time_taken:.2f}s")
            features['step_timings'] = '; '.join(step_timings_summary)
            
        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {e}")
            import traceback
            traceback.print_exc()
            
        return features

    def _clean_gpt4o_results_for_csv(self, gpt4o_results):
        """Clean and flatten comprehensive GPT4O results for CSV output, extracting all information from the new unified JSON structure."""
        cleaned_results = {}
        
        # Extract basic video analysis
        cleaned_results.update({
            'video_description': gpt4o_results.get('video_description', ''),
            'primary_category': gpt4o_results.get('primary_category', 'Unknown'),
            'secondary_category': gpt4o_results.get('secondary_category', 'Unknown'),
            'tertiary_category': gpt4o_results.get('tertiary_category', 'Unknown'),
        })
        
        # Extract creator analysis
        creator_analysis = gpt4o_results.get('creator_analysis', {})
        if isinstance(creator_analysis, dict):
            cleaned_results.update({
                'creator_gender': creator_analysis.get('gender', 'unknown'),
                'creator_age': creator_analysis.get('age', 'unknown'),
                'creator_physical_appearance': creator_analysis.get('physical_appearance', 'unknown'),
                'creator_body_type': creator_analysis.get('body_type', 'unknown'),
                'creator_hair_color': creator_analysis.get('hair_color', 'unknown'),
                'creator_hair_type': creator_analysis.get('hair_type', 'unknown'),
                'creator_persona': ', '.join(creator_analysis.get('persona', [])[:3]) if isinstance(creator_analysis.get('persona'), list) else str(creator_analysis.get('persona', '')),
                'creator_charisma_authenticity': creator_analysis.get('charisma_authenticity', 'unknown'),
                'creator_economic_status': creator_analysis.get('economic_status', 'unknown'),
            })
        
        # Extract content analysis
        content_analysis = gpt4o_results.get('content_analysis', {})
        if isinstance(content_analysis, dict):
            cleaned_results.update({
                'content_niche_specialty': ', '.join(content_analysis.get('niche_specialty', [])[:3]) if isinstance(content_analysis.get('niche_specialty'), list) else str(content_analysis.get('niche_specialty', '')),
                'content_format': ', '.join(content_analysis.get('content_format', [])[:3]) if isinstance(content_analysis.get('content_format'), list) else str(content_analysis.get('content_format', '')),
                'content_overall_tone': ', '.join(content_analysis.get('overall_tone', [])[:3]) if isinstance(content_analysis.get('overall_tone'), list) else str(content_analysis.get('overall_tone', '')),
                'content_energy_levels': content_analysis.get('energy_levels', 'unknown'),
                'content_visual_presentation': ', '.join(content_analysis.get('visual_presentation', [])[:3]) if isinstance(content_analysis.get('visual_presentation'), list) else str(content_analysis.get('visual_presentation', '')),
                'content_product_presentation': ', '.join(content_analysis.get('product_presentation', [])[:3]) if isinstance(content_analysis.get('product_presentation'), list) else str(content_analysis.get('product_presentation', '')),
                'content_video_quality': content_analysis.get('video_quality', 'unknown'),
                'content_video_scene': content_analysis.get('video_scene', 'unknown'),
                'content_video_background': content_analysis.get('video_background', 'unknown'),
                'content_pacing': content_analysis.get('pacing', 'unknown'),
                'content_speech_rate': content_analysis.get('speech_rate', 'unknown'),
                'content_face_visibility': content_analysis.get('face_visibility', 'unknown'),
                'content_tiktok_effects': ', '.join(content_analysis.get('tiktok_effects', [])[:3]) if isinstance(content_analysis.get('tiktok_effects'), list) else str(content_analysis.get('tiktok_effects', '')),
            })
        
        # Extract business analysis
        business_analysis = gpt4o_results.get('business_analysis', {})
        if isinstance(business_analysis, dict):
            cleaned_results.update({
                'business_commercial_intent': business_analysis.get('commercial_intent', 'unknown'),
                'business_product_placement': business_analysis.get('product_placement', 'unknown'),
                'business_target_audience': business_analysis.get('target_audience', ''),
                'business_target_audience_fit': business_analysis.get('target_audience_fit', 'unknown'),
                'business_engagement_drivers': ', '.join(business_analysis.get('engagement_drivers', [])[:3]) if isinstance(business_analysis.get('engagement_drivers'), list) else str(business_analysis.get('engagement_drivers', '')),
                'business_trend_adoption': ', '.join(business_analysis.get('trend_adoption', [])[:3]) if isinstance(business_analysis.get('trend_adoption'), list) else str(business_analysis.get('trend_adoption', '')),
                'business_branding_integration': business_analysis.get('branding_integration', 'unknown'),
                'business_call_to_actions': ', '.join(business_analysis.get('call_to_actions', [])[:3]) if isinstance(business_analysis.get('call_to_actions'), list) else str(business_analysis.get('call_to_actions', '')),
                'business_has_discount': business_analysis.get('has_discount', False),
                'business_has_limited_time_discount': business_analysis.get('has_limited_time_discount', False),
                'business_has_exclusive_discount': business_analysis.get('has_exclusive_discount', False),
            })
        
        # Extract product analysis
        product_analysis = gpt4o_results.get('product_analysis', {})
        if isinstance(product_analysis, dict):
            # Handle lists - convert to comma-separated strings
            key_products = product_analysis.get('key_products', [])
            if isinstance(key_products, list):
                cleaned_results['product_key_products'] = ', '.join(key_products)
            else:
                cleaned_results['product_key_products'] = str(key_products)
                
            brands_mentioned = product_analysis.get('brands_mentioned', [])
            if isinstance(brands_mentioned, list):
                cleaned_results['product_brands_mentioned'] = ', '.join(brands_mentioned)
            else:
                cleaned_results['product_brands_mentioned'] = str(brands_mentioned)
                
            selling_points = product_analysis.get('selling_points', [])
            if isinstance(selling_points, list):
                cleaned_results['product_selling_points'] = ', '.join(selling_points)
            else:
                cleaned_results['product_selling_points'] = str(selling_points)
                
            cleaned_results.update({
                'product_content_creative_ideas': product_analysis.get('content_creative_ideas', ''),
                'product_emotional_value': product_analysis.get('emotional_value', ''),
            })
        
        # Extract user analysis
        user_analysis = gpt4o_results.get('user_analysis', {})
        if isinstance(user_analysis, dict):
            cleaned_results.update({
                'user_demographics': user_analysis.get('user_demographics', ''),
                'user_preferences': user_analysis.get('user_preferences', ''),
                'user_pain_points': user_analysis.get('user_pain_points', ''),
                'user_goals': user_analysis.get('user_goals', ''),
            })
        
        # Extract performance analysis
        performance_analysis = gpt4o_results.get('performance_analysis', {})
        if isinstance(performance_analysis, dict):
            cleaned_results.update({
                'performance_content_style': performance_analysis.get('content_style', ''),
                'performance_engagement_potential': performance_analysis.get('engagement_potential', 'unknown'),
            })
            
            trend_alignment = performance_analysis.get('trend_alignment', [])
            if isinstance(trend_alignment, list):
                cleaned_results['performance_trend_alignment'] = ', '.join(trend_alignment)
            else:
                cleaned_results['performance_trend_alignment'] = str(trend_alignment)
                
            compliance_issues = performance_analysis.get('compliance_issues', [])
            if isinstance(compliance_issues, list):
                cleaned_results['performance_compliance_issues'] = ', '.join(compliance_issues)
            else:
                cleaned_results['performance_compliance_issues'] = str(compliance_issues)
        
        # Handle optimization suggestions - extract key points
        optimization_suggestions = gpt4o_results.get('optimization_suggestions', {})
        if isinstance(optimization_suggestions, dict):
            content_suggestions = optimization_suggestions.get('content', [])
            if isinstance(content_suggestions, list):
                cleaned_results['optimization_content_suggestions'] = '; '.join(content_suggestions[:3])  # Limit to 3 suggestions
            else:
                cleaned_results['optimization_content_suggestions'] = str(content_suggestions)
                
            product_placement_suggestions = optimization_suggestions.get('product_placement', [])
            if isinstance(product_placement_suggestions, list):
                cleaned_results['optimization_product_placement_suggestions'] = '; '.join(product_placement_suggestions[:3])
            else:
                cleaned_results['optimization_product_placement_suggestions'] = str(product_placement_suggestions)
            
            content_creative_suggestions = optimization_suggestions.get('content_creative_ideas', [])
            if isinstance(content_creative_suggestions, list):
                cleaned_results['optimization_content_creative_suggestions'] = '; '.join(content_creative_suggestions[:3])
            else:
                cleaned_results['optimization_content_creative_suggestions'] = str(content_creative_suggestions)
            
            emotional_value_suggestions = optimization_suggestions.get('emotional_value', [])
            if isinstance(emotional_value_suggestions, list):
                cleaned_results['optimization_emotional_value_suggestions'] = '; '.join(emotional_value_suggestions[:3])
            else:
                cleaned_results['optimization_emotional_value_suggestions'] = str(emotional_value_suggestions)
            
            compliance_suggestions = optimization_suggestions.get('compliance_issues', [])
            if isinstance(compliance_suggestions, list):
                cleaned_results['optimization_compliance_suggestions'] = '; '.join(compliance_suggestions[:3])
            else:
                cleaned_results['optimization_compliance_suggestions'] = str(compliance_suggestions)
        else:
            cleaned_results['optimization_content_suggestions'] = ''
            cleaned_results['optimization_product_placement_suggestions'] = ''
            cleaned_results['optimization_content_creative_suggestions'] = ''
            cleaned_results['optimization_emotional_value_suggestions'] = ''
            cleaned_results['optimization_compliance_suggestions'] = ''
        
        # Extract embedding analysis summary (if available) - fallback for backward compatibility
        embedding_analysis = gpt4o_results.get('embedding_analysis', {})
        if isinstance(embedding_analysis, dict):
            tags = embedding_analysis.get('tags', {})
            if isinstance(tags, dict):
                # Extract key creator attributes
                creator_attrs = tags.get('creator_attributes', {})
                if not cleaned_results.get('creator_gender') or cleaned_results.get('creator_gender') == 'unknown':
                    cleaned_results['creator_gender'] = creator_attrs.get('gender', 'unknown')
                if not cleaned_results.get('creator_age') or cleaned_results.get('creator_age') == 'unknown':
                    cleaned_results['creator_age'] = creator_attrs.get('age', 'unknown')
                if not cleaned_results.get('creator_charisma') or cleaned_results.get('creator_charisma') == 'unknown':
                    cleaned_results['creator_charisma'] = creator_attrs.get('charisma', 'unknown')
                
                # Extract key content attributes
                content_attrs = tags.get('content_attributes', {})
                if not cleaned_results.get('content_tone') or cleaned_results.get('content_tone') == 'unknown':
                    cleaned_results['content_tone'] = ', '.join(content_attrs.get('overall_tone', [])[:3])
                if not cleaned_results.get('content_format') or cleaned_results.get('content_format') == 'unknown':
                    cleaned_results['content_format'] = ', '.join(content_attrs.get('content_format', [])[:3])
                if not cleaned_results.get('visual_style') or cleaned_results.get('visual_style') == 'unknown':
                    cleaned_results['visual_style'] = ', '.join(content_attrs.get('visual_presentation', [])[:3])
                
                                 # Extract key industry attributes (removed beauty-specific fields as they're not needed in new GPT4O analysis)
        
        # Extract analysis metadata summary
        analysis_metadata = gpt4o_results.get('analysis_metadata', {})
        if isinstance(analysis_metadata, dict):
            cleaned_results.update({
                'frame_count_analyzed': analysis_metadata.get('frame_count', 0),
                'audio_transcript_length': analysis_metadata.get('audio_transcript_length', 0),
            })
        
        # Extract parsed data if available (this contains the actual GPT4O response)
        parsed_data = analysis_metadata.get('parsed_data', {})
        if isinstance(parsed_data, dict):
            # Extract creator analysis from parsed data
            parsed_creator = parsed_data.get('creator_analysis', {})
            if isinstance(parsed_creator, dict):
                # Only update if current value is unknown or empty
                if not cleaned_results.get('creator_gender') or cleaned_results.get('creator_gender') == 'unknown':
                    cleaned_results['creator_gender'] = parsed_creator.get('gender', 'unknown')
                if not cleaned_results.get('creator_age') or cleaned_results.get('creator_age') == 'unknown':
                    cleaned_results['creator_age'] = parsed_creator.get('age', 'unknown')
                if not cleaned_results.get('creator_physical_appearance') or cleaned_results.get('creator_physical_appearance') == 'unknown':
                    cleaned_results['creator_physical_appearance'] = parsed_creator.get('physical_appearance', 'unknown')
                if not cleaned_results.get('creator_body_type') or cleaned_results.get('creator_body_type') == 'unknown':
                    cleaned_results['creator_body_type'] = parsed_creator.get('body_type', 'unknown')
                if not cleaned_results.get('creator_hair_color') or cleaned_results.get('creator_hair_color') == 'unknown':
                    cleaned_results['creator_hair_color'] = parsed_creator.get('hair_color', 'unknown')
                if not cleaned_results.get('creator_hair_type') or cleaned_results.get('creator_hair_type') == 'unknown':
                    cleaned_results['creator_hair_type'] = parsed_creator.get('hair_type', 'unknown')
                if not cleaned_results.get('creator_persona') or cleaned_results.get('creator_persona') == '':
                    persona = parsed_creator.get('persona', [])
                    if isinstance(persona, list):
                        cleaned_results['creator_persona'] = ', '.join(persona[:3])
                    else:
                        cleaned_results['creator_persona'] = str(persona)
                if not cleaned_results.get('creator_charisma_authenticity') or cleaned_results.get('creator_charisma_authenticity') == 'unknown':
                    cleaned_results['creator_charisma_authenticity'] = parsed_creator.get('charisma_authenticity', 'unknown')
                if not cleaned_results.get('creator_economic_status') or cleaned_results.get('creator_economic_status') == 'unknown':
                    cleaned_results['creator_economic_status'] = parsed_creator.get('economic_status', 'unknown')
            
            # Extract content analysis from parsed data
            parsed_content = parsed_data.get('content_analysis', {})
            if isinstance(parsed_content, dict):
                if not cleaned_results.get('content_niche_specialty') or cleaned_results.get('content_niche_specialty') == '':
                    niche = parsed_content.get('niche_specialty', [])
                    if isinstance(niche, list):
                        cleaned_results['content_niche_specialty'] = ', '.join(niche[:3])
                    else:
                        cleaned_results['content_niche_specialty'] = str(niche)
                if not cleaned_results.get('content_format') or cleaned_results.get('content_format') == '':
                    format_list = parsed_content.get('content_format', [])
                    if isinstance(format_list, list):
                        cleaned_results['content_format'] = ', '.join(format_list[:3])
                    else:
                        cleaned_results['content_format'] = str(format_list)
                if not cleaned_results.get('content_overall_tone') or cleaned_results.get('content_overall_tone') == '':
                    tone = parsed_content.get('overall_tone', [])
                    if isinstance(tone, list):
                        cleaned_results['content_overall_tone'] = ', '.join(tone[:3])
                    else:
                        cleaned_results['content_overall_tone'] = str(tone)
                if not cleaned_results.get('content_energy_levels') or cleaned_results.get('content_energy_levels') == 'unknown':
                    cleaned_results['content_energy_levels'] = parsed_content.get('energy_levels', 'unknown')
                if not cleaned_results.get('content_visual_presentation') or cleaned_results.get('content_visual_presentation') == '':
                    visual = parsed_content.get('visual_presentation', [])
                    if isinstance(visual, list):
                        cleaned_results['content_visual_presentation'] = ', '.join(visual[:3])
                    else:
                        cleaned_results['content_visual_presentation'] = str(visual)
                if not cleaned_results.get('content_product_presentation') or cleaned_results.get('content_product_presentation') == '':
                    product_pres = parsed_content.get('product_presentation', [])
                    if isinstance(product_pres, list):
                        cleaned_results['content_product_presentation'] = ', '.join(product_pres[:3])
                    else:
                        cleaned_results['content_product_presentation'] = str(product_pres)
                if not cleaned_results.get('content_video_quality') or cleaned_results.get('content_video_quality') == 'unknown':
                    cleaned_results['content_video_quality'] = parsed_content.get('video_quality', 'unknown')
                if not cleaned_results.get('content_video_scene') or cleaned_results.get('content_video_scene') == 'unknown':
                    cleaned_results['content_video_scene'] = parsed_content.get('video_scene', 'unknown')
                if not cleaned_results.get('content_video_background') or cleaned_results.get('content_video_background') == 'unknown':
                    cleaned_results['content_video_background'] = parsed_content.get('video_background', 'unknown')
                if not cleaned_results.get('content_pacing') or cleaned_results.get('content_pacing') == 'unknown':
                    cleaned_results['content_pacing'] = parsed_content.get('pacing', 'unknown')
                if not cleaned_results.get('content_speech_rate') or cleaned_results.get('content_speech_rate') == 'unknown':
                    cleaned_results['content_speech_rate'] = parsed_content.get('speech_rate', 'unknown')
                if not cleaned_results.get('content_face_visibility') or cleaned_results.get('content_face_visibility') == 'unknown':
                    cleaned_results['content_face_visibility'] = parsed_content.get('face_visibility', 'unknown')
                if not cleaned_results.get('content_tiktok_effects') or cleaned_results.get('content_tiktok_effects') == '':
                    effects = parsed_content.get('tiktok_effects', [])
                    if isinstance(effects, list):
                        cleaned_results['content_tiktok_effects'] = ', '.join(effects[:3])
                    else:
                        cleaned_results['content_tiktok_effects'] = str(effects)
            
            # Extract business analysis from parsed data
            parsed_business = parsed_data.get('business_analysis', {})
            if isinstance(parsed_business, dict):
                if not cleaned_results.get('business_commercial_intent') or cleaned_results.get('business_commercial_intent') == 'unknown':
                    cleaned_results['business_commercial_intent'] = parsed_business.get('commercial_intent', 'unknown')
                if not cleaned_results.get('business_product_placement') or cleaned_results.get('business_product_placement') == 'unknown':
                    cleaned_results['business_product_placement'] = parsed_business.get('product_placement', 'unknown')
                if not cleaned_results.get('business_target_audience') or cleaned_results.get('business_target_audience') == '':
                    cleaned_results['business_target_audience'] = parsed_business.get('target_audience', '')
                if not cleaned_results.get('business_target_audience_fit') or cleaned_results.get('business_target_audience_fit') == 'unknown':
                    cleaned_results['business_target_audience_fit'] = parsed_business.get('target_audience_fit', 'unknown')
                if not cleaned_results.get('business_engagement_drivers') or cleaned_results.get('business_engagement_drivers') == '':
                    drivers = parsed_business.get('engagement_drivers', [])
                    if isinstance(drivers, list):
                        cleaned_results['business_engagement_drivers'] = ', '.join(drivers[:3])
                    else:
                        cleaned_results['business_engagement_drivers'] = str(drivers)
                if not cleaned_results.get('business_trend_adoption') or cleaned_results.get('business_trend_adoption') == '':
                    trends = parsed_business.get('trend_adoption', [])
                    if isinstance(trends, list):
                        cleaned_results['business_trend_adoption'] = ', '.join(trends[:3])
                    else:
                        cleaned_results['business_trend_adoption'] = str(trends)
                if not cleaned_results.get('business_branding_integration') or cleaned_results.get('business_branding_integration') == 'unknown':
                    cleaned_results['business_branding_integration'] = parsed_business.get('branding_integration', 'unknown')
                if not cleaned_results.get('business_call_to_actions') or cleaned_results.get('business_call_to_actions') == '':
                    cta = parsed_business.get('call_to_actions', [])
                    if isinstance(cta, list):
                        cleaned_results['business_call_to_actions'] = ', '.join(cta[:3])
                    else:
                        cleaned_results['business_call_to_actions'] = str(cta)
                
                # Extract discount information from parsed business data
                if not cleaned_results.get('business_has_discount') or cleaned_results.get('business_has_discount') == False:
                    cleaned_results['business_has_discount'] = parsed_business.get('has_discount', False)
                if not cleaned_results.get('business_has_limited_time_discount') or cleaned_results.get('business_has_limited_time_discount') == False:
                    cleaned_results['business_has_exclusive_discount'] = parsed_business.get('has_exclusive_discount', False)
                if not cleaned_results.get('business_has_exclusive_discount') or cleaned_results.get('business_has_exclusive_discount') == False:
                    cleaned_results['business_has_exclusive_discount'] = parsed_business.get('has_exclusive_discount', False)
            
            # Extract product analysis from parsed data
            parsed_product = parsed_data.get('product_analysis', {})
            if isinstance(parsed_product, dict):
                if not cleaned_results.get('product_key_products') or cleaned_results.get('product_key_products') == '':
                    products = parsed_product.get('key_products', [])
                    if isinstance(products, list):
                        cleaned_results['product_key_products'] = ', '.join(products)
                    else:
                        cleaned_results['product_key_products'] = str(products)
                if not cleaned_results.get('product_brands_mentioned') or cleaned_results.get('product_brands_mentioned') == '':
                    brands = parsed_product.get('brands_mentioned', [])
                    if isinstance(brands, list):
                        cleaned_results['product_brands_mentioned'] = ', '.join(brands)
                    else:
                        cleaned_results['product_brands_mentioned'] = str(brands)
                if not cleaned_results.get('product_selling_points') or cleaned_results.get('product_selling_points') == '':
                    points = parsed_product.get('selling_points', [])
                    if isinstance(points, list):
                        cleaned_results['product_selling_points'] = ', '.join(points)
                    else:
                        cleaned_results['product_selling_points'] = str(points)
                if not cleaned_results.get('product_content_creative_ideas') or cleaned_results.get('product_content_creative_ideas') == '':
                    cleaned_results['product_content_creative_ideas'] = parsed_product.get('content_creative_ideas', '')
                if not cleaned_results.get('product_emotional_value') or cleaned_results.get('product_emotional_value') == '':
                    cleaned_results['product_emotional_value'] = parsed_product.get('emotional_value', '')
                
            
            # Extract user analysis from parsed data
            parsed_user = parsed_data.get('user_analysis', {})
            if isinstance(parsed_user, dict):
                if not cleaned_results.get('user_demographics') or cleaned_results.get('user_demographics') == '':
                    cleaned_results['user_demographics'] = parsed_user.get('user_demographics', '')
                if not cleaned_results.get('user_preferences') or cleaned_results.get('user_preferences') == '':
                    cleaned_results['user_preferences'] = parsed_user.get('user_preferences', '')
                if not cleaned_results.get('user_pain_points') or cleaned_results.get('user_pain_points') == '':
                    cleaned_results['user_pain_points'] = parsed_user.get('user_pain_points', '')
                if not cleaned_results.get('user_goals') or cleaned_results.get('user_goals') == '':
                    cleaned_results['user_goals'] = parsed_user.get('user_goals', '')
            
            # Extract performance analysis from parsed data
            parsed_performance = parsed_data.get('performance_analysis', {})
            if isinstance(parsed_performance, dict):
                if not cleaned_results.get('performance_content_style') or cleaned_results.get('performance_content_style') == '':
                    cleaned_results['performance_content_style'] = parsed_performance.get('content_style', '')
                if not cleaned_results.get('performance_engagement_potential') or cleaned_results.get('performance_engagement_potential') == 'unknown':
                    cleaned_results['performance_engagement_potential'] = parsed_performance.get('engagement_potential', 'unknown')
                if not cleaned_results.get('performance_trend_alignment') or cleaned_results.get('performance_trend_alignment') == '':
                    trends = parsed_performance.get('trend_alignment', [])
                    if isinstance(trends, list):
                        cleaned_results['performance_trend_alignment'] = ', '.join(trends)
                    else:
                        cleaned_results['performance_trend_alignment'] = str(trends)
                if not cleaned_results.get('performance_compliance_issues') or cleaned_results.get('performance_compliance_issues') == '':
                    issues = parsed_performance.get('compliance_issues', [])
                    if isinstance(issues, list):
                        cleaned_results['performance_compliance_issues'] = ', '.join(issues)
                    else:
                        cleaned_results['performance_compliance_issues'] = str(issues)
            
            # Extract optimization suggestions from parsed data
            parsed_optimization = parsed_data.get('optimization_suggestions', {})
            if isinstance(parsed_optimization, dict):
                if not cleaned_results.get('optimization_content_suggestions') or cleaned_results.get('optimization_content_suggestions') == '':
                    content_sugg = parsed_optimization.get('content', [])
                    if isinstance(content_sugg, list):
                        cleaned_results['optimization_content_suggestions'] = '; '.join(content_sugg[:3])
                    else:
                        cleaned_results['optimization_content_suggestions'] = str(content_sugg)
                if not cleaned_results.get('optimization_product_placement_suggestions') or cleaned_results.get('optimization_product_placement_suggestions') == '':
                    product_sugg = parsed_optimization.get('product_placement', [])
                    if isinstance(product_sugg, list):
                        cleaned_results['optimization_product_placement_suggestions'] = '; '.join(product_sugg[:3])
                    else:
                        cleaned_results['optimization_product_placement_suggestions'] = str(product_sugg)
                if not cleaned_results.get('optimization_content_creative_suggestions') or cleaned_results.get('optimization_content_creative_suggestions') == '':
                    creative_sugg = parsed_optimization.get('content_creative_ideas', [])
                    if isinstance(creative_sugg, list):
                        cleaned_results['optimization_content_creative_suggestions'] = '; '.join(creative_sugg[:3])
                    else:
                        cleaned_results['optimization_content_creative_suggestions'] = str(creative_sugg)
                if not cleaned_results.get('optimization_emotional_value_suggestions') or cleaned_results.get('optimization_emotional_value_suggestions') == '':
                    emotional_sugg = parsed_optimization.get('emotional_value', [])
                    if isinstance(emotional_sugg, list):
                        cleaned_results['optimization_emotional_value_suggestions'] = '; '.join(emotional_sugg[:3])
                    else:
                        cleaned_results['optimization_emotional_value_suggestions'] = str(emotional_sugg)
                if not cleaned_results.get('optimization_compliance_suggestions') or cleaned_results.get('optimization_compliance_suggestions') == '':
                    compliance_sugg = parsed_optimization.get('compliance_issues', [])
                    if isinstance(compliance_sugg, list):
                        cleaned_results['optimization_compliance_suggestions'] = '; '.join(compliance_sugg[:3])
                    else:
                        cleaned_results['optimization_compliance_suggestions'] = str(compliance_sugg)
        
        return cleaned_results

    def _clean_audio_results_for_csv(self, audio_results):
        """Clean and flatten audio results for CSV output, extracting only essential information."""
        cleaned_results = {}
        
        if not isinstance(audio_results, dict):
            return cleaned_results
        
        # Extract basic audio information
        cleaned_results.update({
            'has_speech': audio_results.get('has_speech', False),
            'speech_text': audio_results.get('speech_text', ''),
            'speech_confidence': audio_results.get('speech_confidence', 0.0),
            'speech_filtered': audio_results.get('speech_filtered', False),
            'music_title': audio_results.get('music_title', ''),
            'music_artist': audio_results.get('music_artist', ''),
        })
        
        # Handle speech rate analysis
        speech_rate_analysis = audio_results.get('speech_rate_analysis', {})
        if isinstance(speech_rate_analysis, dict):
            cleaned_results.update({
                'speech_rate_wpm': speech_rate_analysis.get('speech_rate_wpm', 0),
                'speech_rate_category': speech_rate_analysis.get('rate_category', 'unknown'),
                'speech_rate_score': speech_rate_analysis.get('rate_score', 0.0),
            })
        else:
            cleaned_results.update({
                'speech_rate_wpm': 0,
                'speech_rate_category': 'unknown',
                'speech_rate_score': 0.0,
            })
        
        # Handle structured data if available
        structured_data = audio_results.get('structured_data', {})
        if isinstance(structured_data, dict):
            speech_analysis = structured_data.get('speech_analysis', {})
            if isinstance(speech_analysis, dict):
                cleaned_results.update({
                    'speech_segments_count': len(speech_analysis.get('segments', [])),
                    'speech_total_duration': speech_analysis.get('total_duration', 0),
                })
        
        return cleaned_results

    @log_time("extract_features_from_folder")
    def extract_features_from_folder(self, video_folder, output_folder, csv_output_path=None, parallel=None, max_workers=None):
        video_files = []
        for file in os.listdir(video_folder):
            if file.lower().endswith('.mp4'):
                video_files.append(file)
        
        self.logger.info(f"Found {len(video_files)} video files to process")
        
        if not video_files:
            self.logger.warning("No video files found!")
            return pd.DataFrame()
        batch_size = self.processing_config.get('batch_size', 10)
        enable_progress_bar = self.processing_config.get('enable_progress_bar', True)
        if parallel is None:
            parallel = self.processing_config.get('parallel', False)
        if max_workers is None:
            max_workers = self.processing_config.get('max_workers', None)
        
        all_features = []
        
        if parallel:
            max_workers = max_workers or min(os.cpu_count() or 4, 8)
            self.logger.info(f"Using parallel processing with {max_workers} workers...")
            tasks = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for video_file in video_files:
                    video_path = os.path.join(video_folder, video_file)
                    video_name = os.path.splitext(video_file)[0]
                    if self.output_config.get('create_video_subdirs', True):
                        video_output_dir = os.path.join(output_folder, video_name)
                    else:
                        video_output_dir = output_folder
                    tasks.append(executor.submit(
                        TikTokFeatureExtractor._extract_video_features_static,
                        video_path, video_output_dir, self.config
                    ))
                if enable_progress_bar:
                    iterator = tqdm(as_completed(tasks), total=len(tasks), desc='Processing videos (parallel)')
                else:
                    iterator = as_completed(tasks)
                for future in iterator:
                    try:
                        features = future.result()
                        all_features.append(features)
                    except Exception as e:
                        self.logger.error(f"Error in parallel processing: {e}")
        else:
            if enable_progress_bar:
                iterator = tqdm(video_files, desc='Processing videos')
            else:
                iterator = video_files
            for video_file in iterator:
                video_path = os.path.join(video_folder, video_file)
                video_name = os.path.splitext(video_file)[0]
                if self.output_config.get('create_video_subdirs', True):
                    video_output_dir = os.path.join(output_folder, video_name)
                else:
                    video_output_dir = output_folder
                features = self.extract_video_features(video_path, video_output_dir)
                all_features.append(features)
        
        df = pd.DataFrame(all_features)
        if not df.empty:
            output_formats = self.output_config.get('output_format', ['csv'])
            if 'csv' in output_formats and csv_output_path:
                df.to_csv(csv_output_path, index=False, encoding='utf-8')
                self.logger.info(f"CSV results saved to: {csv_output_path}")
            if 'json' in output_formats:
                json_path = csv_output_path.replace('.csv', '.json') if csv_output_path else 'results.json'
                df.to_json(json_path, orient='records', indent=2)
                self.logger.info(f"JSON results saved to: {json_path}")
        return df

    @staticmethod
    def _extract_video_features_static(video_path, output_dir, config):
        extractor = TikTokFeatureExtractor(config)
        return extractor.extract_video_features(video_path, output_dir)

    def extract_features_from_single_video(self, video_path, output_folder, csv_output_path=None):
        """Extract features from a single video."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_folder, video_name)
        
        features = self.extract_video_features(video_path, video_output_dir)
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Save to CSV if path provided
        if csv_output_path and not df.empty:
            df.to_csv(csv_output_path, index=False, encoding='utf-8')
            self.logger.info(f"\nResults saved to: {csv_output_path}")
        
        return df