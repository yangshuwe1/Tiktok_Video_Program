import os
from PIL import Image
import torch
import numpy as np
import pandas as pd
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from shared_models import (
    yolo_model, blip_processor, blip_model, clip_model, clip_preprocess,
    gpt4o_config, TIKTOK_CATEGORIES, calc_gpt4o_price, image_to_data_url
)
from image_compressor import ImageCompressor
from embedding_analyzer import EmbeddingAnalyzer
import time
from logger import get_logger
import openai
logger = get_logger("multimodal_extractor")

class MultimodalExtractor:
    """Enhanced multimodal feature extraction using all available models effectively."""
    def __init__(self, config=None):
        self.logger = get_logger("MultimodalExtractor")
        self.config = config or {}
        self.yolo_config = self.config.get("yolo", {})
        self.blip_config = self.config.get("blip", {})
        self.clip_config = self.config.get("clip", {})
        self.embedding_config = self.config.get("embedding_analyzer", {})
        # Use GPT4O config from shared_models (this loads from config.yml)
        self.gpt4o_config = gpt4o_config
        self.yolo_model = yolo_model
        self.blip_processor = blip_processor
        self.blip_model = blip_model
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tiktok_categories = TIKTOK_CATEGORIES
        self.visual_batch_size = (
            self.config.get('visual', {}).get('batch_size',
                self.config.get('batch_size', 16))
        )
        self.multimodal_batch_size = (
            self.config.get('multimodal', {}).get('batch_size',
                self.config.get('batch_size', 16))
        )
        # Initialize image compressor for token optimization
        image_compression_config = self.config.get('multimodal', {}).get('image_compression', {})
        self.image_compressor = ImageCompressor(image_compression_config)
        
        # Initialize embedding analyzer if enabled
        self.embedding_analyzer = None
        self.text_only_mode = False
        
        # Check if embedding analyzer should be enabled (from multimodal config)
        use_embedding_analyzer = self.config.get('use_embedding_analyzer', True)
        
        if use_embedding_analyzer and self.embedding_config.get('enabled', False):
            self.embedding_analyzer = EmbeddingAnalyzer(self.embedding_config)
            self.text_only_mode = self.embedding_config.get('text_only_mode', True)
            self.logger.info(f"EmbeddingAnalyzer initialized (text_only_mode: {self.text_only_mode})")
        else:
            if not use_embedding_analyzer:
                self.logger.info("EmbeddingAnalyzer disabled by multimodal config (use_embedding_analyzer: false)")
                # Force disable embedding analyzer regardless of embedding_config.enabled
                self.embedding_analyzer = None
            else:
                self.logger.info("EmbeddingAnalyzer disabled by embedding config (enabled: false)")
                self.embedding_analyzer = None
        
        self.logger.debug(f"MultimodalExtractor initialized with config: {self.config}")
        self.logger.info(f"GPT4O config loaded: {self.gpt4o_config is not None}")
        if self.gpt4o_config:
            # Check for prompts in the config
            prompts = self.gpt4o_config.get('prompts', {})
            if prompts:
                comprehensive_analysis_prompt = prompts.get('comprehensive_analysis', '')
                self.logger.info(f"GPT4O prompt length: {len(comprehensive_analysis_prompt)} characters")
            else:
                self.logger.warning("No prompts found in GPT4O config")
        self.logger.info(f"Image compressor initialized: {self.image_compressor is not None}")
        if self.image_compressor:
            self.logger.info(f"Image compression config: max_dimension={self.image_compressor.max_dimension}, quality={self.image_compressor.quality}")
        if self.yolo_model is None:
            self.logger.warning("YOLO model is not loaded.")
        if self.blip_model is None:
            self.logger.warning("BLIP model is not loaded.")
        if self.clip_model is None:
            self.logger.warning("CLIP model is not loaded.")
        if self.gpt4o_config is None:
            self.logger.warning("GPT4O configuration is not loaded.")

    def extract_comprehensive_features(self, image_path, ocr_text="", audio_text=""):
        """
        Extract comprehensive features using all available models.
        Returns detailed analysis with proper categorization.
        """
        try:
            # Step 1: Extract features from all models
            if not os.path.exists(image_path): logger.warning(f"Warning: file does not exist: {image_path}"); return
            yolo_objects = self._extract_yolo_features(image_path)
            blip_caption = self._extract_blip_caption(image_path)
            clip_embedding = self._extract_clip_embedding(image_path)
            
            # Step 2: Use GPT4O for intelligent analysis with all context
            analysis_result = self._analyze_with_gpt4o_enhanced(
                image_path, yolo_objects, blip_caption, ocr_text, audio_text
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive feature extraction: {e}")
            return self._create_fallback_result(image_path, ocr_text, audio_text)

    def extract_comprehensive_features_batch(self, image_paths, ocr_texts=None, audio_texts=None):
        """
        Batch extract comprehensive features for a list of images.
        """
        if ocr_texts is None:
            ocr_texts = [""] * len(image_paths)
        if audio_texts is None:
            audio_texts = [""] * len(image_paths)
        # Batch YOLO
        yolo_objects_batch = self._extract_yolo_features_batch(image_paths)
        # Batch BLIP
        blip_captions_batch = self._extract_blip_caption_batch(image_paths)
        # Batch CLIP
        clip_embeddings_batch = self._extract_clip_embedding_batch(image_paths)
        # Combine results
        results = []
        for i, image_path in enumerate(image_paths):
            # Optionally, GPT4O or fallback per image
            analysis_result = self._analyze_with_gpt4o_enhanced(
                image_path,
                yolo_objects_batch[i],
                blip_captions_batch[i],
                ocr_texts[i],
                audio_texts[i]
            )
            results.append(analysis_result)
        return results

    def _extract_yolo_features_batch(self, image_paths):
        self.logger.info(f"Running YOLO batch inference on {len(image_paths)} images.")
        all_objects = []
        for i in range(0, len(image_paths), self.visual_batch_size):
            batch_paths = image_paths[i:i+self.visual_batch_size]
            self.logger.debug(f"YOLO batch: {batch_paths}")
            results = self.yolo_model(batch_paths)
            for r in results:
                objects = []
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.yolo_model.names.get(cls_id, str(cls_id))
                    confidence = float(box.conf[0])
                    if confidence > 0.3:
                        objects.append({
                            "class": cls_name,
                            "confidence": confidence,
                            "bbox": [float(x) for x in box.xyxy[0]]
                        })
                all_objects.append(objects)
        self.logger.info(f"YOLO batch inference finished. Total objects: {sum(len(o) for o in all_objects)}")
        return all_objects

    def _extract_blip_caption_batch(self, image_paths):
        self.logger.info(f"Running BLIP batch captioning on {len(image_paths)} images.")
        captions = []
        for i in range(0, len(image_paths), self.multimodal_batch_size):
            batch_paths = image_paths[i:i+self.multimodal_batch_size]
            self.logger.debug(f"BLIP batch: {batch_paths}")
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.blip_processor(images, return_tensors="pt", padding=True)
            device_inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.blip_model.generate(**device_inputs, max_new_tokens=50)
                batch_captions = [self.blip_processor.decode(o, skip_special_tokens=True) for o in outputs]
            captions.extend(batch_captions)
        self.logger.info(f"BLIP batch captioning finished. Total captions: {len(captions)}")
        return captions

    def _extract_clip_embedding_batch(self, image_paths):
        self.logger.info(f"Running CLIP batch embedding on {len(image_paths)} images.")
        all_embeddings = []
        for i in range(0, len(image_paths), self.multimodal_batch_size):
            batch_paths = image_paths[i:i+self.multimodal_batch_size]
            self.logger.debug(f"CLIP batch: {batch_paths}")
            images = [self.clip_preprocess(Image.open(p).convert('RGB')) for p in batch_paths]
            images = torch.stack(images).to(self.device)
            with torch.no_grad():
                embeddings = self.clip_model.encode_image(images)
            all_embeddings.extend(embeddings.cpu().numpy())
        self.logger.info(f"CLIP batch embedding finished. Total embeddings: {len(all_embeddings)}")
        return all_embeddings

    def _extract_yolo_features(self, image_path):
        return self._extract_yolo_features_batch([image_path])[0]
    def _extract_blip_caption(self, image_path):
        return self._extract_blip_caption_batch([image_path])[0]
    def _extract_clip_embedding(self, image_path):
        return self._extract_clip_embedding_batch([image_path])[0]

    def _analyze_with_gpt4o_enhanced(self, image_path, yolo_objects, blip_caption, ocr_text, audio_text):
        """Analyze image using GPT4O with comprehensive context"""
        if self.gpt4o_config is None:
            logger.debug("DEBUG: GPT4O config is None, fallback.")
            return self._create_fallback_result(image_path, ocr_text, audio_text)
        
        try:
            # Build context from all available information
            context_parts = []
            if blip_caption:
                context_parts.append(f"BLIP caption: {blip_caption}")
            if yolo_objects:
                context_parts.append(f"YOLO objects: {', '.join([obj['class'] for obj in yolo_objects[:3]])}")
            if ocr_text:
                context_parts.append(f"Text in image: {ocr_text}")
            if audio_text:
                context_parts.append(f"Audio transcription: {audio_text}")
            
            context = " | ".join(context_parts) if context_parts else ""
            
            # Get prompt from config
            prompt = self.gpt4o_config.get('prompts', {}).get('comprehensive_analysis', 
                "Analyze this TikTok video frame and provide a detailed description of what you see. Focus on products, people, actions, and any text visible in the image. Return a JSON response with the following structure: {\"description\": \"detailed visual description\", \"primary_category\": \"main product category\", \"secondary_category\": \"sub-category\", \"tertiary_category\": \"specific product type\", \"confidence\": 0.95}")
            
            # Add context to prompt if available
            if context:
                prompt = f"{prompt}\n\nAdditional context: {context}"
            
            # Log the prompt for monitoring
            self.logger.info("=" * 80)
            self.logger.info("GPT4O PROMPT MONITORING")
            self.logger.info("=" * 80)
            self.logger.info(f"Image: {os.path.basename(image_path)}")
            self.logger.info(f"Prompt length: {len(prompt)} characters")
            self.logger.info("Prompt content:")
            self.logger.info("-" * 40)
            self.logger.info(prompt)
            self.logger.info("-" * 40)
            
            # Prepare content for GPT4O API
            content = []
            content.append({"type": "text", "text": prompt})
            
            # Compress image to reduce token consumption (replaces original file)
            if os.path.exists(image_path):
                try:
                    # Compress the image (replaces original)
                    self.image_compressor.compress_image(image_path)
                    
                    # Use the same image path (now compressed) for API
                    data_url = image_to_data_url(image_path)
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
                except Exception as e:
                    logger.error(f"Could not compress or load image for GPT4O: {e}")
                    return self._create_fallback_result(image_path, ocr_text, audio_text)
            
            # Call GPT4O API
            self.logger.info(f"Sending request to GPT4O for image: {image_path}")
            response = openai.chat.completions.create(
                model=self.gpt4o_config.get('model_name', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": content}],
                max_tokens=self.gpt4o_config.get('max_tokens', 2048),
                temperature=self.gpt4o_config.get('temperature', 0.1),
            )
            
            answer = response.choices[0].message.content or ""
            self.logger.info(f"GPT4O response received: {len(answer)} characters")
            
            usage = getattr(response, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)
                from shared_models import load_config
                current_config = load_config()
                price = calc_gpt4o_price(prompt_tokens, completion_tokens, current_config)
                self.logger.info(f"GPT4O token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}, cost=${price:.6f}")
            
            # Parse response
            result = self._parse_gpt4o_response(answer)
            return result
            
        except Exception as e:
            logger.error(f"ERROR in enhanced GPT4O analysis: {e}")
            return self._create_fallback_result(image_path, ocr_text, audio_text)

    def _parse_gpt4o_response(self, response):
        """Parse GPT4O response with improved JSON extraction."""
        try:
            # First try to extract JSON from code blocks
            code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response, re.DOTALL)
            if code_block_match:
                json_str = code_block_match.group(1)
            else:
                # Fallback to original method - extract JSON content wrapped in braces
                match = re.search(r'\{[\s\S]*?\}', response)
                if match:
                    json_str = match.group(0)
                else:
                    logger.warning("Warning: No JSON found in response, using text extraction")
                    return self._extract_info_from_text(response)
            
            # Clean JSON string
            json_str = re.sub(r'[\n\r\t]', ' ', json_str)
            json_str = re.sub(r'\s+', ' ', json_str)
            # Parse JSON
            result = json.loads(json_str)
            
            # Ensure all required fields are present with default values
            default_fields = {
                'video_description': 'No description available',
                'primary_category': 'Unknown',
                'secondary_category': 'Unknown',
                'tertiary_category': 'Unknown',
                'commercial_intent': 'unknown',
                'product_placement': 'unknown',
                'key_products': [],
                'brands_mentioned': [],
                'content_creative_ideas': 'Unknown',
                'emotional_value': 'Unknown',
                'selling_points': [],
                'user_demographics': 'Unknown',
                'user_preferences': 'Unknown',
                'user_pain_points': 'Unknown',
                'user_goals': 'Unknown',
                'compliance_issues': [],
                'engagement_potential': 'unknown',
                'trend_alignment': []
            }
            
            # Set default values for missing fields
            for field, default_value in default_fields.items():
                if field not in result:
                    result[field] = default_value
            
            # Handle nested JSON structures
            if 'optimization_suggestions' not in result:
                result['optimization_suggestions'] = {
                    'content': [],
                    'product_placement': [],
                    'content_creative_ideas': [],
                    'emotional_value': [],
                    'compliance_issues': []
                }
            else:
                # Ensure all optimization_suggestions fields exist
                opt_fields = {
                    'content': [],
                    'product_placement': [],
                    'content_creative_ideas': [],
                    'emotional_value': [],
                    'compliance_issues': []
                }
                for field, default_value in opt_fields.items():
                    if field not in result['optimization_suggestions']:
                        result['optimization_suggestions'][field] = default_value
            
            # Check for placeholder text and try to extract meaningful content
            placeholder_detected = False
            
            if 'detailed description of what you see' in result['video_description'] or 'detailed visual description' in result['video_description']:
                logger.warning("Warning: Detected placeholder text in video_description")
                placeholder_detected = True
                # Try to extract actual description from the response
                result['video_description'] = self._extract_actual_description(response)
            
            if 'specific primary category' in result['primary_category'] or 'your specific primary category' in result['primary_category']:
                logger.warning("Warning: Detected placeholder text in primary_category")
                placeholder_detected = True
                result['primary_category'] = self._extract_actual_category(response, 'primary')
            
            if 'specific secondary category' in result['secondary_category'] or 'your specific secondary category' in result['secondary_category']:
                logger.warning("Warning: Detected placeholder text in secondary_category")
                placeholder_detected = True
                result['secondary_category'] = self._extract_actual_category(response, 'secondary')
            
            if 'specific tertiary category' in result['tertiary_category'] or 'your specific tertiary category' in result['tertiary_category']:
                logger.warning("Warning: Detected placeholder text in tertiary_category")
                placeholder_detected = True
                result['tertiary_category'] = self._extract_actual_category(response, 'tertiary')
            
            if placeholder_detected:
                logger.debug("Attempting to extract actual content from GPT4O response...")
            
            # Validate and clean results - return all fields
            return result
                
        except Exception as e:
            logger.error(f"Error parsing GPT4O response: {e}")
            logger.debug(f"Response was: {response[:200]}...")
            return self._extract_info_from_text(response)

    def _extract_actual_description(self, response):
        """Extract actual description from GPT4O response."""
        try:
            # Look for video_description in the response text
            lines = response.split('\n')
            for line in lines:
                if '"video_description"' in line and ':' in line:
                    # Extract content between quotes
                    start = line.find('"video_description"') + len('"video_description"')
                    content = line[start:].strip()
                    if content.startswith(':'):
                        content = content[1:].strip()
                    if content.startswith('"'):
                        content = content[1:]
                    if content.endswith('"'):
                        content = content[:-1]
                    if content and not 'detailed description of what you see' in content:
                        return content
            
            # Fallback: look for any meaningful text
            for line in lines:
                if ':' in line and len(line) > 20 and not line.startswith('"'):
                    return line.strip()
            
            return "Image analysis completed"
        except:
            return "Image analysis completed"

    def _extract_actual_category(self, response, category_type):
        """Extract actual category from Qwen response."""
        try:
            lines = response.split('\n')
            for line in lines:
                if f'"{category_type}_category"' in line and ':' in line:
                    # Extract content between quotes
                    start = line.find(f'"{category_type}_category"') + len(f'"{category_type}_category"')
                    content = line[start:].strip()
                    if content.startswith(':'):
                        content = content[1:].strip()
                    if content.startswith('"'):
                        content = content[1:]
                    if content.endswith('"'):
                        content = content[:-1]
                    if content and not 'specific' in content.lower():
                        return content
            
            return "Unknown"
        except:
            return "Unknown"

    def _extract_info_from_text(self, text):
        """Extract information from text when JSON parsing fails."""
        # Simple text parsing as fallback
        lines = text.split('\n')
        description = ""
        primary = "Unknown"
        secondary = "Unknown"
        tertiary = "Unknown"
        
        for line in lines:
            line_lower = line.lower()
            if 'description' in line_lower and ':' in line:
                description = line.split(':', 1)[1].strip().strip('"')
            elif 'primary' in line_lower and 'category' in line_lower and ':' in line:
                primary = line.split(':', 1)[1].strip().strip('"')
            elif 'secondary' in line_lower and 'category' in line_lower and ':' in line:
                secondary = line.split(':', 1)[1].strip().strip('"')
            elif 'tertiary' in line_lower and 'category' in line_lower and ':' in line:
                tertiary = line.split(':', 1)[1].strip().strip('"')
        
        if not description:
            description = text[:200] + "..." if len(text) > 200 else text
        
        return {
            'video_description': description,
            'primary_category': primary,
            'secondary_category': secondary,
            'tertiary_category': tertiary,
            'content_type': 'other',
            'target_audience': 'general',
            'audio_relevance': 'unknown',
            'audio_summary': 'none',
            # Add new fields with default values
            'commercial_intent': 'unknown',
            'product_placement': 'unknown',
            'key_products': [],
            'brands_mentioned': [],
            'content_creative_ideas': 'Unknown',
            'emotional_value': 'Unknown',
            'selling_points': [],
            'user_demographics': 'Unknown',
            'user_preferences': 'Unknown',
            'user_pain_points': 'Unknown',
            'user_goals': 'Unknown',
            'compliance_issues': [],
            'engagement_potential': 'unknown',
            'trend_alignment': [],
            'optimization_suggestions': {
                'content': [],
                'product_placement': [],
                'content_creative_ideas': [],
                'emotional_value': [],
                'compliance_issues': []
            }
        }

    def _create_fallback_result(self, image_path, ocr_text, audio_text):
        """Create fallback result when Qwen is not available."""
        try:
            # Use BLIP and YOLO for fallback
            if not os.path.exists(image_path): logger.warning(f"Warning: file does not exist: {image_path}"); return
            blip_caption = self._extract_blip_caption(image_path)
            yolo_objects = self._extract_yolo_features(image_path)
            
            description_parts = []
            if blip_caption:
                description_parts.append(f"BLIP: {blip_caption}")
            
            if yolo_objects:
                objects = [obj['class'] for obj in yolo_objects[:3]]
                description_parts.append(f"Objects: {', '.join(objects)}")
            
            if ocr_text:
                description_parts.append(f"OCR: {ocr_text}")
            
            if audio_text:
                description_parts.append(f"Audio: {audio_text}")
            
            description = " | ".join(description_parts) if description_parts else "No description available"
            
            # Simple categorization based on content
            primary, secondary, tertiary = self._simple_categorization(description, yolo_objects)
            
            return {
                'video_description': description,
                'primary_category': primary,
                'secondary_category': secondary,
                'tertiary_category': tertiary,
                'content_type': 'other',
                'target_audience': 'general',
                'audio_relevance': 'unknown',
                'audio_summary': 'none'
            }
            
        except Exception as e:
            logger.error(f"Error in fallback result creation: {e}")
            return {
                'video_description': 'Analysis failed',
                'primary_category': 'Unknown',
                'secondary_category': 'Unknown',
                'tertiary_category': 'Unknown',
                'content_type': 'other',
                'target_audience': 'general',
                'audio_relevance': 'unknown',
                'audio_summary': 'none'
            }

    def _simple_categorization(self, description, yolo_objects):
        """Simple categorization based on keywords and detected objects."""
        description_lower = description.lower()
        object_classes = [obj['class'].lower() for obj in yolo_objects]
        
        # Enhanced keyword matching with more specific categories
        categories = {
            'Beauty and Personal Care': {
                'keywords': ['makeup', 'cosmetic', 'lipstick', 'foundation', 'skincare', 'beauty', 'perfume', 'shampoo'],
                'secondary': 'Makeup',
                'tertiary': 'Foundation'
            },
            'Fashion': {
                'keywords': ['clothing', 'dress', 'shirt', 'pants', 'shoes', 'jewelry', 'accessory', 'bag', 'watch', 'top', 'jeans', 'black'],
                'secondary': 'Clothing',
                'tertiary': 'Dresses'
            },
            'Electronics': {
                'keywords': ['phone', 'laptop', 'computer', 'camera', 'tv', 'electronic', 'device'],
                'secondary': 'Mobile Devices',
                'tertiary': 'Smartphones'
            },
            'Food and Beverages': {
                'keywords': ['food', 'drink', 'snack', 'beverage', 'meal', 'grocery', 'store', 'aisle'],
                'secondary': 'Snacks',
                'tertiary': 'Other'
            },
            'Home and Garden': {
                'keywords': ['furniture', 'home', 'kitchen', 'appliance', 'decoration', 'shelves', 'boxes'],
                'secondary': 'Furniture',
                'tertiary': 'Other'
            }
        }
        
        # Check each category with scoring
        best_score = 0
        best_category = ('Other', 'Other', 'Other')
        
        for primary, info in categories.items():
            score = 0
            # Check description keywords
            for keyword in info['keywords']:
                if keyword in description_lower:
                    score += 1
            
            # Check detected objects
            for keyword in info['keywords']:
                if keyword in object_classes:
                    score += 2  # Objects get higher weight
            
            if score > best_score:
                best_score = score
                best_category = (primary, info['secondary'], info['tertiary'])
        
        return best_category

    def _find_representative_frame_paths(self, frame_files, video_name):
        """Find the correct representative frame paths after they've been renamed"""
        if not frame_files:
            return []
        
        # Get the directory from the first frame path
        frame_dir = os.path.dirname(frame_files[0])
        representative_paths = []
        
        # Look for representative_* files in the directory
        for filename in os.listdir(frame_dir):
            if filename.startswith('representative_') and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Check if this is for the correct video
                if video_name and video_name in filename:
                    representative_paths.append(os.path.join(frame_dir, filename))
        
        # Sort by timestamp if available, otherwise by filename
        representative_paths.sort()
        
        # If we found representative files, use them
        if representative_paths:
            self.logger.info(f"Found {len(representative_paths)} representative frame files")
            return representative_paths[:len(frame_files)]  # Limit to original count
        
        # Fallback to original paths if no representative files found
        self.logger.warning("No representative frame files found, using original paths")
        return frame_files

    def _get_empty_analysis_result(self):
        """Get empty analysis result structure"""
        return {
            'video_description': '',
            'primary_category': 'Unknown',
            'secondary_category': 'Unknown',
            'tertiary_category': 'Unknown',
            'commercial_intent': 'unknown',
            'product_placement': 'unknown',
            'key_products': [],
            'brands_mentioned': [],
            'content_creative_ideas': '',
            'emotional_value': '',
            'selling_points': [],
            'user_demographics': '',
            'user_preferences': '',
            'user_pain_points': '',
            'user_goals': '',
            'compliance_issues': [],
            'engagement_potential': 'unknown',
            'trend_alignment': [],
            'optimization_suggestions': {
                'content': [],
                'product_placement': [],
                'content_creative_ideas': [],
                'emotional_value': [],
                'compliance_issues': []
            }
        }
    
    def extract_gpt4o_features(self, frame_files, video_name=None, audio_transcript="", speech_rate_analysis=None, api_metadata: Dict = None):
        """
        Extract GPT4O features with configurable analysis modes:
        - Batch image upload mode: when enable_batch_image_upload=True
        - Embedding analysis mode: when enable_embedding_analysis=True
        """
        try:
            if not frame_files:
                return self._get_empty_analysis_result()
            
            # Get configuration settings
            batch_upload_enabled = self.config.get('enable_batch_image_upload', True)
            embedding_analysis_enabled = self.config.get('enable_embedding_analysis', True)
            
            self.logger.info(f"Configuration: batch_image_upload={batch_upload_enabled}, embedding_analysis={embedding_analysis_enabled}")
            
            # Select representative frames
            max_frames = self.config.get('representative_frames', 10)
            if len(frame_files) > max_frames:
                step = len(frame_files) // max_frames
                selected_frames = [frame_files[i] for i in range(0, len(frame_files), step)][:max_frames]
                self.logger.info(f"Selected {len(selected_frames)} frames from {len(frame_files)} total frames for analysis")
            else:
                selected_frames = frame_files
            
            # Mode 1: Embedding Analysis (when enabled)
            if embedding_analysis_enabled and self.embedding_analyzer:
                try:
                    self.logger.info("Using embedding analyzer for comprehensive analysis")
                    # Get comprehensive analysis results
                    embedding_results = self.embedding_analyzer.generate_prompt_from_images(selected_frames, audio_transcript)
                    
                    # Extract components from embedding results
                    embedding_tags = embedding_results.get("embedding_tags", {})
                    structured_analysis = embedding_results.get("structured_analysis", {})
                    comprehensive_prompt = embedding_results.get("comprehensive_prompt", "")
                    
                    self.logger.info(f"Generated embedding analysis with {len(embedding_tags)} tag categories")
                    self.logger.info("=" * 80)
                    self.logger.info("EMBEDDING ANALYSIS RESULTS:")
                    self.logger.info("=" * 80)
                    self.logger.info(f"Embedding tags: {list(embedding_tags.keys())}")
                    self.logger.info(f"Structured analysis: {list(structured_analysis.keys())}")
                    self.logger.info("=" * 80)
                    
                    # Use the comprehensive analysis for enhanced GPT4O input
                    return self._analyze_with_comprehensive_prompt(
                        embedding_tags, structured_analysis, comprehensive_prompt, 
                        selected_frames, video_name, audio_transcript, speech_rate_analysis, api_metadata or {}
                    )
                except Exception as e:
                    self.logger.error(f"Error in embedding analysis: {e}")
                    self.logger.info("Falling back to standard batch analysis")
            
            # Mode 2: Batch Image Upload (when enabled)
            if batch_upload_enabled:
                self.logger.info("Using batch image upload mode for GPT4O analysis")
                return self._analyze_video_frames_batch(selected_frames, video_name, audio_transcript, api_metadata or {})
            
            # Mode 3: Fallback to frame-by-frame analysis
            self.logger.info("Using fallback frame-by-frame analysis")
            return self._fallback_frame_by_frame_analysis(frame_files, video_name, audio_transcript)
                
        except Exception as e:
            self.logger.error(f"Error in GPT4O feature extraction: {e}")
            return self._fallback_frame_by_frame_analysis(frame_files, video_name, audio_transcript)
    
    def _analyze_with_comprehensive_prompt(self, embedding_tags: Dict, structured_analysis: Dict, 
                                   comprehensive_prompt: str, frame_files: List[str], 
                                   video_name: str = None, audio_transcript: str = "", 
                                   speech_rate_analysis: Dict = None, api_metadata: Dict = None) -> Dict[str, Any]:
        """
        Analyze video using comprehensive analysis prompt that combines all embedding analysis results.
        This method creates a comprehensive prompt that includes all relevant tags and analysis
        to generate a complete JSON response with all required columns.
        """
        try:
            self.logger.info("Analyzing video with comprehensive analysis prompt")
            
            # Build comprehensive prompt that includes all embedding analysis
            comprehensive_prompt = self._build_comprehensive_analysis_prompt(
                embedding_tags, structured_analysis, comprehensive_prompt, 
                frame_files, audio_transcript, video_name, speech_rate_analysis, api_metadata or {}
            )
            
            # Call GPT4O API with comprehensive prompt (text-only to reduce token consumption)
            response = self._call_gpt4o_api([{"type": "text", "text": comprehensive_prompt}])
            
            if response:
                # Parse the comprehensive response
                analysis_result = self._parse_comprehensive_gpt4o_response(
                    response, embedding_tags, structured_analysis, frame_files, video_name, audio_transcript
                )
                
                # Save results
                self._save_comprehensive_analysis_results(analysis_result, frame_files[0].rsplit('/', 1)[0] if frame_files else "", video_name)
                
                return analysis_result
            else:
                self.logger.warning("No response from GPT4O API, using fallback analysis")
                return self._fallback_frame_by_frame_analysis(frame_files, video_name, audio_transcript)
                
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {e}")
            return self._fallback_frame_by_frame_analysis(frame_files, video_name, audio_transcript)
    
    def _build_comprehensive_analysis_prompt(self, embedding_tags: Dict, structured_analysis: Dict, 
                                     comprehensive_prompt: str, frame_files: List[str], 
                                     audio_transcript: str, video_name: str = None, 
                                     speech_rate_analysis: Dict = None, api_metadata: Dict = None) -> str:
        """
        Build comprehensive analysis prompt that combines all embedding analysis results.
        This prompt is designed to generate a comprehensive JSON response with all required columns.
        """
        try:
            # Get the comprehensive analysis prompt from config
            base_prompt = self.gpt4o_config.get('prompts', {}).get('comprehensive_analysis', '')
            
            if not base_prompt:
                # Fallback to comprehensive unified prompt
                base_prompt = """You are a professional TikTok video content analysis expert with comprehensive knowledge of TikTok's content creator ecosystem and commercial analysis.

Please analyze the following video content based on the detailed embedding analysis results provided below.

DETAILED EMBEDDING ANALYSIS RESULTS:
{comprehensive_prompt}

EMBEDDING TAGS SUMMARY:
{embedding_tags_summary}

STRUCTURED ANALYSIS:
{structured_analysis_summary}

VIDEO FRAME INFORMATION:
- Total representative frames analyzed: {frame_count}
- Frame sequence: {frame_sequence}
- Analysis method: CLIP + V-JEPA2 embedding analysis + comprehensive tag system

AUDIO TRANSCRIPT:
{audio_transcript}

SPEECH RATE ANALYSIS:
{speech_rate_summary}

ANALYSIS REQUIREMENTS:
Based on the comprehensive embedding analysis results above, please provide a detailed analysis covering all aspects of TikTok content creation and commercial potential.

        {
          "video_description": "comprehensive description of the entire video content",
          "primary_category": "beauty & personal care|fashion & accessories|home & lifestyle|sports & fitness|health & wellness|parenting & family|food & beverage|technology & gadgets|entertainment & gaming|travel & adventure|education & learning|pets & animals|automotive & transportation|business & finance|art & creativity|diy & crafts|comedy|dance|music|news & politics|science & nature|other",
          "secondary_category": "skincare|makeup|haircare|fragrance|clothing|shoes|accessories|jewelry|home decor|kitchenware|gardening|fitness equipment|supplements|diet|mental health|recipes|restaurants|baking|adventure|hotels|budget travel|gadgets|apps|software|consoles|mobile games|team sports|yoga|running|early education|language learning|toddler care|teen issues|dog care|cat care|car reviews|motorcycles|investing|entrepreneurship|movies|tv shows|painting|knitting|woodworking|standup|skits|choreography|cover songs|current events|environment|other",
          "tertiary_category": "specific product type or niche within secondary category",
          
          "creator_analysis": {
            "gender": "male|female|multiple-people|non-human|unknown",
            "age": "child|teen|20s|30s|40s|50+|unknown",
            "physical_appearance": "very appealing|appealing|average|distinctive|unknown",
            "body_type": "slim|average|athletic|curvy|plus-size|unknown",
            "hair_color": "black|brunette|blonde|red|mixed-colored|unknown|not-visible",
            "hair_type": "straight|wavy|curly|coily|updo|tied|braids|unknown|not-visible",
            "persona": ["fashionista|beauty guru|fitness enthusiast|foodie|traveler|tech geek|gamer|mommy blogger|pet lover|financial advisor|comedian|dancer|musician|educator|activist|artist|craftsman|lifestyle coach|health expert|reviewer|unboxer|vlogger|storyteller|influencer|model|expert"],
            "charisma_authenticity": "enthusiastic & energetic|professional & authoritative|authentic & relatable|engaging & expressive|calm & soothing|unknown",
            "economic_status": "luxury-focused|middle-class|budget-conscious|minimalist|unknown"
          },
          
          "content_analysis": {
            "niche_specialty": ["skincare routines|makeup tutorials|hair styling|outfit ideas|home organization|workout plans|healthy recipes|travel hacks|tech reviews|gaming walkthroughs|parenting advice|pet training|car maintenance|investment tips|comedy skits|dance tutorials|song covers|educational content|art tutorials|diy projects|fashion hauls|beauty hauls|food reviews|budget travel tips|luxury unboxing|fitness challenges|mental health awareness|sustainable living"],
            "content_format": ["tutorial|review|vlog|Q&A|challenge|prank|haul|day-in-the-life|how-to|tips & tricks|storytime|pov|comedy sketch|dance video|music video|educational|inspirational|motivational|interview|behind-the-scenes|unboxing|comparison|transformation|live stream"],
            "overall_tone": ["informative|humorous|inspirational|relaxing|dramatic|romantic|sarcastic|upbeat|serious|mysterious|suspenseful|heartwarming|emotional|empowering|casual|formal"],
            "energy_levels": "vibrant & passionate|composed & professional|calm & soothing|energetic & dynamic|relaxed & peaceful|unknown",
            "visual_presentation": ["minimalist|bright & colorful|dark & moody|warm & cozy|professional lighting|natural lighting|dynamic editing|cinematic|aesthetic|raw & unfiltered|retro|futuristic|vintage"],
            "product_presentation": ["product showcase|unboxing & reviews|voiceover content|live demonstrations|before & after comparisons|product comparisons|tutorial|haul|gift guide|collection|restock|routine|get ready with me|what's in my bag|try on haul"],
            "video_quality": "very-high|high|moderate|low|unknown",
            "video_scene": "daily life|family life|at work|professional settings|outdoor activities|social activities|travel scenes|special occasions|studio|bedroom|bathroom|kitchen|living room|garden|gym|cafe|restaurant|store|beach|mountains|city|countryside|other",
            "video_background": "indoor|outdoor|kitchen|bedroom|living room|bathroom|office|studio|gym|cafe|restaurant|store|beach|mountains|city street|park|other",
            "pacing": "very-fast|fast|moderate|slow|very-slow|unknown",
            "speech_rate": "very-fast|fast|moderate|slow|very-slow|no-speech|unknown",
            "face_visibility": "always-visible|frequently-visible|occasionally-visible|rarely-visible|never-visible|unknown",
            "tiktok_effects": ["filters|ar-effects|text-overlay|green-screen|split-screens|stitches|duets|slow-motion|time-lapse|zoom-effects|transition-effects|voice-effects|stickers|emoji|none"]
          },
          
          "business_analysis": {
            "commercial_intent": "commercial|non-commercial|educational|promotional|informational|entertainment|mixed",
            "product_placement": "explicit|subtle|none|prominent|background|integrated",
            "target_audience": "Detailed description of intended viewer demographics and interests",
            "target_audience_fit": "low|moderate|high|very-high|unknown",
            "engagement_drivers": ["promotional-sales|controversial-topics|clickbait|exaggerated-expression|relatable-struggle|satisfying-process|motivational-message|humor|emotional-story|shocking-facts|interactive-polls|user-participation|trending-music|celebration-events|exclusive-content"],
            "trend_adoption": ["dance-challenges|audio-trends|meme-adaptations|pov-storytelling|seasonal-themes|hashtag-challenges|transition-challenges|outfit-challenges|recipe-trends|fitness-challenges|beauty-trends|comedy-trends|duet-chains|stitch-reactions"],
            "branding_integration": "subtle|moderate|overt|not-present|seamless|forced|other",
            "call_to_actions": ["like|share|comment|follow|link-in-bio|swipe-up|buy-now|join-live-stream|save|tag-friends|visit-website|download-app|sign-up|enter-giveaway|use-code"],
            "has_discount": true,
            "has_limited_time_discount": true,
            "has_exclusive_discount": true
          },
          
          "product_analysis": {
            "key_products": ["Product Name 1", "Product Name 2"],
            "brands_mentioned": ["Brand 1", "Brand 2"],
            "selling_points": ["Key benefit 1", "Key benefit 2"],
            "content_creative_ideas": "Description of innovative presentation approaches",
            "emotional_value": "Description of emotional resonance created",
          },
          
          "user_analysis": {
            "user_demographics": "Target user group description",
            "user_preferences": "Typical preferences of target audience",
            "user_pain_points": "Key frustrations addressed",
            "user_goals": "Aspirations and objectives of users"
          },
          
          "performance_analysis": {
            "content_style": "Description of stylistic approach",
            "engagement_potential": "high|medium|low|very-high|unknown",
            "trend_alignment": ["Current trend 1", "Current trend 2"],
            "compliance_issues": ["auctions|exaggerated promises|false or misleading info|gambling & gamification|illegal or criminal activity|other"]
          },
          
          "optimization_suggestions": {
            "content": ["Suggestion 1", "Suggestion 2"],
            "product_placement": ["Suggestion 1", "Suggestion 2"],
            "content_creative_ideas": ["Suggestion 1", "Suggestion 2"],
            "emotional_value": ["Suggestion 1", "Suggestion 2"],
            "compliance_issues": ["Suggestion 1", "Suggestion 2"]
          }
        }

        IMPORTANT: Return ONLY the JSON object, no additional text or explanations. Use the exact field names and structure provided above. Choose values ONLY from the enumerated options provided for each field.

IMPORTANT GUIDELINES:
1. Use the embedding tags and structured analysis to inform all aspects of your response
2. Ensure all fields are populated based on the available analysis data
3. Be specific and detailed in your descriptions
4. Consider the creator demographics, content characteristics, and commercial aspects
5. Provide actionable optimization suggestions
6. Consider compliance with TikTok's content policies
7. Base your analysis on the comprehensive embedding analysis, not assumptions

The embedding analysis provides detailed insights into:
- Creator attributes (gender, age, personas, charisma)
- Content characteristics (niche, format, tone, energy)
- Industry-specific analysis (beauty, fashion, lifestyle)
- Object detection and action analysis
- Temporal consistency and content complexity
- Visual quality and presentation style"""
            
            # Inject TikTok API metadata context to guide analysis
            try:
                api_md = api_metadata or {}
                api_context_lines = ["TIKTOK API METADATA:"]
                for k in [
                    'video_description','hashtags','music_title','music_author','is_ad','is_commerce',
                    'ad_authorization','likes','comments','shares','views','duration','created_time']:
                    v = api_md.get(k, None)
                    if v is None:
                        continue
                    if isinstance(v, list):
                        v = ', '.join([str(x) for x in v][:30])
                    api_context_lines.append(f"- {k}: {v}")
                api_context = "\n".join(api_context_lines)
            except Exception:
                api_context = ""

            # Format the prompt with actual data
            frame_sequence = f"Frame 1-{len(frame_files)}"
            
            # Create embedding tags summary
            embedding_tags_summary = self._format_embedding_tags_summary(embedding_tags)
            
            # Create structured analysis summary
            structured_analysis_summary = self._format_structured_analysis_summary(structured_analysis)
            
            # Create speech rate analysis summary
            speech_rate_summary = self._format_speech_rate_summary(speech_rate_analysis)
            
            # Format the complete prompt using safe string replacement
            # Use string replacement instead of .format() to avoid issues with curly braces in content
            formatted_prompt = base_prompt
            formatted_prompt = formatted_prompt.replace("{comprehensive_prompt}", str(comprehensive_prompt))
            formatted_prompt = formatted_prompt.replace("{embedding_tags_summary}", str(embedding_tags_summary))
            formatted_prompt = formatted_prompt.replace("{structured_analysis_summary}", str(structured_analysis_summary))
            formatted_prompt = formatted_prompt.replace("{frame_count}", str(len(frame_files)))
            formatted_prompt = formatted_prompt.replace("{frame_sequence}", str(frame_sequence))
            formatted_prompt = formatted_prompt.replace("{audio_transcript}", str(audio_transcript))
            formatted_prompt = formatted_prompt.replace("{speech_rate_summary}", str(speech_rate_summary))
            # Append API metadata summary if provided
            try:
                api_md = api_metadata or {}
                if api_md:
                    hashtags = api_md.get('hashtags', [])
                    if isinstance(hashtags, list):
                        hashtags_str = ", ".join([str(h) for h in hashtags[:30]])
                    else:
                        hashtags_str = str(hashtags)
                    api_block = (
                        "\nAPI METADATA (from TikTok API):\n"
                        f"- Creator: {api_md.get('creator_username', '')}\n"
                        f"- Video ID: {api_md.get('video_id', '')}\n"
                        f"- Description: {str(api_md.get('video_description', ''))[:500]}\n"
                        f"- Hashtags: {hashtags_str}\n"
                        f"- Is Ad: {bool(api_md.get('is_ad', False))}, Is Commerce: {bool(api_md.get('is_commerce', False))}, Ad Authorization: {bool(api_md.get('ad_authorization', False))}\n"
                        f"- Stats: likes={int(api_md.get('likes', 0) or 0)}, comments={int(api_md.get('comments', 0) or 0)}, shares={int(api_md.get('shares', 0) or 0)}, views={int(api_md.get('views', 0) or 0)}\n"
                        f"- Duration: {int(api_md.get('duration', 0) or 0)}s, Created: {api_md.get('created_time', '')}\n"
                    )
                    formatted_prompt = formatted_prompt + api_block
            except Exception:
                pass
            
            # Log the prompt for monitoring
            self.logger.info("=" * 80)
            self.logger.info("UNIFIED ANALYSIS PROMPT MONITORING")
            self.logger.info("=" * 80)
            self.logger.info(f"Video: {video_name or 'unknown'}")
            self.logger.info(f"Prompt length: {len(formatted_prompt)} characters")
            self.logger.info(f"Embedding tags: {list(embedding_tags.keys())}")
            self.logger.info(f"Structured analysis: {list(structured_analysis.keys())}")
            self.logger.info("=" * 80)
            
            return formatted_prompt
            
        except Exception as e:
            self.logger.error(f"Error building unified prompt: {e}")
            return f"Analyze this TikTok video with {len(frame_files)} frames. Audio: {audio_transcript}"
    
    def _format_embedding_tags_summary(self, embedding_tags: Dict) -> str:
        """Format embedding tags into a readable summary"""
        try:
            summary_parts = []
            
            # Creator attributes
            if embedding_tags.get("creator_attributes"):
                creator = embedding_tags["creator_attributes"]
                summary_parts.append("CREATOR ATTRIBUTES:")
                summary_parts.append(f"  Gender: {creator.get('gender', 'unknown')}")
                summary_parts.append(f"  Age: {creator.get('age', 'unknown')}")
                summary_parts.append(f"  Personas: {', '.join(creator.get('personas', []))}")
                summary_parts.append(f"  Charisma: {creator.get('charisma', 'unknown')}")
                summary_parts.append(f"  Economic Status: {creator.get('economic_status', 'unknown')}")
            
            # Content attributes
            if embedding_tags.get("content_attributes"):
                content = embedding_tags["content_attributes"]
                summary_parts.append("\nCONTENT ATTRIBUTES:")
                summary_parts.append(f"  Niche: {', '.join(content.get('niche_specialty', []))}")
                summary_parts.append(f"  Format: {', '.join(content.get('content_format', []))}")
                summary_parts.append(f"  Tone: {', '.join(content.get('overall_tone', []))}")
                summary_parts.append(f"  Energy: {content.get('energy_levels', 'unknown')}")
                summary_parts.append(f"  Visual Style: {', '.join(content.get('visual_presentation', []))}")
            
            # Industry attributes
            if embedding_tags.get("industry_attributes"):
                industry = embedding_tags["industry_attributes"]
                summary_parts.append("\nINDUSTRY ATTRIBUTES:")
                if industry.get("beauty"):
                    beauty_items = []
                    for category, items in industry["beauty"].items():
                        beauty_items.extend(items[:3])  # Top 3 items per category
                    summary_parts.append(f"  Beauty: {', '.join(beauty_items)}")
                if industry.get("fashion"):
                    summary_parts.append(f"  Fashion: {', '.join(industry['fashion'][:5])}")
                if industry.get("home_lifestyle"):
                    summary_parts.append(f"  Home/Lifestyle: {', '.join(industry['home_lifestyle'][:5])}")
            
            # Objects and actions
            if embedding_tags.get("objects"):
                objects = embedding_tags["objects"]
                summary_parts.append("\nDETECTED OBJECTS:")
                for category, items in objects.items():
                    if items:
                        object_names = [item["object"] for item in items[:3]]
                        summary_parts.append(f"  {category}: {', '.join(object_names)}")
            
            if embedding_tags.get("actions"):
                actions = embedding_tags["actions"]
                summary_parts.append("\nDETECTED ACTIONS:")
                for category, items in actions.items():
                    if items:
                        action_names = [item["action"] for item in items[:3]]
                        summary_parts.append(f"  {category}: {', '.join(action_names)}")
            
            # Overall summary
            if embedding_tags.get("overall_summary"):
                overall = embedding_tags["overall_summary"]
                summary_parts.append("\nOVERALL SUMMARY:")
                summary_parts.append(f"  Primary Content Type: {overall.get('primary_content_type', 'unknown')}")
                summary_parts.append(f"  Target Audience: {overall.get('target_audience', 'unknown')}")
                summary_parts.append(f"  Commercial Focus: {overall.get('commercial_focus', 'unknown')}")
                summary_parts.append(f"  Content Quality: {overall.get('content_quality', 'unknown')}")
                if overall.get("key_highlights"):
                    summary_parts.append(f"  Key Highlights: {'; '.join(overall['key_highlights'])}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error formatting embedding tags summary: {e}")
            return "Embedding analysis available but formatting failed"
    
    def _format_structured_analysis_summary(self, structured_analysis: Dict) -> str:
        """Format structured analysis into a readable summary"""
        try:
            summary_parts = []
            
            # Creator profile
            if structured_analysis.get("creator_profile"):
                creator = structured_analysis["creator_profile"]
                summary_parts.append("CREATOR PROFILE:")
                if creator.get("gender_distribution"):
                    gender_dist = creator["gender_distribution"]
                    primary_gender = max(gender_dist.items(), key=lambda x: x[1])[0] if gender_dist else "unknown"
                    summary_parts.append(f"  Primary Gender: {primary_gender}")
                if creator.get("age_distribution"):
                    age_dist = creator["age_distribution"]
                    primary_age = max(age_dist.items(), key=lambda x: x[1])[0] if age_dist else "unknown"
                    summary_parts.append(f"  Primary Age: {primary_age}")
                if creator.get("persona_types"):
                    summary_parts.append(f"  Personas: {', '.join(creator['persona_types'][:3])}")
            
            # Content profile
            if structured_analysis.get("content_profile"):
                content = structured_analysis["content_profile"]
                summary_parts.append("\nCONTENT PROFILE:")
                if content.get("niche_focus"):
                    niche_items = sorted(content["niche_focus"].items(), key=lambda x: x[1], reverse=True)[:3]
                    niches = [item[0] for item in niche_items]
                    summary_parts.append(f"  Top Niches: {', '.join(niches)}")
                if content.get("tone_profile"):
                    tone_items = sorted(content["tone_profile"].items(), key=lambda x: x[1], reverse=True)[:3]
                    tones = [item[0] for item in tone_items]
                    summary_parts.append(f"  Top Tones: {', '.join(tones)}")
            
            # Commercial profile
            if structured_analysis.get("commercial_profile"):
                commercial = structured_analysis["commercial_profile"]
                summary_parts.append("\nCOMMERCIAL PROFILE:")
                summary_parts.append(f"  Commercial Intent: {commercial.get('commercial_intent', 'unknown')}")
                if commercial.get("product_categories"):
                    categories = list(commercial["product_categories"].keys())
                    summary_parts.append(f"  Product Categories: {', '.join(categories)}")
                if commercial.get("selling_points"):
                    summary_parts.append(f"  Selling Points: {', '.join(commercial['selling_points'][:5])}")
            
            # Technical profile
            if structured_analysis.get("technical_profile"):
                technical = structured_analysis["technical_profile"]
                summary_parts.append("\nTECHNICAL PROFILE:")
                if technical.get("video_quality"):
                    quality_items = sorted(technical["video_quality"].items(), key=lambda x: x[1], reverse=True)[:2]
                    qualities = [item[0] for item in quality_items]
                    summary_parts.append(f"  Video Quality: {', '.join(qualities)}")
                summary_parts.append(f"  Temporal Consistency: {technical.get('temporal_consistency', 0.5):.2f}")
                summary_parts.append(f"  Content Complexity: {technical.get('content_complexity', 0.5):.2f}")
            
            # Engagement profile
            if structured_analysis.get("engagement_profile"):
                engagement = structured_analysis["engagement_profile"]
                summary_parts.append("\nENGAGEMENT PROFILE:")
                summary_parts.append(f"  Audience Appeal: {engagement.get('audience_appeal', 'unknown')}")
                if engagement.get("engagement_factors"):
                    summary_parts.append(f"  Engagement Factors: {', '.join(engagement['engagement_factors'])}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error formatting structured analysis summary: {e}")
            return "Structured analysis available but formatting failed"

    def _format_speech_rate_summary(self, speech_rate_analysis: Dict) -> str:
        """Format speech rate analysis into a readable summary"""
        try:
            if not speech_rate_analysis:
                return "No speech rate analysis available"
            
            summary_parts = []
            summary_parts.append("SPEECH RATE ANALYSIS:")
            
            # Basic metrics
            if speech_rate_analysis.get('speech_rate_wpm'):
                summary_parts.append(f"  Words per minute: {speech_rate_analysis['speech_rate_wpm']:.1f} WPM")
            if speech_rate_analysis.get('syllables_per_minute'):
                summary_parts.append(f"  Syllables per minute: {speech_rate_analysis['syllables_per_minute']:.1f} SPM")
            if speech_rate_analysis.get('rate_category'):
                summary_parts.append(f"  Rate category: {speech_rate_analysis['rate_category']}")
            
            # Scoring breakdown
            if speech_rate_analysis.get('overall_score'):
                summary_parts.append(f"  Overall score: {speech_rate_analysis['overall_score']:.1f}/10")
            if speech_rate_analysis.get('rate_score'):
                summary_parts.append(f"  Rate score: {speech_rate_analysis['rate_score']:.1f}/10")
            if speech_rate_analysis.get('consistency_score'):
                summary_parts.append(f"  Consistency score: {speech_rate_analysis['consistency_score']:.1f}/10")
            if speech_rate_analysis.get('pause_score'):
                summary_parts.append(f"  Pause score: {speech_rate_analysis['pause_score']:.1f}/10")
            if speech_rate_analysis.get('clarity_score'):
                summary_parts.append(f"  Clarity score: {speech_rate_analysis['clarity_score']:.1f}/10")
            
            # Analysis and recommendations
            if speech_rate_analysis.get('rate_analysis'):
                rate_analysis = speech_rate_analysis['rate_analysis']
                if rate_analysis.get('recommendation'):
                    summary_parts.append(f"  Recommendation: {rate_analysis['recommendation']}")
                if rate_analysis.get('strengths'):
                    summary_parts.append(f"  Strengths: {', '.join(rate_analysis['strengths'])}")
                if rate_analysis.get('areas_for_improvement'):
                    summary_parts.append(f"  Areas for improvement: {', '.join(rate_analysis['areas_for_improvement'])}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error formatting speech rate summary: {e}")
            return "Speech rate analysis available but formatting failed"

    def _analyze_video_frames_batch(self, frame_files, video_name=None, audio_transcript="", api_metadata: Dict = None):
        if not frame_files:
            return {
                'video_description': '',
                'primary_category': 'Unknown',
                'secondary_category': 'Unknown',
                'tertiary_category': 'Unknown'
            }
        
        try:
            # Use representative_frames from config instead of hardcoded 5
            max_frames = self.config.get('representative_frames', 10)
            if len(frame_files) > max_frames:
                step = len(frame_files) // max_frames
                selected_frames = [frame_files[i] for i in range(0, len(frame_files), step)][:max_frames]
                self.logger.info(f"Selected {len(selected_frames)} frames from {len(frame_files)} total frames for batch analysis")
            else:
                selected_frames = frame_files
            
            # Compress images to reduce token consumption (replaces original files)
            self.logger.info("Compressing images to optimize token usage...")
            self.image_compressor.compress_images_batch(selected_frames)
            
            # Use the same frames (now compressed) for analysis
            yolo_objects_batch = self._extract_yolo_features_batch(selected_frames)
            blip_captions_batch = self._extract_blip_caption_batch(selected_frames)
            
            prompt = self._build_batch_analysis_prompt(selected_frames, yolo_objects_batch, blip_captions_batch, audio_transcript, api_metadata or {})
            
            content = [{"type": "text", "text": prompt}]
            
            # Use compressed images for GPT4O API
            for frame_path in selected_frames:
                try:
                    data_url = image_to_data_url(frame_path)
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
                except Exception as e:
                    self.logger.warning(f"Failed to add compressed image {frame_path}: {e}")
            
            response = self._call_gpt4o_api(content)
            
            if response:
                analysis_result = self._parse_batch_gpt4o_response(response, selected_frames, video_name, audio_transcript)
                
                if video_name:
                    self._save_batch_analysis_results(analysis_result, frame_files[0].rsplit('/', 1)[0], video_name)
                
                return analysis_result
            else:
                self.logger.warning("Batch GPT4O analysis failed, falling back to frame-by-frame analysis")
                return self._fallback_frame_by_frame_analysis(frame_files, video_name, audio_transcript)
                
        except Exception as e:
            self.logger.error(f"Error in batch frame analysis: {e}")
            return self._fallback_frame_by_frame_analysis(frame_files, video_name, audio_transcript)

    def _build_batch_analysis_prompt(self, frame_files, yolo_objects_batch, blip_captions_batch, audio_transcript, api_metadata: Dict = None):
        # Use the comprehensive prompt from config
        batch_prompt = self.gpt4o_config.get('prompts', {}).get('comprehensive_analysis', 
            """You are a professional TikTok video content analysis expert with comprehensive knowledge of TikTok's content creator tag system. 
        Please analyze the following video keyframes and provide comprehensive content analysis.

        Please carefully observe all frames and analyze based on the following information:
        1. Visual content of each frame
        2. Detected objects and products
        3. Audio transcription content (if available)
        4. Coherence and storytelling between frames

        Please return a JSON format response with the following structure:
        {{
            "video_description": "comprehensive description of the entire video",
            "primary_category": "main product category",
            "secondary_category": "secondary product category", 
            "tertiary_category": "specific product type",
            "commercial_intent": "commercial/non-commercial/educational",
            "product_placement": "explicit/subtle/none",
            "key_products": ["product1", "product2", ...],
            "brands_mentioned": ["brand1", "brand2", ...],
            "content_creative_ideas": "content creative approach (Before/After, Tutorial, Review, etc.)",
            "emotional_value": "emotional appeal (confidence, beauty, empowerment, etc.)",
            "selling_points": ["key selling points like long-wear, oil-control, time-saving, eco-friendly, etc."],
            "user_demographics": "specific target user group (age, gender, lifestyle, etc. )",
            "user_preferences": "specific target user preferences (color, size, etc. )",
            "user_pain_points": "specific target user pain points (skin type, skin concerns, etc. )",
            "user_goals": "specific target user goals (skin care, makeup, etc. )",
            "compliance_issues": ["issue1", "issue2", ...],
            "engagement_potential": "high/medium/low",
            "trend_alignment": ["current_trend1", "current_trend2", ...],
            "optimization_suggestions": {{
                "content": ["suggestion1", "suggestion2", ...],
                "product_placement": ["suggestion1", "suggestion2", ...],
                "content_creative_ideas": ["suggestion1", "suggestion2", ...],
                "emotional_value": ["suggestion1", "suggestion2", ...],
                "compliance_issues": ["suggestion1", "suggestion2", ...],
            }}
        }}

        Analysis requirements:
        - Focus on commercial intent and product showcase
        - Identify main product categories and brands
        - Analyze target audience demographics and preferences
        - Evaluate commercial value of content
        - Identify content creative approach (Before/After, Tutorial, etc.)
        - Analyze emotional value and appeal
        - Extract key selling points and product benefits
        - Provide brief analysis for each frame
        - Compliance considers 25 items total, such as:
            - Auctions: Flags content promoting bidding or auction-based sales models.
            - Exaggerated promises: Detects unrealistic claims such as "instant results" or "guaranteed outcomes."
            - False or misleading info: Identifies fake reviews, unverified claims, or inconsistent product messaging.
            - Gambling & gamification: Flags lottery, prize wheels, or game-like mechanics that simulate gambling.
            - Illegal or criminal activity: Detects promotion of counterfeits, unlicensed products, or criminal behavior.
        - Analyze user journey and conversion funnel
        - Identify content effectiveness indicators
        - Evaluate brand safety and compliance risks
        - Assess competitive positioning
        Audio transcription content: {audio_transcript}

        Frames:""")
        
        # Inject API metadata block if available
        try:
            meta = api_metadata or {}
            if meta:
                hashtags = meta.get('hashtags', [])
                if isinstance(hashtags, list):
                    hashtags_str = ", ".join([str(h) for h in hashtags[:30]])
                else:
                    hashtags_str = str(hashtags)
                api_block = (
                    "\nAPI METADATA (from TikTok API):\n"
                    f"- Creator: {meta.get('creator_username', '')}\n"
                    f"- Video ID: {meta.get('video_id', '')}\n"
                    f"- Description: {str(meta.get('video_description', ''))[:500]}\n"
                    f"- Hashtags: {hashtags_str}\n"
                    f"- Is Ad: {bool(meta.get('is_ad', False))}, Is Commerce: {bool(meta.get('is_commerce', False))}, Ad Authorization: {bool(meta.get('ad_authorization', False))}\n"
                    f"- Stats: likes={int(meta.get('likes', 0) or 0)}, comments={int(meta.get('comments', 0) or 0)}, shares={int(meta.get('shares', 0) or 0)}, views={int(meta.get('views', 0) or 0)}\n"
                    f"- Duration: {int(meta.get('duration', 0) or 0)}s, Created: {meta.get('created_time', '')}\n"
                )
                batch_prompt = batch_prompt + api_block
        except Exception:
            pass

        # Handle the case where the comprehensive prompt contains JSON structure
        # Replace the audio_transcript placeholder safely
        if "{audio_transcript}" in batch_prompt:
            prompt = batch_prompt.replace("{audio_transcript}", audio_transcript)
        else:
            # Fallback: append audio transcript at the end
            prompt = batch_prompt + f"\n\nAudio transcription content: {audio_transcript}"

        # Add frame information to prompt
        for i, (frame_path, yolo_objects, blip_caption) in enumerate(zip(frame_files, yolo_objects_batch, blip_captions_batch)):
            frame_name = os.path.basename(frame_path)
            objects_str = ", ".join([obj['class'] for obj in yolo_objects[:5]]) if yolo_objects else "No objects detected"
            prompt += f"\n\nFrame {i+1} ({frame_name}):"
            prompt += f"\n- BLIP description: {blip_caption}"
            prompt += f"\n- Detected objects: {objects_str}"
        
        # Log the batch prompt for monitoring
        self.logger.info("=" * 80)
        self.logger.info("GPT4O BATCH PROMPT MONITORING")
        self.logger.info("=" * 80)
        self.logger.info(f"Number of frames: {len(frame_files)}")
        self.logger.info(f"Prompt length: {len(prompt)} characters")
        self.logger.info("Prompt content:")
        self.logger.info("-" * 40)
        self.logger.info(prompt)
        self.logger.info("-" * 40)
        
        return prompt

    def _call_gpt4o_api(self, content):
        if self.gpt4o_config is None:
            return None
        
        try:
            from shared_models import load_config
            current_config = load_config()
            current_gpt4o_config = current_config.get('models', {}).get('gpt4o', {})
            
            response = openai.chat.completions.create(
                model=current_gpt4o_config.get('model_name', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": content}],
                max_tokens=current_gpt4o_config.get('max_tokens', 2048),
                temperature=current_gpt4o_config.get('temperature', 0.1)
            )
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            cost = calc_gpt4o_price(prompt_tokens, completion_tokens, current_config)
            self.logger.info(f"GPT4O API call completed - input tokens: {prompt_tokens}, output tokens: {completion_tokens}, cost: ${cost:.4f}")
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"GPT4O API Failed: {e}")
            return None

    def _parse_batch_gpt4o_response(self, response, frame_files, video_name, audio_transcript):
        try:
            import re
            
            # First try to extract JSON from code blocks
            code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response, re.DOTALL)
            if code_block_match:
                json_str = code_block_match.group(1)
            else:
                # Fallback to original method - extract JSON content wrapped in braces
                json_match = re.search(r'\{[\s\S]*?\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    self.logger.warning("No JSON found in response, using text extraction")
                    return self._extract_info_from_text(response)
            
            # Clean JSON string - be more careful with cleaning
            json_str = json_str.strip()
            # Remove any trailing commas before closing braces/brackets
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            # Fix common JSON formatting issues
            json_str = re.sub(r'(["\w])\s*,\s*(["\w])', r'\1, \2', json_str)
            
            self.logger.info(f"Cleaned JSON string: {json_str[:300]}...")
            
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON decode error: {e}")
                self.logger.warning(f"Attempting to fix JSON: {json_str}...")
                
                # Try to fix common JSON issues
                # Remove trailing commas
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                # Fix unescaped quotes in strings
                json_str = re.sub(r'([^\\])"([^"]*)"([^\\])', r'\1"\2"\3', json_str)
                
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse JSON even after cleaning")
                    return self._extract_info_from_text(response)
            
            # Ensure all required fields are present with default values
            default_fields = {
                'video_description': 'No description available',
                'primary_category': 'Unknown',
                'secondary_category': 'Unknown',
                'tertiary_category': 'Unknown',
                'commercial_intent': 'unknown',
                'product_placement': 'unknown',
                'key_products': [],
                'brands_mentioned': [],
                'content_creative_ideas': 'Unknown',
                'emotional_value': 'Unknown',
                'selling_points': [],
                'user_demographics': 'Unknown',
                'user_preferences': 'Unknown',
                'user_pain_points': 'Unknown',
                'user_goals': 'Unknown',
                'compliance_issues': [],
                'engagement_potential': 'unknown',
                'trend_alignment': []
            }
            
            # Set default values for missing fields
            for field, default_value in default_fields.items():
                if field not in result:
                    result[field] = default_value
            
            # Handle nested JSON structures
            if 'optimization_suggestions' not in result:
                result['optimization_suggestions'] = {
                    'content': [],
                    'product_placement': [],
                    'content_creative_ideas': [],
                    'emotional_value': [],
                    'compliance_issues': []
                }
            else:
                # Ensure all optimization_suggestions fields exist
                opt_fields = {
                    'content': [],
                    'product_placement': [],
                    'content_creative_ideas': [],
                    'emotional_value': [],
                    'compliance_issues': []
                }
                for field, default_value in opt_fields.items():
                    if field not in result['optimization_suggestions']:
                        result['optimization_suggestions'][field] = default_value
            
            return result
                
        except Exception as e:
            self.logger.error(f"Failed to parse batch GPT4O response: {e}")
            self.logger.error(f"Response was: {response}")
            return self._extract_info_from_text(response)

    def _save_batch_analysis_results(self, analysis_result, frame_dir, video_name):
        try:
            json_filename = f"{video_name}_batch_gpt4o_analysis.json"
            json_path = os.path.join(frame_dir, json_filename)
            
            json_data = {
                "video_name": video_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_type": "batch_gpt4o",
                "audio_transcript": analysis_result.get('audio_transcript', ''),
                "analysis_result": analysis_result
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Batch GPT4O analysis results saved to: {json_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save batch analysis results: {e}")

    def _fallback_frame_by_frame_analysis(self, frame_files, video_name, audio_transcript):
        self.logger.info("Using frame-by-frame analysis as fallback")
        ocr_texts = ["" for _ in frame_files]
        audio_texts = [audio_transcript for _ in frame_files]
        batch_results = self.extract_comprehensive_features_batch(frame_files, ocr_texts, audio_texts)
        
        results = []
        for i, analysis in enumerate(batch_results):
            results.append({
                "frame": os.path.basename(frame_files[i]),
                "frame_path": frame_files[i],
                "video_description": analysis['video_description'],
                "primary_category": analysis['primary_category'],
                "secondary_category": analysis['secondary_category'],
                "tertiary_category": analysis['tertiary_category'],
                "content_type": analysis['content_type'],
                "target_audience": analysis['target_audience'],
                "audio_relevance": analysis.get('audio_relevance', 'unknown'),
                "audio_summary": analysis.get('audio_summary', 'none'),
                "ocr_text": ocr_texts[i],
                # Add new fields with default values
                "commercial_intent": "unknown",
                "product_placement": "unknown",
                "key_products": [],
                "brands_mentioned": [],
                "content_creative_ideas": "Unknown",
                "emotional_value": "Unknown",
                "selling_points": [],
                "user_demographics": "Unknown",
                "user_preferences": "Unknown",
                "user_pain_points": "Unknown",
                "user_goals": "Unknown",
                "compliance_issues": [],
                "engagement_potential": "unknown",
                "trend_alignment": [],
                "optimization_suggestions": {
                    "content": [],
                    "product_placement": [],
                    "content_creative_ideas": [],
                    "emotional_value": [],
                    "compliance_issues": []
                }
            })
        
        if results and video_name:
            try:
                json_filename = f"{video_name}_fallback_gpt4o_analysis.json"
                json_path = os.path.join(frame_files[0].rsplit('/', 1)[0], json_filename)
                
                json_data = {
                    "video_name": video_name,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "analysis_type": "fallback_frame_by_frame",
                    "total_frames_analyzed": len(results),
                    "audio_transcript": audio_transcript,
                    "frame_analyses": results,
                    "summary": {
                        "video_description": " | ".join([r['video_description'] for r in results[:3]]),
                        "primary_category": self._get_most_common_category(results, 'primary_category'),
                        "secondary_category": self._get_most_common_category(results, 'secondary_category'),
                        "tertiary_category": self._get_most_common_category(results, 'tertiary_category'),
                        "commercial_intent": "unknown",
                        "product_placement": "unknown",
                        "key_products": [],
                        "brands_mentioned": [],
                        "content_creative_ideas": "Unknown",
                        "emotional_value": "Unknown",
                        "selling_points": [],
                        "user_demographics": "Unknown",
                        "user_preferences": "Unknown",
                        "user_pain_points": "Unknown",
                        "user_goals": "Unknown",
                        "compliance_issues": [],
                        "engagement_potential": "unknown",
                        "trend_alignment": [],
                        "optimization_suggestions": {
                            "content": [],
                            "product_placement": [],
                            "content_creative_ideas": [],
                            "emotional_value": [],
                            "compliance_issues": []
                        }
                    }
                }
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Fallback analysis results saved to: {json_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to save fallback analysis results: {e}")
        
        if not results:
            return {
                'video_description': '',
                'primary_category': 'Unknown',
                'secondary_category': 'Unknown',
                'tertiary_category': 'Unknown',
                'commercial_intent': 'unknown',
                'product_placement': 'unknown',
                'key_products': [],
                'brands_mentioned': [],
                'content_creative_ideas': 'Unknown',
                'emotional_value': 'Unknown',
                'selling_points': [],
                'user_demographics': 'Unknown',
                'user_preferences': 'Unknown',
                'user_pain_points': 'Unknown',
                'user_goals': 'Unknown',
                'compliance_issues': [],
                'engagement_potential': 'unknown',
                'trend_alignment': [],
                'optimization_suggestions': {
                    'content': [],
                    'product_placement': [],
                    'content_creative_ideas': [],
                    'emotional_value': [],
                    'compliance_issues': []
                }
            }
        
        descriptions = [r['video_description'] for r in results]
        categories = {}
        for result in results:
            for key in ['primary_category', 'secondary_category', 'tertiary_category']:
                if key not in categories:
                    categories[key] = {}
                cat = result[key]
                categories[key][cat] = categories[key].get(cat, 0) + 1
        
        most_common = {}
        for key in categories:
            most_common[key] = max(categories[key].items(), key=lambda x: x[1])[0]
        
        summary_description = " | ".join(descriptions[:3])
        return {
            'video_description': summary_description,
            'primary_category': most_common['primary_category'],
            'secondary_category': most_common['secondary_category'],
            'tertiary_category': most_common['tertiary_category'],
            'commercial_intent': 'unknown',
            'product_placement': 'unknown',
            'key_products': [],
            'brands_mentioned': [],
            'content_creative_ideas': 'Unknown',
            'emotional_value': 'Unknown',
            'selling_points': [],
            'user_demographics': 'Unknown',
            'user_preferences': 'Unknown',
            'user_pain_points': 'Unknown',
            'user_goals': 'Unknown',
            'compliance_issues': [],
            'engagement_potential': 'unknown',
            'trend_alignment': [],
            'optimization_suggestions': {
                'content': [],
                'product_placement': [],
                'content_creative_ideas': [],
                'emotional_value': [],
                'compliance_issues': []
            }
        }

    def generate_video_summary_csv(self, gpt4o_results, output_path):
        """Generate CSV with video description and AI categorization"""
        if not gpt4o_results:
            return
        
        # Aggregate results
        descriptions = []
        primary_categories = {}
        secondary_categories = {}
        tertiary_categories = {}
        
        for result in gpt4o_results:
            descriptions.append(result['description'])
            
            # Count categories
            primary = result['primary_category']
            secondary = result['secondary_category']
            tertiary = result['tertiary_category']
            
            primary_categories[primary] = primary_categories.get(primary, 0) + 1
            secondary_categories[secondary] = secondary_categories.get(secondary, 0) + 1
            tertiary_categories[tertiary] = tertiary_categories.get(tertiary, 0) + 1
        
        # Get most common categories
        most_common_primary = max(primary_categories.items(), key=lambda x: x[1])[0] if primary_categories else "Unknown"
        most_common_secondary = max(secondary_categories.items(), key=lambda x: x[1])[0] if secondary_categories else "Unknown"
        most_common_tertiary = max(tertiary_categories.items(), key=lambda x: x[1])[0] if tertiary_categories else "Unknown"
        
        # Create summary description
        summary_description = " | ".join(descriptions[:3])  # Take first 3 descriptions
        
        # Create DataFrame
        df = pd.DataFrame([{
            'Video_Name': gpt4o_results[0]['frame'].split('_representative_')[0] if gpt4o_results else "Unknown",
            'Description_of_Video': summary_description,
            'Primary_Category': most_common_primary,
            'Secondary_Category': most_common_secondary,
            'Tertiary_Category': most_common_tertiary,
            'Category_Confidence_Primary': primary_categories.get(most_common_primary, 0) / len(gpt4o_results),
            'Category_Confidence_Secondary': secondary_categories.get(most_common_secondary, 0) / len(gpt4o_results),
            'Category_Confidence_Tertiary': tertiary_categories.get(most_common_tertiary, 0) / len(gpt4o_results)
        }])
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Video summary saved to: {output_path}")
        
        return df 
    
    def _get_most_common_category(self, results, category_key):
        """Helper method to get the most common category from results."""
        categories = {}
        for result in results:
            cat = result.get(category_key, 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        if not categories:
            return 'Unknown'
        
        return max(categories.items(), key=lambda x: x[1])[0]
    
    def _parse_comprehensive_gpt4o_response(self, response, embedding_tags: Dict, structured_analysis: Dict, 
                                     frame_files: List[str], video_name: str = None, audio_transcript: str = "") -> Dict[str, Any]:
        """
        Parse comprehensive GPT4O response with comprehensive embedding analysis context.
        This method parses the complete JSON response with all required columns.
        """
        try:
            self.logger.info("Parsing comprehensive GPT4O response")
            
            # Extract the actual response content
            if isinstance(response, dict):
                content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            else:
                content = str(response)
            
            # Try to parse JSON response
            parsed_data = self._extract_json_from_response(content)
            
            if not parsed_data:
                self.logger.warning("Failed to parse JSON from GPT4O response, using fallback")
                return self._create_unified_fallback_result(frame_files, video_name, audio_transcript, response)
            
            # Validate and enhance the parsed data with embedding analysis
            enhanced_data = self._enhance_parsed_data_with_embedding_analysis(
                parsed_data, embedding_tags, structured_analysis
            )
            
            # Create comprehensive result
            result = {
                'video_description': enhanced_data.get('video_description', ''),
                'primary_category': enhanced_data.get('primary_category', 'Unknown'),
                'secondary_category': enhanced_data.get('secondary_category', 'Unknown'),
                'tertiary_category': enhanced_data.get('tertiary_category', 'Unknown'),
                'commercial_intent': enhanced_data.get('commercial_intent', 'unknown'),
                'product_placement': enhanced_data.get('product_placement', 'unknown'),
                'target_audience': enhanced_data.get('target_audience', ''),
                'content_style': enhanced_data.get('content_style', ''),
                'key_products': enhanced_data.get('key_products', []),
                'brands_mentioned': enhanced_data.get('brands_mentioned', []),
                'content_creative_ideas': enhanced_data.get('content_creative_ideas', ''),
                'emotional_value': enhanced_data.get('emotional_value', ''),
                'selling_points': enhanced_data.get('selling_points', []),
                'user_demographics': enhanced_data.get('user_demographics', ''),
                'user_preferences': enhanced_data.get('user_preferences', ''),
                'user_pain_points': enhanced_data.get('user_pain_points', ''),
                'user_goals': enhanced_data.get('user_goals', ''),
                'compliance_issues': enhanced_data.get('compliance_issues', []),
                'engagement_potential': enhanced_data.get('engagement_potential', 'unknown'),
                'trend_alignment': enhanced_data.get('trend_alignment', []),
                'optimization_suggestions': enhanced_data.get('optimization_suggestions', {
                    'content': [],
                    'product_placement': [],
                    'content_creative_ideas': [],
                    'emotional_value': [],
                    'compliance_issues': []
                }),
                'embedding_analysis': {
                    'tags': embedding_tags,
                    'structured_analysis': structured_analysis
                },
                'analysis_metadata': {
                    'frame_count': len(frame_files),
                    'video_name': video_name,
                    'audio_transcript_length': len(audio_transcript.split()) if audio_transcript else 0,
                    'gpt4o_response': response,
                    'parsed_data': parsed_data
                }
            }
            
            self.logger.info(f"Successfully parsed comprehensive GPT4O response for {video_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing comprehensive GPT4O response: {e}")
            return self._create_comprehensive_fallback_result(frame_files, video_name, audio_transcript, response)
    
    def _enhance_parsed_data_with_embedding_analysis(self, parsed_data: Dict, embedding_tags: Dict, 
                                                   structured_analysis: Dict) -> Dict[str, Any]:
        """Enhance parsed GPT4O data with embedding analysis insights"""
        try:
            enhanced_data = parsed_data.copy()
            
            # Enhance target audience with creator demographics
            if embedding_tags.get("creator_attributes"):
                creator = embedding_tags["creator_attributes"]
                if creator.get("gender") and creator.get("age"):
                    current_audience = enhanced_data.get("target_audience", "")
                    demographic_info = f"{creator['gender']} {creator['age']}"
                    if demographic_info not in current_audience:
                        enhanced_data["target_audience"] = f"{demographic_info} - {current_audience}".strip(" -")
            
            # Enhance user demographics
            if embedding_tags.get("creator_attributes"):
                creator = embedding_tags["creator_attributes"]
                if creator.get("personas"):
                    enhanced_data["user_demographics"] = f"{', '.join(creator['personas'][:3])}"
            
            # Enhance key products with detected objects
            if embedding_tags.get("objects", {}).get("beauty_products"):
                beauty_products = [obj["object"] for obj in embedding_tags["objects"]["beauty_products"][:5]]
                current_products = enhanced_data.get("key_products", [])
                enhanced_data["key_products"] = list(set(current_products + beauty_products))
            
            # Enhance content style with visual analysis
            if embedding_tags.get("content_attributes"):
                content = embedding_tags["content_attributes"]
                if content.get("visual_presentation"):
                    visual_styles = content["visual_presentation"][:3]
                    current_style = enhanced_data.get("content_style", "")
                    if current_style:
                        enhanced_data["content_style"] = f"{current_style} - {', '.join(visual_styles)}"
                    else:
                        enhanced_data["content_style"] = ", ".join(visual_styles)
            
            # Enhance emotional value with content analysis
            if embedding_tags.get("content_attributes"):
                content = embedding_tags["content_attributes"]
                if content.get("overall_tone"):
                    tones = content["overall_tone"][:2]
                    current_emotional = enhanced_data.get("emotional_value", "")
                    if current_emotional:
                        enhanced_data["emotional_value"] = f"{current_emotional} - {', '.join(tones)}"
                    else:
                        enhanced_data["emotional_value"] = ", ".join(tones)
            
            # Enhance engagement potential with structured analysis
            if structured_analysis.get("engagement_profile"):
                engagement = structured_analysis["engagement_profile"]
                if engagement.get("audience_appeal"):
                    enhanced_data["engagement_potential"] = engagement["audience_appeal"]
            
            # Enhance commercial intent with structured analysis
            if structured_analysis.get("commercial_profile"):
                commercial = structured_analysis["commercial_profile"]
                if commercial.get("commercial_intent"):
                    enhanced_data["commercial_intent"] = commercial["commercial_intent"]
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"Error enhancing parsed data: {e}")
            return parsed_data
    
    def _create_unified_fallback_result(self, frame_files: List[str], video_name: str = None, 
                                       audio_transcript: str = "", response: str = "") -> Dict[str, Any]:
        """Create fallback result when unified analysis fails"""
        try:
            # Try to extract basic information from embedding analysis if available
            basic_info = self._extract_basic_info_from_embedding_analysis(frame_files)
            
            return {
                'video_description': basic_info.get('description', f'TikTok video with {len(frame_files)} frames'),
                'primary_category': basic_info.get('primary_category', 'Unknown'),
                'secondary_category': basic_info.get('secondary_category', 'Unknown'),
                'tertiary_category': basic_info.get('tertiary_category', 'Unknown'),
                'commercial_intent': basic_info.get('commercial_intent', 'unknown'),
                'product_placement': basic_info.get('product_placement', 'unknown'),
                'target_audience': basic_info.get('target_audience', ''),
                'content_style': basic_info.get('content_style', ''),
                'key_products': basic_info.get('key_products', []),
                'brands_mentioned': basic_info.get('brands_mentioned', []),
                'content_creative_ideas': basic_info.get('content_creative_ideas', ''),
                'emotional_value': basic_info.get('emotional_value', ''),
                'selling_points': basic_info.get('selling_points', []),
                'user_demographics': basic_info.get('user_demographics', ''),
                'user_preferences': basic_info.get('user_preferences', ''),
                'user_pain_points': basic_info.get('user_pain_points', ''),
                'user_goals': basic_info.get('user_goals', ''),
                'compliance_issues': basic_info.get('compliance_issues', []),
                'engagement_potential': basic_info.get('engagement_potential', 'unknown'),
                'trend_alignment': basic_info.get('trend_alignment', []),
                'optimization_suggestions': {
                    'content': [],
                    'product_placement': [],
                    'content_creative_ideas': [],
                    'emotional_value': [],
                    'compliance_issues': []
                },
                'embedding_analysis': {},
                'analysis_metadata': {
                    'frame_count': len(frame_files),
                    'video_name': video_name,
                    'audio_transcript_length': len(audio_transcript.split()) if audio_transcript else 0,
                    'gpt4o_response': response,
                    'error': 'Unified analysis failed, using fallback'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error creating unified fallback result: {e}")
            return {
                'video_description': f'TikTok video analysis failed for {video_name or "unknown"}',
                'primary_category': 'Unknown',
                'secondary_category': 'Unknown',
                'tertiary_category': 'Unknown',
                'commercial_intent': 'unknown',
                'product_placement': 'unknown',
                'target_audience': '',
                'content_style': '',
                'key_products': [],
                'brands_mentioned': [],
                'content_creative_ideas': '',
                'emotional_value': '',
                'selling_points': [],
                'user_demographics': '',
                'user_preferences': '',
                'user_pain_points': '',
                'user_goals': '',
                'compliance_issues': [],
                'engagement_potential': 'unknown',
                'trend_alignment': [],
                'optimization_suggestions': {
                    'content': [],
                    'product_placement': [],
                    'content_creative_ideas': [],
                    'emotional_value': [],
                    'compliance_issues': []
                },
                'embedding_analysis': {},
                'analysis_metadata': {
                    'frame_count': len(frame_files),
                    'video_name': video_name,
                    'error': str(e)
                }
            }
    
    def _extract_basic_info_from_embedding_analysis(self, frame_files: List[str]) -> Dict[str, Any]:
        """Extract basic information from embedding analysis for fallback"""
        try:
            if not self.embedding_analyzer:
                return {}
            
            # Get basic embedding analysis
            embedding_results = self.embedding_analyzer.generate_prompt_from_images(frame_files, "")
            embedding_tags = embedding_results.get("embedding_tags", {})
            
            basic_info = {}
            
            # Extract basic description
            if embedding_tags.get("overall_summary"):
                overall = embedding_tags["overall_summary"]
                basic_info["primary_category"] = overall.get("primary_content_type", "Unknown")
                basic_info["target_audience"] = overall.get("target_audience", "")
                basic_info["commercial_intent"] = overall.get("commercial_focus", "unknown")
            
            # Extract key products
            if embedding_tags.get("objects", {}).get("beauty_products"):
                beauty_products = [obj["object"] for obj in embedding_tags["objects"]["beauty_products"][:3]]
                basic_info["key_products"] = beauty_products
            
            # Extract content style
            if embedding_tags.get("content_attributes"):
                content = embedding_tags["content_attributes"]
                if content.get("niche_specialty"):
                    basic_info["content_style"] = ", ".join(content["niche_specialty"][:2])
            
            return basic_info
            
        except Exception as e:
            self.logger.error(f"Error extracting basic info from embedding analysis: {e}")
            return {}
    
    def _save_comprehensive_analysis_results(self, analysis_result, frame_dir, video_name):
        """
        Save comprehensive analysis results to files.
        """
        try:
            if not video_name:
                return
            
            # Create output directory
            output_dir = os.path.join(frame_dir, 'comprehensive_analysis')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save comprehensive JSON result
            json_path = os.path.join(output_dir, f'{video_name}_comprehensive_analysis.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            
            # Save embedding analysis separately
            if analysis_result.get('embedding_analysis'):
                embedding_path = os.path.join(output_dir, f'{video_name}_embedding_analysis.json')
                with open(embedding_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result['embedding_analysis'], f, indent=2, ensure_ascii=False)
            
            # Save analysis metadata
            if analysis_result.get('analysis_metadata'):
                metadata_path = os.path.join(output_dir, f'{video_name}_analysis_metadata.json')
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result['analysis_metadata'], f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Comprehensive analysis results saved to: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving comprehensive analysis results: {e}")

    def _extract_json_from_response(self, response_content: str) -> Dict[str, Any]:
        """Extract JSON data from GPT4O response content"""
        try:
            # Log the response for debugging
            self.logger.info(f"GPT4O Response {response_content}...")
            
            # Try multiple patterns to find JSON
            patterns = [
                r'\{.*\}',  # Basic JSON object
                r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
                r'```\s*(\{.*?\})\s*```',  # JSON in generic code blocks
            ]
            
            for pattern in patterns:
                json_match = re.search(pattern, response_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1) if len(json_match.groups()) > 0 else json_match.group(0)
                    try:
                        parsed_data = json.loads(json_str)
                        self.logger.info(f"Successfully parsed JSON with pattern: {pattern}")
                        return parsed_data
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing JSON from response with pattern {pattern}: {e}")
                        continue
            
            # If no JSON found, try to extract key-value pairs from the text
            self.logger.warning("No JSON found in GPT4O response, attempting to extract key-value pairs")
            extracted_data = {}
            
            # Look for common patterns in the response
            lines = response_content.split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line and not line.startswith('#'):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().lower().replace(' ', '_')
                        value = parts[1].strip()
                        if value and value != 'unknown':
                            extracted_data[key] = value
            
            if extracted_data:
                self.logger.info(f"Extracted {len(extracted_data)} key-value pairs from text response")
                return extracted_data
            
            self.logger.warning("No JSON or structured data found in GPT4O response")
            return {}
                
        except Exception as e:
            self.logger.error(f"Error extracting JSON from response: {e}")
            return {}