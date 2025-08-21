import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import pandas as pd

from progress_manager import ProgressManager
from storage_manager import StorageManager
from tiktok_crawler import TikTokCrawler
from tiktok_feature_extractor import TikTokFeatureExtractor

logger = logging.getLogger(__name__)

@dataclass
class CreatorTask:
    """A single creator task to be processed"""
    username: str
    platform: str = "tiktok"
    category: str = ""
    notes: str = ""
    priority: int = 1
    max_videos: int = 30

@dataclass
class BatchResult:
    """High-level result for a processed batch"""
    batch_id: str
    total_creators: int
    successful_creators: int
    failed_creators: int
    total_videos: int
    total_analysis_time: float
    start_time: datetime
    end_time: Optional[datetime] = None
    errors: List[str] = None

class BatchProcessor:
    """Batch processor that orchestrates crawling and analysis for creators"""
    
    def __init__(self, config):
        """Accepts ConfigManager or raw dict config. Normalizes to dict internally."""
        # Normalize config
        if hasattr(config, 'config') and isinstance(getattr(config, 'config'), dict):
            self.config = config
            cfg_dict = config.config
        elif isinstance(config, dict):
            from config_manager import ConfigManager
            self.config = ConfigManager()
            self.config.config = config
            cfg_dict = config
        else:
            raise TypeError("BatchProcessor requires ConfigManager or dict config")

        self.tiktok_config = cfg_dict.get('tiktok', {})
        
        # Initialize components
        self.progress_manager = ProgressManager(
            progress_file=self.tiktok_config.get('progress_tracking', {}).get('progress_file', 'data/batch_progress.json'),
            auto_save_interval=self.tiktok_config.get('progress_tracking', {}).get('auto_save_interval', 60)
        )
        
        self.storage_manager = StorageManager(
            max_disk_usage_percent=self.tiktok_config.get('storage_management', {}).get('max_disk_usage_percent', 80),
            temp_file_retention_hours=self.tiktok_config.get('storage_management', {}).get('temp_file_retention_hours', 24),
            cleanup_strategy=self.tiktok_config.get('storage_management', {}).get('video_cleanup_strategy', 'after_analysis')
        )
        
        # Batch processing configuration
        self.batch_config = self.tiktok_config.get('batch_processing', {})
        self.batch_size = self.batch_config.get('batch_size', 5)
        self.max_concurrent_creators = self.batch_config.get('max_concurrent_creators', 2)
        self.delay_between_batches = self.batch_config.get('delay_between_batches', 10)
        
        # Creator limits
        self.creator_limits = self.tiktok_config.get('creator_limits', {})
        self.max_videos_per_creator = self.creator_limits.get('max_videos_per_creator', 30)
        self.max_creators_per_batch = self.creator_limits.get('max_creators_per_batch', 10)
        
        # Enhanced analysis toggles
        self.enhanced_analysis = self.tiktok_config.get('enhanced_analysis', {})
        
        logger.info(f"✅ Batch processor ready: batch_size={self.batch_size}, concurrency={self.max_concurrent_creators}")
    
    async def process_creators_from_csv(self, csv_path: str, batch_id: Optional[str] = None) -> str:
        """Process a list of creators loaded from a CSV file"""
        try:
            # Read CSV file with encoding fallbacks
            creators_df, used_enc = self._read_creators_csv(csv_path)
            logger.info(f"📊 Loaded {len(creators_df)} creators from CSV (encoding={used_enc})")
            
            # Build tasks
            creator_tasks = []
            for _, row in creators_df.iterrows():
                task = CreatorTask(
                    username=row.get('username', ''),
                    platform=row.get('platform', 'tiktok'),
                    category=row.get('category', ''),
                    notes=row.get('notes', ''),
                    max_videos=self.max_videos_per_creator
                )
                creator_tasks.append(task)
            
            # Generate batch id if not provided
            if batch_id is None:
                batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create batch entry in progress store
            self.progress_manager.create_batch(batch_id, [task.username for task in creator_tasks])
            
            # Process in batches
            await self._process_creators_in_batches(creator_tasks, batch_id)
            
            return batch_id
            
        except Exception as e:
            logger.error(f"❌ Failed to process CSV: {e}")
            raise

    def _read_creators_csv(self, csv_path: str):
        """Read creators CSV with common encoding fallbacks.
        Returns: (DataFrame, used_encoding)
        """
        encodings = [
            'utf-8',
            'utf-8-sig',
            'gb18030',  # Simplified Chinese superset
            'big5',     # Traditional Chinese
            'shift_jis',
            'latin-1',
        ]
        last_err = None
        for enc in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                # normalize column names lower
                df.columns = [str(c).strip() for c in df.columns]
                return df, enc
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"Unable to read CSV with common encodings. Last error: {last_err}")
    
    async def _process_creators_in_batches(self, creator_tasks: List[CreatorTask], batch_id: str) -> None:
        """Process creators in batches according to config"""
        total_creators = len(creator_tasks)
        logger.info(f"🚀 Starting batch processing for {total_creators} creators, batch_size={self.batch_size}")
        
        # Iterate by batch
        for i in range(0, total_creators, self.batch_size):
            batch_tasks = creator_tasks[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_creators + self.batch_size - 1) // self.batch_size
            
            logger.info(f"📦 Processing batch {batch_num}/{total_batches}: {len(batch_tasks)} creators")
            
            # Process current batch
            await self._process_single_batch(batch_tasks, batch_id, batch_num)
            
            # Inter-batch delay
            if i + self.batch_size < total_creators:
                logger.info(f"⏳ Waiting {self.delay_between_batches} seconds before next batch...")
                await asyncio.sleep(self.delay_between_batches)
        
        logger.info(f"✅ All batches completed: {batch_id}")
    
    async def _process_single_batch(self, batch_tasks: List[CreatorTask], batch_id: str, batch_num: int) -> None:
        """Process one batch of creators"""
        # 检查磁盘空间
        if not self.storage_manager.can_continue_processing():
            logger.warning("⚠️ Low disk space, forcing cleanup")
            self.storage_manager.force_cleanup()
            
            if not self.storage_manager.can_continue_processing():
                logger.error("❌ Disk space still low, aborting batch")
                return
        
        # Concurrency control for creators
        semaphore = asyncio.Semaphore(self.max_concurrent_creators)
        
        tasks = []
        for task in batch_tasks:
            task_wrapper = self._process_creator_with_semaphore(task, batch_id, batch_num, semaphore)
            tasks.append(task_wrapper)
        
        # Wait for all creators in this batch
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate batch results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = sum(1 for r in results if isinstance(r, Exception))
        
        logger.info(f"📊 Batch {batch_num} finished: success={successful}, failed={failed}")
    
    async def _process_creator_with_semaphore(self, task: CreatorTask, batch_id: str, 
                                            batch_num: int, semaphore: asyncio.Semaphore):
        """Wrapper that applies the semaphore to creator processing"""
        async with semaphore:
            return await self._process_single_creator(task, batch_id, batch_num)
    
    async def _process_single_creator(self, task: CreatorTask, batch_id: str, batch_num: int) -> Dict[str, Any]:
        """Process a single creator: crawl and analyze videos"""
        username = task.username
        logger.info(f"👤 Processing creator: {username}")
        
        try:
            # Update status: processing
            self.progress_manager.update_creator_status(batch_id, username, "processing")
            
            # Check storage
            if not self.storage_manager.can_continue_processing():
                raise Exception("Insufficient disk space")
            
            # Crawl creator videos
            videos = await self._crawl_creator_videos(username, task.max_videos)
            
            if not videos:
                logger.warning(f"⚠️ No videos found for creator {username}")
                self.progress_manager.update_creator_status(
                    batch_id, username, "completed", 0, 0, 0
                )
                return {"username": username, "status": "no_videos", "videos": []}
            
            # Update video count
            self.progress_manager.update_creator_status(
                batch_id, username, "processing", len(videos), 0, 0
            )
            
            # Analyze videos
            analysis_results = await self._analyze_creator_videos(videos, username, batch_id)

            # Persist per-creator results promptly to minimize memory/IO bursts
            self._save_creator_results(username, analysis_results, batch_id)

            # Aggregate per-creator final outputs and clean creator folder to keep only final aggregate
            try:
                await self._aggregate_creator_final_outputs(username, videos, analysis_results)
            except Exception as agg_e:
                logger.warning(f"Creator-level aggregation failed for {username}: {agg_e}")
            
            # Update completion status
            successful_videos = len([r for r in analysis_results if r.get('status') == 'success'])
            failed_videos = len(analysis_results) - successful_videos
            
            self.progress_manager.update_creator_status(
                batch_id, username, "completed", 
                len(videos), successful_videos, failed_videos
            )
            
            # Cleanup temporary files according to strategy
            if self.storage_manager.cleanup_strategy == "after_analysis":
                cleanup_stats = self.storage_manager.cleanup_creator_files(username, batch_id)
                logger.debug(f"🧹 Cleanup for creator {username}: {cleanup_stats}")
            
            logger.info(f"✅ Creator {username} done: {successful_videos}/{len(videos)} videos succeeded")
            
            return {
                "username": username,
                "status": "success",
                "videos_found": len(videos),
                "videos_analyzed": successful_videos,
                "videos_failed": failed_videos,
                "analysis_results": analysis_results
            }
            
        except Exception as e:
            error_msg = f"Processing creator {username} failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            # 更新失败状态
            self.progress_manager.update_creator_status(
                batch_id, username, "failed", 0, 0, 0, error_msg
            )
            
            return {
                "username": username,
                "status": "failed",
                "error": str(e)
            }

    def _save_creator_results(self, username: str, analysis_results: List[Dict], batch_id: str) -> None:
        """Write per-creator CSV and JSON files with minimal IO.
        Output layout: data/creators/{username}/{username}_videos.csv and .json
        """
        try:
            from pathlib import Path
            import json as _json
            import pandas as _pd
            from collections import Counter

            base_dir = Path('data/creators') / username
            base_dir.mkdir(parents=True, exist_ok=True)
            json_path = base_dir / f"{username}_videos.json"
            csv_path = base_dir / f"{username}_videos.csv"
            # Also save a minimal summary for quick indexing
            summary_csv_path = base_dir / f"{username}_summary.csv"
            # Run-level summary path
            run_dir = Path('data/run_summaries')
            run_dir.mkdir(parents=True, exist_ok=True)
            run_summary_path = run_dir / f"{batch_id}_summary.csv"

            # Flatten records for CSV
            flat_rows = []
            for item in analysis_results:
                res = item.get('result', {})
                features = res.get('features', {})
                ad_flags = item.get('ad_flags', {})
                music = item.get('music', {})
                enhanced = res.get('enhanced_analysis', {}) if isinstance(res, dict) else {}
                promo = enhanced.get('promotion_analysis', {}) if isinstance(enhanced, dict) else {}
                flat_rows.append({
                    'video_id': item.get('video_id', ''),
                    'creator_username': features.get('creator_username', username),
                    'video_description': features.get('video_description', ''),
                    'hashtags': ','.join(features.get('hashtags', [])[:20]) if isinstance(features.get('hashtags'), list) else str(features.get('hashtags', '')),
                    'music_title': music.get('title', features.get('music_title', '')),
                    'music_author': music.get('author', features.get('music_author', '')),
                    'is_ad': ad_flags.get('is_ad', features.get('is_ad', False)),
                    'is_commerce': ad_flags.get('is_commerce', features.get('is_commerce', False)),
                    'ad_authorization': ad_flags.get('ad_authorization', features.get('ad_authorization', False)),
                    'likes': features.get('likes', 0),
                    'comments': features.get('comments', 0),
                    'shares': features.get('shares', 0),
                    'views': features.get('views', 0),
                    'duration': features.get('duration', 0),
                    'created_time': features.get('created_time', ''),
                    'primary_category': features.get('primary_category', 'Unknown'),
                    'secondary_category': features.get('secondary_category', 'Unknown'),
                    'tertiary_category': features.get('tertiary_category', 'Unknown'),
                    'promotion_is_promotional': promo.get('is_promotional', False),
                    'promotion_ad_authorization': promo.get('ad_authorization', False),
                    'promotion_brand_mentions': ','.join(promo.get('brand_mentions', [])[:20]) if isinstance(promo.get('brand_mentions'), list) else '',
                    'promotion_ctas': ','.join(promo.get('call_to_actions', [])[:20]) if isinstance(promo.get('call_to_actions'), list) else ''
                })

            # Save JSON (full results)
            with open(json_path, 'w', encoding='utf-8') as f:
                _json.dump(analysis_results, f, ensure_ascii=False, indent=2)

            # Save CSV (flattened)
            if flat_rows:
                df = _pd.DataFrame(flat_rows)
                df.to_csv(csv_path, index=False, encoding='utf-8')
                # Minimal summary: counts per ad/music flags
                # Pandas iat is an indexer, use bracket indexing
                top_music = ''
                if 'music_title' in df and not df['music_title'].dropna().empty:
                    m = df['music_title'].mode()
                    if not m.empty:
                        try:
                            top_music = m.iat[0]
                        except Exception:
                            top_music = m.iloc[0]
                summary = {
                    'creator_username': username,
                    'videos_total': int(len(df)),
                    'ads_count': int(df['is_ad'].sum()) if 'is_ad' in df else 0,
                    'commerce_count': int(df['is_commerce'].sum()) if 'is_commerce' in df else 0,
                    'top_music_title': top_music
                }
                _pd.DataFrame([summary]).to_csv(summary_csv_path, index=False, encoding='utf-8')
                # Append to run-level summary
                header_needed = not run_summary_path.exists()
                _pd.DataFrame([summary]).to_csv(run_summary_path, index=False, encoding='utf-8', mode='a', header=header_needed)

            logger.info(f"💾 Saved results for {username} -> {json_path}, {csv_path}")
        except Exception as e:
            logger.error(f"❌ Failed saving results for {username}: {e}")
    
    async def _aggregate_creator_final_outputs(self, username: str, videos: List[Any], analysis_results: List[Dict]) -> None:
        """Aggregate n video-level final analyses into a single creator-level summary.
        - Merge with creator API attributes if available (followers, etc.).
        - Write only two files in data/creators/{username}: creator_final_aggregate.json and .csv
        - Remove other files in that folder.
        """
        from pathlib import Path
        import json as _json
        import pandas as _pd
        import math
        from collections import Counter, defaultdict

        base_dir = Path('data/creators') / username
        base_dir.mkdir(parents=True, exist_ok=True)

        def safe_get(d: Dict, key: str, default=None):
            try:
                v = d.get(key, default)
                return default if v is None else v
            except Exception:
                return default

        def mode_or_empty(counter: Counter):
            return counter.most_common(1)[0][0] if counter else ''

        # Collect creator API attributes (best-effort)
        creator_api = {}
        try:
            src = videos[0] if videos else None
            ai = getattr(src, 'author_info', None)
            if isinstance(ai, dict):
                creator_api = {
                    'creator_username': ai.get('uniqueId') or ai.get('unique_id') or username,
                    'creator_nickname': ai.get('nickname') or ai.get('nickName') or '',
                    'followers': ai.get('followerCount') or ai.get('followers') or 0,
                    'following': ai.get('followingCount') or ai.get('following') or 0,
                    'hearts': ai.get('heartCount') or ai.get('likes') or 0,
                    'videos_count': ai.get('videoCount') or ai.get('videosCount') or 0,
                    'verified': bool(ai.get('verified', False)),
                }
            else:
                # Fallback: scan raw_data in videos for author fields
                for v in videos or []:
                    rd = getattr(v, 'raw_data', None)
                    if isinstance(rd, dict):
                        author = rd.get('author') or rd.get('authorStats') or {}
                        if isinstance(author, dict):
                            creator_api = {
                                'creator_username': author.get('uniqueId') or username,
                                'creator_nickname': author.get('nickname') or '',
                                'followers': author.get('followerCount') or 0,
                                'following': author.get('followingCount') or 0,
                                'hearts': author.get('heartCount') or 0,
                                'videos_count': author.get('videoCount') or 0,
                                'verified': bool(author.get('verified', False)),
                            }
                            break
        except Exception:
            pass

        # Aggregate across videos
        num_videos = 0
        sums = defaultdict(float)
        counts = defaultdict(int)
        cat_counters = defaultdict(Counter)
        list_counters = defaultdict(Counter)

        def add_num(key: str, value):
            try:
                if value is None:
                    return
                val = float(value)
                if not math.isfinite(val):
                    return
                sums[key] += val
                counts[key] += 1
            except Exception:
                pass

        def add_cat(key: str, value: str):
            if not value:
                return
            cat_counters[key][str(value)] += 1

        def add_list(key: str, value):
            if not value:
                return
            if isinstance(value, str):
                items = [x.strip() for x in value.split(',') if x.strip()]
            elif isinstance(value, list):
                items = [str(x).strip() for x in value if str(x).strip()]
            else:
                return
            for it in items:
                list_counters[key][it] += 1

        ad_true = commerce_true = auth_true = 0

        for item in analysis_results or []:
            res = item.get('result', {})
            feats = res.get('features', {}) if isinstance(res, dict) else {}
            num_videos += 1

            # Numeric engagement
            add_num('likes', safe_get(feats, 'likes', 0))
            add_num('comments', safe_get(feats, 'comments', 0))
            add_num('shares', safe_get(feats, 'shares', 0))
            add_num('views', safe_get(feats, 'views', 0))
            add_num('duration', safe_get(feats, 'duration', 0))

            # Booleans as rates
            if bool(safe_get(feats, 'is_ad', False)):
                ad_true += 1
            if bool(safe_get(feats, 'is_commerce', False)):
                commerce_true += 1
            if bool(safe_get(feats, 'ad_authorization', False)):
                auth_true += 1

            # Categories
            add_cat('primary_category', safe_get(feats, 'primary_category', ''))
            add_cat('secondary_category', safe_get(feats, 'secondary_category', ''))
            add_cat('tertiary_category', safe_get(feats, 'tertiary_category', ''))

            # Lists / strings to lists
            add_list('hashtags', safe_get(feats, 'hashtags', []))
            add_list('product_key_products', safe_get(feats, 'product_key_products', ''))
            add_list('product_brands_mentioned', safe_get(feats, 'product_brands_mentioned', ''))
            add_list('business_call_to_actions', safe_get(feats, 'business_call_to_actions', ''))
            add_list('business_engagement_drivers', safe_get(feats, 'business_engagement_drivers', ''))
            add_list('performance_trend_alignment', safe_get(feats, 'performance_trend_alignment', ''))

            # Other categories
            add_cat('business_commercial_intent', safe_get(feats, 'business_commercial_intent', ''))
            add_cat('business_branding_integration', safe_get(feats, 'business_branding_integration', ''))
            add_cat('performance_engagement_potential', safe_get(feats, 'performance_engagement_potential', ''))

        # Build aggregate
        agg = {
            'creator_username': username,
            'videos_analyzed': num_videos,
            'likes_total': int(sums['likes']),
            'likes_avg': round(sums['likes'] / counts['likes'], 2) if counts['likes'] else 0.0,
            'comments_total': int(sums['comments']),
            'comments_avg': round(sums['comments'] / counts['comments'], 2) if counts['comments'] else 0.0,
            'shares_total': int(sums['shares']),
            'shares_avg': round(sums['shares'] / counts['shares'], 2) if counts['shares'] else 0.0,
            'views_total': int(sums['views']),
            'views_avg': round(sums['views'] / counts['views'], 2) if counts['views'] else 0.0,
            'duration_total': int(sums['duration']),
            'duration_avg': round(sums['duration'] / counts['duration'], 2) if counts['duration'] else 0.0,
            'ad_rate': round(ad_true / num_videos, 3) if num_videos else 0.0,
            'commerce_rate': round(commerce_true / num_videos, 3) if num_videos else 0.0,
            'ad_authorization_rate': round(auth_true / num_videos, 3) if num_videos else 0.0,
            'top_primary_category': mode_or_empty(cat_counters['primary_category']),
            'top_secondary_category': mode_or_empty(cat_counters['secondary_category']),
            'top_tertiary_category': mode_or_empty(cat_counters['tertiary_category']),
            'top_business_commercial_intent': mode_or_empty(cat_counters['business_commercial_intent']),
            'top_branding_integration': mode_or_empty(cat_counters['business_branding_integration']),
            'top_engagement_potential': mode_or_empty(cat_counters['performance_engagement_potential']),
            'hashtags_top': [k for k, _ in list_counters['hashtags'].most_common(20)],
            'product_key_products_top': [k for k, _ in list_counters['product_key_products'].most_common(20)],
            'brands_top': [k for k, _ in list_counters['product_brands_mentioned'].most_common(20)],
            'call_to_actions_top': [k for k, _ in list_counters['business_call_to_actions'].most_common(20)],
            'engagement_drivers_top': [k for k, _ in list_counters['business_engagement_drivers'].most_common(20)],
            'trend_alignment_top': [k for k, _ in list_counters['performance_trend_alignment'].most_common(20)],
            'distributions': {
                'primary_category': dict(cat_counters['primary_category']),
                'secondary_category': dict(cat_counters['secondary_category']),
                'tertiary_category': dict(cat_counters['tertiary_category']),
                'business_commercial_intent': dict(cat_counters['business_commercial_intent']),
                'branding_integration': dict(cat_counters['business_branding_integration']),
                'performance_engagement_potential': dict(cat_counters['performance_engagement_potential']),
            },
            'creator_api': creator_api,
        }

        # Save JSON
        json_path = base_dir / 'creator_final_aggregate.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            _json.dump(agg, f, ensure_ascii=False, indent=2)

        # Save CSV (flatten list fields)
        csv_row = {}
        for k, v in agg.items():
            if isinstance(v, list):
                csv_row[k] = ','.join([str(x) for x in v])
            elif isinstance(v, dict):
                # For nested dicts, JSON-encode into a single cell
                csv_row[k] = _json.dumps(v, ensure_ascii=False)
            else:
                csv_row[k] = v
        csv_path = base_dir / 'creator_final_aggregate.csv'
        _pd.DataFrame([csv_row]).to_csv(csv_path, index=False, encoding='utf-8')

        # Cleanup: keep only the final aggregate files in creator folder
        try:
            keep = {'creator_final_aggregate.json', 'creator_final_aggregate.csv'}
            for name in os.listdir(base_dir):
                if name in keep:
                    continue
                try:
                    (base_dir / name).unlink()
                except IsADirectoryError:
                    # Remove directories if any
                    import shutil
                    shutil.rmtree(base_dir / name, ignore_errors=True)
                except Exception:
                    pass
        except Exception:
            pass

    async def _crawl_creator_videos(self, username: str, max_videos: int) -> List[Dict]:
        """Crawl creator videos using TikTokCrawler"""
        try:
            # Extract plain tiktok settings
            cfg_dict = self.config.config if hasattr(self.config, 'config') else self.config
            tcfg = cfg_dict.get('tiktok', {})
            ms_token = tcfg.get('ms_token', '')
            browser = tcfg.get('browser', 'chromium')
            download_dir = tcfg.get('download_dir', 'data/tiktok_videos')
            max_retries = tcfg.get('max_retries', 3)
            headless = bool(tcfg.get('playwright_headless', True))
            cookies_file = tcfg.get('cookies_file')
            custom_user_agent = tcfg.get('custom_user_agent')
            proxy_url = None
            if bool(tcfg.get('use_proxy', False)):
                proxy_cfg = tcfg.get('proxy_config', {})
                host = proxy_cfg.get('host')
                port = proxy_cfg.get('port')
                user = proxy_cfg.get('username')
                pwd = proxy_cfg.get('password')
                if host and port:
                    if user and pwd:
                        proxy_url = f"http://{user}:{pwd}@{host}:{port}"
                    else:
                        proxy_url = f"http://{host}:{port}"
            prefer_yt_dlp_first = bool(tcfg.get('prefer_yt_dlp_first', False))

            enable_download = bool(tcfg.get('enable_video_download', False))
            async with TikTokCrawler(ms_token=ms_token,
                                     browser=browser,
                                     download_dir=download_dir,
                                     max_retries=max_retries,
                                     headless=headless,
                                     cookies_file=cookies_file,
                                     custom_user_agent=custom_user_agent,
                                     proxy_url=proxy_url,
                                     prefer_yt_dlp_first=prefer_yt_dlp_first) as crawler:
                videos = await crawler.get_creator_videos(username, max_videos)

                # Optionally download videos for full multimodal analysis
                if enable_download:
                    for v in videos:
                        try:
                            path = await crawler.download_video(v)
                            if path:
                                v.download_path = path
                                self.storage_manager.register_temp_file(
                                    path, "video", username, ""
                                )
                        except Exception as _e:
                            logger.warning(f"Video download failed for {getattr(v, 'video_id', '')}: {_e}")
                
                return videos
                
        except Exception as e:
            logger.error(f"❌ Crawling videos failed for {username}: {e}")
            raise
    
    async def _analyze_creator_videos(self, videos: List, username: str, batch_id: str) -> List[Dict]:
        """Analyze a list of videos for a given creator"""
        try:
            # Initialize feature extractor (respect config to use full multimodal)
            cfg_dict = self.config.config if hasattr(self.config, 'config') else self.config
            use_lightweight = bool(cfg_dict.get('tiktok', {}).get('use_lightweight_analyzer', False))
            feature_extractor = TikTokFeatureExtractor(self.config, lightweight=use_lightweight)
            analysis_results = []
            
            for i, video in enumerate(videos):
                try:
                    logger.debug(f"🔍 Analyzing video {i+1}/{len(videos)}: {getattr(video, 'video_id', 'unknown')}")
                    
                    # Analyze: if local file exists -> full pipeline; else try metadata-only
                    result = await feature_extractor.analyze_video(video)
                    if result.get('mode') == 'metadata-only' and not use_lightweight:
                        # If we expected full pipeline but no file present, log a warning
                        logger.warning(f"Full analysis requested but no local file for {getattr(video, 'video_id','')}. Running metadata-only.")
                    
                    # Optional: promotion enhancement
                    if self.enhanced_analysis.get('promotion_detection'):
                        result = await self._enhance_analysis_with_promotion_data(result, video)
                    
                    analysis_results.append({
                        "video_id": getattr(video, 'video_id', 'unknown'),
                        "status": "success",
                        "result": result,
                        "ad_flags": {
                            "is_ad": bool(getattr(video, 'is_ad', False) or getattr(video, 'isAd', False)),
                            "is_commerce": bool(getattr(video, 'is_commerce', False) or getattr(video, 'isCommerce', False)),
                            "ad_authorization": bool(getattr(video, 'ad_authorization', False) or getattr(video, 'adAuthorization', False))
                        },
                        "music": {
                            "title": getattr(video, 'music_title', ''),
                            "author": getattr(video, 'music_author', '')
                        }
                    })
                    
                    # Update progress
                    self.progress_manager.update_creator_status(
                        batch_id, username, "processing", 
                        len(videos), i + 1, 0
                    )
                    
                except Exception as e:
                    logger.error(f"❌ Video analysis failed: {e}")
                    analysis_results.append({
                        "video_id": getattr(video, 'video_id', 'unknown'),
                        "status": "failed",
                        "error": str(e)
                    })
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Creator-level analysis failed for {username}: {e}")
            raise
    
    async def _enhance_analysis_with_promotion_data(self, analysis_result: Dict, video: Any) -> Dict:
        """Attach promotion-related signals to the model analysis result"""
        try:
            # Extract promotion-related signals from the crawled video
            promotion_data = {
                "is_promotional": getattr(video, 'isAd', False) or getattr(video, 'isCommerce', False),
                "ad_authorization": getattr(video, 'adAuthorization', False),
                "brand_mentions": self._extract_brand_mentions(video),
                "partnership_indicators": self._extract_partnership_indicators(video),
                "call_to_actions": self._extract_call_to_actions(video)
            }
            
            # Attach to analysis result
            if 'enhanced_analysis' not in analysis_result:
                analysis_result['enhanced_analysis'] = {}
            
            analysis_result['enhanced_analysis']['promotion_analysis'] = promotion_data
            
            return analysis_result
            
        except Exception as e:
            logger.warning(f"⚠️ Promotion enhancement failed: {e}")
            return analysis_result
    
    def _extract_brand_mentions(self, video: Any) -> List[str]:
        """Extract brand mentions from description and hashtags"""
        brands = []
        try:
            # Mentions from description
            description = getattr(video, 'video_description', '') or getattr(video, 'desc', '')
            if description:
                import re
                mentions = re.findall(r'@(\w+)', description)
                brands.extend(mentions)
            
            # Hashtags that imply brand cooperation
            hashtags = getattr(video, 'hashtags', [])
            if hashtags:
                for tag in hashtags:
                    tag_text = getattr(tag, 'name', str(tag))
                    if any(brand in tag_text.lower() for brand in ['brand', 'partner', 'sponsored']):
                        brands.append(tag_text)
            
        except Exception as e:
            logger.debug(f"Brand mention extraction failed: {e}")
        
        return list(set(brands))
    
    def _extract_partnership_indicators(self, video: Any) -> List[str]:
        """Extract partnership indicators from description"""
        indicators = []
        try:
            description = getattr(video, 'video_description', '') or getattr(video, 'desc', '')
            if description:
                import re
                # partnership-related keywords
                partnership_keywords = ['partner', 'sponsored', 'ad', 'collab', 'brand', 'ambassador']
                for keyword in partnership_keywords:
                    if keyword.lower() in description.lower():
                        indicators.append(keyword)
            
        except Exception as e:
            logger.debug(f"Partnership indicator extraction failed: {e}")
        
        return list(set(indicators))
    
    def _extract_call_to_actions(self, video: Any) -> List[str]:
        """Extract call-to-action phrases from description"""
        ctas = []
        try:
            description = getattr(video, 'video_description', '') or getattr(video, 'desc', '')
            if description:
                # Common CTA phrases
                cta_patterns = [
                    'link in bio', 'swipe up', 'click link', 'visit', 'buy now',
                    'shop now', 'download', 'sign up', 'follow', 'like', 'comment'
                ]
                
                for pattern in cta_patterns:
                    if pattern.lower() in description.lower():
                        ctas.append(pattern)
            
        except Exception as e:
            logger.debug(f"CTA extraction failed: {e}")
        
        return list(set(ctas))
    
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get batch-level status summary"""
        return self.progress_manager.get_progress_summary()
    
    def can_resume_batch(self, batch_id: str) -> bool:
        """Check whether a batch can be resumed"""
        return self.progress_manager.can_resume(batch_id)
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """Get storage usage summary"""
        return self.storage_manager.get_storage_summary()
    
    def cleanup_completed_batches(self, max_retention_days: int = 7) -> None:
        """Cleanup completed batches older than retention"""
        self.progress_manager.cleanup_completed_batches(max_retention_days)
    
    def force_cleanup_storage(self, min_free_space_gb: float = 5.0) -> Dict[str, int]:
        """Force cleanup to free disk space"""
        return self.storage_manager.force_cleanup(min_free_space_gb)
