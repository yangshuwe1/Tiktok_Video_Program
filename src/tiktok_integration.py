import asyncio
import os
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import time

from tiktok_crawler import TikTokCrawler, TikTokVideo
from tiktok_feature_extractor import TikTokFeatureExtractor
from config_manager import get_config

class TikTokIntegration:
    """Integration module that combines TikTok crawling with video analysis"""
    
    def __init__(self, config=None):
        """
        Initialize TikTok integration
        
        Args:
            config: Configuration manager instance or None to use default
        """
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize the existing feature extractor
        self.feature_extractor = TikTokFeatureExtractor(self.config)
        
        # Get TikTok configuration
        self.tiktok_config = self.config.get_tiktok_config()
        
    def get_tiktok_config(self) -> Dict:
        """Get TikTok-specific configuration"""
        return {
            'ms_token': os.environ.get('TIKTOK_MS_TOKEN'),
            'browser': os.environ.get('TIKTOK_BROWSER', 'chromium'),
            'download_dir': 'data/tiktok_videos',
            'max_videos_per_creator': 30,
            'enable_video_download': False,  # Set to True if you want to download videos
            'rate_limit_delay': 2,  # Delay between API calls in seconds
            'max_retries': 3
        }
    
    async def crawl_and_analyze_creators(self, creators_csv_path: str, 
                                       output_dir: str = "data/analysis_results") -> Dict:
        """
        Main workflow: Crawl TikTok creators and analyze their videos
        
        Args:
            creators_csv_path: Path to CSV file with creator usernames
            output_dir: Directory to save analysis results
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        results = {
            'total_creators': 0,
            'total_videos': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'processing_time': 0,
            'errors': [],
            'analysis_results': []
        }
        
        try:
            # Step 1: Validate and read creators CSV
            if not os.path.exists(creators_csv_path):
                raise FileNotFoundError(f"Creators CSV not found: {creators_csv_path}")
            
            df = pd.read_csv(creators_csv_path)
            if 'username' not in df.columns:
                raise ValueError("CSV must contain 'username' column")
            
            creators = df['username'].dropna().unique()
            results['total_creators'] = len(creators)
            
            self.logger.info(f"Starting analysis for {len(creators)} creators")
            
            # Step 2: Initialize TikTok crawler
            tiktok_config = self.get_tiktok_config()
            if not tiktok_config['ms_token']:
                raise ValueError("TIKTOK_MS_TOKEN environment variable not set")
            
            async with TikTokCrawler(
                ms_token=tiktok_config['ms_token'],
                browser=tiktok_config['browser'],
                download_dir=tiktok_config['download_dir'],
                max_retries=tiktok_config['max_retries']
            ) as crawler:
                
                # Step 3: Process each creator
                for i, username in enumerate(creators):
                    try:
                        self.logger.info(f"Processing creator {i+1}/{len(creators)}: {username}")
                        
                        # Crawl creator's videos
                        videos = await crawler.get_creator_videos(
                            username, 
                            tiktok_config['max_videos_per_creator']
                        )
                        
                        if not videos:
                            self.logger.warning(f"No videos found for creator: {username}")
                            continue
                        
                        results['total_videos'] += len(videos)
                        
                        # Step 4: Analyze each video
                        creator_results = await self._analyze_creator_videos(
                            username, videos, output_dir
                        )
                        
                        results['analysis_results'].append(creator_results)
                        results['successful_analyses'] += len(creator_results['videos'])
                        
                        # Rate limiting
                        if i < len(creators) - 1:  # Don't delay after last creator
                            await asyncio.sleep(tiktok_config['rate_limit_delay'])
                        
                    except Exception as e:
                        error_msg = f"Error processing creator {username}: {e}"
                        self.logger.error(error_msg)
                        results['errors'].append(error_msg)
                        results['failed_analyses'] += 1
                        continue
            
            # Step 5: Generate summary report
            results['processing_time'] = time.time() - start_time
            self._generate_summary_report(results, output_dir)
            
            self.logger.info(f"Analysis completed. Processed {results['total_videos']} videos from {results['total_creators']} creators")
            
        except Exception as e:
            error_msg = f"Fatal error in crawl_and_analyze_creators: {e}"
            self.logger.error(error_msg)
            results['errors'].append(error_msg)
            results['processing_time'] = time.time() - start_time
        
        return results
    
    async def _analyze_creator_videos(self, username: str, videos: List[TikTokVideo], 
                                    output_dir: str) -> Dict:
        """
        Analyze videos for a specific creator
        
        Args:
            username: Creator's username
            videos: List of TikTokVideo objects
            output_dir: Base output directory
            
        Returns:
            Dictionary with analysis results for the creator
        """
        creator_results = {
            'creator_username': username,
            'total_videos': len(videos),
            'videos': [],
            'summary_stats': {}
        }
        
        # Create creator-specific output directory
        creator_output_dir = Path(output_dir) / username
        creator_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw video data
        self._save_creator_video_data(username, videos, creator_output_dir)
        
        # Analyze each video if it has a local file
        for video in videos:
            try:
                if video.download_path and os.path.exists(video.download_path):
                    # Use existing feature extractor
                    video_analysis = await self._analyze_single_video(
                        video, creator_output_dir
                    )
                    creator_results['videos'].append(video_analysis)
                else:
                    # Video not downloaded, create metadata-only analysis
                    video_analysis = self._create_metadata_analysis(video)
                    creator_results['videos'].append(video_analysis)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing video {video.video_id}: {e}")
                # Create error analysis record
                error_analysis = self._create_error_analysis(video, str(e))
                creator_results['videos'].append(error_analysis)
        
        # Calculate summary statistics
        creator_results['summary_stats'] = self._calculate_creator_stats(creator_results['videos'])
        
        return creator_results
    
    async def _analyze_single_video(self, video: TikTokVideo, output_dir: Path) -> Dict:
        """
        Analyze a single video using the existing feature extractor
        
        Args:
            video: TikTokVideo object
            output_dir: Output directory for analysis results
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Create video-specific output directory
            video_output_dir = output_dir / video.video_id
            video_output_dir.mkdir(exist_ok=True)
            
            # Use existing feature extractor
            analysis_result = self.feature_extractor.extract_video_features(
                video.download_path, 
                str(video_output_dir)
            )
            
            # Add TikTok-specific metadata
            analysis_result.update({
                'tiktok_video_id': video.video_id,
                'tiktok_creator_username': video.creator_username,
                'tiktok_creator_nickname': video.creator_nickname,
                'tiktok_description': video.video_description,
                'tiktok_tags': video.tags,
                'tiktok_hashtags': video.hashtags,
                'tiktok_likes': video.likes,
                'tiktok_comments': video.comments,
                'tiktok_shares': video.shares,
                'tiktok_views': video.views,
                'tiktok_duration': video.duration,
                'tiktok_created_time': video.created_time.isoformat() if video.created_time else "",
                'tiktok_music_title': video.music_title,
                'tiktok_music_author': video.music_author
            })
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in video analysis: {e}")
            return self._create_error_analysis(video, str(e))
    
    def _create_metadata_analysis(self, video: TikTokVideo) -> Dict:
        """Create analysis record for videos without local files"""
        return {
            'video_path': '',
            'tiktok_video_id': video.video_id,
            'tiktok_creator_username': video.creator_username,
            'tiktok_creator_nickname': video.creator_nickname,
            'tiktok_description': video.video_description,
            'tiktok_tags': video.tags,
            'tiktok_hashtags': video.hashtags,
            'tiktok_likes': video.likes,
            'tiktok_comments': video.comments,
            'tiktok_shares': video.shares,
            'tiktok_views': video.views,
            'tiktok_duration': video.duration,
            'tiktok_created_time': video.created_time.isoformat() if video.created_time else "",
            'tiktok_music_title': video.music_title,
            'tiktok_music_author': video.music_author,
            'analysis_status': 'metadata_only',
            'error_message': 'Video file not available for analysis'
        }
    
    def _create_error_analysis(self, video: TikTokVideo, error_msg: str) -> Dict:
        """Create analysis record for videos with errors"""
        return {
            'video_path': video.download_path or '',
            'tiktok_video_id': video.video_id,
            'tiktok_creator_username': video.creator_username,
            'tiktok_creator_nickname': video.creator_nickname,
            'tiktok_description': video.video_description,
            'tiktok_tags': video.tags,
            'tiktok_hashtags': video.hashtags,
            'tiktok_likes': video.likes,
            'tiktok_comments': video.comments,
            'tiktok_shares': video.shares,
            'tiktok_views': video.views,
            'tiktok_duration': video.duration,
            'tiktok_created_time': video.created_time.isoformat() if video.created_time else "",
            'tiktok_music_title': video.music_title,
            'tiktok_music_author': video.music_author,
            'analysis_status': 'error',
            'error_message': error_msg
        }
    
    def _save_creator_video_data(self, username: str, videos: List[TikTokVideo], output_dir: Path):
        """Save raw video data for a creator"""
        try:
            # Save to CSV
            video_data = [video.to_dict() for video in videos]
            df = pd.DataFrame(video_data)
            csv_path = output_dir / f"{username}_videos.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            # Save to JSON
            import json
            json_path = output_dir / f"{username}_videos.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(video_data, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Saved video data for {username} to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving video data for {username}: {e}")
    
    def _calculate_creator_stats(self, videos: List[Dict]) -> Dict:
        """Calculate summary statistics for a creator's videos"""
        if not videos:
            return {}
        
        stats = {
            'total_videos': len(videos),
            'total_likes': 0,
            'total_comments': 0,
            'total_shares': 0,
            'total_views': 0,
            'avg_likes': 0,
            'avg_comments': 0,
            'avg_shares': 0,
            'avg_views': 0,
            'successful_analyses': 0,
            'metadata_only': 0,
            'errors': 0
        }
        
        for video in videos:
            # Engagement metrics
            stats['total_likes'] += video.get('tiktok_likes', 0)
            stats['total_comments'] += video.get('tiktok_comments', 0)
            stats['total_shares'] += video.get('tiktok_shares', 0)
            stats['total_views'] += video.get('tiktok_views', 0)
            
            # Analysis status
            status = video.get('analysis_status', 'unknown')
            if status == 'successful':
                stats['successful_analyses'] += 1
            elif status == 'metadata_only':
                stats['metadata_only'] += 1
            elif status == 'error':
                stats['errors'] += 1
        
        # Calculate averages
        if stats['total_videos'] > 0:
            stats['avg_likes'] = stats['total_likes'] / stats['total_videos']
            stats['avg_comments'] = stats['total_comments'] / stats['total_videos']
            stats['avg_shares'] = stats['total_shares'] / stats['total_videos']
            stats['avg_views'] = stats['total_views'] / stats['total_videos']
        
        return stats
    
    def _generate_summary_report(self, results: Dict, output_dir: str):
        """Generate a comprehensive summary report"""
        try:
            report_path = Path(output_dir) / "analysis_summary.md"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# TikTok Creator Analysis Summary Report\n\n")
                f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Overview\n\n")
                f.write(f"- **Total Creators Processed:** {results['total_creators']}\n")
                f.write(f"- **Total Videos Collected:** {results['total_videos']}\n")
                f.write(f"- **Successful Analyses:** {results['successful_analyses']}\n")
                f.write(f"- **Failed Analyses:** {results['failed_analyses']}\n")
                f.write(f"- **Total Processing Time:** {results['processing_time']:.2f} seconds\n\n")
                
                if results['errors']:
                    f.write("## Errors\n\n")
                    for error in results['errors']:
                        f.write(f"- {error}\n")
                    f.write("\n")
                
                f.write("## Creator Results\n\n")
                for creator_result in results['analysis_results']:
                    username = creator_result['creator_username']
                    stats = creator_result['summary_stats']
                    
                    f.write(f"### {username}\n\n")
                    f.write(f"- **Videos:** {stats['total_videos']}\n")
                    f.write(f"- **Total Views:** {stats['total_views']:,}\n")
                    f.write(f"- **Total Likes:** {stats['total_likes']:,}\n")
                    f.write(f"- **Avg Views:** {stats['avg_views']:,.0f}\n")
                    f.write(f"- **Avg Likes:** {stats['avg_likes']:,.0f}\n")
                    f.write(f"- **Analysis Status:** {stats['successful_analyses']} successful, {stats['metadata_only']} metadata-only, {stats['errors']} errors\n\n")
            
            self.logger.info(f"Summary report generated: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")

async def main():
    """Example usage of TikTokIntegration"""
    # Check environment variables
    if not os.environ.get('TIKTOK_MS_TOKEN'):
        print("Please set TIKTOK_MS_TOKEN environment variable")
        print("You can get this from your browser cookies on tiktok.com")
        return
    
    # Initialize integration
    integration = TikTokIntegration()
    
    # Process creators from CSV
    creators_csv = "data/creators.csv"
    if os.path.exists(creators_csv):
        results = await integration.crawl_and_analyze_creators(creators_csv)
        print(f"Analysis completed: {results['total_videos']} videos processed")
    else:
        print(f"Creators CSV not found: {creators_csv}")
        print("Please create a CSV file with 'username' column containing TikTok usernames")

if __name__ == "__main__":
    asyncio.run(main())

