#!/usr/bin/env python3
"""
Main entrypoint for batch creator analysis.
Reads a CSV of creators and orchestrates crawl + analysis.
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add current directory to PYTHONPATH for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_manager import ConfigManager
from batch_processor import BatchProcessor
from logger import get_logger

async def main():
    """主函数"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='TikTok creator batch analysis tool')
    parser.add_argument('--csv', '-c', required=True, help='Path to creators CSV')
    parser.add_argument('--config', '-f', default='config/config.yml', help='Path to config.yml')
    parser.add_argument('--batch-id', '-b', help='Specify batch id (optional)')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--dry-run', '-d', action='store_true', help='Dry run (no crawling or analysis)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--concurrency', type=int, help='Override max concurrent creators')
    parser.add_argument('--limit-per-creator', type=int, help='Override max videos per creator')
    
    args = parser.parse_args()
    
    # Set up basic logger
    logger = get_logger("main_batch")
    
    logger.info("🚀 Starting TikTok creator batch analysis")
    
    try:
        # Validate CSV path
        csv_path = Path(args.csv)
        if not csv_path.exists():
            logger.error(f"❌ CSV file not found: {csv_path}")
            return 1
        
        # Load config
        cm = ConfigManager(args.config)
        config = cm.config
        
        if not config:
            logger.error("❌ Failed to load config")
            return 1
        
        logger.info(f"✅ Config loaded: {args.config}")
        
        # Check TikTok config exists
        tiktok_config = config.get('tiktok', {})
        if not tiktok_config:
            logger.error("❌ Missing 'tiktok' section in config")
            return 1
        
        # Initialize batch processor
        batch_processor = BatchProcessor(cm)

        # Apply CLI overrides if provided
        if args.batch_size:
            batch_processor.batch_size = max(1, int(args.batch_size))
        if args.concurrency:
            batch_processor.max_concurrent_creators = max(1, int(args.concurrency))
        if args.limit_per_creator:
            batch_processor.max_videos_per_creator = max(1, int(args.limit_per_creator))
        
        # Dry-run validation
        if args.dry_run:
            logger.info("🔍 Dry-run: validating inputs and config only")
            
            # Validate CSV format
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                logger.info(f"✅ CSV valid: {len(df)} rows")
                logger.info(f"   Columns: {list(df.columns)}")
                
                # Preview first rows
                if len(df) > 0:
                    logger.info("📊 First 3 rows preview:")
                    for i, row in df.head(3).iterrows():
                        logger.info(f"   {i+1}: {dict(row)}")
                
                # Show key config values
                logger.info("⚙️ Config checks:")
                logger.info(f"   batch_processing: {tiktok_config.get('batch_processing', {}).get('enabled')}")
                logger.info(f"   batch_size: {tiktok_config.get('batch_processing', {}).get('batch_size')}")
                logger.info(f"   max_concurrent_creators: {tiktok_config.get('batch_processing', {}).get('max_concurrent_creators')}")
                logger.info(f"   progress_tracking: {tiktok_config.get('progress_tracking', {}).get('enabled')}")
                logger.info(f"   auto_cleanup: {tiktok_config.get('storage_management', {}).get('auto_cleanup')}")
                
                return 0
                
            except Exception as e:
                logger.error(f"❌ CSV validation failed: {e}")
                return 1
        
        # Resume support
        if args.resume and args.batch_id:
            if not batch_processor.can_resume_batch(args.batch_id):
                logger.error(f"❌ Cannot resume batch: {args.batch_id}")
                return 1
            
            logger.info(f"🔄 Resuming batch: {args.batch_id}")
        
        # Start processing
        logger.info(f"📊 Processing creators list: {csv_path}")
        logger.info(f"   File size: {csv_path.stat().st_size / 1024:.1f} KB")
        
        # Print key runtime config
        logger.info("⚙️ Runtime config:")
        logger.info(f"   batch_size: {batch_processor.batch_size}")
        logger.info(f"   concurrency: {batch_processor.max_concurrent_creators}")
        logger.info(f"   delay_between_batches: {batch_processor.delay_between_batches} sec")
        logger.info(f"   max_videos_per_creator: {batch_processor.max_videos_per_creator}")
        
        # 执行批量处理
        start_time = asyncio.get_event_loop().time()
        
        batch_id = await batch_processor.process_creators_from_csv(
            str(csv_path), 
            batch_id=args.batch_id
        )
        
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        
        # Summary
        logger.info("📊 Batch processing finished!")
        logger.info(f"   batch_id: {batch_id}")
        logger.info(f"   elapsed: {total_time:.1f} sec")
        
        # Final status
        status = batch_processor.get_batch_status(batch_id)
        logger.info(f"   progress: {status}")
        
        # Storage summary
        storage_summary = batch_processor.get_storage_summary()
        logger.info(f"   storage: {storage_summary}")
        
        # Cleanup expired batches
        batch_processor.cleanup_completed_batches()
        
        logger.info("🎉 Done!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        return 1

def run_sync():
    """Run the async main synchronously"""
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return 130

if __name__ == "__main__":
    sys.exit(run_sync())
