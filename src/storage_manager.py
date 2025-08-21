import os
import shutil
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
try:
    import psutil  # Optional, for robust disk stats
except Exception:  # ImportError or runtime constraints
    psutil = None

logger = logging.getLogger(__name__)

class StorageManager:
    """Storage manager for disk space monitoring and temporary file cleanup"""
    
    def __init__(self, 
                 max_disk_usage_percent: int = 80,
                 temp_file_retention_hours: int = 24,
                 cleanup_strategy: str = "after_analysis",
                 min_free_space_gb: float = 5.0):
        self.max_disk_usage_percent = max_disk_usage_percent
        self.temp_file_retention_hours = temp_file_retention_hours
        self.cleanup_strategy = cleanup_strategy
        self.min_free_space_gb = float(min_free_space_gb)
        self.temp_files: List[Dict] = []
        
    def check_disk_space(self, path: str = "/app") -> Dict[str, float]:
        """Check disk space usage for a given path"""
        try:
            if psutil is not None:
                disk_usage = psutil.disk_usage(path)
                total = disk_usage.total
                used = disk_usage.used
                free = disk_usage.free
            else:
                # Fallback using shutil
                total, used, free = shutil.disk_usage(path)

            total_gb = total / (1024**3)
            used_gb = used / (1024**3)
            free_gb = free / (1024**3)
            usage_percent = (used / total) * 100 if total else 0.0
            
            space_info = {
                "total_gb": round(total_gb, 2),
                "used_gb": round(used_gb, 2),
                "free_gb": round(free_gb, 2),
                "usage_percent": round(usage_percent, 2),
                "is_critical": usage_percent >= self.max_disk_usage_percent
            }
            
            if space_info["is_critical"]:
                logger.warning(f"⚠️ Low disk space: {usage_percent:.1f}% used")
            else:
                logger.debug(f"💾 Disk usage: {usage_percent:.1f}%")
            
            return space_info
        except Exception as e:
            logger.error(f"❌ Failed to check disk space: {e}")
            return {
                "total_gb": 0,
                "used_gb": 0,
                "free_gb": 0,
                "usage_percent": 0,
                "is_critical": False
            }
    
    def can_continue_processing(self, path: str = "/app") -> bool:
        """Return True if disk space is acceptable.
        Accept if usage is not critical OR free space >= min_free_space_gb.
        """
        space_info = self.check_disk_space(path)
        if not space_info["is_critical"]:
            return True
        # Percent critical but enough absolute free space
        if space_info.get("free_gb", 0) >= self.min_free_space_gb:
            logging.getLogger(__name__).info(
                f"Proceeding despite high usage: free_gb={space_info.get('free_gb'):.2f} >= min_free_space_gb={self.min_free_space_gb:.2f}"
            )
            return True
        return False
    
    def register_temp_file(self, file_path: str, file_type: str = "video", 
                          creator: str = "", batch_id: str = "") -> None:
        """Register a temporary file for later cleanup"""
        if os.path.exists(file_path):
            file_info = {
                "path": file_path,
                "type": file_type,
                "creator": creator,
                "batch_id": batch_id,
                "size_mb": self._get_file_size_mb(file_path),
                "created_time": datetime.now(),
                "access_count": 0
            }
            self.temp_files.append(file_info)
            logger.debug(f"📝 Registered temp file: {file_path}")
    
    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in megabytes"""
        try:
            size_bytes = os.path.getsize(file_path)
            return round(size_bytes / (1024**2), 2)
        except:
            return 0.0
    
    def cleanup_temp_files(self, strategy: Optional[str] = None) -> Dict[str, int]:
        """Cleanup temporary files according to a strategy"""
        if strategy is None:
            strategy = self.cleanup_strategy
        
        cleanup_stats = {
            "files_removed": 0,
            "space_freed_mb": 0.0,
            "errors": 0
        }
        
        current_time = datetime.now()
        files_to_remove = []
        
        for file_info in self.temp_files:
            should_remove = False
            
            if strategy == "after_analysis":
                should_remove = True
            elif strategy == "after_batch":
                should_remove = True
            elif strategy == "time_based":
                age_hours = (current_time - file_info["created_time"]).total_seconds() / 3600
                should_remove = age_hours >= self.temp_file_retention_hours
            
            if should_remove and os.path.exists(file_info["path"]):
                try:
                    file_size_mb = file_info["size_mb"]
                    os.remove(file_info["path"])
                    cleanup_stats["files_removed"] += 1
                    cleanup_stats["space_freed_mb"] += file_size_mb
                    files_to_remove.append(file_info)
                    logger.debug(f"🗑️ Removed temp file: {file_info['path']}")
                except Exception as e:
                    cleanup_stats["errors"] += 1
                    logger.error(f"❌ Failed to remove {file_info['path']}: {e}")
        
        # Remove cleaned files from registry
        for file_info in files_to_remove:
            self.temp_files.remove(file_info)
        
        if cleanup_stats["files_removed"] > 0:
            logger.info(f"🧹 Cleanup complete: removed {cleanup_stats['files_removed']} files, "
                       f"freed {cleanup_stats['space_freed_mb']:.2f} MB")
        
        return cleanup_stats
    
    def cleanup_creator_files(self, creator: str, batch_id: str = "") -> Dict[str, int]:
        """Cleanup all temporary files for a specific creator"""
        cleanup_stats = {
            "files_removed": 0,
            "space_freed_mb": 0.0,
            "errors": 0
        }
        
        files_to_remove = []
        
        for file_info in self.temp_files:
            if file_info["creator"] == creator and (not batch_id or file_info["batch_id"] == batch_id):
                if os.path.exists(file_info["path"]):
                    try:
                        file_size_mb = file_info["size_mb"]
                        os.remove(file_info["path"])
                        cleanup_stats["files_removed"] += 1
                        cleanup_stats["space_freed_mb"] += file_size_mb
                        files_to_remove.append(file_info)
                        logger.debug(f"🗑️ Removed creator {creator} file: {file_info['path']}")
                    except Exception as e:
                        cleanup_stats["errors"] += 1
                        logger.error(f"❌ Failed to remove {file_info['path']}: {e}")
        
        # Remove cleaned files from registry
        for file_info in files_to_remove:
            self.temp_files.remove(file_info)
        
        if cleanup_stats["files_removed"] > 0:
            logger.info(f"🧹 Cleanup complete for creator {creator}: removed {cleanup_stats['files_removed']} files, "
                       f"freed {cleanup_stats['space_freed_mb']:.2f} MB")
        
        return cleanup_stats
    
    def cleanup_batch_files(self, batch_id: str) -> Dict[str, int]:
        """Cleanup all temporary files for a specific batch"""
        cleanup_stats = {
            "files_removed": 0,
            "space_freed_mb": 0.0,
            "errors": 0
        }
        
        files_to_remove = []
        
        for file_info in self.temp_files:
            if file_info["batch_id"] == batch_id:
                if os.path.exists(file_info["path"]):
                    try:
                        file_size_mb = file_info["size_mb"]
                        os.remove(file_info["path"])
                        cleanup_stats["files_removed"] += 1
                        cleanup_stats["space_freed_mb"] += file_size_mb
                        files_to_remove.append(file_info)
                        logger.debug(f"🗑️ Removed batch {batch_id} file: {file_info['path']}")
                    except Exception as e:
                        cleanup_stats["errors"] += 1
                        logger.error(f"❌ Failed to remove {file_info['path']}: {e}")
        
        # Remove cleaned files from registry
        for file_info in files_to_remove:
            self.temp_files.remove(file_info)
        
        if cleanup_stats["files_removed"] > 0:
            logger.info(f"🧹 Cleanup complete for batch {batch_id}: removed {cleanup_stats['files_removed']} files, "
                       f"freed {cleanup_stats['space_freed_mb']:.2f} MB")
        
        return cleanup_stats
    
    def get_storage_summary(self) -> Dict[str, any]:
        """Get a summary of temporary storage usage and disk space"""
        total_files = len(self.temp_files)
        total_size_mb = sum(f["size_mb"] for f in self.temp_files)
        
        # Summarize by type
        type_stats = {}
        for file_info in self.temp_files:
            file_type = file_info["type"]
            if file_type not in type_stats:
                type_stats[file_type] = {"count": 0, "size_mb": 0.0}
            type_stats[file_type]["count"] += 1
            type_stats[file_type]["size_mb"] += file_info["size_mb"]
        
        # Summarize by creator
        creator_stats = {}
        for file_info in self.temp_files:
            creator = file_info["creator"]
            if creator not in creator_stats:
                creator_stats[creator] = {"count": 0, "size_mb": 0.0}
            creator_stats[creator]["count"] += 1
            creator_stats[creator]["size_mb"] += file_info["size_mb"]
        
        return {
            "total_temp_files": total_files,
            "total_temp_size_mb": round(total_size_mb, 2),
            "type_distribution": type_stats,
            "creator_distribution": creator_stats,
            "disk_space": self.check_disk_space()
        }
    
    def force_cleanup(self, min_free_space_gb: float = 5.0) -> Dict[str, int]:
        """Force cleanup to ensure a minimum amount of free space"""
        space_info = self.check_disk_space()
        free_gb = space_info["free_gb"]
        
        if free_gb >= min_free_space_gb:
            logger.info(f"💾 Disk space is sufficient: {free_gb:.2f} GB free")
            return {"files_removed": 0, "space_freed_mb": 0.0, "errors": 0}
        
        logger.warning(f"⚠️ Low disk space, starting forced cleanup. Free: {free_gb:.2f} GB, target: {min_free_space_gb} GB")
        
        # Sort by creation time: remove older files first
        sorted_files = sorted(self.temp_files, key=lambda x: x["created_time"])
        
        cleanup_stats = {
            "files_removed": 0,
            "space_freed_mb": 0.0,
            "errors": 0
        }
        
        for file_info in sorted_files:
            if free_gb >= min_free_space_gb:
                break
            
            if os.path.exists(file_info["path"]):
                try:
                    file_size_mb = file_info["size_mb"]
                    os.remove(file_info["path"])
                    cleanup_stats["files_removed"] += 1
                    cleanup_stats["space_freed_mb"] += file_size_mb
                    free_gb += file_size_mb / 1024  # MB -> GB
                    logger.debug(f"🗑️ Forced cleanup removed: {file_info['path']}")
                except Exception as e:
                    cleanup_stats["errors"] += 1
                    logger.error(f"❌ Forced cleanup failed for {file_info['path']}: {e}")
        
        # Remove cleaned files from registry
        self.temp_files = [f for f in self.temp_files if os.path.exists(f["path"])]
        
        if cleanup_stats["files_removed"] > 0:
            logger.info(f"🧹 Forced cleanup complete: removed {cleanup_stats['files_removed']} files, "
                       f"freed {cleanup_stats['space_freed_mb']:.2f} MB")
        
        return cleanup_stats
    
    def cleanup_old_files(self, max_age_hours: Optional[int] = None) -> Dict[str, int]:
        """Cleanup registered temp files older than max_age_hours"""
        if max_age_hours is None:
            max_age_hours = self.temp_file_retention_hours
        
        current_time = datetime.now()
        files_to_remove = []
        
        for file_info in self.temp_files:
            age_hours = (current_time - file_info["created_time"]).total_seconds() / 3600
            if age_hours >= max_age_hours and os.path.exists(file_info["path"]):
                try:
                    os.remove(file_info["path"])
                    files_to_remove.append(file_info)
                    logger.debug(f"🗑️ Removed expired file: {file_info['path']} (age: {age_hours:.1f}h)")
                except Exception as e:
                    logger.error(f"❌ Failed to remove expired file {file_info['path']}: {e}")
        
        # Remove cleaned files from registry
        for file_info in files_to_remove:
            self.temp_files.remove(file_info)
        
        if files_to_remove:
            logger.info(f"🧹 Expired file cleanup: removed {len(files_to_remove)} files")
        
        return {"files_removed": len(files_to_remove), "space_freed_mb": 0.0, "errors": 0}
