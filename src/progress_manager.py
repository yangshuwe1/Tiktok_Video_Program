import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class CreatorProgress:
    """Processing progress for a single creator"""
    username: str
    status: str  # pending, processing, completed, failed
    videos_found: int = 0
    videos_processed: int = 0
    videos_failed: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    last_updated: Optional[datetime] = None

@dataclass
class BatchProgress:
    """Processing progress for a batch"""
    batch_id: str
    status: str  # pending, processing, completed, failed
    creators: List[CreatorProgress]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_creators: int = 0
    completed_creators: int = 0
    failed_creators: int = 0

class ProgressManager:
    """Progress manager responsible for checkpointing and persistence"""
    
    def __init__(self, progress_file: str, auto_save_interval: int = 60):
        self.progress_file = progress_file
        self.auto_save_interval = auto_save_interval
        self.last_save_time = time.time()
        self.progress_data: Dict[str, Any] = {}
        self.load_progress()
    
    def load_progress(self) -> None:
        """Load existing progress file if present"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert timestamp strings to datetime objects
                    self._convert_timestamps(data)
                    self.progress_data = data
                    logger.info(f"✅ Loaded progress file: {self.progress_file}")
            else:
                self.progress_data = {
                    "batches": {},
                    "overall_progress": {
                        "total_creators": 0,
                        "completed_creators": 0,
                        "failed_creators": 0,
                        "start_time": None,
                        "last_updated": None
                    }
                }
                logger.info("🆕 Initialized new progress store")
        except Exception as e:
            logger.error(f"❌ Failed to load progress file: {e}")
            self.progress_data = {
                "batches": {},
                "overall_progress": {
                    "total_creators": 0,
                    "completed_creators": 0,
                    "failed_creators": 0,
                    "start_time": None,
                    "last_updated": None
                }
            }
    
    def _convert_timestamps(self, data: Dict[str, Any]) -> None:
        """Convert timestamp strings to datetime objects in-place"""
        for batch in data.get("batches", {}).values():
            if batch.get("start_time"):
                try:
                    batch["start_time"] = datetime.fromisoformat(batch["start_time"])
                except:
                    batch["start_time"] = None
            
            if batch.get("end_time"):
                try:
                    batch["end_time"] = datetime.fromisoformat(batch["end_time"])
                except:
                    batch["end_time"] = None
            
            for creator in batch.get("creators", []):
                if creator.get("start_time"):
                    try:
                        creator["start_time"] = datetime.fromisoformat(creator["start_time"])
                    except:
                        creator["start_time"] = None
                
                if creator.get("end_time"):
                    try:
                        creator["end_time"] = datetime.fromisoformat(creator["end_time"])
                    except:
                        creator["end_time"] = None
                
                if creator.get("last_updated"):
                    try:
                        creator["last_updated"] = datetime.fromisoformat(creator["last_updated"])
                    except:
                        creator["last_updated"] = None
    
    def create_batch(self, batch_id: str, creator_usernames: List[str]) -> str:
        """Create a new batch entry"""
        batch = BatchProgress(
            batch_id=batch_id,
            status="pending",
            creators=[
                CreatorProgress(username=username, status="pending")
                for username in creator_usernames
            ],
            start_time=datetime.now(),
            total_creators=len(creator_usernames)
        )
        
        self.progress_data["batches"][batch_id] = asdict(batch)
        self.progress_data["overall_progress"]["total_creators"] += len(creator_usernames)
        self.progress_data["overall_progress"]["start_time"] = datetime.now().isoformat()
        
        logger.info(f"🆕 Created batch {batch_id} with {len(creator_usernames)} creators")
        self.save_progress()
        return batch_id
    
    def update_creator_status(self, batch_id: str, username: str, status: str, 
                            videos_found: int = 0, videos_processed: int = 0, 
                            videos_failed: int = 0, error_message: str = None) -> None:
        """Update a creator's processing status"""
        if batch_id not in self.progress_data["batches"]:
            logger.warning(f"⚠️ Batch {batch_id} does not exist")
            return
        
        batch = self.progress_data["batches"][batch_id]
        creator = None
        
        for c in batch["creators"]:
            if c["username"] == username:
                creator = c
                break
        
        if creator is None:
            logger.warning(f"⚠️ Creator {username} does not exist in batch {batch_id}")
            return
        
        # 更新状态
        creator["status"] = status
        creator["videos_found"] = videos_found
        creator["videos_processed"] = videos_processed
        creator["videos_failed"] = videos_failed
        creator["last_updated"] = datetime.now().isoformat()
        
        if status == "processing" and not creator["start_time"]:
            creator["start_time"] = datetime.now().isoformat()
        elif status in ["completed", "failed"]:
            creator["end_time"] = datetime.now().isoformat()
            if error_message:
                creator["error_message"] = error_message
        
        # Update batch status
        self._update_batch_status(batch_id)
        
        # Update overall progress
        self._update_overall_progress()
        
        # Auto-save
        self._auto_save()
    
    def _update_batch_status(self, batch_id: str) -> None:
        """Update batch-level status based on creator statuses"""
        batch = self.progress_data["batches"][batch_id]
        creators = batch["creators"]
        
        completed = sum(1 for c in creators if c["status"] == "completed")
        failed = sum(1 for c in creators if c["status"] == "failed")
        total = len(creators)
        
        batch["completed_creators"] = completed
        batch["failed_creators"] = failed
        
        if completed + failed == total:
            batch["status"] = "completed"
            batch["end_time"] = datetime.now().isoformat()
        elif completed + failed > 0:
            batch["status"] = "processing"
    
    def _update_overall_progress(self) -> None:
        """Update overall progress counters across all batches"""
        overall = self.progress_data["overall_progress"]
        total_completed = 0
        total_failed = 0
        
        for batch in self.progress_data["batches"].values():
            total_completed += batch.get("completed_creators", 0)
            total_failed += batch.get("failed_creators", 0)
        
        overall["completed_creators"] = total_completed
        overall["failed_creators"] = total_failed
        overall["last_updated"] = datetime.now().isoformat()
    
    def get_pending_creators(self, batch_id: str) -> List[str]:
        """Get the list of creators in pending/failed state for a batch"""
        if batch_id not in self.progress_data["batches"]:
            return []
        
        batch = self.progress_data["batches"][batch_id]
        return [
            c["username"] for c in batch["creators"] 
            if c["status"] in ["pending", "failed"]
        ]
    
    def get_failed_creators(self, batch_id: str) -> List[str]:
        """Get the list of failed creators for a batch"""
        if batch_id not in self.progress_data["batches"]:
            return []
        
        batch = self.progress_data["batches"][batch_id]
        return [
            c["username"] for c in batch["creators"] 
            if c["status"] == "failed"
        ]
    
    def can_resume(self, batch_id: str) -> bool:
        """Check whether a batch can be resumed"""
        if batch_id not in self.progress_data["batches"]:
            return False
        
        batch = self.progress_data["batches"][batch_id]
        return batch["status"] in ["pending", "processing"]
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a compact progress summary for dashboards/logs"""
        return {
            "total_batches": len(self.progress_data["batches"]),
            "overall_progress": self.progress_data["overall_progress"],
            "batch_details": {
                batch_id: {
                    "status": batch["status"],
                    "progress": f"{batch['completed_creators']}/{batch['total_creators']}"
                }
                for batch_id, batch in self.progress_data["batches"].items()
            }
        }
    
    def _auto_save(self) -> None:
        """Auto-save progress if interval elapsed"""
        current_time = time.time()
        if current_time - self.last_save_time >= self.auto_save_interval:
            self.save_progress()
            self.last_save_time = current_time
    
    def save_progress(self) -> None:
        """Persist current progress to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
            
            # Convert datetime objects to strings
            save_data = self._prepare_data_for_save()
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"💾 Progress saved to: {self.progress_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save progress: {e}")
    
    def _prepare_data_for_save(self) -> Dict[str, Any]:
        """Prepare data for persistence, converting datetime to strings"""
        save_data = self.progress_data.copy()
        
        for batch in save_data["batches"].values():
            if batch.get("start_time") and hasattr(batch["start_time"], "isoformat"):
                batch["start_time"] = batch["start_time"].isoformat()
            if batch.get("end_time") and hasattr(batch["end_time"], "isoformat"):
                batch["end_time"] = batch["end_time"].isoformat()
            
            for creator in batch["creators"]:
                if creator.get("start_time") and hasattr(creator["start_time"], "isoformat"):
                    creator["start_time"] = creator["start_time"].isoformat()
                if creator.get("end_time") and hasattr(creator["end_time"], "isoformat"):
                    creator["end_time"] = creator["end_time"].isoformat()
                if creator.get("last_updated") and hasattr(creator["last_updated"], "isoformat"):
                    creator["last_updated"] = creator["last_updated"].isoformat()
        
        if save_data["overall_progress"].get("start_time") and hasattr(save_data["overall_progress"]["start_time"], "isoformat"):
            save_data["overall_progress"]["start_time"] = save_data["overall_progress"]["start_time"].isoformat()
        if save_data["overall_progress"].get("last_updated") and hasattr(save_data["overall_progress"]["last_updated"], "isoformat"):
            save_data["overall_progress"]["last_updated"] = save_data["overall_progress"]["last_updated"].isoformat()
        
        return save_data
    
    def cleanup_completed_batches(self, max_retention_days: int = 7) -> None:
        """Cleanup completed batches older than a retention threshold"""
        current_time = datetime.now()
        batches_to_remove = []
        
        for batch_id, batch in self.progress_data["batches"].items():
            if batch["status"] == "completed":
                if batch.get("end_time"):
                    try:
                        end_time = datetime.fromisoformat(batch["end_time"])
                        days_old = (current_time - end_time).days
                        if days_old > max_retention_days:
                            batches_to_remove.append(batch_id)
                    except:
                        pass
        
        for batch_id in batches_to_remove:
            del self.progress_data["batches"][batch_id]
            logger.info(f"🗑️ Removed expired batch: {batch_id}")
        
        if batches_to_remove:
            self.save_progress()
