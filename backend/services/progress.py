"""
Progress Tracking for Manga Generation
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class GenerationProgress:
    """Current generation progress"""
    stage: str = "idle"  # idle, storyboard, generating, completed, error
    current_panel: int = 0
    total_panels: int = 0
    message: str = ""
    started_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "current_panel": self.current_panel,
            "total_panels": self.total_panels,
            "message": self.message,
            "progress_percent": round(self.current_panel / self.total_panels * 100) if self.total_panels > 0 else 0
        }


# Global progress instance
_progress = GenerationProgress()


def get_progress() -> GenerationProgress:
    """Get current progress"""
    return _progress


def set_stage(stage: str, message: str = ""):
    """Set current stage"""
    global _progress
    _progress.stage = stage
    _progress.message = message
    if stage == "generating":
        _progress.started_at = datetime.now()


def set_panel_progress(current: int, total: int):
    """Update panel progress"""
    global _progress
    _progress.current_panel = current
    _progress.total_panels = total


def reset_progress():
    """Reset progress to idle"""
    global _progress
    _progress = GenerationProgress()
