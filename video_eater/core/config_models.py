# config_models.py
from pydantic import BaseModel, Field, validator
from pathlib import Path
from typing import Optional, Literal
from enum import Enum


class TranscriptionProvider(str, Enum):
    OPENAI = "openai"
    ASSEMBLY_AI = "assembly_ai"
    LOCAL_WHISPER = "local_whisper"


class ProcessingConfig(BaseModel):
    """Central configuration for the video processing pipeline."""

    # Audio chunking
    chunk_length_seconds: int = Field(600, ge=60, le=3600)
    chunk_overlap_seconds: int = Field(15, ge=0, le=60)

    # AI models
    analysis_model: str = Field("gpt-4.1", description="Model for analysis")
    transcription_provider: TranscriptionProvider = TranscriptionProvider.OPENAI
    whisper_model: Optional[str] = "large"

    # Concurrency
    max_concurrent_chunks: int = Field(50, ge=1, le=100)
    batch_size: int = Field(10, ge=1, le=50)

    # Force flags
    force_chunk_audio: bool = False
    force_transcribe: bool = False
    force_analyze: bool = False

    @validator('chunk_overlap_seconds')
    def validate_overlap(cls, v, values):
        if 'chunk_length_seconds' in values:
            if v >= values['chunk_length_seconds']:
                raise ValueError("Overlap must be less than chunk length")
        return v


class ProcessingStats(BaseModel):
    """Track processing statistics."""
    audio_chunks_created: int = 0
    audio_chunks_cached: int = 0
    transcripts_created: int = 0
    transcripts_cached: int = 0
    analyses_created: int = 0
    analyses_cached: int = 0
    total_duration_seconds: float = 0
    errors: list[str] = Field(default_factory=list)

    def add_error(self, error: str):
        self.errors.append(error)

    @property
    def cache_hit_rate(self) -> float:
        total = (self.audio_chunks_created + self.audio_chunks_cached +
                 self.transcripts_created + self.transcripts_cached +
                 self.analyses_created + self.analyses_cached)
        if total == 0:
            return 0
        cached = self.audio_chunks_cached + self.transcripts_cached + self.analyses_cached
        return cached / total


class VideoProject(BaseModel):
    """Represents a video processing project."""
    video_path: Path
    title: str = ""

    # Computed paths
    @property
    def audio_chunks_folder(self) -> Path:
        return self.video_path.parent / "audio_chunks"

    @property
    def transcript_chunks_folder(self) -> Path:
        return self.video_path.parent / "transcript_chunks"

    @property
    def analysis_folder(self) -> Path:
        return self.video_path.parent / "analysis_chunks"

    @property
    def output_folder(self) -> Path:
        return self.video_path.parent / "outputs"

    class Config:
        arbitrary_types_allowed = True