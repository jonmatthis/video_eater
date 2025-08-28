import re
from pydantic import BaseModel
from openai.types.audio import TranscriptionVerbose

class TranscriptSegment(BaseModel):
    text: str
    start: float
    dur: float = None
    end: float = None

    def model_post_init(self, __context) -> None:
        # Calculate end time if not provided
        if self.end is None and self.dur is None:
            raise ValueError("Must specify either `end` or `dur`")
        if self.end is None:
            self.end = self.start + self.dur
        if self.dur is None:
            self.dur = self.end - self.start



class VideoTranscript(BaseModel):
    video_id: str = ""
    transcript_segments: list[TranscriptSegment]
    full_transcript: str = ""



    @classmethod
    def from_openai_transcript(cls, transcript_data: TranscriptionVerbose) -> 'VideoTranscript':
        return cls(
            full_transcript=transcript_data.text,
            transcript_segments=[
                TranscriptSegment(text=segment.text, start=segment.start, end=segment.end)
                for segment in transcript_data.segments
            ],
        )
class ProcessedTranscript(BaseModel):
    video_id: str
    title: str
    transcript_chunks: list[TranscriptSegment]
    full_transcript: str = ""

    @property
    def key_name(self) -> str:
        return f"{self.title}_{self.video_id}"