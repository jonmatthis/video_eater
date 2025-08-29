import re
from pydantic import BaseModel
from openai.types.audio import TranscriptionVerbose

class TranscriptSegment(BaseModel):
    text: str
    start: float  # seconds
    dur: float = None  # seconds
    end: float = None  # seconds

    def model_post_init(self, __context) -> None:
        # Calculate end time if not provided
        if self.end is None and self.dur is None:
            raise ValueError("Must specify either `end` or `dur`")
        if self.end is None:
            self.end = self.start + self.dur
        if self.dur is None:
            self.dur = self.end - self.start


class VideoTranscript(BaseModel):
    transcript_segments: list[TranscriptSegment]
    full_transcript: str = ""

    @classmethod
    def from_assembly_ai_output(cls, aai_result):
        # Extract full transcript text
        full_transcript = ""

        # Create transcript segments from either words or segments/utterances
        transcript_segments = []

        # If we have segments/utterances, use those (they have start/end times)
        if aai_result.get("paragraphs"):
            for segment in aai_result["paragraphs"]:
                transcript_segments.append(
                    TranscriptSegment(
                        text=segment.text,
                        start=segment.start / 1000.0,  # Convert ms to seconds
                        end=segment.end / 1000.0      # Convert ms to seconds
                    )
                )
                full_transcript += f"{segment.text}\n\n"

        return cls(
            full_transcript=full_transcript,
            transcript_segments=transcript_segments
        )

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
