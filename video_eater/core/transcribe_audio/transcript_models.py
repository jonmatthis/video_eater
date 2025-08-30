import re
from pydantic import BaseModel
from openai.types.audio import TranscriptionVerbose
from assemblyai.transcriber import Transcript
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
    full_transcript: str
    @property
    def start_time(self) -> float:
        if self.transcript_segments:
            return self.transcript_segments[0].start
        return 0.0
    @property
    def end_time(self) -> float:
        if self.transcript_segments:
            return self.transcript_segments[-1].end
        return 0.0
    @classmethod
    def from_assembly_ai_output(cls, start_time:float, transcript:Transcript):
        # Extract full transcript text
        full_transcript = ""

        # Create transcript segments from either words or segments/utterances
        pararaphs = transcript.get_paragraphs()
        transcript_segments = []

        # If we have segments/utterances, use those (they have start/end times)
        if pararaphs:
            for segment in pararaphs:
                transcript_segments.append(
                    TranscriptSegment(
                        text=segment.text,
                        start=(segment.start / 1000.0)+start_time,  # Convert ms to seconds
                        end=(segment.end / 1000.0)+start_time     # Convert ms to seconds
                    )
                )
                full_transcript += f"{segment.text}\n"

        return cls(
            full_transcript=full_transcript,
            transcript_segments=transcript_segments,
        )

    @classmethod
    def from_openai_transcript(cls, start_time: float, transcript_data: TranscriptionVerbose) -> 'VideoTranscript':
        return cls(
            full_transcript=transcript_data.text,
            transcript_segments=[
                TranscriptSegment(
                    text=segment.text,
                    start=segment.start + start_time,  # Add chunk start time
                    end=segment.end + start_time  # Add chunk start time
                )
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
