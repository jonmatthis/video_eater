from pydantic import BaseModel
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
    full_transcript_raw: str
    full_transcript_timestamps_srt: str
    chunk_offset_seconds: float = 0.0

    @property
    def start_time(self) -> float:
        if self.transcript_segments:
            return self.transcript_segments[0].start + self.chunk_offset_seconds
        return self.chunk_offset_seconds

    @property
    def end_time(self) -> float:
        if self.transcript_segments:
            return self.transcript_segments[-1].end + self.chunk_offset_seconds
        return self.chunk_offset_seconds

    @property
    def absolute_start_time(self) -> float:
        """Raw first segment start time (chunk-relative, no offset)."""
        if self.transcript_segments:
            return self.transcript_segments[0].start
        return 0.0
    @classmethod
    def from_whisper_response(cls, transcript_data) -> 'VideoTranscript':
        """Convert a Groq/OpenAI whisper verbose_json response to VideoTranscript."""
        full_transcript_raw = ""
        full_transcript_timestamps_srt = ""
        transcript_segments = []

        for seg_number, segment in enumerate(transcript_data.segments):
            segment_start = segment["start"] if isinstance(segment, dict) else segment.start
            segment_end = segment["end"] if isinstance(segment, dict) else segment.end
            segment_text = segment["text"] if isinstance(segment, dict) else segment.text

            from video_eater.core.output_templates import format_timestamp_srt
            srt_formatted_timestamp = f"{format_timestamp_srt(segment_start)} --> {format_timestamp_srt(segment_end)}"
            transcript_segments.append(
                TranscriptSegment(
                    text=segment_text.strip(),
                    start=segment_start,
                    end=segment_end
                )
            )
            full_transcript_raw += f"{segment_text.strip()}\n"
            full_transcript_timestamps_srt += f"{seg_number}\n{srt_formatted_timestamp}\n{segment_text.strip()}\n\n"

        return cls(
            full_transcript_raw=full_transcript_raw,
            full_transcript_timestamps_srt=full_transcript_timestamps_srt,
            transcript_segments=transcript_segments,
        )

