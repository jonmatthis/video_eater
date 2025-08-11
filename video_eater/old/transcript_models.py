import re
from pydantic import BaseModel, Field


class TranscriptEntry(BaseModel):
    text: str = Field(default='',
        description="Text of the transcript entry, which has been altered to make it more grammatically correct and readable. "
                    "Segments may have been combined or split as necessary to improve readability.")
    start: float = Field(default=0.0,
        description="Start time of the transcript entry in seconds, relative to the beginning of the video. ")
    dur: float = Field(default=0.0,
        description="Duration of the transcript entry in seconds, indicating how long the text is spoken in the video. "
                    "This may be different from the original duration if segments were combined or split.")
    end: float = Field(default=0.0,
        description="End time of the transcript entry in seconds, calculated as start + dur. If not provided, it will be "
                    "calculated automatically based on the start and duration.")



class ProcessedTranscript(BaseModel):
    title: str = Field(default='',description="Title of the transcribed video, based on its content")
    short_description: str = Field(default='',
        description="Short description of the transcribed video, summarizing its content",
    )
    youtube_formatted_chapters: str = Field(default='',
        description="Formatted chapters for YouTube, with each chapter on a new line in the format 'HH:MM:SS - Chapter Title'.",
    )
    transcript_chunks: list[TranscriptEntry] = Field(default_factory=list,
        description="List of transcript entries, each containing text, start time, duration, and end time. The text has "
                    "been altered to make it more gramatically correct and readable, with segments combined or split as necessary.")

    @property
    def full_text(self) -> str:
        """Concatenate all transcript texts into a single string."""
        return ' '.join(entry.text for entry in self.transcript_chunks)


