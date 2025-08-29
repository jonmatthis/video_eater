from pydantic import BaseModel, Field


class ChapterHeading(BaseModel):
    timestamp_seconds: float = Field(description="Start time in seconds")
    title: str = Field(description="Chapter title")
    description: str = Field(description="Brief description of what happens in this chapter")


class PromptModel(BaseModel):
    pass


class PullQuote(PromptModel):
    timestamp_seconds: float = Field(description="Start time in seconds when the quote was spoken")
    text_content: str = Field(
        description="The text of the pull quote, as a list of strings (one per line if multiline)")

    reason_for_selection: str = Field(description="The reason this quote was selected as a pull quote")
    context_around_quote: str = Field(description="Brief context around the quote to explain its significance")


class ParticularlyInteresting60SecondClip(PromptModel):
    start_timestamp_seconds: float = Field(description="Start time in seconds of the 60-second clip")
    end_timestamp_seconds: float = Field(description="End time in seconds of the 60-second clip")
    text_content: str = Field(description="Full text content of the 60-second clip")
    reason_for_selection: str = Field(description="The reason this clip was selected as particularly interesting")
    context_around_clip: str = Field("Brief context around the clip to explain its significance")


class SubTopicOutlineItem(PromptModel):
    subtopic: str = Field(description="Subtopic under a main topic")
    details: list[str] = Field(description="List of details or points under this subtopic")


class TopicOutlineItem(PromptModel):
    topic: str = Field(description="Main topic")
    topic_overview: str = Field(description="Brief overview of the main topic")
    subtopics: list[SubTopicOutlineItem] = Field(description="List of subtopics under this main topic")


class ChunkAnalysis(PromptModel):
    summary: str = Field(description="Comprehensive summary of the chunk content")
    key_topics: list[str] = Field(description="list of main topics discussed")
    chunk_outline: list[TopicOutlineItem] = Field(description="Hierarchical outline of topics and subtopics")
    chapters: list[ChapterHeading] = Field(description="Timestamped chapter headings")
    pull_quotes: list[PullQuote] = Field(
        description="Most impactful, interesting, or otherwise notable pull quotes from the video")


class FullVideoAnalysis(PromptModel):
    executive_summary: str = Field(description="High-level summary of the entire video")
    detailed_summary: str = Field(description="Comprehensive summary with key points")
    main_themes: list[str] = Field(description="Primary themes covered across the video")
    complete_outline: list[TopicOutlineItem] = Field(description="Hierarchical outline of topics and subtopics")
    chapters: list[ChapterHeading] = Field(description="Full video chapter list with adjusted timestamps")
    key_takeaways: list[str] = Field(description="Main insights and conclusions")
    pull_quotes: list[PullQuote] = Field(
        description="Most impactful, interesting, or otherwise notable pull quotes from the video")
    particularly_interesting_60_second_clips: list[ParticularlyInteresting60SecondClip] = Field(
        description="List of 60-second clips that are particularly interesting or noteworthy, which be extract into standalong short video clips for tiktok, youtube shorts, instagram reels, etc")
