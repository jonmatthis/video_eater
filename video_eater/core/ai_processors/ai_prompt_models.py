from pydantic import BaseModel, Field


class PromptModel(BaseModel):
    pass

class ChapterHeading(PromptModel):
    chapter_start_timestamp_seconds: float = Field(description="Start time in seconds, this MUST MATCH the timestamp from the video transcript PRECISELY")
    title: str = Field(description="Chapter title")
    description: str = Field(description="Brief description of what happens in this chapter")



class PullQuote(PromptModel):
    text_content: str = Field(
        description="The text of the pull quote, as a list of strings (one per line if multiline)")

    reason_for_selection: str = Field(description="The reason this quote was selected as a pull quote")
    context_around_quote: str = Field(description="Brief context around the quote to explain its significance")


class MostInterestingShortSection(PromptModel):
    """
    Model representing a particularly interesting or noteworthy 60-ish-second clip from the video, e.g.
    the part of this section that would be the best candidate for a standalone short video clip for tiktok, youtube shorts, instagram reels, etc.
    """
    text_content: str = Field(description="The text content from the clip, copied verbatim with no changes precisely as spoken in the video")
    reason_for_selection: str = Field(description="The reason this clip was selected as particularly interesting")
    context_around_clip: str = Field("Brief context around the clip to explain its significance")


class SubTopicOutlineItem(PromptModel):
    subtopic: str = Field(description="Subtopic under a main topic")
    details: list[str] = Field(description="List of details or points under this subtopic")

    def __str__(self):
        details_str = "\n".join(f"  - {detail}" for detail in self.details)
        return f"- {self.subtopic}\n{details_str}"


class TopicOutlineItem(PromptModel):
    topic: str = Field(description="Main topic")
    topic_overview: str = Field(description="Brief overview of the main topic")
    subtopics: list[SubTopicOutlineItem] = Field(description="List of subtopics under this main topic")

    def __str__(self):
        subtopics_str = "\n".join(str(subtopic) for subtopic in self.subtopics)
        return f"### {self.topic}\n> {self.topic_overview}\n{subtopics_str}"



class SummaryGeneration(PromptModel):
    """
    Model for generating Summaries and Outlines

    Considers both topic-based and chronological structures to provide a comprehensive understanding of the video's content.

    Topic based summaries and outlines focus on the main themes and topics discussed, while chronological descriptions and outlines follow the sequence of events as they occurred in the video.

    It is imperative to work as closely as possible to the provided material, using the EXACT and SPECIFIC wording and terminology from the transcript whenever possible.
    """

    executive_summary: str = Field(description="3-5 sentence high-level summary capturing the essence of the video")

    topics_detailed_summary: str = Field(
        description="Detailed summary organized by main topics, covering all key points and topics discussed in the video in detail organized in a logical way, not necessarily in the order they were presented"
    )
    complete_topic_outline: list[TopicOutlineItem] = Field(
        description="Hierarchical outline combining all chunk outlines, organized logically with 5-10 main topics"
    )
    chronological_description: str = Field(
        description="Detailed chronological description of the video, covering all the key moments and topics in the video described with great specificity and detail in the order they were presented")

    complete_chronological_outline: str = Field(
        description="Comprehensive outline combining all chunk outlines in chronological order, covering all key points as they were presented in the video"
    )

    main_themes: list[str] = Field(
        description="5-8 primary themes covered across the video, listed in order of priority or emphasis"
    )
    key_takeaways: list[str] = Field(
        description="5-10 main insights and conclusions from the video, summarized as concise bullet points and listed in order of importance"
    )

    def __str__(self):
        topics_outline_str = "\n".join(str(item) for item in self.complete_topic_outline)
        return (f"## Executive Summary\n{self.executive_summary}\n\n"
                f"## Topic Summaries\n{self.topics_detailed_summary}\n\n"
                f"## Chronological Description\n{self.chronological_description}\n\n"
                f"## Complete Topic Outline\n{topics_outline_str}\n\n"
                f"## Complete Chronological Outline\n{self.complete_chronological_outline}\n\n"
                f"## Main Themes\n" + "\n".join(f"- {theme}" for theme in self.main_themes) + "\n\n"
                f"## Key Takeaways\n" + "\n".join(f"- {takeaway}" for takeaway in self.key_takeaways)
               )

class ChunkAnalysis(PromptModel):
    summary: SummaryGeneration
    pull_quotes: list[PullQuote] = Field(
        description="Most impactful, interesting, or otherwise notable pull quotes from the video")
    most_interesting_short_section: MostInterestingShortSection = Field(
        description="Particularly interesting or noteworthy section of this chunk, e.g. the part of this section that would be the best candidate for a standalone short video clip for tiktok, youtube shorts, instagram reels, etc."
    )
    def __str__(self):
        return (f"## Summary\n{self.summary}\n\n"
                f"## Pull Quotes\n" + "\n".join(f"- \"{quote.text_content}\" (Reason: {quote.reason_for_selection})" for quote in self.pull_quotes) + "\n\n"
                f"## Most Interesting Short Section\n- \"{self.most_interesting_short_section.text_content}\" (Reason: {self.most_interesting_short_section.reason_for_selection})"
               )

class ClipSelection(PromptModel):
    """Model for selecting top 60-second clips"""
    top_clips: list[MostInterestingShortSection] = Field(
        description="The most interesting or noteworthy short clips from the video"
    )

class FullVideoAnalysis(PromptModel):
    summary: SummaryGeneration = Field(description="Comprehensive summary of the entire video")
    chunk_analyses: list[ChunkAnalysis] = Field(description="List of analyses for each chunk of the video")
    pull_quotes: list[PullQuote] = Field(
        description="Most impactful, interesting, or otherwise notable pull quotes from the video")
    most_interesting_clips: list[MostInterestingShortSection] = Field(
        description="Particularly interesting or noteworthy short clips from the video")
