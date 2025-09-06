from typing import Literal

from pydantic import BaseModel, Field

import logging
logger = logging.getLogger(__name__)
class PromptModel(BaseModel):
    pass


class ChapterHeading(PromptModel):
    chapter_start_timestamp_seconds: float = Field(
        description="Start time in seconds, this MUST MATCH the timestamp from the video transcript PRECISELY")
    title: str = Field(description="Chapter title")
    description: str = Field(description="Brief description of what happens in this chapter")

    def __str__(self):
        hours = int(self.chapter_start_timestamp_seconds // 3600)
        minutes = int((self.chapter_start_timestamp_seconds % 3600) // 60)
        seconds = int(self.chapter_start_timestamp_seconds % 60)

        if self.description:
            return f"{hours:02}:{minutes:02}:{seconds:02} - {self.title}:\n {self.description}"
        else:
            return f"{hours:02}:{minutes:02}:{seconds:02} - {self.title}"


class PullQuote(PromptModel):
    quality: int = Field(
        description="Quality of this pull quote on a scale from 1 to 1000, where 1000 is the highest quality, evaluated on the basis of interestingness, uniqueness, and potential interest and relevance to the deeper themes and meaning of this video",
        ge=1, le=1000)

    text_content: str = Field(
        description="The text of the pull quote precisely as spoken in the video transcript, copied verbatim with no changes")

    reason_for_selection: str = Field(description="The reason this quote was selected as a pull quote")
    context_around_quote: str = Field(description="Brief context around the quote to explain its significance")


class PullQuoteWithTimestamp(PullQuote):
    timestamp_seconds: float


    def __str__(self):
        output_str = (f"> '{self.text_content}'\n"
                      f"- **Start (w/in full recording):** {float(self.timestamp_seconds):.2f}s\n"
                      f"- **Quality (1-1000):** {self.quality}\n"
                      f"- **Reason for Selection:** {self.reason_for_selection}\n"
                      f"- **Context Around Quote:** {self.context_around_quote}\n")
        return output_str

    @property
    def as_string_youtube_formatted_timestamp(self) -> str:
        hours = int(self.timestamp_seconds // 3600)
        minutes = int((self.timestamp_seconds % 3600) // 60)
        seconds = int(self.timestamp_seconds % 60)

        return f"{hours:02}:{minutes:02}:{seconds:02} - '{self.text_content}'"



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
        subtopics_str = "\n\n".join(str(subtopic) for subtopic in self.subtopics)
        return f"### {self.topic}\n> {self.topic_overview}\n{subtopics_str}\n"


class TranscriptSummaryPromptModel(PromptModel):
    """
    Model for generating Summaries and Outlines

    Considers both topic-based and chronological structures to provide a comprehensive understanding of the video's content.

    Topic based summaries and outlines focus on the main themes and topics discussed, while chronological descriptions and outlines follow the sequence of events as they occurred in the video.

    It is imperative to work as closely as possible to the provided material, using the EXACT and SPECIFIC wording and terminology from the transcript whenever possible.
    """
    transcript_title: str = Field(
        description="The a descriptive title of the transcript, will be used as the H1 header, the filename slug, and the URL slug. "
                    "It should be short (only a few words) and provide a terse preview of the basic content of the full transcript, it should include NO colons")
    transcript_title_slug: str = Field(
        description="The a descriptive title of the transcript, will be used as the H1 header, the filename slug, and the URL slug. "
                    "It should be short (only a few words) and provide a terse preview of the basic content of the full transcript, it should include NO colons")

    one_sentence_summary: str = Field(
        description="A single sentence summary of the content of the transcript. This should be a concise summary that captures the main points and themes of the transcript.")

    executive_summary: str = Field(description="3-5 sentence high-level summary capturing the essence of the video")

    topics_detailed_summary: str = Field(
        description="Detailed summary organized by main topics, covering all key points and topics discussed in the video in detail organized in a logical way, not necessarily in the order they were presented"
    )
    covered_topics_outline: list[TopicOutlineItem] = Field(
        description="Hierarchical outline combining all chunk outlines, organized logically with 5-10 main topics"
    )

    @property
    def formatted_title(self) -> str:
        title_str = (f"# {self.transcript_title} \n\n"
                     f"Title slug: {self.transcript_title_slug}\n\n")
        return title_str

    @property
    def formatted_short_content(self) -> str:
        content_str = (f"## One Sentence Summary\n{self.one_sentence_summary}\n\n"
                       f"## Executive Summary\n{self.executive_summary}\n\n")

        return content_str

    @property
    def formatted_full_content(self) -> str:
        content_str = (f"## One Sentence Summary\n{self.one_sentence_summary}\n\n"
                       f"## Executive Summary\n{self.executive_summary}\n\n"
                       f"## Topics Detailed Summary\n{self.topics_detailed_summary}\n\n"
                       f"## Complete Topic Outline\n" + "\n".join(
            str(item) for item in self.covered_topics_outline) + "\n\n")
        return content_str

    def __str__(self):
        return f"{self.formatted_title}\n{self.formatted_full_content}"


class TopicAreaPromptModel(BaseModel):
    name: str = Field(
        description="The name of this topic area. This should be a single word or hyphenated phrase that describes the topic area. For example, 'machine-learning', 'python', 'oculomotor-control', 'neural-networks', 'computer-vision', etc. Do not include conversational aspects such as 'greetings', 'farewells', 'thanks', etc.")
    category: str = Field(
        description="The general category or field of interest (e.g., 'science', 'technology', 'arts', 'activities', 'health', etc).")
    subject: str = Field(
        description="A more specific subject or area of interest within the category (e.g., 'biology', 'computer science', 'music', 'sports', 'medicine', etc).")
    topic: str = Field(
        description="More specific topic or subfield within the category (e.g., 'neuroscience', 'machine learning',  'classical music', 'basketball', 'cardiology', etc).")
    subtopic: str = Field(
        description="An even more specific subtopic or area of interest within the topic (e.g., 'oculomotor-control', 'neural-networks', 'Baroque music', 'NBA', 'heart surgery', etc).")
    niche: str = Field(
        description="A very specific niche or focus area within the subtopic (e.g., 'gaze-stabilization', 'convolutional-neural-networks', 'Bach', 'NBA playoffs', 'pediatric cardiology', etc).")
    description: str = Field(
        description="A brief description of this interest, including any relevant background information, key concepts, notable figures, recent developments, and related topics. This should be a concise summary that provides context and depth to the interest")

    def __str__(self):
        return f"**#{self.name.replace(' ','-')}**\n \t(#{self.category.replace(' ','-')} | #{self.subject.replace(' ','-')} | #{self.topic.replace(' ','-')} | #{self.subtopic.replace(' ','-')} | #{self.niche.replace(' ','-')}):\n\t\t {self.description}"


class ThemesAndTakeawaysPromptModel(PromptModel):
    main_themes: list[str] = Field(
        description="5-8 primary themes covered across the video transcript, listed in order of priority or emphasis"
    )
    key_takeaways: list[str] = Field(
        description="5-10 main insights and conclusions from the video transcript, summarized as concise bullet points and listed in order of importance"
    )
    topic_areas: list[TopicAreaPromptModel] = Field(
        description="A list of topic areas that describe the content of the transcript. These will be used to categorize the transcript within a larger collection of texts. Ignore conversational aspects (such as 'greetings', 'farewells', 'thanks', etc.).  These should almost always be single word, unless the tag is a multi-word phrase that is commonly used as a single tag, in which case it should be hyphenated. For example, 'machine-learning, python, oculomotor-control,neural-networks, computer-vision', but NEVER things like 'computer-vision-conversation', 'computer-vision-questions', etc.")

    def __str__(self):
        output_str = (f"## Main Themes\n" + "\n".join(f"- {theme}" for theme in self.main_themes) + "\n\n" +
                      f"## Key Takeaways\n" + "\n".join(f"- {takeaway}" for takeaway in self.key_takeaways) + "\n\n" +
                      f"## Topic Areas\n" + "\n".join(str(area) for area in self.topic_areas) + "\n\n"
                      )
        return output_str


class PullQuotesSelectionPromptModel(PromptModel):
    pull_quotes: list[PullQuoteWithTimestamp] = Field(
        description="A list of 5-10 of the most most impactful, interesting, or otherwise notable pull quotes from the video transcript - ORDERED IN TERMS OF QUALITY WITH THE BEST QUOTE FIRST")

    def __str__(self):
        output_str = (f"## Pull Quotes\n" + "\n".join(str(quote) for quote in self.pull_quotes) + "\n\n"
                      )
        return output_str


class ChunkAnalysis(PromptModel):
    summary: TranscriptSummaryPromptModel = Field(description="Comprehensive summary of this chunk of the video")
    main_themes: list[str] = Field(
        description="5-8 primary themes covered across the video transcript, listed in order of priority or emphasis"
    )
    key_takeaways: list[str] = Field(
        description="5-10 main insights and conclusions from the video transcript, summarized as concise bullet points and listed in order of importance"
    )
    topic_areas: list[TopicAreaPromptModel] = Field(
        description="A list of topic areas that describe the content of the transcript. These will be used to categorize the transcript within a larger collection of texts. Ignore conversational aspects (such as 'greetings', 'farewells', 'thanks', etc.).  These should almost always be single word, unless the tag is a multi-word phrase that is commonly used as a single tag, in which case it should be hyphenated. For example, 'machine-learning, python, oculomotor-control,neural-networks, computer-vision', but NEVER things like 'computer-vision-conversation', 'computer-vision-questions', etc.")

    pull_quotes: list[PullQuote] = Field(
        description="Most impactful, interesting, or otherwise notable pull quotes from the video transcript")

    def get_pull_quotes(self, normalize_quality:bool, sort_by: Literal["time","quality"]|None = "quality") -> list[PullQuote]:
        quotes = self.pull_quotes.copy()
        try:
            min_quality = min(quote.quality for quote in quotes)
            max_quality = max(quote.quality for quote in quotes)
            if normalize_quality and max_quality > min_quality:
                for quote in quotes:
                    quote.quality = int(1 + 999 * (quote.quality - min_quality) / (max_quality - min_quality))
            if sort_by:
                if sort_by == "quality":
                    quotes.sort(key=lambda q: q.quality, reverse=True)
                elif sort_by == "time":
                    quotes.sort(key=lambda q: q.timestamp_seconds)
        except Exception as e:
            logger.warning(f"Error normalizing or sorting pull quotes: {e}")
            # If there's an error, just return unsorted/unmodified
            pass
        return quotes


    def __str__(self):
        output_str = (f"## Chunk Summary\n{self.summary}\n\n"
                      f"## Main Themes\n" + "\n".join(f"- {theme}" for theme in self.main_themes) + "\n\n"
                                                                                                    f"## Key Takeaways\n" + "\n".join(
            f"- {takeaway}" for takeaway in self.key_takeaways) + "\n\n"
                                                                  f"## Topic Areas\n" + "\n".join(
            str(area) for area in self.topic_areas) + "\n\n"
                                                      f"## Pull Quotes\n" + "\n".join(
            str(quote) for quote in self.pull_quotes) + "\n\n"
                      )
        return output_str


class ChunkAnalysisWithTimestamp(ChunkAnalysis):
    starting_timestamp_string: str  # e.g. "123.45" for a chunk starting at 123.45 seconds - to match with filename
    pull_quotes: list[PullQuoteWithTimestamp]

    @property
    def as_chapter_heading_with_description(self) -> str:
        return str(ChapterHeading(
            chapter_start_timestamp_seconds=float(self.starting_timestamp_string),
            title=self.summary.transcript_title,
            description=self.summary.one_sentence_summary
        ))

    @property
    def as_chapter_heading_without_description(self) -> str:
        return str(ChapterHeading(
            chapter_start_timestamp_seconds=float(self.starting_timestamp_string),
            title=self.summary.transcript_title,
            description=""
        ))


StartingTimeString = str  # e.g. "123.45" for a chunk starting at 123.45 seconds


class FullVideoAnalysis(BaseModel):
    summary: TranscriptSummaryPromptModel
    chunk_analyses: list[ChunkAnalysisWithTimestamp]
    themes: list[str]
    topics: list[TopicAreaPromptModel]
    takeaways: list[str]
    pull_quotes: list[PullQuoteWithTimestamp]

    def trim(self) -> bool:
        """
        Remove one element based on a general heuristic to reduce size.

        intended to be used iteratively until the model/string output fits within a particular limit

        Checks the length of each NON-chunk-analysis list and removes one element from the longest list.
        """
        # if len(self.most_interesting_clips) > 1: # Always keep at least one clip
        #     print(f"Trimming most_interesting_clips from {len(self.most_interesting_clips)} to {len(self.most_interesting_clips)-1}")
        #     del self.most_interesting_clips[-1]
        #     return True

        if len(self.pull_quotes) > 10:
            print(f"Trimming pull_quotes from {len(self.pull_quotes)} to {len(self.pull_quotes) - 1}")
            del self.pull_quotes[-1]
            return True

        lists = {
            'themes': self.themes,
            'topics': self.topics,
            'takeaways': self.takeaways,
        }
        longest_lists = sorted(lists, key=lambda k: len(lists[k]), reverse=True)
        if longest_lists:
            for longest_list_name in longest_lists:
                if len(lists[longest_list_name]) > 3:
                    print(
                        f"Trimming {longest_list_name} from {len(lists[longest_list_name])} to {len(lists[longest_list_name]) - 1}")
                    del lists[longest_list_name][-1]
                    return True
        return False

    def to_markdown(self, disclaimer_text: str | None = None, header:str=None) -> str:
        out_string = ""
        out_string += self.summary.formatted_title.replace("# ","# [FULL] ")
        out_string += self.summary.formatted_full_content
        if disclaimer_text:
            out_string += f"\n\n> **Disclaimer:** {disclaimer_text}\n\n"
        if header:
            out_string += f"\n\n{header}\n\n"
        out_string += "## Overall Main Themes\n" + "\n".join(f"- {theme}" for theme in self.themes) + "\n\n"
        out_string += "## Overall Key Takeaways\n" + "\n".join(f"- {takeaway}" for takeaway in self.takeaways) + "\n\n"
        out_string += "## Overall Topic Areas\n" + "\n".join(str(area) for area in self.topics) + "\n\n"
        out_string += "## Overall Pull Quotes\n" + "\n".join(str(quote) for quote in self.pull_quotes) + "\n\n"


        out_string += "\n".join(
            f"\\n\\n------------------------TRANSCRIPT CHUNK ANALYSES------------------------n\n\n### Analysis for Chunk Starting at {chunk.starting_timestamp_string} seconds\n\n{str(chunk)}"
            for chunk in self.chunk_analyses
        )

        return out_string

    def get_pull_quotes(self, sort_by: Literal["time","quality"]|None = "quality") -> list[PullQuoteWithTimestamp]:
        quotes = self.pull_quotes.copy()
        if sort_by:
            if sort_by == "quality":
                quotes.sort(key=lambda q: q.quality, reverse=True)
            elif sort_by == "time":
                quotes.sort(key=lambda q: q.timestamp_seconds)
        return quotes