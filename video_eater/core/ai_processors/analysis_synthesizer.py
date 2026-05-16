"""Cross-chunk synthesis: combines N chunk analyses into 1 FullVideoAnalysis."""

import logging
from copy import deepcopy

from video_eater.core.ai_processors.ai_prompt_models import (
    FullVideoAnalysis, ChunkAnalysisWithTimestamp,
    TranscriptSummaryPromptModel, ThemesAndTakeawaysPromptModel,
)
from video_eater.core.ai_processors.base_processor import BaseAIProcessor

logger = logging.getLogger(__name__)


class AnalysisSynthesizer:
    """Aggregates chunk-level analyses and runs LLM calls to produce a full video analysis."""

    def __init__(self, processor: BaseAIProcessor):
        self._processor = processor

    async def combine_analyses(
        self,
        chunk_analyses: list[ChunkAnalysisWithTimestamp],
    ) -> FullVideoAnalysis:
        """Combine individual chunk analyses into a complete video analysis.

        Stages:
        1. Aggregate summaries, themes, takeaways, topics, and pull quotes from all chunks.
        2. LLM call: produce executive summary + detailed summary + topic outline.
        3. LLM call: produce main themes + key takeaways + topic areas.
        4. Rank pull quotes by quality score.
        5. Construct and return FullVideoAnalysis.
        """

        # Stage 1: Aggregate all data from chunks
        all_summaries = []
        all_themes = []
        all_takeaways = []
        all_topics = []
        all_pull_quotes = []

        for chunk in chunk_analyses:
            chunk_copy = deepcopy(chunk)
            all_summaries.append(chunk_copy.summary)
            all_themes.extend(chunk_copy.main_themes)
            all_takeaways.extend(chunk_copy.key_takeaways)
            all_topics.extend(chunk_copy.topic_areas)
            all_pull_quotes.extend(chunk_copy.get_pull_quotes(normalize_quality=True, sort_by="quality"))

        # Stage 2: Generate executive and detailed summaries
        chunk_summary_string = ""
        for i, summary in enumerate(all_summaries):
            chunk_summary_string += f"<<<START Chunk {i + 1}>>>>\n\n{summary}  \n\n<<<END Chunk {i + 1}>>>>\n\n----------------\n\n"

        summary_prompt = f"""Based on these chunk summaries and analysis from a video transcrpt, create:

        Chunk Summaries/Analysis:
        [[[[[START CHUNK SUMMARIES and ANALYSIS]]]]]
        {chunk_summary_string}
        [[[[[END CHUNK SUMMARIES and ANALYSIS]]]]]

        Using the above chunk summaries, provide a an OVERALL summary and analysis of the entire video in accordance to  JSON format schema provided.
        """

        summary_response = await self._processor.async_make_openai_json_mode_ai_request(
            system_prompt=summary_prompt,
            input_data={},
            output_model=TranscriptSummaryPromptModel
        )
        logger.debug(f"Summary Response:\n{summary_response}")

        # Stage 3: Extract main themes and key takeaways
        all_topic_areas_prompt_string = "- " + "\n\n- ".join([str(t) for t in all_topics])
        all_themes_prompt_string = "- " + "\n\n- ".join(all_themes)
        all_takeaways_prompt_string = "- " + "\n\n- ".join(all_takeaways)
        themes_prompt = f"""You will be given a summary and analysis of an extended video transcript along with topics and themes that were extracted from chunks of the video.

        Using this information, identify the MAIN THEMES,  and KEY TAKEAWAYS from the entire video.

        <<<<Full Video Summary and Analysis>>>>
        {summary_response}
        <<<<End Full Video Summary and Analysis>>>>
        ----------------------------------------------------------------------------------------------------------------------------------------
        ----------------------------------------------------------------------------------------------------------------------------------------
        <<<<All Topics, Themes, and TakeAways from Chunks>>>>
        <<<< All Topics>>>>
        {all_topic_areas_prompt_string}
        <<<< End All Topics>>>>
        <<<< All Themes>>>>
        {all_themes_prompt_string}
        <<<< End All Themes>>>>
        <<<< All TakeAways>>>>
        {all_takeaways_prompt_string}
        <<<< End All TakeAways>>>>
        <<<< End All Topics, Themes, and Take Aways from Chunks>>>>
        ----------------------------------------------------------------------------------------------------------------------------------------
        ----------------------------------------------------------------------------------------------------------------------------------------

        Based on the above, provide the TOPIC AREAS, MAIN THEMES, and KEY TAKEAWAYS from the entire video in accordance to  JSON format schema provided.
        """

        themes_response = await self._processor.async_make_openai_json_mode_ai_request(
            system_prompt=themes_prompt,
            input_data={},
            output_model=ThemesAndTakeawaysPromptModel
        )
        logger.debug(f"Themes Response:\n{themes_response}")

        # Stage 4: Rank pull quotes by quality
        top_pull_quotes = sorted(deepcopy(all_pull_quotes), key=lambda x: x.quality, reverse=True)
        top_pull_quotes = top_pull_quotes[:20] if len(top_pull_quotes) > 10 else top_pull_quotes
        logger.debug(f"Pull Quotes:\n" + "\n".join(str(q) for q in top_pull_quotes))

        return FullVideoAnalysis(
            summary=summary_response,
            chunk_analyses=chunk_analyses,
            themes=themes_response.main_themes,
            topics=themes_response.topic_areas,
            takeaways=themes_response.key_takeaways,
            pull_quotes=top_pull_quotes,
        )
