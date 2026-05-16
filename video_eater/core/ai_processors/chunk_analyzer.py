"""Single-chunk AI inference — pure logic, no orchestration."""

import logging

from video_eater.core.ai_processors.ai_prompt_models import ChunkAnalysis, ChunkAnalysisWithTimestamp, StartingTimeString
from video_eater.core.ai_processors.base_processor import BaseAIProcessor
from video_eater.core.ai_processors.fuzz_matching_helpers import match_quotes_to_transcript_srt
from video_eater.core.transcribe_audio.transcript_models import VideoTranscript
from video_eater.core.ai_processors.transcript_utils import format_timestamp

logger = logging.getLogger(__name__)


class ChunkAnalyzer:
    """Analyzes a single transcript chunk via LLM and enriches with timestamps."""

    def __init__(self, processor: BaseAIProcessor, chunk_length_seconds: int):
        self._processor = processor
        self._chunk_length_seconds = chunk_length_seconds

    async def analyze_chunk(
        self,
        transcript_data: VideoTranscript,
        chunk_start_time_string: StartingTimeString,
        chunk_index: int = 0,
    ) -> ChunkAnalysisWithTimestamp:
        """Analyze a single transcript chunk.

        1. Constructs a system prompt with the SRT transcript text.
        2. Calls the LLM for structured output (ChunkAnalysis).
        3. Runs fuzzy timestamp matching on the generated pull quotes.
        4. Returns a ChunkAnalysisWithTimestamp.
        """

        system_prompt = f"""You are analyzing a transcript chunk from a longer video.
        This is chunk #{chunk_index} starting at {format_timestamp(transcript_data.start_time)} in the video.

       <<<Transcript-START>>>

        {transcript_data.full_transcript_timestamps_srt}

        <<<Transcript-END>>>

        Use this information and provie your answer in JSON format according to the provided schema.

        You must use the information from the transcript to fill in the fields as accurately as possible, in effect to best
        summarize and outline the content of this chunk of the video. Ensure precise and careful copying of the direct quotes.
        """

        try:
            response = await self._processor.async_make_openai_json_mode_ai_request(
                system_prompt=system_prompt,
                input_data={},
                output_model=ChunkAnalysis
            )
            chunk_dict = response.model_dump()

            chunk_start_time = float(chunk_start_time_string)
            chunk_end_time = chunk_start_time + float(self._chunk_length_seconds)

            pull_quotes_with_timestamps = await match_quotes_to_transcript_srt(
                quotes=response.pull_quotes,
                transcript_srt=transcript_data.full_transcript_timestamps_srt,
                chunk_start_time=chunk_start_time,
                chunk_end_time=chunk_end_time,
            )

            chunk_dict['pull_quotes'] = pull_quotes_with_timestamps
            response_with_timestamps = ChunkAnalysisWithTimestamp(**chunk_dict,
                                                                  starting_timestamp_string=chunk_start_time_string)
            return response_with_timestamps

        except Exception as e:
            logger.error(f"Error analyzing chunk {chunk_index}: {e}")
            raise
