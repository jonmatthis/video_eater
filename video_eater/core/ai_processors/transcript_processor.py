import asyncio
import json
import logging
import re
import time
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple

import yaml

from video_eater.core.ai_processors.ai_prompt_models import ChunkAnalysis, FullVideoAnalysis, \
    TranscriptSummaryPromptModel, ThemesAndTakeawaysPromptModel, PullQuotesSelectionPromptModel, \
    MostInterestingShortSectionSelectionPromptModel, StartingTimeString, ChunkAnalysisWithTimestamp, \
    PullQuoteWithTimestamp
from video_eater.core.ai_processors.base_processor import BaseAIProcessor
from video_eater.core.output_templates import YouTubeDescriptionFormatter, JsonFormatter, MarkdownReportFormatter, \
    SimpleTextFormatter
from video_eater.core.transcribe_audio.transcript_models import VideoTranscript

logger = logging.getLogger(__name__)


async def _return_cached(idx: int, analysis: ChunkAnalysisWithTimestamp) -> Tuple[int, ChunkAnalysisWithTimestamp]:
    """Helper to return cached analysis in async context."""
    return idx, analysis


class TranscriptProcessor:
    """Process transcribed chunks to generate summaries, outlines, and chapters."""

    def __init__(self,
                 model: str,
                 use_async: bool = True,
                 max_concurrent_chunks: int = 50,
                 batch_size: int = 10,
                 chunk_length_seconds: int = 600,  # Store chunk length
                 chunk_overlap_seconds: int = 15):  # Store overlap
        self.processor = BaseAIProcessor(model=model, use_async=use_async)
        self.model = model
        self.max_concurrent_chunks = max_concurrent_chunks
        self.batch_size = batch_size
        self.chunk_length_seconds = chunk_length_seconds
        self.chunk_overlap_seconds = chunk_overlap_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent_chunks)
        self.processing_stats = {
            'chunks_processed': 0,
            'chunks_cached': 0,
            'processing_time': 0,
            'errors': []
        }

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to YouTube-style timestamp (HH:MM:SS)."""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def parse_chunk_filename(filename: str) -> Tuple[int, float, StartingTimeString]:
        """Extract chunk index and start time from chunk filename.
        Returns: (chunk_index, start_time_seconds)
        """
        # Match the actual format: *_chunk_XXX__YYY.Ysec* where XXX is the index and YYY.Y is the start time
        pattern = r"chunk_(\d{3})__(\d+(?:\.\d+)?)sec"
        match = re.search(pattern, filename)
        if match:
            chunk_index = int(match.group(1))
            start_time_string = match.group(2)
            start_time_float = float(start_time_string)
            return chunk_index, start_time_float, start_time_string

        raise ValueError(f"Filename does not match expected pattern: {filename}")

    async def analyze_chunk_with_semaphore(self,
                                           transcript_data: VideoTranscript,
                                           chunk_start_time_string: StartingTimeString,
                                           chunk_index: int,
                                           chunk_name: str) -> Tuple[int, ChunkAnalysisWithTimestamp]:
        """Analyze a single chunk with rate limiting via semaphore."""
        async with self._semaphore:
            try:
                start_time = time.time()
                print(
                    f"  📝 Processing chunk {chunk_index + 1}: {chunk_name} (starts at {self.format_timestamp(transcript_data.start_time)})")

                analysis = await self.analyze_chunk(
                    transcript_data=transcript_data,
                    chunk_start_time_string=chunk_start_time_string,
                    chunk_index=chunk_index
                )

                elapsed = time.time() - start_time
                print(f"  ✅ Completed chunk {chunk_index + 1} in {elapsed:.1f}s")

                return chunk_index, analysis

            except Exception as e:
                error_msg = f"Error analyzing chunk {chunk_index}: {str(e)}"
                print(f"  ❌ Failed chunk {chunk_index + 1}: {e}")
                self.processing_stats['errors'].append(error_msg)
                raise

    async def analyze_chunk(self,
                            transcript_data: VideoTranscript,
                            chunk_start_time_string: StartingTimeString,
                            chunk_index: int = 0) -> ChunkAnalysisWithTimestamp:
        """Analyze a single transcript chunk."""

        system_prompt = f"""You are analyzing a transcript chunk from a longer video. 
        This is chunk #{chunk_index} starting at {self.format_timestamp(transcript_data.start_time)} in the video.

       <<<Transcript-START>>>
        
        {transcript_data.full_transcript_timestamps_srt}
        
        <<<Transcript-END>>>
        
        Use this information and provie your answer in JSON format according to the provided schema. 
        
        You must use the information from the transcript to fill in the fields as accurately as possible, in effect to best
        summarize and outline the content of this chunk of the video. Ensure precise and careful copying of the direct quotes.
        """

        try:
            response = await self.processor.async_make_openai_json_mode_ai_request(
                system_prompt=system_prompt,
                input_data={},
                output_model=ChunkAnalysis
            )
            chunk_dict = response.model_dump()
            pull_quotes_with_timestamps = []
            for quote in chunk_dict['pull_quotes']:
                pull_quotes_with_timestamps.append(
                    PullQuoteWithTimestamp(**quote, starting_timestamp_string=chunk_start_time_string)
                )
            chunk_dict['pull_quotes'] = pull_quotes_with_timestamps
            response_with_timestamps = ChunkAnalysisWithTimestamp(**chunk_dict,
                                                                  starting_timestamp_string=chunk_start_time_string)
            return response_with_timestamps

        except Exception as e:
            print(f"Error analyzing chunk {chunk_index}: {e}")
            raise

    async def process_transcript_batch(self,
                                       batch: list[tuple[int, StartingTimeString, Path]],
                                       output_folder: Path) -> List[ChunkAnalysisWithTimestamp]:
        """Process a batch of transcript files in parallel."""
        tasks = []

        for batch_idx, (chunk_idx, chunk_start_time_string, transcript_file) in enumerate(batch):
            # Check if analysis already exists
            analysis_file = output_folder / f"{transcript_file.stem}.analysis.yaml"

            if analysis_file.exists():
                print(f"  📂 Using cached analysis for chunk {chunk_idx + 1}")
                self.processing_stats['chunks_cached'] += 1
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    chunk_analysis = ChunkAnalysisWithTimestamp(**yaml.safe_load(f))
                    tasks.append(asyncio.create_task(
                        _return_cached(chunk_idx, chunk_analysis)
                    ))
            else:
                # Load transcript
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript_data = VideoTranscript(**json.load(f))

                # Create analysis task
                task = asyncio.create_task(
                    self.analyze_chunk_with_semaphore(
                        transcript_data=transcript_data,
                        chunk_start_time_string=chunk_start_time_string,
                        chunk_index=chunk_idx,
                        chunk_name=transcript_file.name
                    )
                )
                tasks.append(task)

        # Wait for all tasks in this batch to complete
        results = await asyncio.gather(*tasks)

        # Process results and save analyses
        chunk_analyses: list[Tuple[int, ChunkAnalysisWithTimestamp]] = []
        for idx, analysis in results:

            chunk_analyses.append((idx, analysis))

            # Save successful analysis if it's new
            batch_item = next((item for item in batch if item[0] == idx), None)
            chunk_index, starting_time_string,  transcript_file = batch_item
            if batch_item:
                analysis_file = output_folder / f"{transcript_file.stem}.analysis.yaml"

                if not analysis_file.exists():
                    with open(analysis_file, 'w', encoding='utf-8') as f:
                        yaml.dump(analysis.model_dump(), f,
                                  default_flow_style=False,
                                  sort_keys=False,
                                  allow_unicode=True)
                    self.processing_stats['chunks_processed'] += 1

        # Sort by chunk index (idx) to maintain order
        chunk_analyses.sort(key=lambda x: x[0])


        return [analysis for _, analysis in chunk_analyses]

    async def process_transcript_folder(self,
                                        transcript_folder: Path,
                                        chunk_analysis_output_folder: Path,
                                        full_output_folder: Path) -> FullVideoAnalysis:
        """Process all transcript chunks in a folder with parallel processing."""

        start_time = time.perf_counter()

        chunk_analysis_output_folder.mkdir(parents=True, exist_ok=True)

        # Find all transcript JSON files
        transcript_files = sorted(transcript_folder.glob("*.transcript.json"))
        if not transcript_files:
            raise ValueError(f"No transcript files found in {transcript_folder}")

        print(f"\n🎬 Processing {len(transcript_files)} transcript files")
        print(f"⚡ Max concurrent processing: {self.max_concurrent_chunks}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"⏱️ Chunk length: {self.chunk_length_seconds}s, Overlap: {self.chunk_overlap_seconds}s\n")

        # Prepare file data with corrected timestamps
        file_data: list[tuple[int,StartingTimeString,  Path]] = []
        for transcript_file in transcript_files:
            chunk_index, start_time, start_time_string = self.parse_chunk_filename(transcript_file.name)
            file_data.append((chunk_index,start_time_string, transcript_file))

        # Process in batches
        all_chunk_analyses: list[ChunkAnalysisWithTimestamp]= []
        total_batches = (len(file_data) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(0, len(file_data), self.batch_size):
            batch = file_data[batch_idx:batch_idx + self.batch_size]
            current_batch_num = batch_idx // self.batch_size + 1


            batch_analyses = await self.process_transcript_batch(batch, chunk_analysis_output_folder)
            all_chunk_analyses.extend(batch_analyses)

            print(f"✅ Batch {current_batch_num} complete")

        # Print processing statistics
        elapsed = time.perf_counter() - start_time
        print(f"\n📊 Processing Statistics:")
        print(f"  • Total time: {elapsed:.1f}s")
        print(f"  • Chunks processed: {self.processing_stats['chunks_processed']}")
        print(f"  • Chunks cached: {self.processing_stats['chunks_cached']}")
        print(f"  • Average time per chunk: {elapsed / len(transcript_files):.1f}s")

        if self.processing_stats['errors']:
            print(f"  • Errors encountered: {len(self.processing_stats['errors'])}")
            for error in self.processing_stats['errors'][:5]:  # Show first 5 errors
                print(f"    - {error}")

        # Combine all analyses (or load from cache if already done)
        combined_file = chunk_analysis_output_folder.parent.parent / "full_video_analysis.yaml"


        # if combined_file.exists():
        #     print(f"\n📂 Using cached full video analysis from {combined_file}")
        #     with open(combined_file, 'r', encoding='utf-8') as f:
        #         full_analysis = FullVideoAnalysis(**yaml.safe_load(f))
        #     return full_analysis
        print("\n🔄 Combining all chunk analyses...")
        full_analysis = await self.combine_analyses(all_chunk_analyses)

        # Save combined analysis as YAML
        with open(combined_file, 'w', encoding='utf-8') as f:
            yaml.dump(full_analysis.model_dump(), f,
                      default_flow_style=False,
                      sort_keys=False,
                      allow_unicode=True)
        print(f"💾 Saved combined analysis to {combined_file}")


        return full_analysis

    async def combine_analyses(self,
                               chunk_analyses: list[ChunkAnalysisWithTimestamp]) -> FullVideoAnalysis:
        """Combine individual chunk analyses into a complete video analysis."""

        # Stage 1: Aggregate all data from chunks
        all_summaries = []
        all_themes = []
        all_takeaways = []
        all_topics = []
        all_pull_quotes = []
        all_clips = []

        for chunk in chunk_analyses:
            chunk_copy = deepcopy(chunk)
            all_summaries.append(chunk_copy.summary)
            all_themes.extend(chunk_copy.main_themes)
            all_takeaways.extend(chunk_copy.key_takeaways)
            all_topics.extend(chunk_copy.topic_areas)
            all_pull_quotes.extend(deepcopy(chunk_copy.pull_quotes))
            all_clips.append(chunk_copy.most_interesting_short_section)

        # Stage 2: Generate executive and detailed summaries first
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

        summary_response = await self.processor.async_make_openai_json_mode_ai_request(
            system_prompt=summary_prompt,
            input_data={},
            output_model=TranscriptSummaryPromptModel
        )
        print(
            f"________________________________________\n\n Summary Response:\n{summary_response}\n\n________________________________________")
        # Stage 5: Extract main themes and key takeaways
        all_topic_areas_prompt_string = "- " +"\n\n- ".join([str(t) for t in all_topics])
        all_themes_prompt_string = "- " + "\n\n- ".join(all_themes)
        all_takeaways_prompt_string = "- " +"\n\n- ".join(all_takeaways)
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

        themes_response = await self.processor.async_make_openai_json_mode_ai_request(
            system_prompt=themes_prompt,
            input_data={},
            output_model=ThemesAndTakeawaysPromptModel
        )

        print(
            f"________________________________________\n\n Themes Response:\n{themes_response}\n\n________________________________________")

        # Stage 6: Rank and select top pull quotes
        # all_pull_quotes_string = "\n".join(str(quote) for quote in all_pull_quotes)
        # if len(all_pull_quotes) > 3:
        #     quotes_prompt = f""" You will be given a summary and analysis of an extended video transcript along with pull quotes that were extracted from chunks of the video.
        #     Using this information, identify the TOP PULL QUOTES from the entire video, based on the following criteria:
        #     - Relevance to main themes and topics
        #     - Uniqueness, interestingness, non-obvious, non-repetitive
        #     - Interesting language, phrasing, or storytelling
        #
        #     Add the quotes to the list IN ORDER OF QUALITY, so that the BEST quotes are at the top of the list.
        #
        #     <<<<Full Video Summary and Analysis>>>>
        #     {summary_response}
        #     <<<<End Full Video Summary and Analysis>>>>
        #
        #     ----------------------------------------------------------------------------------------------------------------------------------------
        #     ----------------------------------------------------------------------------------------------------------------------------------------
        #     <<<<All Pull Quotes from Chunks>>>>
        #     {all_pull_quotes_string}
        #     <<<<End All Pull Quotes from Chunks>>>>
        #     ----------------------------------------------------------------------------------------------------------------------------------------
        #     ----------------------------------------------------------------------------------------------------------------------------------------
        #     Based on the above, provide the TOP PULL QUOTES from the entire video in accordance to  JSON format schema provided.
        #
        #     """
        #
        #     quotes_response = await self.processor.async_make_openai_json_mode_ai_request(
        #         system_prompt=quotes_prompt,
        #         input_data={},
        #         output_model=PullQuotesSelectionPromptModel
        #     )
        #
        #     top_pull_quotes = quotes_response.pull_quotes
        # else:
        #     top_pull_quotes = all_pull_quotes
        #sort by quality field (highest first)
        top_pull_quotes = sorted(deepcopy(all_pull_quotes), key=lambda x: x.quality, reverse=True)

        #Sort according to "quality" field (highest first)
        top_pull_quotes.sort(key=lambda x: x.quality, reverse=True)
        pull_quote_response_string = "\n".join(str(quote) for quote in top_pull_quotes)
        print(
            f"_________________________________________\n\n Pull Quotes Response:\n{pull_quote_response_string}\n\n________________________________________")

        # Stage 7: Rank and select top intesesting short clips
        if len(all_clips) > 3:
            all_clips_string = "\n".join(str(clip) for clip in all_clips)
            clips_prompt = f""" You will be given a summary and analysis of an extended video transcript along with particularly interesting 60 second clips that were extracted from chunks of the video.
            Using this information, identify the TOP particularly interesting 60-ish second short clips from the entire video, based on the following criteria:
            - Relevance to main themes and topics
            - Uniqueness, interestingness, non-obvious, non-repetitive
            - Interesting language, phrasing, or storytelling
            - Standalone interest and coherence (i.e. makes sense on its own without the rest of the video providing context)

            <<<<Full Video Summary and Analysis>>>>
            {summary_response}
            <<<<End Full Video Summary and Analysis>>>>
            
            ----------------------------------------------------------------------------------------------------------------------------------------     
            ----------------------------------------------------------------------------------------------------------------------------------------
            <<<<All Particularly Interesting
            {all_clips_string}
            <<<<End All Particularly Interesting 60 Second Clips from Chunks>>>>
            ----------------------------------------------------------------------------------------------------------------------------------------
            ----------------------------------------------------------------------------------------------------------------------------------------    
            Based on the above, provide the TOP Particularly Interesting 60 Second Clips from the entire video in accordance to  JSON format schema provided.
            
            """

            clips_response = await self.processor.async_make_openai_json_mode_ai_request(
                system_prompt=clips_prompt,
                input_data={},
                output_model=MostInterestingShortSectionSelectionPromptModel
            )

            top_clips = clips_response.most_interesting_short_section_candidates
        else:
            top_clips = all_clips
        # remove duplicates based on text_content
        seen_clips = set()
        unique_clips = []
        for clip in top_clips:
            if clip.text_content not in seen_clips:
                unique_clips.append(clip)
                seen_clips.add(clip.text_content)
        top_clips = unique_clips

        print(
            f"_________________________________________\n\n Clips Response:\n{[str(clip) for clip in top_clips]}\n\n________________________________________")

        # Create final FullVideoAnalysis
        full_analysis = FullVideoAnalysis(
            summary=summary_response,
            chunk_analyses=chunk_analyses,
            themes=themes_response.main_themes,
            topics=themes_response.topic_areas,
            takeaways=themes_response.key_takeaways,
            pull_quotes=top_pull_quotes,
            most_interesting_clips=top_clips
        )

        return full_analysis

