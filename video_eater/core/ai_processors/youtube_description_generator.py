# youtube_description_generator.py
import json
from copy import copy
from typing import Optional

from video_eater.core.ai_processors.base_processor import BaseAIProcessor
from video_eater.core.ai_processors.ai_prompt_models import FullVideoAnalysis, YouTubeDescriptionPromptModel
from video_eater.core.output_templates import YouTubeDescriptionFormatter


class YouTubeDescriptionGenerator:
    """Generate a YouTube description from FullVideoAnalysis with a simple refinement loop."""

    def __init__(self, model: str, max_len: int = 5000, target_min: float = 0.85, target_max: float = 0.99999,
                 max_iters: int = 3):
        self.processor = BaseAIProcessor(model=model, use_async=True)
        self.max_len = max_len
        self.target_min = target_min
        self.target_max = target_max
        self.max_iters = max_iters

    @staticmethod
    def _count_chars(text: str) -> int:
        return len(text or "")

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total = int(seconds)
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    async def generate(self, analysis: FullVideoAnalysis) -> YouTubeDescriptionPromptModel:
        """Generate and refine a description to fall within the target character range."""
        target_low = int(self.max_len * self.target_min)
        target_high = min(int(self.max_len * self.target_max), len(analysis.model_dump_json()))

        draft_text: Optional[str] = None
        too_short = False
        too_long = False
        og_system_prompt = ""
        revise_system_prompt = None
        candidate_model: YouTubeDescriptionPromptModel|None = None
        for attempt in range(self.max_iters):
            if draft_text is None:
                og_system_prompt = f"""
You are writing a YouTube video description as a structured JSON object.
Fill the YouTubeDescriptionPromptModel fields using the provided sources.

Here is the full video analysis data:

<<<ANALYSIS-START>>>
{analysis.model_dump_json(indent=2)}
<<<ANALYSIS-END>>>

"""
            else:
                if too_short:
                    revise_system_prompt = copy(og_system_prompt)
                    revise_system_prompt += f"""
                    Here is your previous output
                    <<<PREVIOUS-OUTPUT-START>>>
                    {candidate_model.model_dump_json(indent=2) if candidate_model else 'N/A'}
                    <<<PREVIOUS-OUTPUT-END>>>
                    Your previous output was TOO SHORT.
                    Please EXPAND the content by pulling more information from the full analsyis and adding to the Youtube description. 
                     while keeping it relevant and engaging,and ONLY USING INFO FROM THE ORIGINAL ANALYSIS. DO NOT ADD ANYTHING THAT IS NOT PRESENT IN THE ORIGINAL ANALYSIS JSON 
                    """
                else:  # too_long
                    revise_system_prompt = copy(og_system_prompt)
                    revise_system_prompt += f"""
                    Here is your previous output
                    <<<PREVIOUS-OUTPUT-START>>>
                    {candidate_model.model_dump_json(indent=2) if candidate_model else 'N/A'}
                    <<<PREVIOUS-OUTPUT-END>>>
                    Your previous output was TOO LONG .
                    Please CONDENSE the content while ensuring that the main points and key information are retained.
                    """

            # Request structured JSON and render to text for length checks
            candidate_model = await self.processor.async_make_openai_json_mode_ai_request(
                system_prompt=og_system_prompt if draft_text is None else revise_system_prompt,
                input_data={},
                output_model=YouTubeDescriptionPromptModel
            )
            candidate_text = YouTubeDescriptionFormatter().format(candidate_model).strip()
            length = self._count_chars(candidate_text)

            if target_low <= length <= target_high:
                return candidate_model
            elif length < target_low:
                too_short = True
                too_long = False
            else:
                too_short = False
                too_long = True

            draft_text = candidate_text  # keep best-so-far

        return candidate_model