import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Type, TypeVar

from openai import AsyncOpenAI, OpenAI
from openai.types.audio import TranscriptionVerbose
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseAIProcessor:
    """Base class for AI processors with common functionality."""

    def __init__(self, force_refresh: bool = False, model: str = "gpt-4o-mini", use_async:bool = True):
        
        self.force_refresh = force_refresh
        if use_async:
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=600)
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=600)
        self.model = model

    async def async_make_openai_json_mode_ai_request(
        self, system_prompt: str, input_data: dict, output_model: Type[T]
    ) -> T:
        """Make an OpenAI request with JSON mode for structured output."""
        messages = [{"role": "system", "content": system_prompt}]
        if input_data:
            messages.append({"role": "user", "content": json.dumps(input_data)})

        try:
            # Use the parse method with the Pydantic model directly
            response = await self.client.responses.parse(
                model=self.model,
                input=messages,
                text_format=output_model,
            )

            # The parse method returns the parsed model directly
            return response.output_parsed
        except Exception as e:
            logger.error(f"Error in OpenAI JSON request: {e}")
            raise

    async def async_make_openai_text_request(self, system_prompt: str) -> str:
        """Make an OpenAI request for text generation."""
        messages = [{"role": "system", "content": system_prompt}]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in OpenAI text request: {e}")
            raise

    async def async_make_whisper_transcription_request(
        self, audio_file_path: str, prompt: str | None = None
    ) -> TranscriptionVerbose:
        audio_file = open(audio_file_path, "rb")
        print(f"Transcribing {audio_file_path} ")
        transcript_response = await self.client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            prompt=prompt,
            timestamp_granularities=["segment"],
        )
        print(f"Transcription completed for {audio_file_path}!")
        return transcript_response
    
 