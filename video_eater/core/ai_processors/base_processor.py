import json
import json
import logging
import os
from pathlib import Path
from typing import Type, TypeVar

from openai import AsyncOpenAI, OpenAI
from openai.types.audio import TranscriptionVerbose
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

load_dotenv()
use_deepseek = True
if use_deepseek:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise EnvironmentError("DEEPSEEK_API_KEY not found in env")

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not found in env")

class BaseAIProcessor:
    """Base class for AI processors with common functionality."""

    def __init__(self, force_refresh: bool = False, model: str = "deepseek-chat", use_async:bool = True):
        
        self.force_refresh = force_refresh
        api_url = "https://api.deepseek.com" if use_deepseek else None
        if use_async:
            self.text_client = AsyncOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY") if use_deepseek else os.getenv('OPENAI_API_KEY'),
                                           base_url=api_url,
                                           timeout=600)
            self.transcription_client = AsyncOpenAI(api_key= os.getenv('OPENAI_API_KEY'),
                                           timeout=600)
        else:
            self.text_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY") if use_deepseek else os.getenv('OPENAI_API_KEY'),
                                      base_url=api_url,
                                      timeout=600)
            self.transcription_client = OpenAI(api_key= os.getenv('OPENAI_API_KEY'),
                                           timeout=600)

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
            if use_deepseek:
                messages[0]['content']+=f"""
                YOUR RESPONSE MUST MATCH THE FOLLOWING JSON SCHEMA:
                ==================================
                {output_model.model_json_schema()}
                ==================================
                
                DO NOT INCLUDE ANY TEXT IN YOUR RESPONSE OTHER THAN THE JSON FORMATED RESPONSE STRING
                """
                response = await self.text_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={
                        'type': 'json_object'
                    }
                )
                return output_model(**json.loads(response.choices[0].message.content))

            else:
                response = await self.text_client.responses.parse(
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
            response = await self.text_client.chat.completions.create(
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
        transcript_response = await self.transcription_client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            prompt=prompt,
            timestamp_granularities=["segment"],
        )
        print(f"Transcription completed for {Path(audio_file_path).stem}!")
        return transcript_response
    
 