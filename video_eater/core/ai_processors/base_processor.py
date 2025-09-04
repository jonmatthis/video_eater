import json
import logging
import os
from pathlib import Path
from typing import Type, TypeVar

from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI, OpenAI
from openai.types.audio import TranscriptionVerbose
from pydantic import BaseModel

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not found in env")


class BaseAIProcessor:
    """Base class for AI processors with common functionality."""

    def __init__(self, model: str|None, force_refresh: bool = False, use_async: bool = True):
        self.use_deepseek = "deepseek" in model
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise EnvironmentError("DEEPSEEK_API_KEY not found in env")

        self.force_refresh = force_refresh
        api_url = "https://api.deepseek.com" if self.use_deepseek else None
        if use_async:
            self.text_client = AsyncOpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY") if self.use_deepseek else os.getenv('OPENAI_API_KEY'),
                base_url=api_url,
                timeout=600)
            self.transcription_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                                                    timeout=600)
        else:
            self.text_client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY") if self.use_deepseek else os.getenv('OPENAI_API_KEY'),
                base_url=api_url,
                timeout=600)
            self.transcription_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                                               timeout=600)

        self.model = model

    async def async_make_openai_json_mode_ai_request(
            self, system_prompt: str, input_data: dict, output_model: Type[T]
    ) -> T:
        """Make an OpenAI request with JSON mode for structured output."""
        if not self.model:
            raise ValueError("Model must be specified for to make OpenAI requests.")
        messages = [{"role": "system", "content": system_prompt}]
        if input_data:
            messages.append({"role": "user", "content": json.dumps(input_data)})

        try:
            # Use the parse method with the Pydantic model directly
            if self.use_deepseek:
                messages[0]['content'] += f"""
                YOUR RESPONSE MUST MATCH THE FOLLOWING JSON SCHEMA:
                ==================================
                ==================================
                
                {output_model.model_json_schema()}
                
                ==================================
                ==================================
                Please provide a VALID JSON response that:
                    1. Contains ONLY valid JSON (no extra text before or after)
                    2. Matches the required schema
                    3. Has properly escaped strings and valid JSON syntax
                DO NOT INCLUDE ANY TEXT IN YOUR RESPONSE OTHER THAN THE JSON FORMATED RESPONSE STRING
                """

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = await self.text_client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            response_format={
                                'type': 'json_object'
                            }
                        )

                        response_content = response.choices[0].message.content

                        # Try to parse the JSON
                        try:
                            parsed_json = json.loads(response_content)
                            return output_model(**parsed_json)
                        except json.JSONDecodeError as json_error:
                            if attempt < max_retries - 1:
                                # Create a fix-it message
                                logger.warning(f"JSON decode error on attempt {attempt + 1}: {json_error}")

                                fix_messages = messages.copy()
                                fix_messages.append({
                                    "role": "assistant",
                                    "content": response_content
                                })
                                fix_messages.append({
                                    "role": "user",
                                    "content": f"""The JSON response you provided is malformed and caused this error:
                                    {str(json_error)}

                                    Your response was:
                                    {response_content[:500]}{'...' if len(response_content) > 500 else ''}

                                    Please provide a VALID JSON response that:
                                    1. Contains ONLY valid JSON (no extra text before or after)
                                    2. Matches the required schema
                                    3. Has properly escaped strings and valid JSON syntax

                                    Return ONLY the corrected JSON object, nothing else."""
                                })

                                # Update messages for the next attempt
                                messages = fix_messages
                            else:
                                # Final attempt failed
                                logger.error(f"Failed to get valid JSON after {max_retries} attempts")
                                raise json_error
                        except Exception as pydantic_error:
                            if attempt < max_retries - 1:
                                # Pydantic validation error
                                logger.warning(f"Pydantic validation error on attempt {attempt + 1}: {pydantic_error}")

                                fix_messages = messages.copy()
                                fix_messages.append({
                                    "role": "assistant",
                                    "content": response_content
                                })
                                fix_messages.append({
                                    "role": "user",
                                    "content": f"""The JSON you provided doesn't match the required schema. Error:
                                    {str(pydantic_error)}

                                    The required schema is:
                                    {output_model.model_json_schema()}

                                    Please provide a corrected JSON response that exactly matches this schema.
                                    Return ONLY the valid JSON object."""
                                })

                                messages = fix_messages
                            else:
                                logger.error(f"Failed to get valid schema after {max_retries} attempts")
                                raise pydantic_error

                    except Exception as api_error:
                        # API call failed
                        logger.error(f"API call failed on attempt {attempt + 1}: {api_error}")
                        if attempt < max_retries - 1:
                            continue
                        else:
                            raise api_error

            else:  # OpenAI (not deepseek)

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
        if not self.model:
            raise ValueError("Model must be specified for to make OpenAI requests.")
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
