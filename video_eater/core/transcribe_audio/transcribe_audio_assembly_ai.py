# Simple AssemblyAI transcription function
import asyncio

from video_eater.core.transcribe_audio.transcript_models import VideoTranscript

from assemblyai.types import SummarizationType, SummarizationModel


async def async_transcribe_with_assemblyai(audio_file_path: str, api_key: str) -> VideoTranscript:
    """
    Transcribe an audio file using AssemblyAI.

    Args:
        audio_file_path (str): Path to the audio file to transcribe
        api_key (str): AssemblyAI API key

    Returns:
        dict: Transcription result with text and other details
    """
    import assemblyai as aai

    # Set API key
    aai.settings.api_key = api_key

    config = aai.TranscriptionConfig(punctuate=True,
                                     format_text=True,
                                     word_boost=["FreeMoCap", "SkellyCam"],
                                     # auto_chapters=True,
                                     # summarization=True,
                                     # summary_model=SummarizationModel.informative,
                                     # summary_type=SummarizationType.bullets_verbose,
                                     # entity_detection=True,
                                     # auto_highlights=True,
                                     # iab_categories=True
                                     )

    future = aai.Transcriber(config=config).transcribe_async(audio_file_path)
    transcript = await asyncio.wrap_future(future)
    # Check for errors
    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    # Return transcription result
    return VideoTranscript.from_assembly_ai_output({
        "text": transcript.text,
        "paragraphs":transcript.get_paragraphs()
    })