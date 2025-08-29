import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt

from openai.types.audio import TranscriptionVerbose


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

TRANSCRIPTION_BASE_PROMPT ="Words you may hear, with their correct spelling: `Jon Matthis`, `Jonathan Samir Matthis`, `FreeMoCap`, SkellyCam"

def validate_audio_path(audio_path: str) -> None:
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"File not found: {audio_path}")
    if not Path(audio_path).is_file():
        raise ValueError(f"Path is not a file: {audio_path}")
    if not Path(audio_path).suffix in [".mp3", ".ogg", ".wav"]:
        raise ValueError(f"Unsupported file format: {audio_path}")



def transcribe_audio_with_local_whisper(audio_path: str, model_name: str = "large",save_mel_spectrogram_image:bool=False) -> TranscriptionVerbose:
    import whisper
    import torch
    logger.info(
        f"Transcribing audio: {audio_path} with whisper model: {model_name} - import torch; torch.cuda_is_available(): {torch.cuda.is_available()}")

    validate_audio_path(audio_path)
    model = whisper.load_model(model_name)
    # make log-Mel spectrogram and move to the same device as the model

    result = model.transcribe(audio = audio_path,
                              word_timestamps=True,
                              temperature=0.0,
                              no_speech_threshold=0.5,
                              hallucination_silence_threshold=0.5,
                              initial_prompt=TRANSCRIPTION_BASE_PROMPT,
                              )
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    if save_mel_spectrogram_image:
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        save_spectrogram_image(audio_path, mel)
    return TranscriptionVerbose(**result)


def transcribe_audio_detailed(audio_path: str,
                              model_name: str = "turbo",
                              ):
    import whisper

    model = whisper.load_model(model_name)

    # validate/load audio and pad/trim it to fit 30 seconds
    validate_audio_path(audio_path)
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    save_spectrogram_image(audio_path, mel)
    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode (transcribe) the audio
    transcription_result = whisper.decode(model=model,
                                          mel=mel,
                                          options=whisper.DecodingOptions())

    # print the recognized text
    print(transcription_result.text)
    return transcription_result


def save_spectrogram_image(audio_path, mel):
    # Using matplotlib instead of cv2
    mel_as_numpy = mel.cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.imshow(mel_as_numpy, aspect='auto', origin='lower', cmap='plasma')
    plt.colorbar()
    plt.title('Log-Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')

    output_path = Path(audio_path).with_suffix(".log_mel_spectrogram.png")
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    _audio_path = r"\\jon-nas\jon-nas\videos\livestream_videos\2025-08-14-JSM-Livestream-Skellycam\audio_chunks\2025-08-14-JSM-Livestream-Skellycam_chunk_013_02h-06m-45sec.mp3"
    transcribe_audio_with_local_whisper(audio_path=_audio_path)