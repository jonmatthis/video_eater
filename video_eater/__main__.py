# __main__.py (updated with YouTube support)
import asyncio
from pathlib import Path
import click
import yaml
import logging

from video_eater.core.config_models import VideoProject, ProcessingConfig
from video_eater.core.handle_video.youtube_getter import YouTubeDownloader, CachedYouTubeDownloader
from video_eater.core.pipeline import VideoProcessingPipeline

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

DEFAULT_VIDEO_INPUTS = [
    # Your existing default paths
    # r"C:\Users\jonma\syncthing-folders\jon-alienware-pc-synology-nas-sync\videos\livestream_videos\2025-08-14-JSM-Livestream-Skellycam\2025-08-14-JSM-Livestream-Skellycam.mp4",
    # r"C:\Users\jonma\syncthing-folders\jon-alienware-pc-synology-nas-sync\videos\livestream_videos\2025-08-07-JSM-Livestream\2025-08-07-JSM-Livestream-RAW.mp4",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
]

DEFAULT_DOWNLOAD_DIR = r"C:\Users\jonma\syncthing-folders\jon-alienware-pc-synology-nas-sync\videos\video_eater_downloads"

async def process_input(input_path: str,
                        downloader: YouTubeDownloader,
                        pipeline: VideoProcessingPipeline) -> None:
    """Process a single input (local file or YouTube URL)."""

    # Check if it's a YouTube URL
    if YouTubeDownloader.is_youtube_url(url=input_path):
        print(f"Detected YouTube URL: {input_path}")

        try:
            # Download the video
            video_path: Path = await downloader.download(url=input_path)
            print(f"Downloaded to: {video_path}")
        except Exception as e:
            logger.error(f"Failed to download {input_path}: {e}")
            raise
    else:
        # Local file path
        video_path = Path(input_path)
        if not video_path.exists():
            logger.error(f"File not found: {video_path}")
            raise

    # Process the video through the existing pipeline
    project: VideoProject = VideoProject(video_path=video_path)
    result = await pipeline.process_video(project=project)
    print(result.summary_report())


@click.command()
@click.argument('inputs', nargs=-1)
@click.option('--config', type=click.Path(), help='Config file path')
@click.option('--force-all', is_flag=True, help='Force reprocess everything')
@click.option('--download-dir', type=click.Path(),
              default=DEFAULT_DOWNLOAD_DIR, help='Directory for YouTube downloads')
@click.option('--youtube-quality', default='best',
              help='YouTube video quality (best, 1080, 720, etc.)')
@click.option('--audio-only', is_flag=True,
              help='Download only audio from YouTube videos')
@click.option('--use-cache', is_flag=True,
              help='Use cached YouTube downloads if available')
@click.option('--from-file', type=click.Path(exists=True),
              help='Read input URLs/paths from a text file (one per line)')
@click.option('--playlist', is_flag=True,
              help='Treat YouTube URLs as playlists')
@click.option('--max-videos', type=int,
              help='Maximum videos to download from playlist')
def main(inputs: tuple[str, ...],
         config: str | None,
         force_all: bool,
         download_dir: str,
         youtube_quality: str,
         audio_only: bool,
         use_cache: bool,
         from_file: str | None,
         playlist: bool,
         max_videos: int | None) -> None:
    """
    Process videos through the transcription and analysis pipeline.

    Accepts both local video files and YouTube URLs.

    Examples:
        # Process local video
        python -m video_eater /path/to/video.mp4

        # Download and process YouTube video
        python -m video_eater https://www.youtube.com/watch?v=VIDEO_ID

        # Process multiple inputs
        python -m video_eater video1.mp4 https://youtu.be/VIDEO_ID video2.mp4

        # Download YouTube playlist
        python -m video_eater --playlist https://www.youtube.com/playlist?list=PLAYLIST_ID

        # Read inputs from file
        python -m video_eater --from-file urls.txt
    """

    # Setup logging

    # Load configuration
    processing_config: ProcessingConfig
    if config:
        config_data: dict[str, object] = yaml.safe_load(Path(config).read_text())
        processing_config = ProcessingConfig(**config_data)
    else:
        processing_config = ProcessingConfig()

    if force_all:
        processing_config.force_chunk_audio = True
        processing_config.force_transcribe = True
        processing_config.force_analyze = True

    # Gather all inputs
    all_inputs: list[str] = list(inputs)
    if not all_inputs:
        all_inputs = DEFAULT_VIDEO_INPUTS.copy()

    # Add inputs from file if specified
    if from_file:
        with open(file=from_file, mode='r') as f:
            file_inputs: list[str] = [line.strip() for line in f if line.strip()]
            all_inputs.extend(file_inputs)

    if not download_dir:
        download_dir = DEFAULT_DOWNLOAD_DIR

        if playlist:
            download_dir = str(Path(download_dir) / "playlists")


    # Setup YouTube downloader
    download_path: Path = Path(download_dir)
    downloader: YouTubeDownloader
    if use_cache:
        downloader = CachedYouTubeDownloader(
            output_dir=download_path,
            quality=youtube_quality,
            audio_only=audio_only,
        )
    else:
        downloader = YouTubeDownloader(
            output_dir=download_path,
            quality=youtube_quality,
            audio_only=audio_only,
        )

    # Setup processing pipeline
    pipeline: VideoProcessingPipeline = VideoProcessingPipeline(
        config=processing_config,
    )

    # Run async processing
    async def process_all() -> None:
        # Handle playlist URLs specially
        playlist_urls: list[str] = []
        regular_inputs: list[str] = []

        for input_path in all_inputs:
            if playlist and YouTubeDownloader.is_youtube_url(url=input_path):
                playlist_urls.append(input_path)
            else:
                regular_inputs.append(input_path)

        # Download playlists first
        for playlist_url in playlist_urls:
            try:
                print(f"Processing playlist: {playlist_url}")
                video_files: list[Path] = await downloader.download_playlist(
                    url=playlist_url,
                    max_videos=max_videos
                )
                # Add downloaded videos to regular inputs
                regular_inputs.extend(str(f) for f in video_files)
            except Exception as e:
                logger.error(f"Failed to process playlist {playlist_url}: {e}")

        # Process all inputs (including downloaded playlist videos)
        for input_path in regular_inputs:
            await process_input(
                input_path=input_path,
                downloader=downloader,
                pipeline=pipeline,
            )

    # Run the async processing
    asyncio.run(main=process_all())


if __name__ == "__main__":
    main()