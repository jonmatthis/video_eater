# __main__.py (updated with YouTube support)
import asyncio
import logging
from pathlib import Path

import yaml
# Add to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))
from video_eater.core.config_models import ProcessingConfig, VideoProject, SourceType
from video_eater.core.handle_video.youtube_getter import YouTubeDownloader, CachedYouTubeDownloader
from video_eater.core.pipeline import VideoProcessingPipeline

logger = logging.getLogger(__name__)


# Accepts: Local files, single YouTube URLs, YouTube playlists
DEFAULT_VIDEO_INPUTS = [
    # "https://www.youtube.com/watch?v=A-YC6a6VTGs", #JKL AI Brief History
    # "https://www.youtube.com/playlist?list=PLCZJ-1jWKKw4GMxu8VvIPupXmJVG0Wrt5", #JKL AI Short Course playlist
    # r"C:\Users\jonma\syncthing-folders\jon-alienware-pc-synology-nas-sync\Sync\freemocap-stuff\freemocap-clients\ben-scholl\paper-review\New folder\BS_ferret_paper_review_video.mp4"
    # r"C:\Users\jonma\syncthing-folders\jon-alienware-pc-synology-nas-sync\videos\livestream_videos\2025-08-14-JSM-Livestream-Skellycam\2025-08-14-JSM-Livestream-Skellycam.mp4",
    # r"C:\Users\jonma\syncthing-folders\jon-alienware-pc-synology-nas-sync\videos\livestream_videos\2025-08-07-JSM-Livestream\2025-08-07-JSM-Livestream-RAW.mp4",
    # "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
     "https://www.youtube.com/playlist?list=PLWxH2Ov17q5HRyRc7_HD5baSYB6kBgsTj", # [2024-Fall] Neural Control of Real-World Human Movement playlist
     "https://www.youtube.com/playlist?list=PLWxH2Ov17q5HDfMBJxD_cE1lowM1cr_BV", # [2025-Spring] Neural Control of Real-World Human Movement playlist
]

DEFAULT_DOWNLOAD_DIR = r"C:\Users\jonma\syncthing-folders\jon-alienware-pc-synology-nas-sync\videos\video_eater_downloads"


async def process_input(input_path: str,
                        downloader: YouTubeDownloader,
                        pipeline: VideoProcessingPipeline,
                        is_from_playlist: bool = False,
                        playlist_name: str | None = None) -> None:
    """Process a single input (local file or YouTube URL)."""
    
    # Initialize source tracking
    source_type: str = SourceType.FILE
    source_url: str | None = None
    video_id: str | None = None
    
    # Check if it's a YouTube URL
    if YouTubeDownloader.is_youtube_url(url=input_path):
        print(f"Detected YouTube URL: {input_path}")
        source_type = SourceType.PLAYLIST if is_from_playlist else SourceType.YOUTUBE
        source_url = input_path
        
        try:
            # Get video info to extract video ID
            video_info = await downloader.get_video_info(url=input_path)
            video_id = video_info.get('video_id', None)
            
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
    
    # Create project with source information
    project: VideoProject = VideoProject(
        video_path=video_path,
        source_type=source_type,
        source_url=source_url,
        playlist_name=playlist_name,
        video_id=video_id
    )
    
    result = await pipeline.process_video(project=project)
    print(result.summary_report())

def main(inputs: tuple[str, ...] | None = None,
         config: ProcessingConfig | None = None,
         force_all: bool = False,
         download_dir: str | None = None,
         youtube_quality: str = "best",
         audio_only: bool = False,
         use_cache: bool = True,
         from_file: str | None = None,
         max_videos: int | None = None) -> None:
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

    if config:
        if isinstance(config, ProcessingConfig):
            processing_config = config
        else:
            config_data: dict[str, object] = yaml.safe_load(Path(config).read_text())
            processing_config = ProcessingConfig(**config_data)
    else:
        processing_config = ProcessingConfig()

    if force_all:
        processing_config.force_chunk_audio = True
        processing_config.force_transcribe = True
        processing_config.force_analyze = True

    # Gather all inputs
    all_inputs: list[str] = list(inputs) if inputs else []
    if not all_inputs:
        all_inputs = DEFAULT_VIDEO_INPUTS.copy()

    # Add inputs from file if specified
    if from_file:
        with open(file=from_file, mode='r') as f:
            file_inputs: list[str] = [line.strip() for line in f if line.strip()]
            all_inputs.extend(file_inputs)

    if not download_dir:
        download_dir = DEFAULT_DOWNLOAD_DIR

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
            if "playlist" in input_path and YouTubeDownloader.is_youtube_url(url=input_path):
                playlist_urls.append(input_path)
            else:
                regular_inputs.append(input_path)

        # Download playlists first
        download_tasks = []
        for playlist_url in playlist_urls:

            print(f"Processing playlist: {playlist_url}")

            # Get playlist info
            logger.info(f"Fetching playlist info for {playlist_url}")
            playlist_info = await downloader.get_video_info(url=playlist_url)
            playlist_name = playlist_info.get('title', 'Unknown Playlist')
            logger.info(f"Playlist title: {playlist_name}")
            # video_files: list[Path] =\
            download_tasks.append(asyncio.create_task(downloader.download_playlist(
                url=playlist_url,
                max_videos=max_videos
            )))

        results = await asyncio.gather(*download_tasks, return_exceptions=True)
        video_paths = results[0] if results else []
        for v in results:
            if isinstance(v, Exception):
                logger.error(f"Error downloading playlist: {v}")
        video_files: list[Path] = [item for sublist in results if isinstance(sublist, list) for item in sublist]

        # Process each video from playlist with playlist context
        for video_file in video_files:
            # Extract video URL from the downloaded file's parent directory name
            # (YouTube downloader creates folders with video IDs)
            video_id = video_file.parent.name.split('-')[-1] if '-' in video_file.parent.name else None
            video_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else None

            await process_input(
                input_path=str(video_file),
                downloader=downloader,
                pipeline=pipeline,
                is_from_playlist=True,
                playlist_name=playlist_name
            )
            # Add downloaded videos to regular inputs
            regular_inputs.extend(str(f) for f in video_files)

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
    main()#config=ProcessingConfig(force_analyze=True))
