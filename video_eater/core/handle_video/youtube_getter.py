import asyncio
import logging
import re
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)

class YouTubeDownloader:
    """Downloads videos from YouTube URLs using yt-dlp."""

    def __init__(self,
                 output_dir: Path,
                 quality: str = "best",
                 audio_only: bool = False) -> None:
        """
        Initialize YouTube downloader.

        Args:
            output_dir: Directory to save downloaded videos
            quality: Video quality (best, 1080, 720, etc.)
            audio_only: If True, download only audio
        """
        self.output_dir: Path = output_dir
        self.quality: str = quality
        self.audio_only: bool = audio_only
        self._browser_cookies_cache: tuple[str, ...] | None = None  # Cache the browser selection

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_youtube_url(url: str) -> bool:
        """Check if the given string is a YouTube URL."""
        youtube_patterns: list[str] = [
            r'(https?://)?(www\.)?(youtube\.com|youtu\.be|m\.youtube\.com)/',
            r'(https?://)?(www\.)?youtube\.com/watch\?v=',
            r'(https?://)?(www\.)?youtu\.be/',
            r'(https?://)?(www\.)?youtube\.com/embed/',
            r'(https?://)?(www\.)?youtube\.com/v/',
        ]
        return any(re.match(pattern=pattern, string=url) for pattern in youtube_patterns)

    def _get_browser_cookies(self) -> tuple[str, ...] | None:
        """
        Try to find a browser with valid YouTube cookies.
        Returns browser cookie config or None if no valid browser found.
        """
        # Use cached value if available
        if self._browser_cookies_cache is not None:
            return self._browser_cookies_cache

        browsers: list[str] = ['firefox', 'chrome', 'edge', 'brave', 'chromium', 'opera', 'safari']

        for browser in browsers:
            try:
                logger.info(f"Attempting to extract YouTube cookies from {browser}...")

                # Test if we can extract cookies from this browser
                test_opts: dict[str, object] = {
                    'quiet': True,
                    'cookiesfrombrowser': (browser,),
                }

                # Try to create a YoutubeDL instance with these options
                with yt_dlp.YoutubeDL(params=test_opts) as ydl:
                    # If we get here, browser cookies are accessible
                    logger.info(f"✓ Successfully configured cookies from {browser}")
                    self._browser_cookies_cache = (browser,)
                    return self._browser_cookies_cache

            except Exception as e:
                logger.debug(f"Could not use {browser}: {str(e)}")
                continue

        # If we get here, no browser worked
        logger.error(
            "=" * 60 + "\n" +
            "YOUTUBE AUTHENTICATION REQUIRED!\n" +
            "=" * 60 + "\n" +
            "YouTube is blocking bot access. To fix this:\n\n" +
            "1. Open one of these browsers: Firefox, Chrome, Edge, Brave, or Chromium\n" +
            "2. Go to youtube.com and sign in to your account\n" +
            "3. Make sure you can play videos (proves you're logged in)\n" +
            "4. Keep the browser profile/session active (don't clear cookies)\n" +
            "5. Close the browser before running this script\n\n" +
            "The script will automatically extract cookies from your browser.\n" +
            "=" * 60
        )
        self._browser_cookies_cache = None
        return None

    def _get_ydl_opts(self, output_template: str) -> dict[str, object]:
        """Get yt-dlp options configuration."""
        opts: dict[str, object] = {
            'outtmpl': output_template,
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
            'cachedir': str(self.output_dir / '.cache'),
            'progress_hooks': [self._progress_hook],
            'postprocessor_hooks': [self._postprocessor_hook],
        }

        # Try to add browser cookies
        browser_cookies: tuple[str, ...] | None = self._get_browser_cookies()
        if browser_cookies:
            opts['cookiesfrombrowser'] = browser_cookies
        else:
            logger.warning("No browser cookies configured - YouTube downloads may fail!")

        if self.audio_only:
            opts.update({
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
            })
        else:
            # Video quality selection
            if self.quality == "best":
                opts['format'] = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
            elif self.quality.isdigit():
                # Specific resolution like "1080" or "720"
                opts[
                    'format'] = f'bestvideo[height<={self.quality}][ext=mp4]+bestaudio[ext=m4a]/best[height<={self.quality}][ext=mp4]/best'
            else:
                opts['format'] = self.quality

        return opts

    def _progress_hook(self, d: dict[str, object]) -> None:
        """Hook called during download progress."""
        if d['status'] == 'downloading':
            percent: str = d.get('_percent_str', 'N/A')
            speed: str = d.get('_speed_str', 'N/A')
            eta: str = d.get('_eta_str', 'N/A')
            logger.info(f"Downloading: {percent} | Speed: {speed} | ETA: {eta}")
        elif d['status'] == 'finished':
            logger.info(f"Download complete: {d.get('filename', 'Unknown')}")

    def _postprocessor_hook(self, d: dict[str, object]) -> None:
        """Hook called during post-processing."""
        if d['status'] == 'started':
            logger.info(f"Post-processing: {d.get('postprocessor', 'Unknown')}")
        elif d['status'] == 'finished':
            logger.info("Post-processing complete")

    def _sanitize_filename(self, title: str) -> str:
        """Sanitize video title for use as filename."""
        # Remove invalid characters for filenames
        invalid_chars: str = '<>:"/\\|?*'
        for char in invalid_chars:
            title = title.replace(char, '')
        # Replace multiple spaces with single space
        title = ' '.join(title.split())
        # Limit length
        if len(title) > 200:
            title = title[:200]
        return title.strip()

    async def get_video_info(self, url: str) -> dict[str, object]:
        """Get video metadata without downloading."""
        opts: dict[str, object] = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True,
        }

        # Try to add browser cookies
        browser_cookies: tuple[str, ...] | None = self._get_browser_cookies()
        if browser_cookies:
            opts['cookiesfrombrowser'] = browser_cookies

        def _extract_info() -> dict[str, object]:
            with yt_dlp.YoutubeDL(params=opts) as ydl:
                return ydl.extract_info(url=url, download=False)

        try:
            info: dict[str, object] = await asyncio.to_thread(_extract_info)
            return {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'uploader': info.get('uploader', 'Unknown'),
                'upload_date': info.get('upload_date', ''),
                'view_count': info.get('view_count', 0),
                'url': url,
                'video_id': info.get('id', ''),
            }
        except Exception as e:
            if "Sign in to confirm you're not a bot" in str(e):
                logger.error(
                    "\n" + "=" * 60 + "\n" +
                    "YOUTUBE BOT DETECTION TRIGGERED!\n" +
                    "=" * 60 + "\n" +
                    "YouTube requires authentication to download this video.\n\n" +
                    "Quick fix:\n" +
                    "1. Open Firefox, Chrome, Edge, Brave, or Chromium\n" +
                    "2. Sign in to your YouTube account\n" +
                    "3. Try running this script again\n\n" +
                    "The script tried to find cookies from your browsers but couldn't\n" +
                    "find a valid logged-in session.\n" +
                    "=" * 60
                )
            else:
                logger.error(f"Failed to get video info: {e}")
            raise

    async def download(self,
                       url: str,
                       custom_filename: str | None = None,
                       subfolder: str | None = None) -> Path:
        """
        Download video from YouTube URL.

        Args:
            url: YouTube video URL
            custom_filename: Custom filename (without extension)
            subfolder: Optional subfolder within output_dir. defaults to /[video_name-and-id]

        Returns:
            Path to downloaded video file
        """
        if not self.is_youtube_url(url=url):
            raise ValueError(f"Not a valid YouTube URL: {url}")

        # Get video info first
        logger.info(f"Fetching video info from: {url}")
        info: dict[str, object] = await self.get_video_info(url=url)

        # Determine output path
        output_dir: Path = self.output_dir
        if subfolder:
            output_dir = output_dir / subfolder
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Default subfolder: title-videoid
            safe_title: str = self._sanitize_filename(title=str(info['title']))
            default_subfolder: str = f"{safe_title}-{info.get('video_id')}"
            output_dir = output_dir / default_subfolder
            output_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename
        if custom_filename:
            base_filename: str = self._sanitize_filename(title=custom_filename)
        else:
            # Use video title and upload date
            title: str = self._sanitize_filename(title=str(info['title']))
            upload_date: str = str(info.get('upload_date', ''))
            if upload_date:
                date_str: str = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
                base_filename = f"{date_str}-{title}"
            else:
                base_filename = title

        # Set file extension based on download type
        extension: str = "mp3" if self.audio_only else "mp4"
        if 'playlist' in str(output_dir):
            output_dir = output_dir / base_filename

        output_template: str = str(output_dir / f"{base_filename}.%(ext)s")
        expected_output: Path = output_dir / f"{base_filename}.{extension}"

        # Check if already downloaded
        if expected_output.exists():
            logger.info(f"Video already downloaded: {expected_output}")
            return expected_output

        # Configure yt-dlp
        ydl_opts: dict[str, object] = self._get_ydl_opts(output_template=output_template)

        # Download video
        logger.info(f"Starting download: {info['title']}")
        logger.info(f"Duration: {info['duration']} seconds")

        def _download() -> None:
            with yt_dlp.YoutubeDL(params=ydl_opts) as ydl:
                ydl.download(url_list=[url])

        try:
            await asyncio.to_thread(_download)

            # Find the actual downloaded file (in case extension differs)
            pattern: str = f"{base_filename}.*"
            downloaded_files: list[Path] = list(output_dir.glob(pattern=pattern))

            if not downloaded_files:
                raise FileNotFoundError(f"Downloaded file not found: {pattern}")

            downloaded_file: Path = downloaded_files[0]
            logger.info(f"Download complete: {downloaded_file}")

            return downloaded_file

        except Exception as e:
            if "Sign in to confirm you're not a bot" in str(e):
                logger.error(
                    "\n" + "=" * 60 + "\n" +
                    "YOUTUBE BOT DETECTION - DOWNLOAD FAILED!\n" +
                    "=" * 60 + "\n" +
                    "YouTube blocked the download. Please:\n" +
                    "1. Open Firefox, Chrome, Edge, Brave, or Chromium\n" +
                    "2. Sign in to YouTube and verify you can play videos\n" +
                    "3. Close the browser and try again\n" +
                    "=" * 60
                )
            else:
                logger.error(f"Download failed: {e}")
            raise

    async def download_playlist(self,
                                url: str,
                                max_videos: int | None = None) -> list[Path]:
        """
        Download all videos from a YouTube playlist.

        Args:
            url: YouTube playlist URL
            max_videos: Maximum number of videos to download

        Returns:
            List of paths to downloaded videos
        """
        logger.info(f"Fetching playlist info from: {url}")

        opts: dict[str, object] = {
            'quiet': False,
            'extract_flat': True,
            'skip_download': True,
        }

        # Try to add browser cookies
        browser_cookies: tuple[str, ...] | None = self._get_browser_cookies()
        if browser_cookies:
            opts['cookiesfrombrowser'] = browser_cookies

        def _get_playlist_info() -> dict[str, object]:
            with yt_dlp.YoutubeDL(params=opts) as ydl:
                return ydl.extract_info(url=url, download=False)

        try:
            playlist_info: dict[str, object] = await asyncio.to_thread(_get_playlist_info)
            entries: list[dict[str, object]] = playlist_info.get('entries', [])

            if max_videos:
                entries = entries[:max_videos]

            logger.info(f"Found {len(entries)} videos in playlist")

            downloaded_files: list[Path] = []
            for i, entry in enumerate(entries, 1):
                video_url: str = f"https://www.youtube.com/watch?v={entry['id']}"
                logger.info(f"Downloading video {i}/{len(entries)}: {entry.get('title', 'Unknown')}")

                try:
                    playlist_title: str = self._sanitize_filename(title=str(playlist_info.get('title')))
                    file_path: Path = await self.download(
                        url=video_url,
                        subfolder=f"playlists/{playlist_title}"
                    )
                    downloaded_files.append(file_path)
                except Exception as e:
                    if "Sign in to confirm you're not a bot" in str(e):
                        logger.error(
                            f"Failed to download video {i} due to YouTube bot detection. "
                            f"Please ensure you're logged into YouTube in one of these browsers: "
                            f"Firefox, Chrome, Edge, Brave, or Chromium"
                        )
                    else:
                        logger.error(f"Failed to download video {i}: {e}")
                    continue

            return downloaded_files

        except Exception as e:
            if "Sign in to confirm you're not a bot" in str(e):
                logger.error(
                    "\n" + "=" * 60 + "\n" +
                    "YOUTUBE AUTHENTICATION REQUIRED FOR PLAYLIST!\n" +
                    "=" * 60 + "\n" +
                    "Cannot access this playlist without authentication.\n" +
                    "Please sign in to YouTube in Firefox, Chrome, Edge, Brave, or Chromium\n" +
                    "and ensure you have access to this playlist.\n" +
                    "=" * 60
                )
            else:
                logger.error(f"Failed to process playlist: {e}")
            raise


class CachedYouTubeDownloader(YouTubeDownloader):
    """YouTube downloader with caching support."""

    def __init__(self,
                 output_dir: Path | None = None,
                 quality: str = "best",
                 audio_only: bool = False) -> None:
        super().__init__(
            output_dir=output_dir,
            quality=quality,
            audio_only=audio_only,
        )
        self.cache_dir: Path = self.output_dir / '.cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_index: dict[str, Path] = self._load_cache_index()

    def _load_cache_index(self) -> dict[str, Path]:
        """Load cache index mapping URLs to downloaded files."""
        cache_file: Path = self.cache_dir / 'download_index.txt'
        cache: dict[str, Path] = {}

        if cache_file.exists():
            for line in cache_file.read_text().splitlines():
                if '|' in line:
                    url, path = line.split('|', 1)
                    cached_path: Path = Path(path)
                    if cached_path.exists():
                        cache[url] = cached_path

        return cache

    def _save_cache_index(self) -> None:
        """Save cache index to disk."""
        cache_file: Path = self.cache_dir / 'download_index.txt'
        lines: list[str] = [f"{url}|{path}" for url, path in self._cache_index.items()]
        cache_file.write_text(data='\n'.join(lines))

    async def download(self,
                       url: str,
                       custom_filename: str | None = None,
                       subfolder: str | None = None) -> Path:
        """Download with caching support."""
        # Check cache first
        if url in self._cache_index and self._cache_index[url].exists():
            logger.info(f"Using cached file: {self._cache_index[url]}")
            return self._cache_index[url]

        # Download and cache
        file_path: Path = await super().download(
            url=url,
            custom_filename=custom_filename,
            subfolder=subfolder
        )
        self._cache_index[url] = file_path
        self._save_cache_index()

        return file_path

#
# # example_youtube_usage.py
# """Example usage of the YouTube downloader module."""
# import asyncio
# import logging
# from pathlib import Path
# from video_eater.youtube_downloader import YouTubeDownloader, CachedYouTubeDownloader
#
#
# async def basic_download_example() -> None:
#     """Basic video download example."""
#     # Setup logging
#     logging.basicConfig(level=logging.INFO)
#
#     # Create downloader
#     downloader: YouTubeDownloader = YouTubeDownloader(
#         output_dir=Path("downloads"),
#         quality="720",  # Download 720p quality
#         audio_only=False
#     )
#
#     # Download a video
#     url: str = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
#     video_path: Path = await downloader.download(url=url)
#     print(f"Downloaded to: {video_path}")
#
#
# async def audio_only_example() -> None:
#     """Download only audio from YouTube video."""
#     downloader: YouTubeDownloader = YouTubeDownloader(
#         output_dir=Path("audio_downloads"),
#         audio_only=True  # This will extract audio as MP3
#     )
#
#     url: str = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
#     audio_path: Path = await downloader.download(url=url)
#     print(f"Audio extracted to: {audio_path}")
#
#
# async def playlist_example() -> None:
#     """Download multiple videos from a playlist."""
#     downloader: YouTubeDownloader = YouTubeDownloader(
#         output_dir=Path("playlist_downloads"),
#         quality="best"
#     )
#
#     playlist_url: str = "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"
#     video_paths: list[Path] = await downloader.download_playlist(
#         url=playlist_url,
#         max_videos=5  # Download only first 5 videos
#     )
#
#     for path in video_paths:
#         print(f"Downloaded: {path}")
#
#
# async def cached_download_example() -> None:
#     """Use cached downloader to avoid re-downloading."""
#     # This will skip downloading if the video was already downloaded
#     downloader: CachedYouTubeDownloader = CachedYouTubeDownloader(
#         output_dir=Path("cached_downloads")
#     )
#
#     url: str = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
#
#     # First download
#     path1: Path = await downloader.download(url=url)
#     print(f"First download: {path1}")
#
#     # Second call will use cached version
#     path2: Path = await downloader.download(url=url)
#     print(f"Cached file: {path2}")
#     assert path1 == path2
#
#
# async def get_video_info_example() -> None:
#     """Get video metadata without downloading."""
#     downloader: YouTubeDownloader = YouTubeDownloader()
#
#     url: str = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
#     info: dict[str, object] = await downloader.get_video_info(url=url)
#
#     print(f"Title: {info['title']}")
#     print(f"Duration: {info['duration']} seconds")
#     print(f"Uploader: {info['uploader']}")
#     print(f"Views: {info['view_count']:,}")
#
#
# async def batch_processing_example() -> None:
#     """Process multiple URLs from a list."""
#     urls: list[str] = [
#         "https://www.youtube.com/watch?v=VIDEO_ID_1",
#         "https://youtu.be/VIDEO_ID_2",
#         "https://www.youtube.com/watch?v=VIDEO_ID_3",
#         "/local/path/to/video.mp4",  # Can mix local files
#     ]
#
#     downloader: YouTubeDownloader = YouTubeDownloader(output_dir=Path("batch_downloads"))
#
#     for url in urls:
#         if YouTubeDownloader.is_youtube_url(url=url):
#             try:
#                 path: Path = await downloader.download(url=url)
#                 print(f"Downloaded: {path}")
#             except Exception as e:
#                 print(f"Failed to download {url}: {e}")
#         else:
#             print(f"Local file: {url}")
#
#
# async def custom_filename_example() -> None:
#     """Download with custom filename."""
#     downloader: YouTubeDownloader = YouTubeDownloader(output_dir=Path("custom_downloads"))
#
#     url: str = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
#     video_path: Path = await downloader.download(
#         url=url,
#         custom_filename="my_custom_video_name",
#         subfolder="tutorials"
#     )
#     print(f"Downloaded to: {video_path}")
#
#
# # Integration with your existing pipeline
# async def pipeline_integration_example() -> None:
#     """Example of integrating with your video processing pipeline."""
#     from video_eater.core.config_models import VideoProject, ProcessingConfig
#     from video_eater.core.pipeline import VideoProcessingPipeline
#     from video_eater.logging_config import PipelineLogger
#
#     # Setup
#     logger: PipelineLogger = PipelineLogger(log_level="INFO")
#     config: ProcessingConfig = ProcessingConfig()
#     pipeline: VideoProcessingPipeline = VideoProcessingPipeline(
#         config=config,
#         logger=logger
#     )
#     downloader: CachedYouTubeDownloader = CachedYouTubeDownloader(
#         output_dir=Path("pipeline_downloads"),
#         logger=logger
#     )
#
#     # Process YouTube video
#     youtube_url: str = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
#
#     # Download
#     video_path: Path = await downloader.download(url=youtube_url)
#
#     # Process through your existing pipeline
#     project: VideoProject = VideoProject(video_path=video_path)
#     result = await pipeline.process_video(project=project)
#
#     print(result.summary_report())
#
#
# if __name__ == "__main__":
#     # Run one of the examples
#     asyncio.run(main=basic_download_example())