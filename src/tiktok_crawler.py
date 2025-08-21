import asyncio
import random
import aiohttp
import time
import csv
import os
import logging
import re
import json
from typing import List, Dict, Optional, AsyncGenerator, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
from pathlib import Path

try:
    from TikTokApi import TikTokApi
except ImportError:
    print("Please install TikTokApi first: pip install TikTokApi")
    TikTokApi = None

@dataclass
class TikTokVideo:
    """TikTok video data structure"""
    video_id: str
    creator_username: str
    creator_nickname: str
    video_description: str
    tags: List[str]
    hashtags: List[str]
    likes: int
    comments: int
    shares: int
    views: int
    duration: int
    created_time: datetime
    video_url: str
    cover_url: str
    music_title: str
    music_author: str
    download_path: str = ""
    
    # Additional fields for comprehensive data
    raw_data: Dict = None  # Store complete raw API response
    category_type: str = ""
    ad_authorization: bool = False
    hashtag_list: List[Dict] = None
    sound_info: Dict = None
    author_info: Dict = None
    # Advertising flags
    is_ad: bool = False
    is_commerce: bool = False
    # Internal: original API video object for robust downloading
    api_video: Any = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        data = asdict(self)
        data['created_time'] = self.created_time.isoformat() if self.created_time else ""
        return data

class TikTokCrawler:
    """TikTok data crawler core class"""
    
    def __init__(self, ms_token: str,
                 browser: str = "chromium",
                 download_dir: str = "data/tiktok_videos",
                 max_retries: int = 3,
                 headless: bool = True,
                 cookies_file: Optional[str] = None,
                 custom_user_agent: Optional[str] = None,
                 proxy_url: Optional[str] = None,
                 prefer_yt_dlp_first: bool = False):
        """
        Initialize TikTok crawler
        
        Args:
            ms_token: TikTok ms_token (get from browser cookies)
            browser: Browser type (chromium, firefox, webkit)
            download_dir: Video download directory
            max_retries: Maximum retry attempts
        """
        if not TikTokApi:
            raise ImportError("TikTokApi not installed. Please run: pip install TikTokApi")
            
        self.ms_token = ms_token
        self.browser = browser
        self.download_dir = Path(download_dir)
        self.max_retries = max_retries
        self.api = None
        self._api_cm = None  # holds TikTokApi context manager instance
        self.logger = logging.getLogger(__name__)
        self.headless = headless
        self.cookies_file = cookies_file
        self.custom_user_agent = custom_user_agent
        self.proxy_url = proxy_url
        self.session_ready = False
        self.prefer_yt_dlp_first = bool(prefer_yt_dlp_first)
        
        # Create download directory
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Workaround for TikTokApi stealth scripts scope issue where `opts` is not defined
        # in some Playwright versions when scripts are added separately. We replace the
        # default stealth_async to inject a single combined init script so that `opts`
        # is available to all injected snippets.
        try:
            if TikTokApi:
                import TikTokApi.stealth as _stealth  # type: ignore
                _orig_stealth_async = getattr(_stealth, "stealth_async", None)
                _StealthConfig = getattr(_stealth, "StealthConfig", None)
                # Disable the navigator_user_agent stealth override to avoid 'opts' reference errors
                try:
                    if isinstance(getattr(_stealth, 'SCRIPTS', None), dict) and 'navigator_user_agent' in _stealth.SCRIPTS:
                        _stealth.SCRIPTS['navigator_user_agent'] = "/* navigator_user_agent disabled by app workaround */"
                except Exception:
                    pass
                if _orig_stealth_async and _StealthConfig:
                    async def _combined_stealth_async(page, config=None):  # type: ignore
                        cfg = config or _StealthConfig()
                        # Disable navigator_user_agent override to prevent 'opts' reference issues
                        try:
                            setattr(cfg, 'navigator_user_agent', False)
                        except Exception:
                            pass
                        scripts = []
                        for script in cfg.enabled_scripts:
                            scripts.append(str(script))
                        combined = "\n;\n".join(scripts)
                        await page.add_init_script(combined)
                    # Patch both the stealth module and the tiktok module binding
                    _stealth.stealth_async = _combined_stealth_async  # type: ignore
                    try:
                        import TikTokApi.tiktok as _tiktok_mod  # type: ignore
                        setattr(_tiktok_mod, 'stealth_async', _combined_stealth_async)
                    except Exception:
                        pass
        except Exception:
            # Best-effort; if patching fails we proceed with library default
            pass
        
    async def __aenter__(self):
        """Async context manager entry with retries for session creation"""
        # Prefer ms_token from cookies file when available; include both as fallbacks
        cookie_token = self._extract_ms_token_from_cookies(self.cookies_file) if self.cookies_file else ""
        ms_tokens = []
        if self.ms_token and self.ms_token.strip() not in {"YOUR_MS_TOKEN_HERE", ""}:
            ms_tokens.append(self.ms_token.strip())
        if cookie_token:
            # Avoid duplicates
            if cookie_token not in ms_tokens:
                ms_tokens.append(cookie_token)
        if not ms_tokens:
            self.logger.warning("No valid ms_token provided or found in cookies. TikTok sessions may fail to create.")

        # Initialize TikTokApi as async context manager to ensure proper browser lifecycle
        self._api_cm = TikTokApi()
        try:
            self.api = await self._api_cm.__aenter__()
        except Exception:
            # Fallback if __aenter__ not implemented to return distinct object
            self.api = self._api_cm
        backoff = 2
        for attempt in range(1, self.max_retries + 1):
            try:
                # Try preferred settings first, then fallbacks to survive container constraints
                tried = []
                combos = [
                    (self.browser or 'chromium', bool(self.headless)),
                    (self.browser or 'chromium', True),
                    ('chromium', True),
                ]
                last_error = None
                for browser_name, headless_flag in combos:
                    key = (browser_name, headless_flag)
                    if key in tried:
                        continue
                    tried.append(key)
                    try:
                        await self.api.create_sessions(
                            ms_tokens=ms_tokens or [""],
                            num_sessions=1,
                            sleep_after=3,
                            browser=browser_name,
                            headless=headless_flag
                        )
                        sessions_obj = getattr(self.api, "_sessions", None) or getattr(self.api, "sessions", None)
                        if sessions_obj:
                            self.session_ready = True
                            self.logger.info(f"TikTok sessions created using browser={browser_name}, headless={headless_flag}")
                            return self
                        last_error = RuntimeError("create_sessions returned no sessions")
                    except Exception as ie:
                        last_error = ie
                        continue
                # If we got here, all combos failed for this attempt
                raise last_error or RuntimeError("Unable to create sessions")
            except Exception as e:
                self.logger.error(f"Failed to create sessions (attempt {attempt}/{self.max_retries}): {e}")
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 20)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        try:
            if self._api_cm and hasattr(self._api_cm, "__aexit__"):
                await self._api_cm.__aexit__(exc_type, exc_val, exc_tb)
                return
        finally:
            # Fallback close/cleanup if context manager path isn't available
            if self.api:
                if hasattr(self.api, 'close'):
                    try:
                        await self.api.close()
                    except Exception:
                        pass
                else:
                    try:
                        if hasattr(self.api, 'cleanup'):
                            await self.api.cleanup()
                    except Exception:
                        pass
    
    def _extract_tags(self, description: str) -> List[str]:
        """Extract tags from video description"""
        if not description:
            return []
        
        # Extract hashtags and mentions
        tags = []
        
        # Find hashtags
        hashtags = re.findall(r'#(\w+)', description)
        tags.extend(hashtags)
        
        # Find mentions
        mentions = re.findall(r'@(\w+)', description)
        tags.extend(mentions)
        
        # Find other potential tags (words with special characters)
        words = re.findall(r'\b\w+[^\s#@]*\b', description)
        # Filter out common words and keep potential product/brand names
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        potential_tags = [word for word in words if len(word) > 2 and word.lower() not in common_words]
        tags.extend(potential_tags[:5])  # Limit to 5 potential tags
        
        return list(set(tags))  # Remove duplicates
    
    def _extract_hashtags(self, description: str) -> List[str]:
        """Extract only hashtags from video description"""
        if not description:
            return []
        return re.findall(r'#(\w+)', description)
    
    async def get_creator_videos(self, username: str, max_videos: int = 50) -> List[TikTokVideo]:
        """Get videos from specified creator with yt-dlp fallback to avoid EmptyResponse without proxy."""
        # Prefer yt-dlp-based listing when configured
        if self.prefer_yt_dlp_first:
            try:
                ytv = await self._list_videos_with_ytdlp(username, max_videos)
                if ytv:
                    self.logger.info(f"Successfully fetched {len(ytv)} videos for {username} via yt-dlp (preferred)")
                    return ytv
            except Exception as yerr:
                self.logger.warning(f"yt-dlp preferred listing failed for {username}: {yerr}. Falling back to API.")

        for attempt in range(1, 3):
            try:
                self.logger.info(f"Fetching videos for creator: {username}")
                # Gentle jitter to avoid synchronized requests that trigger bot detection
                await asyncio.sleep(0.5 + random.random() * 1.0)
                # Ensure session exists; avoid strict reliance on internal attributes
                if not self.session_ready:
                    # Attempt lazy session creation once here to be resilient
                    cookie_token = self._extract_ms_token_from_cookies(self.cookies_file) if self.cookies_file else ""
                    ms_tokens = []
                    if self.ms_token and self.ms_token.strip() not in {"YOUR_MS_TOKEN_HERE", ""}:
                        ms_tokens.append(self.ms_token.strip())
                    if cookie_token and cookie_token not in ms_tokens:
                        ms_tokens.append(cookie_token)
                    try:
                        # Try requested settings then fallback headless chromium
                        tried = []
                        combos = [
                            (self.browser or 'chromium', bool(self.headless)),
                            (self.browser or 'chromium', True),
                            ('chromium', True),
                        ]
                        last_error = None
                        for browser_name, headless_flag in combos:
                            key = (browser_name, headless_flag)
                            if key in tried:
                                continue
                            tried.append(key)
                            try:
                                await self.api.create_sessions(
                                    ms_tokens=ms_tokens or [""],
                                    num_sessions=1,
                                    sleep_after=3,
                                    browser=browser_name,
                                    headless=headless_flag
                                )
                                sessions_obj = getattr(self.api, "_sessions", None) or getattr(self.api, "sessions", None)
                                if sessions_obj:
                                    self.session_ready = True
                                    break
                            except Exception as ie:
                                last_error = ie
                                continue
                    except Exception as _e:
                        self.logger.error(f"Lazy session creation failed: {_e}")
                    if not self.session_ready:
                        self.logger.warning("Proceeding to fetch without explicit session_ready; API may still work on this TikTokApi version.")
                user = self.api.user(username)
                videos = []
                
                count = 0
                async for video in user.videos(count=max_videos):
                    try:
                        # Get complete raw data
                        raw_data = video.as_dict if hasattr(video, 'as_dict') else {}
                        
                        # Extract basic stats
                        stats = getattr(video, 'stats', {})
                        likes = int(stats.get('diggCount', 0)) if isinstance(stats.get('diggCount'), str) else int(stats.get('diggCount', 0))
                        comments = int(stats.get('commentCount', 0)) if isinstance(stats.get('commentCount'), str) else int(stats.get('commentCount', 0))
                        shares = int(stats.get('shareCount', 0)) if isinstance(stats.get('shareCount'), str) else int(stats.get('shareCount', 0))
                        views = int(stats.get('playCount', 0)) if isinstance(stats.get('playCount'), str) else int(stats.get('playCount', 0))
                        
                        # Extract description from various possible sources
                        description = ""
                        if hasattr(video, 'desc') and video.desc:
                            description = video.desc
                        elif hasattr(video, 'description') and video.description:
                            description = video.description
                        elif hasattr(video, 'title') and video.title:
                            description = video.title
                        elif raw_data and 'desc' in raw_data:
                            description = raw_data['desc']
                        
                        # Extract hashtags
                        hashtags = []
                        if hasattr(video, 'hashtags') and video.hashtags:
                            hashtags = [tag.name for tag in video.hashtags if hasattr(tag, 'name')]
                        elif raw_data and 'hashtags' in raw_data:
                            hashtags = [tag.get('name', '') for tag in raw_data['hashtags'] if isinstance(tag, dict)]
                        
                        # Extract sound/music info
                        sound_info = {}
                        music_title = ""
                        music_author = ""
                        if hasattr(video, 'sound') and video.sound:
                            sound_info = video.sound.as_dict if hasattr(video.sound, 'as_dict') else {}
                            music_title = sound_info.get('title', '')
                            music_author = sound_info.get('author', '')
                        
                        # Extract author info
                        author_info = {}
                        creator_nickname = username
                        if hasattr(video, 'author') and video.author:
                            author_info = video.author.as_dict if hasattr(video.author, 'as_dict') else {}
                            creator_nickname = author_info.get('nickname', username)
                        
                        # Extract URLs
                        video_url, cover_url = self._extract_urls_from_raw(raw_data)
                        
                        video_data = TikTokVideo(
                            video_id=str(video.id),
                            creator_username=username,
                            creator_nickname=creator_nickname,
                            video_description=description,
                            tags=self._extract_tags(description),
                            hashtags=hashtags,
                            likes=likes,
                            comments=comments,
                            shares=shares,
                            views=views,
                            duration=0,  # Will need to find duration from sound_info or other sources
                            created_time=video.create_time if hasattr(video, 'create_time') else datetime.now(),
                            video_url=video_url or "",
                            cover_url=cover_url or "",
                            music_title=music_title,
                            music_author=music_author,
                            raw_data=raw_data,
                            category_type=str(raw_data.get('CategoryType', '')),
                            ad_authorization=raw_data.get('adAuthorization', False),
                            hashtag_list=hashtags,
                            sound_info=sound_info,
                            author_info=author_info,
                            is_ad=bool(raw_data.get('isAd', False)),
                            is_commerce=bool(raw_data.get('isCommerce', False))
                        ,
                        api_video=video
                        )
                        videos.append(video_data)
                        count += 1
                        # Enforce strict limit
                        if count >= max_videos:
                            break
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing video {getattr(video, 'id', 'unknown')}: {e}")
                        continue
                        
                self.logger.info(f"Successfully fetched {len(videos)} videos for {username}")
                if videos:
                    return videos
                # API returned empty list, try yt-dlp as structured fallback
                try:
                    ytv = await self._list_videos_with_ytdlp(username, max_videos)
                    if ytv:
                        self.logger.info(f"Fetched {len(ytv)} videos for {username} via yt-dlp fallback after empty API response")
                        return ytv
                except Exception as y2:
                    self.logger.warning(f"yt-dlp listing fallback failed for {username}: {y2}")
                
            except Exception as e:
                err_text = str(e) if e is not None else ''
                self.logger.error(f"Error fetching videos for {username}: {e}")
                # Try yt-dlp listing immediately on EmptyResponse-type errors
                try:
                    if any(s in err_text for s in ['EmptyResponse', 'empty response']):
                        ytv = await self._list_videos_with_ytdlp(username, max_videos)
                        if ytv:
                            self.logger.info(f"Fetched {len(ytv)} videos for {username} via yt-dlp after API error")
                            return ytv
                except Exception as y3:
                    self.logger.warning(f"yt-dlp listing attempt after API error failed for {username}: {y3}")
                # Retry once on common transient/anti-bot errors
                retryable = any(s in err_text for s in [
                    'EmptyResponse',
                    'empty response',
                    'Target page, context or browser has been closed',
                    'Connection closed while reading from the driver'
                ])
                if attempt < 2 and retryable:
                    # Reset and try to recreate sessions with jitter
                    self.session_ready = False
                    await asyncio.sleep(1.0 + random.random() * 1.5)
                    try:
                        cookie_token = self._extract_ms_token_from_cookies(self.cookies_file) if self.cookies_file else ""
                        ms_tokens = []
                        if self.ms_token and self.ms_token.strip() not in {"YOUR_MS_TOKEN_HERE", ""}:
                            ms_tokens.append(self.ms_token.strip())
                        if cookie_token and cookie_token not in ms_tokens:
                            ms_tokens.append(cookie_token)
                        tried = []
                        combos = [
                            (self.browser or 'chromium', bool(self.headless)),
                            (self.browser or 'chromium', True),
                            ('chromium', True),
                        ]
                        for browser_name, headless_flag in combos:
                            key = (browser_name, headless_flag)
                            if key in tried:
                                continue
                            tried.append(key)
                            try:
                                await self.api.create_sessions(
                                    ms_tokens=ms_tokens or [""],
                                    num_sessions=1,
                                    sleep_after=3,
                                    browser=browser_name,
                                    headless=headless_flag
                                )
                                sessions_obj = getattr(self.api, "_sessions", None) or getattr(self.api, "sessions", None)
                                if sessions_obj:
                                    self.session_ready = True
                                    break
                            except Exception:
                                continue
                        continue  # retry outer loop
                    except Exception:
                        pass
                # Non-retryable or exhausted attempts
                break

        # Final attempt: yt-dlp listing before giving up
        try:
            ytv = await self._list_videos_with_ytdlp(username, max_videos)
            if ytv:
                self.logger.info(f"Fetched {len(ytv)} videos for {username} via final yt-dlp attempt")
                return ytv
        except Exception as yf:
            self.logger.warning(f"Final yt-dlp attempt failed for {username}: {yf}")
        return []
    
    async def download_video(self, video: TikTokVideo, filename: str = None) -> str:
        """Download video to local storage (best-effort)"""
        try:
            if not filename:
                filename = f"{video.creator_username}_{video.video_id}.mp4"
            filepath = self.download_dir / filename

            # Prefer yt-dlp when configured to minimize HTTP/anti-bot issues
            if self.prefer_yt_dlp_first:
                try:
                    ytdlp_path = await self._download_with_ytdlp(video, filepath)
                    if ytdlp_path:
                        return ytdlp_path
                except Exception as yerr:
                    self.logger.warning(f"yt-dlp preferred path failed for {video.video_id}: {yerr}. Falling back to API/HTTP.")

            # Prefer direct API download if available
            if hasattr(video, 'api_video') and video.api_video is not None and hasattr(video.api_video, 'bytes'):
                try:
                    data = await video.api_video.bytes()
                    if data:
                        with open(filepath, 'wb') as f:
                            f.write(data)
                        video.download_path = str(filepath)
                        return str(filepath)
                except Exception as e:
                    self.logger.warning(f"API bytes download failed for {video.video_id}: {e}")

            if not video.video_url:
                self.logger.warning(f"No video URL for video {video.video_id}")
                return ""
            
            # Check if file already exists
            if filepath.exists():
                self.logger.info(f"Video already exists: {filepath}")
                video.download_path = str(filepath)
                return str(filepath)
            
            # Download via HTTP
            self.logger.info(f"Downloading video {video.video_id} to {filepath}")
            headers = {
                "User-Agent": self.custom_user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
                "Referer": "https://www.tiktok.com/",
                "Cookie": self._cookies_header_from_file(self.cookies_file) if self.cookies_file else f"msToken={self.ms_token};"
            }
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(video.video_url, proxy=self.proxy_url) as resp:
                    if resp.status != 200:
                        self.logger.warning(f"Download failed HTTP {resp.status} for video {video.video_id}")
                        # yt-dlp fallback
                        fallback = await self._download_with_ytdlp(video, filepath)
                        return fallback
                    with open(filepath, 'wb') as f:
                        async for chunk in resp.content.iter_chunked(1 << 14):
                            if chunk:
                                f.write(chunk)
            video.download_path = str(filepath)
            return str(filepath)
            
        except Exception as e:
            self.logger.warning(f"HTTP download error for {video.video_id}: {e}. Trying yt-dlp fallback...")
            try:
                fallback = await self._download_with_ytdlp(video, filepath)
                return fallback
            except Exception as ee:
                self.logger.error(f"Error downloading video {video.video_id}: {ee}")
                return ""

    def _extract_urls_from_raw(self, raw: dict):
        """Extract video and cover URLs from raw data using common TikTok fields"""
        if not isinstance(raw, dict):
            return None, None
        # Common nesting: raw['video']['downloadAddr']/['playAddr'] and raw['video']['cover']
        video_block = raw.get('video', {}) if isinstance(raw.get('video', {}), dict) else {}
        candidates = [
            video_block.get('downloadAddr'),
            video_block.get('playAddr'),
            raw.get('downloadAddr'),
            raw.get('playAddr'),
        ]
        video_url = next((u for u in candidates if isinstance(u, str) and u.startswith('http')), None)
        cover_url = None
        for key in ['cover', 'originCover', 'dynamicCover']:
            val = video_block.get(key)
            if isinstance(val, str) and val.startswith('http'):
                cover_url = val
                break
        return video_url, cover_url

    def _extract_ms_token_from_cookies(self, cookies_path: Optional[str]) -> str:
        """Extract msToken from a Netscape-format cookies.txt"""
        try:
            if not cookies_path or not os.path.exists(cookies_path):
                return ""
            with open(cookies_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line or line.startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) >= 7:
                        name = parts[5]
                        value = parts[6]
                        if name == 'msToken' and value:
                            return value
        except Exception:
            pass
        return ""

    def _cookies_header_from_file(self, cookies_path: Optional[str]) -> str:
        """Build Cookie header from a Netscape-format cookies.txt"""
        cookies = []
        try:
            if not cookies_path or not os.path.exists(cookies_path):
                return ""
            with open(cookies_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line or line.startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) >= 7:
                        name = parts[5]
                        value = parts[6]
                        if name and value:
                            cookies.append(f"{name}={value}")
            return '; '.join(cookies)
        except Exception:
            return ''

    async def _list_videos_with_ytdlp(self, username: str, max_videos: int) -> List[TikTokVideo]:
        """List creator videos via yt-dlp without downloading (no proxy required)."""
        try:
            import yt_dlp  # type: ignore
        except Exception:
            self.logger.error("yt-dlp not installed; cannot list videos via fallback.")
            return []

        profile_url = f"https://www.tiktok.com/@{username}"
        ydl_opts = {
            'quiet': True,
            'noprogress': True,
            'skip_download': True,
            'playlistend': int(max(1, max_videos)),
            'http_headers': {
                'User-Agent': self.custom_user_agent or 'Mozilla/5.0',
                'Referer': 'https://www.tiktok.com/'
            },
            'retries': 3,
        }
        if self.cookies_file and os.path.exists(self.cookies_file):
            ydl_opts['cookiefile'] = self.cookies_file

        results: List[TikTokVideo] = []
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(profile_url, download=False)
                entries = []
                if isinstance(info, dict) and 'entries' in info and isinstance(info['entries'], list):
                    entries = info['entries'][:max_videos]
                elif isinstance(info, dict):
                    entries = [info]

                for ent in entries:
                    if not isinstance(ent, dict):
                        continue
                    vid = str(ent.get('id') or ent.get('webpage_url_basename') or '')
                    if not vid:
                        continue
                    desc = ent.get('description') or ent.get('title') or ''
                    ts = ent.get('timestamp')
                    ctime = datetime.fromtimestamp(ts) if isinstance(ts, (int, float)) else datetime.now()
                    duration = int(ent.get('duration') or 0) if isinstance(ent.get('duration'), (int, float)) else 0
                    views = int(ent.get('view_count') or 0) if isinstance(ent.get('view_count'), (int, float)) else 0
                    likes = int(ent.get('like_count') or 0) if isinstance(ent.get('like_count'), (int, float)) else 0
                    comments = int(ent.get('comment_count') or 0) if isinstance(ent.get('comment_count'), (int, float)) else 0
                    shares = int(ent.get('repost_count') or 0) if isinstance(ent.get('repost_count'), (int, float)) else 0
                    vurl = ent.get('webpage_url') or ent.get('url') or ''
                    cover = ent.get('thumbnail') or ''

                    hashtags = self._extract_hashtags(desc)

                    results.append(TikTokVideo(
                        video_id=vid,
                        creator_username=username,
                        creator_nickname=username,
                        video_description=desc,
                        tags=self._extract_tags(desc),
                        hashtags=hashtags if isinstance(hashtags, list) else [],
                        likes=likes,
                        comments=comments,
                        shares=shares,
                        views=views,
                        duration=duration,
                        created_time=ctime,
                        video_url=vurl,
                        cover_url=cover,
                        music_title='',
                        music_author='',
                        raw_data=ent,
                        category_type='',
                        ad_authorization=False,
                        hashtag_list=hashtags if isinstance(hashtags, list) else [],
                        sound_info={},
                        author_info={},
                        is_ad=False,
                        is_commerce=False,
                        api_video=None
                    ))
            return results
        except Exception as e:
            self.logger.warning(f"yt-dlp listing failed for @{username}: {e}")
            return []

    async def _download_with_ytdlp(self, video: TikTokVideo, target_path: Path) -> str:
        """Download using yt-dlp as a robust fallback. Returns local file path or ''."""
        try:
            import yt_dlp  # type: ignore
        except Exception:
            self.logger.error("yt-dlp not installed; cannot use fallback downloader.")
            return ""

        # Ensure directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Use a base template without extension; yt-dlp will choose the best format and extension
        base_tmpl = str(target_path.with_suffix(''))

        ydl_opts = {
            'outtmpl': base_tmpl + '.%(ext)s',
            'quiet': True,
            'nocheckcertificate': True,
            'noprogress': True,
            'retries': 3,
            'concurrent_fragment_downloads': 1,
            'format': 'bv*+ba/best',
            'http_headers': {
                'User-Agent': self.custom_user_agent or 'Mozilla/5.0',
                'Referer': 'https://www.tiktok.com/'
            }
        }

        # Cookies and proxy
        if self.cookies_file and os.path.exists(self.cookies_file):
            ydl_opts['cookiefile'] = self.cookies_file
        if self.proxy_url:
            ydl_opts['proxy'] = self.proxy_url

        url = video.video_url
        if not url:
            self.logger.warning(f"yt-dlp fallback: no URL for video {video.video_id}")
            return ""

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                # Determine downloaded file path
                out_file = None
                if 'requested_downloads' in info and info['requested_downloads']:
                    out_file = info['requested_downloads'][0].get('filepath')
                if not out_file:
                    out_file = info.get('filepath')
                if not out_file:
                    # Try to locate file with base prefix
                    parent = target_path.parent
                    stem = target_path.stem
                    for p in parent.glob(stem + '.*'):
                        if p.is_file():
                            out_file = str(p)
                            break
                if not out_file:
                    self.logger.error(f"yt-dlp fallback: could not determine output file for {video.video_id}")
                    return ""

                # If extension differs, move to target_path name
                try:
                    final_path = str(target_path)
                    if not out_file == final_path:
                        # Prefer keeping real extension; rename target accordingly
                        import os as _os
                        real_ext = _os.path.splitext(out_file)[1] or '.mp4'
                        final_path = str(target_path.with_suffix(real_ext))
                        if _os.path.abspath(out_file) != _os.path.abspath(final_path):
                            _os.replace(out_file, final_path)
                    video.download_path = final_path
                    self.logger.info(f"Downloaded via yt-dlp to {final_path}")
                    return final_path
                except Exception as move_e:
                    self.logger.warning(f"yt-dlp fallback: move/rename issue: {move_e}")
                    video.download_path = out_file
                    return out_file
        except Exception as e:
            self.logger.error(f"yt-dlp fallback failed for {video.video_id}: {e}")
            return ""
    
    async def process_creator_csv(self, csv_path: str, max_videos_per_creator: int = 30) -> List[TikTokVideo]:
        """Process CSV file with creator usernames and fetch their videos"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            if 'username' not in df.columns:
                raise ValueError("CSV must contain 'username' column")
            
            all_videos = []
            creators = df['username'].dropna().unique()
            
            self.logger.info(f"Processing {len(creators)} creators from CSV")
            
            for username in creators:
                try:
                    videos = await self.get_creator_videos(username, max_videos_per_creator)
                    all_videos.extend(videos)
                    
                    # Add delay to avoid rate limiting
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    self.logger.error(f"Error processing creator {username}: {e}")
                    continue
            
            self.logger.info(f"Total videos collected: {len(all_videos)}")
            return all_videos
            
        except Exception as e:
            self.logger.error(f"Error processing CSV file: {e}")
            return []
    
    def save_videos_to_csv(self, videos: List[TikTokVideo], output_path: str):
        """Save video data to CSV file"""
        try:
            if not videos:
                self.logger.warning("No videos to save")
                return
            
            # Convert videos to list of dictionaries
            video_data = [video.to_dict() for video in videos]
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(video_data)
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            self.logger.info(f"Saved {len(videos)} videos to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving videos to CSV: {e}")
    
    def save_videos_to_json(self, videos: List[TikTokVideo], output_path: str):
        """Save video data to JSON file"""
        try:
            if not videos:
                self.logger.warning("No videos to save")
                return
            
            # Convert videos to list of dictionaries
            video_data = [video.to_dict() for video in videos]
            
            # Save to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(video_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved {len(videos)} videos to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving videos to JSON: {e}")

async def main():
    """Example usage of TikTokCrawler"""
    # Get ms_token from environment variable
    ms_token = os.environ.get("TIKTOK_MS_TOKEN")
    
    if not ms_token:
        print("Please set TIKTOK_MS_TOKEN environment variable")
        print("You can get this from your browser cookies on tiktok.com")
        return
    
    # Initialize crawler
    async with TikTokCrawler(ms_token=ms_token) as crawler:
        # Example 1: Get videos from a single creator
        username = "example_user"
        videos = await crawler.get_creator_videos(username, max_videos=10)
        
        if videos:
            # Save to CSV
            crawler.save_videos_to_csv(videos, f"data/{username}_videos.csv")
            crawler.save_videos_to_json(videos, f"data/{username}_videos.json")
        
        # Example 2: Process CSV file with multiple creators
        csv_path = "data/creators.csv"
        if os.path.exists(csv_path):
            all_videos = await crawler.process_creator_csv(csv_path, max_videos_per_creator=20)
            if all_videos:
                crawler.save_videos_to_csv(all_videos, "data/all_creator_videos.csv")
                crawler.save_videos_to_json(all_videos, "data/all_creator_videos.json")

if __name__ == "__main__":
    asyncio.run(main())

