# Copyright (c) Microsoft. All rights reserved.
"""Voice service for VibeVoice TTS integration.

This module provides voice synthesis capabilities using local VibeVoice API.
Supports both local inference (tts_server.py) and remote HTTP API.
"""

import os
import asyncio
import aiohttp
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class VoiceConfig:
    """Configuration for voice synthesis."""
    server_url: str = "http://localhost:8001"  # Local TTS server
    tts_endpoint: str = "/tts"  # Streaming endpoint
    tts_full_endpoint: str = "/tts/full"  # Non-streaming full-text endpoint
    config_endpoint: str = "/config"
    voice_presets: Dict[str, str] = field(default_factory=dict)
    timeout: float = 120.0  # Longer timeout for non-streaming model
    max_retries: int = 2
    use_non_streaming: bool = False  # Use non-streaming mode by default for full-text

    def __post_init__(self):
        if not self.voice_presets:
            self.voice_presets = {
                "小红": "小红",
                "小明": "小明",
                "张三": "张三",
                "李四": "李四",
                "王五": "王五",
                "赵六": "赵六",
                "钱七": "钱七",
                "孙八": "孙八",
                "你": "WHMale",  # Default voice for human player
            }

    def get_tts_url(self) -> str:
        """Get the full TTS endpoint URL."""
        return f"{self.server_url}{self.tts_endpoint}"

    def get_config_url(self) -> str:
        """Get the config endpoint URL."""
        return f"{self.server_url}{self.config_endpoint}"

    def get_tts_full_url(self) -> str:
        """Get the full TTS endpoint URL (non-streaming)."""
        return f"{self.server_url}{self.tts_full_endpoint}"


class VoiceService:
    """Service for converting text to speech using local VibeVoice API."""

    def __init__(self, config: Optional[VoiceConfig] = None):
        """Initialize the voice service.

        Args:
            config: Voice configuration. If None, uses default config.
        """
        self.config = config or VoiceConfig()
        self._cache: Dict[str, bytes] = {}

    async def text_to_speech(
        self, text: str, voice_key: str = "小明", *, debug: bool = False, use_full: bool = True, speaker_index: int = 0
    ) -> Tuple[bytes, Optional[str]]:
        """Convert text to speech audio.

        Args:
            text: Text to convert to speech.
            voice_key: Voice preset key (speaker name).
            debug: Enable debug output.
            use_full: If True, use non-streaming full-text mode (recommended for shorter texts).
                      If False, use streaming mode (for long texts or real-time playback).
            speaker_index: Speaker index for VibeVoice format (Speaker X: text).

        Returns:
            Tuple of (audio_data: bytes, error_message: str or None)
        """
        # Format text for VibeVoice: "Speaker X: text"
        formatted_text = f"Speaker {speaker_index}: {text}"

        # Check cache
        cache_key = f"{voice_key}:{formatted_text}"
        if cache_key in self._cache:
            return self._cache[cache_key], None

        voice_preset = self.config.voice_presets.get(voice_key, voice_key)

        # Choose endpoint based on mode
        if use_full:
            tts_url = self.config.get_tts_full_url()
            if debug:
                print(f"[VoiceService] POST to (non-streaming): {tts_url}")
        else:
            tts_url = self.config.get_tts_url()
            if debug:
                print(f"[VoiceService] POST to (streaming): {tts_url}")

        if debug:
            print(f"[VoiceService] Voice: {voice_preset}, Text: {formatted_text[:50]}...")

        # Use formatted text for VibeVoice
        payload = {
            "text": formatted_text,
            "voice_key": voice_preset,
            "cfg_scale": 1.5
        }

        for attempt in range(self.config.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        tts_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                    ) as resp:
                        if resp.status == 200:
                            audio_data = await resp.read()

                            if len(audio_data) > 0:
                                self._cache[cache_key] = audio_data
                                return audio_data, None
                            else:
                                if debug:
                                    print(f"[VoiceService] Empty response")
                                if attempt < self.config.max_retries:
                                    await asyncio.sleep(0.5)
                                    continue
                                return b"", "Empty response from server"

                        elif resp.status == 503:
                            error_msg = "TTS server not ready. Please start tts_server.py"
                            if debug:
                                print(f"[VoiceService] {error_msg}")
                            return b"", error_msg

                        elif resp.status == 422:
                            error_msg = "Invalid request parameters"
                            try:
                                detail = await resp.text()
                                error_msg = f"{error_msg}: {detail}"
                            except:
                                pass
                            return b"", error_msg

                        else:
                            try:
                                error_msg = await resp.text()
                            except:
                                error_msg = f"HTTP {resp.status}"

                            if debug:
                                print(f"[VoiceService] Error: {error_msg}")

                            if attempt < self.config.max_retries:
                                await asyncio.sleep(0.5)
                                continue
                            return b"", error_msg

            except aiohttp.ClientError as e:
                error_msg = str(e)
                if "Connection refused" in error_msg:
                    error_msg = "Cannot connect to TTS server. Is tts_server.py running?"
                elif "timeout" in error_msg.lower():
                    error_msg = "TTS request timed out"

                if debug:
                    print(f"[VoiceService] Network error: {error_msg}")

                if attempt < self.config.max_retries:
                    await asyncio.sleep(0.5)
                    continue
                return b"", error_msg

            except Exception as e:
                if debug:
                    print(f"[VoiceService] Error: {type(e).__name__}: {e}")
                return b"", str(e)

        return b"", "Max retries exceeded"

    async def text_to_speech_full(
        self, text: str, voice_key: str = "小明", *, debug: bool = False
    ) -> Tuple[bytes, Optional[str]]:
        """Convert full text to speech using non-streaming mode.

        This method processes the entire text at once, which provides
        better quality for shorter texts and more consistent prosody.

        Args:
            text: Text to convert to speech.
            voice_key: Voice preset key (speaker name).
            debug: Enable debug output.

        Returns:
            Tuple of (audio_data: bytes, error_message: str or None)
        """
        return await self.text_to_speech(text, voice_key, debug=debug, use_full=True)

    async def test_connection(self, *, debug: bool = False) -> Tuple[bool, str]:
        """Test if the TTS service is available.

        Args:
            debug: Enable debug output.

        Returns:
            Tuple of (success: bool, message: str)
        """
        config_url = self.config.get_config_url()

        if debug:
            print(f"[VoiceService] Testing connection: {config_url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    config_url,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        voices = data.get("voices", [])
                        if debug:
                            print(f"[VoiceService] Available voices: {voices}")
                        return True, f"Connected! {len(voices)} voices available."
                    elif resp.status == 503:
                        return False, "TTS server not ready. Start tts_server.py first."
                    else:
                        return False, f"HTTP {resp.status}"

        except aiohttp.ClientError as e:
            error_msg = str(e)
            if "Connection refused" in error_msg:
                return False, "Cannot connect. Is tts_server.py running on port 8001?"
            return False, f"Connection error: {error_msg}"

        except Exception as e:
            return False, f"Error: {str(e)}"

    def get_voice_for_player(self, player_name: str) -> str:
        """Get the voice preset key for a player."""
        return self.config.voice_presets.get(player_name, player_name)

    async def refresh_voices(self, *, debug: bool = False) -> bool:
        """Refresh voice presets from the TTS server config.

        Calls the /config endpoint to get available voices and updates
        the voice_presets accordingly.

        Args:
            debug: Enable debug output.

        Returns:
            True if successful, False otherwise.
        """
        config_url = self.config.get_config_url()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    config_url,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        voices = data.get("voices", [])
                        default_voice = data.get("default_voice", "")

                        if voices:
                            # Build voice_presets from available voices
                            # Map player names to available voice names
                            new_presets = {}
                            player_names = ["小红", "小明", "张三", "李四", "王五", "赵六", "钱七", "孙八", "你"]

                            for i, voice in enumerate(voices):
                                if i < len(player_names):
                                    # Map player name to this voice
                                    new_presets[player_names[i]] = voice
                                else:
                                    # Extra voices use themselves as keys
                                    new_presets[voice] = voice

                            # Add "你" (human player) mapping to default voice
                            new_presets["你"] = default_voice or voices[0]

                            self.config.voice_presets = new_presets

                            if debug:
                                print(f"[VoiceService] Updated voice_presets: {new_presets}")
                            return True
                        else:
                            if debug:
                                print(f"[VoiceService] No voices found in config")
                            return False
                    else:
                        if debug:
                            print(f"[VoiceService] Config request failed: HTTP {resp.status}")
                        return False

        except Exception as e:
            if debug:
                print(f"[VoiceService] Error refreshing voices: {e}")
            return False

    def clear_cache(self):
        """Clear the audio cache."""
        self._cache.clear()


# Default voice service instance
_default_service = None


def get_voice_service(config: Optional[VoiceConfig] = None) -> VoiceService:
    """Get the default voice service instance.

    If no config is provided and it's the first call, this will attempt
    to fetch available voices from the TTS server.
    """
    global _default_service, _voices_refreshed

    if _default_service is None or config is not None:
        _default_service = VoiceService(config)
        _voices_refreshed = False  # Reset flag for new service

    return _default_service


# For synchronous usage - create service and refresh voices asynchronously
def create_voice_service_with_voices(
    server_url: str = "http://localhost:8001",
    *,
    debug: bool = False
) -> VoiceService:
    """Create a voice service and refresh voices from the server.

    This is a convenience function for initialization.

    Args:
        server_url: The TTS server URL.
        debug: Enable debug output.

    Returns:
        VoiceService with voices loaded from server.
    """
    import asyncio

    config = VoiceConfig(server_url=server_url)
    service = VoiceService(config)

    # Try to refresh voices (async operation)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If event loop is running, schedule the refresh
            async def async_refresh():
                await service.refresh_voices(debug=debug)
            loop.create_task(async_refresh())
        else:
            # Blocking call for sync context
            loop.run_until_complete(service.refresh_voices(debug=debug))
    except Exception as e:
        if debug:
            print(f"[VoiceService] Could not refresh voices: {e}")

    return service
