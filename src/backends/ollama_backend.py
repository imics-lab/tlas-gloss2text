"""
Local Ollama backend (GPT-OSS 32K or any Ollama-served model).

Configure OLLAMA_URL and OLLAMA_MODEL in your .env file.
"""

import asyncio
import logging
import re

import aiohttp

from src.backends.base import Direction, TranslationBackend, TranslationResult, clean_model_output
from src.config import cfg

logger = logging.getLogger(__name__)


class OllamaBackend:

    def __init__(
        self,
        url: str = None,
        model: str = None,
        api_delay: float = None,
        timeout: int = None,
    ):
        self.url   = url   or cfg.api.ollama_url
        self.model = model or cfg.api.ollama_model
        self.api_delay = api_delay if api_delay is not None else cfg.api.api_delay
        self.timeout = timeout or cfg.api.api_timeout
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return f"GPT-OSS ({self.model.split(':')[0]})"

    @property
    def supports_native_readiness(self) -> bool:
        return False

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def _call(self, prompt: str) -> str:
        if self.api_delay > 0:
            await asyncio.sleep(self.api_delay)
        session = await self._get_session()
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        for attempt in range(cfg.api.max_retries):
            try:
                async with session.post(self.url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        return data.get("response", "").strip()
            except Exception as e:
                logger.warning(f"Ollama attempt {attempt + 1} failed: {e}")
                if attempt < cfg.api.max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
        return ""

    def _translate_prompt(
        self, glosses: str, context: str, direction: Direction
    ) -> str:
        if direction == Direction.GLOSS_TO_TEXT:
            ctx = f"\n\nPrior context (for coherence only — do NOT repeat or copy): {context}" if context else ""
            return (
                f"Translate ONLY the following ASL gloss sequence to natural English.{ctx}\n"
                f"Output ONLY the English translation of the given glosses, nothing else.\n\n"
                f"ASL Gloss: {glosses}\n\nEnglish:"
            )
        else:
            ctx = f"\n\nPrior context (for coherence only — do NOT repeat or copy): {context}" if context else ""
            return (
                f"Convert ONLY the following English sentence to ASL gloss notation.{ctx}\n"
                f"Output ONLY the ASL gloss sequence in UPPERCASE, nothing else.\n\n"
                f"English: {glosses}\n\nASL Gloss:"
            )

    def _readiness_prompt(self, glosses: str) -> str:
        return (
            "Rate from 0 to 100 how COMPLETE and ready for translation these ASL glosses are:\n"
            "- 0-30: Very incomplete, need many more glosses\n"
            "- 31-60: Partially complete, missing important context\n"
            "- 61-80: Mostly complete, can translate reasonably\n"
            "- 81-100: Fully complete, ready for accurate translation\n\n"
            f"ASL Glosses: {glosses}\n\n"
            "Respond with ONLY a number between 0 and 100:"
        )

    def _wait_prompt(self, glosses: str) -> str:
        return (
            "You are a simultaneous ASL-to-English interpreter.\n"
            "Given the accumulated ASL glosses below, decide:\n"
            "- If the meaning is complete, output ONLY the English translation.\n"
            "- If you need more signs to understand the full meaning, output exactly: <WAIT>\n\n"
            f"ASL glosses: {glosses}\n\nOutput:"
        )

    async def wait_or_translate(self, glosses: str) -> str:
        return await self._call(self._wait_prompt(glosses))

    async def translate(
        self,
        glosses: str,
        context: str = "",
        direction: Direction = Direction.GLOSS_TO_TEXT,
    ) -> TranslationResult:
        prompt = self._translate_prompt(glosses, context, direction)
        raw = await self._call(prompt)
        return TranslationResult(translation=clean_model_output(raw), raw_response=raw)

    async def get_readiness_score(self, glosses: str) -> float:
        raw = await self._call(self._readiness_prompt(glosses))
        nums = re.findall(r"\d+(?:\.\d+)?", raw)
        if nums:
            return min(1.0, max(0.0, float(nums[0]) / 100.0))
        return 0.5

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
