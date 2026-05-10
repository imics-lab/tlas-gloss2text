"""
Google Gemini API backend (Gemini 2.5 Flash or other Gemini models).

Requires GEMINI_API_KEY in .env.
"""

import asyncio
import logging
import re

from google import genai

from src.backends.base import Direction, TranslationResult, clean_model_output
from src.config import cfg

logger = logging.getLogger(__name__)


class GeminiBackend:

    def __init__(self, api_key: str = None, model: str = None):
        api_key = api_key or cfg.api.gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Check your .env file.")
        self._client = genai.Client(api_key=api_key)
        self.model_name = model or cfg.api.gemini_model

    @property
    def name(self) -> str:
        return f"Gemini ({self.model_name})"

    @property
    def supports_native_readiness(self) -> bool:
        return False

    async def _generate(self, prompt: str) -> str:
        if cfg.api.api_delay > 0:
            await asyncio.sleep(cfg.api.api_delay)
        for attempt in range(cfg.api.max_retries):
            try:
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self._client.models.generate_content(
                            model=self.model_name, contents=prompt
                        ),
                    ),
                    timeout=cfg.api.api_timeout,
                )
                return response.text.strip()
            except Exception as e:
                logger.warning(f"Gemini attempt {attempt + 1} failed: {e}")
                if attempt < cfg.api.max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
        return ""

    def _translate_prompt(self, glosses: str, context: str, direction: Direction) -> str:
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
            "- 0-30: Very incomplete\n- 31-60: Partial\n"
            "- 61-80: Mostly complete\n- 81-100: Fully complete\n\n"
            f"ASL Glosses: {glosses}\n\nRespond with ONLY a number 0-100:"
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
        return await self._generate(self._wait_prompt(glosses))

    async def translate(
        self,
        glosses: str,
        context: str = "",
        direction: Direction = Direction.GLOSS_TO_TEXT,
    ) -> TranslationResult:
        raw = await self._generate(self._translate_prompt(glosses, context, direction))
        return TranslationResult(translation=clean_model_output(raw), raw_response=raw)

    async def get_readiness_score(self, glosses: str) -> float:
        raw = await self._generate(self._readiness_prompt(glosses))
        nums = re.findall(r"\d+(?:\.\d+)?", raw)
        if nums:
            return min(1.0, max(0.0, float(nums[0]) / 100.0))
        return 0.5

    async def close(self):
        pass  # google-genai client has no explicit close
