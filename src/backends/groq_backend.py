"""
Groq API backend (Llama 3.1 8B or other Groq-hosted models).

Requires GROQ_API_KEY in .env.
"""

import asyncio
import logging
import re
from typing import Optional

from groq import AsyncGroq

from src.backends.base import Direction, TranslationResult, clean_model_output
from src.config import cfg

logger = logging.getLogger(__name__)


class GroqBackend:

    def __init__(self, api_key: str = None, model: str = None):
        api_key = api_key or cfg.api.groq_api_key
        if not api_key:
            raise ValueError("GROQ_API_KEY not set. Check your .env file.")
        self.model = model or cfg.api.groq_model
        self._client = AsyncGroq(api_key=api_key)

    @property
    def name(self) -> str:
        return f"Llama3.1 (Groq)"

    @property
    def supports_native_readiness(self) -> bool:
        return False

    async def _chat(self, prompt: str, temperature: float = 0.3) -> str:
        if cfg.api.api_delay > 0:
            await asyncio.sleep(cfg.api.api_delay)
        for attempt in range(cfg.api.max_retries):
            try:
                resp = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=256,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Groq attempt {attempt + 1} failed: {e}")
                if attempt < cfg.api.max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
        return ""

    def _translate_prompt(self, glosses: str, context: str, direction: Direction) -> str:
        if direction == Direction.GLOSS_TO_TEXT:
            ctx = f"\n\nPrevious context: {context}" if context else ""
            return (
                f"Translate this ASL gloss sequence to natural English.{ctx}\n"
                f"Output ONLY the English translation, nothing else.\n\n"
                f"ASL Gloss: {glosses}\n\nEnglish:"
            )
        else:
            ctx = f"\n\nPrevious context: {context}" if context else ""
            return (
                f"Convert this English sentence to ASL gloss notation.{ctx}\n"
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
        return await self._chat(self._wait_prompt(glosses), temperature=0.1)

    async def translate(
        self,
        glosses: str,
        context: str = "",
        direction: Direction = Direction.GLOSS_TO_TEXT,
    ) -> TranslationResult:
        raw = await self._chat(self._translate_prompt(glosses, context, direction))
        return TranslationResult(translation=clean_model_output(raw), raw_response=raw)

    async def get_readiness_score(self, glosses: str) -> float:
        raw = await self._chat(self._readiness_prompt(glosses), temperature=0.1)
        nums = re.findall(r"\d+(?:\.\d+)?", raw)
        if nums:
            return min(1.0, max(0.0, float(nums[0]) / 100.0))
        return 0.5

    async def close(self):
        await self._client.close()
