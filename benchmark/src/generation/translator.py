"""Translator for QA pairs (Chinese to English).

Uses SiliconFlow API to translate question_zh, answer_zh, and choices.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List

from openai import OpenAI

from src.schemas import QAPair, QADataset

logger = logging.getLogger(__name__)


class Translator:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "Qwen/Qwen2.5-72B-Instruct",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        batch_size: int = 10,
    ):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size

    def _translate_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []

        items_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        prompt = f"""Translate the following Chinese texts to English. Keep the numbering format.
Output only the translated texts, one per line, with the same numbering.

{items_text}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional translator. Translate Chinese to English accurately."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw = response.choices[0].message.content.strip()
            translations = []
            for line in raw.split("\n"):
                line = line.strip()
                if line and ". " in line:
                    text = line.split(". ", 1)[1]
                    translations.append(text)
                elif line:
                    translations.append(line)
            return translations[:len(texts)]
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return [""] * len(texts)

    def translate_qa_pair(self, qa: QAPair) -> QAPair:
        """Translate a single QA pair's Chinese fields to English."""
        texts_to_translate = [qa.question_zh, qa.answer_zh]
        if qa.choices:
            texts_to_translate.extend(qa.choices)

        translations = self._translate_batch(texts_to_translate)

        qa.question_en = translations[0] if len(translations) > 0 else ""
        qa.answer_en = translations[1] if len(translations) > 1 else ""

        if qa.choices and len(translations) > 2:
            qa.choices_en = translations[2:2 + len(qa.choices)]

        return qa

    def translate_dataset(self, dataset: QADataset) -> QADataset:
        """Translate all QA pairs in a dataset."""
        for i, qa in enumerate(dataset.qa_pairs):
            if i % self.batch_size == 0:
                logger.info(f"Translating {i+1}/{len(dataset.qa_pairs)}...")
            try:
                self.translate_qa_pair(qa)
            except Exception as e:
                logger.warning(f"Failed to translate {qa.qa_id}: {e}")

        dataset.language = "both"
        return dataset
