"""Independent Hausa-to-English NLLB text translation stage."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .hardware import select_precision
from .revisions import NLLB_MODEL_ID, NLLB_REVISION

HAUSA_CODE = "hau_Latn"
ENGLISH_CODE = "eng_Latn"


class NLLBTranslator:
    def __init__(
        self,
        model_id: str = NLLB_MODEL_ID,
        *,
        revision: str | None = None,
        source_language: str = HAUSA_CODE,
        target_language: str = ENGLISH_CODE,
        precision: str = "auto",
        max_new_tokens: int = 256,
        num_beams: int = 4,
    ) -> None:
        self.model_id = model_id
        self.revision = (
            NLLB_REVISION if model_id == NLLB_MODEL_ID and revision is None else revision
        )
        self.source_language = source_language
        self.target_language = target_language
        self.precision = precision
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.device = "cpu"
        self.target_token_id: int | None = None

    def load(self) -> NLLBTranslator:
        if self.model is not None:
            return self
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        selected = select_precision(torch, self.precision)
        self.device = selected.device
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[selected.dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, src_lang=self.source_language
        )
        self.target_token_id = self.tokenizer.convert_tokens_to_ids(self.target_language)
        if self.target_token_id is None or self.target_token_id == self.tokenizer.unk_token_id:
            raise ValueError(
                f"Target language code {self.target_language!r} is not in the tokenizer"
            )
        source_id = self.tokenizer.convert_tokens_to_ids(self.source_language)
        if source_id is None or source_id == self.tokenizer.unk_token_id:
            raise ValueError(
                f"Source language code {self.source_language!r} is not in the tokenizer"
            )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, revision=self.revision, dtype=dtype
        ).to(self.device)
        self.model.eval()
        return self

    def translate_batch(self, texts: Iterable[str]) -> list[str]:
        materialized = [str(text).strip() for text in texts]
        if not materialized:
            return []
        if any(not text for text in materialized):
            raise ValueError("NLLB inputs must be nonempty Hausa text")
        self.load()
        import torch

        batch = self.tokenizer(
            materialized, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.inference_mode():
            tokens = self.model.generate(
                **batch,
                forced_bos_token_id=self.target_token_id,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
            )
        return [
            text.strip()
            for text in self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        ]

    def translate(self, text: str) -> str:
        return self.translate_batch([text])[0]
