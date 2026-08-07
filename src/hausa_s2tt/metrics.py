"""Traceable ASR and speech-translation metrics."""

from __future__ import annotations

import importlib.metadata
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})


def normalize_hausa(text: str) -> str:
    """Normalize Hausa for diagnostic WER/CER without removing diacritics.

    The transform applies Unicode NFKC, lowercasing, apostrophe unification,
    punctuation/symbol removal, and whitespace collapse. Hausa letters such as
    ɓ, ɗ, and ƙ and all decimal digits are preserved.
    """
    normalized = unicodedata.normalize("NFKC", str(text)).translate(_APOSTROPHES).lower()
    chars: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if char == "'" or category[0] in {"L", "N", "M"}:
            chars.append(char)
        elif char.isspace() or category[0] in {"P", "S", "Z"}:
            chars.append(" ")
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (ref_item != hyp_item)
            deletion = previous[j] + 1
            insertion = current[j - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class ErrorRate:
    errors: int
    reference_units: int

    @property
    def rate(self) -> float:
        if self.reference_units == 0:
            return 0.0 if self.errors == 0 else float("inf")
        return self.errors / self.reference_units


def corpus_error_rate(
    references: Iterable[str],
    predictions: Iterable[str],
    *,
    unit: str,
    normalizer: Callable[[str], str] | None = None,
) -> ErrorRate:
    if unit not in {"word", "character"}:
        raise ValueError("unit must be word or character")
    refs = list(references)
    preds = list(predictions)
    if len(refs) != len(preds):
        raise ValueError("references and predictions must have the same length")
    errors = 0
    reference_units = 0
    for reference, prediction in zip(refs, preds):
        ref = normalizer(reference) if normalizer else str(reference)
        pred = normalizer(prediction) if normalizer else str(prediction)
        ref_units = ref.split() if unit == "word" else list(ref)
        pred_units = pred.split() if unit == "word" else list(pred)
        errors += _edit_distance(ref_units, pred_units)
        reference_units += len(ref_units)
    return ErrorRate(errors, reference_units)


def compute_asr_metrics(references: Iterable[str], predictions: Iterable[str]) -> dict[str, float | int | str]:
    refs = list(references)
    preds = list(predictions)
    raw_wer = corpus_error_rate(refs, preds, unit="word")
    normalized_wer = corpus_error_rate(refs, preds, unit="word", normalizer=normalize_hausa)
    raw_cer = corpus_error_rate(refs, preds, unit="character")
    normalized_cer = corpus_error_rate(
        refs, preds, unit="character", normalizer=normalize_hausa
    )
    return {
        "raw_wer": raw_wer.rate,
        "normalized_wer": normalized_wer.rate,
        "raw_cer": raw_cer.rate,
        "normalized_cer": normalized_cer.rate,
        "raw_word_errors": raw_wer.errors,
        "raw_reference_words": raw_wer.reference_units,
        "normalizer": "hausa_nfkc_lower_punctuation_v1",
    }


def compute_translation_metrics(
    references: Iterable[str], predictions: Iterable[str]
) -> dict[str, float | str]:
    refs = [str(value) for value in references]
    preds = [str(value) for value in predictions]
    if len(refs) != len(preds):
        raise ValueError("references and predictions must have the same length")
    try:
        import sacrebleu
    except ImportError as exc:  # pragma: no cover - dependency message
        raise RuntimeError("sacrebleu is required for BLEU and chrF++") from exc
    bleu_metric = sacrebleu.metrics.BLEU(tokenize="13a")
    chrf_metric = sacrebleu.metrics.CHRF(char_order=6, word_order=2, beta=2)
    bleu = bleu_metric.corpus_score(preds, [refs])
    chrf = chrf_metric.corpus_score(preds, [refs])
    try:
        version = importlib.metadata.version("sacrebleu")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        version = "unknown"
    return {
        "sacrebleu": float(bleu.score),
        "chrf_pp": float(chrf.score),
        "sacrebleu_signature": str(bleu_metric.get_signature()),
        "chrf_signature": str(chrf_metric.get_signature()),
        "sacrebleu_version": version,
    }
