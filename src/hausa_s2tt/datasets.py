"""Dataset loading, NaijaS2ST pairing, audits, and speaker-safe splitting."""

from __future__ import annotations

import json
import math
import random
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .revisions import (
    FLEURS_DATASET_ID,
    FLEURS_REVISION,
    NAIJA_DATASET_ID,
    NAIJA_REVISION,
)

ARTIFACT_SCHEMA_VERSION = "1.0"
PAIRING_REJECTION_REASONS = (
    "missing_audio_reference",
    "missing_speaker_id",
    "missing_alignment_key",
    "invalid_duration",
    "nonpositive_duration",
    "duration_over_limit",
    "missing_english_target",
    "conflicting_english_targets",
    "duplicate_source_record",
    "invalid_target_record",
)
PAIRING_ARTIFACT_REQUIRED_FIELDS = (
    "artifact_schema_version",
    "audio_locator",
    "source_text",
    "target_text",
    "duration",
    "speaker_id",
    "split",
    "source_language",
    "target_language",
    "source_text_id",
    "target_text_ids",
    "alignment_key",
    "dataset_row_index",
    "source_dataset",
    "dataset_revision",
)


@dataclass
class PairingAudit:
    split: str
    total_rows: int = 0
    source_rows: int = 0
    target_rows: int = 0
    paired_examples: int = 0
    missing_target_ids: list[str] = field(default_factory=list)
    conflicting_target_ids: list[str] = field(default_factory=list)
    invalid_source_ids: list[str] = field(default_factory=list)
    duplicate_source_records: int = 0
    duplicate_source_text_ids: int = 0
    duplicate_target_text_ids: int = 0
    rejected_records: int = 0
    rejection_reasons: dict[str, int] = field(
        default_factory=lambda: {reason: 0 for reason in PAIRING_REJECTION_REASONS}
    )
    invalid_target_records: int = 0
    source_speakers: int = 0
    source_audio_hours: float = 0.0
    duration_min_seconds: float | None = None
    duration_median_seconds: float | None = None
    duration_max_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_language(value: Any) -> str:
    aliases = {
        "h": "hausa",
        "ha": "hausa",
        "hau": "hausa",
        "e": "english",
        "en": "english",
        "eng": "english",
    }
    language = str(value or "").strip().lower()
    return aliases.get(language, language)


def alignment_key(text_id: Any, language: Any) -> str:
    """Return NaijaS2ST's cross-language text key while preserving raw IDs elsewhere.

    The live dataset uses a language prefix on otherwise aligned identifiers, for
    example ``ETE_0001`` (English), ``HTE_0001`` (Hausa), ``ITE_0001`` (Igbo),
    and ``YTE_0001`` (Yoruba). The first character is removed only when it agrees
    with the row language. IDs without a recognized prefix are left unchanged so
    this function also supports synthetic and future shared-ID records.
    """
    raw_id = str(text_id or "").strip()
    expected_prefix = {
        "english": "E",
        "hausa": "H",
        "igbo": "I",
        "yoruba": "Y",
    }.get(canonical_language(language))
    if expected_prefix and raw_id.startswith(expected_prefix) and len(raw_id) > 1:
        return raw_id[1:]
    return raw_id


def _valid_audio_reference(audio: Any) -> bool:
    if isinstance(audio, (str, Path, bytes)):
        return bool(audio)
    if isinstance(audio, Mapping):
        array = audio.get("array")
        if array is not None:
            return int(getattr(array, "size", 1)) > 0
        if audio.get("bytes"):
            return True
        return any(str(audio.get(key) or "").strip() for key in ("path", "src"))
    return audio is not None


def _clean_target(text: Any) -> str:
    return " ".join(str(text or "").split())


def _stable_relative_audio_path(audio: Any) -> str | None:
    """Return a safe relative path, never a local absolute path or remote URL."""
    if isinstance(audio, Mapping):
        value = audio.get("path")
    elif isinstance(audio, (str, Path)):
        value = audio
    else:
        return None
    path = str(value or "").strip().replace("\\", "/")
    parsed = urllib.parse.urlsplit(path)
    if (
        not path
        or parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or Path(path).is_absolute()
        or ".." in Path(path).parts
    ):
        return None
    return path


def _stable_parquet_reference(value: Any) -> str | None:
    """Reduce a Parquet URL/path to a non-secret shard name when possible."""
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return None
    parsed = urllib.parse.urlsplit(text)
    candidate = parsed.path.rsplit("/", 1)[-1] if parsed.scheme else text
    lowered = candidate.casefold()
    if not candidate or any(term in lowered for term in ("token=", "signature=", "expires=")):
        return None
    return candidate


def build_stable_audio_locator(
    row: Mapping[str, Any],
    *,
    dataset_id: str,
    dataset_revision: str,
    split: str,
    row_index: int,
) -> dict[str, Any]:
    """Describe dataset-backed audio without persisting audio data or signed URLs."""
    if not dataset_revision:
        raise ValueError("A nonempty immutable dataset_revision is required")
    locator: dict[str, Any] = {
        "source_dataset": dataset_id,
        "dataset_revision": dataset_revision,
        "split": split,
        "dataset_row_index": int(row_index),
    }
    parquet_file = _stable_parquet_reference(row.get("_parquet_file"))
    if parquet_file:
        locator["dataset_parquet_file"] = parquet_file
    relative_path = _stable_relative_audio_path(row.get("audio"))
    if relative_path:
        locator["relative_audio_path"] = relative_path
    return locator


def validate_pairing_artifact(record: Mapping[str, Any]) -> None:
    """Reject incomplete or unsafe JSONL records before they reach disk."""
    missing = [field for field in PAIRING_ARTIFACT_REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Pairing artifact is missing required fields: {missing}")
    if "audio" in record:
        raise ValueError("Pairing artifacts must use audio_locator, not an audio payload")
    if record.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unexpected pairing artifact schema version")
    if not record.get("dataset_revision"):
        raise ValueError("Pairing artifact dataset_revision must be nonempty")

    def visit(value: Any, path: str) -> None:
        if isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError(f"Raw audio/binary data is forbidden at {path}")
        value_type = type(value)
        if value_type.__module__.startswith("numpy") and value_type.__name__ == "ndarray":
            raise ValueError(f"NumPy arrays are forbidden at {path}")
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).casefold()
                if any(term in lowered for term in ("token", "signature", "authorization")):
                    raise ValueError(f"Sensitive locator field is forbidden at {path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            if len(value) > 1_024:
                raise ValueError(f"Large arrays/sequences are forbidden at {path}")
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, str):
            parsed = urllib.parse.urlsplit(value)
            query_names = {name.casefold() for name, _ in urllib.parse.parse_qsl(parsed.query)}
            if parsed.scheme in {"http", "https"} and (
                parsed.query
                or query_names
                & {"token", "x-amz-signature", "x-amz-credential", "expires", "signature"}
            ):
                raise ValueError(f"Signed or expiring URL is forbidden at {path}")

    visit(record, "record")


def pairing_artifact_record(pair: Mapping[str, Any]) -> dict[str, Any]:
    """Select the durable, versioned representation consumed by later notebooks."""
    fields = list(PAIRING_ARTIFACT_REQUIRED_FIELDS) + [
        "source_text_original",
        "target_text_original",
        "source_text_whitespace_normalized",
        "target_text_whitespace_normalized",
        "dataset_parquet_file",
    ]
    record = {name: deepcopy(pair[name]) for name in fields if name in pair}
    validate_pairing_artifact(record)
    return record


def align_naija_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str,
    source_language: str = "hausa",
    target_language: str = "english",
    max_duration_seconds: float = 30.0,
    dataset_id: str = NAIJA_DATASET_ID,
    dataset_revision: str = NAIJA_REVISION,
) -> tuple[list[dict[str, Any]], PairingAudit]:
    """Pair every valid Hausa recording with one unambiguous English text.

    Alignment removes the language-specific first character from NaijaS2ST IDs
    only when it agrees with the row language. Multiple English recordings for
    one canonical key are allowed only when their reference text agrees. Each
    Hausa recording remains a distinct training example.
    """
    if not dataset_revision:
        raise ValueError("NaijaS2ST pairing requires an immutable dataset_revision")
    materialized = []
    for index, source_row in enumerate(rows):
        row = dict(source_row)
        row.setdefault("_row_idx", index)
        materialized.append(row)
    audit = PairingAudit(split=split, total_rows=len(materialized))
    source_name = canonical_language(source_language)
    target_name = canonical_language(target_language)
    source_rows = [
        row for row in materialized if canonical_language(row.get("language")) == source_name
    ]
    target_rows = [
        row for row in materialized if canonical_language(row.get("language")) == target_name
    ]
    audit.source_rows = len(source_rows)
    audit.target_rows = len(target_rows)

    targets: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    target_id_counts: Counter[str] = Counter()
    for row in target_rows:
        text_id = str(row.get("text_id") or "").strip()
        key = alignment_key(text_id, row.get("language"))
        original_text = str(row.get("text") or "")
        text = _clean_target(original_text)
        if key and text:
            targets[key].add((text_id, text, original_text))
            target_id_counts[key] += 1
        else:
            audit.invalid_target_records += 1
    audit.duplicate_target_text_ids = sum(count > 1 for count in target_id_counts.values())
    audit.conflicting_target_ids = sorted(
        key
        for key, variants in targets.items()
        if len({text for _, text, _ in variants}) > 1
    )

    source_id_counts: Counter[str] = Counter(
        alignment_key(row.get("text_id"), row.get("language")) for row in source_rows
    )
    audit.duplicate_source_text_ids = sum(count > 1 for count in source_id_counts.values())
    seen_records: set[tuple[str, str, str]] = set()
    rejection_reasons: Counter[str] = Counter({reason: 0 for reason in PAIRING_REJECTION_REASONS})
    pairs: list[dict[str, Any]] = []
    durations: list[float] = []
    speakers: set[str] = set()

    for row in source_rows:
        text_id = str(row.get("text_id") or "").strip()
        key = alignment_key(text_id, row.get("language"))
        speaker = str(row.get("user_id") or "").strip()
        record_key = (text_id, speaker, str(row.get("recorded_at") or ""))
        if record_key in seen_records:
            audit.duplicate_source_records += 1
            audit.rejected_records += 1
            rejection_reasons["duplicate_source_record"] += 1
            continue
        seen_records.add(record_key)

        try:
            duration = float(row.get("duration"))
        except (TypeError, ValueError):
            duration = float("nan")
        invalid_reason = None
        if not key:
            invalid_reason = "missing_alignment_key"
        elif not speaker:
            invalid_reason = "missing_speaker_id"
        elif not _valid_audio_reference(row.get("audio")):
            invalid_reason = "missing_audio_reference"
        elif not math.isfinite(duration):
            invalid_reason = "invalid_duration"
        elif duration <= 0:
            invalid_reason = "nonpositive_duration"
        elif duration > max_duration_seconds:
            invalid_reason = "duration_over_limit"
        if invalid_reason:
            audit.invalid_source_ids.append(text_id or "<missing>")
            audit.rejected_records += 1
            rejection_reasons[invalid_reason] += 1
            continue
        variants = targets.get(key, set())
        if not variants:
            audit.missing_target_ids.append(key)
            audit.rejected_records += 1
            rejection_reasons["missing_english_target"] += 1
            continue
        target_texts = {text for _, text, _ in variants}
        if len(target_texts) != 1:
            audit.rejected_records += 1
            rejection_reasons["conflicting_english_targets"] += 1
            continue
        target_text_clean = next(iter(target_texts))
        matching_targets = sorted(
            (target_id, original)
            for target_id, clean, original in variants
            if clean == target_text_clean
        )
        target_ids = sorted({target_id for target_id, _ in matching_targets})
        target_text_original = matching_targets[0][1]
        source_text_original = str(row.get("text") or "")
        row_index = int(row["_row_idx"])
        audio_locator = build_stable_audio_locator(
            row,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            split=split,
            row_index=row_index,
        )
        parquet_file = audio_locator.get("dataset_parquet_file")
        pairs.append(
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "audio_locator": audio_locator,
                "source_text": source_text_original,
                "target_text": target_text_original,
                "source_text_original": source_text_original,
                "target_text_original": target_text_original,
                "source_text_whitespace_normalized": _clean_target(source_text_original),
                "target_text_whitespace_normalized": target_text_clean,
                "duration": duration,
                "speaker_id": speaker,
                "split": split,
                "source_language": source_name,
                "target_language": target_name,
                "text_id": text_id,
                "source_text_id": text_id,
                "target_text_ids": target_ids,
                "alignment_key": key,
                "dataset_row_index": row_index,
                "dataset_parquet_file": parquet_file,
                "source_dataset": dataset_id,
                "dataset_revision": dataset_revision,
            }
        )
        durations.append(duration)
        speakers.add(speaker)

    audit.paired_examples = len(pairs)
    rejection_reasons["invalid_target_record"] = audit.invalid_target_records
    audit.rejection_reasons = {
        reason: rejection_reasons[reason] for reason in PAIRING_REJECTION_REASONS
    }
    audit.source_speakers = len(speakers)
    audit.source_audio_hours = sum(durations) / 3600
    if durations:
        ordered = sorted(durations)
        middle = len(ordered) // 2
        audit.duration_min_seconds = ordered[0]
        audit.duration_median_seconds = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
        audit.duration_max_seconds = ordered[-1]
    audit.missing_target_ids = sorted(set(audit.missing_target_ids))
    audit.invalid_source_ids = sorted(set(audit.invalid_source_ids))
    return pairs, audit


def speaker_sets(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, set[str]]:
    return {
        split: {
            str(row.get("speaker_id") or row.get("user_id") or "").strip()
            for row in rows
            if str(row.get("speaker_id") or row.get("user_id") or "").strip()
        }
        for split, rows in rows_by_split.items()
    }


def speaker_leakage(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[str]]:
    sets = speaker_sets(rows_by_split)
    names = sorted(sets)
    overlaps: dict[str, list[str]] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = sorted(sets[left] & sets[right])
            if shared:
                overlaps[f"{left}__{right}"] = shared
    return overlaps


def assert_no_speaker_leakage(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    overlaps = speaker_leakage(rows_by_split)
    if overlaps:
        summary = ", ".join(f"{pair}={len(ids)}" for pair, ids in overlaps.items())
        raise ValueError(f"Speaker leakage detected: {summary}")


def split_by_speaker(
    rows: Sequence[Mapping[str, Any]],
    *,
    test_fraction: float,
    seed: int,
    train_name: str = "validation",
    test_name: str = "test",
) -> dict[str, list[dict[str, Any]]]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        speaker = str(row.get("speaker_id") or row.get("user_id") or "").strip()
        if not speaker:
            raise ValueError("Every row must have a speaker ID")
        grouped[speaker].append(dict(row))
    speakers = sorted(grouped)
    if len(speakers) < 2:
        raise ValueError("At least two speakers are required for a speaker-safe split")
    random.Random(seed).shuffle(speakers)
    desired_test_rows = max(1, round(len(rows) * test_fraction))
    test_speakers: set[str] = set()
    count = 0
    for speaker in speakers:
        if count >= desired_test_rows and test_speakers:
            break
        test_speakers.add(speaker)
        count += len(grouped[speaker])
    if len(test_speakers) == len(speakers):
        test_speakers.remove(speakers[-1])
    result = {train_name: [], test_name: []}
    for speaker, speaker_rows in grouped.items():
        destination = test_name if speaker in test_speakers else train_name
        for row in speaker_rows:
            row["split"] = destination
            result[destination].append(row)
    assert_no_speaker_leakage(result)
    return result


def load_fleurs_splits(
    *,
    revision: str | None = FLEURS_REVISION,
    decode_audio: bool = False,
    sampling_rate: int = 16_000,
    splits: Sequence[str] | None = None,
) -> Any:
    """Load only the requested official FLEURS splits."""
    try:
        from datasets import Audio, DatasetDict, load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("datasets is required to load FLEURS") from exc
    requested = tuple(splits or ("train", "validation", "test"))
    allowed = {"train", "validation", "test"}
    invalid = set(requested) - allowed
    if invalid:
        raise ValueError(f"Unknown FLEURS splits: {sorted(invalid)}")
    dataset = DatasetDict(
        {
            split: load_dataset(
                FLEURS_DATASET_ID,
                "ha_ng",
                split=split,
                revision=revision,
            )
            for split in requested
        }
    )
    return dataset.cast_column("audio", Audio(sampling_rate=sampling_rate, decode=decode_audio))


def load_naija_split(
    split: str,
    *,
    revision: str | None = NAIJA_REVISION,
    sampling_rate: int = 16_000,
) -> Any:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("datasets is required to load NaijaS2ST") from exc
    dataset = load_dataset(NAIJA_DATASET_ID, "default", split=split, revision=revision)
    return dataset.cast_column("audio", Audio(sampling_rate=sampling_rate, decode=False))


def pair_naija_dataset(
    dataset: Any,
    *,
    split: str,
    max_duration_seconds: float = 30.0,
    dataset_id: str = NAIJA_DATASET_ID,
    dataset_revision: str = NAIJA_REVISION,
) -> tuple[Any, PairingAudit]:
    """Pair a Dataset without materializing its embedded audio bytes in Python."""
    metadata = dataset.remove_columns("audio")
    metadata_rows = (
        {
            **row,
            "audio": {"path": "dataset-backed-audio"},
            "_row_idx": index,
        }
        for index, row in enumerate(metadata)
    )
    pairs, audit = align_naija_rows(
        metadata_rows,
        split=split,
        max_duration_seconds=max_duration_seconds,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
    )
    indices = [int(pair["dataset_row_index"]) for pair in pairs]
    paired = dataset.select(indices)
    additions = {
        "source_text": [pair["source_text"] for pair in pairs],
        "target_text": [pair["target_text"] for pair in pairs],
        "speaker_id": [pair["speaker_id"] for pair in pairs],
        "source_language": [pair["source_language"] for pair in pairs],
        "target_language": [pair["target_language"] for pair in pairs],
        "source_text_id": [pair["source_text_id"] for pair in pairs],
        "target_text_ids": [pair["target_text_ids"] for pair in pairs],
        "alignment_key": [pair["alignment_key"] for pair in pairs],
        "dataset_row_index": indices,
        "audio_locator": [pair["audio_locator"] for pair in pairs],
        "artifact_schema_version": [ARTIFACT_SCHEMA_VERSION] * len(pairs),
        "source_dataset": [dataset_id] * len(pairs),
        "dataset_revision": [dataset_revision] * len(pairs),
    }
    for name, values in additions.items():
        paired = paired.add_column(name, values)
    return paired, audit


def split_dataset_by_speaker(
    dataset: Any,
    *,
    test_fraction: float,
    seed: int,
    train_name: str = "train",
    test_name: str = "validation",
) -> dict[str, Any]:
    """Select speaker-disjoint Dataset views without reading the audio column."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, speaker_value in enumerate(dataset["speaker_id"]):
        speaker = str(speaker_value or "").strip()
        if not speaker:
            raise ValueError("Every row must have a speaker ID")
        grouped[speaker].append(index)
    speakers = sorted(grouped)
    if len(speakers) < 2:
        raise ValueError("At least two speakers are required for a speaker-safe split")
    random.Random(seed).shuffle(speakers)
    desired_test_rows = max(1, round(len(dataset) * test_fraction))
    test_speakers: set[str] = set()
    count = 0
    for speaker in speakers:
        if count >= desired_test_rows and test_speakers:
            break
        test_speakers.add(speaker)
        count += len(grouped[speaker])
    if len(test_speakers) == len(speakers):
        test_speakers.remove(speakers[-1])
    train_indices = [
        index
        for speaker, indices in grouped.items()
        if speaker not in test_speakers
        for index in indices
    ]
    test_indices = [
        index
        for speaker, indices in grouped.items()
        if speaker in test_speakers
        for index in indices
    ]
    result = {
        train_name: dataset.select(train_indices),
        test_name: dataset.select(test_indices),
    }
    if set(result[train_name]["speaker_id"]) & set(result[test_name]["speaker_id"]):
        raise AssertionError("Speaker-disjoint Dataset selection failed")
    return result


def iter_dataset_viewer_rows(
    split: str,
    *,
    dataset_id: str = NAIJA_DATASET_ID,
    config: str = "default",
    page_size: int = 100,
    workers: int = 1,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Fetch metadata through the Dataset Viewer without downloading 70 GB of audio."""
    if not 1 <= page_size <= 100:
        raise ValueError("Dataset Viewer page_size must be in [1, 100]")
    if workers <= 0:
        raise ValueError("workers must be positive")

    def fetch_page(offset: int) -> tuple[int, list[dict[str, Any]], int]:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset_id,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page_size,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 and exc.code < 500:
                    raise
                if attempt == 5:
                    raise
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(2**attempt, 30)
                time.sleep(delay)
        rows: list[dict[str, Any]] = []
        for item in payload.get("rows", []):
            row = dict(item.get("row", item))
            row["_row_idx"] = item.get("row_idx")
            rows.append(row)
        return offset, rows, int(payload.get("num_rows_total", offset + len(rows)))

    _, first_rows, total = fetch_page(0)
    if limit is not None:
        first_rows = first_rows[:limit]
    for row in first_rows:
        yield row
    requested_total = min(total, limit) if limit is not None else total
    offsets = list(range(page_size, requested_total, page_size))
    if workers == 1:
        pages = [fetch_page(offset) for offset in offsets]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pages = list(pool.map(fetch_page, offsets))
    yielded = len(first_rows)
    for _, rows, _ in sorted(pages, key=lambda page: page[0]):
        for row in rows:
            if limit is not None and yielded >= limit:
                return
            yield row
            yielded += 1


def iter_dataset_viewer_filtered_rows(
    split: str,
    *,
    where: str,
    dataset_id: str = NAIJA_DATASET_ID,
    config: str = "default",
    orderby: str | None = None,
    limit: int = 10,
) -> Iterator[dict[str, Any]]:
    """Fetch a small filtered Viewer slice for runtime inspection, not provenance.

    Dataset Viewer audio fields can contain temporary signed URLs. Callers may
    decode those URLs during the current process, but accepted artifact records
    must be built through :func:`align_naija_rows`, which replaces them with
    revision-pinned dataset coordinates.
    """
    if not where.strip():
        raise ValueError("Dataset Viewer filter requires a nonempty where clause")
    if not 1 <= limit <= 100:
        raise ValueError("Dataset Viewer filtered limit must be in [1, 100]")
    parameters: dict[str, Any] = {
        "dataset": dataset_id,
        "config": config,
        "split": split,
        "where": where,
        "offset": 0,
        "length": limit,
    }
    if orderby:
        parameters["orderby"] = orderby
    url = "https://datasets-server.huggingface.co/filter?" + urllib.parse.urlencode(
        parameters
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise
            if attempt == 5:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(2**attempt, 30)
            time.sleep(delay)
        except (TimeoutError, urllib.error.URLError):
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 30))
    for item in payload.get("rows", []):
        row = dict(item.get("row", item))
        row["_row_idx"] = item.get("row_idx")
        yield row


def iter_dataset_parquet_metadata(
    split: str,
    *,
    dataset_id: str = NAIJA_DATASET_ID,
    config: str = "default",
    workers: int = 4,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Read all non-audio metadata using HTTP range reads over Viewer Parquet.

    Only the small ``audio.path`` child is read from the audio struct. Audio
    bytes are deliberately excluded, so a complete pairing/leakage audit does
    not download the approximately 70 GB corpus.
    """
    query = urllib.parse.urlencode({"dataset": dataset_id})
    with urllib.request.urlopen(
        f"https://datasets-server.huggingface.co/parquet?{query}", timeout=60
    ) as response:
        payload = json.load(response)
    files = sorted(
        (
            item
            for item in payload.get("parquet_files", [])
            if item.get("config") == config and item.get("split") == split
        ),
        key=lambda item: item.get("filename", item.get("url", "")),
    )
    if not files:
        raise RuntimeError(f"No Parquet files found for {dataset_id}/{config}/{split}")
    if workers < 1:
        raise ValueError("workers must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    try:
        import fsspec
        from pyarrow import parquet
    except ImportError as exc:  # pragma: no cover - datasets installs both
        raise RuntimeError("fsspec and pyarrow are required for metadata audit") from exc
    columns = [
        "audio.path",
        "user_id",
        "language",
        "text_id",
        "text",
        "duration",
        "recorded_at",
        "original_sample_rate",
        "silence_ratio",
        "snr_db",
        "speech_rate",
        "volume_db",
        "split",
    ]
    def read_file(item: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        url = str(item["url"])
        filename = str(item.get("filename") or url.rsplit("/", 1)[-1])
        with fsspec.open(url, "rb", block_size=1 << 20, cache_type="readahead") as handle:
            table = parquet.ParquetFile(handle).read(columns=columns)
        return filename, table.to_pylist()

    effective_workers = 1 if limit is not None else workers
    if effective_workers == 1:
        file_rows = map(read_file, files)
    else:
        pool = ThreadPoolExecutor(max_workers=effective_workers)
        file_rows = pool.map(read_file, files)
    row_index = 0
    try:
        for filename, rows in file_rows:
            for row in rows:
                if limit is not None and row_index >= limit:
                    return
                audio = row.get("audio")
                audio_path = audio.get("path") if isinstance(audio, Mapping) else None
                row["audio"] = {"path": audio_path} if audio_path else None
                row["_row_idx"] = row_index
                row["_parquet_file"] = filename
                row_index += 1
                yield row
    finally:
        if effective_workers != 1:
            pool.shutdown(wait=True)


def write_pairing_artifacts(
    pairs: Sequence[Mapping[str, Any]], audit: PairingAudit, output_dir: str | Path
) -> dict[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / f"{audit.split}.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            serializable = pairing_artifact_record(pair)
            handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")
    audit_path = directory / f"{audit.split}_audit.json"
    audit_path.write_text(
        json.dumps(audit.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"examples": str(metadata_path), "audit": str(audit_path)}


def load_revision_matched_audit(
    path: str | Path,
    *,
    expected_dataset_id: str,
    expected_dataset_revision: str,
) -> dict[str, Any]:
    """Load a tracked audit only when its resource identity matches the code."""
    audit_path = Path(path)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    actual_id = payload.get("dataset")
    actual_revision = payload.get("dataset_revision")
    if actual_id != expected_dataset_id or actual_revision != expected_dataset_revision:
        raise ValueError(
            f"Tracked audit provenance mismatch in {audit_path}: expected "
            f"{expected_dataset_id}@{expected_dataset_revision}, got "
            f"{actual_id}@{actual_revision}"
        )
    return payload


def current_git_commit(repository_root: str | Path = ".") -> str:
    """Return the exact checkout commit for a generated manifest."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repository_root),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_pairing_manifest(
    audits: Sequence[PairingAudit],
    *,
    git_commit: str,
    dataset_id: str = NAIJA_DATASET_ID,
    dataset_revision: str = NAIJA_REVISION,
    split_policy: str,
    seed: int,
    max_duration_seconds: float,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the small metadata contract accompanying ignored JSONL artifacts."""
    if not git_commit:
        raise ValueError("git_commit must be recorded in the data manifest")
    if not dataset_revision:
        raise ValueError("dataset_revision must be recorded in the data manifest")
    split_counts = {
        audit.split: {
            "accepted": audit.paired_examples,
            "rejected": audit.rejected_records,
            "rejection_reasons": audit.rejection_reasons,
        }
        for audit in audits
    }
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "split_policy": split_policy,
        "seed": seed,
        "max_duration_seconds": max_duration_seconds,
        "splits": split_counts,
    }


def write_pairing_manifest(manifest: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Validate and write manifest metadata next to ignored pairing JSONL files."""
    required = {
        "artifact_schema_version",
        "generated_at",
        "git_commit",
        "dataset_id",
        "dataset_revision",
        "split_policy",
        "seed",
        "max_duration_seconds",
        "splits",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Data manifest is missing required fields: {missing}")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    path.write_text(json.dumps(dict(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
