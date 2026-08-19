"""Immutable Hugging Face revisions for reproducible experiment reruns.

The completed 2026 GPU run captured these revisions after execution.  Future
repository commands enforce them explicitly; historical enforcement status is
recorded separately in the public provenance artifacts.
"""

NAIJAS2ST_ID = "McGill-NLP/NaijaS2ST"
NAIJAS2ST_REVISION = "898f51582750fe244693794f22e3f4b32c5baf95"

WHISPER_HAUSA_ID = "nahomazmach/whisper-small-ha"
WHISPER_HAUSA_REVISION = "c4e2b47d88ae8b3ee0a605e09863b93aafca72e3"

NLLB_600M_ID = "facebook/nllb-200-distilled-600M"
NLLB_600M_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"

AFRINLLB_ID = "AfriNLP/AfriNLLB-12enc-12dec-full-ft"
AFRINLLB_REVISION = "53b1bf8d09454d092a474a8e78d5c95a32b53154"

NLLB_3_3B_ID = "facebook/nllb-200-3.3B"
NLLB_3_3B_REVISION = "1a07f7d195896b2114afcb79b7b57ab512e7b43e"

SSA_COMET_ID = "McGill-NLP/ssa-comet-mtl"
SSA_COMET_REVISION = "6e64e0a56ce69524c67f304b092725687a362ef8"

OPENAI_WHISPER_SMALL_ID = "openai/whisper-small"
OPENAI_WHISPER_SMALL_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"

FLEURS_ID = "google/fleurs"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"

DIRECT_PILOT_ADAPTER_ID = "nahomazmach/whisper-small-ha-en-direct-pilot"
DIRECT_PILOT_ADAPTER_REVISION = "a91a4a1155c574a24226de53f053e08b6446806d"

C1_MODEL_ID = "lEtoileNoir/Hausa_English_Direct_S2TT"
C1_MODEL_REVISION = "cd84a6c2e447b098d772d6ad59b247f16c29075d"

MODEL_REVISIONS = {
    NAIJAS2ST_ID: NAIJAS2ST_REVISION,
    NLLB_600M_ID: NLLB_600M_REVISION,
    AFRINLLB_ID: AFRINLLB_REVISION,
    NLLB_3_3B_ID: NLLB_3_3B_REVISION,
    WHISPER_HAUSA_ID: WHISPER_HAUSA_REVISION,
    SSA_COMET_ID: SSA_COMET_REVISION,
    OPENAI_WHISPER_SMALL_ID: OPENAI_WHISPER_SMALL_REVISION,
    FLEURS_ID: FLEURS_REVISION,
    DIRECT_PILOT_ADAPTER_ID: DIRECT_PILOT_ADAPTER_REVISION,
    C1_MODEL_ID: C1_MODEL_REVISION,
}


def revision_for(model_id: str) -> str:
    """Return the immutable default for a known Hub model."""

    try:
        return MODEL_REVISIONS[model_id]
    except KeyError as exc:
        raise KeyError(
            f"No immutable revision is registered for Hub model {model_id!r}. "
            "Pass an explicit revision for custom models."
        ) from exc
