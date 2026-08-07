# Source and revision verification

Verified against live authoritative sources on 2026-08-07.

| Resource | Verified revision | Relevant evidence |
|---|---|---|
| `google/fleurs` | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` | `ha_ng`; official train/validation/test; CC BY 4.0; FLEURS documentation says train/development and test speakers are disjoint |
| `McGill-NLP/NaijaS2ST` | `898f51582750fe244693794f22e3f4b32c5baf95` | train 52,000; dev 5,500; no Hub test split; lower-case language values; `user_id`, language-prefixed `text_id`, text, audio, duration; CC BY 4.0 |
| NaijaS2ST Parquet conversion | `fc2d78277d86da24ba1ab5ca55568f39c13cd825` | 115 train and 9 dev shards used for the metadata-only audit |
| `openai/whisper-small` | `973afd24965f72e36ca33b3055d56a652f456b4d` | base zero-shot/direct checkpoint |
| `openai/whisper-tiny` | `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` | public CPU smoke checkpoint |
| `nahomazmach/whisper-small-ha` | `c4e2b47d88ae8b3ee0a605e09863b93aafca72e3` | ASR pipeline; base `openai/whisper-small`; reported Apache-2.0 |
| `facebook/nllb-200-distilled-600M` | `f8d333a098d19b4fd9a8b18f94170487ad3f821d` | translation model; `hau_Latn` → `eng_Latn`; CC BY-NC 4.0 |

Primary links:

- FLEURS dataset and card: https://huggingface.co/datasets/google/fleurs
- FLEURS paper: https://arxiv.org/abs/2205.12446
- NaijaS2ST dataset and card: https://huggingface.co/datasets/McGill-NLP/NaijaS2ST
- NaijaS2ST paper: https://arxiv.org/abs/2604.16287
- Hausa ASR model card: https://huggingface.co/nahomazmach/whisper-small-ha
- Whisper small: https://huggingface.co/openai/whisper-small
- NLLB-200 distilled 600M: https://huggingface.co/facebook/nllb-200-distilled-600M
- Transformers Whisper documentation: https://huggingface.co/docs/transformers/model_doc/whisper
- Transformers Trainer documentation: https://huggingface.co/docs/transformers/main_classes/trainer
- PEFT LoRA documentation: https://huggingface.co/docs/peft/package_reference/lora

## NaijaS2ST alignment semantics

The live rows do not share identical IDs across languages. They share the suffix after a language-specific first character: for example, English `ETE_0001` aligns with Hausa `HTE_0001`. The implementation strips only the prefix matching the declared row language and retains both original IDs. Live values are `english`, `hausa`, `igbo`, and `yoruba`.

The paper describes approximately 50 hours per language. This project reports its own measured accepted-pair hours instead of substituting that approximate publication figure.
