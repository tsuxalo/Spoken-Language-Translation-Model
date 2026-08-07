# Ethics, limitations, and intended use

## Intended use

This code supports reproducible research, graduate instruction, error analysis, and carefully supervised prototyping for Hausa ASR and Hausa→English speech translation.

It should not be used as the sole source for medical care, legal advice or evidence, immigration decisions, policing, benefits, hiring, education access, emergency response, or other consequential decisions. Human review by fluent speakers is required wherever an error can harm a person.

## Known risks

- “Hausa” is not homogeneous. The data emphasizes Nigerian recordings and does not establish coverage across regional varieties, dialects, code-switching patterns, or diaspora speech.
- The measured datasets contain speaker, accent, gender, topic, and recording-condition imbalances. FLEURS `ha_ng` metadata notably has an uneven recorded gender distribution by split; gender labels and their collection process were not independently validated.
- Direct models can hallucinate fluent English. Cascades expose both ASR propagation and MT errors. A fluent output is not evidence that meaning was preserved.
- Names, numbers, dates, negation, dosage, locations, and culturally specific expressions deserve targeted review.
- Automatic BLEU/chrF/WER/CER do not fully measure adequacy, fairness, pragmatics, or real-world harm.
- The complete NaijaS2ST audit checked paths and metadata but did not decode all 69.9 GB of audio. Full corruption and acoustic-quality distributions remain unmeasured.
- Neither full direct training nor held-out comparison nor human evaluation has been run in this repository state.

## Privacy, consent, and licensing

Users must review original dataset documentation for collection, consent, attribution, and removal procedures. Do not infer sensitive traits or identify speakers. Do not redistribute audio or predictions containing personal data without authority.

FLEURS and NaijaS2ST are CC BY 4.0. The NLLB cascade includes a CC BY-NC 4.0 component and is not represented as commercially cleared. The published Hausa ASR model card reports Apache-2.0. A future checkpoint card must document every inherited obligation and the exact training data.

## Responsible release checklist

- Freeze and publish the experimental protocol before final evaluation.
- Run dialect/region, accent, gender, duration, noise, names, and numbers error slices.
- Obtain fluent Hausa/English human review using the supplied rubric.
- Document failures prominently and include a reporting/removal contact.
- Confirm licenses and intended deployment jurisdiction with qualified reviewers.
- Never publish weights, data, or external resources automatically.
