# Optional Hausa→English human-evaluation rubric

No human evaluation has been conducted. This rubric is a protocol for future fluent Hausa/English reviewers.

## Sampling

Blind and randomize system names. Use the same held-out examples for every system. Stratify for short/long audio, clean/noisy conditions, numbers/dates, names/locations, dialect or code-switching where known, and ASR disagreement. Do not let reviewers score their own recordings.

## Per-example form

Record anonymized example ID, reviewer ID, system-hidden output, and:

- Meaning preservation (1–5): 1 = unrelated/contradictory, 3 = main idea with material errors, 5 = complete and faithful.
- English fluency (1–5): 1 = unusable, 3 = understandable with problems, 5 = natural and grammatical.
- Names and numbers (1–5): 1 = materially wrong, 3 = partial, 5 = all preserved; use N/A when absent.
- Omission severity: none / minor / major.
- Added or hallucinated information: yes / no.
- Negation or polarity error: yes / no.
- Unsafe if used without correction: yes / no.
- Short free-text explanation.

## Reporting

Use at least two reviewers where feasible. Report sample size, reviewer qualifications, agreement, score distributions, and adjudication rules. Keep automatic and human results separate. Never fill this report with model-generated reviewer scores.
