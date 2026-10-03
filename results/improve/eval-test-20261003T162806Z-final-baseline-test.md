# Eval: split `test` (n=15)

model `claude-opus-5-5` (effort medium), prompt `3e8b2cdaa6e9` (0 lessons), git `ca4ca1a`, 2026-10-03T16:28:06+00:00

## Per-field accuracy

| field | n | accuracy | coverage |
|---|---|---|---|
| company | 13 | 1.0 | 1.0 |
| domain | 13 | 1.0 | 1.0 |
| role | 14 | 1.0 | 1.0 |
| seniority | 14 | 1.0 | 1.0 |
| industry | 13 | 0.769 | 1.0 |
| accepts_email | 13 | 1.0 | 1.0 |

## Calibration (ECE = 0.0926)

| confidence bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| 0.5-0.6 | 2 | 0.525 | 1.0 | 0.475 |
| 0.6-0.7 | 6 | 0.6 | 0.833 | 0.233 |
| 0.7-0.8 | 6 | 0.7 | 0.667 | -0.033 |
| 0.8-0.9 | 9 | 0.84 | 1.0 | 0.16 |
| 0.9-1.0 | 57 | 0.94 | 1.0 | 0.06 |

## Review queue

| precision | recall | F1 | review rate | unsafe approval rate |
|---|---|---|---|---|
| 0.857 | 1.0 | 0.923 | 0.467 | 0.0 |

## Cost and latency per lead

mean $0.040827, p95 $0.058126; latency p50 14969 ms, p95 23535 ms; 3.33 model calls per lead; agent total $0.612412

## Outreach

```json
{
  "n": 15,
  "deterministic_pass_rate": {
    "length_ok": 1.0,
    "word_count_ok": 1.0,
    "not_generic": 1.0,
    "personalized": 1.0,
    "grounded": 0.867,
    "no_unsupported_numbers": 1.0,
    "no_placeholders": 1.0,
    "all_pass": 0.867
  }
}
```
