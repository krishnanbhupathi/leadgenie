# Eval: split `test` (n=15)

model `claude-opus-5-5` (effort medium), prompt `1ece14027510` (3 lessons), git `ca4ca1a`, 2026-10-03T16:29:06+00:00

## Per-field accuracy

| field | n | accuracy | coverage |
|---|---|---|---|
| company | 13 | 1.0 | 1.0 |
| domain | 13 | 1.0 | 1.0 |
| role | 14 | 1.0 | 1.0 |
| seniority | 14 | 1.0 | 1.0 |
| industry | 13 | 0.769 | 1.0 |
| accepts_email | 13 | 1.0 | 1.0 |

## Calibration (ECE = 0.1129)

| confidence bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| 0.4-0.5 | 1 | 0.4 | 1.0 | 0.6 |
| 0.5-0.6 | 6 | 0.542 | 1.0 | 0.458 |
| 0.6-0.7 | 5 | 0.6 | 0.6 | 0.0 |
| 0.7-0.8 | 2 | 0.725 | 0.5 | -0.225 |
| 0.8-0.9 | 10 | 0.83 | 1.0 | 0.17 |
| 0.9-1.0 | 56 | 0.937 | 1.0 | 0.063 |

## Review queue

| precision | recall | F1 | review rate | unsafe approval rate |
|---|---|---|---|---|
| 0.857 | 1.0 | 0.923 | 0.467 | 0.0 |

## Cost and latency per lead

mean $0.043473, p95 $0.0461532; latency p50 14495 ms, p95 17932 ms; 3.0 model calls per lead; agent total $0.652097

## Outreach

```json
{
  "n": 15,
  "deterministic_pass_rate": {
    "length_ok": 0.933,
    "word_count_ok": 1.0,
    "not_generic": 1.0,
    "personalized": 0.933,
    "grounded": 0.867,
    "no_unsupported_numbers": 1.0,
    "no_placeholders": 1.0,
    "all_pass": 0.867
  }
}
```
