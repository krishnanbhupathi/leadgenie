# Eval: split `train` (n=30)

model `claude-opus-5-5` (effort medium), prompt `3e8b2cdaa6e9` (0 lessons), git `ca4ca1a`, 2026-10-03T16:20:18+00:00

## Per-field accuracy

| field | n | accuracy | coverage |
|---|---|---|---|
| company | 27 | 1.0 | 1.0 |
| domain | 27 | 1.0 | 1.0 |
| role | 28 | 0.964 | 1.0 |
| seniority | 28 | 0.929 | 1.0 |
| industry | 27 | 1.0 | 1.0 |
| accepts_email | 27 | 1.0 | 1.0 |

## Calibration (ECE = 0.1028)

| confidence bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| 0.4-0.5 | 1 | 0.45 | 1.0 | 0.55 |
| 0.5-0.6 | 3 | 0.533 | 0.333 | -0.2 |
| 0.6-0.7 | 10 | 0.61 | 1.0 | 0.39 |
| 0.7-0.8 | 11 | 0.718 | 1.0 | 0.282 |
| 0.8-0.9 | 14 | 0.849 | 1.0 | 0.151 |
| 0.9-1.0 | 125 | 0.939 | 0.992 | 0.053 |

## Review queue

| precision | recall | F1 | review rate | unsafe approval rate |
|---|---|---|---|---|
| 0.636 | 1.0 | 0.778 | 0.367 | 0.0 |

## Cost and latency per lead

mean $0.039604, p95 $0.042484; latency p50 13842 ms, p95 20064 ms; 3.03 model calls per lead; agent total $1.188126

## Outreach

```json
{
  "n": 30,
  "deterministic_pass_rate": {
    "length_ok": 1.0,
    "word_count_ok": 0.967,
    "not_generic": 1.0,
    "personalized": 1.0,
    "grounded": 0.9,
    "no_unsupported_numbers": 1.0,
    "no_placeholders": 1.0,
    "all_pass": 0.867
  }
}
```
