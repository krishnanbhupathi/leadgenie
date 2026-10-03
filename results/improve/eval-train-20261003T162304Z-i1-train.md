# Eval: split `train` (n=30)

model `claude-opus-5-5` (effort medium), prompt `a29028119aae` (3 lessons), git `ca4ca1a`, 2026-10-03T16:23:04+00:00

## Per-field accuracy

| field | n | accuracy | coverage |
|---|---|---|---|
| company | 27 | 1.0 | 1.0 |
| domain | 27 | 1.0 | 1.0 |
| role | 28 | 1.0 | 1.0 |
| seniority | 28 | 1.0 | 1.0 |
| industry | 27 | 1.0 | 1.0 |
| accepts_email | 27 | 1.0 | 1.0 |

## Calibration (ECE = 0.1157)

| confidence bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| 0.3-0.4 | 1 | 0.35 | 1.0 | 0.65 |
| 0.4-0.5 | 1 | 0.45 | 1.0 | 0.55 |
| 0.5-0.6 | 6 | 0.542 | 1.0 | 0.458 |
| 0.6-0.7 | 6 | 0.608 | 1.0 | 0.392 |
| 0.7-0.8 | 10 | 0.73 | 1.0 | 0.27 |
| 0.8-0.9 | 15 | 0.837 | 1.0 | 0.163 |
| 0.9-1.0 | 125 | 0.94 | 1.0 | 0.06 |

## Review queue

| precision | recall | F1 | review rate | unsafe approval rate |
|---|---|---|---|---|
| 0.7 | 1.0 | 0.824 | 0.333 | 0.0 |

## Cost and latency per lead

mean $0.041665, p95 $0.0452148; latency p50 13397 ms, p95 16923 ms; 2.9 model calls per lead; agent total $1.249956

## Outreach

```json
{
  "n": 30,
  "deterministic_pass_rate": {
    "length_ok": 1.0,
    "word_count_ok": 1.0,
    "not_generic": 1.0,
    "personalized": 1.0,
    "grounded": 0.9,
    "no_unsupported_numbers": 1.0,
    "no_placeholders": 1.0,
    "all_pass": 0.9
  }
}
```
