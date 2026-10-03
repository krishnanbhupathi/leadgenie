# Eval: split `test` (n=15)

model `claude-opus-5-5` (effort medium), prompt `3e8b2cdaa6e9` (0 lessons), git `433755f`, 2026-10-03T16:15:47+00:00

## Per-field accuracy

| field | n | accuracy | coverage |
|---|---|---|---|
| company | 13 | 1.0 | 1.0 |
| domain | 13 | 1.0 | 1.0 |
| role | 14 | 1.0 | 1.0 |
| seniority | 14 | 1.0 | 1.0 |
| industry | 13 | 0.769 | 1.0 |
| accepts_email | 13 | 1.0 | 1.0 |

## Calibration (ECE = 0.0795)

| confidence bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| 0.5-0.6 | 2 | 0.525 | 1.0 | 0.475 |
| 0.6-0.7 | 8 | 0.606 | 0.625 | 0.019 |
| 0.7-0.8 | 1 | 0.75 | 1.0 | 0.25 |
| 0.8-0.9 | 11 | 0.852 | 1.0 | 0.148 |
| 0.9-1.0 | 58 | 0.942 | 1.0 | 0.058 |

## Review queue

| precision | recall | F1 | review rate | unsafe approval rate |
|---|---|---|---|---|
| 0.857 | 1.0 | 0.923 | 0.467 | 0.0 |

## Cost and latency per lead

mean $0.040244, p95 $0.047036999999999995; latency p50 14132 ms, p95 19451 ms; 3.2 model calls per lead; agent total $0.603664

## Outreach

```json
{
  "n": 15,
  "deterministic_pass_rate": {
    "length_ok": 1.0,
    "word_count_ok": 1.0,
    "not_generic": 1.0,
    "personalized": 0.933,
    "grounded": 0.867,
    "no_unsupported_numbers": 1.0,
    "no_placeholders": 1.0,
    "all_pass": 0.8
  },
  "judge": {
    "n": 15,
    "pass_rate": 0.067,
    "mean_scores": {
      "specificity": 2.8,
      "grounding": 4.2,
      "relevance": 3.0,
      "tone": 3.87
    },
    "cost_usd": 0.047376
  }
}
```
