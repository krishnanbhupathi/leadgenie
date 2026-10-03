# Eval: split `train` (n=30)

model `claude-opus-5-5` (effort medium), prompt `1ece14027510` (3 lessons), git `ca4ca1a`, 2026-10-03T16:26:01+00:00

## Per-field accuracy

| field | n | accuracy | coverage |
|---|---|---|---|
| company | 27 | 1.0 | 1.0 |
| domain | 27 | 1.0 | 1.0 |
| role | 28 | 1.0 | 1.0 |
| seniority | 28 | 0.964 | 1.0 |
| industry | 27 | 1.0 | 1.0 |
| accepts_email | 27 | 1.0 | 1.0 |

## Calibration (ECE = 0.1212)

| confidence bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| 0.3-0.4 | 1 | 0.3 | 0.0 | -0.3 |
| 0.4-0.5 | 1 | 0.4 | 1.0 | 0.6 |
| 0.5-0.6 | 8 | 0.531 | 1.0 | 0.469 |
| 0.6-0.7 | 7 | 0.621 | 1.0 | 0.379 |
| 0.7-0.8 | 8 | 0.738 | 1.0 | 0.262 |
| 0.8-0.9 | 11 | 0.837 | 1.0 | 0.163 |
| 0.9-1.0 | 128 | 0.932 | 1.0 | 0.068 |

## Review queue

| precision | recall | F1 | review rate | unsafe approval rate |
|---|---|---|---|---|
| 0.636 | 1.0 | 0.778 | 0.367 | 0.0 |

## Cost and latency per lead

mean $0.043452, p95 $0.047487; latency p50 13847 ms, p95 16501 ms; 2.87 model calls per lead; agent total $1.303549

## Outreach

```json
{
  "n": 30,
  "deterministic_pass_rate": {
    "length_ok": 0.967,
    "word_count_ok": 1.0,
    "not_generic": 1.0,
    "personalized": 0.967,
    "grounded": 0.9,
    "no_unsupported_numbers": 1.0,
    "no_placeholders": 1.0,
    "all_pass": 0.9
  }
}
```
