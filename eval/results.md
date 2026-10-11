# vachan-eval results

engine=mock seed=7 n=30
mean_wer=0.1617 mean_cer=0.1181 p50=438.94ms p95=594.46ms cost=$0.0
GATE PASS (wer<=0.25, p95<=1200.0ms)
BASELINE OK mean_wer: current=0.1617 baseline=0.1617 delta=+0.0000 (tol +0.03)
BASELINE OK mean_cer: current=0.1181 baseline=0.1181 delta=+0.0000 (tol +0.03)
BASELINE OK cat:easy: current=0.0286 baseline=0.0286 delta=+0.0000 (tol +0.03)
BASELINE OK cat:hard: current=0.2029 baseline=0.2029 delta=+0.0000 (tol +0.03)
BASELINE OK cat:medium: current=0.123 baseline=0.123 delta=+0.0000 (tol +0.03)
BASELINE OK cat:noise: current=0.1111 baseline=0.1111 delta=+0.0000 (tol +0.03)
OVERALL PASS (gates + baseline)

per-category:
| category | n | mean_wer | gate_wer_max | status |
|---|---|---|---|---|
| adversarial | 5 | 0.3267 | report-only | PASS |
| easy | 5 | 0.0286 | 0.1 | PASS |
| hard | 8 | 0.2029 | 0.3 | PASS |
| medium | 10 | 0.123 | 0.2 | PASS |
| noise | 2 | 0.1111 | 0.35 | PASS |

| id | lang | diff | wer | cer | lat_ms |
|---|---|---|---|---|---|
| ration-01 | hi | easy | 0.0 | 0.0 | 258.62 |
| ration-02 | hi | medium | 0.0 | 0.0 | 397.65 |
| ration-03 | bho | hard | 0.25 | 0.1622 | 524.5 |
| ration-04 | mag | hard | 0.1429 | 0.1389 | 509.51 |
| ration-05 | hi | medium | 0.2222 | 0.1163 | 409.53 |
| scholar-01 | hi | easy | 0.0 | 0.0 | 338.4 |
| scholar-02 | hi | medium | 0.1429 | 0.0278 | 333.39 |
| scholar-03 | bho | hard | 0.3 | 0.1087 | 598.77 |
| scholar-04 | hi | medium | 0.0 | 0.0 | 426.94 |
| scholar-05 | mag | hard | 0.25 | 0.2632 | 547.99 |
| rte-01 | hi | easy | 0.0 | 0.0 | 343.41 |
| rte-02 | hi | medium | 0.1111 | 0.1429 | 411.66 |
| rte-03 | bho | hard | 0.0 | 0.0 | 605.15 |
| rte-04 | hi | medium | 0.1429 | 0.0882 | 348.64 |
| rte-05 | mag | hard | 0.4444 | 0.2791 | 572.64 |
| pension-01 | hi | easy | 0.1429 | 0.1579 | 301.1 |
| pension-02 | bho | hard | 0.1111 | 0.1579 | 563.15 |
| pension-03 | hi | medium | 0.375 | 0.2432 | 382.67 |
| health-01 | hi | easy | 0.0 | 0.0 | 214.96 |
| health-02 | bho | hard | 0.125 | 0.1351 | 559.2 |
| mixed-01 | hi | medium | 0.1111 | 0.0233 | 422.45 |
| mixed-02 | hi | medium | 0.125 | 0.1818 | 363.79 |
| mixed-03 | hi | medium | 0.0 | 0.0 | 410.75 |
| noise-01 | hi | hard | 0.1111 | 0.0222 | 564.76 |
| noise-02 | bho | hard | 0.1111 | 0.0714 | 570.55 |
| adv-01 | hi | hard | 0.3333 | 0.1905 | 563.36 |
| adv-02 | hi | hard | 0.3 | 0.1277 | 589.2 |
| adv-03 | hi | hard | 0.3333 | 0.4074 | 450.95 |
| adv-04 | bho | hard | 0.5 | 0.4286 | 536.76 |
| adv-05 | hi | hard | 0.1667 | 0.069 | 464.39 |
