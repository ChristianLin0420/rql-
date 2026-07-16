# DQL v11.2 -- 50-task OGBench results vs RQL
> **INTERIM 1M SNAPSHOT** (2026-07-15 11:50) — all 150 seeds passed 1M; the 2M extension is running, so rows of early families (antmaze, humanoidmaze) may include evals past 1M (report takes each run's last 3 evals). Clean <=1M numbers: overall 18.6 vs RQL 55.5, W/T/L 1/5/44. Final 2M report will replace RESULTS_DQL112.md.


Runs `DQL112-50` (DQL v11.2), 3 seeds x 1M offline steps, success rate (%) averaged over the last 3 evals (50 episodes each). RQL reference: [arxiv 2606.17551](https://arxiv.org/abs/2606.17551) appendix Table 1.

## Category aggregates

| category | DQL v11.2 | RQL | delta |
|---|---|---|---|
| antmaze-large-navigate | 63 | 83 | -20 |
| antmaze-giant-navigate | 16 | 37 | -21 |
| humanoidmaze-medium-navigate | 65 | 93 | -28 |
| humanoidmaze-large-navigate | 8 | 39 | -31 |
| scene-play | 10 | 89 | -79 |
| puzzle-3x3-play | 19 | 100 | -81 |
| puzzle-4x4-play | 0 | 37 | -37 |
| cube-double-play | 3 | 23 | -20 |
| cube-triple-play | 0 | 4 | -3 |
| cube-quadruple-play | 0 | 51 | -51 |
| **all (50 tasks)** | **18** | **56** | **-37** |

Per-task (+-3pt band): **1 wins / 5 ties / 44 losses** vs RQL.

## Per-task results

| task | DQL v11.2 (mean+-std, n) | RQL | delta | min step |
|---|---|---|---|---|
| antmaze-large-navigate-task1 | 71 +- 8 (n=3) | 84 | -13 | 1,000,000 |
| antmaze-large-navigate-task2 | 12 +- 1 (n=3) | 80 | -68 | 1,000,000 |
| antmaze-large-navigate-task3 | 88 +- 5 (n=3) | 95 | -7 | 1,000,000 |
| antmaze-large-navigate-task4 | 73 +- 3 (n=3) | 81 | -8 | 1,000,000 |
| antmaze-large-navigate-task5 | 73 +- 2 (n=3) | 76 | -3 | 1,000,000 |
| antmaze-giant-navigate-task1 | 3 +- 1 (n=3) | 15 | -12 | 1,000,000 |
| antmaze-giant-navigate-task2 | 0 +- 0 (n=3) | 44 | -44 | 1,000,000 |
| antmaze-giant-navigate-task3 | 0 +- 0 (n=3) | 21 | -21 | 1,000,000 |
| antmaze-giant-navigate-task4 | 1 +- 1 (n=3) | 35 | -34 | 1,000,000 |
| antmaze-giant-navigate-task5 | 76 +- 3 (n=3) | 69 | +7 | 1,000,000 |
| humanoidmaze-medium-navigate-task1 | 64 +- 4 (n=3) | 96 | -32 | 1,000,000 |
| humanoidmaze-medium-navigate-task2 | 77 +- 4 (n=3) | 99 | -22 | 1,000,000 |
| humanoidmaze-medium-navigate-task3 | 56 +- 10 (n=3) | 99 | -43 | 1,000,000 |
| humanoidmaze-medium-navigate-task4 | 37 +- 8 (n=3) | 72 | -35 | 1,000,000 |
| humanoidmaze-medium-navigate-task5 | 90 +- 1 (n=3) | 99 | -9 | 1,000,000 |
| humanoidmaze-large-navigate-task1 | 5 +- 2 (n=3) | 76 | -71 | 1,000,000 |
| humanoidmaze-large-navigate-task2 | 1 +- 0 (n=3) | 4 | -3 | 1,000,000 |
| humanoidmaze-large-navigate-task3 | 14 +- 1 (n=3) | 36 | -22 | 1,000,000 |
| humanoidmaze-large-navigate-task4 | 11 +- 1 (n=3) | 42 | -31 | 1,000,000 |
| humanoidmaze-large-navigate-task5 | 9 +- 3 (n=3) | 37 | -28 | 1,000,000 |
| scene-play-task1 | 31 +- 4 (n=3) | 100 | -69 | 1,000,000 |
| scene-play-task2 | 3 +- 0 (n=3) | 72 | -69 | 1,000,000 |
| scene-play-task3 | 2 +- 1 (n=3) | 96 | -94 | 1,000,000 |
| scene-play-task4 | 13 +- 3 (n=3) | 100 | -87 | 1,000,000 |
| scene-play-task5 | 2 +- 1 (n=3) | 79 | -77 | 1,000,000 |
| puzzle-3x3-play-task1 | 61 +- 12 (n=3) | 100 | -39 | 1,000,000 |
| puzzle-3x3-play-task2 | 16 +- 3 (n=3) | 100 | -84 | 1,000,000 |
| puzzle-3x3-play-task3 | 5 +- 2 (n=3) | 100 | -95 | 1,000,000 |
| puzzle-3x3-play-task4 | 6 +- 2 (n=3) | 100 | -94 | 1,000,000 |
| puzzle-3x3-play-task5 | 6 +- 4 (n=3) | 100 | -94 | 1,000,000 |
| puzzle-4x4-play-task1 | 0 +- 0 (n=3) | 64 | -64 | 1,000,000 |
| puzzle-4x4-play-task2 | 0 +- 0 (n=3) | 26 | -26 | 1,000,000 |
| puzzle-4x4-play-task3 | 0 +- 0 (n=3) | 32 | -32 | 1,000,000 |
| puzzle-4x4-play-task4 | 0 +- 0 (n=3) | 40 | -40 | 1,000,000 |
| puzzle-4x4-play-task5 | 0 +- 0 (n=3) | 21 | -21 | 1,000,000 |
| cube-double-play-task1 | 12 +- 2 (n=3) | 51 | -39 | 1,000,000 |
| cube-double-play-task2 | 0 +- 0 (n=3) | 25 | -25 | 1,000,000 |
| cube-double-play-task3 | 0 +- 0 (n=3) | 19 | -19 | 1,000,000 |
| cube-double-play-task4 | 0 +- 0 (n=3) | 6 | -6 | 1,000,000 |
| cube-double-play-task5 | 2 +- 1 (n=3) | 12 | -10 | 1,000,000 |
| cube-triple-play-task1 | 2 +- 1 (n=3) | 11 | -9 | 1,000,000 |
| cube-triple-play-task2 | 0 +- 0 (n=3) | 1 | -1 | 1,000,000 |
| cube-triple-play-task3 | 0 +- 0 (n=3) | 1 | -1 | 1,000,000 |
| cube-triple-play-task4 | 0 +- 0 (n=3) | 0 | +0 | 1,000,000 |
| cube-triple-play-task5 | 0 +- 0 (n=3) | 5 | -5 | 1,000,000 |
| cube-quadruple-play-task1 | 0 +- 0 (n=3) | 87 | -87 | 1,000,000 |
| cube-quadruple-play-task2 | 0 +- 0 (n=3) | 81 | -81 | 1,000,000 |
| cube-quadruple-play-task3 | 0 +- 0 (n=3) | 62 | -62 | 1,000,000 |
| cube-quadruple-play-task4 | 0 +- 0 (n=3) | 25 | -25 | 1,000,000 |
| cube-quadruple-play-task5 | 0 +- 0 (n=3) | 0 | +0 | 1,000,000 |

## Comparison caveats

- RQL trains **2M** gradient steps (paper Table 2); these runs use **1M**.
- Paper's `puzzle-4x4` and `cube-quadruple` rows use the **100M-transition** dataset variants; we use the standard `-v0` datasets.
- Ours: 3 seeds, mean of last 3 evals; paper: bootstrap CI over its own seeds/protocol.
