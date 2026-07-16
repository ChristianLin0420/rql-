# DQL v11.2 -- 50-task OGBench results vs RQL

Runs `DQL112-50` (DQL v11.2), 3 seeds x 2M offline steps, success rate (%) averaged over the last 3 evals (50 episodes each). RQL reference: [arxiv 2606.17551](https://arxiv.org/abs/2606.17551) appendix Table 1.

## Category aggregates

| category | DQL v11.2 | RQL | delta |
|---|---|---|---|
| antmaze-large-navigate | 60 | 83 | -24 |
| antmaze-giant-navigate | 16 | 37 | -21 |
| humanoidmaze-medium-navigate | 66 | 93 | -27 |
| humanoidmaze-large-navigate | 7 | 39 | -32 |
| scene-play | 8 | 89 | -81 |
| puzzle-3x3-play | 15 | 100 | -85 |
| puzzle-4x4-play | 0 | 37 | -36 |
| cube-double-play | 2 | 23 | -20 |
| cube-triple-play | 0 | 4 | -4 |
| cube-quadruple-play | 0 | 51 | -51 |
| **all (50 tasks)** | **17** | **56** | **-38** |

Per-task (+-3pt band): **1 wins / 4 ties / 45 losses** vs RQL.

## Per-task results

| task | DQL v11.2 (mean+-std, n) | RQL | delta | min step |
|---|---|---|---|---|
| antmaze-large-navigate-task1 | 66 +- 5 (n=3) | 84 | -18 | 2,000,000 |
| antmaze-large-navigate-task2 | 11 +- 2 (n=3) | 80 | -69 | 2,000,000 |
| antmaze-large-navigate-task3 | 82 +- 3 (n=3) | 95 | -13 | 2,000,000 |
| antmaze-large-navigate-task4 | 68 +- 1 (n=3) | 81 | -13 | 2,000,000 |
| antmaze-large-navigate-task5 | 71 +- 0 (n=3) | 76 | -5 | 2,000,000 |
| antmaze-giant-navigate-task1 | 2 +- 2 (n=3) | 15 | -13 | 2,000,000 |
| antmaze-giant-navigate-task2 | 0 +- 0 (n=3) | 44 | -44 | 2,000,000 |
| antmaze-giant-navigate-task3 | 0 +- 0 (n=3) | 21 | -21 | 2,000,000 |
| antmaze-giant-navigate-task4 | 0 +- 0 (n=3) | 35 | -35 | 2,000,000 |
| antmaze-giant-navigate-task5 | 76 +- 3 (n=3) | 69 | +7 | 2,000,000 |
| humanoidmaze-medium-navigate-task1 | 59 +- 4 (n=3) | 96 | -37 | 2,000,000 |
| humanoidmaze-medium-navigate-task2 | 74 +- 9 (n=3) | 99 | -25 | 2,000,000 |
| humanoidmaze-medium-navigate-task3 | 63 +- 12 (n=3) | 99 | -36 | 2,000,000 |
| humanoidmaze-medium-navigate-task4 | 45 +- 7 (n=3) | 72 | -27 | 2,000,000 |
| humanoidmaze-medium-navigate-task5 | 87 +- 3 (n=3) | 99 | -12 | 2,000,000 |
| humanoidmaze-large-navigate-task1 | 3 +- 1 (n=3) | 76 | -73 | 2,000,000 |
| humanoidmaze-large-navigate-task2 | 0 +- 0 (n=3) | 4 | -4 | 2,000,000 |
| humanoidmaze-large-navigate-task3 | 15 +- 5 (n=3) | 36 | -21 | 2,000,000 |
| humanoidmaze-large-navigate-task4 | 11 +- 1 (n=3) | 42 | -31 | 2,000,000 |
| humanoidmaze-large-navigate-task5 | 6 +- 2 (n=3) | 37 | -31 | 2,000,000 |
| scene-play-task1 | 24 +- 4 (n=3) | 100 | -76 | 2,000,000 |
| scene-play-task2 | 3 +- 1 (n=3) | 72 | -69 | 2,000,000 |
| scene-play-task3 | 3 +- 2 (n=3) | 96 | -93 | 2,000,000 |
| scene-play-task4 | 10 +- 3 (n=3) | 100 | -90 | 2,000,000 |
| scene-play-task5 | 1 +- 1 (n=3) | 79 | -78 | 2,000,000 |
| puzzle-3x3-play-task1 | 47 +- 1 (n=3) | 100 | -53 | 2,000,000 |
| puzzle-3x3-play-task2 | 12 +- 2 (n=3) | 100 | -88 | 2,000,000 |
| puzzle-3x3-play-task3 | 3 +- 1 (n=3) | 100 | -97 | 2,000,000 |
| puzzle-3x3-play-task4 | 6 +- 2 (n=3) | 100 | -94 | 2,000,000 |
| puzzle-3x3-play-task5 | 6 +- 3 (n=3) | 100 | -94 | 2,000,000 |
| puzzle-4x4-play-task1 | 0 +- 0 (n=3) | 64 | -64 | 2,000,000 |
| puzzle-4x4-play-task2 | 0 +- 0 (n=3) | 26 | -26 | 2,000,000 |
| puzzle-4x4-play-task3 | 0 +- 0 (n=3) | 32 | -32 | 2,000,000 |
| puzzle-4x4-play-task4 | 0 +- 0 (n=3) | 40 | -40 | 2,000,000 |
| puzzle-4x4-play-task5 | 0 +- 0 (n=3) | 21 | -21 | 2,000,000 |
| cube-double-play-task1 | 11 +- 2 (n=3) | 51 | -40 | 2,000,000 |
| cube-double-play-task2 | 0 +- 0 (n=3) | 25 | -25 | 2,000,000 |
| cube-double-play-task3 | 0 +- 0 (n=3) | 19 | -19 | 2,000,000 |
| cube-double-play-task4 | 0 +- 0 (n=3) | 6 | -6 | 2,000,000 |
| cube-double-play-task5 | 1 +- 1 (n=3) | 12 | -11 | 2,000,000 |
| cube-triple-play-task1 | 0 +- 0 (n=3) | 11 | -11 | 2,000,000 |
| cube-triple-play-task2 | 0 +- 0 (n=3) | 1 | -1 | 2,000,000 |
| cube-triple-play-task3 | 0 +- 0 (n=3) | 1 | -1 | 2,000,000 |
| cube-triple-play-task4 | 0 +- 0 (n=3) | 0 | +0 | 2,000,000 |
| cube-triple-play-task5 | 0 +- 0 (n=3) | 5 | -5 | 2,000,000 |
| cube-quadruple-play-task1 | 0 +- 0 (n=3) | 87 | -87 | 2,000,000 |
| cube-quadruple-play-task2 | 0 +- 0 (n=3) | 81 | -81 | 2,000,000 |
| cube-quadruple-play-task3 | 0 +- 0 (n=3) | 62 | -62 | 2,000,000 |
| cube-quadruple-play-task4 | 0 +- 0 (n=3) | 25 | -25 | 2,000,000 |
| cube-quadruple-play-task5 | 0 +- 0 (n=3) | 0 | +0 | 2,000,000 |

## Comparison caveats

- RQL trains **2M** gradient steps (paper Table 2); these runs use **2M**.
- Paper's `puzzle-4x4` and `cube-quadruple` rows use the **100M-transition** dataset variants; we use the standard `-v0` datasets.
- Ours: 3 seeds, mean of last 3 evals; paper: bootstrap CI over its own seeds/protocol.
