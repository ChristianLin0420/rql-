# dql113 -- 50-task OGBench results vs RQL

Runs `DQL113-50` (dql113), 3 seeds x 1M offline steps, success rate (%) averaged over the last 3 evals (50 episodes each). RQL reference: [arxiv 2606.17551](https://arxiv.org/abs/2606.17551) appendix Table 1.

## Category aggregates

| category | dql113 | RQL | delta |
|---|---|---|---|
| antmaze-large-navigate | 48 | 83 | -35 |
| antmaze-giant-navigate | 1 | 37 | -36 |
| humanoidmaze-medium-navigate | 0 | 93 | -93 |
| humanoidmaze-large-navigate | 0 | 39 | -39 |
| scene-play | 42 | 89 | -48 |
| puzzle-3x3-play | 48 | 100 | -52 |
| puzzle-4x4-play | 2 | 37 | -35 |
| cube-double-play | 3 | 23 | -20 |
| cube-triple-play | 0 | 4 | -3 |
| cube-quadruple-play | 0 | 51 | -51 |
| **all (50 tasks)** | **14** | **56** | **-41** |

Per-task (+-3pt band): **0 wins / 5 ties / 45 losses** vs RQL.

## Per-task results

| task | dql113 (mean+-std, n) | RQL | delta | min step |
|---|---|---|---|---|
| antmaze-large-navigate-task1 | 78 +- 3 (n=3) | 84 | -6 | 1,000,000 |
| antmaze-large-navigate-task2 | 8 +- 6 (n=3) | 80 | -72 | 1,000,000 |
| antmaze-large-navigate-task3 | 65 +- 7 (n=3) | 95 | -30 | 1,000,000 |
| antmaze-large-navigate-task4 | 21 +- 15 (n=3) | 81 | -60 | 1,000,000 |
| antmaze-large-navigate-task5 | 69 +- 1 (n=3) | 76 | -7 | 1,000,000 |
| antmaze-giant-navigate-task1 | 1 +- 0 (n=3) | 15 | -14 | 1,000,000 |
| antmaze-giant-navigate-task2 | 2 +- 1 (n=3) | 44 | -42 | 1,000,000 |
| antmaze-giant-navigate-task3 | 1 +- 1 (n=3) | 21 | -20 | 1,000,000 |
| antmaze-giant-navigate-task4 | 2 +- 2 (n=3) | 35 | -33 | 1,000,000 |
| antmaze-giant-navigate-task5 | 0 +- 0 (n=3) | 69 | -69 | 1,000,000 |
| humanoidmaze-medium-navigate-task1 | 0 +- 0 (n=3) | 96 | -96 | 1,000,000 |
| humanoidmaze-medium-navigate-task2 | 0 +- 0 (n=3) | 99 | -99 | 1,000,000 |
| humanoidmaze-medium-navigate-task3 | 0 +- 0 (n=3) | 99 | -99 | 1,000,000 |
| humanoidmaze-medium-navigate-task4 | 0 +- 0 (n=3) | 72 | -72 | 1,000,000 |
| humanoidmaze-medium-navigate-task5 | 1 +- 0 (n=3) | 99 | -98 | 1,000,000 |
| humanoidmaze-large-navigate-task1 | 0 +- 0 (n=3) | 76 | -76 | 1,000,000 |
| humanoidmaze-large-navigate-task2 | 0 +- 0 (n=3) | 4 | -4 | 1,000,000 |
| humanoidmaze-large-navigate-task3 | 0 +- 0 (n=3) | 36 | -36 | 1,000,000 |
| humanoidmaze-large-navigate-task4 | 0 +- 0 (n=3) | 42 | -42 | 1,000,000 |
| humanoidmaze-large-navigate-task5 | 0 +- 0 (n=3) | 37 | -37 | 1,000,000 |
| scene-play-task1 | 70 +- 8 (n=3) | 100 | -30 | 1,000,000 |
| scene-play-task2 | 30 +- 16 (n=3) | 72 | -42 | 1,000,000 |
| scene-play-task3 | 18 +- 9 (n=3) | 96 | -78 | 1,000,000 |
| scene-play-task4 | 71 +- 3 (n=3) | 100 | -29 | 1,000,000 |
| scene-play-task5 | 20 +- 11 (n=3) | 79 | -59 | 1,000,000 |
| puzzle-3x3-play-task1 | 100 +- 0 (n=3) | 100 | -0 | 1,000,000 |
| puzzle-3x3-play-task2 | 63 +- 8 (n=3) | 100 | -37 | 1,000,000 |
| puzzle-3x3-play-task3 | 26 +- 5 (n=3) | 100 | -74 | 1,000,000 |
| puzzle-3x3-play-task4 | 14 +- 2 (n=3) | 100 | -86 | 1,000,000 |
| puzzle-3x3-play-task5 | 39 +- 12 (n=3) | 100 | -61 | 1,000,000 |
| puzzle-4x4-play-task1 | 3 +- 3 (n=3) | 64 | -61 | 1,000,000 |
| puzzle-4x4-play-task2 | 0 +- 0 (n=3) | 26 | -26 | 1,000,000 |
| puzzle-4x4-play-task3 | 5 +- 4 (n=3) | 32 | -27 | 1,000,000 |
| puzzle-4x4-play-task4 | 1 +- 0 (n=3) | 40 | -39 | 1,000,000 |
| puzzle-4x4-play-task5 | 1 +- 0 (n=3) | 21 | -20 | 1,000,000 |
| cube-double-play-task1 | 6 +- 1 (n=3) | 51 | -45 | 1,000,000 |
| cube-double-play-task2 | 0 +- 0 (n=3) | 25 | -25 | 1,000,000 |
| cube-double-play-task3 | 1 +- 1 (n=3) | 19 | -18 | 1,000,000 |
| cube-double-play-task4 | 1 +- 1 (n=3) | 6 | -5 | 1,000,000 |
| cube-double-play-task5 | 4 +- 2 (n=3) | 12 | -8 | 1,000,000 |
| cube-triple-play-task1 | 1 +- 1 (n=3) | 11 | -10 | 1,000,000 |
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
