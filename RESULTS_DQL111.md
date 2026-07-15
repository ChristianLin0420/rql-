# DQL v11.1 -- 50-task OGBench results vs RQL

Runs `DQL111-50` (DQL v11.1), 3 seeds x 1M offline steps, success rate (%) averaged over the last 3 evals (50 episodes each). RQL reference: [arxiv 2606.17551](https://arxiv.org/abs/2606.17551) appendix Table 1.

> **PARTIAL RESULTS** -- 100 run(s) not yet at 1,000,000 steps; numbers below use each run's latest eval.

## Category aggregates

| category | DQL v11.1 | RQL | delta |
|---|---|---|---|
| antmaze-large-navigate | 54 | 83 | -29 |
| antmaze-giant-navigate | 9 | 37 | -28 |
| humanoidmaze-medium-navigate | 47 | 93 | -46 |
| humanoidmaze-large-navigate | 4 | 39 | -35 |
| scene-play | 10 | 89 | -79 |
| puzzle-3x3-play | 24 | 100 | -76 |
| puzzle-4x4-play | 0 | 37 | -36 |
| cube-double-play | 2 | 23 | -20 |
| cube-triple-play | 0 | 4 | -3 |
| cube-quadruple-play | 0 | 51 | -51 |
| **all (50 tasks)** | **15** | **56** | **-40** |

Per-task (+-3pt band): **0 wins / 4 ties / 46 losses** vs RQL.

## Per-task results

| task | DQL v11.1 (mean+-std, n) | RQL | delta | min step |
|---|---|---|---|---|
| antmaze-large-navigate-task1 | 71 +- 10 (n=2) | 84 | -13 | 800,000 |
| antmaze-large-navigate-task2 | 1 +- 0 (n=2) | 80 | -79 | 800,000 |
| antmaze-large-navigate-task3 | 84 +- 0 (n=2) | 95 | -11 | 800,000 |
| antmaze-large-navigate-task4 | 43 +- 0 (n=2) | 81 | -38 | 800,000 |
| antmaze-large-navigate-task5 | 71 +- 2 (n=2) | 76 | -5 | 800,000 |
| antmaze-giant-navigate-task1 | 0 +- 0 (n=2) | 15 | -15 | 800,000 |
| antmaze-giant-navigate-task2 | 0 +- 0 (n=2) | 44 | -44 | 800,000 |
| antmaze-giant-navigate-task3 | 0 +- 0 (n=2) | 21 | -21 | 800,000 |
| antmaze-giant-navigate-task4 | 0 +- 0 (n=2) | 35 | -35 | 800,000 |
| antmaze-giant-navigate-task5 | 45 +- 11 (n=2) | 69 | -24 | 800,000 |
| humanoidmaze-medium-navigate-task1 | 47 +- 18 (n=2) | 96 | -49 | 550,000 |
| humanoidmaze-medium-navigate-task2 | 62 +- 7 (n=2) | 99 | -37 | 550,000 |
| humanoidmaze-medium-navigate-task3 | 30 +- 10 (n=2) | 99 | -69 | 550,000 |
| humanoidmaze-medium-navigate-task4 | 16 +- 4 (n=2) | 72 | -56 | 550,000 |
| humanoidmaze-medium-navigate-task5 | 77 +- 7 (n=2) | 99 | -22 | 700,000 |
| humanoidmaze-large-navigate-task1 | 2 +- 0 (n=2) | 76 | -74 | 550,000 |
| humanoidmaze-large-navigate-task2 | 0 +- 0 (n=2) | 4 | -4 | 550,000 |
| humanoidmaze-large-navigate-task3 | 10 +- 4 (n=2) | 36 | -26 | 550,000 |
| humanoidmaze-large-navigate-task4 | 5 +- 2 (n=2) | 42 | -37 | 550,000 |
| humanoidmaze-large-navigate-task5 | 5 +- 4 (n=2) | 37 | -32 | 550,000 |
| scene-play-task1 | 37 +- 8 (n=2) | 100 | -63 | 550,000 |
| scene-play-task2 | 2 +- 0 (n=2) | 72 | -70 | 550,000 |
| scene-play-task3 | 2 +- 2 (n=2) | 96 | -94 | 550,000 |
| scene-play-task4 | 10 +- 7 (n=2) | 100 | -90 | 550,000 |
| scene-play-task5 | 1 +- 0 (n=2) | 79 | -78 | 550,000 |
| puzzle-3x3-play-task1 | 83 +- 2 (n=2) | 100 | -17 | 800,000 |
| puzzle-3x3-play-task2 | 19 +- 1 (n=2) | 100 | -81 | 700,000 |
| puzzle-3x3-play-task3 | 6 +- 3 (n=2) | 100 | -94 | 600,000 |
| puzzle-3x3-play-task4 | 6 +- 4 (n=2) | 100 | -94 | 700,000 |
| puzzle-3x3-play-task5 | 5 +- 2 (n=2) | 100 | -95 | 800,000 |
| puzzle-4x4-play-task1 | 0 +- 0 (n=2) | 64 | -64 | 550,000 |
| puzzle-4x4-play-task2 | 0 +- 0 (n=2) | 26 | -26 | 550,000 |
| puzzle-4x4-play-task3 | 0 +- 0 (n=2) | 32 | -32 | 550,000 |
| puzzle-4x4-play-task4 | 0 +- 0 (n=2) | 40 | -40 | 550,000 |
| puzzle-4x4-play-task5 | 0 +- 0 (n=2) | 21 | -21 | 550,000 |
| cube-double-play-task1 | 10 +- 2 (n=2) | 51 | -41 | 800,000 |
| cube-double-play-task2 | 0 +- 0 (n=2) | 25 | -25 | 800,000 |
| cube-double-play-task3 | 0 +- 0 (n=2) | 19 | -19 | 800,000 |
| cube-double-play-task4 | 0 +- 0 (n=2) | 6 | -6 | 800,000 |
| cube-double-play-task5 | 2 +- 1 (n=2) | 12 | -10 | 800,000 |
| cube-triple-play-task1 | 1 +- 2 (n=2) | 11 | -10 | 550,000 |
| cube-triple-play-task2 | 0 +- 0 (n=2) | 1 | -1 | 550,000 |
| cube-triple-play-task3 | 0 +- 0 (n=2) | 1 | -1 | 550,000 |
| cube-triple-play-task4 | 0 +- 0 (n=2) | 0 | +0 | 550,000 |
| cube-triple-play-task5 | 0 +- 0 (n=2) | 5 | -5 | 450,000 |
| cube-quadruple-play-task1 | 0 +- 0 (n=2) | 87 | -87 | 550,000 |
| cube-quadruple-play-task2 | 0 +- 0 (n=2) | 81 | -81 | 450,000 |
| cube-quadruple-play-task3 | 0 +- 0 (n=2) | 62 | -62 | 550,000 |
| cube-quadruple-play-task4 | 0 +- 0 (n=2) | 25 | -25 | 550,000 |
| cube-quadruple-play-task5 | 0 +- 0 (n=2) | 0 | +0 | 550,000 |

## Comparison caveats

- RQL trains **2M** gradient steps (paper Table 2); these runs use **1M**.
- Paper's `puzzle-4x4` and `cube-quadruple` rows use the **100M-transition** dataset variants; we use the standard `-v0` datasets.
- Ours: 3 seeds, mean of last 3 evals; paper: bootstrap CI over its own seeds/protocol.

## Incomplete runs

- antmaze-large-navigate-singletask-task1-v0 sd1: at step 800000
- antmaze-large-navigate-singletask-task1-v0 sd2: no eval yet
- antmaze-large-navigate-singletask-task2-v0 sd1: at step 800000
- antmaze-large-navigate-singletask-task2-v0 sd2: no eval yet
- antmaze-large-navigate-singletask-task3-v0 sd1: at step 800000
- antmaze-large-navigate-singletask-task3-v0 sd2: no eval yet
- antmaze-large-navigate-singletask-task4-v0 sd1: at step 800000
- antmaze-large-navigate-singletask-task4-v0 sd2: no eval yet
- antmaze-large-navigate-singletask-task5-v0 sd1: at step 800000
- antmaze-large-navigate-singletask-task5-v0 sd2: no eval yet
- antmaze-giant-navigate-singletask-task1-v0 sd1: at step 800000
- antmaze-giant-navigate-singletask-task1-v0 sd2: no eval yet
- antmaze-giant-navigate-singletask-task2-v0 sd1: at step 800000
- antmaze-giant-navigate-singletask-task2-v0 sd2: no eval yet
- antmaze-giant-navigate-singletask-task3-v0 sd1: at step 800000
- antmaze-giant-navigate-singletask-task3-v0 sd2: no eval yet
- antmaze-giant-navigate-singletask-task4-v0 sd1: at step 800000
- antmaze-giant-navigate-singletask-task4-v0 sd2: no eval yet
- antmaze-giant-navigate-singletask-task5-v0 sd1: at step 800000
- antmaze-giant-navigate-singletask-task5-v0 sd2: no eval yet
- humanoidmaze-medium-navigate-singletask-task1-v0 sd1: at step 550000
- humanoidmaze-medium-navigate-singletask-task1-v0 sd2: no eval yet
- humanoidmaze-medium-navigate-singletask-task2-v0 sd1: at step 550000
- humanoidmaze-medium-navigate-singletask-task2-v0 sd2: no eval yet
- humanoidmaze-medium-navigate-singletask-task3-v0 sd1: at step 550000
- humanoidmaze-medium-navigate-singletask-task3-v0 sd2: no eval yet
- humanoidmaze-medium-navigate-singletask-task4-v0 sd1: at step 550000
- humanoidmaze-medium-navigate-singletask-task4-v0 sd2: no eval yet
- humanoidmaze-medium-navigate-singletask-task5-v0 sd1: at step 700000
- humanoidmaze-medium-navigate-singletask-task5-v0 sd2: no eval yet
- ... and 70 more
