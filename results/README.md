# Results

Cube layout 0, episodes up to 120 chunks (96 s), 60 episodes per row. The
numbers below are recomputed from the per-episode files in `n60/`.

| policy | per-cube placed | full 3/3 | ≤ 1 cube ("stall") | wrong basket | robot fell |
|---|---|---|---|---|---|
| frozen GR00T N1.7, its own noise (native) | 78.9 % | 48.3 % | 10.0 % | 0 | 0 |
| **+ DSRL (champion `r9-pixel-anneal-chain`, step 4000)** | **88.9 %** | **78.3 %** | 11.7 % | 0 | 0 |

The VLA's weights are unchanged; only the initial noise of its flow-matching
action head is chosen by the learned agent.

- **Full 3/3 difference:** 47/60 vs 29/60 episodes (Fisher's exact test,
  two-sided, p ≈ 0.001).
- **Per-cube difference:** +10 points. The bootstrap 95 % intervals overlap
  (native 72.8–84.4 %, champion 82.8–94.4 %).
- **Where the gain comes from:** turning 2-of-3 episodes into 3-of-3. The
  stall rate, a failed first grasp, is unchanged.

## Files

| path | content |
|---|---|
| `n60/baseline_native_120ch_n30.jsonl`, `..._seed456.jsonl` | native GR00T, 30 episodes each (seeds 0 and 456) |
| `n60/confirm_r9-pixel-anneal-chain_best_s123.jsonl`, `..._s456.jsonl` | champion, 30 episodes each (seeds 123 and 456, never used in training) |
| `champion/config.json` | the champion run's full resolved configuration |
| `champion/metrics.jsonl` | its training log (losses, critic health, 10-episode evaluations) |
| `champion/ckpt_best.json` | the step its best checkpoint was taken at |
| `figures/plot_leaderboard_3of3.png`, `plot_leaderboard_1of3.png` | all 17 confirmed n = 60 results, by full 3/3 and by per-cube rate |
| `figures/plot_champion_lineage.png` | 10-episode training evaluations of the champion and its forks; note how far they swing |
| `figures/plot_critic_health.png` | critic calibration before and after the pessimism fix (round 1 vs round 2) |
| `videos/champion_rollout.mp4`, `native_rollout.mp4` | the robot's ego camera on a held-out seed, one frame per chunk |

Each episode line: `{"ep", "correct", "wrong", "fell", "chunks"}`.

## Reproducing

```bash
# native baseline (two seeds)
GPU=0 scripts/baseline.sh --mode native --episodes 30 --layout 0 --max-chunks 120 --out native_s0.jsonl
GPU=0 scripts/baseline.sh --mode native --episodes 30 --layout 0 --max-chunks 120 --seed 456 --out native_s456.jsonl

# the champion recipe: three chained runs with the same overrides
OV="env.max_chunks=120 sac.use_pixels=true sac.critic_subset_size=4 sac.q_bound=45 sac.gamma=0.97 groot.resample_unsteered=true"
GPU=0 scripts/train.sh r5-pixel 30000 $OV
GPU=0 RESUME=runs/<r5>/ckpt_best_<step>.pt RESUME_BUF=runs/<r5>/buffer_<step>.npz scripts/train.sh r8-pixel-anneal 30000 $OV
GPU=0 RESUME=runs/<r8>/ckpt_best_<step>.pt RESUME_BUF=runs/<r8>/buffer_<step>.npz scripts/train.sh r9-pixel-anneal-chain 30000 $OV

# confirm a checkpoint
GPU=0 scripts/eval.sh runs/<r9>/ckpt_best_<step>.pt --config runs/<r9>/config.json \
    --max-chunks 120 --episodes 30 --seed 123 --out confirm_s123.jsonl
```

`--resume` restores the networks, optimisers and α, but not the configuration,
so restate every override. Runs differ from seed to seed, and a chained run's
best checkpoint falls at a different step each time. Expect results in the
80–89 % per-cube band that the campaign's configurations clustered in, rather
than these exact numbers.

## Caveats

- **Selection.** The champion is the best of ~26 checkpoints confirmed on the
  same two evaluation seeds. The mean over all confirmed DSRL checkpoints is
  about 82 % per-cube.
- **One layout.** All results are cube layout 0, with no position jitter.
- **Privileged state.** The agent's observation includes simulator cube
  positions, alongside the ego-camera pixels.
- **No independent seeds.** The champion's lineage is one chain of runs, not
  repeated from scratch.
- **Simulation only.** No real-robot result.
- **No trained agent here.** The champion's actor checkpoint is not included.
