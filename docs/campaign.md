# The training campaign: what was tried, what worked

Ten rounds of experiments (E1–E10), 46 training runs of up to 30k steps each
(about 650 GPU-hours), and 17 independent 60-episode confirmations.
The code comments cite the round (`E…`) and earlier port-era findings (`F…`)
behind each design choice. They are listed here.

**How it was run.** After the initial setup, the campaign was driven largely
by an autonomous AI coding agent working under a standing human instruction:
reach ≥ 90 % full-episode success, verify the reward before trusting any
result, log every decision. Every launch, kill, confirmation and
course-correction was logged as it happened. One rule shaped which results
exist: **no number counts unless confirmed at n = 60 on seeds never used in
training**.

## Rounds

| round | question | result |
|---|---|---|
| **E1** floors | what do untrained choices of noise score? | no-op (one frozen seed per episode) 47.8 % per-cube; random steering actions 64.4 %; native GR00T 72.2 % (90-chunk episodes). Per-chunk noise diversity alone is worth ~17 points, so the actor must not collapse to near-constant corrections |
| **E2** reward audit | is the reward correct? | replaying a known 3/3 episode paid exactly the expected 43.0, milestones in order, no spurious penalties |
| **E3** round 1 | which SAC axis matters (224 vs 56 steered dims, noise scale, update ratio, entropy)? | nothing beat the 64 % floor. Every critic pinned at the Q clamp within ~1.5k steps, unable to rank actions |
| **E4** round 2 | does critic pessimism fix it? | yes: min over a random 4-of-10 subset plus a Q bound of 45. Evaluations climbed ~30 points over round 1 at the same step count. Resampling the unsteered noise every chunk was the most reliable variant |
| **E5–E6** round 3 | why do strong training evals not confirm? | 97 % / 90 % training spikes confirmed at 72.2 %. The return–cubes correlation was 0.986 (reward fine), but **91 % of two-cube episodes were truncated at 90 chunks while still working**. The episode cap rose to 120 chunks and all baselines were re-measured (native 72.2 → 77.8 %) |
| **E7** | did DSRL beat native yet? | no: parity at n = 60. Outcomes are decided at the first grasp; a 150-chunk probe showed extra time does not rescue a failed episode |
| **E8** round 4 | train at the deployment horizon | first confirmed gain: **84.4 % per-cube / 68.3 % full 3/3** (warm-started from a mature 90-chunk policy) |
| **E9** round 5 | add vision, as in the original DSRL implementation | PixelSAC (DrQ encoder on the 84×84 ego view): **85.0 % / 63.3 %** |
| **E10** | learning-rate annealing + warm-chaining | champion **r9-pixel-anneal-chain: 88.9 % / 78.3 %, 0 wrong placements**. A dozen further forks (anneal timing, update ratio, 2nd/3rd-generation chains) landed in the same 76–89 % / 52–78 % band; none beat it on full 3/3 |

The champion's lineage:
1. `r5-pixel-g097`: a fresh PixelSAC run.
2. `r8-pixel-anneal`: resumed from r5; its best was at step 6k, 89.4 % / 71.7 %.
3. `r9-pixel-anneal-chain`: resumed from r8's best checkpoint and replay
   buffer; its best was at step 4k.

## What worked

- **Fixing the environment first.** A broken episode reset had capped native
  success at about 11 %. Fixing it alone brought native to 72 %, within ~3
  points of the reference stack ([sim_port.md](sim_port.md)).
- **Residual steering plus resampling the unsteered noise** removed a
  degenerate zero-noise floor and restored per-chunk diversity.
- **Critic pessimism** (subset-min, tight Q bound) turned round 1's
  no-learning into learning.
- **Training at the evaluation horizon** (120 chunks).
- **Vision for the actor and critic.**
- **Warm-chaining.** Every top result descends from a chained run.

## What did not

- **Best-of-K noise selection by the learned critic** (sample K actions, run
  the one with the highest mean Q): −10 points, because the argmax exploits
  critic overestimation. Using the minimum over heads was neutral.
- **Longer episodes as a fix for stalls:** stalled episodes stay stalled.
- **High update ratio** (UTD 20): about 10 points below native.
- **Trusting 10-episode evaluations.** They swing ±15–20 points.
- **Annealing as a full fix for SAC's late-run decay:** it softens the decay
  but doesn't remove it. Best checkpoints sit mid-run.

## Where it stopped

10+ independent configurations cluster at 80–85 % per-cube / 60–68 % full
3/3, with the champion at 88.9 / 78.3. The original target was 90 % full 3/3,
which was not reached. The leading explanation: noise steering can only choose
among behaviours the frozen policy already has, and it cannot teach recovery
from an off-distribution state. The remaining failures are mostly a bad first
grasp that knocks a cube out of reach. The champion's stall rate (episodes
ending with ≤ 1 cube placed) is 11.7 %, against 10.0 % for native, so the gain
comes from turning 2/3 episodes into 3/3.

## Earlier findings cited in the code

| id | finding |
|---|---|
| F17 | the episode reset that fixes the native baseline (settle, snap the base to the canonical pose, re-place cubes) |
| F18 | replacing the steered noise with the action (zero action → zero noise) scores 33.3 % vs 61.1 % for random on-distribution noise, so steering is residual |
| v1 authority sweep | before the campaign, steering only the first 4 rows (56 dims) looked best; in round 1 it underperformed full-row steering (35 % vs 63 %) |
| B3 | executing fewer than all 16 chunk rows (to get more decisions per second) degraded grasps on the reference stack, so the whole chunk is always executed |
