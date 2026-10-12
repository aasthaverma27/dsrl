# Method

## Task and frozen policy

A Unitree G1 humanoid (29 body DoF, two 7-DoF Dex3 hands) stands at a table in
MuJoCo and must sort three coloured cubes (red, green, blue) into the matching
baskets. The policy is **GR00T N1.7**, fine-tuned on this task
(`rahul-ai-01/red_green_blue_cube_sorting_in_sim`, `checkpoint-22000`) and kept
**completely frozen**. GR00T outputs upper-body and hand joint targets in
16-step chunks at 20 Hz. NVIDIA's whole-body controller (GEAR-WBC balance and
walk policies, from GR00T-WholeBodyControl) turns them into joint torques while
keeping the robot balanced, exactly as on the real robot.

Simulator, controller and GR00T run **in one Python process**
(`dsrl/env/sort_env.py`, `dsrl/sim/wbc_adapter.py`). The deployment stack,
by contrast, is four processes over ROS 2 / DDS / ZMQ. How the port was
validated against it is in [sim_port.md](sim_port.md).

## Diffusion steering via RL (DSRL)

GR00T's action head is a flow-matching model. It turns an initial noise tensor
`z ∈ R^{40×132}` (action horizon × padded action width) into an action chunk
in 4 denoising steps. With the weights fixed, `z` decides which of the policy's
behaviours comes out. DSRL ([Wagenmaker et al., 2025](https://arxiv.org/abs/2506.15799))
trains a small SAC agent whose **action is that noise**. RL then steers a large
frozen policy without touching its weights.

The patch in `patches/Isaac-GR00T_dsrl_noise.patch` lets the caller pass `z`
(`options["dsrl_noise"]`) instead of sampling `torch.randn`.

### What the agent controls

| choice | value | where |
|---|---|---|
| steered action groups | right arm + right hand (14 of the 35 live action dims) | `GrootConfig.steerable_groups` |
| steered rows | all 16 executed rows of the chunk | `GrootConfig.steer_rows` |
| **agent action** | **224-dimensional** (16 × 14) of the 5,280-dimensional seed | `SeededGrootPolicy.build_seed` |
| parameterisation | **residual**: `z_steered = z_base + 1.5 · clip(a, −1, 1)` | `steer_residual`, `noise_scale` |
| unsteered slots (5,056) | redrawn from N(0, 1) every chunk | `resample_unsteered` |
| decision rate | once per 16-step chunk (0.8 s) | `EnvConfig.execute_steps` |

The left arm, left hand, waist, base height and navigation commands were
constant in the fine-tuning data. Leaving them out keeps the action space at
the 224 dimensions that matter.

**Why residual, and why resample.** Replacing the steered slots outright made a
zero action write literal zeros into the seed. That is off-distribution for a
model that denoises from N(0, 1), and it scored 33 % per-cube against 61 % for
random on-distribution slots (finding F18). Adding the action to a base draw
keeps every seed on-distribution, and an untrained actor starts from the
policy's own behaviour. Freezing the unsteered slots for a whole episode cost
about 24 points (47.8 % no-op floor vs 72.2 % native, E1), because per-chunk
noise diversity is itself worth that much. Redrawing them every chunk restores
it.

### Observation and reward

- **Observation** (105-d, `CubeSortEnv.observation`): joint positions and
  velocities, cube-to-basket and cube-to-hand relative positions, pelvis height.
  The cube positions come from the simulator (privileged state).
- **Pixels** (champion only): the robot's 84×84 ego-camera view, encoded by a
  small DrQ-style CNN trained by the critic loss, concatenated with the
  observation (`dsrl/rl/sac_pixel.py`).
- **Reward** (`dsrl/rewards.py`, `RewardConfig`):
  - per-cube milestones: first contact +0.5, lifted +1.5, over the correct
    basket +3, correct placement +10;
  - penalties: wrong basket −10, drop −3, disturbing another cube −2, tipping a
    basket −1, grasp slip −1, robot falls −10.

  An audit over 1,353 training episodes found the return correlates 0.986
  with cubes placed (E6).

### SAC details

| component | value | why |
|---|---|---|
| critics | ensemble of 10, target = min over a random subset of 4 (REDQ-style) | the mean of the ensemble overestimated Q until it saturated against its clamp and stopped ranking actions (round 1 learned nothing) |
| Q clamp | 45 (≈ the undiscounted return of a perfect episode) | keeps targets physically meaningful |
| discount | γ = 0.97 | |
| networks | 3 × 2048, tanh, layer-norm on the actor | |
| update-to-data | 10 gradient steps per transition | |
| entropy | initial α = 0.01, target entropy 0 per dim | |
| learning-rate annealing | halve the LR at each new best evaluation (after 8k steps, floor 5 %) | SAC tends to optimise its way back out of a good region |
| warm-chaining | a new run resumes actor, critics, optimisers, α and replay buffer from a previous run's best | compounds gains across runs |

### Episode and evaluation protocol

- **Episodes:** cube layout 0, 120 chunks (96 s) maximum. An episode ends early
  when all three cubes are placed. The original 90-chunk cap truncated 91 % of
  two-cube episodes while they were still working (E6).
- **Training:** 10-episode evaluations every 2,000 steps, used only to pick
  checkpoints.
- **Every reported number:** an independent confirmation over 60 episodes
  (two seeds, 123 and 456, × 30 episodes) with the actor sampled
  stochastically (`scripts/eval.sh`).

  Episode outcomes are close to all-or-nothing, so a 10-episode evaluation
  routinely reads 20–30 points above the confirmed value. In this campaign,
  in-training spikes of 97 % and 100 % confirmed at 76–89 %.

The **native baseline** is the frozen policy with its own `torch.randn` noise,
on the same protocol (`scripts/baseline.sh --mode native`).
