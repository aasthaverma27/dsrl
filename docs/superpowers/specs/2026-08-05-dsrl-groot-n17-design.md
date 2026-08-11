# DSRL Steering of a Frozen GR00T N1.7 — Design Spec

**Date:** 2026-08-05
**Status:** design approved, prerequisites outstanding
**Supersedes/refines:** `DSRL_GROOT_N17_PLAN.md` (the original concept doc)

---

## 1. Goal

Improve the reliability of colored-cube sorting on a Unitree G1 in MuJoCo **without changing
GR00T N1.7's weights**, by training a small SAC agent whose action *is* the flow-matching initial
noise. Success is measured as end-to-end 3-cube sort rate against the frozen policy's own
random-noise baseline, on identical held-out layout seeds.

**Method:** DSRL (arXiv 2506.15799), noise-aliased variant.

**Non-goal:** real-robot deployment. The actor is nonetheless restricted to deploy-available
observations so transfer stays possible later.

---

## 2. What is already established

Verified by inspection of the actual repos, not assumed:

| Fact | Evidence |
|---|---|
| Seed is sampled in exactly one place | `gr00t/model/gr00t_n1d7/gr00t_n1d7.py:349` |
| `options` dict is plumbed end-to-end and unused | `Gr00tPolicy.get_action(obs, options)` → `model.get_action` → `get_action_with_features(..., options=options)`; docstring says "currently unused" |
| Backbone pass is separable from the flow ODE | `_encode_features()` (line ~289) is distinct from `get_action_with_features()` (line 326), which takes features as arguments |
| Sim is steppable in-process | `gear_sonic/utils/mujoco_sim/base_sim.py` — `sim_step()` calls `mujoco.mj_step` directly |
| Reward / privileged-obs / reset are unimplemented stubs | same file: `update_reward` sets `last_reward = 0`; `get_privileged_obs` returns `{}`; `reset` is a bare `mj_resetData` |
| The shipped VLA runner cannot be used for DSRL | `gear_sonic/scripts/run_vla_inference.py` drives GR00T via `PolicyClient`/`PolicyServer` over ZMQ REQ/REP — the seed is server-side and unreachable |
| Concrete timing constants | same file: `action_horizon = 40`, `action_publish_rate = 50` Hz, inference `rate = 1/0.4` → 2.5 Hz |
| `gear_sonic/trl/` is not reusable | on-policy PPO for WBC motion imitation (`ppo_im_phc.yaml`), not chunk-level off-policy SAC |

### 2.1 Environment layout (pinned)

```
/mnt/ssd2/Isaac-GR00T                 @ b995540   ← receives the 4-line seed patch
/mnt/ssd2/vla_for_simulation_testing  @ f6b05b0   ← unmodified; we subclass DefaultEnv
/mnt/ssd2/dsrl                        ← this project
```

Both clones are `--depth 1`. If the checkpoint is tied to a specific upstream commit,
`git fetch --unshallow` first. Nothing else on disk imports `/mnt/ssd2/Isaac-GR00T`, so patching it
in place is safe. A CUDA 12.8 / sm_120 venv (torch 2.7.1+cu128) must be built — this is real
Milestone 0 work.

---

## 3. Locked decisions

> **Calibration note (added after review of the authors' reference implementation,
> github.com/nakamotoo/dsrl_pi0).** The architecture, milestones, and task
> decomposition of this spec are unchanged. Several hyperparameters and one
> algorithm default were corrected to match what the authors actually ran on π0 —
> the closest published analogue to steering a large flow-matching VLA. Corrected
> values are marked **REF**. `dsrl/config.py` is the authority.

| Decision | Choice | Rationale |
|---|---|---|
| RL algorithm | **Vanilla DSRL** default; DSRL-NA behind a config flag | **REF** — `dsrl_pi0` contains no noise-aliasing at all: a single critic ensemble over `(obs, w)` with plain SAC TD. Since that is what the authors validated on π0, it is the default here and the lower-risk path. NA stays available as an experiment (§7.2). |
| Training episode | **one cube per episode**; reset pre-places 0/1/2 cubes at random | Sparse terminal reward over ≤40 chunks is tractable; 100+ chunk episodes are not. Random pre-placement preserves the late-task state distribution that per-cube resets would otherwise never visit. |
| Headline metric | full 3-cube sort rate, 3 episodes chained without reset | This is the task as stated. Per-cube episodes are a training device, not the deliverable. |
| Actor observations | pooled `backbone_features` + pooled `state_features` + proprioception | Deploy-available and **free** — already computed inside GR00T's forward pass. |
| Critic observations | actor obs **+ ~60 dims privileged truth** | Asymmetric actor-critic. Legitimate because the actor's gradient flows through the critic only w.r.t. `w`; privileged inputs are fixed conditioning. |
| Reward shape | outcome-sparse, symmetric ±10 correct/wrong; dense shaping **off** by default | Symmetry makes indiscriminate dumping strictly worse than inaction, which is what forces color matching. Dense shaping on a steering problem adds bias without information and invites hover-farming. |
| Noise bound | tanh-squashed, scaled to **±1.0** | **REF** `action_magnitude=1.0`. Bounding matters because unbounded SAC walks to ‖w‖≫1 where the flow ODE emits garbage the critic has never scored — but the reference bound is tighter than the "cover 99.7% of the Gaussian mass" reasoning would suggest (±3). ±3 is an ablation, not the default. |
| Upstream diff | exactly **one** 4-line change | Keeps the NVIDIA diff reviewable and re-appliable. Everything else is additive. |

---

## 4. Architecture

```
dsrl/
  patches/isaac_gr00t_seed.patch   the 4-line diff; idempotent; verified by test
  groot/steerable.py               SteerableGr00t: (obs_batch, noise) → (chunk, pooled_features)
  sim/scene.py                     3-cube/3-target MJCF generation + layout randomization
  sim/reward.py                    PURE functions: privileged_state → (reward, terms, done)
  sim/sort_env.py                  DefaultEnv subclass; fills the three stubs
  env/chunk_env.py                 chunk-level env: step(noise) → obs, r, done
  env/vec.py                       N worker processes + batched GR00T in the main process
  rl/sac.py                        actor, twin Q^a, Q^w, entropy temperature
  rl/replay.py                     chunk-level replay buffer
  eval/rollout.py                  baseline + eval harness (shared code path)
  train.py                         the training loop
```

**Dependency direction is strictly one-way:** `train → rl → env → {groot, sim}`. Nothing in `sim/`
knows RL exists; nothing in `rl/` knows MuJoCo exists.

**The boundary that matters:** `sim/reward.py` and `rl/sac.py` take no heavy dependencies. Reward is
pure functions over poses — unit-testable with synthetic numbers, no MuJoCo, no GPU. SAC is pure
torch against an abstract env. The two components most likely to hide silent bugs are the two we can
test first, and neither needs the checkpoint.

`env/chunk_env.py` is the only place GR00T, the WBC, and MuJoCo compose, and it owns the temporal
abstraction.

---

## 5. The seed hook

### 5.1 Why it costs nothing

`_encode_features` (the expensive VLM/cross-attention/state pass) is already separate from
`get_action_with_features` (the flow ODE), and the latter accepts features as arguments. So:

```
backbone_output = model.backbone(images, language)          # expensive VLM pass — ONCE
features        = model._encode_features(backbone_output, action_input)
obs_actor       = pool(features.backbone_features, features.state_features)   # FREE
noise           = actor(obs_actor)                          # the RL decision
chunk           = model.get_action_with_features(*features, options={"dsrl_noise": noise})
```

The actor's observation is an intermediate GR00T already computes, and the noise is consumed *after*
it. **Zero extra GR00T cost per RL decision.**

### 5.2 The patch

Replacing `gr00t_n1d7.py:349`:

```python
noise = (options or {}).get("dsrl_noise")
if noise is None:
    actions = torch.randn(size=(batch_size, self.config.action_horizon,
                                self.action_dim), dtype=vl_embeds.dtype, device=device)
else:
    actions = noise.to(device=device, dtype=vl_embeds.dtype)
```

The fallback preserves normal inference exactly, so baseline runs and upstream tests are unaffected.
No signature anywhere changes.

### 5.3 Consequences that must be handled

**RTC is disabled during DSRL.** The RTC branch (lines 358–394) overwrites
`actions[:, :rtc_overlap_steps, :]` with the previous chunk and zeroes `vel_strength` on frozen
steps — it would silently null out part of the RL action. We train with no `action` in
`action_input`. Recorded as a train/deploy mismatch to revisit only if deployment requires RTC.

**Padded action dims.** Seed shape is `(action_horizon, max_action_dim)` because
`self.action_dim = config.max_action_dim` (line 62). Padded columns still feed `action_encoder`, so
they influence the live columns' output — they are not harmlessly ignored. The actor controls only
live dims; padded columns receive a **fixed per-run random draw** (in-distribution magnitude, but
deterministic). Verified empirically in Milestone 2 against zeros and actor-controlled variants.

**Seed width ≠ executed length.** `action_horizon = 40`, but re-inference fires every 0.4 s at a
50 Hz publish rate, so only **~20 actions are executed** per decision and the tail is discarded.
`execute_steps` is therefore its own config parameter, distinct from seed width. Note the
implication: roughly half of each seed influences only actions that are thrown away. Whether to
shrink the actor's output to the executed prefix is an open empirical question (§10).

---

## 6. Scene, reward, environment

### 6.1 Scene

Preferred source is the sort-scene MJCF used to collect the demos — reward geometry must match the
geometry the policy was trained in. Closest existing asset is
`gear_sonic/data/robots/g1/g1_loco_manip.xml`: **2 colored cubes (red, green) + 2 colored target
zones**, where targets are **flat pads** (`size="0.1 0.1 0.001"`), not containers. Extending to
3 colors is a small edit.

`sim/scene.py` generates and validates the scene either way, so randomization has a schema to work
against. Containment geometry depends on which the demos used:

- **flat pads:** cube center within the pad's 2D extent, cube resting on it
- **open boxes:** cube center within the box's inner AABB

### 6.2 Settled definition

A cube is **settled** in a target when the containment test holds **and**
`|linear velocity| < 0.01 m/s` for 10 consecutive `sim_step`s.

The velocity clause is load-bearing: without it a cube tumbling *through* the target region scores,
and the picker learns to throw.

### 6.3 Reward

`sim/reward.py` signature: `(privileged_state, prev_state, cfg) → (reward, terms_dict, done_flags)`.
No MuJoCo, no torch.

| Term | Weight | Fires when |
|---|---|---|
| `correct_placement` | **+10**, terminal | active cube settled in the color-matching target |
| `wrong_target` | **−10**, terminal | active cube settled in a mismatched target |
| `dropped` | −3, terminal | active cube at rest outside all targets, ungrasped, 10 steps |
| `disturb` | −2 per event | an already-correct cube knocked out of its target |
| `target_tipped` | −1 per event | receptacle up-axis deviates > 30° (boxes only) |
| `grasp_slip` | −1, non-terminal | cube lost from grip mid-carry; recovery still allowed |
| `fell` | −10, terminal | robot pelvis height < 0.2 m |
| `jerk` | −0.01·‖Δchunk‖ | between consecutive chunks |
| `approach` | **disabled by default**; +0.1·Δdistance, capped at +1 cumulative | active cube nearing its target |

**Density.** Informative reward is one terminal number per episode. This is sparse in time but
**dense in information** — every episode ends in a labeled outcome (+10 / −10 / −3 / fall); only
timeouts return ~0. Combined with ~20× temporal densification from chunking (`execute_steps`, §5.3)
and a base policy that already succeeds sometimes, this needs no shaping. `approach` is enabled only
if the Milestone 1 baseline is near zero, and then measured with/without rather than assumed.

Note that `jerk` is dense but is a regularizer, not shaping: it carries no task information and is
two orders of magnitude below the terminal terms. "Shaping off by default" refers to `approach`.

### 6.4 Environment

- **Episode:** sort the **active cube**. Cap 40 chunks.
- **Reset:** random color permutation, cube/target XY within table bounds, and a random count of
  already-correctly-placed cubes (0/1/2). Optionally seeded from demo states for realism.
- **Termination:** active cube settled (either polarity) · dropped · robot fell · timeout.
- **Actor obs:** mean-pooled `backbone_features`, pooled `state_features`, proprioception. Active
  cube and target identity reach the actor via the language instruction already encoded in those
  features.
- **Critic obs:** the above plus 3 cube poses, 3 target poses, color assignment, active index,
  placed mask, gripper poses (~60 dims).

**Upstream bug to work around:** `check_fall()` (`base_sim.py:508`) calls `self.reset()` *inside
itself*. Convenient for teleop, corrupting for RL — it resets mid-step and poisons the transition.
`SortEnv` overrides it to report the fall and let `chunk_env` own all resets.

---

## 7. The DSRL-NA core

Three networks, and the **bootstrap deliberately routed through `Q^w`**:

```
target   = r + γ(1−d)·( min Q^w_target(s'_priv, w') − α·log π(w'|s'_actor) ),  w' ~ π(·|s'_actor)
Q^a loss = MSE( Q^a(s_priv, a_stored),  target )              ← TD, on the STORED chunk
Q^w loss = MSE( Q^w(s_priv, w_stored),  stopgrad·min Q^a(s_priv, a_stored) )   ← distillation
actor    = maximize  min Q^w(s_priv, w_π) − α·log π(w_π|s_actor),             w_π ~ π(·|s_actor)
```

`Q^a` and `Q^w` are both twin; target networks are kept on `Q^w` only, since it is the only network
bootstrapped from.

**Why this needs no GR00T forward passes during updates.** The naive arrangement would need
`a' = g(s', w')` to bootstrap `Q^a`, which is a base-policy call per update. Routing the bootstrap
through `Q^w` avoids it: `Q^w` consumes noise directly, so the value at `s'` is available without
mapping `w'` through the flow ODE. Meanwhile `Q^a` is only ever evaluated at `(s, a_stored)` — an
action already in the buffer. Every quantity in the four lines above comes from stored tuples or from
networks that accept noise, so the update path never touches GR00T.

The division of labor is the point of NA: TD bootstrapping is anchored by `Q^a`, which sees a smooth,
well-conditioned action space, while the awkward noise→action composition is absorbed by supervised
regression rather than by bootstrapping. Coverage in noise space comes from entropy-driven
exploration; `Q^w` must generalize to unseen `w` the way any network does — the core algorithmic risk
(§9.2).

Actor: tanh-squashed Gaussian scaled to ±3, MLP 3×512 with LayerNorm.

| Hyperparameter | Value | Rationale |
|---|---|---|
| γ | 0.99 | effective horizon ~100 vs. the 40-chunk cap |
| τ | 0.005 | standard soft update |
| **UTD ratio** | **8** | env steps cost a GR00T pass; gradient steps are nearly free. The main sample-efficiency lever. |
| Critic LayerNorm | on | required for high UTD not to diverge (RLPD) |
| Entropy target | −dim(w) | automatic temperature tuning |
| Replay | ~3–4 GB @ 100k transitions | obs + next_obs + noise + chunk + reward + done ≈ 30–40 KB each in fp32; fp16 storage halves it |

### 7.1 Action-space dimensionality — a live concern

Confirmed from `gr00t/configs/model/gr00t_n1d7.py`: `action_horizon = 40`, **`max_action_dim = 132`**,
`num_inference_timesteps = 4`.

The raw seed is therefore `40 × 132 = ` **5280 dimensions** — far beyond what SAC can handle as an
action space. Masking is not an optimization here; it is mandatory. The G1's live action dim is
roughly 23–29, so the overwhelming majority of the tensor is padding.

Three levers, in preference order, each strictly narrowing the space:

1. **Mask to live action dims** — `40 × ~23 ≈ 920`. Mandatory. Padded columns receive the fixed
   per-run draw (§5.3). Requires prerequisite #5 to know the live width exactly.
2. **Restrict to the executed prefix** — `execute_steps × live_dim ≈ 20 × 23 = 460`. The discarded
   tail never reaches the robot, so steering it is largely wasted capacity. Must be validated rather
   than assumed, because the tail still participates in the flow ODE and so does influence the
   executed prefix.
3. **Temporal structure in the actor head** — predict low-rank or basis coefficients over the
   horizon instead of 460 independent scalars. Held in reserve for if 1+2 remain too large.

Levers 1 and 2 are implemented and measured in Milestone 2, before the training loop is built.

`num_inference_timesteps = 4` is a welcome finding: the flow ODE is only 4 DiT passes, so the VLM
backbone dominates per-decision cost. This reinforces §5.1 — the actor rides along inside a forward
pass whose expensive part has already happened.

**Warm start.** Milestone 1's baseline rollouts record their noise, making them valid off-policy
transitions. The baseline measurement therefore **doubles as offline pretraining data** for `Q^a`
(RLPD-style offline/online mix). The number needed for the go/no-go gate is also the free
pretraining set.

**Throughput.** N worker processes each own a `SortEnv` + WBC on CPU; the main process batches all
pending GR00T calls into one GPU forward. DDS/ZMQ and the real-time rate limiter are bypassed —
`DefaultEnv.sim_step()` is called directly, so sim runs at CPU speed, not wall-clock. N is tuned to
saturate the GPU batch after measuring chunks/sec in Milestone 1.

**Eval.** Every K updates: deterministic rollouts (actor mean, no exploration noise), 3 episodes
chained without reset, reported against the Milestone 1 baseline on the same held-out layout seeds.

---

## 8. Milestones and gates

| # | Deliverable | Gate |
|---|---|---|
| 0 | CUDA 12.8 / sm_120 venv; checkpoint loads; sim runs headless with cameras; WBC closes the loop | sim + WBC + GR00T all instantiate |
| 1 | Closed-loop runner with **default random noise**; baseline 3-cube sort rate over randomized layouts; chunks/sec measured; rollouts recorded with noise | **If baseline ≈ 0%, stop.** DSRL cannot steer an absent skill. |
| 2 | Seed patch + determinism and effectiveness tests; padded-dim A/B | same `(s,w)` → same chunk; **different `w` → different chunk** |
| 3 | Scene, reward, `SortEnv`, `chunk_env`, all unit-tested | scripted placement scores +10 and terminates |
| 4 | DSRL-NA + replay + training loop; eval vs. baseline | eval sort rate > Milestone 1 baseline |

---

## 9. Test strategy and risks

### 9.1 Testable before the checkpoint arrives — deliberately most of it

| Component | Test | Needs GR00T? |
|---|---|---|
| `sim/reward.py` | per-term unit tests with synthetic poses; boundaries — cube on the lip, cube tumbling through, cube at rest just outside | No |
| `rl/sac.py` | **toy DSRL problem**: a fake frozen base policy (fixed random nonlinear noise→action map) on a simple continuous env where only some noises earn reward. DSRL-NA must find them. Validates the NA mechanism standalone. | No |
| `sim/sort_env.py` | scripted-placement smoke test: teleport a cube into the matching target, assert +10 and terminal | No |
| `env/chunk_env.py` | `step()` advances exactly `execute_steps` sim steps; reward aggregation; seeded-reset determinism | Stub |
| patch | idempotency; unpatched behavior preserved when `dsrl_noise` is absent | Any weights |

**The single most important test:** different injected noise must produce different action chunks,
and identical noise identical chunks. If `options` fails to plumb through the real `Gr00tPolicy`
path, everything still runs — SAC trains against an input with no effect, we observe zero
improvement, and we blame the RL for weeks. Written first in Milestone 2.

Determinism is asserted with a tolerance, not exact equality: `vl_embeds.dtype` is bf16 and
kernel-level nondeterminism is possible. Not fatal if it appears — it makes the base policy mildly
stochastic, which DSRL tolerates.

### 9.2 Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Baseline ≈ 0% → nothing to steer | **fatal** | Milestone 1 gate precedes all RL code |
| Our scene ≠ demo scene → baseline meaningless | **high** | obtain the MJCF; validate by replaying a demo and confirming GR00T succeeds in *our* scene |
| WBC missing → no closed loop | **high** | resolve prerequisite #2 below |
| `Q^w` fails to generalize to unseen noise | high | vanilla DSRL behind the config flag |
| Throughput too low for SAC | medium | measure chunks/sec in Milestone 1 before fixing N; UTD=8 buys sample efficiency |
| Padded-dim handling wrong | medium | A/B in Milestone 2 |
| Raw seed is 5280-dim (`40 × max_action_dim 132`) — unusable as a SAC action space | **high** | live-dim masking is mandatory (→~920), executed-prefix restriction (→~460), basis head in reserve (§7.1) |
| Noise bound ±3 mis-set | low | config; ablate |
| RTC train/deploy mismatch | low | documented; only matters if deployment needs RTC |

---

## 10. Prerequisites and open items

### Blocking

1. **GR00T N1.7 checkpoint** — on another system, to be transferred.
2. **The WBC that closes the loop in sim.** *Resolved in principle — lives in a separate repo the
   user is providing.* `deploy-wbc-on-robot/` in the sim repo is an empty directory with no
   `.gitmodules`, so the control loop that `run_vla_inference.py` publishes latent motion tokens to
   is not present in what we have. Until that repo lands, nothing converts a GR00T chunk into joint
   commands, so Milestone 1 cannot run. On arrival, record: repo + commit, whether the WBC is a
   Python/ONNX policy (callable in-process) or a C++ binary behind ZMQ, and its expected input
   contract (latent motion token + hand joints, per the runner's "protocol v4").

   **This distinction drives the throughput design (§7).** In-process/ONNX → N parallel workers as
   specified. C++-behind-ZMQ → one WBC process per worker with separate ports, which caps N and may
   force a lower UTD; the plan sequences a spike to determine which.
3. **The sort-scene MJCF the demos used** — 3 cubes + 3 boxes, or the 2-cube flat-pad
   `g1_loco_manip` scene?

### Required before the baseline number is meaningful

4. **Exact language instruction strings** from finetuning. GR00T is language-conditioned; a
   paraphrase yields a different policy and a falsely low baseline.
5. **Data-config / embodiment tag** used — fixes live action dim and state layout, which cannot be
   inferred because `action_dim` is the padded `max_action_dim`.
6. **Demo dataset** — realistic reset states, and the means of validating that GR00T still succeeds
   in our scene.

### Nice to have

7. Confirmation that `gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml` is the
   WBC config the demos ran with — it is the only one in the repo.

### Deferred empirical questions

- Actor output width: executed prefix (~460 dims) vs. full 40-wide seed (~920). See §7.1.
- Padded-dim strategy: fixed draw vs. zeros vs. actor-controlled.
- Is ±3 the right noise bound for this checkpoint?
- Is a curriculum needed? Decided by the Milestone 1 baseline (>20–30% → skip).

### Scope note

This spec covers the whole pipeline (hook, runner, env, trainer) at the user's explicit request,
rather than being split after a baseline exists. The consequence is that the §7 trainer design rests
on quantities Milestone 1 has not yet measured — chunks/sec, episode duration, and the baseline
success rate itself. Specifically, **UTD ratio, worker count N, and replay size are provisional**;
they are sized from measurements taken in Milestone 1 and should be expected to change. The §5–§6
design (hook, scene, reward, env) does not depend on those measurements and is stable.

---

## 11. Summary

GR00T N1.7 turns a random noise tensor into an action chunk by integrating a flow ODE; the noise
silently selects which of its many behaviors emerges. We freeze GR00T entirely and train a SAC
actor whose action *is* that noise, so good behavior is chosen deliberately instead of by luck. The
hook is four lines reading `options["dsrl_noise"]`, and because GR00T's backbone pass is already
separate from its flow ODE, the actor's observation and decision cost no extra compute. The sim is
driven in-process through `DefaultEnv.sim_step()`, bypassing DDS/ZMQ and real-time pacing. Reward is
outcome-sparse and symmetric (±10 correct vs. wrong target) so color matching is forced rather than
merely encouraged. Learning uses DSRL-NA — TD in smooth action space, regression for the noise-space
critic — which needs no GR00T calls in the update path. The Milestone 1 baseline is simultaneously
the go/no-go gate and the offline warm-start data.
