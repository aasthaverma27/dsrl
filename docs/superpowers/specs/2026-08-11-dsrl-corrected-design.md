# DSRL on frozen GR00T N1.7 — corrected design

Date: 2026-08-11

Goal: improve a frozen GR00T N1.7 checkpoint on three-cube colour-matched sorting by
running RL over its latent-noise space, without touching the VLA weights.

This spec replaces an implementation that was built from recalled knowledge of DSRL
rather than from the paper and reference code. Every value below is traced to a
source: `trial/dsrl_paper.pdf`, `trial/dsrl/` (official repo), or
`trial/sb3_dsrl/stable_baselines3/dsrl/dsrl.py` (their SB3 fork, where DSRL-NA lives).

---

## 1. What the reference implementation actually does

### 1.1 Algorithm 1, DSRL-NA (paper §4.2)

```
Update Q^A:  min E_{(s,a,r,s')~B, a'~pi_dp(s',pi^W(s'))} [ (Q^A(s,a) - r - g*Qbar^A(s',a'))^2 ]
Update Q^W:  min E_{s~B, w~N(0,I)}                       [ (Q^W(s,w) - Q^A(s,pi_dp(s,w)))^2 ]
Update pi^W: max E_{s~B}                                 [ Q^W(s, pi^W(s)) ]
```

`Q^A` is a critic over **environment actions**, trained by TD. `Q^W` is a critic over
**noise**, trained by *supervised regression* against `Q^A` — no bootstrapping. The
actor optimises `Q^W`.

Confirmed in `dsrl.py`, with two details the paper's algorithm box omits:
- the entropy term IS present in both the `Q^A` target and the actor loss
- `Q^A` and `Q^W` are paired head-by-head in the distillation loss
- noise for distillation is fresh `randn`, **unclipped**, and passed through
  `scale_action`, so `Q^W` is deliberately trained on tails the actor cannot reach
- `actor_gradient_steps` defaults to `-1`, so the actor updates every gradient step

### 1.2 Action space (`utils.py:167-170`, `env_utils.py:199-200`)

A per-coordinate **box**, not a norm constraint:

```python
noise = torch.randn(n_envs, act_steps, action_dim)
noise[noise < -action_magnitude] = -action_magnitude   # clip to +/- 1.5
noise[noise > action_magnitude] =  action_magnitude
# SAC action space: Box(low=-1.5, high=+1.5, shape=(act_steps*action_dim,))
```

Standard tanh-squashed SAC actor over that box. `dsrl_sac` clips base-policy noise;
`dsrl_na` does not clip.

### 1.3 Hyperparameters (`cfg/robomimic/dsrl_can.yaml`)

```
act_steps 4, action_dim 7      -> noise dim 28
utd 20, noise_critic_grad_steps 10, train_freq 1
layer_size 2048, num_layers 3, use_layer_norm True, activation Tanh
n_critics 2, critic_backup_combine_type min
discount 0.99, tau 0.005, actor_lr 3e-4, batch_size 256
ent_coef -1 (auto),  target_ent 0.0
init_rollout_steps 1501  (x4 envs = ~6000 steps ~ 80 episodes)
action_magnitude 1.5
env: n_envs 4, n_eval_envs 25, max_episode_steps 300, reward_offset 1
```

### 1.4 Reported results

- Real world, Franka pick-and-place, 10 demos, 0-1 sparse reward, **DSRL-SAC**:
  base 2/10 -> DSRL **9/10** in **3500 online steps (~40 episodes)**
- Real world multi-task (Bridge V2, WidowX): improvement in **100-150 episodes**
- Simulation (Robomimic, Gym): **1e6 - 1e7 steps**

The 3500-step figure is the outlier, on an easy task with a weak (20%) base policy.
Our task is harder and our base policy stronger, so the sim scale is the better guide.

---

## 2. Our deviations, and the correction

| item | reference | ours (before) | change |
|---|---|---|---|
`target_entropy` | **0.0** | +224 | **set 0.0** |
warmup / init rollouts | ~6000 steps (~80 eps) | 500 (~11 eps) | **3500 steps** |
action space | box +/-1.5, tanh actor | shell projection, unsquashed | **box + tanh, magnitude 1.5** |
warmup noise | clipped standard normal | uniform, then shell-projected | **clip(randn, +/-1.5)/1.5** |
gamma | 0.99 | 0.95 | **0.99** |
n_critics / reduction | 2 / min | 10 / subset-min-2 | **2 / min** |
activation | Tanh | ReLU | **Tanh** |
LayerNorm | actor and critic | critic only | **both** |
utd / width / layers / batch / lr / tau | 20 / 2048 / 3 / 256 / 3e-4 / 0.005 | same | no change |
noise dim | 28 | 224 | **56 via B3, conditional on the §3.1 gates** |
algorithm | dsrl_na | dsrl_sac | dsrl_sac now, NA next phase |

### 2.1 Deliberate deviations, with reasons

**`init_alpha` = 0.01, not SB3's 1.0.** Their `|logp|` is ~19 over 28 dims; ours is
~150 over 224. At `alpha=1.0` that places a ~148 entropy bonus in every Bellman
target, amplified 100x by `gamma=0.99` — the measured `q -> 13700` pathology from
2026-08-10. The tuner raises alpha if entropy falls below target.

**Milestone rewards kept** (first_contact 0.5, lifted 1.5, over_correct_basket 3.0,
correct_placement 10.0) instead of their sparse 0-1. Raises event density from 5.6%
to ~20% of steps. Measured harmless: warmup 1.50 against a 1.71 reference.
Their `reward_offset: 1` is not adopted — it is specific to a 0-1 reward.

**Frozen-policy reference = 1.71 cubes (n=38).** Pooled over three warmup phases on
layout 0, all with 224-dim per-row noise and per-episode base_noise: 1.92 (n=13),
1.69 (n=13), 1.50 (n=12). Individual phases have SE ~0.3, so the pooled figure is the
one to compare against. A separate 8-episode measurement with no seed published at
all gave 2.00; the discrepancy is within the error of both.

**Layout pinned** (`--layout 0`) so results compare against a measured single-layout
baseline. Rotation across arrangements returns once steering is shown to work.

---

## 3. B3: shorten the executed chunk

GR00T generates 40 rows, serves 16, and their client executes all 16 before
re-querying. The reference uses `act_steps 4` — generate 4, execute 4, replan.

Change the client to execute **4 of the 16** rows and replan.

```
                  steps/hour   noise dim   decisions/episode
now                  2,250        224              44
B3 (4 rows)          9,000         56             175
reference               --          28              75
```

This is the only change that fixes both remaining structural gaps at once: it
multiplies throughput 4x and divides the noise dimension by 4.

### 3.1 Feasibility gates (no robot time, ~15 min)

1. **VLA query rate 1.25 Hz -> 5 Hz.** If server inference is ~200 ms, that is ~1 s
   of GPU per second of wall clock. Measure against the running server.
2. **Seed publish/ack loop at 5 Hz.** Today one action spans two chunks because the
   client re-reads a cached seed before we can publish; tighter timing may worsen
   this. Measure ack intervals.

If either fails, fall back to 8 rows: 2x throughput, 112 dims.

---

## 4. Our own fixes, retained

- **`correct_placement` guarded per cube and revoked if the cube leaves its basket.**
  Measured: an episode returned +40 with a final state worth 20, because placements
  are paid on transition with no once-per-cube guard and no revocation. Return must
  track the reported metric.
- **`ep_terms` / `ep_counts` written into `metrics.jsonl`.** Currently terminal-only,
  so per-term reward analysis is impossible after the fact.
- **Live critic calibration** (`cal_q`, `cal_true_return`, `cal_bias_x`, `cal_corr`)
  stays. It is the instrument that showed the critic becoming informative
  (`cal_corr` last-8 mean +0.240 at 35 episodes, against -0.041 in the prior run).

---

## 5. Run plan

```
1. current run s1-l0 finishes            -> uncorrected baseline, n~85 post-warmup
2. B3 feasibility gates                   ~15 min, no robot time
3. apply section 2 + section 4 changes
4. warmup 3500 steps                      ~95 min without B3, ~24 min with
5. train long: 20,000-50,000 steps        ~2.2-5.5 h with B3
6. paired A/B vs frozen baseline, layout 0, common random numbers
```

Run length is decided by `cal_corr`, not fixed in advance: continue while it stays
above 0.3 and episode returns are still trending up; stop at 20,000 if both have been
flat for ~50 episodes; extend toward 50,000 if either is still improving.

### 5.1 Success criteria

- **Primary:** mean cubes sorted beats the frozen reference of 1.71 (§2.1, n=38)
  with non-overlapping standard errors. Needs ~40 episodes per arm.
- **Mechanism check:** `cal_corr` sustained above 0.3.
- **Stop condition:** `cal_corr` below 0.1 after 30 post-warmup episodes with the
  corrected config means the config was not the cause, and the remaining gaps
  (partial steering of 224/5280 dims, actor blind to the camera) are.

### 5.2 Expected outcome

Plausible: 1.8-2.2 cubes against the 1.73 baseline. A 2/10 -> 9/10 style
transformation is not a reasonable expectation — that result was on a single-object
task with a 20% base policy, where most failures were systematic. Ours is three
sequential colour-matched placements from a 67%-per-cube base policy.

---

## 6. Known gaps not addressed here

1. **We steer 224 of 5280 noise dims.** Measured: our slots carry 74.6% of the
   influence on the served action, the free 5056 carry 67.1% — so ~45% of the action
   variation is uncontrolled and unobserved. The reference steers the whole vector.
2. **The actor cannot see the camera.** Their noise policy receives the same
   observation as the base policy (23-dim state for Robomimic; a CNN encoder for
   image tasks). Ours gets 117-dim privileged state — richer in some ways (exact
   world-frame cube and basket poses, contact flags) but with no image. Fixable via a
   two-phase policy-server API returning `vl_embeds`, but a 2048-dim observation
   would worsen overfitting until the sample budget is much larger.
3. **DSRL-NA not implemented.** Needs the executed action chunk in the buffer plus
   the ability to denoise at arbitrary buffer states, which for a 3.14B VLA means
   storing backbone features and running the action head locally (verified feasible
   on 2026-08-11). Their real-world results used `dsrl_sac`, so this is an upgrade.
