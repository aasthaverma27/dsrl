# DSRL Fine-Tuning of GR00T N1.7 for Color-Sorted Pick-and-Place

**Purpose of this document.** A self-contained plan and explainer for RL-fine-tuning a frozen
GR00T N1.7 VLA policy using **DSRL (Diffusion Steering RL)** on a Unitree G1 in simulation.
It is written to be handed to a fresh workspace/agent to implement. It covers the concept, the
exact code hook, what to clone and build, and the full RL pipeline spec (observations, action,
reward, terminations, resets, curriculum, actor-critic).

---

## 1. The task

A Unitree G1 humanoid, in a MuJoCo-based simulator, must sort **3 colored cubes into 3 matching
colored boxes** (e.g. red cube → red box, green → green, blue → blue). The robot has a high-level
VLA policy (GR00T N1.7) that already produces pick-and-place motion; a low-level whole-body
controller (WBC) executes it; the sim provides physics and camera feeds.

**Goal of this project:** make the sorting *more reliable* (higher success rate, fewer wrong-box
placements, fewer dropped/slipped cubes) **without retraining or fine-tuning GR00T itself**, by
training a small RL network that *steers* GR00T's action generation. This is DSRL.

---

## 2. Core concept: DSRL (Diffusion Steering RL)

**Reference:** DSRL — "Diffusion Steering via Reinforcement Learning", arXiv **2506.15799**
(Wagenmaker, Levine et al., CoRL 2025).

### The idea in one line
> GR00T stays 100% frozen. We train a small network that **chooses the noise seed** GR00T uses to
> generate its action, so GR00T dispenses its *good* behavior instead of a random one.

### Why this works (mental model)
GR00T N1.7 has a **flow-matching action head**. To produce an action chunk it:
1. samples an initial noise tensor `x0 ~ N(0, I)` (a `torch.randn`), then
2. deterministically integrates a learned flow ODE that turns that noise into an action chunk.

So the action is a **deterministic function of `(state, noise_seed)`**. The same state + same seed
always gives the same motion. GR00T already "contains" many possible pick-and-place behaviors —
some clean, some that drop or misplace a cube — and **the noise seed selects which one comes out.**
Normally that seed is random, which is why behavior is inconsistent.

**DSRL replaces the random seed with a learned one.** We train a small actor-critic (SAC) whose
*action is the noise seed*. GR00T does not change or improve internally; the **system** improves
because it is fed a better seed, so GR00T's good behavior is selected on purpose instead of by luck.

### What DSRL can and cannot do
- **Can:** re-weight / reliably select among behaviors GR00T *already* produces.
- **Cannot:** invent a new skill GR00T never learned. If GR00T cannot color-sort at all, DSRL has
  nothing to steer. → **Establishing GR00T's baseline success is a prerequisite (see Milestone 1).**

---

## 3. The exact hook point (where DSRL plugs in)

GR00T N1.7 lives in the **NVIDIA Isaac-GR00T** repo. The noise is sampled in exactly one place:

- **File:** `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`
- **Method:** `get_action_with_features(...)` — "Generate actions using the flow matching diffusion process."
- **The seed line (approx. line 349):**

```python
# Set initial actions as the sampled noise.
actions = torch.randn(
    size=(batch_size, self.config.action_horizon, self.action_dim),
    dtype=vl_embeds.dtype,
    device=device,
)

dt = 1.0 / self.num_inference_timesteps
...
for t in range(self.num_inference_timesteps):     # flow ODE integration
    ...
    actions = actions + dt * pred_velocity * vel_strength   # Euler step (unchanged)
```

**The DSRL modification is this one line:**

```python
# OLD (random seed, hidden):
actions = torch.randn(size=(batch_size, action_horizon, action_dim), ...)

# NEW (seed supplied by the DSRL actor):
actions = seed_from_dsrl_actor    # same shape, injected from outside
```

Everything after it — the `for t in range(num_inference_timesteps)` integration loop — stays
**byte-for-byte identical**. Nothing else in GR00T changes.

### The seed / "RL action" shape
The DSRL actor's output (its action) is a tensor of shape:

```
(action_horizon, action_dim)          # per environment
```

- `action_horizon` = number of future timesteps GR00T predicts per call (the action-chunk length).
- `action_dim` = width of one action vector GR00T outputs to the WBC.

Read these two numbers off the loaded GR00T config (`self.config.action_horizon`, `self.action_dim`).
The RL seed dimensionality is `action_horizon * action_dim`.

### Temporal abstraction (important)
One RL decision = one seed = **one whole action chunk** executed over `action_horizon` sim steps.
So the RL environment steps at the **chunk level**, not the physics level. The reward the actor sees
is aggregated (summed / terminal) over the chunk's sim steps.

---

## 4. Components and where they come from

| Component | Source | Status | Role |
|---|---|---|---|
| **Simulator** (`gear_sonic`) | `github.com/MethdAI-Modinoid/vla_for_simulation_testing` | have it | MuJoCo physics + camera feeds (ZMQ), DDS bridge, pick-and-place scene (cubes + boxes) |
| **WBC** (`gr00t_wbc`) | same repo above | have it | Whole-body control (Balance/Walk ONNX); executes GR00T's action chunk on the G1 |
| **GR00T N1.7 VLA + inference code** | `github.com/NVIDIA/Isaac-GR00T` (clone locally) | to clone | The frozen policy DSRL steers; contains the seed hook (§3) |
| **GR00T N1.7 checkpoint** | from labmate | to obtain | The trained weights loaded by `Gr00tPolicy` |
| **Closed-loop GR00T-in-sim runner** | build | new | Feed sim state+camera → GR00T → action chunk → WBC → sim |
| **Reward function** | build (in sim) | new | Score the sort outcome (§7) |
| **SAC die-picker (actor-critic)** | build | new | The only thing that trains; outputs the seed (§8) |
| **DSRL training loop** | build | new | Glue: run GR00T, step sim, collect reward, update SAC |

### Important naming clarification
The sim repo contains files named `GR00T-WholeBodyControl-*.onnx`. **That is NOT the VLA.** Those are
the low-level balance/walk controllers (the WBC). The **VLA** we steer is the separate **GR00T N1.7**
flow-matching model from the Isaac-GR00T repo. Two different "GR00T" things — keep them distinct:

```
GR00T N1.7 VLA   (Isaac-GR00T repo)  ← decides the pick-and-place; DSRL steers THIS
      │ action chunk
      ▼
GR00T-WBC (ONNX) (sim repo)          ← keeps G1 upright, executes motion
      │ joint commands
      ▼
gear_sonic sim   (sim repo)          ← physics + cameras + reward
```

### The dataset (teleop demos)
The teleop demonstrations were used to **train GR00T** (already baked into the checkpoint). **DSRL
does NOT train on that dataset** — it is online RL and learns purely from live sim rollouts + reward.
The demos are still useful (optional) for: realistic **reset states**, sanity-checking the reward,
and defining the demo layout used by an early curriculum stage.

---

## 5. Why run GR00T locally (not via the inference server)

Isaac-GR00T ships a **server/client** setup (`gr00t/eval/run_gr00t_server.py`,
`gr00t/policy/server_client.py`). The client API only exposes `observation → action` — **the noise
seed is rolled inside the server and cannot be reached.** DSRL must *control* that seed.

Therefore:
- **Server/client** is fine for **Milestone 1** (playing/testing the checkpoint, measuring baseline).
- **DSRL training requires GR00T running locally, in-process**, so the seed line (§3) is editable and
  so we avoid per-chunk network round-trips across millions of steps.

The "rebuild the Docker image for your GPU" step is about running GR00T **locally** with a
PyTorch/CUDA build matching the target GPU (developed on RTX 5090 → CUDA 12.8, sm_120). This is the
correct direction — local is what DSRL needs.

**Confirm with the model owner:** you get the **policy code + checkpoint** (editable), not just a
sealed server binary. Editing the `torch.randn` line requires the source.

---

## 6. Milestones (build order)

### Milestone 0 — Environment up
- Clone `Isaac-GR00T` locally; build/rebuild the container for the target GPU (CUDA 12.8 / sm_120).
- Obtain the GR00T N1.7 checkpoint from the labmate.
- Bring up `gear_sonic` + `gr00t_wbc` from the sim repo; confirm sim runs headless with cameras.

### Milestone 1 — GR00T plays in sim (baseline, NO DSRL yet)
- Wire the closed loop: `sim state + camera → Gr00tPolicy.get_action → action chunk → WBC → sim`.
- Run the frozen policy with the **default random seed**.
- **Measure the baseline sort success rate** across randomized cube/box layouts.
- This is the number DSRL must beat, and it decides whether a curriculum is needed (§9).
- *Gate:* if GR00T essentially never sorts correctly, stop — DSRL cannot steer a skill that isn't there.

### Milestone 2 — Expose the seed (DSRL-ready)
- Edit `get_action_with_features` so the initial `actions` noise can be **passed in** instead of
  sampled (§3). Keep a fallback to `torch.randn` when no seed is supplied (so normal inference and
  Milestone 1 still work).
- Verify: same `(state, seed)` → same action chunk (determinism check).

### Milestone 3 — Reward + RL env
- Implement the reward (§7) and the RL environment wrapper (chunk-level step, obs, termination, reset).

### Milestone 4 — DSRL training
- Add the SAC actor-critic (§8) + replay buffer + training loop.
- Train the die-picker; compare success rate vs. the Milestone 1 baseline.
- Add curriculum / domain randomization only as needed (§9, §7).

---

## 7. RL pipeline specification

DSRL is a standard RL environment over a **"noise MDP"**: the only unusual part is that the *action
is the noise seed*. Everything else (obs, reward, termination, reset, curriculum) is normal.

### 7.1 Action
- **What:** the flow-matching initial-noise seed, shape `(action_horizon, action_dim)` per env.
- **Range:** unbounded real values (it replaces a Gaussian sample). SAC actor outputs a mean+std;
  sample for exploration, use the mean for evaluation. (A common choice is to keep it near the
  N(0,I) scale the model expects; do not over-constrain.)
- One action = one chunk executed for `action_horizon` sim steps.

### 7.2 Observations (what the die-picker reads to choose a seed)
Two sources, can be combined:

| Option | Observation = | Notes |
|---|---|---|
| **A. GR00T's own features (recommended)** | the vision-language / backbone embedding GR00T already computes from the camera+state each step | Free (already computed); no separate vision encoder needed |
| **B. Explicit privileged state** | cube positions, box positions, color assignments, gripper pose, which cubes already placed | Easy to expose in sim; ideal for the **critic** |

Use **asymmetric actor-critic**: the **actor** sees what will be available at deploy time (A, or A +
modest state); the **critic** may see full privileged ground truth (B) since it is only used during
training. (Same asymmetric-value idea used in standard sim RL.)

### 7.3 Reward (score the sort outcome; keep it lean)
Because GR00T supplies the motion, grade **outcomes**, not steps. Per episode, summed over the 3 cubes:

| Term | Sign | Measures |
|---|---|---|
| `correct_placement` | + (big) | cube settled inside the **color-matching** box |
| `wrong_box_penalty` | − (big) | cube inside a **mismatched** box (**critical** — forces color matching) |
| `drop_penalty` | − | cube outside all boxes, not in hand (dropped on table/floor) |
| `grasp_slip` | − | cube lost from grip while being carried |
| `disturb_penalty` | − | a box tipped over, or other cubes displaced |
| `jerk_penalty` (optional) | − | Δaction between consecutive chunks (smoothness) |
| `approach_nudge` (optional, small) | + | current cube getting closer to its correct box |
| `grasp_nudge` (optional, small) | + | a cube successfully lifted (progress marker) |

Minimal reward:
```
reward = Σ_cubes [ W1·correct_placement − W2·wrong_box − W3·dropped ]  (− small jerk)
```
Keep optional shaping small so it never overrides the true outcome. **Success measurement is pure
geometry** (is the cube's position inside the matching box's volume) — cheap and unambiguous.

### 7.4 Termination
| Condition | Type | Effect |
|---|---|---|
| all 3 cubes in correct boxes | success | end episode, terminal reward |
| cube unrecoverable (off table / stuck) | failure | end early with penalty (optional; can also let it run) |
| max steps / chunks reached | timeout | end episode |

### 7.5 Reset
Each episode, respawn **cube positions, box positions, and color assignments** within a randomization
range. Optionally seed some resets from recorded demo states for realism. Randomization (not a reward
term) is what teaches generalization across layouts.

### 7.6 Curriculum (OPTIONAL — only if GR00T's baseline is weak)
Unlike from-scratch RL, DSRL usually does **not** need a curriculum, because GR00T already succeeds
some of the time and thus supplies positive reward signal from step one. Add a curriculum **only if**
Milestone 1 shows GR00T's success is near zero under full randomization. If needed, its job is to keep
episodes in GR00T's competence zone so the picker gets reward signal:

| Stage | Setting |
|---|---|
| Easy | 1 cube, box near demo position, fixed colors, tight spawn |
| Medium | all 3 cubes, moderate randomization |
| Hard | full position + color randomization, wide spawn, distractors |

Widen as success rate climbs. **Decision rule:** GR00T baseline high enough (say >20–30% across
random layouts) → skip curriculum; otherwise start easy and expand.

### 7.7 Domain randomization (for sim-to-real, optional first pass)
Randomize lighting, textures, cube sizes, friction, camera pose. Does not change the task; makes the
learned seeds robust to real-world variation. Needed before real-robot deployment, not for a first
sim result.

---

## 8. The SAC die-picker (actor-critic)

DSRL uses an **off-policy** RL algorithm (SAC is standard) because it is sample-efficient — GR00T
forward passes are the expensive part, so we reuse every transition via a replay buffer.

| Piece | What it is |
|---|---|
| **Actor** (die-picker) | small network: `observation → (mean, std)` over the seed; sample for exploration, mean for eval. Output shape = `action_horizon × action_dim`. ~1–2M params. |
| **Critic** (judge) | `(observation, seed) → expected return`. May use **privileged** obs (§7.2). Twin critics (SAC). |
| **Replay buffer** | stores `(obs, seed, reward, next_obs, done)` transitions at the **chunk level** |
| **Entropy term** | SAC's automatic temperature encourages seed exploration |
| **Target networks** | standard SAC target critics / soft updates |

### One training step (data flow)
```
obs (from sim, via GR00T features + state)
   │
   ▼
[SAC actor] → seed z  (shape action_horizon × action_dim)
   │
   ▼
GR00T get_action_with_features(state, z)  → action chunk   [FROZEN]
   │
   ▼
WBC → joint commands → sim steps action_horizon times      [FROZEN]
   │
   ▼
reward (§7) + next_obs + done  → store in replay buffer
   │
   ▼
sample batch → update critic (TD) and actor (maximize critic) → repeat
```
Only the SAC actor and critic have gradients. GR00T, WBC, and the sim are forward-only black boxes.

### Practical notes
- **Batch/parallel envs:** run many sim envs in parallel; batch the GR00T forward pass over envs to
  amortize its cost (the dominant compute).
- **Determinism:** GR00T must be deterministic given `(state, seed)` — verify in Milestone 2.
- **Eval:** periodically run the actor's mean seed (no exploration noise) and report sort success rate
  vs. the Milestone 1 baseline.

---

## 9. Open questions to resolve during implementation

1. **GR00T baseline success rate** (Milestone 1) — decides if a curriculum is needed (§7.6) and
   whether DSRL is viable at all (needs a non-zero base skill to steer).
2. **`action_horizon` and `action_dim`** — read from the loaded GR00T config; sets the seed shape.
3. **Episode structure** — is one episode "sort all 3 cubes" (long horizon, running reward) or "one
   cube then reset" (short)? Affects reward weighting and credit assignment.
4. **Observation choice** — confirm how to extract GR00T's backbone features for the actor, and which
   privileged state the sim can expose for the critic.
5. **Seed scale/parameterization** — whether to keep the seed near N(0,I) scale or let SAC roam;
   affects how far DSRL can push GR00T's behavior.
6. **GR00T ↔ WBC ↔ sim latency** — must run local/in-process for training throughput (§5).
7. **Access to editable GR00T source + checkpoint** (not just a sealed server) — required to edit §3.

---

## 10. References
- **DSRL paper:** arXiv 2506.15799 — Diffusion Steering via Reinforcement Learning.
- **VLA + hook:** `github.com/NVIDIA/Isaac-GR00T`
  — `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` (`get_action_with_features`, the `torch.randn` seed line).
  — `gr00t/policy/gr00t_policy.py`, `gr00t/policy/server_client.py`, `gr00t/eval/run_gr00t_server.py`.
- **Sim + WBC + teleop/data:** `github.com/MethdAI-Modinoid/vla_for_simulation_testing`
  — `gear_sonic/` (MuJoCo sim, pick-and-place scene), `gr00t_wbc/` (WBC ONNX, teleop, data exporter).

---

## 11. One-paragraph summary
GR00T N1.7 generates a pick-and-place action chunk by sampling a random noise seed and integrating a
flow ODE; the seed silently selects which of GR00T's many behaviors comes out. DSRL freezes GR00T
entirely and trains a small SAC actor-critic whose action **is** that seed, so GR00T's good behavior
is chosen on purpose. Clone Isaac-GR00T locally, load the checkpoint, and edit one line
(`actions = torch.randn(...)`) so the seed can be injected. Run GR00T in closed loop inside the
`gear_sonic` sim (state+camera → GR00T → WBC → physics), score each episode by whether cubes land in
their color-matching boxes (big + for correct, big − for wrong box, − for drops/slips), observe via
GR00T's own features plus privileged state for the critic, and train the die-picker with SAC until the
sort succeeds reliably across randomized layouts — adding a curriculum only if GR00T's raw baseline is
too weak to give reward signal.
