# The in-process environment, and how it was validated

The reference deployment of this task is four processes: the gear_sonic MuJoCo
simulator, the whole-body controller (in a container), the GR00T policy server
and a client, wired together with ROS 2 / DDS / ZMQ. SAC cannot sit inside that
loop efficiently at the sample counts DSRL needs. So everything runs **in one
Python process**:

| piece | in-process replacement |
|---|---|
| simulator | `gear_sonic.utils.mujoco_sim.base_sim.DefaultEnv`, stepped directly (`dsrl/env/sort_env.py`) |
| whole-body controller | the same `G1DecoupledWholeBodyPolicy` and ONNX policies, called directly on the sim clock (`dsrl/sim/wbc_adapter.py`) |
| GR00T server + client | `Gr00tPolicy` loaded in-process, with the noise hook (`dsrl/groot/policy.py`) |
| DDS / ROS | none: state and commands are passed as arrays |

That made an episode about 2× faster (33 s vs 70 s before any parallelism). It
also made the environment something a training loop can own.

## Five bugs the port had to fix

All five present as "the robot runs but behaves oddly", never as an error. The
fixes are in this repository's code.

| bug | effect | fix |
|---|---|---|
| hand joint order differs between the GR00T dataset and MuJoCo's native order | every grasp closed the wrong fingers (a 1.3 rad hand error in a one-chunk replay) | channel permutation on the hand state fed to GR00T and on the hand targets (`groot/policy.py`, `sim/wbc_adapter.py`) |
| hand PD gains ~100× too stiff | fingers drove to the angle instead of complying; the contact solver ejected the cube | the Dex3 command gains the deployed controller sends (`HAND_KP`, `HAND_KD`) |
| elastic band never applied or released | the robot hung from the support band or collapsed during settling | band applied every sub-step and released after settling (`sort_env._apply_elastic_band`, `reset`) |
| controller call contract | goals and observations reached the controller at the wrong rates | goal once per GR00T waypoint (20 Hz), observation and action at the controller's 50 Hz on the sim clock (`WbcAdapter.set_goal` / `tick`) |
| episode reset | the base settled ~4 cm and 9° off the reference pose, the episode started mid-transient, and the settle motion nudged the cubes | longer settle, base snapped to the canonical table-relative pose, cubes re-placed after the snap (`sort_env.reset`) |

The reset bug was the one that mattered for the numbers: with it, native GR00T
placed **11 %** of cubes; after the fix, **72 %**.

## Validation against the reference stack

The match was established, not assumed:

1. **Record on the reference stack:** client observations, action chunks and
   the full MuJoCo state (qpos, qvel, ctrl) per step.
2. **Test each half separately.**
   - *Policy:* identical observations and seeds give matching action chunks
     on both (differences at GPU floating-point noise level).
   - *Simulator:* replaying the recorded action chunks through the in-process
     sim reproduces the recorded joint trajectory to within 0.02 rad.

   This one-half-at-a-time replay is what exposed the finger-order bug: every
   joint group tracked within 0.1 rad except the hands, at 1.3 rad.
3. **Close the loop** and compare success rates end to end.

| regime | per-cube | full 3/3 | n |
|---|---|---|---|
| reference ROS/DDS stack | 75.6 % | 46.7 % | 30 |
| in-process port, after the reset fix | 72.2 % | — | 12 |

The ~3-point gap was within the tolerance set for the port in advance (about
10 points). The leading suspect is execution timing: the reference client acts
on ~100 ms-old observations while inference overlaps motion.

**Health check before trusting any number:** pelvis height at reset ≈ 0.746 m.
If the elastic band is still engaged it reads 0.965 m, with the cubes out of
reach.

## What this repository uses from NVIDIA's code

`external/GR00T-WholeBodyControl` (NVIDIA, pinned) provides the simulator,
controller, robot model and the Unitree SDK. Our changes to it are in
`patches/GR00T-WholeBodyControl_dsrl_sim.patch`:
- the cube-sorting scene (table, three baskets, three cubes);
- the cube placement schedule (six permutations; `--layout 0` is red–green–blue);
- the robot's ego and wrist cameras at the poses used to collect GR00T's
  fine-tuning data;
- right-hand grasp smoothing in the controller.
