from dataclasses import dataclass, field

@dataclass(frozen=True)
class GrootConfig:
    action_horizon: int = 40
    max_action_dim: int = 132
    max_state_dim: int = 132
    num_inference_timesteps: int = 4
    backbone_embedding_dim: int = 2048
    input_embedding_dim: int = 1536
    hidden_size: int = 1024
    state_history_length: int = 1
    max_num_embodiments: int = 32
    # Served chunk is 16 steps (action delta_indices [0..15]), not the 40 the
    # model generates internally. Seed stays (40, max_action_dim); only 16 execute.
    served_action_horizon: int = 16
    live_action_dim: int = 35
    policy_port: int = 5560

    state_groups: tuple[tuple[str, int], ...] = (
        ("left_leg", 6), ("right_leg", 6), ("waist", 3), ("left_arm", 7),
        ("right_arm", 7), ("left_hand", 7), ("right_hand", 7),
    )
    action_groups: tuple[tuple[str, int], ...] = (
        ("left_arm", 7), ("right_arm", 7), ("left_hand", 7), ("right_hand", 7),
        ("waist", 3), ("base_height_command", 1), ("navigate_command", 3),
    )
    # Verified constant in live inference and across 10k demo frames: left_hand
    # exactly 0, navigate_command 0, base_height pinned 0.740. Legs are absent
    # from the action entirely and owned by the WBC.
    steerable_groups: tuple[str, ...] = ("right_arm", "right_hand")

    # Shell radius multiplier, not a raw gain: _build_seed rescales the actor's
    # output to norm sqrt(224) * noise_scale. 1.0 puts steered seeds exactly on the
    # N(0,1) shell the model denoises from. The old 3.0 was a plain multiplier on a
    # tanh box, which pushed seed norm to ~26 against an expected ~15.
    noise_scale: float = 1.0

    # Share one 14-value vector across all 16 chunk rows instead of steering each
    # row independently: 224 knobs -> 14. Measured justification: resampling only
    # our 224 slots moves the served action 74.6% as much as resampling all 5280,
    # so these columns hold the leverage; and the chunk's across-row spread barely
    # changes under a shared vector (0.707 vs 0.732) because that spread comes from
    # the DiT's positional structure, not from row-to-row noise variation.
    # REFUTED on the robot. Gate run gate14 (11 warmup episodes, layout 0) scored
    # 0.82 cubes against 1.69 and 1.92 for full-rank warmup on the same layout, with
    # 0 perfect episodes where full-rank had 5/13. Forcing all 16 steps to share one
    # vector makes the deviation coherent across the whole 0.8 s chunk instead of
    # averaging along it, which shows up as sustained wrist swings that jump at each
    # chunk boundary. The offline checks missed this: across-row spread of the OUTPUT
    # barely moved (0.707 vs 0.732), but nobody measured the spread of the chunk MEAN,
    # which is the quantity that actually changed.
    # A smooth temporal basis (constant + ramp = 28 dims) is the untested middle
    # ground: 8x fewer dims without forcing the 16 steps to be identical.
    share_across_chunk: bool = False

    # Hold the 5056 unsteered slots fixed for the whole run instead of redrawing
    # them each episode. Motivation was sound -- they carry ~45% of the action
    # variation, are unobserved, and so act as a hidden variable the critic cannot
    # account for. But it locks the run to ONE draw, and an unlucky draw handicaps
    # every episode. Suspected cause of both failed gates: 1.17 (224 dims, frozen)
    # and 0.82 (14 dims, frozen) against 1.80 pooled for per-episode redraw.
    # Off until a control run with per-episode redraw reproduces ~1.8.
    freeze_base_noise: bool = False

@dataclass(frozen=True)
class RewardConfig:
    correct_placement: float = 10.0
    wrong_target: float = -10.0
    dropped: float = -3.0
    disturb: float = -2.0
    target_tipped: float = -1.0
    grasp_slip: float = -1.0
    fell: float = -10.0
    # Off: |joint_acc| over 43 joints is ~330 even when near-still, so -0.01 charged
    # ~-3.3/step (~-270/episode) against +30 for a perfect sort, making "hold still"
    # the optimal policy. Re-enable only with a coefficient measured from real runs.
    jerk_coeff: float = 0.0
    # Off by default: the real stack places 1.33/3 cubes unaided, so placement and
    # the event penalties carry enough signal on their own. Distance shaping was
    # only needed to paper over a broken in-process env.
    approach_coeff: float = 0.0
    approach_cap: float = 1.0
    settle_speed: float = 0.01
    settle_steps: int = 10
    tip_angle_deg: float = 30.0
    fall_height: float = 0.2

    # Milestones, paid once per cube per episode. Reward density was 5.6% of steps
    # (~3 events/episode), which is what let the critic drift to a 22x overestimate
    # with no ability to rank actions. These raise it to ~12 events/episode.
    # Deliberately a ladder summing to 5.0 against correct_placement's 10.0, so
    # completing a placement always dominates stopping partway. Once-per-cube, so
    # none of them can be farmed by repeating a motion.
    # NOTE: distinct from the shaping terms zeroed above. jerk_coeff was continuous
    # and swamped the task (-270/episode vs a +30 ceiling); these are one-time
    # payments on irreversible progress.
    first_contact: float = 0.5
    lifted: float = 1.5
    over_correct_basket: float = 3.0
    # Below table_z minus this, a cube is on the floor and unrecoverable. Read by
    # RewardFn (which computes the off_table flag), used by StackEnv to end an
    # episode that has nothing left to earn. Generous, so a cube bouncing on the
    # table surface is never mistaken for one that is gone.
    off_table_drop: float = 0.15

@dataclass(frozen=True)
class EnvConfig:
    execute_steps: int = 16
    # 82 chunks x 16 steps / 20 Hz = 65.6 s. Demos run 36-53 s, so this leaves
    # margin for a slow-but-succeeding episode before truncation.
    max_chunks: int = 82
    n_cubes: int = 3
    preplaced_choices: tuple[int, ...] = (0, 1, 2)
    control_hz: int = 20
    cube_xy_jitter: float = 0.0
    scene_xml: str = (
        "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
    )
    camera_keys: tuple[str, ...] = ("ego_view",)
    # Stack env timing. reset_settle lets their WBC re-stabilise after the sim
    # clock rewinds; ack_timeout catches a stalled client; max_secs is the wall
    # clock twin of max_chunks, since the stack runs in real time.
    reset_settle_secs: float = 8.0
    ack_timeout_secs: float = 5.0
    max_secs: float = 70.0
    # End an episode once every cube is either placed or knocked off the table, so
    # a failed episode stops instead of burning the remaining 50 s. Buys roughly
    # 1.5x more episodes per hour, and episodes/hour is the price of all information
    # here. The threshold is deliberately far below the table (0.795 - 0.15) so a
    # cube bouncing on the surface is never mistaken for one that is gone.
    early_terminate: bool = True
    instruction: str = (
        "observe the entire scene first. identify the red cube, blue cube, and "
        "green cube. then pick up each cube and place it into the basket with "
        "the matching color. place the red cube in the red basket, the blue cube "
        "in the blue basket, and the green cube in the green basket. ensure "
        "every cube is placed in the correct colored basket."
    )

@dataclass(frozen=True)
class SacConfig:
    # 0.99 implies a ~100-step horizon against 24-45 step episodes, so the critic
    # bootstraps past where the episode exists. 0.95 gives ~20 steps (32 s), about
    # one pick-and-place cycle, which is the right credit scope. It also divides the
    # entropy bonus accumulating in the Bellman target by 5 (that constant reached
    # ~300 against a +30 return ceiling).
    gamma: float = 0.95
    tau: float = 0.005
    # Back to the official DSRL default of 20. It was cut to 2 after q reached 674
    # against a +30 ceiling, which we later traced to the entropy bonus accumulating
    # in the Bellman target (alpha 1.0 x |logp| 150, amplified 100x by gamma 0.99) --
    # not to overestimation. That bug is fixed, so the reason for the cut is gone.
    # This is the main lever on our binding constraint: 6000 robot steps buys 120,000
    # updates instead of 12,000, converting cheap GPU time into sample efficiency.
    # Measured at 0.38 s for 20 updates (2048x3, 10 heads, batch 256) against a 1.6 s
    # step budget.
    utd: int = 20
    lr: float = 3e-4
    batch_size: int = 256
    # The official repo recommends a 3-layer MLP of width 2048; we were at 128, i.e.
    # 16x narrower. Held-out regression on our real buffer showed the critic in the
    # overfitting regime, but under-capacity is the other failure mode and their
    # tuned setting is the better prior. Latency measured, see utd above.
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    noise_scale: float = 1.0
    num_critics: int = 10
    # "mean" applies no pessimism, so the actor exploits critic error unopposed:
    # q reached 674 against a +30 return ceiling. REDQ's subset-min keeps the heads
    # disagreeing, which is what actually bounds overestimation. See sac._reduce.
    critic_reduction: str = "subset"
    critic_subset_size: int = 2
    # Target entropy PER DIM, positive = keep the policy stochastic. An unsquashed
    # Gaussian at std=1 -- the base policy itself -- sits at 1.419/dim, so 1.0/dim
    # lets the policy sharpen to std ~0.66 and no further. That floor is what stops
    # collapse to one fixed seed per state, which is measurably far below baseline.
    # A NEGATIVE target asks for a concentrated policy and is what small control
    # tasks use; here it would drive exactly that collapse.
    target_entropy_per_dim: float = 1.0
    # alpha starts at 1.0 in textbook SAC, but |logp| ~ 318 across 224 dims, so the
    # entropy term dominated the reward 30x and gamma=0.99 amplified that constant
    # 100x in the Bellman target, leaving the critic fitting a large constant
    # instead of resolving actions. 0.01 puts the entropy term on the reward's
    # scale; the tuner raises it if entropy falls below target.
    init_alpha: float = 0.01
    use_noise_aliasing: bool = False
    restrict_to_prefix: bool = True
    replay_capacity: int = 100_000
    warmup_transitions: int = 500

@dataclass(frozen=True)
class DsrlConfig:
    groot: GrootConfig = field(default_factory=GrootConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    sac: SacConfig = field(default_factory=SacConfig)
