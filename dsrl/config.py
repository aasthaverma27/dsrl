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

    # PER-COORDINATE bound on the noise, matching the reference implementation's
    # `action_magnitude` (cfg/robomimic/dsrl_can.yaml: 1.5). Their action space is
    # Box(-1.5, 1.5) over act_steps*action_dim with a standard tanh-squashed SAC
    # actor, and their base-policy rollouts clip randn to the same range
    # (utils.py:167-170). This replaces the shell projection, which was our own
    # invention: it forced ||seed|| = sqrt(d) exactly, a constraint the reference
    # does not impose.
    noise_scale: float = 1.5

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
    # B3: the client executes this many of the served 16 rows, then re-queries.
    # 8, not the 4 originally planned. Replanning discontinuity measured 2026-08-12:
    # the jump at a chunk boundary is ~equal to the motion the chunk performs at every
    # N, so what matters is the RATE -- 0.765 units/s of discontinuity at N=4 against
    # 0.411 at N=8 and 0.237 at N=16. Visible jerkiness on the robot at N=4.
    # GR00T serves 16 rows and was deployed executing all 16; the reference tolerates
    # act_steps=4 only because their policy's horizon IS 4 (horizon_steps: act_steps).
    # 8 keeps 2x throughput and halves the noise dimension while halving the switch
    # rate. Latency is not the constraint: get_action_chunk is 73 ms against a 400 ms
    # budget at N=8.
    execute_steps: int = 8
    # One env step is 2 chunks, so at execute_steps=4 that is 400 ms and a 70 s
    # episode holds ~175 env steps. Set above that so max_secs is the binding
    # limit rather than this counter (at execute_steps=16 the old value of 82 was
    # the binding one; keeping it here would truncate B3 episodes at 33 s).
    max_chunks: int = 200
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
    # Their value (dsrl_can.yaml: discount 0.99), on 75-step episodes -- so they too
    # bootstrap past the episode length. We had cut this to 0.95 to shrink the entropy
    # bonus in the Bellman target, but that constant is controlled by init_alpha and
    # target_entropy, not by gamma, and shortening the horizon to 20 steps also
    # shortens the credit scope below one pick-and-place.
    gamma: float = 0.99
    tau: float = 0.005
    # Back to the official DSRL default of 20. It was cut to 2 after q reached 674
    # against a +30 ceiling, which we later traced to the entropy bonus accumulating
    # in the Bellman target (alpha 1.0 x |logp| 150, amplified 100x by gamma 0.99) --
    # not to overestimation. That bug is fixed, so the reason for the cut is gone.
    # This is the main lever on our binding constraint: 6000 robot steps buys 120,000
    # updates instead of 12,000, converting cheap GPU time into sample efficiency.
    # Measured at 0.38 s for 20 updates (2048x3, 10 heads, batch 256) against a 1.6 s
    # step budget.
    # Halved from their 20 because B3 shortens the env step to ~400 ms and 20 updates
    # at 2048x3 with 10 heads measured 380 ms -- only 20 ms of margin, and overrunning
    # means missing the seed publish window so the client re-reads a stale seed.
    # Total gradient work still doubles, because B3 gives 4x the env steps:
    #   today  2,250 steps/h x utd 20 = 45,000 updates/h
    #   B3     9,000 steps/h x utd 10 = 90,000 updates/h
    utd: int = 10
    lr: float = 3e-4
    batch_size: int = 256
    # The official repo recommends a 3-layer MLP of width 2048; we were at 128, i.e.
    # 16x narrower. Held-out regression on our real buffer showed the critic in the
    # overfitting regime, but under-capacity is the other failure mode and their
    # tuned setting is the better prior. Latency measured, see utd above.
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    # Tanh, not ReLU (train_dsrl.py: activation_fn=torch.nn.Tanh), and LayerNorm on
    # BOTH actor and critic (net_arch=dict(pi=..., qf=...) with post_linear_modules).
    activation: str = "tanh"
    layer_norm_actor: bool = True
    # MEASURED DEVIATION from dsrl_can.yaml (n_critics 2, backup `min`). Their action
    # space is 28-dim; ours is 224. At that width the twin-critic minimum is too little
    # pessimism and the value estimate diverges:
    #   10 heads, subset-min (run s1-l0):     q stayed 30-104, bias 21x -> 3.4x
    #    2 heads, plain min  (run v2-fidelity): q ran 87 -> 453, bias 33x, cal_corr
    #                                           last-10 fell 0.325 -> 0.085
    # Both runs matched the frozen policy on cubes, but only the ensemble stayed
    # numerically stable, so the ensemble is what we keep.
    num_critics: int = 10
    critic_reduction: str = "subset"
    critic_subset_size: int = 2
    # TOTAL target entropy is 0.0 in the reference (dsrl_can.yaml: target_ent 0.0,
    # explicitly overriding SB3's auto = -dim). Expressed per-dim it is also 0.0, so
    # the value transfers directly despite our 224 dims against their 28.
    # We previously had +1.0/dim = +224 total, which pins the actor near maximum
    # entropy -- i.e. tells it to stay almost random, which is barely steering at all.
    target_entropy_per_dim: float = 0.0
    # DELIBERATE DEVIATION from SB3's ent_coef="auto" default of 1.0. Their |logp| is
    # ~19 over 28 dims; ours is ~150 over 224. At alpha=1.0 that puts a ~148 entropy
    # bonus in every Bellman target, amplified 100x by gamma=0.99 -- the measured
    # q -> 13700 pathology. The tuner raises alpha if entropy falls below target.
    init_alpha: float = 0.01
    # Noise aliasing: two critics, Q^A over environment actions trained by TD, and
    # Q^W over noise trained by regression against Q^A. Their default (algo: dsrl_na).
    # Requires the executed action chunk in the buffer and the ability to denoise at
    # arbitrary buffer states -- see NoiseAliasConfig.
    use_noise_aliasing: bool = True
    # Their value is 10 (dsrl_can.yaml). Measured on our nets: an update is 21.6 ms and
    # a distill step ~11 ms, so utd 10 + 10 distill is 330 ms of a 400 ms B3 step (83%,
    # too tight); 5 distill steps gives 273 ms (68%). Still 1,280 pair-samples per env
    # step, which is abundant for a supervised regression.
    noise_critic_grad_steps: int = 5
    # Distill batch. Cost is flat from 64 to 256 rows (bandwidth-bound), so 32 x 8
    # pairs = 256 rows matches the update's batch at no extra cost.
    na_distill_batch: int = 32
    # Candidate (noise, action) pairs per real step, batched through the same backbone
    # pass. Measured +7 ms for 16 candidates, so the count is limited by memory rather
    # than time: 8 pairs at 200k capacity is ~630 MB in float16.
    na_pairs: int = 8
    # Append a fixed random projection of the VLA's backbone features to the
    # observation, so the actor conditions on what the base policy conditions on --
    # the reference's noise policy sees the same observation as its diffusion policy,
    # ours previously saw only privileged low-dim state and no camera.
    # A projection rather than the raw 64x2048: 64 dims against a 117-dim observation
    # is +55%, where the raw features would be 18x. Fixed and seeded, so nothing to
    # learn. Features lag one env step (~400 ms) because they are produced BY the
    # policy call we are choosing the seed for; the scene changes little in 400 ms.
    visual_features: bool = True
    visual_dim: int = 64
    # 200k, not 1M: our longest planned run is ~40k steps, and the NA buffer also
    # holds the executed action plus na_pairs (noise, action) pairs per transition.
    replay_capacity: int = 200_000
    # Matches their transition count exactly: init_rollout_steps 1501 x 4 envs = 6004.
    # Transitions, not episodes, is the right unit -- it is what the critic fits on.
    # At B3 speed (400 ms/step) this costs ~40 min and yields ~34 episodes; at the old
    # 1.6 s/step it would have been 2.7 h. Previously we ran 500 (~11 episodes).
    warmup_transitions: int = 6000

@dataclass(frozen=True)
class DsrlConfig:
    groot: GrootConfig = field(default_factory=GrootConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    sac: SacConfig = field(default_factory=SacConfig)
