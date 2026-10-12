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

    # How many of the served 16 chunk rows the actor steers. 16 = the whole
    # chunk (224 dims with arm+hand). 4 = the v1 authority-sweep winner
    # (56 dims): the earliest rows carry most of the leverage because the WBC
    # interpolator reaches later waypoints from wherever the early ones put
    # the arm. Rows beyond steer_rows keep the base noise.
    steer_rows: int = 16

    # Redraw the base noise every CHUNK instead of every episode. E1/E3 round 1
    # showed frozen per-episode noise costs ~24 pts of SR (47.8% no-op floor vs
    # 72.2% native), and with steer_rows=4 the 12 unsteered rows drag the run
    # toward that floor (40% eval vs 77% for full-row steering). Resampling
    # restores native-like diversity in the slots the actor does not control,
    # at the cost of making them an unobserved variable for the critic (the v1
    # freeze_base_noise discussion, inverted).
    resample_unsteered: bool = False


    noise_scale: float = 1.5

    # Residual steering (2026-08-18, after F18): the actor's action is ADDED to
    # the base noise slots instead of replacing them. Under replace semantics a
    # zero action wrote literal 0.0 into the 224 steered slots -- off-distribution
    # for the flow model (it denoises from N(0,1)) and measured at 33.3% per-cube
    # against 61.1% for on-distribution random slots (F18). With residual
    # steering the no-op action keeps the seed exactly at the episode's base
    # draw, so an untrained/timid actor starts from the base-seed floor rather
    # than the degenerate zero-noise one.
    steer_residual: bool = True

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
    first_contact: float = 0.5
    lifted: float = 1.5
    over_correct_basket: float = 3.0

    off_table_drop: float = 0.15

@dataclass(frozen=True)
class EnvConfig:
    # 16 = execute the whole served chunk. B3 (shortening this to buy throughput and
    # fewer noise dimensions) is REFUTED on the robot:
    #   N=4  visible jerkiness
    #   N=8  warmup 1.45 (n=33) against a 1.74 reference, 0/3 rate 33% vs 12-18%,
    #        and `dropped`/`grasp_slip` firing for the first time in any run
    # GR00T serves 16 rows and was deployed executing all 16; the reference tolerates
    # act_steps=4 only because their policy's horizon IS 4 (horizon_steps: act_steps).
    # Interrupting mid-chunk breaks grasps, and no amount of RL recovers a degraded
    # base policy. Costs us the 4x throughput; that is the price of a working policy.
    execute_steps: int = 16

    max_chunks: int = 200
    n_cubes: int = 3
    preplaced_choices: tuple[int, ...] = (0, 1, 2)
    control_hz: int = 20
    cube_xy_jitter: float = 0.0
    scene_xml: str = (
        "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
    )
    camera_keys: tuple[str, ...] = ("ego_view",)
    reset_settle_secs: float = 8.0
    ack_timeout_secs: float = 5.0
    max_secs: float = 70.0
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
    # Upper bound on the entropy coefficient. target_entropy=0 gives the tuner no
    # ceiling and it reached 11.2 in final2-l0, at which point alpha*logp swamped the
    # task reward in the actor loss. Inactive while entropy is healthy.
    alpha_max: float = 1.0
    # Clamp on the critic target. A perfect episode pays 45 (3 cubes x 15) and measured
    # return-to-go averaged 4.9 with a max of 27, so |value| > 60 is provably wrong and
    # clamping cannot distort a correct estimate. This is the bound that actually stops
    # the overestimation loop; removing the entropy term alone did not.
    q_bound: float = 60.0
    # Noise aliasing: two critics, Q^A over environment actions trained by TD, and
    # Q^W over noise trained by regression against Q^A. Their default (algo: dsrl_na).
    # Requires the executed action chunk in the buffer and the ability to denoise at
    # arbitrary buffer states -- see NoiseAliasConfig.
    # OFF: reverted to plain DSRL-SAC 2026-08-12. NA ran for the first time in
    # final2-l0 and was inert -- distill_loss diverged to 1.5e7, q to -4037, alpha to
    # 11.2, and cal_corr ended negative, so Q^W carried no action-dependent signal.
    # Result 1.71 (n=84) against a 1.86 baseline. The mechanism is implemented and
    # tested; it needs the alpha*logp term removed from the Bellman target before it
    # can be judged, which is not done yet.
    use_noise_aliasing: bool = False
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
    # OFF: reverted with noise aliasing. Never tested in isolation, and it shares the
    # feedback channel with NA, so with NA off there is nothing to carry the features.
    visual_features: bool = False
    visual_dim: int = 64

    replay_capacity: int = 200_000
    warmup_transitions: int = 600

    # PixelSAC (round 5): give the actor/critic the ego view, matching the
    # reference DSRL implementation (their pi0 agent is `pixel_sac`: raw
    # pixels through a learned CNN + proprio; arXiv 2506.15799,
    # github nakamotoo/dsrl_pi0). We use the DrQ recipe at our sample scale:
    # shared conv encoder trained by the critic only (actor gets detached
    # features), random-crop augmentation, 84x84 downscale of the SAME frame
    # already rendered for GR00T. Replay capacity drops so image storage
    # stays ~2.5 GB per run.
    use_pixels: bool = False
    pixel_size: int = 84
    pixel_latent: int = 128
    pixel_replay_capacity: int = 60_000

    # Peak-anneal (round 6+): every confirmed 84-85% policy came from a checkpoint
    # 10-20k steps before the run's END, then decayed back toward native as
    # training continued (r4-warm 84.4@14k -> 77.8@30k final; r6-chain 83.9@10k
    # -> 75.6@14k). SAC keeps taking full-size gradient steps after finding a
    # good region and walks back out of it. anneal_after halves BOTH actor and
    # critic LR each time a new best eval lands after this many steps, so later
    # updates refine near the peak instead of overwriting it.
    anneal_after: int = 8_000
    anneal_factor: float = 0.5
    anneal_min_lr_frac: float = 0.05

@dataclass(frozen=True)
class DsrlConfig:
    groot: GrootConfig = field(default_factory=GrootConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    sac: SacConfig = field(default_factory=SacConfig)
