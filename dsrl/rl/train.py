from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import torch
import tyro

from dsrl.config import DsrlConfig
from dsrl.env.sort_env import CubeSortEnv
from dsrl.rl.sac import SAC

@dataclass(frozen=True)
class TrainArgs:
    steps: int = 20_000
    seed: int = 0
    log_root: Path = Path("runs")
    run_name: str = "dsrl"
    eval_every: int = 1000
    eval_episodes: int = 6
    eval_chunks: int = 44
    save_every: int = 2000
    resume: Path | None = None
    # Replay-buffer npz (buffer_*.npz saved next to interval checkpoints) to
    # preload; with an actor-only resume this lets the fresh critic relearn
    # from the run's own data instead of restarting collection from zero.
    resume_buffer: Path | None = None
    # Pin every episode to one cube arrangement. Layout variance otherwise spends
    # the sample budget on generalisation before there is any evidence seed
    # selection works at all, and it makes comparison against a measured
    # single-layout baseline impossible.
    layout: int | None = None
    # "sort" runs simulator, whole-body controller and GR00T in one process (what
    # every reported result used). "stack" drives an external ROS/DDS deployment
    # of the same task over ZMQ (dsrl/env/stack_env.py); it needs that stack.
    env: str = "sort"
    # How many recent episodes the critic-calibration check averages over. Small
    # enough to reflect the current policy, large enough for a stable correlation.
    calibrate_episodes: int = 4
    # Per-run config overrides, e.g. --override sac.utd=20 groot.noise_scale=0.75
    # Applied on top of DsrlConfig defaults; the resolved config is what lands in
    # the run's config.json, so every run stays self-describing.
    override: tuple[str, ...] = ()


def apply_overrides(cfg: DsrlConfig, sets: tuple[str, ...]) -> DsrlConfig:
    import dataclasses

    groups: dict[str, dict] = {}
    for s in sets:
        path, val = s.split("=", 1)
        grp, field_name = path.split(".")
        cur = getattr(getattr(cfg, grp), field_name)
        if isinstance(cur, bool):
            v: object = val.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            v = int(val)
        elif isinstance(cur, float):
            v = float(val)
        else:
            v = val
        groups.setdefault(grp, {})[field_name] = v
    for grp, kv in groups.items():
        cfg = dataclasses.replace(cfg, **{grp: dataclasses.replace(getattr(cfg, grp), **kv)})
    return cfg


def main(args: TrainArgs) -> None:
    cfg = DsrlConfig()
    if args.override:
        cfg = apply_overrides(cfg, args.override)
        print(f"[INFO] config overrides: {list(args.override)}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = args.log_root / f"{stamp}_{args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] logging to {log_dir}")

    (log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    if args.env == "sort":
        env = CubeSortEnv(cfg, seed=args.seed)
    else:
        from dsrl.env.stack_env import StackEnv

        env = StackEnv(cfg, seed=args.seed)
    if cfg.sac.use_noise_aliasing:
        from dsrl.rl.sac_na import SacNA

        agent = SacNA(env.obs_dim, env.action_dim, env.env_action_dim, cfg.sac)
    elif cfg.sac.use_pixels:
        from dsrl.rl.sac_pixel import PixelSAC

        agent = PixelSAC(env.obs_dim, env.action_dim, cfg.sac)
        print(f"[INFO] PixelSAC: ego view {cfg.sac.pixel_size}px -> "
              f"{cfg.sac.pixel_latent}-d features + proprio")
    else:
        agent = SAC(env.obs_dim, env.action_dim, cfg.sac)
    print(f"[INFO] obs_dim={env.obs_dim} action_dim={env.action_dim}"
          + (f" env_action_dim={env.env_action_dim}" if cfg.sac.use_noise_aliasing else ""))

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=agent.device)
        agent.actor.load_state_dict(ckpt["actor"])
        if "encoder" in ckpt and hasattr(agent, "encoder"):
            agent.encoder.load_state_dict(ckpt["encoder"])
        with torch.no_grad():
            agent.log_alpha.copy_(ckpt["log_alpha"])
        if "critic" in ckpt:
            agent.critic.load_state_dict(ckpt["critic"])
            agent.critic_target.load_state_dict(ckpt["critic_target"])
            agent.actor_opt.load_state_dict(ckpt["actor_opt"])
            agent.critic_opt.load_state_dict(ckpt["critic_opt"])
            agent.alpha_opt.load_state_dict(ckpt["alpha_opt"])
            print(f"[INFO] resumed (full) from {args.resume}")
        else:
            # Interval checkpoints store only the actor (see save_checkpoint).
            # The critic restarts fresh and relearns from the restored buffer —
            # it is off-policy, so this costs minutes of GPU, not robot time.
            print(f"[INFO] resumed (actor-only) from {args.resume}; critic fresh")

    if args.resume_buffer is not None:
        d = np.load(args.resume_buffer)
        n = min(len(d["obs"]), agent.buffer.capacity)
        agent.buffer.obs[:n] = d["obs"][:n]
        agent.buffer.act[:n] = d["act"][:n]
        agent.buffer.rew[:n] = d["rew"][:n].reshape(n, 1)
        agent.buffer.next_obs[:n] = d["next_obs"][:n]
        agent.buffer.done[:n] = d["done"][:n].reshape(n, 1)
        agent.buffer.size = n
        agent.buffer.idx = n % agent.buffer.capacity
        print(f"[INFO] restored replay buffer: {n} transitions from {args.resume_buffer}")

    def save_buffer(agent, path: Path) -> None:
        """Dump real transitions so hyperparameters can be swept offline.

        Each transition costs 1.6 s of robot time, so the collected data is the
        expensive part; replaying it on GPU is free.
        """
        n = agent.buffer.size
        extra = {}
        if hasattr(agent.buffer, "env_act"):
            # The executed action and the (noise, action) pairs are what make offline
            # analysis of Q^A and the distillation possible; without them the dump
            # cannot reproduce anything NA-related.
            extra = {"env_act": agent.buffer.env_act[:n],
                     "pair_w": agent.buffer.pair_w[:n],
                     "pair_a": agent.buffer.pair_a[:n],
                     "has_pairs": agent.buffer.has_pairs[:n]}
        np.savez_compressed(
            path,
            obs=agent.buffer.obs[:n], act=agent.buffer.act[:n],
            rew=agent.buffer.rew[:n], next_obs=agent.buffer.next_obs[:n],
            done=agent.buffer.done[:n], **extra,
        )

    def save_checkpoint(agent, path: Path, step: int, full: bool = False) -> None:
        """Periodic saves store only what eval needs; `full` adds targets and optimisers.

        With noise aliasing there are three 10-head 2048x3 critics (Q^A, its target,
        Q^W) plus Adam state, which makes a complete checkpoint ~2.7 GB -- 14 GB over a
        10k-step run, on top of the 8 GB runs/ already holds. The actor is ~22 MB and is
        all play.py loads, so intervals save that and the final save keeps everything.
        """
        enc = ({"encoder": agent.encoder.state_dict()}
               if hasattr(agent, "encoder") else {})
        if not full:
            torch.save({"step": step, "actor": agent.actor.state_dict(),
                        "log_alpha": agent.log_alpha.detach(), **enc}, path)
            return
        torch.save(
            {
                "step": step,
                **enc,
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "critic_target": agent.critic_target.state_dict(),
                "log_alpha": agent.log_alpha.detach(),
                "actor_opt": agent.actor_opt.state_dict(),
                "critic_opt": agent.critic_opt.state_dict(),
                "alpha_opt": agent.alpha_opt.state_dict(),
                # Under noise aliasing the ACTOR is trained against Q^W, so a
                # checkpoint without it cannot reproduce or resume the policy.
                **({"critic_w": agent.critic_w.state_dict(),
                    "critic_w_opt": agent.critic_w_opt.state_dict()}
                   if hasattr(agent, "critic_w") else {}),
            },
            path,
        )

    def reset_env():
        if args.env == "sort":
            return env.reset(config_idx=args.layout)
        if args.layout is not None:
            env.force_layout(args.layout)
        return env.reset()

    is_pixel = hasattr(agent, "encoder")

    def select_action(obs):
        """agent.act, with the current ego view threaded in for PixelSAC."""
        if is_pixel:
            return agent.act(obs, pixels=env.last_pixels)
        return agent.act(obs)

    def evaluate(env, agent, episodes: int, chunks: int) -> dict:
        returns, corrects, falls = [], [], 0
        for _ in range(episodes):
            obs = reset_env()
            total = 0.0
            for _ in range(chunks):
                # Stochastic, not the mean action: the mean discards the noise, and
                # after the shell projection that is one fixed seed per state -- the
                # constant-seed regime that measures far below baseline. The policy
                # being evaluated is a distribution over seeds, so sample it.
                obs, r, done, trunc, info = env.step(select_action(obs))
                total += r
                if done or trunc:
                    break
            returns.append(total)
            corrects.append(info["correct"])
            falls += int(info["fell"])
        return {
            "eval_return": float(np.mean(returns)),
            "eval_correct": float(np.mean(corrects)),
            "eval_success_rate": float(np.mean([c > 0 for c in corrects])),
            "eval_falls": falls,
        }

    def calibration(agent, episodes: list, gamma: float) -> dict:
        """Compare what the critic predicts against what actually happened.

        This is the number that explained the whole previous run -- the critic said
        108 where the true discounted return was 4.9, and its correlation with real
        outcomes was -0.04, meaning it could not rank actions at all. We only saw it
        after three hours. `corr` is the one to watch: near 0 means the actor is
        following a gradient with no information in it, and no amount of further
        training fixes that.
        """
        obs, act, G = [], [], []
        for ep in episodes:
            g = 0.0
            for o, a, r in reversed(ep):
                g = r + gamma * g
                obs.append(o); act.append(a); G.append(g)
        if len(G) < 32:
            return {}
        # The stored action is NOISE, so this must use the noise critic. Under noise
        # aliasing `agent.critic` is Q^A over ENVIRONMENT actions (obs 181 + env 560),
        # and passing a 224-dim noise action there is a shape error -- which is exactly
        # how this crashed at step 2510.
        critic = getattr(agent, "critic_w", None) or agent.critic
        with torch.no_grad():
            q = critic(
                torch.as_tensor(np.array(obs), device=agent.device),
                torch.as_tensor(np.array(act), device=agent.device),
            ).mean(-1).cpu().numpy()
        G = np.array(G, dtype=np.float64)
        corr = float(np.corrcoef(q, G)[0, 1]) if q.std() > 1e-9 and G.std() > 1e-9 else 0.0
        return {
            "cal_true_return": round(float(G.mean()), 2),
            "cal_q": round(float(q.mean()), 2),
            "cal_bias_x": round(float(q.mean() / G.mean()), 1) if abs(G.mean()) > 1e-6 else None,
            "cal_corr": round(corr, 3),
        }

    # Best-eval checkpoint: the 97%-eval actor of r2b-resample was lost because
    # interval saves are every 5k steps and the policy drifts between them.
    # Saving on every new best eval costs ~22 MB once per improvement.
    best_eval = [-1.0]
    lr_scale = [1.0]

    def anneal_lr(agent, step: int) -> None:
        """Halve actor+critic LR on every post-peak new-best, floored at
        anneal_min_lr_frac. Every confirmed 84-85% checkpoint in this campaign
        came from mid-run (r4-warm 84.4@14k -> 77.8@30k final; r6-chain
        83.9@10k -> 75.6@14k) -- SAC keeps taking full-size steps after finding
        a good region and walks back out. This lets later updates refine near
        a peak instead of overwriting it."""
        if step < cfg.sac.anneal_after or lr_scale[0] <= cfg.sac.anneal_min_lr_frac:
            return
        lr_scale[0] = max(cfg.sac.anneal_min_lr_frac, lr_scale[0] * cfg.sac.anneal_factor)
        for opt in (agent.actor_opt, agent.critic_opt):
            for g in opt.param_groups:
                g["lr"] = cfg.sac.lr * lr_scale[0]
        print(f"[INFO] post-peak anneal: lr_scale={lr_scale[0]:.3f} at step {step}", flush=True)

    def maybe_save_best(agent, step: int, eval_correct: float) -> None:
        if eval_correct > best_eval[0]:
            best_eval[0] = eval_correct
            save_checkpoint(agent, log_dir / "ckpt_best.pt", step)
            # dated copy too: later "bests" can confirm worse at n=60 than
            # earlier ones (r4-warm's 24k best < its 14k best), so every best
            # must stay recoverable
            save_checkpoint(agent, log_dir / f"ckpt_best_{step}.pt", step)
            (log_dir / "ckpt_best.json").write_text(json.dumps(
                {"step": step, "eval_correct": eval_correct}))
            print(f"[INFO] new best eval {eval_correct:.2f} at step {step} -> ckpt_best.pt",
                  flush=True)
            anneal_lr(agent, step)

    metrics_path = log_dir / "metrics.jsonl"

    def log(record: dict) -> None:
        print("  " + "  ".join(f"{k}={v}" for k, v in record.items()), flush=True)
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    if args.layout is not None:
        print(f"[INFO] layout pinned to {args.layout} for every episode")
    obs = reset_env()
    ep_return, ep_len, episode = 0.0, 0, 0
    metrics: dict = {}
    # Last few complete episodes, kept for the calibration check. Bounded so this
    # measures the CURRENT policy rather than the whole run's history.
    cur_ep: list = []
    recent_eps: list = []
    na_seen = [0]   # list so the closure-free counter can be mutated in the loop

    for step in range(args.steps):
        if step < cfg.sac.warmup_transitions:
            # The reference's collect_rollouts: sample standard normal, clip each
            # coordinate to +/-action_magnitude, and store scale_action(noise), i.e.
            # noise/action_magnitude, so the stored action lives in (-1,1).
            # Dividing here means build_seed's *noise_scale exactly undoes it, so
            # the steer applied is a clipped standard normal draw (under residual
            # steering -- cfg.groot.steer_residual, F18/F19 -- it is ADDED to the
            # base-noise slots rather than written over them).
            mag = cfg.groot.noise_scale
            z = np.random.standard_normal(env.action_dim)
            action = (np.clip(z, -mag, mag) / mag).astype(np.float32)
        else:
            action = select_action(obs)

        px_t = env.last_pixels.copy() if is_pixel else None
        next_obs, reward, done, truncated, info = env.step(action)
        terminal = float(done and not truncated)
        if cfg.sac.use_noise_aliasing:
            pw, pa = info.get("na_pairs") or (None, None)
            agent.buffer.add(obs, action, reward, next_obs, terminal,
                             info.get("env_action"), pw, pa)
            # Fail loudly. Without pairs Q^W never trains and the actor optimises a
            # randomly-initialised critic for the whole run, while every other metric
            # looks healthy. This chain (seed -> client -> policy server -> feedback)
            # is the one part never exercised outside a mock.
            na_seen[0] += pw is not None
            if step == 40 and na_seen[0] == 0:
                print("\n" + "!" * 70
                      + "\n!! NO NA PAIRS RECEIVED in 40 steps. Q^W cannot train."
                      + "\n!! The feedback channel is not delivering. Check that the"
                      + "\n!! policy server was restarted (it needs the batched-candidate"
                      + "\n!! hook in gr00t_n1d7.py) and that port 5596 is reachable."
                      + "\n!! KILL THIS RUN -- it will produce nothing.\n"
                      + "!" * 70 + "\n", flush=True)
            elif step == 40:
                print(f"[na] pairs arriving: {na_seen[0]}/41 steps had candidates",
                      flush=True)
        elif is_pixel:
            agent.buffer.add(obs, action, reward, next_obs, terminal,
                             px=px_t, next_px=env.last_pixels)
        else:
            agent.buffer.add(obs, action, reward, next_obs, terminal)
        cur_ep.append((obs, action, reward))

        obs = next_obs
        ep_return += reward
        ep_len += 1

        if agent.buffer.size >= cfg.sac.warmup_transitions:
            for _ in range(cfg.sac.utd):
                metrics = agent.update(cfg.sac.batch_size)
            if cfg.sac.use_noise_aliasing:
                # Q^W's regression against Q^A, mirroring their separate loop after
                # the main updates (dsrl.py runs noise_critic_grad_steps here).
                metrics.update(agent.distill())

        phase = "warmup" if step < cfg.sac.warmup_transitions else "train"

        # Every step that paid anything, itemised. This is the signal the policy is
        # actually learning from, so it should be visible as it happens rather than
        # summarised into one scalar.
        if info.get("step_terms"):
            paid = "  ".join(f"{k}={v:+.1f}(cube {info['events'][k]})"
                             for k, v in info.get("step_terms", {}).items())
            print(f"    >>> ep{episode:3d} step {step:5d}  {paid}   "
                  f"step_total={reward:+.1f}  episode_total={ep_return:+.1f}", flush=True)

        if step % 10 == 0:
            print(f"  [{phase}] step {step:5d} ep{episode:3d} step_reward={reward:+6.2f} "
                  f"episode_reward={ep_return:+7.2f} correct={info['correct']} "
                  f"q={metrics.get('q', 0.0):8.1f} alpha={metrics.get('alpha', 0.0):.3f} "
                  f"used={info.get('seed_used', True)}", flush=True)

        if done or truncated:
            recent_eps.append(cur_ep)
            recent_eps[:] = recent_eps[-args.calibrate_episodes:]
            cur_ep = []
            cal = {}
            # calibration replays stored raw obs through the critic; PixelSAC's
            # critic needs image features too, so the check is skipped there
            if metrics and len(recent_eps) >= args.calibrate_episodes and not is_pixel:
                cal = calibration(agent, recent_eps, cfg.sac.gamma)

            # Full per-episode reward breakdown. Ordered by magnitude so the terms
            # that actually moved the return are at the top, and the sum is checked
            # against the accumulated return so accounting errors cannot hide.
            terms = info.get("ep_terms", {})
            counts = info.get("ep_counts", {})
            width = max((len(k) for k in terms), default=12)
            print(f"  {'-' * 58}")
            print(f"  EPISODE {episode} REWARD BREAKDOWN   ({ep_len} steps, "
                  f"{info.get('elapsed', 0)}s, correct={info['correct']}/{cfg.env.n_cubes})")
            for k, v in sorted(terms.items(), key=lambda kv: -abs(kv[1])):
                print(f"    {k:<{width}}  {v:+8.2f}   x{counts[k]:<3d}"
                      f"  ({v / counts[k]:+.2f} each)")
            if not terms:
                print("    (nothing paid this episode)")
            print(f"    {'TOTAL':<{width}}  {sum(terms.values()):+8.2f}"
                  f"   (accumulated {ep_return:+.2f})")
            print(f"  {'-' * 58}", flush=True)
            log({"step": step, "episode": episode, "episode_reward": round(ep_return, 2),
                 "correct": info["correct"], "wrong": info["wrong"],
                 "fell": info["fell"], "len": ep_len,
                 "elapsed": info.get("elapsed", 0), "stalled": info.get("stalled", False),
                 "early": info.get("ended_early", False),
                 "events": sorted(info.get("events", {}).keys()),
                 # per-term totals, so reward composition is analysable after the run
                 # rather than only visible in the terminal
                 "ep_terms": {k: round(v, 2) for k, v in info.get("ep_terms", {}).items()},
                 "ep_counts": info.get("ep_counts", {}),
                 **cal,
                 **{k: round(v, 4) for k, v in metrics.items()}})
            obs = reset_env()
            ep_return, ep_len, episode = 0.0, 0, episode + 1

        if step > 0 and step % args.eval_every == 0:
            ev = evaluate(env, agent, args.eval_episodes, args.eval_chunks)
            log({"step": step, **ev})
            maybe_save_best(agent, step, ev["eval_correct"])
            obs = reset_env()
            ep_return, ep_len = 0.0, 0

        if step > 0 and step % args.save_every == 0:
            save_checkpoint(agent, log_dir / f"ckpt_{step}.pt", step)
            save_buffer(agent, log_dir / f"buffer_{step}.npz")

    save_checkpoint(agent, log_dir / "ckpt_final.pt", args.steps, full=True)
    save_buffer(agent, log_dir / "buffer_final.npz")
    print(f"[INFO] done, checkpoints in {log_dir}")

if __name__ == "__main__":
    main(tyro.cli(TrainArgs))
