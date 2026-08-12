from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import torch
import tyro

from dsrl.config import DsrlConfig
from dsrl.env.stack_env import StackEnv
from dsrl.rl.sac import SAC
from dsrl.rl.sac_na import SacNA

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
    # Pin every episode to one cube arrangement. Layout variance otherwise spends
    # the sample budget on generalisation before there is any evidence seed
    # selection works at all, and it makes comparison against a measured
    # single-layout baseline impossible.
    layout: int | None = None
    # How many recent episodes the critic-calibration check averages over. Small
    # enough to reflect the current policy, large enough for a stable correlation.
    calibrate_episodes: int = 4

def main(args: TrainArgs) -> None:
    cfg = DsrlConfig()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = args.log_root / f"{stamp}_{args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] logging to {log_dir}")

    (log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    env = StackEnv(cfg, seed=args.seed)
    if cfg.sac.use_noise_aliasing:
        agent = SacNA(env.obs_dim, env.action_dim, env.env_action_dim, cfg.sac)
    else:
        agent = SAC(env.obs_dim, env.action_dim, cfg.sac)
    print(f"[INFO] obs_dim={env.obs_dim} action_dim={env.action_dim}"
          + (f" env_action_dim={env.env_action_dim}" if cfg.sac.use_noise_aliasing else ""))

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=agent.device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic_target.load_state_dict(ckpt["critic_target"])
        with torch.no_grad():
            agent.log_alpha.copy_(ckpt["log_alpha"])
        agent.actor_opt.load_state_dict(ckpt["actor_opt"])
        agent.critic_opt.load_state_dict(ckpt["critic_opt"])
        agent.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        print(f"[INFO] resumed from {args.resume}")

    def save_buffer(agent, path: Path) -> None:
        """Dump real transitions so hyperparameters can be swept offline.

        Each transition costs 1.6 s of robot time, so the collected data is the
        expensive part; replaying it on GPU is free.
        """
        n = agent.buffer.size
        np.savez_compressed(
            path,
            obs=agent.buffer.obs[:n], act=agent.buffer.act[:n],
            rew=agent.buffer.rew[:n], next_obs=agent.buffer.next_obs[:n],
            done=agent.buffer.done[:n],
        )

    def save_checkpoint(agent, path: Path, step: int) -> None:
        torch.save(
            {
                "step": step,
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "critic_target": agent.critic_target.state_dict(),
                "log_alpha": agent.log_alpha.detach(),
                "actor_opt": agent.actor_opt.state_dict(),
                "critic_opt": agent.critic_opt.state_dict(),
                "alpha_opt": agent.alpha_opt.state_dict(),
            },
            path,
        )

    def reset_env():
        if args.layout is not None:
            env.force_layout(args.layout)
        return env.reset()

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
                obs, r, done, trunc, info = env.step(agent.act(obs))
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
        with torch.no_grad():
            q = agent.critic(
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

    for step in range(args.steps):
        if step < cfg.sac.warmup_transitions:
            # The reference's collect_rollouts: sample standard normal, clip each
            # coordinate to +/-action_magnitude, and store scale_action(noise), i.e.
            # noise/action_magnitude, so the stored action lives in (-1,1).
            # Dividing here means _build_seed's *noise_scale exactly undoes it, so
            # the seed written is a clipped standard normal draw.
            mag = cfg.groot.noise_scale
            z = np.random.standard_normal(env.action_dim)
            action = (np.clip(z, -mag, mag) / mag).astype(np.float32)
        else:
            action = agent.act(obs)

        next_obs, reward, done, truncated, info = env.step(action)
        terminal = float(done and not truncated)
        if cfg.sac.use_noise_aliasing:
            pw, pa = info.get("na_pairs") or (None, None)
            agent.buffer.add(obs, action, reward, next_obs, terminal,
                             info.get("env_action"), pw, pa)
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
        if info["step_terms"]:
            paid = "  ".join(f"{k}={v:+.1f}(cube {info['events'][k]})"
                             for k, v in info["step_terms"].items())
            print(f"    >>> ep{episode:3d} step {step:5d}  {paid}   "
                  f"step_total={reward:+.1f}  episode_total={ep_return:+.1f}", flush=True)

        if step % 10 == 0:
            print(f"  [{phase}] step {step:5d} ep{episode:3d} step_reward={reward:+6.2f} "
                  f"episode_reward={ep_return:+7.2f} correct={info['correct']} "
                  f"q={metrics.get('q', 0.0):8.1f} alpha={metrics.get('alpha', 0.0):.3f} "
                  f"used={info['seed_used']}", flush=True)

        if done or truncated:
            recent_eps.append(cur_ep)
            recent_eps[:] = recent_eps[-args.calibrate_episodes:]
            cur_ep = []
            cal = {}
            if metrics and len(recent_eps) >= args.calibrate_episodes:
                cal = calibration(agent, recent_eps, cfg.sac.gamma)

            # Full per-episode reward breakdown. Ordered by magnitude so the terms
            # that actually moved the return are at the top, and the sum is checked
            # against the accumulated return so accounting errors cannot hide.
            terms = info["ep_terms"]
            counts = info["ep_counts"]
            width = max((len(k) for k in terms), default=12)
            print(f"  {'-' * 58}")
            print(f"  EPISODE {episode} REWARD BREAKDOWN   ({ep_len} steps, "
                  f"{info['elapsed']}s, correct={info['correct']}/{cfg.env.n_cubes})")
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
                 "elapsed": info["elapsed"], "stalled": info.get("stalled", False),
                 "early": info.get("ended_early", False),
                 "events": sorted(info["events"].keys()),
                 # per-term totals, so reward composition is analysable after the run
                 # rather than only visible in the terminal
                 "ep_terms": {k: round(v, 2) for k, v in info["ep_terms"].items()},
                 "ep_counts": info["ep_counts"],
                 **cal,
                 **{k: round(v, 4) for k, v in metrics.items()}})
            obs = reset_env()
            ep_return, ep_len, episode = 0.0, 0, episode + 1

        if step > 0 and step % args.eval_every == 0:
            log({"step": step, **evaluate(env, agent, args.eval_episodes, args.eval_chunks)})
            obs = reset_env()
            ep_return, ep_len = 0.0, 0

        if step > 0 and step % args.save_every == 0:
            save_checkpoint(agent, log_dir / f"ckpt_{step}.pt", step)
            save_buffer(agent, log_dir / f"buffer_{step}.npz")

    save_checkpoint(agent, log_dir / "ckpt_final.pt", args.steps)
    save_buffer(agent, log_dir / "buffer_final.npz")
    print(f"[INFO] done, checkpoints in {log_dir}")

if __name__ == "__main__":
    main(tyro.cli(TrainArgs))
