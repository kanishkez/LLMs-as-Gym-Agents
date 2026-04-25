"""
Multi-Model LLM-as-RL-Agent Experiment Runner
Uses NVIDIA NIM API (free tier models) to test LLM policies
across CartPole-v1, FrozenLake-v1, and LunarLander-v3.

Usage:
    export NVIDIA_API_KEY=your_key_here
    pip install gymnasium openai numpy
    python multi_model_runner.py
"""

import os
import json
import time
import numpy as np
import gymnasium as gym
from openai import OpenAI
from datetime import datetime

# ── NVIDIA NIM client ──────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY", "YOUR_NVIDIA_API_KEY_HERE"),
)

# ── Models to test ─────────────────────────────────────────────────────────────
# Free-tier models on NVIDIA NIM — comment/uncomment as needed
MODELS = {
    "deepseek-r1":        "deepseek-ai/deepseek-r1",
    "deepseek-v3":        "deepseek-ai/deepseek-v3-0324",
    "llama-3.3-70b":      "meta/llama-3.3-70b-instruct",
    "mistral-7b":         "mistralai/mistral-7b-instruct-v0.3",
    "phi-3-mini":         "microsoft/phi-3-mini-128k-instruct",
    "gemma-2-9b":         "google/gemma-2-9b-it",
    "qwen-2.5-72b":       "qwen/qwen2.5-72b-instruct",
    "kimi-k1.5":          "moonshotai/kimi-k1.5-instruct",
}

# ── Experiment config ──────────────────────────────────────────────────────────
CONFIG = {
    "cartpole": {
        "env_id":      "CartPole-v1",
        "n_episodes":  5,
        "max_steps":   500,
    },
    "frozenlake": {
        "env_id":      "FrozenLake-v1",
        "env_kwargs":  {"is_slippery": False},
        "n_episodes":  10,
        "max_steps":   100,
    },
    "lunarlander": {
        "env_id":      "LunarLander-v3",
        "n_episodes":  5,
        "max_steps":   500,
    },
}

# ── Prompt builders ────────────────────────────────────────────────────────────

CARTPOLE_SYSTEM = """You are a controller for a CartPole balancing task.
The goal is to keep a pole upright on a cart for as long as possible.

State variables:
  cart_position    : position of the cart (centre=0, limits ±2.4)
  cart_velocity    : velocity of the cart
  pole_angle       : angle of the pole in radians (upright=0)
  pole_ang_velocity: angular velocity of the pole

Physics heuristic (use this):
  - If pole_angle > 0 (leaning right) → push RIGHT (action 1)
  - If pole_angle < 0 (leaning left)  → push LEFT  (action 0)
  - If pole_ang_velocity has the SAME sign as pole_angle, the lean is
    accelerating — act more urgently.

Actions: 0 = push LEFT, 1 = push RIGHT
Output ONLY a single integer: 0 or 1. No explanation."""

def cartpole_prompt(obs, history):
    x, xd, theta, thetad = obs
    hist_str = "\n".join(
        f"  step {i}: angle={s[2]:.4f}, act={a}, reward={r:.1f}"
        for i, (s, a, r) in enumerate(history[-10:])  # last 10 steps only
    ) or "  (none yet)"
    return (
        f"Current state:\n"
        f"  cart_position    = {x:.4f}\n"
        f"  cart_velocity    = {xd:.4f}\n"
        f"  pole_angle       = {theta:.4f} rad\n"
        f"  pole_ang_velocity= {thetad:.4f}\n\n"
        f"Recent history (last 10 steps):\n{hist_str}\n\n"
        f"Output ONLY the action integer (0 or 1)."
    )


FROZENLAKE_GRID = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG",
]
FROZENLAKE_SYSTEM = """You are navigating a 4x4 FrozenLake grid (deterministic, no slipping).

Grid layout (row 0 = top):
  Row 0: S F F F
  Row 1: F H F H
  Row 2: F F F H
  Row 3: H F F G
  S=Start(0,0)  F=Safe  H=Hole(avoid, terminates episode)  G=Goal(3,3)

Rules:
  - Stepping on H ends the episode immediately with reward 0.
  - Reaching G gives reward 1.
  - Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP

Strategy hint:
  A safe path from (0,0) to (3,3) is: DOWN, DOWN, RIGHT, RIGHT, DOWN, RIGHT.
  (row 0→1→2, col 0→1→2, row 2→3, col 2→3)
  Verify each step won't land on a Hole before acting.

Output ONLY a single integer: 0, 1, 2, or 3. No explanation."""

def frozenlake_prompt(obs, history):
    row, col = obs // 4, obs % 4
    cell = FROZENLAKE_GRID[row][col]
    hist_str = "\n".join(
        f"  step {i}: state={s}(r{s//4},c{s%4}), act={a}, reward={r}"
        for i, (s, a, r) in enumerate(history[-15:])
    ) or "  (none yet)"
    return (
        f"Current position: state={obs}, row={row}, col={col}, cell={cell}\n"
        f"Goal: reach row=3, col=3\n\n"
        f"Recent history:\n{hist_str}\n\n"
        f"Output ONLY the action integer (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP)."
    )


LUNARLANDER_SYSTEM = """You are controlling a lunar lander spacecraft.

State variables (8 values):
  x_pos      : horizontal position (target: near 0)
  y_pos      : vertical position   (target: near 0 = ground)
  x_vel      : horizontal velocity (target: near 0)
  y_vel      : vertical velocity   (target: small negative, gentle descent)
  angle      : tilt angle in radians (target: near 0 = upright)
  ang_vel    : angular velocity      (target: near 0)
  left_leg   : 1 if left leg touching ground, else 0
  right_leg  : 1 if right leg touching ground, else 0

Actions: 0=do nothing, 1=fire left engine, 2=fire main engine, 3=fire right engine

Landing strategy (reason step by step):
  1. Correct tilt first: if angle > 0.1 fire LEFT engine (1); if angle < -0.1 fire RIGHT (3)
  2. Correct horizontal drift: if x_pos > 0.2 nudge left (1); if x_pos < -0.2 nudge right (3)
  3. Control descent: if y_vel < -1.5 fire main engine (2) to slow fall
  4. When both legs touch ground: do nothing (0)
  5. Avoid firing main engine unnecessarily (costs -0.3 per fire)

Output ONLY a single integer: 0, 1, 2, or 3. No explanation."""

def lunarlander_prompt(obs, history):
    names = ["x_pos","y_pos","x_vel","y_vel","angle","ang_vel","left_leg","right_leg"]
    state_str = "\n".join(f"  {n} = {v:.4f}" for n, v in zip(names, obs))
    hist_str = "\n".join(
        f"  step {i}: act={a}, reward={r:.2f}"
        for i, (s, a, r) in enumerate(history[-10:])
    ) or "  (none yet)"
    total_r = sum(r for _, _, r in history)
    return (
        f"Current state:\n{state_str}\n\n"
        f"Recent history (last 10 steps):\n{hist_str}\n"
        f"Cumulative reward so far: {total_r:.2f}\n\n"
        f"Think step by step, then output ONLY the action integer (0-3)."
    )


# Map env name → (system prompt, prompt builder, action space size)
ENV_PROMPTS = {
    "cartpole":    (CARTPOLE_SYSTEM,    cartpole_prompt,    2),
    "frozenlake":  (FROZENLAKE_SYSTEM,  frozenlake_prompt,  4),
    "lunarlander": (LUNARLANDER_SYSTEM, lunarlander_prompt, 4),
}

# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(model_id, system_prompt, user_prompt, action_space, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=16,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            # Extract first integer found
            for token in raw.replace(",", " ").split():
                try:
                    action = int(token)
                    if 0 <= action < action_space:
                        return action, raw
                except ValueError:
                    continue
            # Fallback: random action
            print(f"  [WARN] Could not parse action from: {repr(raw)}, using random")
            return np.random.randint(action_space), raw
        except Exception as e:
            print(f"  [ERROR] API call failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return np.random.randint(action_space), "ERROR"


# ── Single episode runner ──────────────────────────────────────────────────────

def run_episode(env, model_id, system_prompt, prompt_fn, action_space, max_steps, verbose=True):
    obs, _ = env.reset()
    history = []
    total_reward = 0.0
    step_logs = []

    for step in range(max_steps):
        user_prompt = prompt_fn(obs, history)
        action, raw = call_llm(model_id, system_prompt, user_prompt, action_space)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        log = {
            "step": step,
            "obs": obs.tolist() if hasattr(obs, "tolist") else int(obs),
            "action": action,
            "reward": float(reward),
            "raw_response": raw,
        }
        step_logs.append(log)
        history.append((obs, action, reward))
        total_reward += reward

        if verbose:
            print(f"    step {step:3d} | act={action} | reward={reward:7.3f} | total={total_reward:8.2f} | raw={repr(raw[:40])}")

        obs = next_obs
        if done:
            break

    return {
        "n_steps": len(step_logs),
        "total_reward": total_reward,
        "steps": step_logs,
    }


# ── Main experiment loop ───────────────────────────────────────────────────────

def run_experiments(models_to_test=None, envs_to_test=None):
    if models_to_test is None:
        models_to_test = list(MODELS.keys())
    if envs_to_test is None:
        envs_to_test = list(CONFIG.keys())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for model_name in models_to_test:
        model_id = MODELS[model_name]
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}  ({model_id})")
        print(f"{'='*70}")
        all_results[model_name] = {}

        for env_name in envs_to_test:
            cfg = CONFIG[env_name]
            system_prompt, prompt_fn, action_space = ENV_PROMPTS[env_name]

            print(f"\n  ENV: {cfg['env_id']}  ({cfg['n_episodes']} episodes)")
            print(f"  {'-'*60}")

            env_kwargs = cfg.get("env_kwargs", {})
            env = gym.make(cfg["env_id"], **env_kwargs)

            episodes = []
            for ep in range(cfg["n_episodes"]):
                print(f"\n  -- Episode {ep+1}/{cfg['n_episodes']} --")
                result = run_episode(
                    env, model_id, system_prompt, prompt_fn,
                    action_space, cfg["max_steps"], verbose=True
                )
                result["episode"] = ep
                episodes.append(result)
                print(f"  >> Episode done: {result['n_steps']} steps, "
                      f"total_reward={result['total_reward']:.2f}")
                time.sleep(0.5)  # small delay between episodes

            env.close()

            rewards   = [e["total_reward"] for e in episodes]
            lengths   = [e["n_steps"]      for e in episodes]
            successes = sum(
                1 for e in episodes
                if (env_name == "cartpole"    and e["n_steps"] >= 499)
                or (env_name == "frozenlake"  and e["total_reward"] > 0)
                or (env_name == "lunarlander" and e["total_reward"] > 200)
            )

            summary = {
                "mean_reward":  float(np.mean(rewards)),
                "std_reward":   float(np.std(rewards)),
                "max_reward":   float(np.max(rewards)),
                "min_reward":   float(np.min(rewards)),
                "mean_length":  float(np.mean(lengths)),
                "success_rate": successes / cfg["n_episodes"],
                "n_episodes":   cfg["n_episodes"],
                "episodes":     episodes,
            }
            all_results[model_name][env_name] = summary

            print(f"\n  SUMMARY for {model_name} on {env_name}:")
            print(f"    mean_reward  = {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
            print(f"    mean_length  = {summary['mean_length']:.1f} steps")
            print(f"    success_rate = {summary['success_rate']*100:.0f}%")

        # Save after each model (in case of crash mid-run)
        fname = f"results_{timestamp}.json"
        with open(fname, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  [Saved so far → {fname}]")

    # Final save
    fname = f"results_FINAL_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE. Results saved to: {fname}")
    print_summary_table(all_results, envs_to_test)
    return all_results, fname


# ── Pretty summary table ───────────────────────────────────────────────────────

def print_summary_table(results, envs):
    print(f"\n{'='*70}")
    print("SUMMARY TABLE (mean reward per model per environment)")
    print(f"{'Model':<20}", end="")
    for e in envs:
        print(f"  {e:<18}", end="")
    print()
    print("-" * 70)
    for model_name, env_results in results.items():
        print(f"{model_name:<20}", end="")
        for e in envs:
            if e in env_results:
                r = env_results[e]["mean_reward"]
                sr = env_results[e]["success_rate"]
                print(f"  {r:7.1f} ({sr*100:.0f}%sr)  ", end="")
            else:
                print(f"  {'n/a':<18}", end="")
        print()
    print(f"{'='*70}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── CONFIGURE WHAT TO RUN HERE ─────────────────────────────────────────────
    # Pick a subset of model keys from MODELS dict above
    MODELS_TO_RUN = [
        "deepseek-v3",
        "llama-3.3-70b",
        "qwen-2.5-72b",
        "mistral-7b",
        # add more from MODELS dict as needed
    ]

    # Pick environments to test
    ENVS_TO_RUN = [
        "cartpole",
        "frozenlake",
        "lunarlander",
    ]

    results, output_file = run_experiments(
        models_to_test=MODELS_TO_RUN,
        envs_to_test=ENVS_TO_RUN,
    )
