import gymnasium as gym
import requests
import time
import re
import random
import os
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-4.1-nano"

MAX_EPISODES = 25
MAX_STEPS = 30
EXPLORATION_BIAS = 3.0


env = gym.make("FrozenLake-v1", is_slippery=False,render_mode="human")



memory = defaultdict(lambda: defaultdict(list))
cumulative_paths = []


def query_model(prompt):
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an advanced reinforcement learning agent. "
                        "Use provided transition data as ground truth. "
                        "Think step-by-step. "
                        "At the end output: FINAL_ACTION: <0-3>"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.6,
            "max_tokens": 600,
        },
    )

    data = response.json()

    if "choices" not in data:
        print("API ERROR:", data)
        return None

    return data["choices"][0]["message"]["content"]


def extract_action(text):
    if text is None:
        return None

    match = re.search(r"FINAL_ACTION:\s*([0-3])", text)
    if match:
        return int(match.group(1))

    nums = re.findall(r"\b[0-3]\b", text)
    if nums:
        return int(nums[-1])

    return None



def get_real_transitions(state):
    transitions = []

    for action in range(4):
        temp_env = gym.make("FrozenLake-v1", is_slippery=False)
        temp_env.reset()
        temp_env.unwrapped.s = state
        new_state, reward, terminated, truncated, _ = temp_env.step(action)
        transitions.append(
            f"Action {action} → next_state={new_state}, reward={reward}, terminated={terminated}"
        )
        temp_env.close()

    return "\n".join(transitions)


def format_memory(state):
    lines = []

    for action in range(4):
        outcomes = memory[state].get(action, [])
        tries = len(outcomes)

        if tries > 0:
            avg_reward = sum(r for (_, r, _) in outcomes) / tries
            terminations = sum(1 for (_, _, t) in outcomes if t)
            self_loops = sum(1 for (ns, _, _) in outcomes if ns == state)
            next_states = list(set(ns for (ns, _, _) in outcomes))
        else:
            avg_reward = 0
            terminations = 0
            self_loops = 0
            next_states = []

        exploration_bonus = EXPLORATION_BIAS / (1 + tries)

        lines.append(
            f"""
Action {action}:
  tries={tries}
  avg_reward={avg_reward:.2f}
  terminations={terminations}
  self_loops={self_loops}
  next_states_seen={next_states}
  exploration_bonus={exploration_bonus:.2f}
"""
        )

    return "\n".join(lines)


def format_paths():
    if not cumulative_paths:
        return "No completed episodes yet."

    lines = []
    for i, path in enumerate(cumulative_paths[-5:]):
        states = [step[0] for step in path]
        lines.append(f"Episode {i} path: {states}")

    return "\n".join(lines)



def choose_action(state, episode_history):

    real_transitions = get_real_transitions(state)
    memory_summary = format_memory(state)
    path_summary = format_paths()

    prompt = f"""
FrozenLake 4x4 (non-slippery).
Goal state: 15

Map layout:
0 1 2 3
4 H 6 H
8 9 10 H
H 13 14 15

Current state: {state}

Actual environment transitions (ground truth):
{real_transitions}

Episode history so far:
{episode_history}

Recent completed paths:
{path_summary}

Memory statistics:
{memory_summary}

INSTRUCTIONS:

1. Trust the real transitions above.
2. Strongly penalize:
   - terminated=True
   - self_loops
3. Encourage exploration (low tries).
4. Prefer actions moving toward 15.
5. Avoid repeating useless loops.
6. Think multiple steps ahead.
7. Use past successful paths if available.

Reason carefully step-by-step.

At the end output:
FINAL_ACTION: <0-3>
"""

    response = query_model(prompt)

    print("\n=========== MODEL THINKING ===========\n")
    print(response)
    print("\n======================================\n")

    action = extract_action(response)

    if action is None:
        print("LLM failed, random fallback.")
        action = random.randint(0, 3)

    return action

for episode in range(MAX_EPISODES):

    print(f"\n\n=========== EPISODE {episode} ===========\n")

    state, _ = env.reset()
    total_reward = 0
    episode_path = []
    episode_history = ""

    for step in range(MAX_STEPS):

        print(f"Step {step} | State {state}")

        action = choose_action(state, episode_history)

        new_state, reward, terminated, truncated, _ = env.step(action)

        print(f"Action: {action} → New State: {new_state} | Reward: {reward}")

        memory[state][action].append((new_state, reward, terminated))

        episode_path.append((state, action, new_state, reward, terminated))
        episode_history += f"State {state} → Action {action} → {new_state}\n"

        total_reward += reward
        state = new_state

        if terminated or truncated:
            break

        time.sleep(0.3)

    cumulative_paths.append(episode_path)

    print(f"\nEpisode reward: {total_reward}")

env.close()

print("\nTraining complete.")
