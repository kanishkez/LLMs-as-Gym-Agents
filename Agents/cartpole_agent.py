import gymnasium as gym
import requests
import json
import time
from dotenv import load_dotenv
import os

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-4.1-nano"

MAX_EPISODES = 20
MAX_STEPS = 500



env = gym.make("CartPole-v1",render_mode="human")



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
                        "You are a physics reasoning agent controlling CartPole.\n"
                        "You must think step-by-step before deciding.\n"
                        "After reasoning, output ONLY valid JSON like {\"action\": 0}.\n"
                        "Do not output anything else after the JSON."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2
        },
        timeout=20
    )

    data = response.json()

    if "choices" not in data:
        raise Exception(f"Bad API response: {data}")

    content = data["choices"][0]["message"]["content"]

    # Print reasoning
    print("\nLLM RAW RESPONSE:")
    print(content)

    # Extract JSON from response
    start = content.find("{")
    end = content.rfind("}") + 1
    json_str = content[start:end]

    parsed = json.loads(json_str)
    action = parsed["action"]

    if action not in [0, 1]:
        raise Exception(f"Invalid action from LLM: {action}")

    return action



for episode in range(MAX_EPISODES):

    state, _ = env.reset()
    total_reward = 0

    print(f"\n=========== EPISODE {episode} ===========")

    for step in range(MAX_STEPS):

        cart_position = state[0]
        cart_velocity = state[1]
        pole_angle = state[2]
        pole_angular_velocity = state[3]

        prompt = f"""
You are controlling a CartPole system.

Current state:
- cart_position = {cart_position:.5f}
- cart_velocity = {cart_velocity:.5f}
- pole_angle = {pole_angle:.5f} radians (0 = perfectly vertical)
- pole_angular_velocity = {pole_angular_velocity:.5f}

Failure occurs if:
- |pole_angle| > 0.209 radians (~12 degrees)
- |cart_position| > 2.4

Goal:
Keep the pole balanced upright as long as possible.

Reason step-by-step:

1. Which direction is the pole leaning?
2. Is angular velocity increasing that lean?
3. If the pole is falling right, move right to get under it.
4. If the pole is falling left, move left to get under it.
5. Consider cart position limits.

Actions:
0 = Push LEFT
1 = Push RIGHT

Simulate mentally which action reduces |pole_angle| in the next step.

After reasoning, output ONLY:
{{"action": 0}}
"""

        try:
            action = query_model(prompt)
        except Exception as e:
            print("LLM ERROR:", e)
            break

        state, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        print(f"\nStep {step} | Action: {action} | Reward: {reward}")

        if terminated or truncated:
            break

        time.sleep(0.02)

    print(f"\nEpisode {episode} Total Reward: {total_reward}")

env.close()
