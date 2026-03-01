import gymnasium as gym
import requests
import json
import time
from dotenv import load_dotenv
import os

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


MODEL = "openai/gpt-4.1-nano"

MAX_STEPS = 400



def call_llm(prompt):

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "LunarLander-Target-Agent"
    }

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise landing controller. Output only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    try:
        return result["choices"][0]["message"]["content"]
    except:
        print("LLM ERROR:", result)
        return None




def build_prompt(state):

    x, y, xv, yv, angle, ang_vel, left_leg, right_leg = state

    return f"""
You are landing a rocket.

TARGET:
x = 0
angle = 0
vertical speed ≈ -0.1 (slow descent)

CURRENT STATE:
x = {x}
y = {y}
x_velocity = {xv}
y_velocity = {yv}
angle = {angle}
angular_velocity = {ang_vel}

ACTIONS:
0 = do nothing
1 = fire left engine (push right / rotate left)
2 = fire main engine (push upward)
3 = fire right engine (push left / rotate right)

CONTROL RULES:

1. If angle is tilted, fix angle first.
2. If drifting horizontally, correct toward x=0.
3. If falling too fast (y_velocity < -0.4), use main engine.
4. If rising (y_velocity > 0), DO NOT use main engine.
5. Near ground (y < 0.5), reduce vertical speed carefully.

Choose ONE action that moves the rocket closer to the TARGET.

Respond ONLY with valid JSON:

{{ "action": 0 }}
or
{{ "action": 1 }}
or
{{ "action": 2 }}
or
{{ "action": 3 }}
"""



env = gym.make("LunarLander-v3", render_mode="human")
state, _ = env.reset()

print("\n=========== EPISODE 0 ===========\n")

for step in range(MAX_STEPS):

    prompt = build_prompt(state)
    llm_response = call_llm(prompt)

    if llm_response is None:
        print("No LLM output. Stopping.")
        break

    print("\nLLM RAW RESPONSE:\n")
    print(llm_response)

    try:
        parsed = json.loads(llm_response.strip())
        action = int(parsed["action"])
    except:
        print("Invalid JSON. Stopping.")
        break

    next_state, reward, terminated, truncated, _ = env.step(action)

    print(f"\nStep {step} | Action: {action} | Reward: {reward}")

    state = next_state

    time.sleep(0.1)

    if terminated or truncated:
        print("\nEpisode finished.")
        break

env.close()
