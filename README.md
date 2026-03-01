# LLMs as Gym Agents

**Can large language models actively solve reinforcement learning environments?**

This project explores whether an LLM can function as a policy inside classic Gym environments, without reinforcement learning, gradient updates, or weight training. Instead of updating parameters, the LLM reasons over context and episode history to choose actions.

---

## Motivation

Traditional RL agents learn via gradient descent, update value functions, and improve across episodes. LLMs do none of this. They don't update weights, don't accumulate value estimates, and only reason over what's inside the context window.

So the question becomes: **can reasoning alone simulate reinforcement learning behavior?**

---

## Environments

Three Gymnasium environments of increasing difficulty:

**CartPole-v1** — dense reward, simple feedback control, discrete actions (left/right)

**FrozenLake-v1** — sparse reward, multi-step planning, grid-based navigation

**LunarLander-v3** — continuous dynamics, multi-variable control, long-horizon stabilization

---

## Experimental Setup

At every timestep, the LLM receives the current observation, the episode history, a goal description, and the available action space. It outputs a single action. There is no gradient descent, no policy update, no training between episodes, just zero-shot decision making at each step.

---

## Key Findings

**CartPole** works surprisingly well. The control law is naturally verbalizable ("if the pole leans right, move right") and LLMs are good at executing structured rules expressed in language.

**FrozenLake** fails under pure reward feedback but succeeds when given the full transition function. When the LLM knows exactly where each action leads, it can plan. Without that, sparse rewards provide too weak a signal for incontext reasoning to recover from.

**LunarLander** is the hard case. The dynamics are too continuous and high dimensional for natural language reasoning to reliably track without structured state summaries.

---

## What This Tells Us

LLMs can approximate policies in environments where the optimal behavior is **describable in language** and feedback is **dense enough to reason about**. They break down where dynamics are continuous, rewards are sparse, or multi-step credit assignment is required.


---

## Structure

```
├── cartpole_agent.py
├── frozenlake_agent.py
└── lunarlander_agent.py
```

---

## Requirements

```
gymnasium
openai
numpy
```

---

## Usage

```bash
python cartpole_agent.py
python frozenlake_agent.py
python lunarlander_agent.py
```

---
