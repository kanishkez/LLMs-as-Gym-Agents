# LLMs as Gym Agents

Testing whether large language models can act as policies in Gymnasium environments, without RL training, gradient updates, or value-function learning.

At each timestep, the model gets the current observation and local trajectory context, then outputs an action. This repo supports the CAISc 2026 paper experiments across three environments.

Blog post: https://kanishkk.substack.com/p/can-llms-solve-gym-environments
Paper: *Can LLMs Act as RL Agents?* (CAISc 2026)

---

## Environments

- `cartpole/` for `CartPole-v1`
- `frozenlake/` for `FrozenLake-v1`
- `lunarlander/` for `LunarLander-v3`

Each environment has its own folder with agent code and experiment artifacts.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Add API keys in `.env`:

- `OPENROUTER_API_KEY` for OpenRouter-backed runs (for example GPT-4o)
- `NVIDIA_API_KEY` for NVIDIA NIM-backed runs (Llama, DeepSeek, Qwen, Mistral, etc.)

---

## Run Single-Environment Agents

```bash
python cartpole/cartpole_agent.py
python frozenlake/frozenlake_agent.py
python lunarlander/lunarlander_agent.py
```

---

## Run Multi-Model Experiments

```bash
python multi_model_runner.py
```

Configure `MODELS_TO_RUN` and `ENVS_TO_RUN` near the bottom of `multi_model_runner.py`.

---

## Experiment Logs (JSON)

### CartPole

- `cartpole/cartpole_experiment_log_llama33_20260424_115502.json`
  - Converted from the original text log.
  - Includes per-step actions/rewards for 3 episodes and summary stats.

### FrozenLake

- `frozenlake/frozenlake_results_gpt4o_coordinate_enriched_20260424.json`
  - GPT-4o, 20 episodes, coordinate-enriched prompt.
  - 0% success with coordinate-only world context.

### LunarLander

- `lunarlander/lunarlander_results_gpt4o_variants_20260424.json`
  - GPT-4o across four prompt variants (baseline, state detail, goal with coords, chain-of-thought).
- `lunarlander/lunarlander_experiment_log_llama33_reasoning_20260424_121532.json`
  - Converted from the original reasoning trace text log.
  - Includes parsed step-level telemetry and reasoning segments.

---

## Key Findings (Paper-Aligned)

- CartPole can be solved with strong structured prompting on top-tier models; weaker/open models plateau much lower.
- FrozenLake fails without transition knowledge and improves with explicit path/world-model hints.
- LunarLander remains unsolved across tested zero-shot variants; reasoning traces show a deliberation-reaction gap (under-urgent control decisions in continuous dynamics).

---

## Repo Layout

```text
.
├── cartpole/
│   ├── cartpole_agent.py
│   └── cartpole_experiment_log_llama33_20260424_115502.json
├── frozenlake/
│   ├── frozenlake_agent.py
│   └── frozenlake_results_gpt4o_coordinate_enriched_20260424.json
├── lunarlander/
│   ├── lunarlander_agent.py
│   ├── lunarlander_results_gpt4o_variants_20260424.json
│   └── lunarlander_experiment_log_llama33_reasoning_20260424_121532.json
├── multi_model_runner.py
├── requirements.txt
├── .env.example
└── README.md
```
