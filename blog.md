# 🚗 Teaching an AI to Drive Like It's Rush Hour in Bangalore

*A hackathon journey into one of the hardest unsolved problems in autonomous driving*

---

## The Problem Nobody Builds For

Most autonomous driving systems are trained on one quiet assumption:

> People follow lanes.

That assumption quietly removes most of the world.

On Indian roads, lanes are suggestions — not constraints. Cars, bikes, auto-rickshaws, pedestrians, street vendors, and animals share the same space. There is no strict structure — only **continuous negotiation**.

And yet traffic flows. Nobody taught people to do this explicitly. It emerged.

This is not chaos.  
This is **organized chaos** — and it is one of the hardest environments for an AI to reason in.

Modern AV systems are not built for this. Every major benchmark — CARLA, Waymo, nuScenes — was designed for roads that behave predictably. That is why we built AutoDrive Gym.

---

## What We Built

**AutoDrive Gym** is an OpenEnv-compatible reinforcement learning environment designed to train agents for:

- Context-aware decision making
- Multi-agent interaction
- Social negotiation
- Long-horizon planning under uncertainty

Unlike traditional simulators, it is:

- ⚡ Lightweight and fast
- 🎯 Scenario-driven (not just physics)
- 🧠 Focused on reasoning, not rendering

Each episode is a **carefully designed situation** — not just random traffic. The agent interacts over a simple HTTP API:

```bash
GET  /reset    # start a new episode
POST /step     # send an action, get observation + reward
GET  /grader   # score the agent on a specific task
```

---

## The Core Question

### ❓ *When should an AI NOT brake?*

This turned out to be the most important problem we faced.

Early training results looked great:

- High rewards ✅
- No crashes ✅
- Safe behavior ✅

But the agent had learned a shortcut:

> "Sensitive zone nearby → always brake"

This is not intelligence. This is pattern matching.

Real driving requires contextual reasoning:

| Situation | Correct Action |
|---|---|
| Hospital zone + ambulance arriving | Brake immediately |
| Hospital zone + empty road + green signal | Maintain speed |

Same environment. Opposite decisions.

**The difference is context.**

---

## How We Forced the Agent to Think

### 1. Ambiguous Scenarios

We designed cases where:

- braking = **wrong**
- maintaining speed = **correct**

The agent is penalized for unnecessary braking. This breaks the "always brake = safe" shortcut.

### 2. Dynamic Mid-Episode Events

A scenario can start safe, then change:

- Start: clear road → correct action is `accelerate`
- Step 4: ambulance spawns from driveway → correct action is `brake` and yield

The agent has to **adapt in real time** — not apply a fixed policy.

---

## Real-World Scenarios

We implemented 30+ scenarios across difficulty tiers:

| What happens | Why it's hard |
|---|---|
| Child darts across road near school | Must brake without being told it's a school zone |
| Ambulance approaches from behind | Yield without blocking, no sudden swerves |
| Wedding procession blocks all lanes | Horn is appropriate here — but measured |
| Temple zone, no procession active | No Horn sign is the constraint, not speed |
| Police waves you through a red light | Police authority overrides the traffic signal |
| Hospital zone at 3am, road empty | Ambulances can emerge anytime — still slow |
| Hospital zone midday, green signal, clear | **Maintain speed** — no hazard present |

These are not rare edge cases. These are **everyday driving situations in India**.

---

## Zone Inference — No Shortcuts Allowed

We never tell the agent what zone it is in. No `"zone_type": "hospital"` label.

Instead it observes:

```json
{
  "nearby_places": ["hospital", "pharmacy"],
  "visible_signs": ["Slow — Hospital Zone", "No Honking"],
  "ambient_cues": ["ambulance parked at entrance"],
  "pedestrian_density": "medium",
  "time_of_day": "10:30"
}
```

The agent must infer the environment, infer the risk, and infer the correct action.

Just like humans do.

---

## Architecture Overview

### 🧠 Multi-Agent Reasoning Pipeline

Six LLM sub-agents collaborate on every driving step:

```
Observation
    │
    ├─► Perception Agent     → What are the threats and how severe?
    ├─► Context Agent        → What is the overall situation?
    ├─► Intent Engine        → What is each actor trying to do?
    ├─► Negotiation Agent    → Should I yield or assert my lane?
    ├─► Decision Agent       → What is the optimal action?
    └─► Oversight Agent      → Is this decision safe? (can veto)
```

Running on `Qwen2.5-72B-Instruct` via Hugging Face Inference API.

### ⚡ Q-Learning Policy

- Learns repeated patterns efficiently over episodes
- Reduces reliance on expensive LLM calls for known situations
- Safety defaults for unlearned states: emergency vehicle nearby + close range → always brake-biased

---

## Live Training Dashboard

<!-- Replace with your actual dashboard screenshot -->
![AutoDrive Gym Live Dashboard](./assets/dashboard.png)

The dashboard tracks episode progress, agent decisions, reward signals, success rate, and current scenario state — updated in real time during training.

---

## Training Results

After pipeline training with real LLM calls:

- Step reward improved from **~0.4 → ~0.95**
- Over-braking incidents reduced by **~60%**
- Correct "maintain speed" decisions in clear zones increased significantly
- Dynamic event adaptation (suddenly spawned hazard) reached **~80% accuracy**

**Key insight:** the agent learned not just *what to do* — but *when not to act*.

---

## Hardest Remaining Challenge

> 🚨 Police officer waving you through a red signal

The agent still occasionally trusts the traffic light over the human.

This is a real open problem: **how should AI resolve conflicting authority?**

A traffic signal is a rule. A police officer waving you through is an override. Humans handle this intuitively. Teaching an AI to do the same is genuinely non-trivial.

---

## Demo

<!-- Replace with your actual video link -->
🎥 **[Watch the system in action — add video link]**

---

## Try It Yourself

```bash
docker build -t autodrive-gym .
docker run -p 8000:8000 -e LLM_BACKEND=hf -e HF_TOKEN=hf_xxx autodrive-gym
curl http://localhost:8000/reset
```

All 7 graded tasks available at `/tasks`. Full baseline scores at `/baseline`.

---

## Going Further: GRPO Fine-tuning

The Q-learning agent is a useful baseline — fast, no GPU, trains in seconds. But it cannot generalize to scenarios it has never seen, and it cannot explain its decisions.

For the second training track, we added **GRPO fine-tuning** — the same algorithm behind DeepSeek-R1 and QwQ.

The idea is simple: instead of a lookup table, the model itself (Qwen3-0.6B) generates driving decisions as JSON. GRPO trains it directly on episode outcomes from AutoDrive Gym — no teacher, no labels. Just the environment pushing back.

### Why GRPO fits this problem

PPO needs a value function — an estimate of how good a state is. For Indian road driving, this is hard to learn. The same observation (`hazard at 15m`) can warrant brake *or* accelerate depending on three steps of context the value network doesn't see.

GRPO sidesteps this entirely. It runs **8 parallel rollouts of the same scenario** and compares them: which completions scored higher? That difference *is* the advantage signal. No value network.

This is ideal for Indian road conditions where context matters more than distance. The same `15m gap to auto-rickshaw` might warrant:

- `brake` → if the auto is erratic and accelerating toward you
- `horn` → if it has committed to the lane change but is slowing
- `accelerate` → if it has slipped past and is now 20m ahead and receding

These distinctions live in the **sequence of tokens** leading to the action — exactly what GRPO optimizes.

### What the reward structure teaches

We learned from studying winning RL projects that naive reward signals plateau. The agent finds a local maximum (`always brake`) and stops improving.

Three additional signals break this:

- **Repeat-action penalty** — `-0.12` per consecutive repeat, `-0.20` on the third. Breaks the "always brake" shortcut.
- **Phase-order bonus** — `+0.08` for braking near an approaching hazard, `+0.10` for accelerating once cleared. Rewards *correct sequencing*, not just any action.
- **Resolution bonus** — `+1.0` to `+3.0` at episode end, scaled by speed. Faster correct decisions earn more.

The result: successful fast episodes score **+3 to +8**, failed episodes score **-2.0**. GRPO gets the variance it needs.

### Running it

```bash
# Environment server
uvicorn autodrive_env.server.app:app --host 0.0.0.0 --port 8000

# Fine-tuning (in a separate terminal, GPU required)
pip install -e ".[grpo]"
python train_grpo.py --episodes 50 --model-id Qwen/Qwen3-0.6B
```

Live metrics stream to `reward_log.json` and `live_state.json` every episode. The Training Lab UI (`python grpo_ui.py`) reads these in real time — the **GRPO** tab shows the reward curve updating as training runs.

---

## Why This Matters Beyond the Hackathon

Autonomous driving research has a representation problem. The environments that matter most for real-world AV deployment — South Asia, Southeast Asia, West Africa, Latin America — are almost entirely absent from mainstream benchmarks.

AutoDrive Gym is a step toward fixing that. The core principles here — contextual zone inference, ambiguous social signals, dynamic mid-episode adaptation, mixed-authority signals — apply to any environment where driving is a **social activity**, not just a physics problem.

---

## Links

- 🤗 Hugging Face Space: *[add link]*
- 💻 GitHub Repo: *[add link]*
- 📊 WandB Training Runs: *[add link]*

---

*"The goal isn't to build an agent that always brakes near a hospital. The goal is to build an agent that knows when not to."*