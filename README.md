

---
title: AutoDrive GRPO
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# AutoDrive Gym — Autonomous Driving in Indian Road Conditions

AutoDrive Gym is a complete [OpenEnv](https://huggingface.co/openenv)-compatible reinforcement-learning environment that challenges AI agents with the real-world chaos of Indian road conditions: sudden pedestrian crossings, auto-rickshaw cut-ins, bike blind-spot merges, potholes, ambiguous traffic signals, ambulance corridors, and much more.

> **Hackathon submission for [OpenEnv Round 1](https://huggingface.co/openenv).**
> Problem statement: *Build a real-world OpenEnv environment an AI agent can learn from.*

---

## Why Indian Roads?

Indian urban driving is one of the hardest unsolved problems in autonomous driving research:

- No strict lane discipline — vehicles share space fluidly
- Mixed traffic: cars, bikes, auto-rickshaws, pedestrians, animals, push-carts
- Ambiguous or absent road markings and signals
- Frequent unpredictable micro-events: sudden pedestrian crossings, vehicle cut-ins, potholes
- Human overrides: police hand signals supersede traffic lights

Existing AV benchmarks (CARLA, nuScenes, Waymo) are designed for Western road conditions. AutoDrive Gym fills a real gap for Indian-condition RL research.

---

## Environment Design

### Custom Car Simulator

A lightweight physics simulator (`LightweightDrivingSimulator`) models:

| Property | Detail |
|---|---|
| **Ego vehicle** | position (x, y), speed, steering, lane, lane_position |
| **Objects** | pedestrians, bikes, autos, cars, trucks, ambulances, animals, potholes |
| **Object behaviours** | sudden_cross, cut_in, blind_spot_merge, emergency_pass, zig_zag, static, ridge |
| **Environment** | road_condition, visibility, traffic_signal, lane_status |
| **Sensor model** | distance + angle to each object, on_road flag |

Each timestep: the agent sends one action → simulator executes it → objects move according to their behaviour patterns → sensor snapshot is captured → programmatic checks run → reward is computed → observation returned.

### Scenario Progression

Each episode has a **primary hazard** and optionally a **sudden mid-episode secondary hazard**.

```
reset()           → episode starts with primary alert visible
step() × N        → agent brakes / waits while hazard is close
  trigger_step=4  → dynamic event fires (e.g. "Pedestrian has crossed")
                    scenario_stage changes to "clearing"
                    hint = "Hazard has cleared. Accelerate NOW."
step() continuing → agent accelerates → progress_restored = True → done!
```

The agent is rewarded for:
1. Braking / waiting while the hazard is **approaching**
2. Accelerating once the hazard has **cleared**
3. **Not** oscillating: repeated identical actions are penalised
4. Reaching the goal without collision

### Sudden Mid-Episode Alerts

After a few steps, a secondary hazard can spawn without warning:

| Alert | What happens |
|---|---|
| Ambulance from rear | Siren alert, agent must create a corridor |
| Animal on road | Sudden crossing object appears |
| Traffic police override | Junction flow changes, signal override |
| Traffic jam | Static vehicles block the path ahead |
| Speed breaker | Unmarked ridge appears |

These appear as `active_alerts` in the observation and must be handled **while** the primary scenario is still active.

---

## Action Space

| Action | Value range | Effect |
|---|---|---|
| `accelerate` | 0.0 – 1.0 | Increase speed |
| `brake`       | 0.0 – 1.0 | Decrease speed |
| `steer_left`  | 0.0 – 1.0 | Steer left |
| `steer_right` | 0.0 – 1.0 | Steer right |
| `change_lane_left`  | — | Move to left lane |
| `change_lane_right` | — | Move to right lane |
| `horn`        | — | Use horn (social signal) |
| `wait`        | — | Remain stationary, hold steering |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `command_output` | str | Latest environment event + last action taken |
| `scene_summary`  | str | Compact snapshot: object count, signal, road |
| `event_log`      | str | Environment event (clearing, sudden alert, etc.) |
| `active_alerts`  | list[str] | Sudden mid-episode alerts |
| `hint`           | str | Stage-aware guidance (changes to "ACCELERATE" after clearing) |
| `scenario_type`  | str | Primary hazard type |
| `scenario_stage` | str | `approaching` / `clearing` / `cleared` |
| `hazard_type`    | str | Current hazard being tracked |
| `hazard_distance`| float | Distance to nearest hazard object (metres) |
| `hazard_status`  | str | `approaching` / `clearing` / `cleared` |
| `sensor_data`    | dict | Object list with distance, angle, on_road, behaviour |
| `ego_state`      | dict | speed, steering, position, lane |
| `environment`    | dict | road_condition, visibility, traffic_signal |
| `stage_scores`   | dict | Per-step decision / safety / efficiency scores (0–1) |
| `validation`     | dict | Programmatic checks: collision, near_miss, stuck, etc. |
| `resolution`     | dict | `verified` bool, `reason`, `bonus` |
| `reward`         | float | Step reward in (0, 1) |
| `done`           | bool | Episode ended |

---

## Tasks and Graders

Minimum 3 tasks required — this environment provides **6**, spanning easy → hard.

| Task ID | Difficulty | Scenario |
|---|---|---|
| `pedestrian_crossing` | Easy | Vulnerable road user crosses suddenly ahead |
| `auto_cut_in` | Easy | Auto-rickshaw cuts in unpredictably |
| `bike_blind_spot` | Medium | Bike merges from blind spot |
| `pothole_ahead` | Medium | Deep pothole after rain, requires avoidance |
| `traffic_light_ambiguity` | Medium | Conflicting signals + police manual override |
| `adversarial` | Hard | Multiple unpredictable agents in chaotic traffic |

Each task uses `HeuristicGrader` — a deterministic, callable class that:
- Takes `(observation, action, result_state, scenario, history)`
- Returns `{"score": float, "feedback": str}` with score strictly in **(0.0, 1.0)**
- Is importable at `autodrive_env.server.judge:HeuristicGrader`

### Grader Scoring Logic

| Behaviour | Score effect |
|---|---|
| Collision | ×0.05 penalty (severe) |
| Near miss | ×0.5 penalty |
| Off-road | ×0.2 penalty |
| Safe distance maintained | +0.10 bonus |
| Signal respected | +0.05 bonus |
| Incident cleared + moving | +0.10 bonus |
| Episode failed | ×0.5 penalty |

All scores are clamped into **(0.001, 0.999)** — never exactly 0 or 1.

---

## Reward Function

Step-level reward: strictly in **(0, 1)**, derived from the normalised judge score with multiplicative penalties and additive bonuses.

```
base = (judge_score + 1) / 2          # map [-1,1] → [0,1]
× collision_factor (0.05 if crash)
× near_miss_factor (0.5)
× repeat_factor (reduced for oscillating actions)
+ safe_distance_bonus (0.10)
+ signal_bonus (0.05)
+ incident_cleared_bonus (0.10)
× fail_factor (0.5 if episode ends without success)
→ clamped into (0.001, 0.999)
```

Partial progress is signalled every step — not a sparse end-of-episode reward.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `GET /reset` | GET | Start a new episode, returns initial observation |
| `POST /step` | POST | Send action `{"action": str, "value": float}`, returns obs+reward |
| `GET /state` | GET | Return current episode state |
| `GET /tasks` | GET | List all tasks with difficulty and grader reference |
| `GET /grader?task_id=X` | GET | Run grader on task X, return score in (0, 1) |
| `GET /baseline` | GET | Run grader on all 6 tasks, return baseline scores |
| `GET /healthz` | GET | Health check, returns 200 with environment info |

### Example: Reset + Step

```bash
curl http://localhost:8000/reset
curl -X POST http://localhost:8000/step \
     -H "Content-Type: application/json" \
     -d '{"action": "brake", "value": 0.9}'
```

### Example: Check Grader Score

```bash
curl "http://localhost:8000/grader?task_id=pedestrian_crossing"
# {"task_id":"pedestrian_crossing","score":0.652,"feedback":"braking for visible hazard"}

curl "http://localhost:8000/baseline"
# {"status":"ok","baseline_scores":[...all 6 tasks...]}
```

---

## Local Setup

### Prerequisites

- Python 3.10+
- `uv` (recommended) or `pip`

### Install

```bash
uv sync
# or
pip install -e .
```

### Required Environment Variables

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=your_huggingface_token
```

### Start the Server

```bash
python -m autodrive_env.server.app
```

Health check:
```bash
curl http://localhost:8000/healthz
```

### Run Inference (Hackathon Format)

```bash
python inference.py --base-url http://localhost:8000 --episodes 6
```

Expected output format:
```
[START]
Episode 1: pedestrian_crossing (junior judge)
------------------------------------------------------------------------
Task: ALERT: vulnerable road user crossing suddenly ahead
[STEP] Step 1: brake: 1.00              reward=+0.72
         trace: phase=approaching nearest=14.0 source=remote_llm
[STEP] Step 2: brake: 1.00              reward=+0.68
...
[STEP] Step 5: accelerate: 0.50         reward=+0.81
         trace: phase=clearing nearest=999.0 source=remote_llm
[END]
-> Judge verification: RESOLVED - incident cleared and safe progress restored | score=0.78
```

### Run Evaluation

```bash
python eval.py --base-url http://localhost:8000 --episodes 6
```

---

## Docker

Build:
```bash
docker build -t autodrive-gym .
```

Run:
```bash
docker run --rm -p 8000:8000 \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  -e HF_TOKEN=your_token \
  autodrive-gym
```

---

## Baseline Scores

Baseline collected by running `python inference.py --episodes 6` with `Qwen/Qwen2.5-72B-Instruct`:

| Task | Difficulty | Typical Score |
|---|---|---|
| `pedestrian_crossing` | Easy | 0.65 – 0.80 |
| `auto_cut_in`         | Easy | 0.55 – 0.75 |
| `bike_blind_spot`     | Medium | 0.50 – 0.70 |
| `pothole_ahead`       | Medium | 0.45 – 0.65 |
| `traffic_light_ambiguity` | Medium | 0.40 – 0.60 |
| `adversarial`         | Hard | 0.25 – 0.45 |

Frontier models (GPT-4o, Claude 3.5) score ~0.10–0.15 higher across the board.

---

## Training — Adaptive Policy Learning

AutoDrive Gym includes a full **before-vs-after training demonstration** showing that the agent measurably improves over episodes — no GPU or LLM required.

### Quick Demo (≈5 seconds)

```bash
python demo_training.py                    # 40 episodes across 6 scenarios
python demo_training.py --episodes 80      # longer run → stronger convergence signal
python demo_training.py --plot             # also saves reward curve PNG (needs matplotlib)
python demo_training.py --scenario hospital  # deep-dive on one scenario
```

Expected output (abbreviated):
```
  GLOBAL SUMMARY (6 scenarios)
  Improvement vs heuristic baseline:
    Mean reward:    4.62  →  4.85  (+5.2%)
    Success rate:   91.0% →  90.5%

  Verdict: ✅ PASS — agent measurably learned and improved over baseline!
```

### What the Agent Learns

| Scenario | Behaviour BEFORE | Behaviour AFTER |
|---|---|---|
| 🏥 Hospital Zone | Honks repeatedly (39 times / 13 eps) | Silent — no horns in sensitive zones |
| 📿 Temple Procession | Brakes + honks unpredictably | Waits quietly, navigates smoothly (+9.9%) |
| 🚕 Aggressive Auto | Naive panic or accelerates into cut-in | Predicts aggressor intent → yields correctly |
| 🚨 Ambulance Override | Confused response, blocks siren path | Yields/steers left to give corridor |
| Pedestrian Crossing | Horn-first approach | Brake-first, success rate up 8% |

### How Learning Works

Three modules work together:

1. **`server/policy_learner.py`** — Q-table adaptive policy  
   - State = `(scenario_type, stage, dist_bin, dominant_intent, zone_sensitivity, signal)`  
   - Decaying ε-greedy: starts 80% exploration → 5% by end  
   - TD update: `Q(s,a) ← (1−α)·Q(s,a) + α·[r + γ·maxₐ'Q(s',a')]`

2. **`server/belief_tracker.py`** — Bayesian Theory of Mind  
   - Per-actor belief distribution over 5 intents (rush / cautious / distracted / aggressive / yielding)  
   - Updated every step via `P(intent | signals)` — confidence grows as evidence accumulates

3. **`server/counterfactual.py`** — Counterfactual reasoning  
   - After each step: *"what reward would I have got with each other action?"*  
   - Avoidance bonus +0.08 when agent chose an action that avoids a would-be collision

### Adaptive Training Mode (full train loop)

```bash
python train.py --mode adaptive --episodes 50
```

---

## GRPO Fine-tuning — Teaching Qwen3 to Drive

Beyond the Q-table, AutoDrive Gym includes a full **GRPO fine-tuning pipeline** using [HF TRL](https://github.com/huggingface/trl) — the same algorithm behind DeepSeek-R1 and QwQ.

Instead of a lookup table, the model itself (Qwen3-0.6B or larger) generates driving decisions as JSON text. GRPO fine-tunes it directly on Indian-road driving episodes — no human labels, no separate reward model, just the gym as the training signal.

```
┌─────────────────────────────────────────────────────────────────┐
│                     GRPO TRAINING LOOP                          │
│                                                                 │
│  ┌──────────┐  ┌────────────────┐  ┌──────────────────┐        │
│  │ Indian   │  │ Qwen3 + LoRA   │  │  AutoDrive Gym   │        │
│  │ Scenario │► │ generates JSON │► │  step reward     │        │
│  │  (env)   │  │ action         │  │  (0–1 per step)  │        │
│  └──────────┘  └────────────────┘  └──────┬───────────┘        │
│                                           │                    │
│  ◄── GRPO gradient update (8 rollouts) ───┘                    │
│                                                                 │
│  Curriculum: warmup (0.15) → intermediate → expert (0.85)      │
└─────────────────────────────────────────────────────────────────┘
```

### Reward Structure

| Signal | Effect |
|--------|--------|
| Per-step env reward (0–1) | Dense signal every action |
| Repeat-action penalty (−0.12 / −0.20) | Prevents "always brake" plateau |
| Phase-order bonus (+0.08 / +0.10) | Rewards brake→accelerate sequencing |
| Resolution bonus (+1.0 to +3.0) | Efficiency-scaled — faster success = more |
| Timeout floor (−2.0) | Clean negative signal for failed episodes |

Successful fast episodes score **+3 to +8**; failures score **−2.0** — the variance GRPO needs for stable advantage estimates.

### GPU Requirements

| VRAM | Recommended model | LoRA rank |
|------|-------------------|-----------| 
| 8 GB | `Qwen/Qwen3-0.6B` | r=8 |
| 16 GB | `Qwen/Qwen3-1.7B` | r=16 (HF T4 default) |
| 24 GB | `Qwen/Qwen3-4B` | r=16 |
| 40–80 GB | `Qwen/Qwen2.5-7B-Instruct` | r=32 |

### Quick Start

**Option A — Colab notebook (recommended):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GH_USER/autodrive_openenv/blob/main/train_grpo.ipynb)

The notebook [`train_grpo.ipynb`](train_grpo.ipynb) handles GPU check, install, HF login, training, and reward curve plotting in 8 cells. Just open, set Runtime → T4 GPU, and run all.

**Option B — two terminals:**

```bash
# Terminal 1: environment server
uvicorn autodrive_env.server.app:app --host 0.0.0.0 --port 8000

# Terminal 2: GRPO training  (GPU required)
pip install -e ".[grpo]"
python train_grpo.py --episodes 50 --model-id Qwen/Qwen3-0.6B

# Push fine-tuned checkpoint to HF Hub
python train_grpo.py --episodes 50 --push-to-hub --hub-repo your-name/autodrive-agent
```

Live metrics are written to `reward_log.json` and `live_state.json` every episode. The Training Lab UI (`python grpo_ui.py`) reads them in real time from the **GRPO** tab.

### Why GRPO over PPO

PPO requires a value function — an estimate of how good a state is. For Indian road driving this is hard: the same observation (`hazard at 15m`) can be correct to brake *or* accelerate depending on 3 steps of context. GRPO sidesteps this by running **8 parallel rollouts of the same scenario** and comparing outcomes. No value network. Better suited for the context-dependent, delayed rewards in this domain.

---

## Multi-Agent Coordination (Experimental)

AutoDrive Gym includes experimental multi-agent support via `FleetCoordinator` and `MultiAgentPipeline`. This is not the primary focus of the submission, but it turned out to be a genuinely useful extension — especially for hard Indian traffic scenarios where a single agent can't observe everything.

### What it does

Fleet mode (`/fleet/reset`, `/fleet/step_all`) spawns multiple ego vehicles on a shared city route. Each vehicle has its own observation, but they share:

- **Sudden alerts** — an ambulance or police override broadcast to all agents simultaneously
- **Negotiation layer** — vehicles signal intent before executing a lane change or overtake, reducing conflict
- **Oversight** — a deterministic rule checker vetos any action that would cause a collision the agent could not yet see

### Pipeline architecture (6 stages)

```
Perception → Context → IntentInference → Negotiation → Decision → Oversight
```

The `MultiAgentPipeline` runs this chain for each action request. The result is a traced decision that shows exactly *why* each stage concluded what it did — useful for debugging and for understanding emergent cooperation.

### Results

When two agents train together on the shared city route (`city_route` task), cooperation emerges around ambulance corridors and police overrides — scenarios where a single agent has a blind spot but the fleet does not. In my runs:

| Metric | Single agent | 2-agent fleet |
|--------|-------------|---------------|
| Ambulance corridor success | 62 % | 81 % |
| Police override compliance | 54 % | 76 % |
| City route completion (5 CPs) | 38 % | 57 % |

The gains come almost entirely from the negotiation and overnight layers — the agents are not sharing gradient updates, just structured intent messages.

### Try it

```bash
# Reset a 2-vehicle fleet
curl -X POST "http://localhost:8000/fleet/reset?n_vehicles=2"

# Step all vehicles simultaneously
curl -X POST http://localhost:8000/fleet/step_all \
  -H "Content-Type: application/json" \
  -d '{"vehicle_0": {"action": "brake", "value": 0.8},
       "vehicle_1": {"action": "steer_left", "value": 0.5}}'

# Check fleet status
curl http://localhost:8000/fleet/status
```

Multi-agent training via `python train.py --mode pipeline` runs the 6-stage pipeline for every action decision and logs the full trace per episode.

> **Note:** Fleet mode is experimental. The communication protocol is deterministic (no learned communication policy), and the biggest remaining gap is training the agents jointly with shared gradient updates. That is a natural next step and would likely push the ambulance corridor figure above 90 %.

---

## Project Structure

```
autodrive_env/
├── inference.py           # Hackathon inference runner (required by spec)
├── eval.py                # Full evaluation with report writing
├── train.py               # Training loop (--mode heuristic|adaptive|pipeline|route)
├── demo_training.py       # ★ Before-vs-after learning demo (no GPU/LLM required)
├── agent_baseline.py      # Fallback heuristic agent
├── client.py              # OpenEnv typed client
├── models.py              # Typed Pydantic models
├── openenv_compat.py      # OpenEnv compatibility shim
├── openenv.yaml           # OpenEnv spec + task/grader declarations
├── pyproject.toml
├── Dockerfile
└── server/
    ├── app.py                       # FastAPI app + /tasks /grader /baseline
    ├── autodrive_gym_environment.py # Core env: reset / step / state
    ├── driving_backend.py           # Physics simulator + sensor model
    ├── driving_actions.py           # Action handler
    ├── scenario_generator.py        # Scenario pool + zone-inference scenarios
    ├── scenario_injectors.py        # Inject actors + events into simulator
    ├── curriculum.py                # Adaptive difficulty controller
    ├── judge.py                     # LLMJudge + HeuristicJudge + HeuristicGrader
    ├── grader.py                    # 5-dimension reward (safety/efficiency/compliance/smooth/negotiation)
    ├── intent_engine.py             # Hidden actor intents + observable behavioral signals
    ├── zone_api.py                  # Sensitive zone inference from POI cues (no label given)
    ├── multi_agent_pipeline.py      # 6-stage pipeline: Perception→Context→Intent→Negotiation→Decision→Oversight
    ├── policy_learner.py            # ★ Q-table adaptive policy (ε-greedy, TD updates)
    ├── belief_tracker.py            # ★ Bayesian Theory-of-Mind: per-actor intent beliefs
    ├── counterfactual.py            # ★ Counterfactual reasoning + avoidance bonus
    ├── reward_tracker.py            # Per-episode metric curves
    ├── llm_client.py                # OpenAI-compatible LLM client
    └── constants.py                 # Actions, scenario types, defaults
```

---

## OpenEnv Spec Compliance

- `openenv.yaml` — spec version, runtime, app entrypoint, **6 tasks with grader references**
- Typed `AutoDriveAction`, `AutoDriveObservation`, `AutoDriveState` Pydantic models
- `reset()` → clean state, initial observation
- `step(action)` → observation, reward (0–1), done, info
- `state()` → current episode metadata
- `openenv validate` passes
- Dockerfile builds and runs cleanly
- HF Space deploys and responds to `/reset`


AutoDrive Gym is an OpenEnv-compatible autonomous driving environment built for Indian road conditions. It evaluates agent behavior on dynamic hazards such as pedestrian crossings, auto-rickshaw cut-ins, bike blind spots, potholes, speed breakers, crowded markets, and sudden mid-episode alerts like ambulances or traffic police overrides.

This project is designed for hackathon submission:
- local Python server support
- Docker packaging
- OpenEnv-compatible API
- evaluation with judge verification and structured reports

## Highlights

- OpenEnv-compatible `reset()` / `step()` interaction loop
- FastAPI server for local and containerized execution
- Hybrid evaluation agent:
  - LLM reasoning
  - memory over recent steps
  - phase-aware decision logic
  - safety and anti-stall guardrails
- Judge verification with final outcome:
  - `RESOLVED`
  - `UNRESOLVED`
- Indian-road scenario coverage:
  - pedestrian crossings
  - auto-rickshaw cut-ins
  - bike blind spots
  - potholes
  - speed breakers
  - crowded markets
  - ambulance approach
  - police override
  - traffic jam
  - animal crossing
  - rain-slippery road
- Mid-episode sudden hazards:
  - a primary scenario can remain active while a secondary surprise hazard appears later in the same episode

## Project Structure

```text
autodrive_env/
|- agent_baseline.py
|- client.py
|- Dockerfile
|- eval.py
|- inference_report.json
|- models.py
|- openenv.yaml
|- openenv_compat.py
|- pyproject.toml
|- README.md
|- train.py
|- __init__.py
`- server/
   |- app.py
   |- autodrive_gym_environment.py
   |- constants.py
   |- curriculum.py
   |- driving_actions.py
   |- driving_backend.py
   |- judge.py
   |- llm_client.py
   |- scenario_generator.py
   `- scenario_injectors.py
```

## OpenEnv Interface

The environment exposes the standard loop:

```python
obs = env.reset()
obs, reward, done, info = env.step(action)
```

The server returns a structured observation containing:
- scene summary
- sensor/object snapshot
- ego vehicle state
- environment state
- hazard type
- hazard status
- hazard distance
- scenario stage
- judge metadata

## Example Scenarios

Primary scenarios:
- `pedestrian_crossing`
- `auto_cut_in`
- `bike_blind_spot`
- `pothole_ahead`
- `speed_breaker`
- `crowded_market`

Sudden mid-episode hazards:
- `ambulance_approach`
- `animal_crossing`
- `police_override`
- `traffic_jam`
- `speed_breaker`

Example intended behavior:
- Episode starts as `pedestrian_crossing`
- Agent brakes and stabilizes
- Sudden ambulance appears from the rear
- Agent must adapt while the episode still remains a `pedestrian_crossing` scenario

## Local Setup

### 1. Create or activate your environment

Use your preferred Python 3.10+ environment.

### 2. Install dependencies

If you use `uv`:

```powershell
uv sync
```

Or with pip:

```powershell
pip install -e .
```

### 3. Start the server

From the `autodrive_env` directory:

```powershell
python -m autodrive_env.server.app
```

Health check:

```powershell
curl http://localhost:8000/healthz
```

## Model Configuration

### Ollama / local OpenAI-compatible API

```powershell
set LLM_BACKEND=openai
set LLM_BASE_URL=http://localhost:11434/v1
set LLM_API_KEY=ollama
set LLM_MODEL=llama3
```

### Groq

```powershell
set LLM_PROVIDER=openai
set OPENAI_BASE_URL=https://api.groq.com/openai/v1
set OPENAI_API_KEY=your_groq_key
set OPENAI_MODEL=llama-3.1-8b-instant
```

## Hackathon Inference Contract

For hackathon scoring, use the root-level [inference.py](/c:/Users/samatha/OneDrive/Documents/greenops-openenv/autodrive_env/inference.py) script.

It follows the required contract:
- uses the OpenAI client for LLM calls
- reads:
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`
- emits structured stdout in:
  - `[START]`
  - `[STEP]`
  - `[END]`

### Required variables

```powershell
set API_BASE_URL=https://api.groq.com/openai/v1
set MODEL_NAME=llama-3.1-8b-instant
set HF_TOKEN=your_api_key
```

### Run compliant inference

```powershell
python inference.py --base-url http://localhost:8000 --episodes 6
```

## Evaluation

Run the evaluation script:

```powershell
python eval.py --base-url http://localhost:8000 --episodes 6
```

The evaluation prints:
- per-step actions
- rewards
- decision trace
- sudden alerts when injected mid-episode
- judge verification

It also writes:
- [inference_report.json](/c:/Users/samatha/OneDrive/Documents/greenops-openenv/autodrive_env/inference_report.json)

The report includes:
- task trace per episode
- success rate
- average reward
- stage grades
- weakness tracking
- auto-generated notes

## Docker

Build the image from the `autodrive_env` directory:

```powershell
docker build -t autodrive-gym .
```

Run it:

```powershell
docker run --rm -p 8000:8000 autodrive-gym
```

Run with LLM configuration:

### Ollama on host

```powershell
docker run --rm -p 8000:8000 `
  -e LLM_BACKEND=openai `
  -e LLM_BASE_URL=http://host.docker.internal:11434/v1 `
  -e LLM_API_KEY=ollama `
  -e LLM_MODEL=llama3 `
  autodrive-gym
```

### Groq

```powershell
docker run --rm -p 8000:8000 `
  -e API_BASE_URL=https://api.groq.com/openai/v1 `
  -e MODEL_NAME=llama-3.1-8b-instant `
  -e HF_TOKEN=your_groq_key `
  autodrive-gym
```

## OpenEnv Packaging

This repo includes:
- [openenv.yaml](/c:/Users/samatha/OneDrive/Documents/greenops-openenv/autodrive_env/openenv.yaml)
- a FastAPI runtime
- a Dockerfile

That makes it ready for OpenEnv packaging and hub deployment.

### Push to hub

After authenticating your OpenEnv CLI:

```powershell
openenv push
```

If your workflow uses a specific project or remote target, use the corresponding OpenEnv CLI flags required by your hackathon account.

## Recommended Submission Checklist

- OpenEnv server runs locally
- Docker image builds cleanly
- `/healthz` responds successfully
- `eval.py` produces a fresh `inference_report.json`
- README includes setup, run, eval, and Docker instructions
- `openenv push` is executed from an authenticated CLI session

## Environment Variables

Useful runtime knobs:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `LLM_BACKEND`
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`
- `LLM_PROVIDER`
- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `GROQ_API_KEY`
- `GROQ_MODEL`

## Current Output Signals

During evaluation you will see:
- `trace: phase=...`
- `sudden alert: ...`
- `guardrail adjusted ...`

Example:

```text
sudden alert: ambulance approaching quickly from behind.
```

This means:
- the episode still belongs to its original scenario
- a secondary hazard was injected mid-episode
- the agent is being evaluated on how it adapts mid-episode

---

## Links

- 🎥 YouTube Demo: https://youtu.be/UAaQS-xngz8
- 🤗 Hugging Face Space: https://huggingface.co/spaces/Samatha369/autodrive_grpo
- 💻 GitHub Repo: https://github.com/SamathaAshokkumar/autodrive_grpo
- 📊 WandB Training Runs: https://wandb.ai/samatha45102-scaler/autodrive-gym/runs/6wbcqovs?nw=nwusersamatha45102