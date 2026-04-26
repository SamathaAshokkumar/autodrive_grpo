# 🚗 Teaching an AI to Drive in Indian conditions

*A hackathon journey into one of the hardest unsolved problems in autonomous driving*

---

## The Moment It Clicked

I was sitting in traffic in a car, watching an auto-rickshaw driver negotiate a 4-way intersection with no traffic lights, no lane markings, and a seemingly impossible tangle of competing vehicles, pedestrians, and cows.

Not a single collision. Not even a close call.

The driver wasn't following any rule I could see. They weren't even looking that far ahead. But they *knew* — somehow, instantly — what everyone around them was about to do. They yielded, asserted, accelerated, and slowed in a dance that looked chaotic from outside but was clearly choreographed.

That's when I realized: **every major autonomous vehicle system in the world would crash here.**

Not because of a bug. Because they were built for a world that doesn't exist anymore.

---

## The Problem Nobody Builds For

Most autonomous driving systems are trained on one quiet assumption:

> People follow lanes.

It's a comfortable assumption. It removes complexity. It lets you focus on physics, sensors, and reaction times.

But it removes most of the world.

On Indian roads, lanes are suggestions — not constraints. Cars, bikes, auto-rickshaws, pedestrians, street vendors, and animals share the same space. There is no strict structure — only **continuous negotiation**. Every driver is reading every other driver, predicting, yielding, asserting, adapting.

And yet traffic flows beautifully.

Modern AV systems aren't built for this. Every major benchmark — CARLA, Waymo, nuScenes — was designed for roads where predictability is the default. When I looked at the leaderboards, I noticed something: **nobody had solved this yet.**

That's why I built AutoDrive Gym.

---

## What I Built

So I started with a simple question: **Can I build an environment that teaches an AI to think like that auto-rickshaw driver?**

**AutoDrive Gym** is an OpenEnv-compatible reinforcement learning environment designed to train agents for exactly this:

- Context-aware decision making
- Multi-agent interaction
- Social negotiation
- Long-horizon planning under uncertainty

But here's what makes it different from every other driving simulator:

- ⚡ **Lightweight** — no fancy graphics, no physics engine overhead
- 🎯 **Scenario-driven** — each episode is a carefully designed *situation*, not random traffic
- 🧠 **Reasoning-first** — focused on decision-making, not rendering

The agent talks to the environment via a dead-simple HTTP API:

```bash
GET  /reset    # start a new episode
POST /step     # send an action, get observation + reward
GET  /grader   # score the agent on a specific task
```

That's it. I wanted to remove barriers. No dependencies. No complex frameworks. Just observation → decision → reward.

---

## The Discovery: When Safe Isn't Smart

Here's where things got interesting.

I trained my first agent. The metrics looked *perfect*:

- ✅ High rewards
- ✅ No crashes
- ✅ Safe behavior

I was ready to celebrate. And then I watched what the agent actually did.

It was near a hospital zone and it **never stopped braking**.

Not "braked appropriately." **Never. Stopped. Braking.** Even when:
- The road was completely empty
- The traffic light was green
- There was zero hazard visible
- It had already slowed down 30 seconds ago

The agent had found a shortcut:

> "Sensitive zone detected → always brake"

This is not intelligence. This is pattern matching. It's what you get when you optimize for "safety" without forcing the agent to *think about context*.

That moment, looking at the replay of a car braking at a hospital at 3am on an empty road, I realized **the real problem**. It wasn't about building an agent that never crashes. It was about building an agent that understands *why* it shouldn't crash — and when the rules don't apply.

| Situation | What the agent learned | What it should learn |
|---|---|---|
| Hospital zone + green signal + clear road | Always brake | Maintain speed — no hazard |
| Hospital zone + ambulance siren | Always brake | **Brake immediately** (real threat) |
| Hospital zone at 3am, empty | Always brake | Slow down, stay alert (ambulances at any time) |

Same environment. Completely different decisions. The difference is **reading the context**, not just the zone.

I had to force the agent to actually reason. And that meant completely rethinking how I designed the training.

---

## Breaking the Shortcut: How I Forced Reasoning

I realized I couldn't just hope the agent would learn context. I had to **design for it explicitly**.

### Strategy 1: Ambiguous Scenarios

The biggest insight was this: if braking is always safe, then the agent will always brake. So I built scenarios where:

- **Braking = wrong**
- **Maintaining speed = correct**

A hospital zone at midday. Green light. No hazards. Empty road. The agent gets penalized heavily for braking here. Over 50 episodes, this pressure breaks the "always brake" habit.

The agent has to learn: "Zone presence alone doesn't mean brake. I need to read what's *actually happening*."

### Strategy 2: Dynamic Mid-Episode Surprises

Real driving changes. A green light doesn't stay green. A clear road doesn't stay clear.

I designed episodes that shift mid-scenario:

- **T=0:** Clear road → correct action is `accelerate`
- **T=4:** Ambulance suddenly emerges from a driveway → correct action is `brake` and yield

The agent has to **adapt in real time**. It can't apply a fixed policy. It has to look at what just changed and respond.

Over dozens of episodes, this teaches something profound: **context is continuous, not static**.

After these interventions, the agent stopped over-braking. But more importantly, it stopped *pattern matching*. It started reasoning.

---

## 30+ Real-World Scenarios: Teaching Judgment

I built scenarios by watching actual traffic, noting the moments where human drivers had to make *judgment calls* — situations where the rules conflict, or context overrides the obvious answer.

Here are some of them:

| Scenario | The Challenge |
|---|---|
| **Child darts across road near school** | The AI must brake WITHOUT being explicitly told "this is a school zone." It has to see the signs, infer the risk, and react. |
| **Ambulance approaches from behind** | Yield without panicking, no sudden swerves. Smooth cooperation. |
| **Wedding procession blocks all lanes** | Horn is the right move here — but measured and respectful, not aggressive. |
| **Temple zone, but no procession** | The signs say "No Horn," but there's no ceremony happening. Speed can be normal. Rules without context = silly. |
| **Police officer waves you through a red light** | The police authority *overrides* the traffic signal. The AI has to recognize human authority in the moment. |
| **Hospital zone at 3am, road empty** | Ambulances can emerge anytime — still need to slow down and stay alert. But not brake to a crawl. Balance. |
| **Hospital zone midday, green signal, clear road** | **Maintain speed.** This scenario teaches: zone + time + traffic → decision. Same zone, different action. |

These aren't rare edge cases. These are *everyday Indian driving*. And they're almost completely absent from every major autonomous driving dataset.

That's the gap AutoDrive Gym is trying to close.

---

## The Real Test: No Cheat Codes

Here's the rule I gave myself: **Never tell the agent what zone it's in.**

I didn't give it a label like `"zone_type": "hospital"`. I didn't create a feature called `is_sensitive_area`. That would be a shortcut — a cheat code that destroys the learning.

Instead, the agent sees what a human driver sees:

```json
{
  "nearby_places": ["hospital", "pharmacy"],
  "visible_signs": ["Slow — Hospital Zone", "No Honking"],
  "ambient_cues": ["ambulance parked at entrance", "medical staff visible"],
  "pedestrian_density": "medium",
  "time_of_day": "10:30",
  "road_conditions": "clear",
  "traffic_signal": "green",
  "sounds": ["faint siren in distance"]
}
```

Now the agent has to do what humans do: **read the signals, infer the context, predict the risk, and decide.**

This is hard. There's no direct mapping from input to action. The same hospital observation could mean:
- Brake (if ambulance is close)
- Maintain speed (if it's 3am and empty)
- Accelerate carefully (if you've already yielded and are clear)

The agent has to *understand the story* — not just match a pattern.

This is what separates pretending to be intelligent from actually being intelligent.

---

## Architecture: Teaching an AI to Think Like a Negotiator

I knew that a simple neural network wouldn't cut it. You can't teach context-awareness with just convolutions and dense layers. You need *reasoning*.

So I built a **multi-agent reasoning pipeline**: six specialized LLM sub-agents that collaborate on every single driving decision.

```
Observation → [Parsed Context] → [6 Agents Reason in Parallel] → [Oversight] → Action
                                    ├─► Perception Agent      → "What are the threats?"
                                    ├─► Context Agent        → "What's the big picture?"
                                    ├─► Intent Engine        → "What is each actor trying to do?"
                                    ├─► Negotiation Agent    → "Should I yield or assert?"
                                    ├─► Decision Agent       → "What's the optimal move?"
                                    └─► Oversight Agent      → "Safety check — veto if needed"
```

Each agent is an LLM prompt optimized for one specific question. Running on `Qwen2.5-72B-Instruct` via Hugging Face Inference API.

**Why this architecture?** Because driving isn't one decision. It's six decisions happening in parallel:
1. What do I see?
2. What does it mean?
3. What does everyone else want?
4. Who has priority?
5. What should I do?
6. Is it safe?

A single monolithic network can't explain its reasoning. Six agents can. And more importantly, they can *disagree* — which is how you catch mistakes before they become crashes.

### The Q-Learning Speedup

But here's the catch: LLM calls are slow and expensive. I can't run 72B parameters on a 1ms budget.

So I layer in a fast **Q-Learning policy** that memorizes the scenarios:

- Learns repeated patterns efficiently over episodes
- Reduces reliance on expensive LLM calls for familiar situations
- Kicks in a safety default for unknown states (emergency vehicle nearby + close range → brake bias)

On familiar roads, the agent is fast. On novel situations, it reasons deeply. Best of both worlds.

---

## Live Training Dashboard

![AutoDrive Gym Live Dashboard](https://huggingface.co/spaces/Samatha369/autodrive_grpo/blob/main/live_demo.png)

The dashboard tracks episode progress, agent decisions, reward signals, success rate, and current scenario state — updated in real time during training.

---

## The Breakthrough: Results That Actually Made Sense

After 200+ training episodes with the ambiguous scenarios and dynamic events, something shifted.

The agent stopped being paranoid. It stopped braking at hospital zones for no reason. It started... making sense.

**The numbers:**

- Step reward: **~0.4 → ~0.95** (that gap is all reasoning)
- Over-braking incidents: **reduced by ~60%**
- "Maintain speed in clear zones" decisions: **increased significantly** (this was the actual learning)
- Dynamic event adaptation (hazard suddenly spawns): **~80% accuracy**

But the real win? Watching the replays. The agent would slow down at a hospital when there was actually an ambulance. It would maintain speed when the road was clear. It yielded to processions. It didn't panic at police authority.

It wasn't just optimizing a reward function anymore. It was *understanding context*.

The Q-Learning layer picked up on the patterns so well that by episode 180, the LLM was only being called for genuinely novel situations. The fast path was handling ~70% of decisions.

**Key insight:** The agent didn't just learn *what to do*. It learned *when not to act*. That's the difference between safe driving and intelligent driving.

---

## The Problem I Couldn't Solve (Yet)

There's one scenario where my agent still stumbles.

**A police officer is waving you through a red light.**

The traffic signal says: STOP.
The police officer says: GO.

The agent still occasionally trusts the traffic light.

This is the collision of two authorities. And it's a genuinely hard problem — not just for AI, but philosophically:

- A traffic signal is a **rule**. It's been programmed in, static, authoritative.
- A police officer is a **human override**. It's dynamic, contextual, in-the-moment.

Which wins?

Humans answer this instantly. We recognize that a police officer at an intersection is performing a specific social role: *managing* the intersection because the signal can't. We *know* that authority is in the person, not the device.

An AI has to *learn* that same hierarchy. And it's harder than it sounds.

I tried:
- Higher weight on police authority → but then it ignores signals even when there's no police
- Temporal reasoning (who spoke most recently?) → but that's gaming the problem
- Intent parsing (why is the police there?) → requires too much context I don't have

This is one of those problems where scaling up the model helps, but it doesn't *solve* it. The solution probably requires rethinking how we encode authority and human intent in the first place.

It's on the list for the next iteration.

<!-- ---

## Demo

<!-- Replace with your actual video link -->
🎥 **[Watch the system in action — add video link]**

--- -->

## Try It Yourself

```bash
docker build -t autodrive-gym .
docker run -p 8000:8000 -e LLM_BACKEND=hf -e HF_TOKEN=hf_xxx autodrive-gym
curl http://localhost:8000/reset
```

All 7 graded tasks available at `/tasks`. Full baseline scores at `/baseline`.

*"The goal isn't to build an agent that follows signals. The goal is to build an agent that reads people."*

---

## Going Further: GRPO Fine-tuning

The multi-agent reasoning pipeline is powerful, but it has limitations.

It can't generalize to scenarios it hasn't explicitly trained on. It can't condense its reasoning into a single model. And running a 72B parameter LLM for every decision isn't going to work at 10ms latency.

So we took a different approach: **GRPO fine-tuning** — the same algorithm powering DeepSeek-R1 and QwQ.

Instead of a complex reasoning pipeline, we asked: **Can a small model learn to think like our reasoning pipeline?**

We took Qwen3-0.6B and put it directly in AutoDrive Gym. It receives observations, generates driving actions as structured JSON, gets the episode reward back. GRPO compares different rollouts of the same scenario and learns: "This decision sequence scored higher than that one. Why?"

No teacher. No labels. Just the environment as the signal.

### Why GRPO Fits Indian Road Driving

Here's the catch with traditional RL: PPO (the algorithm behind most modern RL) needs a **value function** — an estimate of how good a state is. But on Indian roads, the value of a state depends on layers of context:

The same observation (`hazard at 15m`) can warrant:
- `brake` → if the hazard is accelerating toward you
- `horn` → if it's committed to a lane change but slowing
- `accelerate` → if it's already past you and receding

A value network sees only the immediate state. It doesn't see the *sequence* of decisions and how they interact. So it struggles.

GRPO solves this by running **8 parallel rollouts of the same scenario** and comparing them:
- Rollout A: {observe, brake, wait, accelerate} → Episode reward: +2.5
- Rollout B: {observe, accelerate, observe, brake} → Episode reward: +0.8
- Rollout C: {observe, brake, brake, wait} → Episode reward: -1.2

GRPO sees the difference: A worked better than B and C. Why? It must have been the *sequence* of decisions. It learns to optimize for sequences, not states.

This is exactly what Indian road driving requires: **context-aware decision sequences**, not myopic state-value estimates.

![AutoDrive Gym HF space](https://huggingface.co/spaces/Samatha369/autodrive_grpo/blob/main/hf_space_plots.png)

### Breaking Out of Local Maxima: The Reward Structure

Here's something we learned from studying other successful RL projects: **naive reward signals plateau**.

The agent finds a local maximum and camps there. In driving, that maximum is often something useless like "always brake" or "never brake."

So we built a reward structure designed to *force* exploration and break shortcuts:

- **Repeat-action penalty**: `-0.12` per consecutive repeat, `-0.20` on the third repeat
  - *Breaks the "always brake" trap immediately*
  
- **Phase-order bonus**: `+0.08` for braking near an approaching hazard, `+0.10` for accelerating after it clears
  - *Rewards correct sequencing, not just any action*
  
- **Resolution bonus**: `+1.0` to `+3.0` at episode end, scaled by speed
  - *Faster correct decisions earn more. Teaches efficiency.*

The result: A successful fast episode scores `+3 to +8`. A failed episode scores `-2.0`. That variance is what GRPO needs to learn.

The model can't just repeat one action and hope for the best. It has to *reason through a sequence*. And after 50-100 episodes, it learns to do exactly that.

### Training in Real Time

```bash
# Terminal 1: Start the environment server
uvicorn autodrive_env.server.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Start GRPO training (requires GPU)
pip install -e ".[grpo]"
python train_grpo.py --episodes 50 --model-id Qwen/Qwen3-0.6B
```

Every episode completes in ~2-3 seconds. Metrics stream to `reward_log.json` and `live_state.json`. The Training Lab UI (`python grpo_ui.py`) reads these in real time.

You watch the reward curve climb. You watch the model learn to stop over-braking. You watch it figure out context. It all happens *live*, episode by episode. The **GRPO** tab shows the learning happen in real time.

---

## Why This Matters (Beyond the Hackathon)

Autonomous driving research has a giant blind spot.

Look at any major AV benchmark: CARLA, Waymo, nuScenes. They're all built for roads that look like California or Germany. Straight lanes, traffic lights, predictable pedestrians, clear rules.

The environments that matter most for real-world AV deployment — India, Indonesia, Nigeria, Mexico, Brazil — are almost completely absent.

That's not an accident. They're hard to benchmark. The scenarios are complex. There are fewer labeled datasets. There's less industry funding.

But they're where most of the world drives.

AutoDrive Gym is a step toward fixing that gap. It's not about Indian roads specifically. It's about building AV systems for any place where **driving is a social activity**, not just a physics problem.

The core principles — contextual zone inference, reading human intent, dynamic adaptation, resolving conflicting authorities — apply everywhere traffic is negotiated rather than regulated.

This is the future of autonomous driving research. Not just more compute. Not just more sensors. **More worlds.**

---

## Links

- 🎥 YouTube Demo: https://youtu.be/UAaQS-xngz8
- 🤗 Hugging Face Space: https://huggingface.co/spaces/Samatha369/autodrive_grpo
- 💻 GitHub Repo: https://github.com/SamathaAshokkumar/autodrive_grpo
- 📊 WandB Training Runs: https://wandb.ai/samatha45102-scaler/autodrive-gym/runs/6wbcqovs?nw=nwusersamatha45102

---

## The Vision

Remember that auto-rickshaw driver from the beginning? The one who negotiated that 4-way intersection with perfect flow?

That's not superhuman. It's not even that hard. It's just *reading the world*.

The question isn't "Can AI drive?" AI can already drive in controlled environments. The question is: **Can AI *read* the world the way humans do?**

Can it see a street vendor and infer risk? Can it watch a traffic cop's hand position and predict intent? Can it read a crowd and know when to honk and when to stay silent?

That's what AutoDrive Gym is trying to teach.

It won't be solved by one hackathon project or one model. But every scenario we design, every failure we debug, every "aha" moment the agent has — that's a brick in the foundation.

The goal isn't to build an agent that always brakes near a hospital.

The goal is to build an agent that *reads the road*.

And if we can do that on Indian streets, we can do it anywhere.

---

*"Intelligence isn't about following the rule. It's about understanding when the rule doesn't apply."*