"""grpo_ui.py -- Autonomous Driving Training Lab UI

Interactive Gradio interface for continuous background training and live analytics.

Features
--------
* Start / Stop continuous background training -- UI is a live monitor, not a trigger
* Manual "Train N Episodes" still available for step-by-step control
* Algorithm badge: always shows which algorithm is active + model version info
* Reward curve: raw + rolling-10 + trend + Q-learner vs heuristic baseline delta
* Safety metrics chart: collision rate, near-miss rate, rule-violation rate
* Scenario breakdown: success rate + mean reward per hazard type
* Curriculum progression: difficulty + epsilon decay over time
* Checkpoint tracker: best policy auto-saved; version history panel
* GRPO tab: reads live output from train_grpo.py when running simultaneously
* Live Simulation tab: real-time road scene with ego car, hazard, action trail
* Failed episode replay: last 5 failed episodes with step-by-step trace
* Chaos mode / difficulty override / scenario filter controls

Runs without a GPU -- uses PolicyLearner (Q-table) for the demo training loop.

Usage
-----
    python grpo_ui.py                  # http://localhost:7860
    python grpo_ui.py --share          # public HF Spaces link
    python grpo_ui.py --port 7861      # custom port
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gym imports
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from autodrive_env.server.autodrive_gym_environment import AutoDriveGymEnvironment
    from autodrive_env.models import AutoDriveAction, AutoDriveObservation
    from autodrive_env.server.policy_learner import PolicyLearner
    from autodrive_env.server.curriculum import CurriculumController, DIFFICULTY_TIERS
    from autodrive_env.agent_baseline import choose_action as baseline_choose
except ImportError:
    from server.autodrive_gym_environment import AutoDriveGymEnvironment
    from models import AutoDriveAction, AutoDriveObservation
    from server.policy_learner import PolicyLearner
    from server.curriculum import CurriculumController, DIFFICULTY_TIERS
    from agent_baseline import choose_action as baseline_choose

import gradio as gr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GRPO_CSV_GLOB   = "outputs/autodrive-grpo-*/reward_log.csv"
_GRPO_JSON_PATH  = Path("reward_log.json")
_CHECKPOINT_DIR  = Path("checkpoints")
_ALGORITHM_LABEL = "Q-learning (PolicyLearner / Q-table)"
# GRPO training script uses Qwen3 + TRL, labelled separately in UI
_GRPO_ALGO_LABEL = "GRPO (Qwen3 fine-tuning via TRL)"

# ---------------------------------------------------------------------------
# Logging verbosity toggle
# VERBOSE_LOGGING = True  → full episode-by-episode chatter (development/debug)
# VERBOSE_LOGGING = False → silent except warnings/errors (demo / hackathon)
# ---------------------------------------------------------------------------
VERBOSE_LOGGING: bool = True

SCENARIO_TYPES = [
    "pedestrian_crossing", "auto_cut_in", "bike_blind_spot",
    "pothole_ahead", "speed_breaker", "crowded_market",
    "ambulance_approach", "police_override", "traffic_jam",
    "animal_crossing", "rain_slippery_road", "traffic_light_ambiguity",
    "hospital_zone_slow_and_quiet", "construction_zone_one_lane",
]

_ACTION_VALUES: dict[str, float] = {
    "brake": 0.8, "accelerate": 0.7, "wait": 0.0,
    "steer_left": 0.6, "steer_right": 0.6, "horn": 0.5,
    "change_lane_left": 0.7, "change_lane_right": 0.7,
}

_PALETTE = {
    "trained":    "#2196F3",
    "baseline":   "#FF9800",
    "success":    "#4CAF50",
    "collision":  "#E53935",
    "nearmiss":   "#FF7043",
    "violation":  "#9C27B0",
    "difficulty": "#7B1FA2",
    "trend_up":   "#C62828",
    "trend_down": "#2E7D32",
    "grpo":       "#00BCD4",
    "ckpt":       "#FDD835",
}

# Action colours mirrored from the /demo page in app.py
_ACTION_COL: dict[str, str] = {
    "brake":              "#EF4444",
    "accelerate":         "#22C55E",
    "wait":               "#EAB308",
    "horn":               "#F97316",
    "steer_left":         "#3B82F6",
    "steer_right":        "#8B5CF6",
    "change_lane_left":   "#06B6D4",
    "change_lane_right":  "#14B8A6",
}

_FIG_KWARGS = {"facecolor": "#F8F9FA"}


# ---------------------------------------------------------------------------
# Checkpoint data class
# ---------------------------------------------------------------------------

@dataclass
class _Checkpoint:
    episode:  int
    reward:   float
    q_size:   int
    epsilon:  float
    scenario: str
    # Store a shallow snapshot of the Q-table (str keys for JSON)
    q_table:  dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "episode":   self.episode,
            "reward":    round(self.reward, 4),
            "q_size":    self.q_size,
            "epsilon":   round(self.epsilon, 4),
            "scenario":  self.scenario,
            "algorithm": _ALGORITHM_LABEL,
        }


# ---------------------------------------------------------------------------
# Global training state (shared between bg thread and Gradio callbacks)
# ---------------------------------------------------------------------------

@dataclass
class _TrainingState:
    """All mutable training data.  Mutations happen under ``self.lock``."""

    learner:    PolicyLearner        = field(default_factory=PolicyLearner)
    curriculum: CurriculumController = field(default_factory=CurriculumController)

    # Per-episode history -- the core data for all charts
    rewards:          list[float] = field(default_factory=list)
    successes:        list[bool]  = field(default_factory=list)
    collisions:       list[int]   = field(default_factory=list)   # 0/1 per episode
    near_misses:      list[int]   = field(default_factory=list)
    violations:       list[int]   = field(default_factory=list)   # rule violations
    difficulties:     list[float] = field(default_factory=list)
    scenarios:        list[str]   = field(default_factory=list)
    epsilons:         list[float] = field(default_factory=list)
    baseline_rewards: list[float] = field(default_factory=list)

    # Live step-by-step action log (last 60 steps shown in UI)
    action_log: list[dict] = field(default_factory=list)

    # Failed episode traces for replay panel (last 5)
    failed_traces: list[dict] = field(default_factory=list)

    # Checkpoint history
    checkpoints:  list[_Checkpoint] = field(default_factory=list)
    best_reward:  float             = field(default=-999.0)
    best_ckpt:    _Checkpoint | None = field(default=None)

    # Background training controls
    _bg_running:  bool               = field(default=False)
    _bg_thread:   threading.Thread | None = field(default=None)
    _stop_event:  threading.Event    = field(default_factory=threading.Event)

    # Settings read by background thread (written by UI)
    bg_difficulty: float = 0.0
    bg_chaos:      bool  = False
    bg_filter:     str   = "all"

    lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def episode_count(self) -> int:
        return len(self.rewards)

    @property
    def is_running(self) -> bool:
        return self._bg_running

    # ------------------------------------------------------------------
    # Checkpoint saving
    # ------------------------------------------------------------------

    def _maybe_checkpoint(self, reward: float, scenario: str) -> bool:
        """Update in-memory checkpoint state.  MUST be called with self.lock held.

        Disk I/O is deliberately NOT done here -- call _write_checkpoint_async()
        AFTER releasing the lock to avoid blocking the UI thread.
        Returns True if a new best was recorded.
        """
        if reward <= self.best_reward:
            return False
        ep = len(self.rewards)  # already appended before this call
        ckpt = _Checkpoint(
            episode  = ep,
            reward   = reward,
            q_size   = len(self.learner._q),
            epsilon  = self.learner._epsilon,
            scenario = scenario,
        )
        self.best_reward = reward
        self.best_ckpt   = ckpt
        self.checkpoints.append(ckpt)
        return True

    # ------------------------------------------------------------------
    # Background thread management
    # ------------------------------------------------------------------

    def start_background(self) -> str:
        if self._bg_running:
            return "Already running."
        self._stop_event.clear()
        self._bg_running = True
        self._bg_thread = threading.Thread(
            target=_background_training_loop,
            args=(self,),
            daemon=True,
            name="autodrive-bg-train",
        )
        self._bg_thread.start()
        return "Continuous training started."

    def stop_background(self, join_timeout: float = 5.0) -> str:
        if not self._bg_running:
            return "Not running."
        self._stop_event.set()
        self._bg_running = False
        # Wait briefly for the thread to exit cleanly so there are no zombies.
        # Use a short timeout so the UI callback doesn't block for too long.
        t = self._bg_thread
        if t is not None and t.is_alive():
            t.join(timeout=join_timeout)
        self._bg_thread = None
        return "Stopped."

    def reset(self) -> None:
        self.stop_background()
        with self.lock:
            self.learner          = PolicyLearner()
            self.curriculum       = CurriculumController()
            self.rewards          = []
            self.successes        = []
            self.collisions       = []
            self.near_misses      = []
            self.violations       = []
            self.difficulties     = []
            self.scenarios        = []
            self.epsilons         = []
            self.baseline_rewards = []
            self.action_log       = []
            self.failed_traces    = []
            self.checkpoints      = []
            self.best_reward      = -999.0
            self.best_ckpt        = None


_STATE = _TrainingState()


# ---------------------------------------------------------------------------
# Background training loop  (runs in daemon thread)
# ---------------------------------------------------------------------------

def _write_checkpoint_async(ckpt: _Checkpoint) -> None:
    """Serialize a checkpoint to disk in a short-lived daemon thread.

    Called OUTSIDE the state lock so disk I/O never blocks the training loop
    or the UI rendering thread.
    """
    def _write() -> None:
        try:
            _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "episode":   ckpt.episode,
                "reward":    round(ckpt.reward, 4),
                "q_size":    ckpt.q_size,
                "epsilon":   round(ckpt.epsilon, 4),
                "scenario":  ckpt.scenario,
                "algorithm": _ALGORITHM_LABEL,
            }
            path = _CHECKPOINT_DIR / f"policy_ep{ckpt.episode:04d}_r{ckpt.reward:.3f}.json"
            path.write_text(json.dumps(data, indent=2))
            logger.info("Checkpoint saved -> %s", path)
        except Exception as exc:
            logger.warning("Checkpoint save failed: %s", exc)

    threading.Thread(target=_write, daemon=True, name="ckpt-writer").start()


def _background_training_loop(state: _TrainingState) -> None:
    """Continuous training loop.  Writes to shared state; never touches Gradio."""
    env          = AutoDriveGymEnvironment()
    baseline_env = AutoDriveGymEnvironment()
    ep_count     = 0

    logger.info("[AUTO-DRIVE][BG-TRAIN] Continuous training loop started.")
    while not state._stop_event.is_set():
        with state.lock:
            diff   = state.bg_difficulty
            chaos  = state.bg_chaos
        ep_num = state.episode_count + ep_count

        if diff > 0.0:
            _set_env_difficulty(env, diff)
            _set_env_difficulty(baseline_env, diff)

        reward, success, collision, near_miss, violation, actions, stype = _run_episode(
            env, state.learner, ep_num, chaos
        )
        b_reward, _, _, _, _, _, _ = _run_episode(
            baseline_env, None, ep_num, chaos_mode=False
        )

        # Curriculum update + difficulty read -- bg-thread-only, done OUTSIDE lock
        # so UI callbacks are never blocked waiting for it.
        env.curriculum.record(stype or "unknown", success, len(actions), reward)
        curr_diff = (
            float(env.curriculum.get_difficulty())
            if hasattr(env, "curriculum") else float(diff or 0.2)
        )
        # Lock scope: only brief list appends -- no I/O, no computation
        new_best = False
        with state.lock:
            state.rewards.append(reward)
            state.successes.append(success)
            state.collisions.append(collision)
            state.near_misses.append(near_miss)
            state.violations.append(violation)
            state.difficulties.append(curr_diff)
            state.scenarios.append(stype or "unknown")
            state.epsilons.append(state.learner._epsilon)
            state.baseline_rewards.append(b_reward)
            state.action_log.extend(actions[-3:])
            state.action_log = state.action_log[-60:]
            if not success:
                state.failed_traces.append({"episode": ep_num + 1, "scenario": stype, "trace": actions})
                state.failed_traces = state.failed_traces[-5:]
            new_best = state._maybe_checkpoint(reward, stype or "unknown")
        # Checkpoint disk write OUTSIDE the lock -- never block the UI
        if new_best and state.best_ckpt:
            _write_checkpoint_async(state.best_ckpt)

        if VERBOSE_LOGGING:
            logger.info(
                "[CURRICULUM] scenario=%s success=%s reward=%.3f -> difficulty=%.2f",
                stype or "unknown", success, reward, curr_diff,
            )
        ep_count += 1
        if VERBOSE_LOGGING and ep_count % 5 == 0:
            logger.info(
                "[BG-TRAIN] episodes=%d latest_reward=%.3f epsilon=%.3f difficulty=%.2f",
                ep_count, reward, state.learner._epsilon, curr_diff,
            )

    logger.info("[AUTO-DRIVE][BG-TRAIN] Continuous training loop stopped after %d episodes.", ep_count)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_value(action: str, hazard_dist: float) -> float:
    if action == "brake":
        if hazard_dist < 4:   return 0.95
        if hazard_dist < 8:   return 0.82
        if hazard_dist < 14:  return 0.58
        return 0.35
    if action == "accelerate":
        return min(0.9, 0.4 + hazard_dist / 60.0)
    return _ACTION_VALUES.get(action, 0.5)


def _obs_to_dict(obs: AutoDriveObservation) -> dict[str, Any]:
    return {
        "scenario_type":   obs.scenario_type or "",
        "scenario_stage":  obs.scenario_stage or "approaching",
        "hazard_distance": obs.hazard_distance or 999.0,
        "hazard_type":     obs.hazard_type or "",
        "sensor_data":     obs.sensor_data or {},
        "ego_state":       obs.ego_state or {},
        "environment":     obs.environment or {},
        "zone_cues":       obs.zone_cues or {},
        "pipeline_trace":  obs.pipeline_trace or {},
        "active_alerts":   obs.active_alerts or [],
        "hint":            obs.hint or "",
    }


def _set_env_difficulty(env: AutoDriveGymEnvironment, target: float) -> None:
    if not hasattr(env, "curriculum"):
        return
    tier_idx = len(DIFFICULTY_TIERS) - 1
    for i, t in enumerate(DIFFICULTY_TIERS):
        if target <= t["max_diff"]:
            tier_idx = i
            break
    env.curriculum._tier_index = tier_idx


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def _run_episode(
    env:         AutoDriveGymEnvironment,
    learner:     PolicyLearner | None,
    episode_num: int,
    chaos_mode:  bool,
) -> tuple[float, bool, int, int, int, list[dict], str]:
    """Run one driving episode.

    Returns:
        (total_reward, success, collision:int, near_miss:int,
         violations:int, action_log, scenario_type)
    """
    try:
        obs: AutoDriveObservation = env.reset()
    except Exception as exc:
        logger.warning("env.reset() failed: %s", exc)
        return 0.0, False, 0, 0, 0, [], "unknown"

    total_reward   = 0.0
    action_log:    list[dict] = []
    scenario_type  = obs.scenario_type or "unknown"
    rule_violation = 0

    if chaos_mode and hasattr(env, "_simulator"):
        try:
            env._simulator.dynamic_events.insert(0, {
                "trigger_step": 3,
                "kind": "spawn_vehicle",
                "message": "Chaos: sudden cut-in",
                "hazard_type": "auto_cut_in",
            })
        except Exception:
            pass

    if learner is not None:
        learner.start_episode(scenario_type)

    prev_state:  tuple | None = None
    prev_action: str   | None = None

    for step in range(obs.max_steps or 20):
        if obs.done:
            break

        obs_dict    = _obs_to_dict(obs)
        hazard_dist = float(obs.hazard_distance or 999.0)

        if learner is not None:
            state_key  = learner.encode_state(obs_dict)
            action_str = learner.select_action(state_key, episode_num)
        else:
            result     = baseline_choose(obs_dict)
            action_str = result.get("action", "brake")

        action_val = _action_value(action_str, hazard_dist)
        action     = AutoDriveAction(action=action_str, value=action_val)

        try:
            next_obs: AutoDriveObservation = env.step(action)
        except Exception as exc:
            logger.warning("env.step() failed at step %d: %s", step, exc)
            break

        reward = float(next_obs.reward or 0.0)
        total_reward += reward

        # Count rule violations: braking hard in a zone marked "cleared" or
        # honking in a silent zone are obvious violations visible in the hint.
        hint_lower = (next_obs.hint or "").lower()
        if "penali" in hint_lower or "violat" in hint_lower or "wrong" in hint_lower:
            rule_violation += 1

        if learner is not None:
            next_state_key = learner.encode_state(_obs_to_dict(next_obs))
            if prev_state is not None:
                learner.update(prev_state, prev_action, reward, next_state_key, next_obs.done)
            prev_state  = next_state_key
            prev_action = action_str

        action_log.append({
            "step":        step + 1,
            "action":      action_str,
            "value":       round(action_val, 2),
            "reward":      round(reward, 3),
            "hazard_dist": round(hazard_dist, 1),
            "hint":        next_obs.hint or "",
            "stage":       next_obs.scenario_stage or "approaching",
            "hazard_type": next_obs.hazard_type or scenario_type,
        })
        obs = next_obs

    validation = obs.validation or {}
    collision  = int(bool(validation.get("collision")))
    near_miss  = int(bool(validation.get("near_miss")))

    # Normalise episode reward to (0.02, 0.98) — same contract as graders.py.
    # Raw sum over 20 steps can reach ~20; averaging then clamping keeps it
    # in range for consistent display in the UI and checkpoint comparisons.
    n_steps = len(action_log)
    if n_steps > 0:
        normalised_reward = max(0.02, min(0.98, total_reward / n_steps))
    else:
        normalised_reward = 0.02

    success = bool(obs.done and normalised_reward > 0.5 and not collision)

    if learner is not None:
        learner.record_validation(collision=bool(collision), near_miss=bool(near_miss))
        learner.end_episode(success, normalised_reward, scenario_type)

    if VERBOSE_LOGGING:
        logger.info(
            "[EPISODE] ep=%d scenario=%s reward=%.3f success=%s collision=%s near_miss=%s steps=%d",
            episode_num, scenario_type, normalised_reward, success,
            bool(collision), bool(near_miss), n_steps,
        )
        if not success:
            logger.warning(
                "[FAIL] ep=%d scenario=%s reward=%.3f collision=%s near_miss=%s",
                episode_num, scenario_type, normalised_reward, bool(collision), bool(near_miss),
            )

    return normalised_reward, success, collision, near_miss, rule_violation, action_log, scenario_type


# ---------------------------------------------------------------------------
# train_n_episodes (step-based, called on button click)
# ---------------------------------------------------------------------------

def train_n_episodes(
    n:                  int,
    difficulty_override: float,
    scenario_filter:    str,
    chaos_mode:         bool,
) -> tuple[dict, str]:
    """Run n episodes synchronously (used by 'Train N' button)."""
    state        = _STATE
    env          = AutoDriveGymEnvironment()
    baseline_env = AutoDriveGymEnvironment()

    with state.lock:
        ep_offset = len(state.rewards)

    for i in range(n):
        ep_num = ep_offset + i
        if difficulty_override > 0.0:
            _set_env_difficulty(env, difficulty_override)
            _set_env_difficulty(baseline_env, difficulty_override)

        reward, success, collision, near_miss, violation, actions, stype = _run_episode(
            env, state.learner, ep_num, chaos_mode
        )
        b_reward, _, _, _, _, _, _ = _run_episode(
            baseline_env, None, ep_num, chaos_mode=False
        )

        # Update curriculum and read back the new difficulty BEFORE the lock.
        # This keeps the lock scope to pure list appends only.
        env.curriculum.record(stype or "unknown", success, len(actions), reward)
        curr_diff = (
            float(env.curriculum.get_difficulty())
            if hasattr(env, "curriculum")
            else float(difficulty_override or 0.2)
        )
        new_best = False
        with state.lock:
            state.rewards.append(reward)
            state.successes.append(success)
            state.collisions.append(collision)
            state.near_misses.append(near_miss)
            state.violations.append(violation)
            state.difficulties.append(curr_diff)
            state.scenarios.append(stype or "unknown")
            state.epsilons.append(state.learner._epsilon)
            state.baseline_rewards.append(b_reward)
            state.action_log.extend(actions[-3:])
            state.action_log = state.action_log[-60:]
            if not success:
                state.failed_traces.append({"episode": ep_num + 1, "scenario": stype, "trace": actions})
                state.failed_traces = state.failed_traces[-5:]
            new_best = state._maybe_checkpoint(reward, stype or "unknown")
        if new_best and state.best_ckpt:
            _write_checkpoint_async(state.best_ckpt)

    with state.lock:
        rewards   = list(state.rewards)
        successes = list(state.successes)
        baselines = list(state.baseline_rewards)
        colls     = list(state.collisions)
        nms       = list(state.near_misses)
        viols     = list(state.violations)
        eps_val   = state.learner._epsilon
        q_size    = len(state.learner._q)
        diff_last = state.difficulties[-1] if state.difficulties else 0.0
        n_ckpts   = len(state.checkpoints)
        best_r    = state.best_reward

    recent_r  = rewards[-10:]
    recent_b  = baselines[-10:] if baselines else recent_r
    sr_last10 = sum(1 for s in successes[-10:] if s) / max(len(successes[-10:]), 1)
    cr_last10 = sum(colls[-10:]) / max(len(colls[-10:]), 1)

    stats: dict[str, Any] = {
        "algorithm":               _ALGORITHM_LABEL,
        "total_episodes":          len(rewards),
        "last_reward":             round(rewards[-1], 3) if rewards else 0.0,
        "avg_reward_last10":       round(sum(recent_r)  / max(len(recent_r),  1), 3),
        "baseline_avg_last10":     round(sum(recent_b)  / max(len(recent_b),  1), 3),
        "improvement_vs_baseline": round((sum(recent_r) - sum(recent_b)) / max(len(recent_r), 1), 3),
        "success_rate_last10_pct": round(sr_last10 * 100, 1),
        "collision_rate_last10":   round(cr_last10, 3),
        "near_miss_count_last10":  sum(nms[-10:]),
        "rule_violations_last10":  sum(viols[-10:]),
        "epsilon":                 round(eps_val, 4),
        "q_table_states":          q_size,
        "difficulty":              round(diff_last, 3),
        "checkpoints_saved":       n_ckpts,
        "best_reward_ever":        round(best_r, 3) if best_r > -999 else "none",
    }

    with state.lock:
        last_5 = list(state.action_log[-5:])
    lines = []
    for a in last_5:
        flag = "OK" if a["reward"] > 0 else "!!"
        hint = f"  <- {a['hint']}" if a["hint"] else ""
        lines.append(
            f"  [{flag}] Step {a['step']}: {a['action']:20s}"
            f"v={a['value']}  r={a['reward']:+.3f}  @{a['hazard_dist']}m{hint}"
        )
    return stats, "\n".join(lines) if lines else "No actions yet."


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _rolling(values: list[float], k: int = 10) -> list[float]:
    out = []
    for i, v in enumerate(values):
        chunk = values[max(0, i - k + 1): i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _empty_fig(title: str, msg: str = "No data yet.\nClick 'Train N Episodes' or 'Start Continuous' to begin.") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5), **_FIG_KWARGS)
    ax.set_facecolor("#F0F4F8")
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="gray")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 1: Reward curve (trained vs baseline + delta panel)
# ---------------------------------------------------------------------------

def make_reward_chart() -> plt.Figure:
    with _STATE.lock:
        rewards   = list(_STATE.rewards)
        baselines = list(_STATE.baseline_rewards)
        ckpts     = list(_STATE.checkpoints)

    if not rewards:
        return _empty_fig("Reward Curve -- Q-learner vs Heuristic Baseline")

    eps    = list(range(1, len(rewards) + 1))
    roll_r = _rolling(rewards, 10)
    roll_b = _rolling(baselines, 10) if baselines else []

    fig, (ax_main, ax_delta) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        **_FIG_KWARGS,
    )
    fig.patch.set_facecolor("#F8F9FA")

    ax_main.set_facecolor("#F0F4F8")
    ax_main.plot(eps, rewards, alpha=0.18, color=_PALETTE["trained"],
                 lw=1, marker="o", markersize=2, label="Q-learner (raw)")
    ax_main.plot(eps, roll_r, color=_PALETTE["trained"], lw=2.8,
                 label="Q-learner rolling(10)")
    if roll_b:
        ax_main.plot(eps[:len(baselines)], baselines, alpha=0.12,
                     color=_PALETTE["baseline"], lw=1, marker="s", markersize=2)
        ax_main.plot(eps[:len(roll_b)], roll_b, color=_PALETTE["baseline"],
                     lw=2, linestyle="--", label="Baseline heuristic rolling(10)")

    # Trend line
    if len(rewards) >= 3:
        z       = np.polyfit(eps, rewards, 1)
        trend   = np.poly1d(z)(eps)
        arrow   = "up" if z[0] > 0 else "down"
        tcol    = _PALETTE["trend_up"] if z[0] > 0 else _PALETTE["trend_down"]
        ax_main.plot(eps, trend, color=tcol, lw=1.6, linestyle=":",
                     label=f"Trend [{arrow}] ({z[0]:+.3f}/ep)")

    # Checkpoint markers
    for ckpt in ckpts:
        if ckpt.episode <= len(rewards):
            ax_main.axvline(ckpt.episode, color=_PALETTE["ckpt"], lw=1.0,
                            linestyle="--", alpha=0.7)
            ax_main.annotate(
                f"ckpt\nr={ckpt.reward:.2f}",
                xy=(ckpt.episode, ckpt.reward),
                xytext=(4, 6), textcoords="offset points",
                fontsize=6.5, color=_PALETTE["ckpt"],
                arrowprops=dict(arrowstyle="-", color=_PALETTE["ckpt"], lw=0.8),
            )

    ax_main.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.4)
    ax_main.set_ylabel("Episode Reward", fontsize=11)
    ax_main.set_title("Reward Curve -- Q-learner vs Heuristic Baseline", fontsize=13, fontweight="bold")
    ax_main.legend(fontsize=9, loc="lower right")
    ax_main.grid(True, alpha=0.2)

    best    = max(rewards)
    last10m = sum(rewards[-10:]) / min(len(rewards), 10)
    ax_main.annotate(
        f"Best: {best:.2f}  |  Last-10 avg: {last10m:.2f}  |  n={len(rewards)}  |  algo: Q-learning",
        xy=(0.02, 0.97), xycoords="axes fraction", fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
    )

    # Delta panel
    ax_delta.set_facecolor("#F0F4F8")
    if baselines:
        n     = min(len(rewards), len(baselines))
        delta = [rewards[i] - baselines[i] for i in range(n)]
        roll_d = _rolling(delta, 10)
        bar_colors = [_PALETTE["success"] if d >= 0 else "#EF5350" for d in delta]
        ax_delta.bar(eps[:n], delta, color=bar_colors, alpha=0.45, width=0.8)
        ax_delta.plot(eps[:n], roll_d, color="#263238", lw=1.5, label="Rolling delta")
        ax_delta.axhline(0, color="gray", lw=0.8)
        ax_delta.set_ylabel("Delta reward", fontsize=9)
        ax_delta.legend(fontsize=8, loc="upper right")
    else:
        ax_delta.set_visible(False)
        ax_main.set_xlabel("Episode", fontsize=11)

    ax_delta.set_xlabel("Episode", fontsize=11)
    ax_delta.grid(True, alpha=0.2)
    plt.tight_layout(h_pad=0.6)
    return fig


# ---------------------------------------------------------------------------
# Chart 2: Safety metrics (collision / near-miss / violation rates)
# ---------------------------------------------------------------------------

def make_safety_chart() -> plt.Figure:
    with _STATE.lock:
        colls  = list(_STATE.collisions)
        nms    = list(_STATE.near_misses)
        viols  = list(_STATE.violations)

    if not colls:
        return _empty_fig("Safety Metrics")

    eps     = list(range(1, len(colls) + 1))
    roll_c  = _rolling([float(v) for v in colls], 10)
    roll_nm = _rolling([float(v) for v in nms],   10)
    roll_vl = _rolling([float(v) for v in viols], 10)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), **_FIG_KWARGS, sharex=True)
    fig.patch.set_facecolor("#F8F9FA")

    def _panel(ax, raw, roll, colour, label, ylabel):
        ax.set_facecolor("#F0F4F8")
        ax.bar(eps, raw, color=colour, alpha=0.3, width=0.8)
        ax.plot(eps, roll, color=colour, lw=2.2, label=f"Rolling(10)  current={roll[-1]:.2f}")
        ax.axhline(0, color="gray", lw=0.6, alpha=0.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2)
        # Add trend annotation
        if len(roll) >= 5:
            trend_val = roll[-1] - roll[max(0, len(roll) - 10)]
            arrow = "improving" if trend_val < 0 else ("worsening" if trend_val > 0.01 else "stable")
            ax.annotate(
                f"Trend: {arrow}",
                xy=(0.02, 0.88), xycoords="axes fraction", fontsize=8,
                color=_PALETTE["trend_down"] if arrow == "improving" else _PALETTE["trend_up"],
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
            )

    _panel(axes[0], colls,  roll_c,  _PALETTE["collision"],  "Collision Rate",    "Collision / ep")
    _panel(axes[1], nms,    roll_nm, _PALETTE["nearmiss"],   "Near-Miss Rate",    "Near miss / ep")
    _panel(axes[2], viols,  roll_vl, _PALETTE["violation"],  "Rule Violations",   "Violations / ep")
    axes[2].set_xlabel("Episode", fontsize=11)

    plt.suptitle("Safety Metrics Over Training", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ---------------------------------------------------------------------------
# Chart 3: Scenario breakdown (success rate + mean reward)
# ---------------------------------------------------------------------------

def make_scenario_chart() -> plt.Figure:
    with _STATE.lock:
        scenarios = list(_STATE.scenarios)
        rewards   = list(_STATE.rewards)
        successes = list(_STATE.successes)

    if not scenarios:
        return _empty_fig("Scenario Breakdown")

    sc_rewards: dict[str, list[float]] = defaultdict(list)
    sc_success: dict[str, list[bool]]  = defaultdict(list)
    for s, r, ok in zip(scenarios, rewards, successes):
        sc_rewards[s].append(r)
        sc_success[s].append(ok)

    types   = sorted(sc_rewards, key=lambda t: -sum(sc_success[t]) / max(len(sc_success[t]), 1))
    sr_vals = [sum(sc_success[t]) / max(len(sc_success[t]), 1) * 100 for t in types]
    avg_r   = [sum(sc_rewards[t]) / max(len(sc_rewards[t]), 1)       for t in types]
    counts  = [len(sc_rewards[t]) for t in types]
    y       = np.arange(len(types))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, max(5, len(types) * 0.55 + 2)), **_FIG_KWARGS
    )
    fig.patch.set_facecolor("#F8F9FA")

    c1 = [_PALETTE["success"] if v >= 50 else "#FF7043" for v in sr_vals]
    bars1 = ax1.barh(y, sr_vals, color=c1, alpha=0.85, height=0.6)
    ax1.set_facecolor("#F0F4F8")
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{t}\n(n={c})" for t, c in zip(types, counts)], fontsize=8)
    ax1.set_xlabel("Success Rate (%)", fontsize=10)
    ax1.set_title("Success Rate by Scenario", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 115)
    ax1.axvline(50, color="gray", lw=0.9, linestyle="--", alpha=0.5)
    ax1.grid(True, axis="x", alpha=0.2)
    for bar, v in zip(bars1, sr_vals):
        ax1.text(v + 1.5, bar.get_y() + bar.get_height() / 2,
                 f"{v:.0f}%", va="center", fontsize=8)

    c2    = [_PALETTE["trained"] if v >= 0 else "#EF5350" for v in avg_r]
    bars2 = ax2.barh(y, avg_r, color=c2, alpha=0.85, height=0.6)
    ax2.set_facecolor("#F0F4F8")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Mean Episode Reward", fontsize=10)
    ax2.set_title("Mean Reward by Scenario", fontsize=12, fontweight="bold")
    ax2.axvline(0, color="gray", lw=0.9, linestyle="--", alpha=0.5)
    ax2.grid(True, axis="x", alpha=0.2)
    for bar, v in zip(bars2, avg_r):
        xpos = v + 0.015 if v >= 0 else v - 0.015
        ax2.text(xpos, bar.get_y() + bar.get_height() / 2,
                 f"{v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)

    plt.suptitle("Scenario Performance Breakdown", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 4: Curriculum + epsilon
# ---------------------------------------------------------------------------

def make_difficulty_chart() -> plt.Figure:
    with _STATE.lock:
        difficulties = list(_STATE.difficulties)
        epsilons     = list(_STATE.epsilons)

    if not difficulties:
        return _empty_fig("Curriculum Progression")

    eps_ax = list(range(1, len(difficulties) + 1))

    fig, ax1 = plt.subplots(figsize=(12, 5), **_FIG_KWARGS)
    ax1.set_facecolor("#F0F4F8")

    tier_bands = [
        (0.00, 0.25, "#E3F2FD", "Warmup"),
        (0.25, 0.45, "#E8F5E9", "Beginner"),
        (0.45, 0.60, "#FFF9C4", "Intermediate"),
        (0.60, 0.75, "#FBE9E7", "Advanced"),
        (0.75, 1.00, "#FCE4EC", "Expert"),
    ]
    for ylo, yhi, colour, label in tier_bands:
        ax1.axhspan(ylo, yhi, alpha=0.28, color=colour, label=label)

    ax1.fill_between(eps_ax, difficulties, alpha=0.12, color=_PALETTE["difficulty"])
    ax1.plot(eps_ax, difficulties, color=_PALETTE["difficulty"], lw=2.5,
             label="Curriculum difficulty")

    ax2 = ax1.twinx()
    if epsilons:
        ax2.plot(eps_ax[:len(epsilons)], epsilons, color="#FF7043",
                 lw=1.8, linestyle="--", alpha=0.85, label="epsilon (exploration)")
        ax2.set_ylabel("Exploration rate (epsilon)", fontsize=10, color="#FF7043")
        ax2.set_ylim(0, max(epsilons) * 1.15)
        ax2.tick_params(axis="y", colors="#FF7043")
        ax2.legend(fontsize=9, loc="upper right")

    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Difficulty", fontsize=11)
    ax1.set_ylim(0, 1.07)
    ax1.set_title("Curriculum Progression + Exploration Decay", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 5: GRPO progress
# ---------------------------------------------------------------------------

def make_grpo_chart() -> tuple[plt.Figure, str]:
    csvs = sorted(glob.glob(_GRPO_CSV_GLOB), reverse=True)
    grpo_rewards:   list[float] = []
    grpo_successes: list[int]   = []
    source_info = ""

    if csvs:
        try:
            with open(csvs[0], newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 3:
                        grpo_rewards.append(float(row[1]))
                        grpo_successes.append(int(row[2]))
            source_info = (
                f"Source: {Path(csvs[0]).parent.name}/reward_log.csv  |  "
                f"algo: {_GRPO_ALGO_LABEL}  |  "
                f"n={len(grpo_rewards)}  |  "
                f"SR={sum(grpo_successes)/max(len(grpo_successes),1):.1%}"
            )
        except Exception as exc:
            return _empty_fig("GRPO Training Progress", f"Error reading CSV:\n{exc}"), f"Error: {exc}"

    elif _GRPO_JSON_PATH.exists():
        try:
            with open(_GRPO_JSON_PATH) as f:
                data = json.load(f)
            grpo_rewards   = [float(r) for r in data.get("reward_curve",  [])]
            grpo_successes = [int(s)   for s in data.get("success_curve", [])]
            live = {}
            try:
                live = json.loads(Path("live_state.json").read_text())
            except Exception:
                pass
            source_info = (
                f"algo: {_GRPO_ALGO_LABEL}  |  "
                f"n={len(grpo_rewards)}  |  "
                f"SR={sum(grpo_successes)/max(len(grpo_successes),1):.1%}"
                + (f"  |  difficulty={live.get('difficulty','?')}  tier={live.get('tier','?')}" if live else "")
            )
        except Exception as exc:
            return _empty_fig("GRPO Training Progress", f"Error reading JSON:\n{exc}"), f"Error: {exc}"

    if not grpo_rewards:
        fig = _empty_fig(
            "GRPO Training Progress",
            "No GRPO run data found.\n\n"
            "Run  python train_grpo.py --episodes 50  in a separate terminal,\n"
            "then click  Refresh GRPO Progress  to see the reward curve here.",
        )
        return fig, "No GRPO output found. Run `python train_grpo.py --episodes 50` first."

    return _plot_grpo_rewards(grpo_rewards, grpo_successes), source_info


def _plot_grpo_rewards(rewards: list[float], successes: list[int] | None = None) -> plt.Figure:
    eps    = list(range(1, len(rewards) + 1))
    roll_r = _rolling(rewards, 10)
    has_sr = bool(successes) and len(successes) == len(rewards)
    h_ratios = [3, 1] if has_sr else [1]

    fig, axes = plt.subplots(
        2 if has_sr else 1, 1,
        figsize=(12, 7 if has_sr else 5),
        gridspec_kw={"height_ratios": h_ratios},
        **_FIG_KWARGS,
    )
    ax_top = axes[0] if has_sr else axes
    ax_top.set_facecolor("#F0F4F8")

    ax_top.plot(eps, rewards, alpha=0.22, color=_PALETTE["grpo"],
                lw=1, marker="o", markersize=2.5, label="GRPO reward (raw)")
    ax_top.plot(eps, roll_r, color=_PALETTE["grpo"], lw=2.8, label="Rolling avg (10)")

    if len(rewards) >= 3:
        z     = np.polyfit(eps, rewards, 1)
        trend = np.poly1d(z)(eps)
        arrow = "up" if z[0] > 0 else "down"
        tcol  = _PALETTE["trend_up"] if z[0] > 0 else _PALETTE["trend_down"]
        ax_top.plot(eps, trend, color=tcol, lw=1.6, linestyle=":",
                    label=f"Trend [{arrow}] ({z[0]:+.3f}/ep)")

    ax_top.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.4)
    ax_top.set_title(f"GRPO Fine-Tuning ({_GRPO_ALGO_LABEL})", fontsize=12, fontweight="bold")
    ax_top.set_ylabel("Total Episode Reward", fontsize=11)
    ax_top.legend(fontsize=9, loc="lower right")
    ax_top.grid(True, alpha=0.22)

    best    = max(rewards)
    last10m = sum(rewards[-10:]) / min(len(rewards), 10)
    ax_top.annotate(
        f"Best: {best:.2f}  |  Last-10 avg: {last10m:.2f}  |  n={len(rewards)}",
        xy=(0.02, 0.97), xycoords="axes fraction", fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
    )

    if has_sr:
        ax_bot = axes[1]
        ax_bot.set_facecolor("#F0F4F8")
        sr_roll = _rolling([float(s) for s in successes], 10)
        ax_bot.fill_between(eps, sr_roll, alpha=0.35, color=_PALETTE["success"])
        ax_bot.plot(eps, sr_roll, color=_PALETTE["success"], lw=1.8, label="Success rate rolling(10)")
        ax_bot.set_ylim(0, 1.08)
        ax_bot.set_ylabel("Success rate", fontsize=9)
        ax_bot.axhline(0.5, color="gray", lw=0.8, linestyle="--", alpha=0.5)
        ax_bot.legend(fontsize=8, loc="upper right")
        ax_bot.set_xlabel("Episode", fontsize=11)
        ax_bot.grid(True, alpha=0.2)
    else:
        ax_top.set_xlabel("Episode", fontsize=11)

    plt.tight_layout(h_pad=0.6)
    return fig


# ---------------------------------------------------------------------------
# Chart 6: Live road simulation
# ---------------------------------------------------------------------------

_HAZARD_ICONS: dict[str, str] = {
    "pedestrian":     "🚶",
    "bike":           "🏍",
    "auto":           "🛺",
    "auto_cut_in":    "🛺",
    "car":            "🚗",
    "truck":          "🚚",
    "pothole":        "⚠️",
    "pothole_ahead":  "⚠️",
    "ambulance":      "🚑",
    "ambulance_approach": "🚑",
    "animal":         "🐄",
    "animal_crossing": "🐄",
    "traffic_police": "👮",
    "police_override": "👮",
    "speed_breaker":  "🚧",
    "bike_blind_spot": "🏍",
    "pedestrian_crossing": "🚶",
}


def make_simulation_chart() -> plt.Figure:
    """Render the current driving state as a live road-scene figure.

    Layout:
    ┌──────────────────────────────┬────────────┐
    │  Road scene (car + hazard)   │ Live stats │
    ├──────────────────────────────┴────────────┤
    │  Action trail (step rewards per action)   │
    └───────────────────────────────────────────┘
    """
    with _STATE.lock:
        action_log = list(_STATE.action_log[-30:])
        scenarios  = list(_STATE.scenarios)
        rewards    = list(_STATE.rewards)
        successes  = list(_STATE.successes)
        running    = _STATE._bg_running
        episodes   = len(rewards)

    latest   = action_log[-1] if action_log else None
    scenario = scenarios[-1] if scenarios else "waiting for training..."

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6), facecolor="#0F172A")
    gs  = fig.add_gridspec(
        2, 2,
        height_ratios=[2.2, 1],
        width_ratios=[3.2, 1],
        hspace=0.15,
        wspace=0.08,
    )
    ax_road  = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_trail = fig.add_subplot(gs[1, :])

    # ── Road scene ────────────────────────────────────────────────────────
    ax_road.set_facecolor("#0F172A")
    ax_road.set_xlim(0, 10)
    ax_road.set_ylim(0, 4.2)
    ax_road.axis("off")

    # Sky gradient (simple fill)
    sky = mpatches.Rectangle((0, 3.2), 10, 1.0, color="#1E3A5F", zorder=0)
    ax_road.add_patch(sky)

    # Road surface
    road = mpatches.Rectangle((0, 0.7), 10, 2.5, color="#374151", zorder=1)
    ax_road.add_patch(road)

    # Kerb lines
    ax_road.add_patch(mpatches.Rectangle((0, 0.65), 10, 0.08, color="#94A3B8", alpha=0.4, zorder=2))
    ax_road.add_patch(mpatches.Rectangle((0, 3.12), 10, 0.08, color="#94A3B8", alpha=0.4, zorder=2))

    # Lane dashes (centre line)
    for x in np.arange(0.4, 9.6, 1.2):
        ax_road.plot([x, x + 0.75], [1.95, 1.95], color="white", alpha=0.22, lw=1.4, zorder=2)

    # ── Determine current state ────────────────────────────────────────────
    hd         = float(latest["hazard_dist"])  if latest else 999.0
    cur_action = latest["action"]              if latest else "wait"
    step_r     = float(latest["reward"])       if latest else 0.0
    hint       = (latest.get("hint") or "").lower()
    step_num   = latest["step"]                if latest else 0
    haz_type   = latest.get("hazard_type") or scenario.split("_")[0]

    # Stage inference from hint
    if "cleared" in hint or "accelerate now" in hint:
        stage     = "cleared"
        stage_col = "#22C55E"
    elif "clearing" in hint or "easing" in hint:
        stage     = "clearing"
        stage_col = "#EAB308"
    else:
        stage     = "approaching"
        stage_col = "#EF4444"

    # ── Ego vehicle ────────────────────────────────────────────────────────
    ego_x, ego_y = 1.3, 1.9
    ego_box = mpatches.FancyBboxPatch(
        (ego_x - 0.45, ego_y - 0.28), 0.9, 0.56,
        boxstyle="round,pad=0.06",
        facecolor="#1D4ED8", edgecolor="#93C5FD", lw=1.8, zorder=5,
    )
    ax_road.add_patch(ego_box)
    ax_road.text(ego_x, ego_y + 0.01, "🚗", ha="center", va="center", fontsize=15, zorder=6)

    # Action badge under ego car
    a_col = _ACTION_COL.get(cur_action, "#94A3B8")
    ax_road.text(
        ego_x, ego_y - 0.55,
        cur_action.replace("_", " ").upper(),
        ha="center", va="top", fontsize=8.5, fontweight="bold",
        color=a_col, zorder=7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#0F172A",
                  alpha=0.9, edgecolor=a_col, lw=1.1),
    )

    # ── Hazard object ──────────────────────────────────────────────────────
    MAX_DIST = 40.0
    if hd < MAX_DIST and latest:
        t     = hd / MAX_DIST          # 0 = close (near ego), 1 = far
        haz_x = min(ego_x + 0.9 + t * 7.0, 9.4)
        icon  = _HAZARD_ICONS.get(haz_type, "⚠️")

        # Glow circle behind icon
        ax_road.plot(haz_x, ego_y, "o", markersize=26, color=stage_col, alpha=0.22, zorder=3)
        ax_road.text(haz_x, ego_y, icon, ha="center", va="center", fontsize=16, zorder=4)

        # Distance label above hazard
        ax_road.text(
            haz_x, ego_y + 0.60,
            f"{hd:.1f} m",
            ha="center", color=stage_col, fontsize=9, fontweight="bold", zorder=5,
        )

        # Range line ego → hazard
        ax_road.annotate(
            "", xy=(haz_x - 0.25, ego_y), xytext=(ego_x + 0.45, ego_y),
            arrowprops=dict(
                arrowstyle="->", color=stage_col, lw=1.2,
                connectionstyle="arc3,rad=0.0", alpha=0.5,
            ),
            zorder=3,
        )

    # ── Stage + step reward overlay ────────────────────────────────────────
    ax_road.text(
        9.6, 3.85,
        f"STAGE: {stage.upper()}",
        ha="right", color=stage_col, fontsize=10, fontweight="bold", zorder=7,
    )
    r_col = "#22C55E" if step_r > 0 else "#EF4444"
    ax_road.text(
        9.6, 3.5,
        f"Step reward: {step_r:+.3f}",
        ha="right", color=r_col, fontsize=9, zorder=7,
    )

    # ── Scenario title ─────────────────────────────────────────────────────
    status_dot = "●" if running else "○"
    sc_label   = scenario.replace("_", " ").title()
    ax_road.set_title(
        f"{status_dot} {'LIVE' if running else 'IDLE'}  —  {sc_label}  —  Step {step_num}",
        color="white", fontsize=11, fontweight="bold", pad=5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E293B", alpha=0.9, edgecolor="#334155"),
    )

    # ── Stats panel ────────────────────────────────────────────────────────
    ax_stats.set_facecolor("#0F172A")
    ax_stats.axis("off")

    sr10   = (sum(1 for s in successes[-10:] if s) / max(len(successes[-10:]), 1) * 100)
    last_r = rewards[-1] if rewards else 0.0
    roll10 = sum(rewards[-10:]) / max(len(rewards[-10:]), 1) if rewards else 0.0
    best_r = max(rewards) if rewards else 0.0

    stat_rows: list[tuple[str, str, str]] = [
        ("Episodes",      f"{episodes}",          "#94A3B8"),
        ("Last reward",   f"{last_r:+.3f}",        "#94A3B8"),
        ("Rolling-10",    f"{roll10:.3f}",          "#3B82F6"),
        ("Best reward",   f"{best_r:.3f}",          "#FDD835"),
        ("Success (10)",  f"{sr10:.0f}%",
            "#22C55E" if sr10 >= 50 else "#EF4444"),
        ("Status",        "RUNNING" if running else "IDLE",
            "#22C55E" if running else "#94A3B8"),
    ]
    for i, (label, val, col) in enumerate(stat_rows):
        yp = 0.92 - i * 0.155
        ax_stats.text(0.06, yp, label, ha="left",  color="#475569", fontsize=9,
                      transform=ax_stats.transAxes)
        ax_stats.text(0.94, yp, val,   ha="right", color=col,       fontsize=10,
                      fontweight="bold", transform=ax_stats.transAxes)

    # ── Action trail ──────────────────────────────────────────────────────
    ax_trail.set_facecolor("#1E293B")
    ax_trail.spines[:].set_color("#334155")

    if action_log:
        steps_t   = [a["step"]   for a in action_log]
        rewards_t = [a["reward"] for a in action_log]
        actions_t = [a["action"] for a in action_log]
        bar_cols  = [_ACTION_COL.get(a, "#94A3B8") for a in actions_t]

        ax_trail.bar(steps_t, rewards_t, color=bar_cols, alpha=0.75, width=0.65, zorder=2)
        ax_trail.axhline(0, color="#475569", lw=0.9, zorder=1)
        ax_trail.set_ylabel("Step reward", fontsize=8, color="#94A3B8")
        ax_trail.tick_params(colors="#94A3B8", labelsize=7)

        # Colour legend (first 4 actions)
        shown = sorted({a["action"] for a in action_log})[:6]
        for act in shown:
            ax_trail.bar([], [], color=_ACTION_COL.get(act, "#94A3B8"),
                         alpha=0.75, label=act)
        ax_trail.legend(
            fontsize=7, loc="upper right", ncol=min(len(shown), 6),
            facecolor="#1E293B", edgecolor="#334155", labelcolor="#CBD5E1",
        )
    else:
        ax_trail.text(
            0.5, 0.5,
            "Action trail appears here once training starts",
            ha="center", va="center",
            transform=ax_trail.transAxes, fontsize=10, color="#475569",
        )

    ax_trail.set_xlabel("Step number (last 30 steps)", fontsize=8, color="#94A3B8")
    ax_trail.set_title(
        "Per-Step Rewards Coloured by Action Type",
        fontsize=9, color="#94A3B8", pad=2,
    )

    fig.patch.set_facecolor("#0F172A")
    plt.tight_layout(pad=0.8)
    return fig


# ---------------------------------------------------------------------------
# Live status helpers (called by polling or refresh buttons)
# ---------------------------------------------------------------------------

def _get_status_text() -> str:
    state = _STATE
    with state.lock:
        n          = len(state.rewards)
        running    = state._bg_running
        algo       = _ALGORITHM_LABEL
        eps_val    = state.learner._epsilon
        diff       = state.difficulties[-1] if state.difficulties else 0.0
        n_ckpts    = len(state.checkpoints)
        best_r     = state.best_reward
        coll_last  = sum(state.collisions[-10:])
        sr_last    = sum(1 for s in state.successes[-10:] if s)

    status   = "RUNNING" if running else "IDLE"
    best_str = f"{best_r:.3f}" if best_r > -999 else "none"
    return (
        f"[{status}]  Algorithm: {algo}\n"
        f"Episodes: {n}  |  Difficulty: {diff:.2f}  |  Epsilon: {eps_val:.3f}\n"
        f"Success(last10): {sr_last}/10  |  Collisions(last10): {coll_last}  |  "
        f"Checkpoints: {n_ckpts}  |  Best reward: {best_str}"
    )


def _get_checkpoint_history() -> str:
    with _STATE.lock:
        ckpts = list(_STATE.checkpoints)
    if not ckpts:
        return "No checkpoints saved yet. Best reward will auto-save on a new high."
    lines = ["Version  Episode  Reward    Q-states  Epsilon   Scenario"]
    lines.append("-" * 70)
    for i, c in enumerate(ckpts, 1):
        lines.append(
            f"  v{i:<5}  ep{c.episode:<6}  {c.reward:+.3f}    {c.q_size:<8}  {c.epsilon:.4f}    {c.scenario}"
        )
    with _STATE.lock:
        best = _STATE.best_ckpt
    if best:
        lines.append(f"\nBest so far: v{len(ckpts)} -- ep{best.episode} -- reward={best.reward:.3f}")
    return "\n".join(lines)


def _get_failed_replay() -> str:
    with _STATE.lock:
        traces = list(_STATE.failed_traces)
    if not traces:
        return "No failed episodes recorded yet."
    lines = []
    for t in reversed(traces):
        lines.append(f"\n--- Episode {t['episode']} | Scenario: {t['scenario']} ---")
        for step in t["trace"]:
            flag = "OK" if step["reward"] > 0 else "!!"
            lines.append(
                f"  [{flag}] Step {step['step']:2d}: {step['action']:20s}"
                f"r={step['reward']:+.3f}  @{step['hazard_dist']}m"
                + (f"  <- {step['hint']}" if step["hint"] else "")
            )
    return "\n".join(lines) if lines else "No failed episodes yet."


# ---------------------------------------------------------------------------
# Gradio callback functions
# ---------------------------------------------------------------------------

def refresh_all_charts() -> tuple:
    """Refresh all charts from the current training state (for live monitoring)."""
    grpo_fig, grpo_info = make_grpo_chart()
    return (
        make_reward_chart(),
        make_safety_chart(),
        make_scenario_chart(),
        make_difficulty_chart(),
        grpo_fig,
        grpo_info,
        make_simulation_chart(),
        _get_status_text(),
        _get_checkpoint_history(),
        _get_failed_replay(),
    )


def train_and_visualize(n: int, difficulty: float, scenario_filter: str, chaos: bool) -> tuple:
    """Button click: run N episodes synchronously, refresh all."""
    n = max(1, int(n))
    stats, latest = train_n_episodes(n, float(difficulty), scenario_filter, bool(chaos))
    grpo_fig, grpo_info = make_grpo_chart()
    return (
        stats,
        latest,
        make_reward_chart(),
        make_safety_chart(),
        make_scenario_chart(),
        make_difficulty_chart(),
        grpo_fig,
        grpo_info,
        make_simulation_chart(),
        _get_status_text(),
        _get_checkpoint_history(),
        _get_failed_replay(),
    )


def start_continuous(difficulty: float, chaos: bool, scenario_filter: str) -> str:
    with _STATE.lock:
        _STATE.bg_difficulty = float(difficulty)
        _STATE.bg_chaos      = bool(chaos)
        _STATE.bg_filter     = scenario_filter
    return _STATE.start_background()


def stop_continuous() -> str:
    return _STATE.stop_background()


def refresh_grpo() -> tuple:
    fig, info = make_grpo_chart()
    return fig, info


def reset_training() -> tuple:
    _STATE.reset()
    empty_reward   = _empty_fig("Reward Curve -- Q-learner vs Heuristic Baseline")
    empty_safety   = _empty_fig("Safety Metrics")
    empty_scenario = _empty_fig("Scenario Breakdown")
    empty_diff     = _empty_fig("Curriculum Progression")
    empty_grpo     = _empty_fig("GRPO Training Progress", "No GRPO data. Reset complete.")
    empty_sim      = _empty_fig("Live Simulation", "Training reset.\nStart training to see the road scene.")
    # Must match _all_chart_outputs order exactly:
    # reward_plot, safety_plot, scenario_plot, difficulty_plot,
    # grpo_plot, grpo_info_md, simulation_plot,
    # status_bar, checkpoint_text, failed_text
    return (
        {},                                    # stats_json
        "Training state reset.",               # latest_action
        empty_reward,                          # reward_plot
        empty_safety,                          # safety_plot
        empty_scenario,                        # scenario_plot
        empty_diff,                            # difficulty_plot
        empty_grpo,                            # grpo_plot
        "*No GRPO data — reset complete.*",    # grpo_info_md (Markdown)
        empty_sim,                             # simulation_plot
        "Reset complete.",                     # status_bar
        "No checkpoints yet.",                 # checkpoint_text
        "No failed episodes yet.",             # failed_text
    )


# ---------------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------------

_SCENARIO_CHOICES = ["all"] + SCENARIO_TYPES

_HEADER_MD = """
# Autonomous Driving Training Lab -- Indian Road Conditions

Two training modes:
- **Continuous** (Start/Stop): background thread trains indefinitely -- click Refresh to see live progress
- **Step-by-step** (Train N): run exactly N episodes on demand

Algorithm in use: Q-learning (PolicyLearner). GRPO (Qwen3 fine-tuning) shown in the GRPO tab when `train_grpo.py` is running separately.
"""


def build_ui() -> gr.Blocks:
    # theme is passed to launch() for Gradio 6 compatibility
    with gr.Blocks(title="AutoDrive Training Lab") as demo:

        gr.Markdown(_HEADER_MD)

        # ---- Status bar --------------------------------------------------
        status_bar = gr.Textbox(
            label="Live Status",
            value=_get_status_text(),
            lines=3,
            interactive=False,
        )

        with gr.Row():

            # ---- Left: controls ------------------------------------------
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Training Controls")

                n_slider = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="Episodes per click (step-by-step mode)",
                )
                diff_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                    label="Difficulty override  (0 = auto curriculum)",
                )
                scenario_dd = gr.Dropdown(
                    choices=_SCENARIO_CHOICES,
                    value="all",
                    label="Scenario filter",
                )
                chaos_cb = gr.Checkbox(label="Chaos mode  (multi-hazard episodes)", value=False)

                with gr.Row():
                    train_btn   = gr.Button("Train N Episodes",  variant="primary", scale=2)
                    refresh_btn = gr.Button("Refresh Charts",    variant="secondary", scale=2)

                with gr.Row():
                    start_btn = gr.Button("Start Continuous", variant="primary",   scale=2)
                    stop_btn  = gr.Button("Stop Continuous",  variant="stop",      scale=2)
                    reset_btn = gr.Button("Reset",            variant="stop",      scale=1)

                bg_msg = gr.Textbox(label="Background training status", lines=2, interactive=False)

                gr.Markdown("---")
                gr.Markdown("### Episode Stats")
                stats_json = gr.JSON(label="", show_label=False)

                gr.Markdown("### Latest Agent Decisions")
                latest_action = gr.Textbox(label="", lines=7, show_label=False,
                    placeholder="Decisions appear here after training...")

            # ---- Right: chart tabs ---------------------------------------
            with gr.Column(scale=3):
                with gr.Tabs():

                    with gr.Tab("Live Simulation"):
                        gr.Markdown(
                            "Real-time road scene rendered from the training state.  "
                            "🚗 ego car on the left — hazard object positioned by distance.  "
                            "Stage colours: 🔴 approaching · 🟡 clearing · 🟢 cleared.  "
                            "Bottom bar = per-step reward coloured by action type.  "
                            "Click **Refresh Charts** or **Refresh Scene** to update."
                        )
                        simulation_plot = gr.Plot(show_label=False)
                        sim_refresh = gr.Button("Refresh Scene", variant="secondary")

                    with gr.Tab("Reward Curve"):
                        gr.Markdown("Blue = Q-learner, orange = fixed baseline. Yellow markers = new-best-reward checkpoints. Bottom panel = per-episode delta.")
                        reward_plot = gr.Plot(show_label=False)

                    with gr.Tab("Safety Metrics"):
                        gr.Markdown("Collision rate, near-miss rate, and rule violations per episode (rolling-10). Downward trend = safer driving.")
                        safety_plot = gr.Plot(show_label=False)

                    with gr.Tab("Scenario Breakdown"):
                        gr.Markdown("Success rate and mean reward per hazard type. Red = below 50% -- agent still weak here.")
                        scenario_plot = gr.Plot(show_label=False)

                    with gr.Tab("Curriculum Progress"):
                        gr.Markdown("Difficulty climbs through tiers as agent succeeds. Epsilon decays (exploration -> exploitation).")
                        difficulty_plot = gr.Plot(show_label=False)

                    with gr.Tab("GRPO Progress"):
                        gr.Markdown(
                            "Live GRPO (LLM fine-tuning) progress from `train_grpo.py`.  "
                            "Run `python train_grpo.py --episodes 50` in a separate terminal, "
                            "then click **Refresh GRPO** to see results."
                        )
                        grpo_info_md  = gr.Markdown("*No GRPO run detected yet.*")
                        grpo_plot     = gr.Plot(show_label=False)
                        grpo_refresh  = gr.Button("Refresh GRPO Progress", variant="secondary")

                    with gr.Tab("Checkpoints"):
                        gr.Markdown("Best reward checkpoints saved automatically. Policy is saved to `checkpoints/` on each new high score.")
                        checkpoint_text = gr.Textbox(
                            label="Checkpoint history",
                            value=_get_checkpoint_history(),
                            lines=15, interactive=False,
                        )
                        ckpt_refresh = gr.Button("Refresh Checkpoints", variant="secondary")

                    with gr.Tab("Failed Episode Replay"):
                        gr.Markdown("Last 5 failed episodes -- step-by-step trace. Useful for debugging which scenario the agent still fails on.")
                        failed_text = gr.Textbox(
                            label="Failed episodes",
                            value=_get_failed_replay(),
                            lines=20, interactive=False,
                        )
                        failed_refresh = gr.Button("Refresh Replay Log", variant="secondary")

        # ---- Wire callbacks ------------------------------------------

        _all_chart_outputs = [
            reward_plot, safety_plot, scenario_plot, difficulty_plot,
            grpo_plot, grpo_info_md,
            simulation_plot,
            status_bar, checkpoint_text, failed_text,
        ]

        train_btn.click(
            fn=train_and_visualize,
            inputs=[n_slider, diff_slider, scenario_dd, chaos_cb],
            outputs=[stats_json, latest_action] + _all_chart_outputs,
        )

        refresh_btn.click(
            fn=refresh_all_charts,
            inputs=[],
            outputs=_all_chart_outputs,
        )

        sim_refresh.click(
            fn=make_simulation_chart,
            inputs=[],
            outputs=[simulation_plot],
        )

        start_btn.click(
            fn=start_continuous,
            inputs=[diff_slider, chaos_cb, scenario_dd],
            outputs=[bg_msg],
        )

        stop_btn.click(
            fn=stop_continuous,
            inputs=[],
            outputs=[bg_msg],
        )

        reset_btn.click(
            fn=reset_training,
            inputs=[],
            outputs=[stats_json, latest_action] + _all_chart_outputs,
        )

        grpo_refresh.click(
            fn=refresh_grpo,
            inputs=[],
            outputs=[grpo_plot, grpo_info_md],
        )

        ckpt_refresh.click(
            fn=_get_checkpoint_history,
            inputs=[],
            outputs=[checkpoint_text],
        )

        failed_refresh.click(
            fn=_get_failed_replay,
            inputs=[],
            outputs=[failed_text],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # -----------------------------------------------------------------------
    # Logging: prefer colorlog for coloured output; fall back to plain text.
    # Failures = red, WARN = yellow, INFO = green, so training signal pops.
    # -----------------------------------------------------------------------
    try:
        import colorlog  # optional dep -- `pip install colorlog`
        _ch = colorlog.StreamHandler()
        _ch.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG":    "cyan",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            },
        ))
        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(_ch)
    except ImportError:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )

    parser = argparse.ArgumentParser(description="AutoDrive Gym -- Training Lab UI")
    parser.add_argument("--share",   action="store_true", help="Create public Gradio link")
    parser.add_argument("--port",    type=int, default=8000)
    parser.add_argument("--server",  default="0.0.0.0")
    parser.add_argument("--verbose", action="store_true", help="Override VERBOSE_LOGGING at runtime")
    args = parser.parse_args()

    global VERBOSE_LOGGING
    if args.verbose:
        VERBOSE_LOGGING = True

    logger.info("[AUTO-DRIVE][INIT] AutoDrive Training Lab starting on http://%s:%d", args.server, args.port)
    demo = build_ui()
    # Build theme safely -- older Gradio versions may not have all attributes
    _theme = None
    try:
        _theme = gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="orange",
            neutral_hue="slate",
        )
    except Exception:
        pass
    # inbrowser is intentionally omitted: headless containers (HF Spaces, Docker)
    # have no display to open.  Gradio 6 also removed/deprecated this parameter.
    launch_kwargs: dict = dict(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
    )
    if _theme is not None:
        launch_kwargs["theme"] = _theme
    logger.info("[AUTO-DRIVE][INIT] Launching Gradio -- server_name=%s server_port=%d share=%s",
                args.server, args.port, args.share)
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()