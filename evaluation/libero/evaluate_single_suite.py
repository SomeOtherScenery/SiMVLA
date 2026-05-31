#!/usr/bin/env python3
"""
Single-suite LIBERO evaluation with parallel rollouts.

Each task's rollout episodes run in parallel via subprocess workers, all
connecting to a shared SimVLA policy server.

Usage:
  python evaluate_single_suite.py --task_suite libero_goal --num_trials 50 --num_workers 4 --port 8102
"""

from __future__ import annotations

import argparse
import collections
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import imageio
import numpy as np

# EGL offscreen rendering must be set before MuJoCo imports
os.environ.setdefault("MUJOCO_GL", "egl")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as ws_client

    HAS_WS_CLIENT = True
except ImportError:
    HAS_WS_CLIENT = False

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

MAX_STEPS = {
    "libero_spatial": 800,
    "libero_object": 800,
    "libero_goal": 800,
    "libero_10": 900,
    "libero_90": 900,
}

NUM_STEPS_WAIT = 10

benchmark_dict = benchmark.get_benchmark_dict()


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class WorkerWSClient:
    """Lightweight WebSocket client used inside each worker process."""

    def __init__(self, host: str, port: int, replan_steps: int = 5, resize_size: int = 224):
        if not HAS_WS_CLIENT:
            raise ImportError("openpi_client not installed. Run: pip install openpi-client")
        self.client = ws_client.WebsocketClientPolicy(host, port)
        self.replan_steps = replan_steps
        self.resize_size = resize_size
        self.action_plan = collections.deque()

    def reset(self) -> None:
        self.action_plan.clear()

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["image"], self.resize_size, self.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["wrist_image"], self.resize_size, self.resize_size)
            )
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": obs["state"],
                "prompt": goal,
            }
            result = self.client.infer(element)
            action_chunk = result["actions"]
            if not isinstance(action_chunk, np.ndarray):
                action_chunk = np.array(action_chunk)
            for i in range(min(self.replan_steps, len(action_chunk))):
                self.action_plan.append(action_chunk[i])
        return self.action_plan.popleft()


def _run_episode(args: tuple) -> dict:
    """Run a single rollout episode (worker entry point, module-level for pickling)."""
    (
        task_suite_name,
        task_id,
        episode_idx,
        seed,
        server_host,
        server_port,
        max_steps,
        save_video,
        video_out_path,
    ) = args

    worker_seed = seed + task_id * 1000 + episode_idx

    try:
        task_suite = benchmark_dict[task_suite_name]()
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_description = task.language

        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(worker_seed)

        client = WorkerWSClient(server_host, server_port)

        env.reset()
        client.reset()
        obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])

        replay_images = []
        t = 0
        done = False

        while t < max_steps + NUM_STEPS_WAIT:
            if t < NUM_STEPS_WAIT:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

            if save_video:
                replay_images.append(img)

            state = np.concatenate([
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ])

            obs_dict = {"image": img, "wrist_image": wrist_img, "state": state}
            action = client.step(obs_dict, task_description)
            obs, reward, done, info = env.step(action.tolist())

            if done:
                break
            t += 1

        env.close()

        suffix = "success" if done else "failure"
        task_segment = task_description.replace(" ", "_")[:50]
        video_path = Path(video_out_path) / f"{task_segment}_ep{episode_idx}_{suffix}.mp4"
        if replay_images and save_video:
            imageio.mimwrite(str(video_path), replay_images, fps=10)

        return {
            "task_id": task_id,
            "episode": episode_idx,
            "success": done,
            "steps": t,
        }

    except Exception as e:
        return {
            "task_id": task_id,
            "episode": episode_idx,
            "success": False,
            "steps": 0,
            "error": str(e),
        }


def main():
    # Use 'spawn' to avoid EGL context issues with fork + MuJoCo
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Single-Suite LIBERO Evaluation (Parallel Rollouts)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task_suite", type=str, required=True,
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--num_trials", type=int, default=50, help="Number of rollouts per task")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--video_out", type=str, default="./eval_results",
                        help="Output directory for videos")
    parser.add_argument("--no_video", action="store_true",
                        help="Disable video recording for faster evaluation")
    args = parser.parse_args()

    if not HAS_WS_CLIENT:
        print("ERROR: openpi_client not installed. Run: pip install openpi-client")
        raise SystemExit(1)

    max_steps = MAX_STEPS.get(args.task_suite, 400)
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = task_suite.n_tasks

    video_out = Path(args.video_out) / args.task_suite
    video_out.mkdir(parents=True, exist_ok=True)
    save_video = not args.no_video

    print(f"Single-suite LIBERO evaluation (parallel rollouts)")
    print(f"  Server:    ws://{args.host}:{args.port}")
    print(f"  Suite:     {args.task_suite}")
    print(f"  Tasks:     {num_tasks}")
    print(f"  Trials:    {args.num_trials} per task")
    print(f"  Workers:   {args.num_workers}")
    print(f"  Max steps: {max_steps}")
    print(f"  Video:     {'on' if save_video else 'off'}")
    print()

    total_successes = 0
    total_episodes = 0

    for task_id in range(num_tasks - 1, -1, -1):
        task = task_suite.get_task(task_id)
        print(f"[Task {task_id}] {task.language[:80]}")

        episode_args = [
            (
                args.task_suite,
                task_id,
                ep,
                args.seed,
                args.host,
                args.port,
                max_steps,
                save_video,
                str(video_out),
            )
            for ep in range(args.num_trials)
        ]

        task_successes = 0
        task_episodes = 0

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(_run_episode, ea) for ea in episode_args]
            for future in as_completed(futures):
                result = future.result()
                task_episodes += 1
                if result["success"]:
                    task_successes += 1

                icon = "[OK]" if result["success"] else "[FAIL]"
                if "error" in result:
                    print(f"   {icon} Ep {result['episode']}: ERROR - {result['error']}")
                else:
                    print(f"   {icon} Ep {result['episode']}: steps={result['steps']}")

        total_successes += task_successes
        total_episodes += task_episodes
        print(f"   => {task_successes}/{task_episodes} ({task_successes / max(task_episodes, 1) * 100:.1f}%)")
        print()

    success_rate = total_successes / max(total_episodes, 1)
    print(f"Overall: {total_successes}/{total_episodes} ({success_rate * 100:.1f}%)")


if __name__ == "__main__":
    main()
