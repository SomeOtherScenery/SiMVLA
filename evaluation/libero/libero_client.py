#!/usr/bin/env python3
"""
SimVLA LIBERO Evaluation Client

Observation format:
1. State: [eef_pos(3), axis_angle(3), gripper_qpos(2)] = 8D
2. Action: delta action (7D)
3. Default delta control mode
4. Images rotated 180 degrees

Supports parallel rollout evaluation via subprocess vectorized environments.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Deque, Dict, List, Optional

import imageio
import json_numpy
import numpy as np
import requests
from tqdm import tqdm

try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as ws_client
    HAS_WS_CLIENT = True
except ImportError:
    HAS_WS_CLIENT = False

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

# CUDA_VISIBLE_DEVICES value -> EGL render device ID
CUDA_TO_EGL_MAP = {
    "0": 2,
    "1": 3,
    "2": 1,
    "3": 0,
    "4": 6,
    "5": 7,
    "6": 5,
    "7": 4,
}


def _resolve_render_gpu_id(explicit: Optional[int]) -> int:
    """Return the EGL render GPU ID from explicit arg or CUDA_VISIBLE_DEVICES."""
    if explicit is not None:
        return explicit
    cuda_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_dev and cuda_dev in CUDA_TO_EGL_MAP:
        return CUDA_TO_EGL_MAP[cuda_dev]
    return -1

# Max steps per task suite (based on longest demo + buffer)
MAX_STEPS = {
    "libero_spatial": 800,   # longest demo: 193
    "libero_object": 800,    # longest demo: 254
    "libero_goal": 800,      # longest demo: 270
    "libero_10": 900,        # longest demo: 505
    "libero_90": 900,        # longest demo: 373
}

NUM_STEPS_WAIT = 10  # Wait for objects to stabilize

benchmark_dict = benchmark.get_benchmark_dict()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to axis-angle representation."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _merge_dict(list_of_dicts: list) -> dict:
    """Merge a list of observation dicts into a dict of batched numpy arrays."""
    if not isinstance(list_of_dicts, (list, tuple)) or not list_of_dicts:
        return list_of_dicts
    keys = list_of_dicts[0].keys()
    return {k: np.stack([d[k] for d in list_of_dicts]) for k in keys}


# -----------------------------------------------------------------------------
# Client Policy Classes
# -----------------------------------------------------------------------------

class WebSocketClient:
    """WebSocket client for SimVLA server."""

    def __init__(self, host: str, port: int, replan_steps: int = 5, resize_size: int = 224):
        if not HAS_WS_CLIENT:
            raise ImportError("openpi_client not installed. Run: pip install openpi-client")
        self.client = ws_client.WebsocketClientPolicy(host, port)
        self.replan_steps = replan_steps
        self.resize_size = resize_size
        self.reset()

    def reset(self) -> None:
        self.action_plan: Deque[np.ndarray] = collections.deque()

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

            assert len(action_chunk) >= self.replan_steps, \
                f"Need {self.replan_steps} steps but got {len(action_chunk)}"

            for i in range(min(self.replan_steps, len(action_chunk))):
                self.action_plan.append(action_chunk[i])

        return self.action_plan.popleft()


class HTTPClient:
    """HTTP client for SimVLA server."""

    def __init__(self, host: str, port: int, replan_steps: int = 5):
        self.url = f"http://{host}:{port}/act"
        self.replan_steps = replan_steps
        self.reset()

    def reset(self) -> None:
        self.action_plan: Deque[np.ndarray] = collections.deque()

    def infer(self, element: Dict) -> Dict:
        try:
            payload = {
                "image0": json_numpy.dumps(element["observation/image"]),
                "image1": json_numpy.dumps(element["observation/wrist_image"]),
                "proprio": json_numpy.dumps(element["observation/state"]),
                "language_instruction": element["prompt"],
                "steps": 10,
            }

            resp = requests.post(self.url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            actions = np.array(data["action"])
            return {"actions": actions}

        except Exception as e:
            raise RuntimeError(f"Policy server request failed: {e}") from e

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            element = {
                "observation/image": obs["image"],
                "observation/wrist_image": obs["wrist_image"],
                "observation/state": obs["state"],
                "prompt": goal,
            }

            result = self.infer(element)
            action_chunk = result["actions"]

            for action in action_chunk[:self.replan_steps]:
                self.action_plan.append(action)

        return self.action_plan.popleft()


# -----------------------------------------------------------------------------
# Subprocess Vectorized Environments
# -----------------------------------------------------------------------------

class _LiberoEnvFn:
    """Picklable environment factory for subprocess workers.

    Stores primitive parameters so the factory can be sent across process
    boundaries.  The LIBERO task is reconstructed inside the worker.
    """

    def __init__(self, task_suite_name: str, task_id: int, gpu_id: int, seed: int):
        self.task_suite_name = task_suite_name
        self.task_id = task_id
        self.gpu_id = gpu_id
        self.seed = seed

    def __call__(self):
        task_suite = benchmark_dict[self.task_suite_name]()
        task = task_suite.get_task(self.task_id)
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env_args = {
            "bddl_file_name": str(bddl_file),
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
        }
        if self.gpu_id >= 0:
            env_args["render_gpu_device_id"] = self.gpu_id
        env = OffScreenRenderEnv(**env_args)
        env.seed(self.seed)
        return env


def _subproc_worker(pipe: mp.connection.Connection, env_fn, worker_id: int):
    """Run a LIBERO environment in a subprocess, communicating via pipe."""
    env = env_fn()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                pipe.send((obs, reward, done, info))
            elif cmd == "reset_with_state":
                init_state = data
                env.reset()
                obs = env.set_init_state(init_state)
                pipe.send(obs)
            elif cmd == "close":
                env.close()
                break
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass


class SubprocVecEnv:
    """Lightweight subprocess vectorized environment for LIBERO."""

    def __init__(self, env_fns: list):
        self.num_envs = len(env_fns)
        self.processes = []
        self.pipes = []

        ctx = mp.get_context("spawn")
        for i, env_fn in enumerate(env_fns):
            parent_pipe, child_pipe = ctx.Pipe()
            proc = ctx.Process(target=_subproc_worker, args=(child_pipe, env_fn, i))
            proc.start()
            self.processes.append(proc)
            self.pipes.append(parent_pipe)
            child_pipe.close()

    def reset_with_states(self, init_states: list) -> dict:
        """Reset all envs and set their initial states."""
        for i, pipe in enumerate(self.pipes):
            pipe.send(("reset_with_state", init_states[i]))
        obs_list = [pipe.recv() for pipe in self.pipes]
        return _merge_dict(obs_list)

    def step(self, actions: np.ndarray):
        """Step all environments with per-env actions (shape [N, action_dim])."""
        for i, pipe in enumerate(self.pipes):
            pipe.send(("step", actions[i]))
        results = [pipe.recv() for pipe in self.pipes]
        obs_list, rewards, dones, infos = zip(*results)
        return _merge_dict(obs_list), np.array(rewards), np.array(dones), list(infos)

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except Exception:
                pass
        for proc in self.processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------
def get_libero_env(task, resolution: int, seed: int, render_gpu_id: int = -1):
    """Initialize a LIBERO environment."""
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    if render_gpu_id >= 0:
        env_args["render_gpu_device_id"] = render_gpu_id
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def eval_libero(
    client,
    task_suite_name: str,
    num_trials: int = 50,
    seed: int = 7,
    video_out_path: str = "data/libero/videos",
    save_video: bool = True,
    num_parallel: int = 1,
    render_gpu_id: int = -1,
    host: str = "127.0.0.1",
    port: int = 8000,
    client_type: str = "websocket",
    replan_steps: int = 5,
) -> float:
    """
    Run LIBERO evaluation across all tasks in a suite.

    When num_parallel > 1, runs multiple rollouts in parallel using subprocess
    vectorized environments. One WebSocket client per environment.
    """
    np.random.seed(seed)

    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = MAX_STEPS.get(task_suite_name, 400)

    Path(video_out_path).mkdir(parents=True, exist_ok=True)

    print(f"Task suite: {task_suite_name}")
    print(f"   Tasks: {num_tasks}, Trials per task: {num_trials}")
    print(f"   Max steps: {max_steps}")
    if num_parallel > 1:
        print(f"   Parallel envs: {num_parallel}")
        print(f"   Render GPU:  {render_gpu_id}")

    total_episodes, total_successes = 0, 0

    for task_id in tqdm(range(num_tasks - 1, -1, -1), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_description = task.language

        if num_parallel > 1:
            task_successes = _eval_task_parallel(
                task, task_id, task_description, initial_states,
                task_suite_name, num_trials, seed, max_steps,
                video_out_path, save_video, num_parallel,
                render_gpu_id,
                host, port, client_type, replan_steps,
            )
            total_successes += task_successes
            total_episodes += num_trials
            print(f"   Task {task_id}: {task_successes}/{num_trials} ({task_successes / num_trials * 100:.1f}%)")
        else:
            # Original single-env path
            env, _ = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed, render_gpu_id=render_gpu_id)
            task_successes = 0
            for ep in tqdm(range(num_trials), desc=f"{task_description[:30]}...", leave=False):
                env.reset()
                client.reset()
                obs = env.set_init_state(initial_states[ep % len(initial_states)])

                replay_images = []
                t = 0
                done = False

                while t < max_steps + NUM_STEPS_WAIT:
                    try:
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
                            task_successes += 1
                            total_successes += 1
                            break

                        t += 1

                    except Exception as e:
                        print(f"Error in rollout: {e}")
                        break

                total_episodes += 1

                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")[:50]
                video_path = Path(video_out_path) / f"{task_segment}_ep{ep}_{suffix}.mp4"
                if replay_images and save_video:
                    imageio.mimwrite(str(video_path), replay_images, fps=10)

                status_icon = "[OK]" if done else "[FAIL]"
                print(f"  {status_icon} Task {task_id} Ep {ep}: {suffix.upper()} (steps={t})")

            env.close()
            print(f"   Task {task_id}: {task_successes}/{num_trials} ({task_successes / num_trials * 100:.1f}%)")

    success_rate = total_successes / max(total_episodes, 1)
    print(f"\nTotal success rate: {total_successes}/{total_episodes} ({success_rate * 100:.1f}%)")

    return success_rate


def _eval_task_parallel(
    task, task_id: int, task_description: str, initial_states: list,
    task_suite_name: str, num_trials: int, seed: int, max_steps: int,
    video_out_path: str, save_video: bool, num_parallel: int,
    render_gpu_id: int, host: str, port: int,
    client_type: str, replan_steps: int,
) -> int:
    """Evaluate a single task with parallel environments."""

    # Distribute initial states across envs
    num_envs = min(num_parallel, num_trials)
    init_states_per_env = [initial_states[i % len(initial_states)] for i in range(num_trials)]
    # Each env gets its own init state for this batch; if num_trials > num_envs,
    # we run multiple batches
    batches = (num_trials + num_envs - 1) // num_envs

    total_successes = 0

    for batch in range(batches):
        start_ep = batch * num_envs
        end_ep = min(start_ep + num_envs, num_trials)
        batch_size = end_ep - start_ep

        env_fns = []
        for i in range(batch_size):
            gpu_id = render_gpu_id
            ep_idx = start_ep + i
            worker_seed = seed + task_id * 1000 + ep_idx
            env_fns.append(_LiberoEnvFn(task_suite_name, task_id, gpu_id, worker_seed))

        # Create vectorized env
        vec_env = SubprocVecEnv(env_fns)

        # Create one client per env
        clients = []
        for i in range(batch_size):
            if client_type == "websocket":
                clients.append(WebSocketClient(host, port, replan_steps=replan_steps))
            else:
                clients.append(HTTPClient(host, port, replan_steps=replan_steps))

        # Reset all envs with their initial states
        batch_init_states = [init_states_per_env[start_ep + i] for i in range(batch_size)]
        obs = vec_env.reset_with_states(batch_init_states)

        # State tracking
        dones = np.zeros(batch_size, dtype=bool)
        episode_dones = np.zeros(batch_size, dtype=bool)  # track which episodes succeeded
        replay_images = [[] for _ in range(batch_size)]
        step_counts = np.zeros(batch_size, dtype=int)

        # Main loop
        for t in range(max_steps + NUM_STEPS_WAIT):
            if np.all(episode_dones):
                break

            actions = np.zeros((batch_size, 7), dtype=np.float64)

            for i in range(batch_size):
                if episode_dones[i]:
                    actions[i] = LIBERO_DUMMY_ACTION
                    continue

                if t < NUM_STEPS_WAIT:
                    actions[i] = LIBERO_DUMMY_ACTION
                    continue

                # Extract per-env observation from batched dict
                img = np.ascontiguousarray(obs["agentview_image"][i][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][i][::-1, ::-1])

                if save_video:
                    replay_images[i].append(img)

                state = np.concatenate([
                    obs["robot0_eef_pos"][i],
                    _quat2axisangle(obs["robot0_eef_quat"][i]),
                    obs["robot0_gripper_qpos"][i],
                ])

                obs_dict = {"image": img, "wrist_image": wrist_img, "state": state}
                try:
                    action = clients[i].step(obs_dict, task_description)
                except Exception as e:
                    print(f"  [ERROR] Task {task_id} Ep {start_ep + i} step {t}: {e}")
                    action = LIBERO_DUMMY_ACTION

                actions[i] = action
                step_counts[i] += 1

            # Step all envs
            obs, rewards, dones, infos = vec_env.step(actions)

            # Check for newly finished envs
            for i in range(batch_size):
                if dones[i] and not episode_dones[i] and t >= NUM_STEPS_WAIT:
                    episode_dones[i] = True
                    total_successes += 1
                    # Save video immediately
                    suffix = "success"
                    task_segment = task_description.replace(" ", "_")[:50]
                    video_path = Path(video_out_path) / f"{task_segment}_ep{start_ep + i}_{suffix}.mp4"
                    if replay_images[i] and save_video:
                        imageio.mimwrite(str(video_path), replay_images[i], fps=10)

        # Handle unfinished envs (failures)
        for i in range(batch_size):
            if not episode_dones[i]:
                suffix = "failure"
                task_segment = task_description.replace(" ", "_")[:50]
                video_path = Path(video_out_path) / f"{task_segment}_ep{start_ep + i}_{suffix}.mp4"
                if replay_images[i] and save_video:
                    imageio.mimwrite(str(video_path), replay_images[i], fps=10)

            status = "[OK]" if episode_dones[i] else "[FAIL]"
            suffix = "SUCCESS" if episode_dones[i] else "FAILURE"
            print(f"  {status} Task {task_id} Ep {start_ep + i}: {suffix} (steps={step_counts[i]})")

        vec_env.close()

    return total_successes


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("LIBERO Evaluation Client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--connection_info", type=str, default=None,
                        help="Path to server connection info JSON")
    parser.add_argument("--client_type", type=str, default="websocket",
                        choices=["websocket", "http"],
                        help="Client type: websocket or http")
    parser.add_argument("--task_suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--video_out", type=str, default="./eval_results")
    parser.add_argument("--no_video", action="store_true", help="Disable video recording for faster evaluation")
    parser.add_argument("--num_parallel", type=int, default=1,
                        help="Number of parallel environments for rollouts (default: 1, sequential)")
    parser.add_argument("--render_gpu_id", type=int, default=None,
                        help="EGL render GPU ID. If not set, inferred from CUDA_VISIBLE_DEVICES via CUDA_TO_EGL_MAP.")

    args = parser.parse_args()

    # Resolve render GPU ID from explicit arg or CUDA_VISIBLE_DEVICES
    render_gpu_id = _resolve_render_gpu_id(args.render_gpu_id)
    print(f"   Render GPU:  {render_gpu_id} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")

    # Load connection info
    if args.connection_info:
        print(f"Loading connection info from: {args.connection_info}")
        while not Path(args.connection_info).exists():
            sys.stdout.write("\rWaiting for server...")
            sys.stdout.flush()
            time.sleep(0.5)
        print()
        with open(args.connection_info) as f:
            info = json.load(f)
            args.host = info["host"]
            args.port = info["port"]

    protocol = "ws" if args.client_type == "websocket" else "http"
    print(f"Starting LIBERO evaluation client")
    print(f"   Client type: {args.client_type}")
    print(f"   Server: {protocol}://{args.host}:{args.port}")
    print(f"   Task suite: {args.task_suite}")
    print(f"   Replan steps: {args.replan_steps}")
    print(f"   Parallel envs: {args.num_parallel}")
    print()

    # Initialize client (for single-env path) or pass factory params to eval_libero
    if args.client_type == "websocket":
        client = WebSocketClient(args.host, args.port, replan_steps=args.replan_steps)
    else:
        client = HTTPClient(args.host, args.port, replan_steps=args.replan_steps)

    # Run evaluation
    video_path = Path(args.video_out) / args.task_suite
    eval_libero(
        client=client,
        task_suite_name=args.task_suite,
        num_trials=args.num_trials,
        seed=args.seed,
        video_out_path=str(video_path),
        save_video=not args.no_video,
        num_parallel=args.num_parallel,
        render_gpu_id=render_gpu_id,
        host=args.host,
        port=args.port,
        client_type=args.client_type,
        replan_steps=args.replan_steps,
    )


if __name__ == "__main__":
    main()
