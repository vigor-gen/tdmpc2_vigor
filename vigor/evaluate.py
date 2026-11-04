import datetime
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import time
import warnings
warnings.filterwarnings("ignore")

import hydra
import imageio
import numpy as np
import pandas as pd
from pathlib import Path
import termcolor
import torch

from vigor import envs
from vigor.algos import ALGORITHMS
from vigor.common import parser
from vigor.common import seeding

torch.backends.cudnn.benchmark = True

DMC_GEN_TYPES = ["original", "background_color", "background_video", "cam_pos",
				"light_position", "light_color", "moving_light", "objects_color"]
DMC_GEN_DIFFS = ["easy", "hard"]
ROBO_GEN_TYPES = ["train", "eval-easy", "eval-hard", "eval-extreme"]
ROBO_SCENES = list(range(10))
NUM_SEEDS = 10

@hydra.main(config_name="config", config_path=".")
def evaluate(cfg: dict):
	"""
	Script for evaluating a TD-MPC2 or TD-MPC2+VIGOR agent.

	Most relevant args:
		`name`: algorithm, TDMPC2 or VIGOR (default: TDMPC2)
		`task`: task name
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`save_video`: whether to save a video of the evaluation (default: True)
		`seed`: random seed (default: 1)
		`gen_type`: generalization type name (default: all)

	See config.yaml for a full list of args.

	Example usage:
	````
		$ python vigor/evaluate.py task=walker-walk save_video=true
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, "Must evaluate at least 1 episode."
	assert cfg.name in ALGORITHMS, f"Unknown algorithm {cfg.name}; choose one of {list(ALGORITHMS)}."
	cfg = parser.parse_cfg(cfg)
	cfg.exp_type = "eval"
	cfg.gen_type = cfg.get("gen_type", "all")
	cfg.checkpoint = Path(hydra.utils.get_original_cwd()) / "logs" / cfg.name / cfg.task / str(cfg.seed) / cfg.exp_name / "models" / "final.pt"
	print(termcolor.colored(f"Task: {cfg.task}", "blue", attrs=["bold"]))
	print(termcolor.colored(f"Gen type: {cfg.gen_type}", "blue", attrs=["bold"]))
	print(termcolor.colored(f"Checkpoint: {cfg.checkpoint}", "blue", attrs=["bold"]))

	gen_diff = cfg.get("gen_diff", "easy")

	if cfg.task in ["walker-walk", "cartpole-swingup", "cheetah-run", "walker-stand", "reacher-easy", "finger-spin"]:
		env_type = "dmc"
		csv_file = "eval_avg.csv" if cfg.gen_type == "all" else f"{cfg.gen_type}_{gen_diff}.csv"
	elif cfg.task in ["Door", "Lift", "TwoArmPegInHole", "NutAssemblyRound"]:
		env_type = "robosuite"
		csv_file = "eval_avg.csv" if cfg.gen_type == "all" else f"{cfg.gen_type}.csv"
	else:
		raise NotImplementedError
	if (cfg.work_dir / csv_file).exists():
		raise FileExistsError(f"{cfg.work_dir / csv_file} already exists.")
	# Evaluation seeds are fixed and disjoint from training seeds (same protocol as the DreamerV3 variant)
	seed_base = int(os.environ.get("EVAL_SEED_BASE", 25))
	test_seed_list = [seed_base+i for i in range(NUM_SEEDS)]

	# Make a dummy environment
	cfg.gen_type = "original" if env_type == "dmc" else "train"
	cfg.gen_diff = "easy"
	cfg.scene_id = 0
	dummy_env = envs.make_env(cfg)
	# Load agent
	agent = ALGORITHMS[cfg.name](cfg)
	assert os.path.exists(cfg.checkpoint), f"Checkpoint {cfg.checkpoint} not found! Must be a valid filepath."
	agent.load(cfg.checkpoint)
	del dummy_env
	
	if cfg.save_video:
		video_dir = os.path.join(cfg.work_dir, "videos")
		os.makedirs(video_dir, exist_ok=True)
	metrics, start_time = [], time.time()
	for exp_name, (gen_type, gen_level) in conditions(env_type, cfg.gen_type, gen_diff):
		for test_seed in test_seed_list:
			seeding.set_seed(test_seed)
			cfg.seed = test_seed
			cfg.gen_type = gen_type
			if env_type == "dmc":
				cfg.gen_diff = gen_level
				num_episodes = cfg.eval_episodes
			else:
				cfg.scene_id = gen_level
				num_episodes = 1 # One episode per seed and scene

			# Make environment
			env = envs.make_env(cfg)

			# Evaluate
			scene_str = f"+scene#{cfg.scene_id}" if env_type == "robosuite" else ""
			print(termcolor.colored(f"Evaluating agent on {cfg.task}+{exp_name}+seed{test_seed}{scene_str}:", "yellow", attrs=["bold"]))
			record = cfg.save_video and test_seed == test_seed_list[0]
			episode_returns, episode_successes = [], []
			for episode in range(num_episodes):
				observation, done, episode_return, timestep = env.reset(), False, 0, 0
				if record and episode == 0:
					if env_type == "dmc":
						frames = [env.render(480, 480, visualization_only=True)]
					else:
						frames = [env.render(480, 480)]
				while not done:
					torch.compiler.cudagraph_mark_step_begin()
					action = agent.act(observation, t0=timestep==0, eval_mode=True)
					observation, reward, done, info = env.step(action)
					episode_return += reward
					timestep += 1
					if record and episode == 0:
						if env_type == "dmc":
							frames.append(env.render(480, 480, visualization_only=True))
						else:
							frames.append(env.render(480, 480))
				episode_returns.append(episode_return.item())
				episode_successes.append(info["success"])
				if record and episode == 0:
					video_name = f"{cfg.task}_{exp_name}" + (f"#{cfg.scene_id}" if env_type == "robosuite" else "")
					imageio.mimsave(os.path.join(video_dir, f"{video_name}.mp4"), frames, fps=15)
			elapsed_time = time.time() - start_time
			print(termcolor.colored(f"  {cfg.task:<22}{exp_name}+seed{test_seed}{scene_str}" \
				f"\tR: {np.mean(episode_returns):.01f}  " \
				f"\tS: {np.mean(episode_successes):.02f}" \
				f"\tT: {datetime.timedelta(seconds=int(elapsed_time))}", "blue"))
			if env_type == "dmc":
				keys = ["exp_name", "seed", "episode_idx", "episode_return"]
				for episode_index, episode_return in enumerate(episode_returns):
					metrics.append(np.array([exp_name, test_seed, episode_index, episode_return]))
			else:
				keys = ["exp_name", "seed", "episode_return", "scene_id", "success_rate"]
				metrics.append(np.array([exp_name, test_seed, episode_returns[0], cfg.scene_id, episode_successes[0]]))
			pd.DataFrame(np.array(metrics)).to_csv(
				cfg.work_dir / csv_file, header=keys, index=None
			)
			del env

def conditions(env_type, gen_type, gen_diff):
	"""Yields (exp_name, (gen_type, gen_diff or scene_id)) for every evaluated condition."""
	if env_type == "dmc":
		types = DMC_GEN_TYPES if gen_type == "all" else [gen_type]
		diffs = DMC_GEN_DIFFS if gen_type == "all" else [gen_diff]
		for t in types:
			for d in diffs:
				if t == "original" and d == "hard":
					continue
				yield (t if t == "original" else f"{t}_{d}"), (t, d)
	else:
		types = ROBO_GEN_TYPES if gen_type == "all" else [gen_type]
		for t in types:
			for scene in ROBO_SCENES:
				yield t, (t, scene)

if __name__ == "__main__":
	evaluate()