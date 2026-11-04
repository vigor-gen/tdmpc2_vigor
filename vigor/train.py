import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TORCH_LOGS"] = "+recompiles"
import warnings
warnings.filterwarnings("ignore")

import hydra
import termcolor
import torch

from vigor import envs
from vigor.algos import ALGORITHMS
from vigor.common import parser
from vigor.common import seeding
from vigor.common.buffer import Buffer
from vigor.common.logger import Logger
from vigor.trainer.online_trainer import OnlineTrainer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

@hydra.main(config_name="config", config_path=".")
def train(cfg: dict):
	"""
	Script for training a TD-MPC2 or TD-MPC2+VIGOR agent.

	Most relevant args:
		`name`: algorithm, TDMPC2 or VIGOR (default: TDMPC2)
		`task`: task name
		`device`: CUDA device name
		`steps`: number of training/environment steps (default: 1M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python vigor/train.py task=walker-walk device=cuda:0
		$ python vigor/train.py name=VIGOR task=walker-walk dataset_dir=/path/to/dataset
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, "Must train for at least 1 step."
	assert cfg.name in ALGORITHMS, f"Unknown algorithm {cfg.name}; choose one of {list(ALGORITHMS)}."
	cfg = parser.parse_cfg(cfg)
	seeding.set_seed(cfg.seed)
	print(termcolor.colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)
 
	cfg.exp_type = "train"

	trainer = OnlineTrainer(
		cfg=cfg,
		env=envs.make_env(cfg),
		agent=ALGORITHMS[cfg.name](cfg),
		buffer=Buffer(cfg),
		logger=Logger(cfg),
	)
	trainer.train()
	print("\nTraining completed successfully")


if __name__ == "__main__":
	train()
