import random
from pathlib import Path
import termcolor
from PIL import Image, ImageFile

import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as TF
import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True

def worker_init_fn(worker_id):
	seed = min(torch.initial_seed() % 2**32 + worker_id, 2**32-1)
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)

def select_places_files(root, num_bg, seed):
	"""
	Selects `num_bg` image files spread across shuffled category folders.
	Mirrors the background selection of the DreamerV3 variant of VIGOR, so that
	both variants use the same bank for the same `num_bg` and `seed`.
	"""
	rng = np.random.RandomState(seed)
	dirs = [d for d in root.iterdir() if d.is_dir()]
	if not dirs:
		dirs = [root]
	rng.shuffle(dirs)
	per_dir = max(1, num_bg // max(1, len(dirs)) + 1)
	files = []
	for d in dirs:
		fs = []
		for pat in ("*.jpg", "*.jpeg", "*.png"):
			fs.extend(sorted(d.glob(pat)))
			if len(fs) >= per_dir:
				break
		files.extend(fs[:per_dir])
		if len(files) >= num_bg:
			break
	files = files[:num_bg]
	if not files:
		raise FileNotFoundError(f"No image files found under {root}.")
	return files

class PlacesFiles(torch.utils.data.Dataset):
	"""Loads image files resized to (image_size, image_size) as uint8 (3, H, W) tensors."""
	def __init__(self, files, image_size):
		self.files = files
		self.image_size = image_size

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		im = Image.open(self.files[idx]).convert("RGB").resize((self.image_size, self.image_size))
		return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1)

class Augmentation:
	"""Class for data augmentation."""
	places_dataloader = None
	places_iter = None
	places_bank = None
	dataset_dir = None
	num_workers = 8
	num_bg = 0
	bank_seed = 0

	@classmethod
	def _load_places_bank(cls, image_size, device=None, partition="train") -> None:
		"""Loads `num_bg` pre-resized Places images into GPU memory as a uint8 (N, 3, H, W) bank."""
		root = Path(Augmentation.dataset_dir) / "places365_standard" / partition
		if not root.exists():
			raise ValueError(f"Path {root} does not exist")
		print(termcolor.colored(f"Caching {Augmentation.num_bg:,} images of {partition} partition of places365_standard...", "yellow", attrs=["bold"]))
		files = select_places_files(root, Augmentation.num_bg, Augmentation.bank_seed)
		loader = torch.utils.data.DataLoader(PlacesFiles(files, image_size), batch_size=256, num_workers=Augmentation.num_workers)
		bank = torch.empty(len(files), 3, image_size, image_size, dtype=torch.uint8, device=device)
		i = 0
		for imgs in loader:
			bank[i:i+len(imgs)] = imgs.to(device)
			i += len(imgs)
		Augmentation.places_bank = bank
		print(termcolor.colored(f"Cached {len(files):,} images from {root} ({bank.numel()/1e9:.2f} GB)", "yellow", attrs=["bold"]))

	@classmethod
	def _get_places_bank_batch(cls, batch_size, device=None) -> torch.Tensor:
		idx = torch.randint(len(Augmentation.places_bank), (batch_size,), device=Augmentation.places_bank.device)
		return Augmentation.places_bank[idx].to(device=device).float().div_(255.)

	@classmethod
	def _load_places(cls, batch_size=256, image_size=84, num_workers=8, use_val=False, device=None) -> None:
		partition = "val" if use_val else "train"
		print(termcolor.colored(f"Loading {partition} partition of places365_standard...", "yellow", attrs=["bold"]))
		dataset_dir = Path(Augmentation.dataset_dir) / "places365_standard" / partition
		if dataset_dir.exists():
			Augmentation.places_dataloader = torch.utils.data.DataLoader(
				datasets.ImageFolder(dataset_dir, TF.Compose([
					TF.RandomResizedCrop(image_size),
					TF.RandomHorizontalFlip(),
					TF.ToTensor()
				])),
				batch_size=batch_size, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn,
				num_workers=num_workers, persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None,
				pin_memory=True, pin_memory_device=str(device) if isinstance(device, torch.device) else device)
			Augmentation.places_iter = iter(Augmentation.places_dataloader)
		else:
			raise ValueError(f"Path {dataset_dir} does not exist")

		if Augmentation.places_iter is None:
			raise FileNotFoundError("failed to find places365 data at any of the specified paths")
		print(termcolor.colored(f"Loaded dataset from {dataset_dir}", "yellow", attrs=["bold"]))

	@classmethod
	def _get_places_batch(cls, batch_size, device=None) -> torch.Tensor:
		try:
			imgs, _ = next(Augmentation.places_iter)
			while imgs.size(0) < batch_size:
				imgs_next, _ = next(Augmentation.places_iter)
				imgs = torch.cat((imgs, imgs_next), dim=0)
		except StopIteration:
			Augmentation.places_iter = iter(Augmentation.places_dataloader)
			imgs, _ = next(Augmentation.places_iter)
			while imgs.size(0) < batch_size:
				imgs_next, _ = next(Augmentation.places_iter)
				imgs = torch.cat((imgs, imgs_next), dim=0)
		return imgs[:batch_size].to(device=device, non_blocking=True)

	@classmethod
	def random_overlay(cls, x: torch.Tensor, dataset="places365_standard", device=None) -> torch.Tensor:
		"""Randomly overlay an image from Places"""
		alpha = 0.5

		if dataset == "places365_standard":
			if x.ndim == 4:
				batch_size = x.size(0)
			elif x.ndim == 5:
				batch_size = x.size(1)
			if Augmentation.num_bg > 0:
				if Augmentation.places_bank is None:
					Augmentation._load_places_bank(image_size=x.size(-1), device=device)
				imgs = Augmentation._get_places_bank_batch(batch_size=batch_size, device=device)
			else:
				if Augmentation.places_dataloader is None:
					Augmentation._load_places(batch_size=batch_size, image_size=x.size(-1), num_workers=Augmentation.num_workers, device=device)
				imgs = Augmentation._get_places_batch(batch_size=batch_size, device=device)
			if x.ndim == 4:
				imgs = imgs.repeat(1, x.size(1)//3, 1, 1)
			elif x.ndim == 5:	
				imgs = imgs.unsqueeze(0).repeat(x.size(0), 1, x.size(2)//3, 1, 1)
		else:
			raise NotImplementedError(f"overlay has not been implemented for dataset '{dataset}'")

		return ((1-alpha)*(x/255.) + (alpha)*imgs)*255.

	@classmethod
	def random_shift(cls, x: torch.Tensor, pad_size=None) -> torch.Tensor:
		if pad_size is None:
			raise ValueError("Please specify the padding size")
		if x.ndim == 4:
			n, c, h, w = x.size()
			padding = tuple([pad_size] * 4)
		elif x.ndim == 5:
			horizon, n, c, h, w = x.size()
			padding = tuple([pad_size] * 4 + [0] * 2)
		assert h == w
		x = x.float()
		x = F.pad(x, padding, "replicate")
		eps = 1.0 / (h + 2 * pad_size)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad_size, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * pad_size + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * pad_size)
		grid = base_grid + shift
		if x.ndim == 4:
			return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
		elif x.ndim == 5:
			xs = []
			for h_idx in range(horizon):
				xs.append(F.grid_sample(x[h_idx], grid, padding_mode="zeros", align_corners=False))
			return torch.stack(xs)

	@classmethod
	def random_conv(cls, x: torch.Tensor, device=None) -> torch.Tensor:
		"""Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
		n, c, h, w = x.shape
		for i in range(n):
			weights = torch.randn(3, 3, 3, 3).to(device=device if device is not None else x.device)
			temp_x = x[i:i+1].reshape(-1, 3, h, w)/255.
			temp_x = F.pad(temp_x, pad=[1]*4, mode="replicate")
			out = torch.sigmoid(F.conv2d(temp_x, weights))*255.
			total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
		return total_out.reshape(n, c, h, w)

	@classmethod
	def random_mask_freq_v2(cls, x: torch.Tensor, device=None) -> torch.Tensor:
		p = random.uniform(0, 1)
		if p > 0.5:
			return x

		# dynamicly select freq range to erase
		A = 0
		B = 0.5
		a = random.uniform(A, B)
		C = 2
		freq_limit_low = round(a, C)

		A = 0
		B = 0.05
		a = random.uniform(A, B)
		C = 2
		diff = round(a, C)
		freq_limit_hi = freq_limit_low + diff

		# b, 9, h, w
		b, c, h, w = x.shape
		x0, x1, x2 = torch.chunk(x, 3, dim=1)
		# b, 3, 3, h, w
		x = torch.cat((x0.unsqueeze(1), x1.unsqueeze(1), x2.unsqueeze(1)), dim=1)

		pass1 = torch.abs(torch.fft.fftfreq(x.shape[-1], device=device if device is not None else x.device)) < freq_limit_hi
		pass2 = torch.abs(torch.fft.fftfreq(x.shape[-2], device=device if device is not None else x.device)) < freq_limit_hi
		kernel1 = torch.outer(pass2, pass1)  # freq_limit_hi square is true

		pass1 = torch.abs(torch.fft.fftfreq(x.shape[-1], device=device if device is not None else x.device)) < freq_limit_low
		pass2 = torch.abs(torch.fft.fftfreq(x.shape[-2], device=device if device is not None else x.device)) < freq_limit_low
		kernel2 = torch.outer(pass2, pass1)  # freq_limit_low square is true

		kernel = kernel1 * (~kernel2)  # a square ring is true
		fft_1 = torch.fft.fftn(x, dim=(2, 3, 4))
		imgs = torch.fft.ifftn(fft_1 * (~kernel), dim=(2, 3, 4)).float()
		x0, x1, x2 = torch.chunk(imgs, 3, dim=1)
		imgs = torch.cat((x0.squeeze(1), x1.squeeze(1), x2.squeeze(1)), dim=1)

		return imgs