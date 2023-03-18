from pathlib import Path
from shutil import rmtree

from beartype import beartype

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid, save_image
from PIL import Image
from einops import rearrange
import torch.nn.functional as F

from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs

from ema_pytorch import EMA
from tqdm import tqdm


import numpy as np

try:
    import wandb
except:
    None


def noop(*args, **kwargs):
    pass


# helper functions


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)


def yes_or_no(question):
    answer = input(f"{question} (y/n) ")
    return answer.lower() in ("yes", "y")


def pair(val):
    return val if isinstance(val, tuple) else (val, val)


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# image related helpers fnuctions and dataset


def get_accelerator(**accelerate_kwargs):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    kwargs_handlers = accelerate_kwargs.get("kwargs_handlers", [])
    kwargs_handlers.append(ddp_kwargs)
    accelerate_kwargs.update(kwargs_handlers=kwargs_handlers)

    accelerator = Accelerator(**accelerate_kwargs)
    return accelerator


def split_dataset(dataset, valid_frac, accelerator, seed=42):
    if valid_frac > 0:
        train_size = int((1 - valid_frac) * len(dataset))
        valid_size = len(dataset) - train_size
        ds, valid_ds = random_split(
            ds,
            [train_size, valid_size],
            generator=torch.Generator().manual_seed(seed),
        )
        accelerator.print(
            f"training with dataset of {len(ds)} samples and validating with randomly splitted {len(valid_ds)} samples"
        )
    else:
        valid_ds = ds
        accelerator.print(
            f"training with shared training and valid dataset of {len(ds)} samples"
        )
    return ds, valid_ds


# main trainer class


@beartype
class BaseAcceleratedTrainer(nn.Module):
    def __init__(
        self,
        dataloader,
        valid_dataloader,
        accelerator,
        *,
        current_step,
        num_train_steps,
        max_grad_norm=None,
        save_results_every=100,
        save_model_every=1000,
        results_dir="./results",
        logging_dir="./results/logs",
        apply_grad_penalty_every=4,
        gradient_accumulation_steps=1,
        clear_previous_experiments=False,
        validation_image_scale=1,
        only_save_last_checkpoint=False,
        use_profiling=False,
        profile_frequency=1,
    ):
        super().__init__()
        self.model = None
        # instantiate accelerator
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.accelerator = accelerator
        self.results_dir = Path(results_dir)
        if clear_previous_experiments:
            rmtree(str(self.results_dir))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.logging_dir = Path(logging_dir)
        self.logging_dir.mkdir(parents=True, exist_ok=True)

        # training params
        self.only_save_last_checkpoint = only_save_last_checkpoint
        self.validation_image_scale = validation_image_scale
        self.register_buffer("steps", torch.Tensor([current_step]))
        self.num_train_steps = num_train_steps
        self.max_grad_norm = max_grad_norm

        self.dl = dataloader
        self.valid_dl = valid_dataloader
        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.apply_grad_penalty_every = apply_grad_penalty_every

        self.use_profiling = use_profiling
        self.profile_frequency =  profile_frequency

    def save(self, path):
        if not self.is_local_main_process:
            return

        pkg = dict(
            model=self.get_state_dict(self.model),
            optim=self.optim.state_dict(),
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(pkg["model"])

        self.optim.load_state_dict(pkg["optim"])
        return pkg

    def log_validation_images(self, images, step, prompts=None):
        if prompts:
            self.print(f"\nStep: {step} | Logging with prompts: {prompts}")
        if self.validation_image_scale != 1:
            # Feel free to make pr for better solution!
            output_size = (
                int(images[0].size[0] * self.validation_image_scale),
                int(images[0].size[1] * self.validation_image_scale),
            )
            for i in range(len(images)):
                images[i] = images[i].resize(output_size)
        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images(
                    "validation", np_images, step, dataformats="NHWC"
                )
            if tracker.name == "wandb":
                tracker.log(
                    {
                        "validation": [
                            wandb.Image(
                                image, caption="" if not prompts else prompts[i]
                            )
                            for i, image in enumerate(images)
                        ]
                    }
                )

    def print(self, msg):
        self.accelerator.print(msg)

    def log(self, log_dict):
        self.accelerator.log(log_dict)

    def prepare(self, *args):
        return self.accelerator.prepare(*args)

    def get_state_dict(self, model):
        return self.accelerator.get_state_dict(model)

    def unwrap_model(self, model):
        return self.accelerator.unwrap_model(model)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (
            self.accelerator.distributed_type == DistributedType.NO
            and self.accelerator.num_processes == 1
        )

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        raise NotImplementedError(
            "You are calling train_step on the base trainer with no models"
        )

    def train(self, log_fn=noop):
        self.model.train()

        # create two tqdm objects, one for showing the progress bar
        # and another one for showing any extra information we want to show on a different line.
        pbar = tqdm(initial=int(self.steps.item()), total=self.num_train_steps)
        info_bar = tqdm(total=0, bar_format='{desc}')
        #profiling_bar = tqdm(total=0, bar_format='{desc}')


        # use pytorch built-in profiler to gather information on the training for improving performance later.
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        if self.use_profiling:
            prof = torch.autograd.profiler.profile(use_cuda=True)
            prof.__enter__()
            counter = 1

        while self.steps < self.num_train_steps:
                with self.accelerator.autocast():
                    logs = self.train_step()
                log_fn(logs)

                # update the tqdm progress bar
                pbar.update(1)

                # show some extra information on the tqdm progress bar.
                #pbar.set_postfix_str(f"Step: {int(self.steps.item())}")
                try:
                    info_bar.set_description_str(f"Loss: {logs['loss']}, lr: {logs['lr']}")
                    print(logs['save_model_every']) if logs['save_model_every'] else None
                    print(logs['save_results_every']) if logs['save_model_every'] else None
                except KeyError:
                    info_bar.set_description_str(f"VAE loss: {logs['Train/vae_loss']} - discr loss: {logs['Train/discr_loss']} - lr: {logs['lr']}")
                    print(logs['save_model_every']) if logs['save_model_every'] else None
                    print(logs['save_results_every']) if logs['save_model_every'] else None

                if self.use_profiling:
                    counter += 1
                    if counter == self.profile_frequency:
                        # in order to use export_chrome_trace we need to first stop the profiler
                        prof.__exit__(None, None, None)
                        # show the information on the console using loguru as it provides better formating and we can later add colors for easy reading.
                        from loguru import logger
                        logger.info(prof.key_averages().table(sort_by='cpu_time_total'))
                        # save the trace.json file with the information we gathered during this training step,
                        # we can use this trace.json file on the chrome tracing page or other similar tool to view more information.
                        prof.export_chrome_trace(f'{self.logging_dir}/trace.json')
                        # then we can restart it to continue reusing the same profiler.
                        prof = torch.autograd.profiler.profile(use_cuda=True)
                        prof.__enter__()
                        counter = 1 # Reset step counter

        # close the progress bar as we no longer need it.
        pbar.close()
        self.print("training complete")