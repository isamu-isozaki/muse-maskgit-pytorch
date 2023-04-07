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
from torch.optim import Adam, AdamW
from torch_optimizer import AdaBound, AdaMod, AccSGD, AdamP, AggMo, DiffGrad, \
     Lamb, NovoGrad, PID, QHAdam, QHM, RAdam, SGDP, SGDW, Shampoo, SWATS, Yogi
from transformers.optimization import Adafactor
from lion_pytorch import Lion

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

def get_optimizer(use_8bit_adam, optimizer, parameters, lr, weight_decay):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:  # bitsandbytes raises a broad exception for cuda setup errors
            raise ImportError("Please install bitsandbytes to use 8-bit optimizers. You can do so by running `pip install "
                        "bitsandbytes` | Defaulting to non 8-bit equivalent...")
    # optimizers
    if optimizer == "Adam":
        if use_8bit_adam:
            optim = bnb.optim.Adam8bit(parameters, lr=lr, weight_decay=weight_decay)
        else:
            optim = Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "AdamW":
        if use_8bit_adam:
            optim = bnb.optim.AdamW8bit(parameters, lr=lr, weight_decay=weight_decay)
        else:
            optim = AdamW(parameters, lr=lr, weight_decay=weight_decay)

    elif optimizer == "Lion":
        optim = Lion(parameters, lr=lr, weight_decay=weight_decay)
        if use_8bit_adam:
            print("8bit is not supported by the Lion optimiser, Using standard Lion instead.")
    elif optimizer == "Adafactor":
        optim = Adafactor(parameters, lr=lr, weight_decay=weight_decay, relative_step=False)
    elif optimizer == "AccSGD":
        optim = AccSGD(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "AdaBound":
        optim = AdaBound(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "AdaMod":
        optim = AdaMod(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "AdamP":
        optim = AdamP(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "AggMo":
        optim = AggMo(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "DiffGrad":
        optim = DiffGrad(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "Lamb":
        optim = Lamb(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "NovoGrad":
        optim = NovoGrad(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "PID":
        optim = PID(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "QHAdam":
        optim = QHAdam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "QHM":
        optim = QHM(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "RAdam":
        optim = RAdam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "SGDP":
        optim = SGDP(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "SGDW":
        optim = SGDW(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "Shampoo":
        optim = Shampoo(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "SWATS":
        optim = SWATS(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "Yogi":
        optim = Yogi(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"{optimizer} optimizer not supported yet.")
    return optim
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
        batch_size=1,
        gradient_accumulation_steps=1,
        clear_previous_experiments=False,
        validation_image_scale=1,
        only_save_last_checkpoint=False,
        use_profiling=False,
        profile_frequency=1,
        row_limit=10,
    ):
        super().__init__()
        self.model = None
        # instantiate accelerator
        self.batch_size = batch_size
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
        self.row_limit = row_limit

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
            # Calculate the new height based on the scale factor
            new_height = int(images[0].shape[0] * self.validation_image_scale)

            # Calculate the aspect ratio of the original image
            aspect_ratio = images[0].shape[1] / images[0].shape[0]

            # Calculate the new width based on the new height and aspect ratio
            new_width = int(new_height * aspect_ratio)

            # Resize the images using the new width and height
            output_size = (new_width, new_height)
            images_pil = [Image.fromarray(image) for image in images]
            images_pil_resized = [image_pil.resize(output_size) for image_pil in images_pil]
            images = [np.array(image_pil) for image_pil in images_pil_resized]

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
                pbar.update(self.batch_size * self.gradient_accumulation_steps)

                # show some extra information on the tqdm progress bar.
                #pbar.set_postfix_str(f"Step: {int(self.steps.item())}")
                #print (logs)
                if logs:
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
                            logger.info(prof.key_averages().table(sort_by='cpu_time_total', row_limit=self.row_limit))
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