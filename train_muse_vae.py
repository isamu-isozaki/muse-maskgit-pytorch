import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from pathlib import Path
from datasets import load_dataset
import os
from muse_maskgit_pytorch import (
    VQGanVAE,
    VQGanVAETrainer,
    get_accelerator
)
from muse_maskgit_pytorch.dataset import get_dataset_from_dataroot, ImageDataset, split_dataset_into_dataloaders


import argparse
def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_grad_norm", type=float, default=None, help="Max gradient norm."
    )
    parser.add_argument(
        "--discr_max_grad_norm", type=float, default=None, help="Max gradient norm for discriminator."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed."
    )
    parser.add_argument(
        "--valid_frac", type=float, default=0.05, help="validation fraction."
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use ema."
    )
    parser.add_argument(
        "--ema_beta", type=float, default=0.995, help="Ema beta."
    )
    parser.add_argument(
        "--ema_update_after_step", type=int, default=1, help="Ema update after step."
    )
    parser.add_argument(
        "--ema_update_every", type=int, default=1, help="Ema update every this number of steps."
    )
    parser.add_argument(
        "--apply_grad_penalty_every", type=int, default=4, help="Apply gradient penalty every this number of steps."
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Precision to train on."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Path to save the training samples and checkpoints",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="results/logs",
        help="Path to log the losses and LR",
    )

    # vae_trainer args
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Path to the vae model. eg. 'results/vae.steps.pt'",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the huggingface dataset used."
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Dataset folder where your input images for training are.",
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=50000,
        help="Total number of steps to train for. eg. 50000.",
    )
    parser.add_argument("--dim", type=int, default=128, help="Model dimension.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning Rate.")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Gradient Accumulation."
    )
    parser.add_argument(
        "--save_results_every",
        type=int,
        default=100,
        help="Save results every this number of steps.",
    )
    parser.add_argument(
        "--save_model_every",
        type=int,
        default=500,
        help="Save the model every this number of steps.",
    )
    parser.add_argument("--vq_codebook_size", type=int, default=256, help="Image Size.")
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size. You may want to start with small images, and then curriculum learn to larger ones, but because the vae is all convolution, it should generalize to 512 (as in paper) without training on it",
    )
    # Parse the argument
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = get_accelerator(log_with=args.log_with, gradient_accumulation_steps=args.gradient_accumulation_steps,mixed_precision=args.mixed_precision, logging_dir=args.logging_dir)
    if accelerator.is_main_process:
        accelerator.init_trackers("muse_vae", config=vars(args))
    if args.train_data_dir:
        dataset = get_dataset_from_dataroot(args.train_data_dir, args)
    elif args.dataset_name:
        dataset = load_dataset(args.dataset_name)["train"]
    vae = VQGanVAE(dim=args.dim, vq_codebook_size=args.vq_codebook_size)
    dataset = ImageDataset(dataset, args.image_size, image_column=args.image_column)
    # dataloader

    dataloader, validation_dataloader = split_dataset_into_dataloaders(dataset, args.valid_frac, args.seed, args.batch_size)
    trainer = VQGanVAETrainer(
        vae,
        dataloader,
        validation_dataloader,
        accelerator,
        current_step=0,
        num_train_steps=args.num_train_steps,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        discr_max_grad_norm=args.discr_max_grad_norm,
        save_results_every=args.save_results_every,
        save_model_every=args.save_model_every,
        results_dir=args.results_dir,
        logging_dir=args.logging_dir,
        use_ema=args.use_ema,
        ema_beta=args.ema_beta,
        ema_update_after_step=args.ema_update_after_step,
        ema_update_every=args.ema_update_every,
        apply_grad_penalty_every=args.apply_grad_penalty_every,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    trainer.train()



if __name__ == "__main__":
    main()