import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from pathlib import Path
from datasets import load_dataset
import os
from muse_maskgit_pytorch import (
    VQGanVAE,
    MaskGitTrainer,
    MaskGit,
    MaskGitTransformer,
    get_accelerator
)
from muse_maskgit_pytorch.dataset import get_dataset_from_dataroot, ImageTextDataset, split_dataset_into_dataloaders

import argparse

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_tokens", type=int, default=256, help="Number of tokens. Must be same as codebook size above"
    )
    parser.add_argument(
        "--seq_len", type=int, default=1024, help="The sequence length. Must be equivalent to fmap_size ** 2 in vae"
    )
    parser.add_argument(
        "--dim", type=int, default=128, help="Model dimension"
    )
    parser.add_argument(
        "--depth", type=int, default=2, help="The depth of model"
    )
    parser.add_argument(
        "--dim_head", type=int, default=64, help="Attention head dimension"
    )
    parser.add_argument(
        "--heads", type=int, default=8, help="Attention heads"
    )
    parser.add_argument(
        "--ff_mult", type=int, default=4, help="Feed forward expansion factor"
    )
    parser.add_argument(
        "--t5_name", type=str, default="t5-small", help="Name of your t5 model"
    )
    parser.add_argument(
        "--cond_image_size", type=int, default=None, help="Conditional image size."
    )
    parser.add_argument(
        "--cond_drop_prob", type=float, default=0.25, help="Conditional dropout. Related to classifier free guidance"
    )
    parser.add_argument(
        "--validation_prompt", type=str, default="A photo of a dog", help="Validation prompt."
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=None, help="Max gradient norm."
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
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
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
    # Duplicated line
    # parser.add_argument("--dim", type=int, default=128, help="Model dimension.")
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
    accelerate_kwargs={
        'mixed_precision': "no", # 'fp16',
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'device_placement': False,
        'split_batches': True,
        "log_with": "tensorboard",
        "logging_dir": "."
    }
    accelerator = get_accelerator(**accelerate_kwargs)
    if args.train_data_dir:
        dataset = get_dataset_from_dataroot(args.train_data_dir, args)
    elif args.dataset_name:
        dataset = load_dataset(args.dataset_name)


    vae = VQGanVAE(
        dim = args.dim,
        vq_codebook_size = args.vq_codebook_size
    ).cuda()

    print ('Resuming VAE from: ', args.resume_from)
    vae.load(args.resume_from)    # you will want to load the exponentially moving averaged VAE

    # then you plug the vae and transformer into your MaskGit as so

    # (1) create your transformer / attention network

    transformer = MaskGitTransformer(
        num_tokens = args.num_tokens,         # must be same as codebook size above
        seq_len = args.seq_len,               # must be equivalent to fmap_size ** 2 in vae
        dim = args.dim,                       # model dimension
        depth = args.depth,                   # depth
        dim_head = args.dim_head,             # attention head dimension
        heads = args.heads,                   # attention heads,
        ff_mult = args.ff_mult,               # feedforward expansion factor
        t5_name = args.t5_name,               # name of your T5
    )

    # (2) pass your trained VAE and the base transformer to MaskGit
    
    # print(dir(transformer.tokenizer))

    maskgit = MaskGit(
        vae = vae,                 # vqgan vae
        transformer = transformer, # transformer
        image_size = args.image_size,          # image size
        cond_drop_prob = args.cond_drop_prob,     # conditional dropout, for classifier free guidance
        cond_image_size = args.cond_image_size
    ).cuda()
    dataset = ImageTextDataset(dataset, args.image_size, transformer.tokenizer, image_column=args.image_column, caption_column=args.caption_column)
    dataloader, validation_dataloader = split_dataset_into_dataloaders(dataset, args.valid_frac, args.seed, args.batch_size)

    trainer = MaskGitTrainer(
        maskgit,\
        dataloader,
        validation_dataloader,
        accelerator,
        current_step=0,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        # image_size=args.image_size,  # you may want to start with small images, and then curriculum learn to larger ones, but because the vae is all convolution, it should generalize to 512 (as in paper) without training on it
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
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
        validation_prompt=args.validation_prompt,
        accelerate_kwargs=accelerate_kwargs
    )

    trainer.train()



if __name__ == "__main__":
    main()