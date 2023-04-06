import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from pathlib import Path
from datasets import load_dataset
import os
from muse_maskgit_pytorch import (
    VQGanVAE,
    VQGanVAETaming,
    MaskGitTrainer,
    MaskGit,
    MaskGitTransformer,
    get_accelerator,
)
from muse_maskgit_pytorch.dataset import (
    get_dataset_from_dataroot,
    ImageTextDataset,
    split_dataset_into_dataloaders,
)

import argparse


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser()
    #parser.add_argument(
        #"--validation_image_scale",
        #default=1,
        #type=float,
        #help="Factor by which to scale the validation images.",
    #)
    parser.add_argument(
        "--clear_previous_experiments",
        action="store_true",
        help="Whether to clear previous experiments.",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=256,
        help="Number of tokens. Must be same as codebook size above",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=1024,
        help="The sequence length. Must be equivalent to fmap_size ** 2 in vae",
    )
    parser.add_argument("--depth", type=int, default=2, help="The depth of model")
    parser.add_argument(
        "--dim_head", type=int, default=64, help="Attention head dimension"
    )
    parser.add_argument("--heads", type=int, default=8, help="Attention heads")
    parser.add_argument(
        "--ff_mult", type=int, default=4, help="Feed forward expansion factor"
    )
    parser.add_argument(
        "--t5_name", type=str, default="t5-large", help="Name of your t5 model"
    )
    parser.add_argument(
        "--cond_image_size", type=int, default=None, help="Conditional image size."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A photo of a dog",
        help="Prompt to use for generation, you can use multiple prompts separated by |.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100,
        help="Number of steps to use for generating the image. Default: 100")
    parser.add_argument(
        "--cond_scale",
        type=float,
        default=3.0,
        help="Conditional Scale to use for generating the image. Default: 3.0")

    parser.add_argument(
        "--max_grad_norm", type=float, default=None, help="Max gradient norm."
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument(
        "--valid_frac", type=float, default=0.05, help="validation fraction."
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use ema.")
    parser.add_argument("--ema_beta", type=float, default=0.995, help="Ema beta.")
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
        help="Precision to train on.",
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
        "--vae_path",
        type=str,
        default=None,
        help="Path to the vae model. eg. 'results/vae.steps.pt'",
    )
    parser.add_argument("--dim", type=int, default=128, help="Model dimension.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch Size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient Accumulation.",
    )
    parser.add_argument("--vq_codebook_size", type=int, default=256, help="Image Size.")
    parser.add_argument(
        "--cond_drop_prob",
        type=float,
        default=0.5,
        help="Conditional dropout, for classifier free guidance.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size. You may want to start with small images, and then curriculum learn to larger ones, but because the vae is all convolution, it should generalize to 512 (as in paper) without training on it",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Path to the last saved checkpoint. 'results/maskgit.steps.pt'",
    )
    parser.add_argument(
        "--taming_model_path",
        type=str,
        default=None,
        help="path to your trained VQGAN weights. This should be a .ckpt file. (only valid when taming option is enabled)",
    )

    parser.add_argument(
        "--taming_config_path",
        type=str,
        default=None,
        help="path to your trained VQGAN config. This should be a .yaml file. (only valid when taming option is enabled)",
    )
    # Parse the argument
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = get_accelerator(
        log_with=args.log_with,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        logging_dir=args.logging_dir,
    )

    if args.vae_path and args.taming_model_path:
        raise Exception("You can't pass vae_path and taming args at the same time.")

    if args.vae_path:
        accelerator.print("Loading Muse VQGanVAE")
        vae = VQGanVAE(dim=args.dim, vq_codebook_size=args.vq_codebook_size).to(
            accelerator.device
        )

        accelerator.print("Resuming VAE from: ", args.vae_path)
        vae.load(
            args.vae_path
        )  # you will want to load the exponentially moving averaged VAE

    elif args.taming_model_path:
        print("Loading Taming VQGanVAE")
        vae = VQGanVAETaming(
            vqgan_model_path=args.taming_model_path,
            vqgan_config_path=args.taming_config_path,
        )
        args.num_tokens = vae.codebook_size
        args.seq_len = vae.get_encoded_fmap_size(args.image_size) ** 2
    if accelerator.is_main_process:
        accelerator.init_trackers("muse_maskgit", config=vars(args))
    # then you plug the vae and transformer into your MaskGit as so

    # (1) create your transformer / attention network
    transformer = MaskGitTransformer(
        num_tokens=args.num_tokens
        if args.num_tokens
        else args.vq_codebook_size,  # must be same as codebook size above
        seq_len=args.seq_len,  # must be equivalent to fmap_size ** 2 in vae
        dim=args.dim,  # model dimension
        depth=args.depth,  # depth
        dim_head=args.dim_head,  # attention head dimension
        heads=args.heads,  # attention heads,
        ff_mult=args.ff_mult,  # feedforward expansion factor
        t5_name=args.t5_name,  # name of your T5
    ).to(accelerator.device)
    transformer.t5.to(accelerator.device)

    # (2) pass your trained VAE and the base transformer to MaskGit

    maskgit = MaskGit(
        vae=vae,  # vqgan vae
        transformer=transformer,  # transformer
        image_size=args.image_size,  # image size
        cond_drop_prob=args.cond_drop_prob,  # conditional dropout, for classifier free guidance
        cond_image_size=args.cond_image_size,
    ).to(accelerator.device)

    # load the maskgit transformer from disk if we have previously trained one
    if args.resume_path:
        accelerator.print(f"Resuming MaskGit from: {args.resume_path}")
        maskgit.load(args.resume_path)
    else:
        accelerator.print("We need a MaskGit model to do inference with. Please provide a path to a checkpoint..")

    # ready your training text and images
    images = maskgit.generate(
        texts=list(args.prompt) if '|' not in args.prompt else str(args.prompt).split("|"),
        #texts = [
            #'a whale breaching from afar',
            #'young girl blowing out candles on her birthday cake',
            #'fireworks with blue and green sparkles'
            #],
        cond_scale = 3.0, # conditioning scale for classifier free guidance
        timesteps = args.timesteps,
        )

    print(images.shape) # (3, 3, 256, 256)

    # save image to disk
    save_path = str(f"{args.results_dir}/result_maskgit.png")
    os.makedirs(str(f"{args.results_dir}/"), exist_ok = True)

    save_image(images, save_path)


if __name__ == "__main__":
    main()