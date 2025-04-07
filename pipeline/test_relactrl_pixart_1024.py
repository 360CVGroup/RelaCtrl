import argparse
import numpy as np
from typing import List, Union

from diffusers.models import AutoencoderKL
import requests
import PIL
from io import BytesIO
from PIL import Image
import json
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor
import cv2

import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
relactrl_path = current_dir.parent
sys.path.insert(0, str(relactrl_path))

import torch
import torchvision.transforms as T
from torchvision.utils import _log_api_usage_once, make_grid, save_image

from diffusion import IDDPM, DPMS, SASolverSampler
from diffusion.data.datasets import *
from diffusion.model.nets import PixArtMS_XL_2, ControlPixArtMSHalf
from diffusion.model.nets import ControlPixArtMSHalf_RelaCtrl
from diffusion.model.t5 import T5Embedder
from diffusion.model.utils import prepare_prompt_ar, resize_and_crop_tensor
from diffusion.utils.misc import read_config
from diffusion.download import find_model, my_load_model

vae_scale = 0.18215

@torch.no_grad()
def ndarr_image(tensor: Union[torch.Tensor, List[torch.Tensor]], **kwargs, ) -> None:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return ndarr


def set_env():
    torch.manual_seed(0)
    torch.set_grad_enabled(False)


@torch.inference_mode()
def generate_img(prompt, given_image_npz, seed):
    torch.manual_seed(seed)
    torch.cuda.empty_cache()

    prompt_clean, prompt_show, hw, ar, custom_hw = prepare_prompt_ar(prompt, base_ratios, device=device)  # ar for aspect ratio
    prompt_clean = prompt_clean.strip()
    if isinstance(prompt_clean, str):
        prompts = [prompt_clean]
    caption_embs, emb_masks = llm_embed_model.get_text_embeddings(prompts)
    caption_embs = caption_embs[:, None]

    null_y = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None]

    if given_image_npz is not None:
        condition = given_image_npz
        c = condition * vae_scale
        c_vis = vae.decode(condition)['sample']
        c_vis = torch.clamp(255 * c_vis, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
    else:
        c_vis = None
        c = None

    latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
    # Sample images:
    if args.sampling_algo == 'iddpm':
        # Create sampling noise:
        n = len(prompts)
        z = torch.randn(n, 4, latent_size, latent_size, device=device).repeat(2, 1, 1, 1)
        model_kwargs = dict(y=torch.cat([caption_embs, null_y]), cfg_scale=args.cfg_scale,
                            data_info={'img_hw': hw, 'aspect_ratio': ar},
                            mask=emb_masks, c=c)
        diffusion = IDDPM(str(args.num_sampling_steps))
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    elif args.sampling_algo == 'dpm-solver':
        # Create sampling noise:
        n = len(prompts)
        z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks, c=c)
        dpm_solver = DPMS(model.forward_with_dpmsolver,
                          condition=caption_embs,
                          uncondition=null_y,
                          cfg_scale=args.cfg_scale,
                          model_kwargs=model_kwargs)
        samples = dpm_solver.sample(
            z,
            steps=args.num_sampling_steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )

    elif args.sampling_algo == 'sa-solver':
        # Create sampling noise:
        n = len(prompts)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks, c=c)
        sas_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
        samples = sas_solver.sample(
            S=args.num_sampling_steps,
            batch_size=n,
            shape=(4, latent_size_h, latent_size_w),
            eta=1,
            conditioning=caption_embs,
            unconditional_conditioning=null_y,
            unconditional_guidance_scale=args.cfg_scale,
            model_kwargs=model_kwargs,
        )[0]

    samples = vae.decode(samples / vae_scale).sample
    torch.cuda.empty_cache()
    samples = resize_and_crop_tensor(samples, custom_hw[0, 1], custom_hw[0, 0])
    return ndarr_image(samples, normalize=True, value_range=(-1, 1)), c_vis, prompt_show


def load_img_2_vae_feature(image):
    transform = T.Compose([
        T.Resize(1024),  # Image.BICUBIC
        T.CenterCrop(1024),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    image = Image.fromarray(image)
    img = transform(image.convert("RGB"))
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        posterior = vae.encode(img).latent_dist
    mean = posterior.mean.squeeze(0)
    std = posterior.std.squeeze(0)
    sample = randn_tensor(mean.shape, generator=None, device=mean.device, dtype=mean.dtype)
    z = mean + std * sample
    return z


def initialize_model(args, config, t5_model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.image_size in [512, 1024], "We only provide pre-trained models for 1024x1024 resolutions."
    
    lewei_scale = {512: 1, 1024: 2}
    latent_size = args.image_size // 8
    weight_dtype = torch.float16
    print(f"Inference with {weight_dtype}")

    model = PixArtMS_XL_2(input_size=latent_size, lewei_scale=lewei_scale[config.image_size])
    print('Model architecture: RelaCtrl_v1, image size: 1024')
    model = ControlPixArtMSHalf_RelaCtrl(model, copy_blocks_num=11).to(device)

    state_dict = find_model(args.model_path)['state_dict']
    if 'pos_embed' in state_dict:
        del state_dict['pos_embed']
    elif 'base_model.pos_embed' in state_dict:
        del state_dict['base_model.pos_embed']
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print('Missing keys (missing pos_embed is normal):', missing)
    print('Unexpected keys:', unexpected)

    model.eval()
    model.to(weight_dtype)

    display_model_info = f'model path: {args.model_path},\n base image size: {args.image_size}'
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

    vae = AutoencoderKL.from_pretrained(args.tokenizer_path).to(device)

    print("Begin loading T5 model...")
    llm_embed_model = T5Embedder(
        device=device, local_cache=True, 
        cache_dir=t5_model_path,
        torch_dtype=torch.float
    )
    print("Finished loading T5 model.")

    return model, vae, llm_embed_model, device, base_ratios


def image2vae_canny(image_path, device, low_threshold=100, high_threshold=200):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    refer_img = cv2.Canny(image, low_threshold, high_threshold)
    refer_img_vae = load_img_2_vae_feature(refer_img).unsqueeze(0).to(device)
    return refer_img_vae


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="config")
    parser.add_argument('--num_sampling_steps', default=20, type=int)
    parser.add_argument('--cfg_scale', default=4.5, type=int)
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--model_path', default='/home/jovyan/maao-data-cephfs-2/dataspace/caoke/PixArt-alpha/output_relactrl/release/relactrl_pixart_canny_1024.pth', type=str)
    parser.add_argument('--tokenizer_path', default='/home/jovyan/maao-data-cephfs-0/dataspace/maao/projects/Common/models/PixArt-alpha/PixArt-XL-2-1024-MS/vae', type=str)
    parser.add_argument('--llm_model', default='t5', type=str)
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--condition_strength', default=1, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    config = read_config(args.config)
    set_env()
    
    t5_model_path = "/home/jovyan/maao-data-cephfs-0/dataspace/maao/projects/Common/models/PixArt-alpha/"
    model, vae, llm_embed_model, device, base_ratios = initialize_model(args, config, t5_model_path)
    
    prompt = "a large, well-maintained estate with a red brick driveway and a beautifully landscaped yard. The property is surrounded by a forest, giving it a serene and peaceful atmosphere. The house is situated in a neighborhood with other homes nearby, creating a sense of community.In the yard, there are several potted plants, adding to the lush greenery of the area. A bench is also present, providing a place for relaxation and enjoyment of the surroundings. The overall scene is picturesque and inviting, making it an ideal location for a family home."
    image_path = "/home/jovyan/maao-data-cephfs-2/workspace/caoke/projects/RelaCtrl/resources/demos/reference_images/example1.png"
    
    refer_img_vae = image2vae_canny(image_path, device)
    output_img, c_vis, prompt_show = generate_img(prompt=prompt, given_image_npz=refer_img_vae, seed=123)
    output_img = Image.fromarray(output_img)
    output_img.save("./output/result1.png")