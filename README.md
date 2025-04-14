# RelaCtrl

This is the official reproduction of [RelaCtrl](https://360cvgroup.github.io/RelaCtrl/), which represents an efficient controlnet-like architecture designed for DiTs.

**[RelaCtrl: Relevance-Guided Efficient Control for Diffusion Transformers](https://arxiv.org/pdf/2502.14377)**
</br>
Ke Cao*, Jing Wang*, Ao Ma*, Jiasong Feng, Zhanjie Zhang, Xuanhua He, Shanyuan Liu, Bo Cheng, Dawei Leng‡, Yuhui Yin, Jie Zhang‡(*Equal Contribution, ‡Corresponding Authors)
</br>
[![arXiv](https://img.shields.io/badge/arXiv-2502.14377-b31b1b.svg)](https://arxiv.org/pdf/2502.14377)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://360cvgroup.github.io/RelaCtrl/)


## 📰 News
- **[2025.04.07]** We released the inference pipeline and some weights of RelaCtrl-PixArt.
- **[2025.02.21]** We have released our paper [RelaCtrl](https://arxiv.org/pdf/2502.14377) and created a dedicated [project homepage](https://360cvgroup.github.io/RelaCtrl/).


## We Are Hiring
We are seeking academic interns in the AIGC field. If interested, please send your resume to [maao@360.cn](mailto:maao@360.cn).

## Inference with RealCtrl on PixArt
### Dependencies and Installation
``` python
conda create -n relactrl python=3.10
conda activate relactrl

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/360CVGroup/RelaCtrl.git
cd RelaCtrl
pip install -r requirements.txt
```

## Download Models

### 1. Required PixArt-related Weights  

Download the necessary model weights for PixArt from the links below:  

| Model         | Parameters | Download Link |
|--------------|------------|----------------------------------------------------------------|
| **T5**       | 4.3B       | [T5](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) |
| **VAE**      | 80M        | [VAE](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema) |
| **PixArt-α-1024** | 0.6B  | [PixArt-XL-2-1024-MS.pth](https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-1024-MS.pth) or [Diffusers Version](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS) |

### 2. RelaCtrl Conditional Weights  

Download the required conditional weights for RelaCtrl:  
| Model         | Parameters | Download Link |
|--------------|------------|----------------------------------------------------------------|
| **RelaCtrl_PixArt_Canny**       | 45M       | [Canny](https://huggingface.co/qihoo360/RelaCtrl/tree/main) |
| **RelaCtrl_PixArt_Canny_Style**       | 45M       | [Style](https://huggingface.co/qihoo360/RelaCtrl/tree/main) |


### Inference with Conditions
``` python
python pipeline/test_relactrl_pixart_1024.py diffusion/configs/config_relactrl_pixart_1024.py
```
Prompt examples for different models can be found in the [prompt_exampeles](resources/demos/prompts_examples.md).

### Acknowledgment  
The PixArt model weights are derived from the open-source project [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha).  
Please refer to the original repository for detailed license information.  

## BibTeX
```
@misc{cao2025relactrl,
                title={RelaCtrl: Relevance-Guided Efficient Control for Diffusion Transformers}, 
                author={Ke Cao and Jing Wang and Ao Ma and Jiasong Feng and Zhanjie Zhang and Xuanhua He and Shanyuan Liu and Bo Cheng and Dawei Leng and Yuhui Yin and Jie Zhang},
                year={2025},
                eprint={2502.14377},
                archivePrefix={arXiv},
                primaryClass={cs.CV},
                url={https://arxiv.org/abs/2502.14377}, 
}
```
