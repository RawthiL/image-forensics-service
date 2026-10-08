# community-forensics

Community Forensics — Park & Owens, CVPR 2025.

| | |
|---|---|
| Paper | [arXiv:2411.04125](https://arxiv.org/abs/2411.04125) · [CVPR 2025 open access](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html) |
| Reference code | [JeongsooP/Community-Forensics](https://github.com/JeongsooP/Community-Forensics) — MIT |
| Weights | [`OwensLab/commfor-model-384`](https://huggingface.co/OwensLab/commfor-model-384) · [`-224`](https://huggingface.co/OwensLab/commfor-model-224) — MIT |
| Training data | [`OwensLab/CommunityForensics`](https://huggingface.co/datasets/OwensLab/CommunityForensics) — CC-BY-4.0 |
| Project page | <https://jespark.net/projects/2024/community_forensics/> |

## Architecture

A **single ViT-S/16 with a one-logit head** — 21.8M parameters, ~44MB in fp16.
The "thousands of generators" in the title is the *dataset* (2.7M images from
4,803 generators), not an ensemble. This is why it runs on a CPU-only box.

Training: AdamW, lr 2e-5, weight decay 1e-2, batch 512, cosine schedule with 20%
warmup, 52K iterations. Backbone trained end-to-end; frozen backbones did worse.

## Options

Pass via `IFS_DETECTOR_OPTIONS` (JSON):

| Option | Default | Notes |
|---|---|---|
| `input_size` | `384` | `384` or `224`. The 384 checkpoint is the stronger one |

## Known limits

* **Modern generators evade it.** An independent 2026 benchmark of open
  detectors out-of-the-box found Flux Dev, Firefly v4 and Midjourney v7 detected
  at only **18–30%** average accuracy across all methods tested, with accuracy
  decaying from ~79% on 2020-era generators to ~38% on 2024-era ones
  ([arXiv:2602.07814](https://arxiv.org/html/2602.07814v1)).
* **Headline numbers depend on who measures.** The paper reports 0.987 mAP /
  89.2% accuracy on its own eval set; the independent benchmark puts the same
  checkpoint at **75.0% mean / 82.1% median** in the wild — still the best of
  everything it tested.
* **Training data is 99.67% latent diffusion** (0.25% GAN, 0.06% pixel
  diffusion). That skew is the direct cause of the Flux/Midjourney weakness.
* **Binary only.** No per-generator head, so the Hive dialect's source head is
  approximated. No face-manipulation head, so `deepfake` is never filled.

## Reproducibility

Preprocessing is `Resize(short side → 440) → CenterCrop(384) → ToTensor →
Normalize(ImageNet)`, reproduced exactly in [`preprocess.py`](preprocess.py).
It moves the score more than the model does — pin it, don't tune it.

Set `IFS_WEIGHTS_REVISION` to a commit sha in production. The resolved sha ends
up in the model fingerprint.

## Citation

```bibtex
@inproceedings{park2025community,
  title     = {Community Forensics: Using Thousands of Generators to Train Fake Image Detectors},
  author    = {Park, Jeongsoo and Owens, Andrew},
  booktitle = {CVPR},
  year      = {2025}
}
```
