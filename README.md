<div align='center'>
<h1>Visual Spatial Tuning</h1>


[![Paper](https://img.shields.io/badge/paper-5f16a8?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.05491)
[![Project Page](https://img.shields.io/badge/Blog-3858bf?style=for-the-badge&logo=homepage&logoColor=white)](https://yangr116.github.io/vst_project/)
[![Weights](https://img.shields.io/badge/Model%20Weights-63cad3?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/collections/rayruiyang/vst)
</div>

We introduce **Visual Spatial Tuning (VST)**, a comprehensive framework designed to cultivate Vision-Language Models (VLMs) with human-like visuospatial abilities—from spatial perception to advanced reasoning.

![Teaser Image](assets/teaser_project_page.jpg)

---

## 💡 Key Highlights

✨ **VST-P**: 4.1M samples across 19 skills, spanning single images, multi-image scenarios, and videos—boosting spatial perception in VLMs.  
✨ **VST-R**: 135K curated samples that teach models to reason in space, including step-by-step reasoning and rule-based data for reinforcement learning.  
✨ **Progressive Training Pipeline**: Start with supervised fine-tuning to build foundational spatial knowledge, then reinforce spatial reasoning abilities via RL. VST achieves state-of-the-art results on spatial benchmarks (34.8% on MMSI-Bench, 61.2% on VSIBench) without compromising general capabilities.  
✨ **Vision-Language-Action Models Enhanced**: The VST paradigm significantly strengthens spatial tuning, paving the way for more physically grounded AI.

---

## 📊 Dataset Overview

![Dataset Image](assets/dataset.jpg)

### 🖼️ VST-Perception (VST-P)
- **4.1M samples** across **19 tasks** for supervised fine-tuning.
- Covers three primary vision scenarios: *single-image*, *multi-image*, and *video*.
- VLMs tuned on VST-P show strong improvements in spatial perception:
  - ~20% boost on CVBench-3D
  - ~5% increase on BLINK
  - ~16% gain on VSIBench

### 🧠 VST-Reasoning (VST-R)
- **135K samples**, split into:
  - **Reasoning steps (CoT)**: Teach models how to reason spatially.
  - **Rule-checkable data**: Used in online RL to further enhance reasoning skills.
- VLMs tuned on VST-R demonstrate:
  - 8.9% improvement on MMSI-Bench

---

## 🏷️ Model Card

| Model Name     | 🤗 HuggingFace Download |
|:-------------- |:----------------------:|
| VST-3B-SFT     | [Download](https://huggingface.co/rayruiyang/VST-3B-SFT)           |
| VST-3B-RL      | [Download](https://huggingface.co/rayruiyang/VST-3B-RL)           |
| VST-7B-SFT     | [Download](https://huggingface.co/rayruiyang/VST-7B-SFT)           |
| VST-7B-RL      | [Download](https://huggingface.co/rayruiyang/VST-7B-RL)           |

### 📈  Spatial & General Benchmarks

| Models              | CV   | 3DSR | MMSI | BLINK | VSI  | MMStar | MMB  | RealworldQA | MMMU | OCRB | AI2D |
|---------------------|------|------|------|-------|------|--------|------|-------------|------|------|------|
| VST-3B-SFT   | 84.4 | 54.1 | 30.2 | 59.1  | 57.9 | 58.0   | 80.9 | 68.4        | 45.2 | 83.7 | 82.5 |
| VST-3B-RL   | 84.2 | 56.5 | 31.3 | 57.2  | 57.7 | 58.9   | 80.5 | 68.5        | 49.8 | 80.9 | 82.4 |
| VST-7B-SFT   | 85.5 | 54.6 | 32.0 | 62.1  | 60.6 | 63.1   | 83.3 | 72.2        | 50.6 | 85.5 | 84.9 |
| VST-7B-RL| 86.5 | 60.1 | 34.8 | 62.6 | 61.2 | 63.5 | 83.0 | 68.5 | 49.4 | 86.1 | 83.5 |

### 📈  VSIBench

| Methods               | Avg. | Obj. Count | Abs. Dist. | Obj. Size | Room Size | Rel. Dist | Rel. Dir. | Route Plan | Appr. Order |
|-----------------------|------|------------|------------|-----------|-----------|-----------|-----------|------------|-------------|
| VST-3B-SFT      | 57.9 | 69.3       | 45.4       | 71.8      | 62.4      | 59.0      | 46.0      | 38.7       | 70.2    |
| VST-3B-RL      | 57.7 | 66.6       | 45.0       | 72.8      | 60.9      | 59.9      | 47.6      | 40.7       | 68.3        |
| VST-7B-SFT      | 60.6 | 72.0   | 44.4       | 74.3      | 68.3      | 59.7      | 55.8      | 44.9       | 65.2        |
| VST-7B-RL      | 61.2 | 71.6   | 43.8       | 75.5  | 69.2  | 60.0      | 55.6      | 44.3       | 69.2        |


---



## ⚡ Getting Started

### Installation

```bash
git clone https://github.com/Yangr116/VST
cd VST
pip install -e .
```

### Train

SFT: [docs/train.md](./docs/train.md)

Adapt to VLA model: [vla.md](./vla.md)


### Evaluation

Please see [docs/evaluation.md](./docs/evaluation.md)


## 📜 License
This project is licensed under the Apache License. See the [LICENSE](./LICENSE) file for details.

The VST-3B model is fine-tuned from Qwen2.5VL-3B, its license is [Qwen2.5VL-3B LICENSE](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE).

## Acknowledgement

Thanks for the projects: [Qwen2.5VL](https://github.com/QwenLM/Qwen3-VL/tree/main), [VeOmni](https://github.com/ByteDance-Seed/VeOmni), [EasyR1](https://github.com/hiyouga/EasyR1), and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

If you find VST useful for your research or applications, please ⭐ star the repo or cite our work:

```bibtex
@article{vst,
  title={Visual Spatial Tuning},
  author={Rui Yang, Ziyu Zhu, Yanwei Li, Jingjia Huang, Shen Yan, Siyuan Zhou, Zhe Liu, Xiangtai Li, Shuangye Li, Wenqian Wang, Yi Lin, Hengshuang Zhao},
  journal={arXiv preprint arXiv:2511.05491},
  year={2025}
}
```
