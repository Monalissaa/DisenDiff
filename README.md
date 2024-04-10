# DisenDiff

This repository is the official implementation of [DisenDiff](https://arxiv.org/abs/2403.18551) [CVPR-2024 Oral Presentation].

> **Attention Calibration for Disentangled Text-to-Image Personalization** <br>
> Yanbing Zhang, Mengping Yang, Qin Zhou, Zhe Wang<br>
> [pdf](https://arxiv.org/abs/2403.18551)

<div>
<p align="center">
<img src='assets/first_figure.jpg' align="center" width=900>
</p>
</div>

## Datasets
The training images are located in `datasets/images`, the test prompts are located in `datasets/prompts`, and the processed images for evaluating image-alignment can be found in `datasets/data_eval`.

## Key modules
The crucial constraints for optimization are implemented in the function `p_losses` within `src/model.py`.

## Results
<div>
<p align="center">
<img src='assets/results_github.jpg' align="center" width=900>
</p>
</div>

## Getting Started
```
conda env create -f environment.yml
conda activate ldm
git clone https://github.com/CompVis/stable-diffusion.git
```

## Fine-tuning
```
## run training
bash run.sh

## sample and evaluate
bash eval.sh
```
The `run.sh` and `eval.sh` scripts include several hyperparameters such as `classes` in the input image,`data_path`, `save_path`, training `caption`, random `seed`, and more. Please modify these executable files to suit your specific requirements.


## Contact Us
**Yanbing Zhang**: [zhangyanbing@mail.ecust.edu.cn](mailto:zhangyanbing@mail.ecust.edu.cn)  
**Mengping Yang**: [kobeshegu@gmail.com](mailto:kobeshegu@gmail.com)  

## BibTeX
```
@article{zhang2024attention,
  title={Attention Calibration for Disentangled Text-to-Image Personalization},
  author={Zhang, Yanbing and Yang, Mengping and Zhou, Qin and Wang, Zhe},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## Acknowledgement
Our code is built upon the excellent codebase of [Custom-Diffusion](https://github.com/adobe-research/custom-diffusion), we thank a lot for their work.
We also kindly refer interesting researchers to these wonderful relted works:

[DreamBooth](https://dreambooth.github.io/)
[Break-A-Scene](https://omriavrahami.com/break-a-scene/)
[Textual Inversion](https://textual-inversion.github.io/)

We also thank the anonymous reviewers for their valuable suggestions during the rebuttal, which greatly help us improve the paper.

## Disclaimer
This project is released for academic use. We disclaim responsibility for user-generated content. Users are solely liable for their actions. The project contributors are not legally affiliated with, nor accountable for, users' behaviors. Use the generative model responsibly, adhering to ethical and legal standards. 
