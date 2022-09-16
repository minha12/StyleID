# StyleID - Official Pytorch implementation

> **StyleID: Identity Disentanglement for Anonymizing Faces**<br>
> Minh-Ha Le and Niklas Carlsson <br>
>
>**Abstract:** Privacy of machine learning models is one of the remaining challenges that hinder the broad adoption of Artificial Intelligent (AI). This paper considers this problem in the context of image datasets containing faces. Anonymization of such datasets is becoming increasingly important due to their central role in the training of autonomous cars, for example, and the vast amount of data generated by surveillance systems. While most prior work de-identifies facial images by modifying identity features in pixel space, we instead project the image onto the latent space of a Generative Adversarial Network (GAN) model, find the features that provide the biggest identity disentanglement, and then manipulate these features in latent space, pixel space, or both. The main contribution of the paper is the design of a feature-preserving anonymization framework, StyleID, which protects the individuals’ identity, while preserving as many characteristics of the original faces in the image dataset as possible. As part of the contribution, we present a novel disentanglement metric, three complementing disentanglement methods, and new insights into identity disentanglement. StyleID provides tunable privacy, has low computational complexity, and is shown to outperform current state-of-the-art solutions.

## Description

This is the official implementation of StyleID, a framework to disentangle and anonymize identity in facial images. The framework uses the highly disentangled latent space of a pre-trained StyleGAN generator. Three methods of disentanglement for anonymizing face are presented:

- (1) Disentanglement in latent space
- (2) Disentanglement in pixel space
- (3) Latent swapper

The face generator is [Rosinality](https://github.com/rosinality/stylegan2-pytorch/) pytorch implementation of StyleGAN2. The model has been modified to fit with our framework.

Following the instruction from [Rosinality](https://github.com/rosinality/stylegan2-pytorch/) to setup the face generator and a pre-trained model of StyleGAN2 will downloaded by our provided script ```download_files.py``` (or manually download from [link](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing)). Furthermore, few others files will be downloaded including:

- Latent codes of faces in CelebA-HQ dataset
- Identity disentanglement rank for single channels
- Identity disentanglement rank for group of 256 channels

In addition, we provide various auxiliary models needed for training the mapper as well as pretrained models needed for computing the identity disentanglement metrics reported in the paper.
| Path | Description
| :--- | :----------
|[Segmentation to Image](https://drive.google.com/file/d/1VpEKc6E6yG3xhYuZ0cq8D2_1CbT0Dstz/view?usp=sharing) | [pSp](https://github.com/eladrich/pixel2style2pixel) trained with the CelebAMask-HQ dataset for image synthesis from segmentation maps.
|[IR-SE50 Model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our ID loss during pSp training.
|[CurricularFace Backbone](https://drive.google.com/file/d/1f4IwVa2-Bn9vWLwB-bUwm53U_MlvinAj/view?usp=sharing)  | Pretrained CurricularFace model taken from [HuangYG123](https://github.com/HuangYG123/CurricularFace) for use in ID similarity metric computation.
|[MTCNN](https://drive.google.com/file/d/1tJ7ih-wbCO6zc3JhI_1ZGjmwXKKaPlja/view?usp=sharing)  | Weights for MTCNN model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in ID similarity metric computation. (Unpack the tar.gz to extract the 3 model weights.)

All of these models should be placed in ```./pretrained_models/``` folder.

## Usage

Given an input facial image, the identity can be anonymized by swaping to a specific target or randomly generated target. For the simplicity of demonstration, we choose both input face and target face randomly.

All the three methods are demonstrated in `notebook.ipynb` notebook ([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github.com/minha12/StyleID/blob/main/StyleID.ipynb)).

## Citation:

If you use our code, please cite our paper:

Minh-Ha Le, Niklas Carlsson, “StyleID: Identity Disentanglement for Anonymizing Faces”, Proceedings on Privacy Enhancing Technologies (PoPETs), Volumn 2023.1, June 15, 2023.
