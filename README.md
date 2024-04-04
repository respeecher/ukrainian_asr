# Ukrainian ASR from Respeecher

Training code for a Ukrainian ASR system is based on the [Data2Vec model](https://huggingface.co/Respeecher/ukrainian-data2vec-asr), which was published by [Respeecher](https://www.respeecher.com/). Respeecher provides Hollywood-quality speech-to-speech and text-to-speech AI voices to businesses and various types of content creators.

# Setup Working Environment

## Pre-requirements

- conda 4.12.0 (later versions may also work) - [Installation](https://docs.anaconda.com/anaconda/install/index.html)
- (Optional) CUDA Version: 11.4; Driver Version: 470.129.06 - [Installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

## Setup environment

```bash
conda env create -f environment.yaml
```

## Train

```bash
CUDA_VISIBLE_DEVICES="{gpu}" python torch_train.py
```

## Results

Best model can be found in `logdirs/torch_asr_on_ukrainian_data2vec_cosinev3/best_model`

Metric JSON can be found in `logdirs/torch_asr_on_ukrainian_data2vec_cosinev3/metric.json`
