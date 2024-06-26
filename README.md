# SwissImage Vision

- [Overview](#overview)
- [Installation](#installation)
  - [Setup Conda Environment](#setup-conda-environment)
  - [Install Dependencies](#install-dependencies)
  - [Setup DVC](#setup-dvc)
- [Reproduce the Experiment](#reproduce-the-experiment)
- [Concepts](#concepts)
- [Setup](#setup)
- [Resources](#resources)
  - [Data](#data)
  - [Projects](#projects)

## Overview

The goal of this project is to detect solar panels in aerial images (SwissImage 0.1m) using deep learning.

MLOps practices are used to manage the project, including DVC for data versioning, Label Studio for data labeling, and BentoML for model serving. This allows for reproducibility, collaboration and scalability of the project.

## Installation

Clone the repository:

```bash
git clone https://github.com/heigvd-software-engineering/swissimage-vision.git
cd swissimage-vision
```

### Setup Conda Environment

> [!NOTE]
> A GPU is required to run the pipeline.

Installing MiniConda is optional but is recommended in environments where you do not have sudo access to install system packages. You can skip this step

To install MiniConda, you can run the following script to install it:

```bash
./scripts/install_conda.sh
```

Next, restart your terminal and create a new conda environment and install the dependencies:

```bash
conda create --name swissimage-vision python=3.12 pip
conda activate swissimage-vision
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Setup DVC

Add the MinIO credentials to the DVC configuration:

```bash
dvc remote modify --local minio access_key_id <ACCESS_KEY_ID>
dvc remote modify --local minio secret_access_key <SECRET_ACCESS_KEY>
```

Pull the data from the remote storage:

```bash
dvc pull
```

## Reproduce the Experiment

To reproduce the experiment, execute the following command:

```bash
dvc repro
```

To view the training logs, run the following command:

```bash
tensorboard --logdir out/pre-train/lightning_logs
```

<!-- DEPRECATED
## Serving the Model

### Gradio Demo

Run the following command to start the Gradio demo interface:

```bash
python3 src/demo.py
```

### BentoML API

```bash
python3 src/serve.py
```
-->

## Concepts

To learn more about the integration and concepts used in this project, refer to the [docs/concepts.md](docs/concepts.md) file.

## Setup

To replicate the setup of this project, refer to the [docs/setup.md](docs/setup.md) file.

## Resources

### Data

- https://zenodo.org/records/7358126 (currently used)
- http://2019.geopython.net/data/solar.zip
- https://ep2020.europython.eu/media/conference/slides/detecting-and-analyzing-solar-panels-switzerland-using-aerial-imagery.pdf
- https://zenodo.org/records/5171712

### Projects

- https://github.com/shashankag14/fine-tune-object-detector/blob/main/fine_tune.ipynb
- https://github.com/swiss-ai-center/giscup2023-resnet-unet
- https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- https://github.com/saizk/Deep-Learning-for-Solar-Panel-Recognition?tab=readme-ov-file
