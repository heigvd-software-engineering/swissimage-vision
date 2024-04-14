# SwissImage Vision

- [TODO](#todo)
  - [Label Studio](#label-studio)
  - [CML](#cml)
  - [Model](#model)
- [About](#about)
- [Installation](#installation)
  - [Setup DVC](#setup-dvc)
- [Reproduce the Experiment](#reproduce-the-experiment)
- [Serving the Model](#serving-the-model)
  - [Gradio Demo](#gradio-demo)
  - [BentoML API](#bentoml-api)
- [Integrations](#integrations)
  - [DVC](#dvc)
  - [Label Studio](#label-studio-1)
- [Resources](#resources)
  - [Data](#data)
  - [Projects](#projects)

![Demo](media/demo.png)

## TODO

### Label Studio

- [ ] Documentation for Label Studio
  - [ ] Installation locally
  - [ ] Installation on Kubernetes
  - [ ] Configuration

- [ ] Deploy label studio instance
  - [ ] Add label studio configuration

- [ ] (Deploy BentoML API to Kubernetes cluster)
- [ ] (Deploy label studio model backend)

### CML

- [ ] Documentation for CML
  - [ ] PR preview
  - [ ] GitHub repository setup (prevent fork to run actions)
  - [ ] (Diagram)

- [ ] CML training on Kubernetes cluster
  - [ ] Self hosted runner with terraform (https://medium.com/simform-engineering/how-to-setup-self-hosted-github-action-runner-on-kubernetes-c8825ccbb63c)

- [ ] PR and main branch CI/CD

- [ ] CML reporting
  - [ ] Better model evaluation metrics and plots
  - [ ] Save metrics to DVC

### Model

- [ ] Detection yaml job on whole dataset
- [ ] Label more data

## About

The goal of this project is to detect solar panels in aerial images (SwissImage 0.1m) using deep learning.

MLOps practices are used to manage the project, including DVC for data versioning, Label Studio for data labeling, and BentoML for model serving. This allows for reproducibility, collaboration and scalability of the project.

## Installation

If you don't have Conda installed, you can run the following script to install it:

```bash
./scripts/install_conda.sh
```

Next, restart your terminal and create a new conda environment and install the dependencies:

```bash
conda create --name swissimage-vision python=3.12 pip
conda activate swissimage-vision
```

```bash
pip install -r requirements.txt
```

### Setup DVC

Add the MinIO credentials to the DVC configuration:

```bash
dvc remove modify --local minio access_key_id <ACCESS_KEY_ID>
dvc remove modify --local minio secret_access_key <SECRET_ACCESS_KEY>
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
tensorboard --logdir lightning_logs
```

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

## Integrations

### DVC

Read more about DVC integration at [docs/dvc.md](docs/dvc.md)

### Label Studio

Read more about LabelStudio integration at [docs/labelstudio.md](docs/labelstudio.md)

## Resources

### Data

- http://2019.geopython.net/data/solar.zip
- https://ep2020.europython.eu/media/conference/slides/detecting-and-analyzing-solar-panels-switzerland-using-aerial-imagery.pdf
- https://zenodo.org/records/5171712

### Projects

- https://github.com/shashankag14/fine-tune-object-detector/blob/main/fine_tune.ipynb
- https://github.com/swiss-ai-center/giscup2023-resnet-unet
- https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- https://github.com/saizk/Deep-Learning-for-Solar-Panel-Recognition?tab=readme-ov-file
