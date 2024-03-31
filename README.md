# SwissImage Vision

- [Installation](#installation)
  - [Setup DVC](#setup-dvc)
- [Resources](#resources)
  - [Data](#data)
  - [Tutorials](#tutorials)

## Installation

```bash
# Conda
./scripts/install_conda.sh
```

Next, restart your terminal and create a new conda environment:

```bash
conda create --name swissimage-vision python=3.12 pip
conda activate swissimage-vision
```

```bash
# Conda
pip install -r requirements.txt
```

### Setup DVC

```bash
dvc remove modify --local minio access_key_id <ACCESS_KEY_ID>
dvc remove modify --local minio secret_access_key <SECRET_ACCESS_KEY>
```

```bash
dvc pull
```

## Resources

### Data

http://2019.geopython.net/data/solar.zip
https://ep2020.europython.eu/media/conference/slides/detecting-and-analyzing-solar-panels-switzerland-using-aerial-imagery.pdf

### Tutorials

- https://github.com/shashankag14/fine-tune-object-detector/blob/main/fine_tune.ipynb
- https://github.com/swiss-ai-center/giscup2023-resnet-unet
- https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
