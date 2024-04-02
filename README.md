# SwissImage Vision

- [Installation](#installation)
  - [Setup DVC](#setup-dvc)
- [Demo](#demo)
- [LabelStudio](#labelstudio)
- [Resources](#resources)
  - [Data](#data)
  - [Projects](#projects)

![Demo](media/demo.png)

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

## Demo

To run the demo locally, execute the following command:

```bash
python src/demo.py
```

## LabelStudio

```bash
conda install conda-forge::psycopg2-binary
```

```bash
pip install label-studio>=0.11.0,<=0.12
```

```bash
python3 ./scripts/serve_data.py
```

```bash
label-studio start ./label-studio/config.xml
```

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
