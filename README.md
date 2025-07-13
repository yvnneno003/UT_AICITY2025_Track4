# UT_2025_T4

# Instructions:

## Installation
Install the following dependencies:
```
Anaconda          # Create inference and training environments on Ubuntu
```
## Data preparation
1. Download the Fisheye8K dataset, and put the data into `datasets/fisheye8k/`. Link to the fisheye8k dataset: [link](https://scidm.nchc.org.tw/en/dataset/fisheye8k/resource/f6e7500d-1d6d-48ea-9d38-c4001a17170e/nchcproxy)

2. Download the VisDrone dataset, and put the data into `datasets/visdrone/`. Link to the VisDrone dataset: [link](https://github.com/VisDrone/VisDrone-Dataset?tab=readme-ov-file)

3. Download the FisheyeEval1k test dataset, and put the data into `datasets/fisheye_test/`. For convenience, all test images should be put into one folder named `images`. The `datasets/` directory will look like below:

```
- datasets
    - fisheye8k
        - test
        - train
    - visdrone
        - VisDrone2019-DET-train
        - VisDrone2019-DET-val
        - VisDrone2019-DET-test_dev
    - fisheye_test
        - images
    - json_labels
```

4. Convert the VisDrone dataset to YOLO format using the following command. Note that when converting the VisDrone dataset, we also map each category to their corresponding one in the Fisheye8k dataset, other categories are ignored. The labels will be saved in the "labels" directory under the corresponding sub-dataset

```
python dataprocessing/format_conversion/visdrone2yolo.py --data_path datasets/visdrone/VisDrone2019-DET-train
```
## Models Training

### YOLOR-D6
Follow these instructions to train the YOLOR-W6:
1. Create the conda environment
```
conda create -n yolor python=3.8
conda activate yolor_2025
```

2. Download the COCO-pretrained YOLOr-D6 model released by the authors and put the checkpoint in `./train/YoloR/`. Pretrained link: [yolor-d6.pt](https://github.com/WongKinYiu/yolor/releases/download/weights/yolor-d6.pt)

3. Install the dependencies
```
pip install -r requirements.txt
```

4. Train the YOLOr-D6 model on the VisDrone+Fisheye8k dataset using the following command

```
# Move to the YOLO-R directory
cd ./train/YoloR

# Train the yolor-d6 model for 250 epochs
python train.py --batch-size 8 --img 1280 1280 --data ../../dataset/visdrone_fisheye8k.yaml --cfg models/yolor-d6.yaml --weights './yolor-d6.pt' --device 0 --name yolor_w6 --hyp hyp.scratch.1280.yaml --epochs 250
```
The checkpoints will be saved in `./train/YoloR/runs/train/`



