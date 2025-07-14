# UT_2025_T4

# Instructions:

## Installation
Install the following dependencies:
```
Anaconda          # Create inference and training environments on Ubuntu
```
Install additional dependencies:
```
pip install -r requirements.txt
```
## Data preparation
1. Download the [Fisheye8K](https://scidm.nchc.org.tw/en/dataset/fisheye8k/resource/f6e7500d-1d6d-48ea-9d38-c4001a17170e/nchcproxy) dataset, and put the data into `datasets/fisheye8k/`.

2. Download the [VisDrone](https://github.com/VisDrone/VisDrone-Dataset?tab=readme-ov-file) dataset, and put the data into `datasets/visdrone/`.

3. Download the FisheyeEval1k test dataset, and put the data into `datasets/fisheye_test/`. For convenience, all test images should be put into one folder named `images`. The `datasets/` directory will look like below:

```
- datasets
    - 8k_random
        - test
        - val
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
python dataprocessing/visdrone2yolo.py --data_path datasets/visdrone/VisDrone2019-DET-train
```
```
python dataprocessing/visdrone2yolo.py --data_path datasets/visdrone/VisDrone2019-DET-test-dev
```
```
python dataprocessing/visdrone2yolo.py --data_path datasets/visdrone/VisDrone2019-DET-val
```
5. Use the test labels under data/aicity/aicity_2024_fisheye8k/test from the 3rd place in 2024, which are publicly available in [Google Drive](https://drive.google.com/file/d/1pj1hWajt2Zh_A7cIQBPyvQg7weXwhaiA/view?usp=sharing)/data.zip
6. Redistribute the Fisheye8K dataset by randomly splitting it into 70:30.
```
mkdir 8k_random
cp datasets/fisheye8k/train/images/* datasets/8k_random/images/
cp datasets/fisheye8k/test/images/* datasets/8k_random/images/
cp datasets/fisheye8k/train/labels/* datasets/8k_random/labels/
cp datasets/fisheye8k/test/labels/* datasets/8k_random/labels/
python dataprocessing/8k_random_split.py
```
7. The download path of our datasets on Google Drive is : [8k_random](https://drive.google.com/file/d/1LruOMKFEB3Svo_EgxLhhN4JUxQGF2mns/view?usp=drive_link) [visdrone](https://drive.google.com/file/d/1kM9rhdhpl7zz1WGRxp9NZM-PoF5nuHh2/view?usp=drive_link) [fisheye_test](https://drive.google.com/file/d/1ekkW2xQgS77NrrFFkn4Ll2AwKj47G-N9/view?usp=drive_link) [fisheye8k](https://drive.google.com/file/d/1ZFtHvsEHxQQ46x6u_BE6rh3QFmUYbZ1X/view?usp=drive_link) 
   
## Models Training

### YOLOR-D6
Follow these instructions to train the YOLOR-D6:
1. Create the conda environment
```
conda create -n yolor python=3.8
conda activate yolor_2025
```

2. Download the COCO-pretrained YOLOr-D6 model released by the authors and put the checkpoint in `train/YoloR/`. Pretrained link: [yolor-d6.pt](https://github.com/WongKinYiu/yolor/releases/download/weights/yolor-d6.pt)

3. Install the dependencies
```
pip install -r requirements.txt
```

4. Train the YOLOr-D6 model on the VisDrone+Fisheye8k dataset using the following command

```
# Move to the YOLO-R directory
cd train/YoloR

# Train the yolor-d6 model for 250 epochs
python train.py --batch-size 2 --img 1920 1920 --data ../../datasets/visdrone_fisheye8k.yaml --cfg models/yolor-d6-SPP.yaml --weights yolor-d6.pt --device 0 --name yolor_d6 --hyp hyp.scratch.1280.yaml --epochs 250
```
The checkpoints will be saved in `train/YoloR/runs/train/`



