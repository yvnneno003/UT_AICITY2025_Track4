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

2. Download the COCO-[pretrained YOLOr-D6 model](https://github.com/WongKinYiu/yolor/releases/download/weights/yolor-d6.pt) released by the authors and put the checkpoint in `train/YoloR/`.

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
### YOLOV10
Follow these instructions to train the YOLOV10-X:
1. Create the conda environment
```
conda create -n yolor python=3.9
conda activate yolov10_2025
```

2. Download the COCO-[pretrained YOLOV10-X model](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt) released by the authors and put the checkpoint in `train/YoloV10/`.

3. Install the dependencies
```
pip install -r requirements.txt
pip install -e .
```

4. Train the YOLOV10 model on the VisDrone+Fisheye8k+Pseudo Labels datasets using the following command

```
# Move to the YOLOV10 directory
cd train/YoloV10

# Train the yolov10-x model for 150 epochs
yolo detect train data=../../datasets/visdrone_fisheye8k_pseudo.yaml model=datasets/FishEye8K/yolov10x-FishEye8K.yaml epochs=150 batch=2 imgsz=1280 device=0 pretrained=yolov10x.pt save_period=25
```

### YOLOV13
Follow these instructions to train the YOLOV13-l:
1. Create the conda environment
```
conda create -n yolor python=3.11
conda activate yolov13_2025
```

2. Download the COCO-[pretrained YOLOV13-l model](https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13l.pt) released by the authors and put the checkpoint in `train/YoloV13/`.

3. Install the dependencies
```
pip install -r requirements.txt
pip install -e .
```

4. Train the YOLOV13 model on the Fisheye8k+Pseudo Labels datasets using the following command

```
# Move to the YOLOV13 directory
cd train/YoloV13

# Train the yolov13-l model for 150 epochs
yolo task=detect mode=train imgsz=1280 batch=2 epochs=150 data=../../datasets/fisheye8k_pseudo.yaml   model=/model/yolov13l.pt hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 flipud=0.0 fliplr=0.5 scale=0.5 mosaic=1.0 mixup=0.1 copy_paste=0.15
```

## Models Inferencing

### Checkpoints
For quick reproduction, download checkpoints and put them in the `checkpoints` : 
- [YoloR](https://drive.google.com/file/d/1qPFxhDH1kOoKYOlHDuZuQ8IE3ZrmAKh5/view?usp=drive_link)
- [YoloV10](https://drive.google.com/file/d/17ByreGsuBy_HJJTRLlziTkyoWQQoV8uK/view?usp=drive_link)
- [YoloV13](https://drive.google.com/file/d/1bpaZ56eBZSmwPzbUOnzkMROzFK5sxkiZ/view?usp=drive_link)

### YOLOR-D6
For inferencing, follow these instructions
1. Move to the YOLOR-D6 directory and activate the yolor conda environment created in the training phase. If you haven't, see the **Training** section for instructions.
```
cd train/YoloR

# Activate the yolor environment
conda activate yolor_2025
```

2. Infer using the yolor model :
```
python detect.py --source ../../datasets/fisheye_test/images --weights ../../checkpoints/yolor_d6_best_checkpoint.pt --conf 0.5 --iou 0.55 --img-size 1280 --device 0 --save-txt --save-conf
```

3. Convert to submission format. Remember to modify the path to the corresponding labels_dir
```
python ../../dataprocessing/yolo2coco.py --images_dir ../../datasets/fisheye_test/images --labels_dir runs/detect/exp/labels --output yolor_d6.json --conf 1 --submission 1 --is_fisheye8k 1
```
