# UT_2025_T4

# Instructions:

## Installation
Install the following dependencies:
```
Anaconda          # Create inference and training environments on Ubuntu
```
## Data preparation
1. Download the Fisheye8K dataset, and put the data into `./dataset/fisheye8k/`. Link to the fisheye8k dataset: [link](https://scidm.nchc.org.tw/en/dataset/fisheye8k/resource/f6e7500d-1d6d-48ea-9d38-c4001a17170e/nchcproxy)

2. Download the VisDrone dataset, and put the data into `./dataset/visdrone/`. Link to the VisDrone dataset: [link](https://github.com/VisDrone/VisDrone-Dataset?tab=readme-ov-file)

3. Download the FisheyeEval1k test dataset, and put the data into `./dataset/fisheye_test/`. For convenience, all test images should be put into one folder named `images`. The `./dataset/` directory will look like below:

```
- dataset
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
python ./dataprocessing/format_conversion/visdrone2yolo.py --data_path ./dataset/visdrone/VisDrone2019-DET-train
```

5. Transform the VisDrone images to fisheye images. The images are split into two squared images, and each of them is converted individually. You can use the code to visualize the labels after conversion, the results will look like below:

```
# Fisheye Conversion
python ./dataprocessing/ifish_augmentation/convert_visdrone.py --src_path ./dataset/visdrone --trg_path ./dataset/synthetic_visdrone --distortion 0.5 --data_type VisDrone2019-DET-train --crop True

# Visualize 4 images for checking
python ./dataprocessing/visualization/visualization.py --image_dir ./dataset/synthetic_visdrone/VisDrone2019-DET-train/images --label_dir ./dataset/synthetic_visdrone/VisDrone2019-DET-train/labels --num_imgs 4 --save_dir ./dataprocessing/visualization
```

6. Convert the Visdrone and Synthetic VisDrone datasets to MS-COCO annotation format. Note that for the VisDrone dataset, the old train.json file will be replaced by the new one. If you want to keep the old file, modify the --output argument in the command.

```
# Convert VisDrone dataset
python ./dataprocessing/format_conversion/yolo2coco.py --images_dir ./dataset/visdrone/VisDrone2019-DET-train/images --labels_dir ./dataset/visdrone/VisDrone2019-DET-train/labels --output ./dataset/visdrone/train.json


