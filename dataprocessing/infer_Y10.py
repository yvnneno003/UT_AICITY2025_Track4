import os
import cv2
import numpy as np
import time
import json
import torch
import argparse
from ultralytics import YOLO
from utils.datasets import letterbox

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv10 inference (single model)")
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--yolov10_model', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=1280)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--yolov10_conf', type=float, default=0.7)
    parser.add_argument('--yolov10_iou', type=float, default=0.45)
    return parser.parse_args()

def get_image_id(img_name):
    img_name = img_name.split('.')[0]
    scene_list = ['M', 'A', 'E', 'N']
    camera_indx = int(img_name.split('_')[0].replace('camera', ''))
    scene_indx = scene_list.index(img_name.split('_')[1])
    frame_indx = int(img_name.split('_')[2])
    return int(f"{camera_indx}{scene_indx}{frame_indx}")

def preprocess_image(image, img_size):
    img, ratio, pad = letterbox(image, new_shape=img_size, auto=False)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img, ratio, pad

def scale_coords(coords, img0_shape, ratio, pad):
    coords = np.array(coords)
    coords[:, [0, 2]] = (coords[:, [0, 2]] - pad[0]) / ratio[0]
    coords[:, [1, 3]] = (coords[:, [1, 3]] - pad[1]) / ratio[1]
    coords[:, [0, 2]] = np.clip(coords[:, [0, 2]], 0, img0_shape[1])
    coords[:, [1, 3]] = np.clip(coords[:, [1, 3]], 0, img0_shape[0])
    return coords.round().tolist()

def postprocess_yolov10(results, img_shape, img_size, conf_thres, ratio, pad):
    boxes, scores, classes = [], [], []
    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            if conf >= conf_thres:
                xyxy = box.cpu().numpy()
                boxes.append(xyxy)
                scores.append(float(conf.cpu()))
                classes.append(int(cls.cpu()))
    if boxes:
        boxes = scale_coords(boxes, img_shape, ratio, pad)
    return boxes, scores, classes

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    model = YOLO(args.yolov10_model)
    submission = []
    sum_time = 0
    max_fps = 25

    # Gather images
    image_files = []
    for root, _, files in os.walk(args.image_folder):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                image_files.append(os.path.join(root, file))

    image_files.sort()

    print(f"Found {len(image_files)} images")

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue
        img_shape = img.shape[:2]
        image_id = get_image_id(os.path.basename(img_path))

        # Preprocess
        img_processed, ratio, pad = preprocess_image(img, args.img_size)
        img_tensor = torch.from_numpy(img_processed).unsqueeze(0).to(device)

        # Timing starts
        t0 = time.time()
        results = model.predict(
            img_tensor, imgsz=args.img_size, conf=args.yolov10_conf,
            iou=args.yolov10_iou, verbose=False
        )
        t1 = time.time()

        boxes, scores, classes = postprocess_yolov10(
            results, img_shape, args.img_size, args.yolov10_conf, ratio, pad
        )

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            submission.append({
                "image_id": image_id,
                "category_id": int(cls),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score)
            })

        sum_time += (t1 - t0) * 1000  # ms

    if sum_time > 0:
        fps = 1000 * len(image_files) / sum_time
        print(f"FPS: {fps:.2f}")
    else:
        print("No images processed; cannot compute FPS.")

    norm_fps = min(fps, max_fps) / max_fps

    print(f"Processed {len(image_files)} images in {sum_time/1000:.2f} sec")
    print(f"FPS: {fps:.2f}")
    print(f"Normalized FPS: {norm_fps:.4f}")

    out_path = os.path.join(args.output_dir, f"infer_Y10_yolov10x_vfp_train4_1280_best_{int(args.yolov10_conf*100)}_{int(args.yolov10_iou*100)}.json")

    with open(out_path, 'w') as f:
        json.dump(submission, f, indent=2)
    print(f"Saved submission to {out_path}")

if __name__ == "__main__":
    main()

