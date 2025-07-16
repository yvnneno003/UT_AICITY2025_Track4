import os
import time
import json
import argparse
import torch
import cv2
from ultralytics import YOLO

CATEGORY_MAP = {
    0: "Bus",
    1: "Bike",
    2: "Car",
    3: "Pedestrian",
    4: "Truck"
}

CATEGORY_ID_MAP = {
    "Bus": 0,
    "Bike": 1,
    "Car": 2,
    "Pedestrian": 3,
    "Truck": 4
}

SCENE_LIST = ['M', 'A', 'E', 'N']

def get_image_id(img_name):
    img_name = os.path.basename(img_name).split('.png')[0]
    cam_idx = int(img_name.split('_')[0].replace('camera', ''))
    scene_idx = SCENE_LIST.index(img_name.split('_')[1])
    frame_idx = int(img_name.split('_')[2])
    return int(f"{cam_idx}{scene_idx}{frame_idx}")

def main(args):
    model = YOLO(args.model)
    image_paths = sorted([
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(args.image_dir)
        for f in filenames if f.endswith(".png")
    ])

    results_json = []
    total_process_time = 0.0

    for img_path in image_paths:
        img = cv2.imread(img_path)

        start_time = time.time()
        pred = model.predict(
            source=img,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False #,
            #half=True  # Enable FP16 inference during prediction
        )[0]
        end_time = time.time()
        total_process_time += end_time - start_time

        boxes = pred.boxes.cpu().numpy()
        for box, score, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
            x1, y1, x2, y2 = box
            result = {
                "image_id": get_image_id(img_path),
                "category_id": int(cls),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score)
            }
            results_json.append(result)

    fps = len(image_paths) / total_process_time
    norm_fps = min(fps, 25.0) / 25.0
    print(f"Processed {len(image_paths)} images in {total_process_time:.2f}s")
    print(f"FPS: {fps:.2f}, Normalized FPS: {norm_fps:.4f}")

    #os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results_json, f)
    print(f"Saved submission JSON to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='YOLOv13 PyTorch .pt model path')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to evaluation image folder')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for NMS')
    parser.add_argument('--device', type=int, default=0, help='CUDA device to use')
    args = parser.parse_args()
    main(args)
