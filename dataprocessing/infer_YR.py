import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.meshgrid.*indexing argument.*")

import os
import cv2
import numpy as np
import time
import json
import torch
import argparse
from models.experimental import attempt_load as load_yolor
from utils.datasets import letterbox
from utils.general import non_max_suppression

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOvR inference (single model)")
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--yolor_model', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=1280)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--yolor_conf', type=float, default=0.35)
    parser.add_argument('--yolor_iou', type=float, default=0.475)
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
    coords[:, [0, 2]] = (coords[:, [0, 2]] - pad[0]) / ratio[0]
    coords[:, [1, 3]] = (coords[:, [1, 3]] - pad[1]) / ratio[1]
    coords[:, [0, 2]] = np.clip(coords[:, [0, 2]], 0, img0_shape[1])
    coords[:, [1, 3]] = np.clip(coords[:, [1, 3]], 0, img0_shape[0])
    return coords.round()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    model = load_yolor(args.yolor_model).to(device)
    model.eval()
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

        img_processed, ratio, pad = preprocess_image(img, args.img_size)
        img_tensor = torch.from_numpy(img_processed).unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            pred = model(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thres=args.yolor_conf, iou_thres=args.yolor_iou)
        t1 = time.time()

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = torch.tensor(scale_coords(det[:, :4].cpu().numpy(), img_shape, ratio, pad),
                                          device=det.device, dtype=det.dtype)
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    submission.append({
                        "image_id": image_id,
                        "category_id": int(cls.cpu()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(conf.cpu())
                    })

        sum_time += (t1 - t0) * 1000

    if sum_time > 0:
        fps = 1000 * len(image_files) / sum_time
        print(f"FPS: {fps:.2f}")
    else:
        print("No images processed; cannot compute FPS.")

    norm_fps = min(fps, max_fps) / max_fps

    print(f"Processed {len(image_files)} images in {sum_time/1000:.2f} sec")
    print(f"FPS: {fps:.2f}")
    print(f"Normalized FPS: {norm_fps:.4f}")

    out_path = os.path.join(args.output_dir, f"infer_YR_YoloR-d6-e90_{int(args.yolor_conf*100)}_{int(args.yolor_iou*100)}.json")

    with open(out_path, 'w') as f:
        json.dump(submission, f, indent=2)
    print(f"Saved submission to {out_path}")

if __name__ == "__main__":
    main()

