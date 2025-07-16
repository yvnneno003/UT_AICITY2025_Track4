import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.meshgrid.*indexing argument.*")

import os
import cv2
import json
import time
import torch
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from models.experimental import attempt_load as load_yolor
from ultralytics import YOLO
from utils.general import non_max_suppression
from utils.datasets import letterbox
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOR + YOLOv10 + YOLOv13 + WBF")
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    #parser.add_argument('--cbam_model', type=str, required=True)
    #parser.add_argument('--siam_model', type=str, required=True)
    #parser.add_argument('--y11_model', type=str, required=True)
    
    parser.add_argument('--yolor_model', type=str, required=True)
    parser.add_argument('--y10_model', type=str, required=True)
    parser.add_argument('--y13_model', type=str, required=True)
    
    parser.add_argument('--img_size', type=int, default=1280)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--wbf_iou', type=float, default=0.65)
    parser.add_argument('--wbf_weights', type=str, default='9,9,9')
    parser.add_argument('--post_conf', type=float, default=0.15)
    # Dynamic conf/iou thresholds
    #parser.add_argument('--cbam_conf', type=float, default=0.8)
    #parser.add_argument('--cbam_iou', type=float, default=0.45)
    #parser.add_argument('--siam_conf', type=float, default=0.8)
    #parser.add_argument('--siam_iou', type=float, default=0.5)
    #parser.add_argument('--y11_conf', type=float, default=0.45)
    #parser.add_argument('--y11_iou', type=float, default=0.525)
    parser.add_argument('--yolor_conf', type=float, default=0.5)
    parser.add_argument('--yolor_iou', type=float, default=0.55)
    parser.add_argument('--y10_conf', type=float, default=0.5)
    parser.add_argument('--y10_iou', type=float, default=0.65)
    parser.add_argument('--y13_conf', type=float, default=0.495)
    parser.add_argument('--y13_iou', type=float, default=0.45)

    return parser.parse_args()

def get_image_id(img_name):
    img_name = img_name.split('.')[0]
    parts = img_name.split('_')
    cam = int(parts[0].replace('camera', ''))
    scene_idx = {'M':0,'A':1,'E':2,'N':3}.get(parts[1], 0)
    frame = int(parts[2])
    return int(f"{cam}{scene_idx}{frame}")

def preprocess_image(image, img_size):
    img, ratio, pad = letterbox(image, new_shape=img_size, auto=False)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img, ratio, pad

def scale_coords(coords, img0_shape, ratio, pad):
    coords = np.array(coords)
    coords[:, [0,2]] = (coords[:, [0,2]] - pad[0]) / ratio[0]
    coords[:, [1,3]] = (coords[:, [1,3]] - pad[1]) / ratio[1]
    coords[:, [0,2]] = np.clip(coords[:, [0,2]], 0, img0_shape[1])
    coords[:, [1,3]] = np.clip(coords[:, [1,3]], 0, img0_shape[0])
    return coords.tolist()

def postprocess_yolor(pred, img_shape, img_size, conf_thres, iou_thres, device):
    boxes, scores, classes = [], [], []
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)
    for det in pred:
        if det is not None and len(det):
            scaled = scale_coords(det[:, :4].cpu().numpy(), img_shape, (img_size,img_size), (0,0))
            for xyxy, conf, cls in zip(scaled, det[:,4], det[:,5]):
                boxes.append(xyxy)
                scores.append(float(conf.cpu()))
                classes.append(int(cls.cpu()))
    return boxes, scores, classes

def postprocess_y11(results, img_shape, ratio, pad, conf_thres):
    boxes, scores, classes = [], [], []
    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            if conf >= conf_thres:
                xyxy = box.cpu().numpy().tolist()
                boxes.append(xyxy)
                scores.append(float(conf.cpu()))
                classes.append(int(cls.cpu()))
    if boxes:
        boxes = scale_coords(np.array(boxes), img_shape, ratio, pad)
    return boxes, scores, classes
    
def postprocess_yolo(results, img_shape, ratio, pad, conf_thres):
    boxes, scores, classes = [], [], []
    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            if conf >= conf_thres:
                xyxy = box.cpu().numpy().tolist()
                boxes.append(xyxy)
                scores.append(float(conf.cpu()))
                classes.append(int(cls.cpu()))
    if boxes:
        boxes = scale_coords(np.array(boxes), img_shape, ratio, pad)
    return boxes, scores, classes	

def wbf_ensemble(boxes_list, scores_list, labels_list, img_shape, weights, iou_thr):
    norm_boxes = [
        [[b[0]/img_shape[1], b[1]/img_shape[0], b[2]/img_shape[1], b[3]/img_shape[0]] for b in boxes]
        for boxes in boxes_list
    ]
    boxes, scores, labels = weighted_boxes_fusion(norm_boxes, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
    boxes = (boxes * [img_shape[1], img_shape[0], img_shape[1], img_shape[0]]).tolist()
    return boxes, scores, labels

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load models
    #cbam_model = load_yolor(args.cbam_model).to(device).half().eval()
    #siam_model = load_yolor(args.siam_model).to(device).half().eval()
    #y11_model = YOLO(args.y11_model)
    yolor_model = load_yolor(args.yolor_model).to(device).half().eval()
    y10_model = YOLO(args.y10_model)
    y13_model = YOLO(args.y13_model)

    image_files = []
    for root, _, files in os.walk(args.image_folder):
        for file in files:
            if file.endswith(('.jpg','.png')):
                image_files.append(os.path.join(root,file))
    image_files.sort()
    print(f"Found {len(image_files)} images")

    weights = [float(w) for w in args.wbf_weights.split(',')]
    sum_time = 0
    submission = []
    max_fps = 25

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue
        img_shape = img.shape[:2]
        image_id = get_image_id(os.path.basename(img_path))
        img_processed, ratio, pad = preprocess_image(img, args.img_size)
        img_tensor = torch.from_numpy(img_processed).unsqueeze(0).to(device).half()

        t0 = time.time()

        # YOLOv7 CBAM
        #with torch.no_grad():
        #    pred_cbam = cbam_model(img_tensor)[0]
        #cbam_boxes, cbam_scores, cbam_classes = postprocess_yolor(pred_cbam, img_shape, args.img_size, args.cbam_conf, args.cbam_iou, device)

        # YOLOv7 SIAM
        #with torch.no_grad():
        #    pred_siam = siam_model(img_tensor)[0]
        #siam_boxes, siam_scores, siam_classes = postprocess_yolor(pred_siam, img_shape, args.img_size, args.siam_conf, args.siam_iou, device)

    
        with torch.no_grad():
            pred_yolor = yolor_model(img_tensor)[0]
        yolor_boxes, yolor_scores, yolor_classes = postprocess_yolor(pred_yolor, img_shape, args.img_size, args.yolor_conf, args.yolor_iou, device)
    
        # YOLOv11
        #with torch.no_grad():
        #    results = y11_model.predict(img_tensor, imgsz=args.img_size, conf=args.y11_conf, iou=args.y11_iou, verbose=False)
        #y11_boxes, y11_scores, y11_classes = postprocess_y11(results, img_shape, ratio, pad, args.y11_conf)

        with torch.no_grad():
            results = y10_model.predict(img_tensor, imgsz=args.img_size, conf=args.y10_conf, iou=args.y10_iou, verbose=False,half=True)
        y10_boxes, y10_scores, y10_classes = postprocess_yolo(results, img_shape, ratio, pad, args.y10_conf)

        with torch.no_grad():
            results = y13_model.predict(img_tensor, imgsz=args.img_size, conf=args.y13_conf, iou=args.y13_iou, verbose=False,half=True)
        y13_boxes, y13_scores, y13_classes = postprocess_yolo(results, img_shape, ratio, pad, args.y13_conf)
        
        
        
        # WBF fusion
        boxes, scores, labels = wbf_ensemble(
            [yolor_boxes, y10_boxes, y13_boxes],
            [yolor_scores, y10_scores, y13_scores],
            [yolor_classes, y10_classes, y13_classes],
            img_shape, weights, args.wbf_iou
        )

        # Apply post_conf filter
        for box, score, cls in zip(boxes, scores, labels):
            if score >= args.post_conf:
                x1,y1,x2,y2 = box
                submission.append({
                    "image_id": image_id,
                    "category_id": int(cls),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score
                })

        t1 = time.time()
        sum_time += (t1 - t0) * 1000

    fps = 1000 * len(image_files) / sum_time if sum_time > 0 else 0
    norm_fps = min(fps, max_fps) / max_fps
    print(f"Processed {len(image_files)} images in {sum_time/1000:.2f} sec")
    print(f"FPS: {fps:.2f}")
    print(f"Normalized FPS: {norm_fps:.4f}")
    print("Submit JSON to evaluate F1 and harmonic mean")

    out_path = os.path.join(
        args.output_dir,
        f"submission_YRY10Y13_WBF_9_9_9_YR{int(args.yolor_conf*100)}-{int(args.yolor_iou*100)}"
        f"_Y10{int(args.y10_conf*100)}-{int(args.y10_iou*100)}"
        f"_Y13{int(args.y13_conf*100)}-{int(args.y13_iou*100)}.json"
    )
    with open(out_path,'w') as f:
        json.dump(submission, f, indent=2)
    print(f"Saved submission to {out_path}")

if __name__ == "__main__":
    main()

