#!/usr/bin/env python3

import argparse
import os
import sys
import cv2
import numpy as np
import onnxruntime as ort
from trt_config import LABELS, efficientdet_model_dict
import time  

def preprocess_image(image_path, input_size):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image at {image_path}")
    orig_size = image.shape[:2]  # (height, width)
    resized = cv2.resize(image, input_size)
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(rgb_image.astype(np.uint8), axis=0)
    return input_tensor, orig_size, image

def postprocess(outputs, orig_size, score_thresh, top_k, input_size=320):
    boxes, class_ids, scores = outputs[0][0], outputs[1][0], outputs[2][0]
    height, width = orig_size
    scale_w = width / input_size
    scale_h = height / input_size

    bboxes = boxes * input_size
    bboxes = np.clip(bboxes, 0, input_size)

    results = []
    for idx in range(len(scores)):
        if scores[idx] < score_thresh or len(results) >= top_k:
            continue
        class_id = int(class_ids[idx])
        label = LABELS[class_id] if 0 <= class_id < len(LABELS) else 'Unknown'
        ymin, xmin, ymax, xmax = bboxes[idx]
        
        left = int(xmin * scale_w)
        top = int(ymin * scale_h)
        right = int(xmax * scale_w)
        bottom = int(ymax * scale_h)
        results.append({'box': (left, top, right, bottom), 'label': label, 'score': scores[idx]})
    return results

def draw_results(image, detections):
    for det in detections:
        left, top, right, bottom = det['box']
        caption = f"{det['label']}: {det['score']:.2f}"
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, caption, (left, max(top - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)
    return image

def parse_arguments():
    parser = argparse.ArgumentParser(description="Image Inference using ONNX Runtime")
    parser.add_argument('-i', '--image', required=True, help='Path to the input image')
    parser.add_argument('-o', '--output', default='onnx_output.jpg', help='Path to save the output image')
    parser.add_argument('--model_type', choices=list(efficientdet_model_dict.keys()), default='efficientdet_lite0',
                        help='Model type to use')
    parser.add_argument('--score_threshold', type=float, default=0.5, help='Score threshold for filtering detections')
    parser.add_argument('--top_k', type=int, default=5, help='Maximum number of detections to display')
    return parser.parse_args()

def main():
    args = parse_arguments()

    model_info = efficientdet_model_dict.get(args.model_type)
    if model_info is None:
        print(f"Model type '{args.model_type}' not found in efficientdet_model_dict.")
        sys.exit(1)

    model_path = model_info['MODEL_PATH']
    input_size = (model_info['TARGET_SIZE'], model_info['TARGET_SIZE'])

    if not os.path.isfile(model_path):
        print(f"Model file '{model_path}' not found.")
        sys.exit(1)
    if not os.path.isfile(args.image):
        print(f"Image file '{args.image}' not found.")
        sys.exit(1)

    try:
        ort_session = ort.InferenceSession(model_path)
        input_name = ort_session.get_inputs()[0].name

        input_tensor, orig_size, orig_image = preprocess_image(args.image, input_size)
        
        start_time = time.perf_counter()
        outputs = ort_session.run(None, {input_name: input_tensor})
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"Inference latency: {latency:.2f} ms")

        detections = postprocess(outputs, orig_size, args.score_threshold, args.top_k, input_size=input_size[0])
        result_image = draw_results(orig_image.copy(), detections)
        cv2.imwrite(args.output, result_image)
        print(f"Output saved to {args.output}")
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
