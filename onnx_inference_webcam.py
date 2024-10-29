#!/usr/bin/env python3

import argparse
import sys
import cv2
import numpy as np
import onnxruntime as ort
from trt_config import LABELS, efficientdet_model_dict
import os
import time  # Import time module for latency and FPS measurement

def preprocess_frame(frame, input_size):
    """
    Preprocesses a video frame for inference.

    Args:
        frame (np.ndarray): Original frame from the webcam.
        input_size (tuple): Desired input size for the model (width, height).

    Returns:
        np.ndarray: Preprocessed frame tensor.
    """
    resized = cv2.resize(frame, input_size)
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(rgb_frame.astype(np.uint8), axis=0)
    return input_tensor

def postprocess(outputs, orig_size, score_thresh, top_k, input_size=320):
    """
    Processes the model outputs to extract detections.

    Args:
        outputs (list): Model outputs.
        orig_size (tuple): Original frame size (height, width).
        score_thresh (float): Minimum score threshold for detections.
        top_k (int): Maximum number of detections.
        input_size (int): Size to which the frame was resized.

    Returns:
        list: List of detection dictionaries.
    """
    boxes, class_ids, scores = outputs[0][0], outputs[1][0], outputs[2][0]
    height, width = orig_size
    scale_w = width / input_size
    scale_h = height / input_size

    # Scale boxes from normalized [0,1] to input_size
    bboxes = boxes * input_size

    # Ensure boxes are within [0, input_size]
    bboxes = np.clip(bboxes, 0, input_size)

    results = []
    for idx in range(len(scores)):
        if scores[idx] < score_thresh or len(results) >= top_k:
            continue
        class_id = int(class_ids[idx])
        label = LABELS[class_id] if 0 <= class_id < len(LABELS) else 'Unknown'
        ymin, xmin, ymax, xmax = bboxes[idx]
        # Correct scaling: x coordinates by width, y coordinates by height
        left = int(xmin * scale_w)
        top = int(ymin * scale_h)
        right = int(xmax * scale_w)
        bottom = int(ymax * scale_h)
        results.append({'box': (left, top, right, bottom), 'label': label, 'score': scores[idx]})
    return results

def draw_results(frame, detections):
    """
    Draws bounding boxes and labels on the frame.

    Args:
        frame (np.ndarray): Original frame.
        detections (list): List of detections.

    Returns:
        np.ndarray: Frame with drawn detections.
    """
    for det in detections:
        left, top, right, bottom = det['box']
        caption = f"{det['label']}: {det['score']:.2f}"
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
        cv2.putText(frame, caption, (left, max(top - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 2)
    return frame

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Video Inference using ONNX Runtime")
    parser.add_argument('--model_type', choices=list(efficientdet_model_dict.keys()), default='efficientdet_lite0',
                        help='Model type to use')
    parser.add_argument('--score_threshold', '-t', type=float, default=0.5, help='Score threshold for filtering detections')
    parser.add_argument('--top_k', '-k', type=int, default=5, help='Maximum number of detections to display')
    parser.add_argument('--camera_id', '-c', type=int, default=0, help='Camera device ID to use')
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

    try:
        ort_session = ort.InferenceSession(model_path)
        input_name = ort_session.get_inputs()[0].name

        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            print(f"Cannot open camera with ID {args.camera_id}")
            sys.exit(1)

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break

            frame_size = frame.shape[:2]  # (height, width)
            input_tensor = preprocess_frame(frame, input_size)

            # Measure inference latency
            inference_start = time.perf_counter()
            outputs = ort_session.run(None, {input_name: input_tensor})
            inference_end = time.perf_counter()
            latency = (inference_end - inference_start) * 1000  # Convert to milliseconds

            # Postprocess to get detections
            detections = postprocess(outputs, frame_size, args.score_threshold, args.top_k, input_size=input_size[0])

            # Draw detections on the frame
            result_frame = draw_results(frame, detections)

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0

            # Display latency and FPS on the frame
            cv2.putText(result_frame, f"Latency: {latency:.2f} ms", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show the frame
            cv2.imshow('Webcam Inference', result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        sys.exit(1)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
