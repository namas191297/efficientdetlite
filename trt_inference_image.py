#!/usr/bin/env python3

import argparse
import cv2
import sys
import os
import time  # For latency measurement

from trt_executor import TRTExecutor
from trt_config import efficientdet_model_dict

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run inference on an image using TensorRT')
    parser.add_argument('--image', '-i', required=True, help='Path to the input image')
    parser.add_argument('--output', '-o', default='trt_output.jpg', help='Path to save the output image with detections')
    parser.add_argument('--model_type', '-m', default='efficientdet_lite0',
                        choices=list(efficientdet_model_dict.keys()),
                        help='Model type to use for inference')
    parser.add_argument('--fp16', '-f', action='store_true', help='Enable FP16 precision')
    parser.add_argument('--score_threshold', '-t', type=float, default=0.4,
                        help='Score threshold for detections')
    parser.add_argument('--top_k', '-k', type=int, default=5,
                        help='Maximum number of detections to display')
    return parser.parse_args()

def draw_detections(image, detections, score_threshold, top_k):
    """
    Draws bounding boxes and labels on the image.

    Args:
        image (np.ndarray): Original image.
        detections (list): List of detection dictionaries.
        score_threshold (float): Minimum score threshold for displaying detections.
        top_k (int): Maximum number of detections to display.

    Returns:
        np.ndarray: Image with drawn detections.
    """
    # Sort detections by score in descending order
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    # Limit to top_k detections
    detections = detections[:top_k]
    
    for det in detections:
        if det['score'] < score_threshold:
            continue
        left, top, right, bottom = det['box']
        label = det['label']
        score = det['score']
        # Draw rectangle
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        # Put label
        cv2.putText(image, f"{label}: {score:.2f}", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def main():
    args = parse_arguments()

    # Verify input image exists
    if not os.path.isfile(args.image):
        print(f"Error: Image file '{args.image}' does not exist.")
        sys.exit(1)

    # Read the input image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Unable to read image '{args.image}'.")
        sys.exit(1)

    try:
        # Initialize the TRTExecutor
        executor = TRTExecutor(MODEL_TYPE=args.model_type, fp16=args.fp16)
        print(f"Initialized TRTExecutor with model type '{args.model_type}' and FP16={'enabled' if args.fp16 else 'disabled'}.")

        # Measure inference latency
        start_time = time.perf_counter()
        detections = executor.inference(image)
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds

        print(f"Inference completed in {latency:.2f} ms.")

        # Draw detections on the image
        output_image = draw_detections(image.copy(), detections, args.score_threshold, args.top_k)

        # Save the output image
        cv2.imwrite(args.output, output_image)
        print(f"Output image with detections saved to '{args.output}'.")

    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
