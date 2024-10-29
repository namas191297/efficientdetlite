
import argparse
import cv2
import sys
import os
import time  # Import time module for latency and FPS measurement

from trt_executor import TRTExecutor
from trt_config import efficientdet_model_dict

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run inference on webcam feed using TensorRT')
    parser.add_argument('--model_type', '-m', default='efficientdet_lite0', choices=list(efficientdet_model_dict.keys()),
                        help='Model type to use for inference')
    parser.add_argument('--fp16', action='store_true', help='Use fp16 precision')
    parser.add_argument('--score_threshold', '-t', type=float, default=0.5,
                        help='Score threshold for detections')
    parser.add_argument('--top_k', '-k', type=int, default=10,
                        help='Maximum number of detections to display')
    parser.add_argument('--camera_id', '-c', type=int, default=0,
                        help='Camera device ID')
    return parser.parse_args()

def main():
    args = parse_arguments()

    try:
        # Initialize the TRTExecutor
        executor = TRTExecutor(MODEL_TYPE=args.model_type, fp16=args.fp16)
        # Open webcam
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            print(f"Error: Unable to open camera with ID {args.camera_id}")
            sys.exit(1)

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame from camera.")
                break

            # Measure total frame processing time
            frame_start_time = time.perf_counter()

            # Run inference
            inference_start = time.perf_counter()
            detections = executor.inference(frame)
            inference_end = time.perf_counter()
            latency = (inference_end - inference_start) * 1000  # Convert to milliseconds

            # Apply score threshold
            filtered_detections = [det for det in detections if det['score'] >= args.score_threshold]

            # Sort detections by score in descending order
            sorted_detections = sorted(filtered_detections, key=lambda x: x['score'], reverse=True)

            # Select top_k detections
            top_detections = sorted_detections[:args.top_k]

            # Draw detections on the frame
            for det in top_detections:
                left, top, right, bottom = det['box']
                label = det['label']
                score = det['score']
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {score:.2f}", (left, top-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0

            # Display latency and FPS on the frame
            cv2.putText(frame, f"Latency: {latency:.2f} ms", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show the frame
            cv2.imshow('Webcam Inference', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error during inference: {e}")
    finally:
        # Release resources
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        sys.exit(1)

if __name__ == '__main__':
    main()
