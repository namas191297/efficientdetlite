import argparse
import sys
from trt_engine_builder import TRTEngineBuilder
from trt_config import efficientdet_model_dict

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Build TensorRT engine from an ONNX model.')
    parser.add_argument('--model_type', '-m',
                        choices=list(efficientdet_model_dict.keys()),
                        default='efficientdet_lite0',
                        help='Model type to use for building the engine.')
    parser.add_argument('--fp16', '-f',
                        action='store_true',
                        help='Enable FP16 precision for the TensorRT engine.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Initialize the TensorRT Engine Builder with parsed arguments
    builder = TRTEngineBuilder(MODEL_TYPE=args.model_type, fp16=args.fp16)
    
    if not builder.initialized:
        print("Failed to initialize the TensorRT Engine Builder. Exiting.")
        sys.exit(1)
    
    # Build the engine
    builder.build_engine()

if __name__ == '__main__':
    main()