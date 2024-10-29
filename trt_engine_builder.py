import tensorrt as trt
from cuda import cuda, cudart
import numpy as np
from typing import Optional, List, Union
import ctypes
import os
from trt_config import efficientdet_model_dict

def cuda_call(call):
        err, res = call[0], call[1:]
        check_cuda_err(err)
        if len(res) == 1:
            res = res[0]
        return res

def check_cuda_err(err):
        if isinstance(err, cuda.CUresult):
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("Cuda Error: {}".format(err))
        if isinstance(err, cudart.cudaError_t):
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError("Cuda Runtime Error: {}".format(err))
        else:
            raise RuntimeError("Unknown error type: {}".format(err))

class TRTEngineBuilder:
    def __init__(self, MODEL_TYPE='efficientdet_lite0', fp16=False):
        self.initialized = True
        if MODEL_TYPE not in efficientdet_model_dict.keys():
            print(f'Model Type:{MODEL_TYPE} does not exist, the available ONNX models are:{list(efficientdet_model_dict.keys())}')
            self.initialized = False
            return
        self.TRT_MODELS_DIR = 'trt_models'
        os.makedirs(self.TRT_MODELS_DIR, exist_ok=True)
        self.MODEL_TYPE = MODEL_TYPE
        self.MODEL_PATH = efficientdet_model_dict[self.MODEL_TYPE]['MODEL_PATH']
        self.TARGET_SIZE = efficientdet_model_dict[self.MODEL_TYPE]['TARGET_SIZE']
        self.ONNX_INPUT_SHAPE = (1, self.TARGET_SIZE, self.TARGET_SIZE, 3)
        self.DTYPE = trt.uint8
        self.FP16 = fp16

        # You can set the logger severity higher to suppress messages (or lower to display more messages).
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING) 
    
    def _build_engine_onnx(self):
        builder = trt.Builder(self.TRT_LOGGER)
        network = builder.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, self.TRT_LOGGER)

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 * 1 << 30)

        profile = builder.create_optimization_profile()
        profile.set_shape("serving_default_images:0", self.ONNX_INPUT_SHAPE, self.ONNX_INPUT_SHAPE, self.ONNX_INPUT_SHAPE)
        config.add_optimization_profile(profile)

        # Check if FP16 is enabled
        if self.FP16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(self.MODEL_PATH, "rb") as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        engine_bytes = builder.build_serialized_network(network, config)
        return engine_bytes
    
    def build_engine(self):
        if not self.initialized:
            print('Error building engine, returning.')
            return None
        
        print(f'Building the engine....')
        engine_bytes = self._build_engine_onnx()
        if self.FP16:
            engine_file = f"{self.TRT_MODELS_DIR}/{self.MODEL_TYPE}_fp16.engine"
        else:
            engine_file = f"{self.TRT_MODELS_DIR}/{self.MODEL_TYPE}_fp32.engine"
            
        with open(engine_file, "wb") as f:
            f.write(engine_bytes)
        print(f"Serialized TensorRT engine saved to '{engine_file}'.")

if __name__ == '__main__':
    builder = TRTEngineBuilder('efficientdet_lite4', fp16=False)
    builder.build_engine()