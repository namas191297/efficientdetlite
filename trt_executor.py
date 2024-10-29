import tensorrt as trt
import os
import numpy as np
from typing import Optional, Union, List
import ctypes
from cuda import cuda, cudart
import cv2
from trt_config import LABELS, efficientdet_model_dict

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

class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: Optional[np.dtype] = None):
        dtype = dtype or np.dtype(np.uint8)
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes
        
    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, data: Union[np.ndarray, bytes]):
        if isinstance(data, np.ndarray):
            if data.size > self.host.size:
                raise ValueError(
                    f"Tried to fit an array of size {data.size} into host memory of size {self.host.size}"
                )
            np.copyto(self.host[:data.size], data.flat, casting='safe')
        else:
            assert self.host.dtype == np.uint8
            self.host[:self.nbytes] = np.frombuffer(data, dtype=np.uint8)

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


class TRTExecutor:

    def __init__(self, MODEL_TYPE='efficientdet_lite0', fp16=False):
        if MODEL_TYPE not in efficientdet_model_dict.keys():
            print(f'{MODEL_TYPE} does not exist, the available ONNX models are:{list(efficientdet_model_dict.keys())}')  

        self.FP16 = fp16
        self.MODEL_TYPE = MODEL_TYPE  
        self.ENGINE_PATH = efficientdet_model_dict[MODEL_TYPE]['ENGINE_PATH_FP16'] if self.FP16 else efficientdet_model_dict[MODEL_TYPE]['ENGINE_PATH_FP32']

        if not os.path.exists(self.ENGINE_PATH):
            print(f'{self.ENGINE_PATH} does not exist, either download the .engine file or build one.')
            return
    
        self.TARGET_SIZE = efficientdet_model_dict[self.MODEL_TYPE]['TARGET_SIZE']
        self.INPUT_SHAPE = (self.TARGET_SIZE, self.TARGET_SIZE, 3)
        self.DTYPE = trt.uint8
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.TRT_LOGGER)
        self.engine = None
        self.engine_data = None
        self.load_engine()

    def load_engine(self):
        with open(self.ENGINE_PATH, 'rb') as f:
            self.engine_data = f.read()
            self.engine = self.runtime.deserialize_cuda_engine(self.engine_data)

    def allocate_buffers(self, engine: trt.ICudaEngine, profile_idx: Optional[int] = None):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda_call(cudart.cudaStreamCreate())
        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        for binding in tensor_names:
            # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
            # Pick out the max shape to allocate enough memory for the binding.
            if binding == 'StatefulPartitionedCall:3':
                shape = (1,25,4)
            elif binding == 'StatefulPartitionedCall:2':
                shape = (1,25)
            elif binding == 'StatefulPartitionedCall:1':
                shape = (1,25)
            elif binding == 'StatefulPartitionedCall:31':
                shape = (1,25,4)
            elif binding == 'StatefulPartitionedCall:32':
                shape = (1,25)
            elif binding == 'StatefulPartitionedCall:33':
                shape = (1,25)
            else:
                shape = engine.get_tensor_shape(binding) if profile_idx is None else engine.get_tensor_profile_shape(binding, profile_idx)[-1]
            shape_valid = np.all([s >= 0 for s in shape])
            if not shape_valid and profile_idx is None:
                raise ValueError(f"Binding {binding} has dynamic shape, " +\
                    "but no profile was specified.")
            size = trt.volume(shape)
            trt_type = engine.get_tensor_dtype(binding)

            # Allocate host and device buffers
            try:
                dtype = np.dtype(trt.nptype(trt_type))
                bindingMemory = HostDeviceMem(size, dtype)
            except TypeError: # no numpy support: create a byte array instead (BF16, FP8, INT4)
                size = int(size * trt_type.itemsize)
                bindingMemory = HostDeviceMem(size)

            # Append the device buffer to device bindings.
            bindings.append(int(bindingMemory.device))

            # Append to the appropriate list.
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                inputs.append(bindingMemory)
            else:
                outputs.append(bindingMemory)
        return inputs, outputs, bindings, stream
    
    def _do_inference_base(self, inputs, outputs, stream, execute_async_func):
        # Transfer input data to the GPU.
        kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream)) for inp in inputs]
        # Run inference.
        execute_async_func()
        # Transfer predictions back from the GPU.
        kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        [cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream)) for out in outputs]
        # Synchronize the stream
        cuda_call(cudart.cudaStreamSynchronize(stream))
        # Return only the host outputs.
        return [out.host for out in outputs]

    def do_inference(self, context, engine, bindings, inputs, outputs, stream):
        def execute_async_func():
            context.execute_async_v3(stream_handle=stream)
        # Setup context tensor address.
        num_io = engine.num_io_tensors
        for i in range(num_io):
            context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
        return self._do_inference_base(inputs, outputs, stream, execute_async_func)

    def free_buffers(self, inputs: List[HostDeviceMem], outputs: List[HostDeviceMem], stream: cudart.cudaStream_t):
        for mem in inputs + outputs:
            mem.free()
        cuda_call(cudart.cudaStreamDestroy(stream))

    def preprocess_image(self, img, target_size=(320, 320)):
        resized_img = cv2.resize(img, target_size)
        resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        input_array = np.expand_dims(resized_img_rgb, axis=0)  # Add batch dimension
        input_array = input_array.astype(trt.nptype(self.DTYPE)).ravel()
        return input_array
    
    def postprocess_outputs(self, boxes, class_ids, scores, original_size, score_threshold=0.5, top_k=3):
        boxes = boxes.reshape(25,4)
        h_original, w_original = original_size

        detections = []
        for i in range(len(scores)):
            if scores[i] < score_threshold:
                continue
            if len(detections) >= top_k:
                break
            class_id = int(class_ids[i])  # Adjust if class IDs are 1-based
            if class_id < 0 or class_id >= len(LABELS):
                label = 'N/A'
            else:
                label = LABELS[class_id]
            ymin, xmin, ymax, xmax = boxes[i]

            # Denormalize coordinates
            left = int(xmin * w_original)
            right = int(xmax * w_original)
            top = int(ymin * h_original)
            bottom = int(ymax * h_original)

            detections.append({
                'box': (left, top, right, bottom),
                'label': label,
                'score': scores[i]
            })
        
        return detections

    def inference(self, image):
        original_size = image.shape[:2]
        image = self.preprocess_image(image, target_size=(self.TARGET_SIZE, self.TARGET_SIZE))
        inputs, outputs, bindings, stream = self.allocate_buffers(self.engine)
        context = self.engine.create_execution_context()
        np.copyto(inputs[0].host, image)
        trt_outputs = self.do_inference(
            context,
            engine=self.engine,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
        )
        boxes = trt_outputs[0]
        class_ids = trt_outputs[1]
        scores = trt_outputs[2]
        detections = self.postprocess_outputs(boxes, class_ids, scores, original_size)
        self.free_buffers(inputs, outputs, stream)
        return detections

if __name__ == '__main__':
    executor = TRTExecutor()
    img_path = 'dog.jpg'
    image = cv2.imread(img_path)
    dets = executor.inference(image)
    print(dets)
    dets = executor.inference(image)
    print(dets)
    dets = executor.inference(image)
    print(dets)