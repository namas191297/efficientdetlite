# EfficientDet Lite Object Detection with ONNX & TensorRT 🚀

![Object Detection GIF](efficientdetlite_demo.gif)

![License](https://creativecommons.org/licenses/by/3.0/deed.en)
![GitHub stars](https://img.shields.io/github/stars/yourusername/efficientdet-lite-onnx-tensorrt?style=social)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)

## Table of Contents
- [📖 Project Overview](#-project-overview)
- [✨ Features](#-features)
- [🛠 Installation](#-installation)
- [🚀 Usage](#-usage)
- [🧠 Models](#-models)
- [⚡ Performance Comparison](#-performance-comparison)
- [📈 Results](#-results)
- [📁 Repository Structure](#-repository-structure)
- [📜 License](#-license)
- [📫 Contact](#-contact)

---

## 📖 Project Overview

**EfficientDet Lite Object Detection with ONNX & TensorRT** is a high-performance project designed to implement EfficientDet Lite models (versions 0 to 4) for object detection. Utilizing ONNX for model inference and TensorRT for optimized engine building, this project enables efficient and rapid deployment of object detection models with support for FP32 and FP16 precision on NVIDIA GPUs.

---

## ✨ Features

- **Support for EfficientDet Lite Models**: Implemented versions 0, 1, 2, 3, and 4.
- **ONNX Inference**: Run inference directly using ONNX models.
- **TensorRT Engine Building**: Optimize models with TensorRT for FP32 and FP16 precision.
- **Inference Scripts**: Execute inference using both ONNX and TensorRT engines seamlessly.
- **Performance Benchmarking**: Compare latency and speed across different models and backends.
- (TO BE IMPLEMENTED) **INT8 Quantization**: INT8 Post-Training Quantization for faster inference. 

---

## 🛠 Installation

### Libraries and Tools

- [ONNX Runtime](https://github.com/microsoft/onnxruntime) (tested with version 1.19.2)
- [TensorRT](https://developer.nvidia.com/tensorrt) (tested with version 10.5.0)
- [PyCUDA] (https://pypi.org/project/pycuda/) (tested with version 2024.1.2)
- [cuda-python] (https://pypi.org/project/cuda-python/) (tested with version 12.2.1 - should be the same as installed CUDA version)

### Installation Steps

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/efficientdet-lite-onnx-tensorrt.git
    cd efficientdet-lite-onnx-tensorrt
    ```

2. **Set Up a Virtual Environment**

    ```bash
    conda create -n efficientdetlite python=3.9
    conda activate efficientdetlite 
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download EfficientDet Lite Models**

    - Follow the instructions in the [Models](#-models) section to obtain the required model files.

---

## 🚀 Usage

### Running Inference with ONNX

```bash
python scripts/infer_onnx.py --model path/to/model.onnx --image path/to/image.jpg
```

### Building TensorRT Engines

- **FP32 Precision**

    ```bash
    python scripts/build_trt_engine.py --model path/to/model.onnx --precision FP32 --output path/to/engine_fp32.trt
    ```

- **FP16 Precision**

    ```bash
    python scripts/build_trt_engine.py --model path/to/model.onnx --precision FP16 --output path/to/engine_fp16.trt
    ```

### Running Inference with TensorRT Engines

- **Using FP32 Engine**

    ```bash
    python scripts/infer_trt.py --engine path/to/engine_fp32.trt --image path/to/image.jpg
    ```

- **Using FP16 Engine**

    ```bash
    python scripts/infer_trt.py --engine path/to/engine_fp16.trt --image path/to/image.jpg
    ```

### Example Usage

- **Building TRT Engine from ONNX models**

```bash
# Build .engine TRT Engine for EfficientDetLit4 with FP32 precision.
python build_engine.py --model_type efficientdet_lite4

# Build .engine TRT Engine for EfficientDetLit4 with FP16 precision.
python build_engine.py --model_type efficientdet_lite4 --fp16
```
- **Single Image**

```bash
# Inference with ONNX on a single image
python onnx_inference_image.py --model_type efficientdet_lite1 --image test.jpg --score_threshold 0.5 --top_k 5

# Inference with TRT Engine on a single image using FP32 precision.
python trt_inference_image.py --model_type efficientdet_lite1 --image test.jpg --score_threshold 0.5 --top_k 5

# Inference with TRT Engine on a single image using FP16 precision.
python trt_inference_image.py --model_type efficientdet_lite1 --image test.jpg --score_threshold 0.5 --top_k 5 --fp16
```

- **Webcam**

```bash
# Inference with ONNX on your webcam
python onnx_inference_webcam.py --model_type efficientdet_lite1 --score_threshold 0.5 --top_k 5

# Inference with TRT Engine on your webcam using FP32 precision.
python trt_inference_webcam.py --model_type efficientdet_lite1 --score_threshold 0.5 --top_k 5

# Inference with TRT Engine on your webcam using FP32 precision.
python trt_inference_webcam.py --model_type efficientdet_lite1 --score_threshold 0.5 --top_k 5 --fp16
```

---

## 🧠 Models

### Supported Models

- **EfficientDet Lite 0**
- **EfficientDet Lite 1**
- **EfficientDet Lite 2**
- **EfficientDet Lite 3**
- **EfficientDet Lite 4**

### Model Details

- **Model Files**: All models are included in this repo but you can still download the pre-trained EfficientDet Lite models from [EfficientDetLite Google Drive Repo](https://drive.google.com/file/d/1Sdk_jRHQprOYP5xaUq0J5fM3sOlRadD6/view?usp=sharing).
- Place all .engine files under `trt_models/`.
- Place all the .onnx files under `onnx_models/`.

---

## ⚡ Performance Comparison

### Latency and Speed Metrics

The following table compares the latency (ms) of each EfficientDet Lite model across different backends when running on an NVIDIA RTX 3060.

| **Model** | **ONNX** | **TensorRT FP32** | **TensorRT FP16** |
|-----------|----------|-------------------|--------------------|
| Lite0     |    27    |        27         |         19         |
| Lite1     |    39    |        33         |         23         |
| Lite2     |    54    |        42         |         27         |
| Lite3     |    78    |        54         |         33         |
| Lite4     |   145    |        82         |         46         |

*Note: Replace the above numbers with your actual benchmark results.*

### Hardware Specifications

- **GPU**: NVIDIA RTX 3060
- **CUDA Version**: 12.2
- **TensorRT Version**: 10.5.0

---

## 📈 Results

### Detection Examples

![Detection Example](trt_output.jpg)
*Inference using EfficientDetLite 4*

### Benchmark Results

The project demonstrates significant improvements in inference speed when utilizing TensorRT, especially with FP16 precision. TensorRT FP16 offers up to **300%** speedup compared to ONNX for larger models, enabling real-time object detection applications.

---

## 📁 Repository Structure

```
root/
├── onnx_models/                 # ONNX model files    
├── trt_models/                  # TensorRT engine files                  
├── build_engine.py              # Script to build a TRT engine from ONNX models.
├── trt_engine_builder.py        # TRTEngineBuilder class implementation.
├── trt_executor.py              # TRTExecutor class implementation for inference.
├── trt_config.py                # Contains LABELS for classes and helper dictionary to build and run models. 
├── onnx_inference_image.py      # Script to run ONNX inference on a single image.
├── onnx_inference_webcam.py     # Script to run ONNX inference on webcam. 
├── trt_inference_image.py       # Script to run TRT inference on a single image.
├── trt_inference_webcam.py      # # Script to run ONNX inference on webcam.
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── LICENSE                      # License information
```

### Scripts Overview

- **build_engine.py**: Builds TensorRT engines from ONNX models with specified precision.
- **onnx_inference_image.py**: Runs inference using ONNX model on a single image. 
- **onnx_inference_webcam.py**: Runs inference using ONNX model on webcam.
- **trt_inference_image.py**: Runs inference using TRT engines on a single image.
- **trt_inference_webcam.py**: Runs inference using TRT engines on webcam.

---

## 📜 License

This project is licensed under the [Creative Commons Attribution 3.0](LICENSE).

---

## 📫 Contact
 
Email: [namas.brd@gmail.com](mailto:namas.brd@gmail.com)  
LinkedIn: [Namas Bhandari](https://www.linkedin.com/in/namas-bhandari/)  

Feel free to reach out for any questions, suggestions, or collaborations!

---
