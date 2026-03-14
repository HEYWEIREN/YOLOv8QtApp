
# YOLOv8 Qt C++ Multi-task Deployment System | 基于 Qt + C++ 的 YOLOv8 多任务部署系统

## 📌 Introduction | 项目简介

This project integrates YOLOv8 with a Qt graphical interface to support **object detection**, **instance segmentation**, and **pose estimation** via ONNX Runtime using C++ and OpenCV.  
本项目通过 Qt 图形界面封装 YOLOv8 模型，实现基于 ONNX Runtime 的 **目标检测**、**实例分割**、**姿态估计** 等多任务功能，使用 C++ 和 OpenCV 构建。

---

## 🎯 Features | 功能特点

- ✅ Real-time image inference with YOLOv8
- ✅ Support for detect / segment / pose tasks
- ✅ Qt interface for image selection and result display
- ✅ GPU acceleration via ONNX Runtime CUDA
- ✅ Supports drawing bbox, mask, and keypoints with OpenCV

- ✅ 实时图像推理
- ✅ 支持检测 / 分割 / 姿态估计三种模型
- ✅ Qt 图形界面加载与展示图像
- ✅ 支持 GPU CUDA 加速推理
- ✅ OpenCV 绘制检测框、掩膜、关键点

---

## 🛠️ Dependencies | 项目依赖

- Qt 6.x or Qt 5.15+
- OpenCV 4.x
- ONNX Runtime (with CUDA support if GPU is used)
- C++17 or later
- Visual Studio 2022 (recommended)

---

## 🚀 How to Run | 如何运行

```bash
# Clone the repository
git clone https://github.com/yourname/yolov8-qt-app.git

# Open the .sln or .vcxproj file using Visual Studio

# Make sure Qt / OpenCV / ONNXRuntime paths are correctly configured

# Build and run
```

- Click **Open Image** to load a test image.
- Choose **Detect / Segment / Pose** to run inference.
- The result will be displayed and saved under `outputs/`.

点击 **Open Image** 加载图像，点击 **Detect / Segment / Pose** 开始推理，推理结果将显示在界面并保存在 `outputs/` 目录。

---

## 📂 Project Structure | 项目结构

```
├── main.cpp                     # Application entry
├── YOLOv8QtApp.*                # Qt main window
├── yolo_wrapper.*               # Inference wrapper for UI
├── autobackend.*                # Unified YOLOv8 backend
├── onnx_model_base.*            # ONNX session base class
├── augment.*                    # Letterbox and mask scaling
├── ops.*                        # NMS, box/keypoints decode
├── common.*                     # Utilities: Timer, parser
├── constants.h                  # Metadata keys and enums
```

---

## 📸 Screenshot | 运行截图

*(Add your image here)*

---

## 📌 Future Work | 后续拓展

- [ ] Support video stream inference
- [ ] Add model auto-download feature
- [ ] Export results as JSON

---

## 🔗 License | 许可协议

This project is licensed under the MIT License.  
本项目基于 MIT 协议开源。

