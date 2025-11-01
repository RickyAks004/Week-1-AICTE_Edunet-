# Week-1-AICTE_Edunet-
# 🚘 Multi-Task 2D Obstacle Detection for Autonomous Driving  
### Detect Vehicles, Pedestrians, Traffic Lights, Road Signs, Potholes & Speed Breakers — all in one model!

![banner](docs/banner.png) <!-- optional visual banner -->

---

## 📖 Overview

This project presents a **multi-task deep learning system** for **2D obstacle detection in autonomous vehicles** using only a **single RGB camera feed**.  
The network is designed to detect and understand various critical road elements in real time:

- 🧍‍♂️ **Pedestrians** and 🚗 **Vehicles**
- 🚦 **Traffic lights** and 🚧 **Traffic signs**
- 🕳️ **Potholes** and 🛑 **Speed breakers**

It uses a **shared backbone with specialized task heads**, enabling **efficient feature sharing** and **real-time inference** — ideal for edge devices (Jetson, Raspberry Pi, or onboard GPU systems).

---


> 💡 **Core idea:** One model learns shared features for the whole scene and performs multiple specialized perception tasks simultaneously.

---

## 🚀 Features

- 🔍 **Multi-task learning** — object detection + segmentation + color classification  
- 🧠 **SOTA architectures** (YOLOv9, RT-DETR, SegFormer, DeepLabV3+)  
- ⚡ **Single-pass inference** (real-time capable)  
- 🌈 **Color-aware traffic light recognition**  
- 🕳️ **Road anomaly segmentation** (potholes, speed bumps)  
- 📦 Modular design: easy to replace backbones or heads  

---

## 🏗️ Model Components

| Component | SOTA Architecture | Output | Description |
|------------|------------------|---------|--------------|
| **Backbone** | ConvNeXt-V2 / YOLOv9-E / Swin Transformer | Multi-scale features | Shared encoder for all tasks |
| **Head 1: Object Detection** | RT-DETR / YOLOv9 | Bounding boxes + class | Detect vehicles, pedestrians, signs |
| **Head 2: Traffic Light** | YOLOv9-S / EfficientDet | Box + light color | Detect and classify traffic lights |
| **Head 3: Road Surface** | SegFormer-B2 / DeepLabV3+ | Segmentation mask | Segment potholes and speed breakers |

---

## 🧰 Installation

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/multi-task-obstacle-detection.git
cd multi-task-obstacle-detection
```

### 2. Create Environment
```bash
conda create -n obstacle-detection python=3.10 -y
conda activate obstacle-detection
```


