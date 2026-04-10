# Robust Cross-Modal Feature Learning and Local Attention Modeling for RGB-Thermal Object Detection

🚀 Overview

RGB-YOLO11 is a real-time RGB-thermal object detection framework built upon YOLOv11, designed for robust perception in low-light and complex environmental scenarios.

Unlike conventional detection models that suffer from unstable cross-modal fusion and limited local spatial modeling capability, RGB-YOLO11 focuses on achieving a better balance between detection accuracy and real-time efficiency, making it suitable for practical deployment in surveillance, autonomous systems, and edge devices.
## Model Architecture
RGB-YOLO11 introduces two key modules to improve cross-modal fusion quality and feature representation capability:
- Stable Cross-Modal Fusion (SCMF) Module
  Enables attention-guided bimodal feature recalibration and residual fusion. It effectively suppresses modality interference while preserving complementary information between RGB and thermal features.
- Window Self-Attention Neck (WSA-Neck)
  Enhances multi-scale feature aggregation and local spatial dependency modeling through window-based self-attention, improving detection performance for small and densely distributed objects.


## Datasets

The experiments are conducted on three medical datasets:

1. LLVIP (Low-Light Visible-Infrared Paired Dataset)

🔗 @inproceedings{llvip,
  title={LLVIP: A visible-infrared paired dataset for low-light vision},
  author={Jia, Xinyu and Zhu, Chuang and Li, Minzhen and Tang, Wenqi and Zhou, Wenli},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={3496--3504},
  year={2021}
}

2. M3FD (Multi-Modal Fusion Detection Dataset)

🔗 @inproceedings{m3fd,
  title={Target-aware dual adversarial learning and a multi-scenario multi-modality benchmark to fuse infrared and visible for object detection},
  author={Liu, Jinyuan and Fan, Xin and Huang, Zhanbo and Wu, Guanyao and Liu, Risheng and Zhong, Wei and Luo, Zhongxuan},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={5802--5811},
  year={2022}
}

3. DroneVehicle Dataset (RGB-IR Vehicle Detection)

🔗 @article{drone,
  title={Drone-based RGB-infrared cross-modality vehicle detection via uncertainty-aware learning},
  author={Sun, Yiming and Cao, Bing and Zhu, Pengfei and Hu, Qinghua},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={32},
  number={10},
  pages={6700--6713},
  year={2022},
  publisher={IEEE}
}


### 3. Install Dependencies
(It is recommended to directly use the YOLOv11 or YOLOv8 environment that has already been set up on this computer, without the need to download again.)
```bash
# Step 1.Create a virtual environment with conda
conda create -n pt121_py38 python=3.8
conda activate pt121_py38

# Step 2: Install pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch


# Step 3: Install the remaining dependencies

pip install -r requirements.txt


# https://pytorch.org/get-started/previous-versions/
## CUDA 10.2
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
## CUDA 11.3
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
## CUDA 11.6
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
## CPU Only
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch

## CUDA 11.8
#conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
## CUDA 12.1
#conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
## CPU Only
#conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch
```


### 4. Run the Program
```bash
python train.py --data your_dataset_config.yaml
```
#### Explanation of Training Modes

Below are the Python script files for different training modes included in the project, each targeting specific training needs and data types.

4.1. **`train.py`**
   - Basic training script.
   - Used for standard training processes, suitable for general image classification or detection tasks.

2. **`train-rtdetr.py`**
   - Training script for RTDETR (Real-Time Detection Transformer).

3. **`train_Gray.py`**
   - Grayscale image training script.
   - Specifically for processing datasets of grayscale images, suitable for tasks requiring image analysis in grayscale space.


### 5. Testing
Run the test script to verify if the data loading is correct:
```bash
python val.py
```
