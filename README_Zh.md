# 基于稳定跨模态融合与局部注意力建模的RGB-T目标检测方法

🚀 概述

RGB-YOLO11 是一种基于 YOLOv11 构建的实时 RGB-热红外（RGB-T）目标检测框架，旨在实现低光照和复杂环境下的鲁棒感知能力。与传统检测方法不同，现有模型通常存在跨模态融合不稳定以及局部空间建模能力不足的问题，从而限制了检测性能。
针对这些问题，RGB-YOLO11 在保证实时性的前提下，实现了检测精度与计算效率之间的平衡，使其适用于监控系统、自动驾驶以及边缘设备等实际部署场景。

## 模型结构
RGB-YOLO11 提出了两个关键模块，用于提升跨模态融合质量与特征表达能力：
- 稳定跨模态融合模块（Stable Cross-Modal Fusion, SCF）
  该模块通过注意力引导的双模态特征重标定与残差融合机制，有效抑制模态间干扰，同时保留RGB与热红外模态之间的互补信息。
- 窗口自注意力颈部结构（Window Self-Attention Neck, WSA-Neck）
  通过基于窗口的自注意力机制，实现多尺度特征的高效聚合与局部空间依赖建模，从而提升对小目标及密集目标的检测能力。


## 数据集

本研究在三个公开的RGB-T目标检测数据集上进行实验评估：

1. LLVIP （低光照可见光-红外配对数据集）

🔗 @inproceedings{llvip,
  title={LLVIP: A visible-infrared paired dataset for low-light vision},
  author={Jia, Xinyu and Zhu, Chuang and Li, Minzhen and Tang, Wenqi and Zhou, Wenli},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={3496--3504},
  year={2021}
}

2. M3FD （多模态融合目标检测数据集）

🔗 @inproceedings{m3fd,
  title={Target-aware dual adversarial learning and a multi-scenario multi-modality benchmark to fuse infrared and visible for object detection},
  author={Liu, Jinyuan and Fan, Xin and Huang, Zhanbo and Wu, Guanyao and Liu, Risheng and Zhong, Wei and Luo, Zhongxuan},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={5802--5811},
  year={2022}
}

3. DroneVehicle Dataset （无人机RGB-红外车辆检测数据集）

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
(环境安装推荐直接使用已配置好的YOLOv11 环境，无需重复安装）
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


### 4. 运行训练
```bash
python train.py --data your_dataset_config.yaml
```
#### 训练脚本说明

本项目包含多个训练脚本，适用于不同任务：

4.1. **`train.py`**
  - 基础训练脚本，适用于通用目标检测任务


4.2. **`train-rtdetr.py`**
   - 用于 RT-DETR 模型的训练

4.3. **`train_Gray.py`**
   - 灰度图训练脚本，适用于单通道图像任务


### 5.测试与验证

运行以下命令进行模型验证：
```bash
python val.py
```
