# Multimodal 3D Object Detection: Robustness, Redundancy, and Embedded Deployment

A comprehensive research and benchmarking framework for evaluating multimodal 3D object detection models (**Camera-LiDAR Fusion**) on autonomous driving datasets (**KITTI**). This project provides end-to-end tooling for:

1. **Synthetic Weather & Sensor Corruption Generation**: Realistic simulation of adverse environmental conditions (fog, rain, snow, darkness, intense sunlight/glare) and sensor dropouts (`no_camera`, `no_lidar`, noise, FOV loss) with 5 severity levels. The synthetic corruption generation suite is adapted from the benchmark repository **[thu-ml/3D_Corruptions_AD](https://github.com/thu-ml/3D_Corruptions_AD)** (*"Benchmarking Robustness of 3D Object Detection to Common Corruptions in Autonomous Driving"*, CVPR 2023).
2. **Multimodal Model Evaluation**: Comparative analysis of state-of-the-art fusion paradigms:
   - **BEVFusion** (Shared Bird's-Eye-View representation via Depth LSS Transform & Swin Transformer).
   - **MVX-Net** (Point-wise and Dynamic Voxel-wise multimodal fusion).
   - **VirConv** (Virtual Sparse Convolutions with PENet depth completion).
3. **Distance-Segmented Robustness & Redundancy Metrics**: Detection evaluation segmented into near ($0\text{--}30\,\text{m}$), mid ($30\text{--}50\,\text{m}$), far ($50\text{--}80\,\text{m}$), and global ($0\text{--}80\,\text{m}$) distance ranges using Relative Accuracy ($\text{RA}$) and Relative Corruption Error ($\text{RCE}\%$).
4. **Embedded Edge Deployment & Hardware Profiling**: Real-time telemetry on **NVIDIA Jetson Orin NX** platform (GPU/CPU usage, power draw in Watts, thermal dynamics, memory footprint, and inference FPS/latency).

---

## Table of Contents

- [System Architecture & Workflow](#system-architecture--workflow)
- [Repository Structure](#repository-structure)
- [Prerequisites & Environment Setup](#prerequisites--environment-setup)
  - [Docker for NVIDIA Jetson](#docker-for-nvidia-jetson)
  - [Python Dependencies (`requirements.txt`)](#python-dependencies-requirementstxt)
  - [NVIDIA Jetson Docker Containers](#nvidia-jetson-docker-containers)
- [Dataset Preparation (KITTI)](#dataset-preparation-kitti)
- [Step-by-Step Usage Guide](#step-by-step-usage-guide)
  - [1. Generating Corrupted Datasets](#1-generating-corrupted-datasets)
    - [1.1 Quick Verification on a Sample Frame](#11-quick-verification-on-a-sample-frame)
    - [1.2 Batch Camera Corruptions](#12-batch-camera-corruptions)
    - [1.3 Batch LiDAR Corruptions](#13-batch-lidar-corruptions)
    - [1.4 Assemble the Complete Corrupted Dataset Structure](#14-assemble-the-complete-corrupted-dataset-structure)
  - [2. Preprocessing & Dataset Info Generation](#2-preprocessing--dataset-info-generation)
    - [2.1 For MMDetection3D (BEVFusion & MVX-Net)](#21-for-mmdetection3d-bevfusion--mvx-net)
    - [2.2 For VirConv (OpenPCDet)](#22-for-virconv-openpcdet)
    - [2.3 Class Distribution by Distance Analysis](#23-class-distribution-by-distance-analysis)
  - [3. Model Training & Evaluation](#3-model-training--evaluation)
    - [3.1 Model Training on Clean KITTI](#31-model-training-on-clean-kitti)
    - [3.2 Baseline Evaluation on Clean KITTI](#32-baseline-evaluation-on-clean-kitti)
    - [3.3 Testing with Corrupted Datasets (Robustness & Sensor Dropout Evaluation)](#33-testing-with-corrupted-datasets-robustness--sensor-dropout-evaluation)
    - [3.4 Confusion Matrix Generation & Classification Metrics](#34-confusion-matrix-generation--classification-metrics)
    - [3.5 Latency & Throughput (FPS) Measurement](#35-latency--throughput-fps-measurement)
  - [4. Edge Hardware Benchmarking on Jetson Orin](#4-edge-hardware-benchmarking-on-jetson-orin)
    - [4.1 Start Telemetry Logger in Background](#41-start-telemetry-logger-in-background)
    - [4.2 Run Inference Benchmark](#42-run-inference-benchmark)
  - [5. Analysis & Visualization](#5-analysis--visualization)
    - [5.1 Robustness Analysis across Severities & Distances](#51-robustness-analysis-across-severities--distances)
    - [5.2 Modal Redundancy Analysis (Sensor Dropout)](#52-modal-redundancy-analysis-sensor-dropout)
    - [5.3 Jetson Embedded Performance Visualization](#53-jetson-embedded-performance-visualization)
    - [5.4 Training Convergence Plots](#54-training-convergence-plots)
- [Evaluation Metrics & Formulation](#evaluation-metrics--formulation)
  - [1. Detection Performance (KITTI AP40)](#1-detection-performance-kitti-ap40)
  - [2. Distance Segmentation](#2-distance-segmentation)
  - [3. Relative Accuracy (RA)](#3-relative-accuracy-ra)
  - [4. Relative Corruption Error (RCE%)](#4-relative-corruption-error-rce)
- [Checkpoints Included](#checkpoints-included)
- [Citations & Tool References](#citations--tool-references)
  - [1. Benchmark Dataset & Corruption Suite](#1-benchmark-dataset--corruption-suite)
  - [2. Multimodal 3D Detection Models](#2-multimodal-3d-detection-models)
  - [3. Deep Learning & 3D Detection Frameworks](#3-deep-learning--3d-detection-frameworks)

---

## System Architecture & Workflow

```
       +-------------------------------------------------------------+
       |               Raw KITTI 3D Dataset                          |
       |     (Camera Images, Velodyne Point Clouds, Calib, Labels)   |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |               Corruptions Module (`Corruptions/`)           |
       |  - Camera: Fog, Rain, Snow, Sunlight, Darkness, No-Camera   |
       |  - LiDAR:  LISA (Rain/Snow), Optical Fog Sim, Glare, Noise  |
       |  - Severities: Level 1 to 5 per condition                   |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |             Dataset Formatting & Info Generation            |
       |  - `generate_corruption_dataset.py` (Organize KITTI dirs)   |
       |  - `generate_infos_mmdet3.py` (MMDetection3D: BEVFusion/MVX)|
       |  - `generate_infos_virconv.py` (OpenPCDet: VirConv)         |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |                Model Evaluation & Inference                 |
       |  - BEVFusion (MMDet3D)  - MVX-Net (MMDet3D)  - VirConv       |
       |  - Distance range evaluation: [0-30m], [30-50m], [50-80m]   |
       +------------------------------+------------------------------+
                    |                                 |
                    v                                 v
  +-----------------------------------+  +----------------------------+
  |    Analysis & Visualizations      |  | Embedded Hardware Profiling|
  |  - Robustness (RA & mRA curves)   |  | - NVIDIA Jetson Orin NX    |
  |  - Redundancy (RCE% sensor loss)  |  | - jtop: Power, Temp, GPU%  |
  |  - Training curves & Confusion M. |  | - Latency & FPS benchmarks |
  +-----------------------------------+  +----------------------------+
```

---

## Repository Structure

```text
proyecto_TFG/
├── Corruptions/                      # Synthetic corruption generator suite
│   ├── Camera_corruptions.py         # Image corruptions (weather, lighting, noise)
│   ├── LiDAR_corruptions.py          # Point cloud corruptions (LISA rain/snow, fog, glare)
│   ├── img_converter.py              # Batch corruption script for camera images
│   ├── point_converter.py            # Batch corruption script for LiDAR point clouds
│   ├── generate_corruption_dataset.py# Copies calibration and labels into corrupted splits
│   ├── proj_velo2cam.py              # Projects 3D LiDAR points onto 2D camera images
│   ├── ver_lidar.py                  # Open3D visualization script for point clouds
│   ├── prueba_corrupciones.py        # Single-sample corruption validation script
│   ├── prueba_sunlight.py            # Sunlight glare validation script
│   ├── kitti_ejemplos/               # Sample KITTI frames for quick testing
│   ├── corruption_examples/          # Generated corruption preview outputs
│   └── utils/                        # Physics models, LISA, fog tables, Automold
│
├── jetson_packages/                  # Embedded deployment modules & Dockerfiles
│   ├── mmdetection3d/                # MMDetection3D container setup for Jetson
│   │   ├── Dockerfile                # L4T-PyTorch base Dockerfile with spconv/cumm SM 8.7
│   │   ├── benchmark/                # Latency and telemetry tools
│   │   │   ├── benchmark.py          # Inference throughput (FPS) and latency
│   │   │   ├── jetson_metrics.py     # Real-time jtop logger (Power, Temp, CPU/GPU)
│   │   │   ├── eval.py               # Custom KITTI eval with distance slicing
│   │   │   ├── kitti_metric.py       # Custom distance-aware KITTI metric
│   │   │   └── test_multiple_ckpt.py # Multi-checkpoint automated evaluator
│   │   ├── models/                   # Configs and modules for BEVFusion and MVX-Net
│   │   │   ├── bevfusion/            # BEVFusion configs and loading routines
│   │   │   └── mvxnet/               # MVX-Net configs and learning rate schedules
│   │   └── spconv/                   # Custom spconv/cumm compilation for Jetson
│   ├── virconv/                      # VirConv container setup with OpenPCDet
│   │   ├── Dockerfile                # VirConv Jetson Dockerfile
│   │   ├── VirConv/                  # Full VirConv OpenPCDet codebase
│   │   └── spconv/                   # Optimized spconv installer
│ 
│
├── visualizar_TFG/                   # Analysis, plotting, and metric calculation
│   ├── robustez.py                   # Computes Relative Accuracy (RA/mRA) & plots AP lines
│   ├── redundancia.py                # Computes Relative Corruption Error (RCE%) for no_camera
│   ├── visualizacion_orin.py         # Multi-panel Jetson telemetry graphs & summary CSV
│   ├── orin.rtf                      # Per-core CPU load and thermal dynamics script
│   ├── bevfusion/                    # BEVFusion evaluation CSVs, curves, and timings
│   ├── mvxnet/                       # MVX-Net evaluation CSVs, curves, and timings
│   ├── virconv/                      # VirConv evaluation CSVs, curves, and timings
│   └── Orin/                         # Aggregated Jetson performance charts and summaries
│
├── resultados/                       # Experimental logs and training progression
│   ├── plot_training_curve.py        # Plots training AP curves (3D and AOS) across epochs
│   ├── bevfusion/                    # Tensorboard logs, confusion matrices, eval logs
│   ├── mvxnet/                       # MVX-Net training outputs and metrics
│   └── virconv/                      # VirConv training outputs and predictions
│
├── generate_infos_mmdet3.py          # Generates kitti_infos_val.pkl for MMDetection3D
├── generate_infos_virconv.py         # Generates kitti_infos_val.pkl for VirConv
├── clases_por_distancia.py           # Counts GT objects per class, difficulty, & distance
├── checkpoint_epoch_60.pth           # Trained model checkpoint (VirConv)
├── epoch_15.pth                      # Trained model checkpoint (MMDetection3D)
├── epoch_20.pth                      # Trained model checkpoint (MMDetection3D)
└── epoch_40.pth                      # Trained model checkpoint (MMDetection3D)
```

---

## Prerequisites & Environment Setup

### Docker for NVIDIA Jetson

To build and run containers with GPU acceleration on NVIDIA Jetson devices, you must have the specific Docker and NVIDIA Container Runtime installed and configured to use the GPU.


### Python Dependencies (`requirements.txt`)

For data generation, corruption simulation, info preprocessing, and result visualization on your host machine:

```bash
pip install -r requirements.txt
```

### NVIDIA Jetson Docker Containers

Pre-configured Dockerfiles are provided in `jetson_packages/` for deployment on NVIDIA Jetson devices running JetPack 5.x / 6.x (L4T):

1. **Build Containers**:
   ```bash
   jetson-containers build --name=<name_image> --package-dirs= <path_to_dockerfile> <alias_dockerfile>
   ```
2. **Run Containers**:
   ```bash
   docker run --runtime nvidia -it -v <path_to_dataset>:<path_to_dataset_container> <name_image>
   ```

## Dataset Preparation (KITTI)

The framework evaluates on the **KITTI 3D Object Detection Benchmark**. Ensure your original dataset is structured as follows:

```text
kitti/
├── ImageSets/
│   ├── train.txt
│   └── val.txt
└── training/
    ├── image_2/               # Left color images (.png)
    ├── velodyne/              # Uncompressed LiDAR scans (.bin)
    ├── calib/                 # Calibration matrices (.txt)
    └── label_2/               # 3D bounding box annotations (.txt)
```

---

## Step-by-Step Usage Guide

### 1. Generating Corrupted Datasets

The corruption pipeline generates degraded camera images and point clouds across 5 severity levels for each condition:
- **Weather conditions**: `rain`, `snow`, `fog`
- **Lighting conditions**: `darkness` (night), `sunlight` (intense glare)
- **Sensor failure/dropout**: `no_camera` (blacked-out images), `no_lidar` (empty point clouds)

#### 1.1 Quick Verification on a Sample Frame
Test corruptions and projection on sample data:
```bash
cd Corruptions

# Run camera and LiDAR corruptions on sample frame 000000
python prueba_corrupciones.py

# Project corrupted LiDAR onto corrupted images to verify calibration & alignment
python proj_velo2cam.py

# Visualize a corrupted point cloud in 3D using Open3D
python ver_lidar.py
```

#### 1.2 Batch Camera Corruptions
Generate corrupted camera images for the entire validation set:
```bash
cd Corruptions

# Syntax: python img_converter.py -c <corruption> -f <severity_1-5> -r <kitti_root> -s <save_root>
python img_converter.py -c fog -f 1 -r /path/to/kitti -s /path/to/kittiCorrupt 
python img_converter.py -c rain -f 3 -r /path/to/kitti -s /path/to/kittiCorrupt 
python img_converter.py -c no_camera -r /path/to/kitti -s /path/to/kittiCorrupt
```

#### 1.3 Batch LiDAR Corruptions
Generate corrupted LiDAR point clouds for the entire validation set:
```bash
cd Corruptions

# Syntax: python point_converter.py -c <corruption> -f <severity_1-5> -r <kitti_root> -s <save_root>
python point_converter.py -c fog -f 1 -r /path/to/kitti -s /path/to/kittiCorrupt 
python point_converter.py -c rain -f 3 -r /path/to/kitti -s /path/to/kittiCorrupt
python point_converter.py -c snow -f 5 -r /path/to/kitti -s /path/to/kittiCorrupt 
```

#### 1.4 Assemble the Complete Corrupted Dataset Structure
Copy `val.txt`, label annotations, and calibration files into each corruption severity directory:
```bash
cd Corruptions
python generate_corruption_dataset.py \
  --dataset_root /path/to/kitti \
  --corrupted_root /path/to/kittiCorrupt
```

The resulting structure will be ready for model evaluation:
```text
kittiCorrupt/
├── fog/
│   ├── severity_1/
│   │   ├── ImageSets/val.txt
│   │   └── training/
│   │       ├── image_2/
│   │       ├── velodyne/
│   │       ├── calib/
│   │       └── label_2/
│   ├── severity_2/ ...
├── rain/ ...
├── snow/ ...
├── sunlight/ ...
├── darkness/ ...
└── no_camera/ ...
```

---

### 2. Preprocessing & Dataset Info Generation

Before evaluating models, generate the required dataset info `.pkl` files and reduced point clouds.

#### 2.1 For MMDetection3D (BEVFusion & MVX-Net)
Run `generate_infos_mmdet3.py` pointing to the corruption and severity level:
```bash
# For a specific weather condition and severity
python generate_infos_mmdet3.py --corruption rain --severity 1
python generate_infos_mmdet3.py --corruption fog --severity 3

# For complete sensor loss
python generate_infos_mmdet3.py --corruption no_camera
python generate_infos_mmdet3.py --corruption no_lidar
```

#### 2.2 For VirConv (OpenPCDet)
Run `generate_infos_virconv.py`:
```bash
python generate_infos_virconv.py --corruption rain --severity 1
python generate_infos_virconv.py --corruption no_camera
```

#### 2.3 Class Distribution by Distance Analysis
Inspect ground-truth sample distribution for each class across distance intervals:
```bash
python clases_por_distancia.py
# Outputs summary statistics into 'clases_por_distancia.txt'
```

---

### 3. Model Evaluation

> [!NOTE]
> All model evaluation and latency benchmarking commands below should be executed **inside the respective Jetson Docker container** (`mmdet3d:r36.4.0` or `virconv:r36.4.0`) where GPU acceleration, CUDA drivers, and compiled extensions (`spconv`, `cumm`) are fully loaded.


#### 3.2 Baseline Evaluation on Clean KITTI

Evaluate models on the clean KITTI validation set to establish baseline detection performance:

```bash
# 1. VirConv-L baseline evaluation
cd /VirConv/tools
python3 test.py \
  --cfg_file cfgs/models/kitti/VirConv-L_3class.yaml \
  --batch_size 1 \
  --ckpt /output/models/kitti/VirConv-L_3class/default/ckpt/epoch_60.pth

# 2. MVX-Net baseline evaluation
cd /mmdetection3d
python3 tools/test.py \
  configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py \
  /work_dirs/mvxnet_kitti3/ckpt/epoch_40.pth

# 3. BEVFusion baseline evaluation
cd /mmdetection3d
python3 tools/test.py \
  projects/BEVFusion/configs/bevfusion_kitti3_lidar-cam.py \
  /work_dirs/bevfusion_kitti3_lidar-cam/ckpt/epoch_20.pth
```

---

#### 3.3 Testing with Corrupted Datasets 

To evaluate model resilience under weather perturbations (`fog`, `rain`, `snow`), lighting variations (`darkness`, `sunlight`), and sensor dropout (`no_camera`, `no_lidar`), models are evaluated sequentially across severities ($1\text{--}5$). Results are automatically logged into `<model_name>_corruption_eval.csv`.

##### A. Evaluating VirConv on Corrupted Splits
1. **Generate virtual points for the target corruption and severity level**:
   ```bash
   cd /VirConv/tools/PENet
   python3 main.py --detpath ../../data/kittiCorrupt/<corruption>/severity_<severity>/training -ce

   # Example for snow severity 1:
   python3 main.py --detpath ../../data/kittiCorrupt/snow/severity_1/training -ce
   ```

2. **Generate corrupted dataset validation info**:
   ```bash
   cd /VirConv/tools
   python3 generate_infos_virconv.py --corruption snow --severity 1
   ```

3. **Configure `DATA_PATH`**:
   In `tools/cfgs/dataset_configs/kitti_dataset.yaml`, set:
   ```yaml
   DATA_PATH: '../data/kittiCorrupt/snow/severity_1'
   ```

4. **Run evaluation with corruption logging**:
   ```bash
   python3 test.py \
     --cfg_file cfgs/models/kitti/VirConv-L_3class.yaml \
     --batch_size 1 \
     --ckpt /output/models/kitti/VirConv-L_3class/default/ckpt/epoch_60.pth \
     --corruption_eval
   ```
   *(Executing consecutively for different corruptions/severities accumulates results into `VirConv_corruption_eval.csv`).*

##### B. Evaluating MMDetection3D (BEVFusion & MVX-Net) on Corrupted Splits
1. **Generate corrupted dataset validation info & reduced point clouds**:
   ```bash
   cd /mmdetection3d
   # For weather and lighting corruptions (severities 1 to 5):
   python3 generate_infos_mmdet3d.py --corruption snow --severity 1
   python3 generate_infos_mmdet3d.py --corruption fog --severity 3

   # For sensor failure / modal dropout:
   python3 generate_infos_mmdet3d.py --corruption no_camera
   python3 generate_infos_mmdet3d.py --corruption no_lidar
   ```

2. **Configure `data_root` and `val_evaluator` in the model config**:
   In `configs/mvxnet/...` or `projects/BEVFusion/...`, set:
   ```python
   data_root = 'data/kittiCorrupt/snow/severity_1/'
   val_evaluator = dict(
       type='KittiMetric',
       ann_file=data_root + 'kitti_infos_val.pkl',
       corruption_savepath='data/kittiCorrupt',
       model_name='mvxnet',  # or 'bevfusion'
       corruption_eval=True
   )
   ```

3. **Execute testing with corruption evaluation**:
   ```bash
   # Test MVX-Net on corrupted split:
   python3 tools/test.py \
     configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py \
     /work_dirs/mvxnet_kitti3/ckpt/epoch_40.pth

   # Test BEVFusion on corrupted split:
   python3 tools/test.py \
     projects/BEVFusion/configs/bevfusion_kitti3_lidar-cam.py \
     /work_dirs/bevfusion_kitti3_lidar-cam/ckpt/epoch_20.pth
   ```
   *(Results for all distance ranges: [0-30m], [30-50m], [50-80m], and [0-80m] are appended into `<model_name>_corruption_eval.csv`).*

---

#### 3.4 Confusion Matrix Generation & Classification Metrics

Compute 3D IoU confusion matrices, recall and F1:

```bash
# 1. VirConv Confusion Matrix (IoU 3D >= 0.5, Score >= 0.5, Moderate difficulty)
cd /VirConv/tools
python3 confusion_matrix.py \
  --config cfgs/models/kitti/VirConv-L_3class.yaml \
  --prediction_path ./result.pkl \
  --save_dir ./matrix \
  --kitti_eval_classes \
  --difficulty 1 \
  --iou_type 3 \
  --score-thr 0.5 \
  --tp-iou-thr 0.5 \
  --normalize

# 2. MVX-Net Confusion Matrix
cd /mmdetection3d
python3 tools/analysis_tools/confusion_matrix.py \
  configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py \
  results/kitti-3class/mvxnet_results/pred_instances_3d.pkl \
  --save_dir tools/analysis_tools/matrix_mvxnet \
  --kitti_eval_classes \
  --difficulty 1 \
  --score-thr 0.5 \
  --tp-iou-thr 0.5 \
  --normalize
```
*(Saves confusion matrix images and outputs `metrics.txt` containing class-wise F1 and recall).*

---

#### 3.5 Latency & Throughput (FPS) Measurement

Measure inference latency and throughput per sample on NVIDIA Jetson with GPU synchronization (`torch.cuda.synchronize`):

```bash
# 1. VirConv Network Latency (batch size = 1, 200 samples)
cd /VirConv/tools
python3 latency_eval.py \
  --config cfgs/models/kitti/VirConv-L_3class.yaml \
  --checkpoint /output/models/kitti/VirConv-L_3class/default/ckpt/epoch_60.pth \
  --samples 200 \
  --network vir

# 2. PENet Depth Completion Latency
python3 latency_eval.py \
  --samples 200 \
  --network pe

# 3. MMDetection3D (BEVFusion & MVX-Net) Latency & FPS
cd /mmdetection3d
python3 tools/analysis_tools/benchmark.py \
  models/bevfusion/bevfusion_kitti3_lidar-cam.py \
  ../../epoch_40.pth \
  --samples 200 \
  --fuse-conv-bn \
  --in_seconds
```
*(Results are appended into `InferenceTime.txt`).*

---

### 4. Edge Hardware Benchmarking on Jetson Orin

Hardware metrics are captured using `jetson-stats` (`jtop`) during inference to profile system power, temperature, and resource bottlenecks.

#### 4.1 Start Telemetry Logger in Background
On the Jetson device, launch the telemetry logger:
```bash
python jetson_packages/mmdetection3d/benchmark/jetson_metrics.py \
  --filepath orin_metrics.csv \
  --wait 0.1
```
This logs:
- GPU utilization (`%`)
- Per-core CPU utilization (`cpu_1` to `cpu_8`)
- GPU, CPU, and junction temperatures (`°C`)
- Total board power (`power_tot`) and Model power rail (`power_model`) in $\text{mW}$
- RAM consumption (`%`)

#### 4.2 Run Inference Benchmark
Simultaneously run model inference:
```bash
python jetson_packages/mmdetection3d/benchmark/benchmark.py \
  jetson_packages/mmdetection3d/models/bevfusion/bevfusion_kitti3_lidar-cam.py \
  epoch_40.pth \
  --samples 200 \
  --fuse-conv-bn \
  --in_seconds
```
Results (FPS and latency per sample) will be appended to `InferenceTime.txt`.

---

### 5. Analysis & Visualization

All analysis scripts are located in `visualizar_TFG/` and operate directly on the evaluation CSVs generated during testing.

#### 5.1 Robustness Analysis across Severities & Distances
Calculates the Relative Accuracy ($\text{RA}$) and mean Relative Accuracy ($\text{mRA}$) across corruption severities ($1\text{--}5$) and distance brackets ($0\text{--}30\,\text{m}$, $30\text{--}50\,\text{m}$, $50\text{--}80\,\text{m}$, $0\text{--}80\,\text{m}$):
```bash
cd visualizar_TFG
python robustez.py
```
**Generated Outputs**:
- `AP_3d_<Class>_moderate_(0,80)_severidades.png`: Evolution of $\text{AP}_{\text{3D}}$ under increasing severity.
- `AR_3d_<Class>_moderate.csv`: Numerical table of $\text{RA}$ values per condition and distance range.
- `RA_3d_<Class>_moderate_por_rango.png`: Bar plots comparing model robustness across distance ranges.

#### 5.2 Modal Redundancy Analysis (Sensor Dropout)
Evaluates how much each model depends on the camera modality by computing the Relative Corruption Error ($\text{RCE}\%$) when the camera stream is lost:
```bash
cd visualizar_TFG
python redundancia.py
```
**Generated Outputs**:
- `redundancia.png`: Multi-panel plot of $\text{RCE}\%$ across distance ranges for Cars, Pedestrians, and Cyclists ($\text{AP}_{\text{3D}}$ and $\text{AP}_{\text{AOS}}$).
- `redundancia(0,80).csv`: Summary of modal degradation over the full $0\text{--}80\,\text{m}$ operational range.

#### 5.3 Jetson Embedded Performance Visualization
Plots comparative hardware consumption profiles across MVX-Net, BEVFusion, and VirConv:
```bash
cd visualizar_TFG
python visualizacion_orin.py
```
**Generated Outputs**:
- `Orin/todos_orin.png`: 7-panel visualization of GPU usage, average CPU usage, GPU/CPU temperatures, power consumption (Total and Model), and RAM %.
- `Orin/resumen_orin.csv`: Summary table with average and peak resource utilization per model.

#### 5.4 Training Convergence Plots
To visualize training progression across epochs:
```bash
cd resultados
python plot_training_curve.py
```

---

## Evaluation Metrics & Formulation

### 1. Detection Performance (KITTI AP40)
Evaluated across classes (**Car**, **Pedestrian**, **Cyclist**) and difficulty levels (**Easy**, **Moderate**, **Hard**) for:
- $\text{AP}_{\text{3D}}$: 3D bounding box Average Precision.
- $\text{AP}_{\text{AOS}}$: Average Orientation Similarity (heading accuracy).
- $\text{AP}_{\text{BEV}}$: Bird's-Eye-View 2D bounding box Average Precision.

### 2. Distance Segmentation
Objects are filtered based on their depth $z_{\text{cam}}$:
$$\text{Near: } z \in [0, 30]\,\text{m}, \quad \text{Mid: } z \in [30, 50]\,\text{m}, \quad \text{Far: } z \in [50, 80]\,\text{m}, \quad \text{Overall: } z \in [0, 80]\,\text{m}$$

### 3. Relative Accuracy (RA)
Measures the retention of performance under a corruption $c$ at severity $s$ relative to clean baseline data ($s=0$):
$$\text{RA}_{c, s} = \frac{\text{AP}_{c, s}}{\text{AP}_{\text{clean}}}$$

The mean Relative Accuracy for a corruption $c$ is averaged across severities $s \in \{1, 3, 5\}$:
$$\text{RA}_{c} = \frac{1}{3} \sum_{s \in \{1, 3, 5\}} \text{RA}_{c, s}, \qquad \text{mRA} = \frac{1}{|C|} \sum_{c \in C} \text{RA}_{c}$$

### 4. Relative Corruption Error (RCE%)
Measures the percentage drop in detection accuracy when a specific sensor modality is lost (e.g., $c = \text{no-camera}$):
$$
\text{RCE}\% = \frac{\text{AP}_{\text{clean}} - \text{AP}_{\text{no-camera}}}{\text{AP}_{\text{clean}}} \times 100\%
$$
- **Low $\text{RCE}\%$**: High sensor redundancy (the model maintains high accuracy even when a sensor fails).
- **High $\text{RCE}\%$**: High sensor dependency (the model severely degrades without that modality).

---

## Checkpoints Included

The repository includes pre-trained model weights ready for testing:

| Checkpoint File | Model Architecture | Framework | Description |
| :--- | :--- | :--- | :--- |
| `checkpoint_epoch_60.pth` | **VirConv-L** | OpenPCDet | 60-epoch trained VirConv checkpoint |
| `epoch_40.pth` | **BEVFusion** | MMDetection3D | 40-epoch trained LiDAR-Camera BEVFusion checkpoint |
| `epoch_20.pth` | **BEVFusion / Baselines** | MMDetection3D | Intermediate checkpoint |
| `epoch_15.pth` | **MVX-Net** | MMDetection3D | Trained MVX-Net multimodal checkpoint |

---

## Citations & Tool References

This project was developed as part of a Bachelor's Thesis on multimodal 3D object detection for autonomous driving and robotics, conducted at **Ikerlan**:

Please cite the corresponding tools, models, benchmarks, and libraries utilized throughout this project:

---

  ```bibtex
  @inproceedings{geiger2012we,
    title={Are we ready for autonomous driving? The KITTI vision benchmark suite},
    author={Geiger, Andreas and Lenz, Philip and Urtasun, Raquel},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages={3354--3361},
    year={2012}
  }

  @inproceedings{he2023benchmarking,
    title={Benchmarking Robustness of 3D Object Detection to Common Corruptions in Autonomous Driving},
    author={He, Jiawei and Chen, Charlie and Gao, Huan-ang and Cao, Yuxuan and Wang, Zhaoxi and Ding, Jianmin and Dong, Yinpeng and Zhu, Jun},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages={21973--21982},
    year={2023}
  }
  
  - Repository: [azureology/kitti-velo2cam](https://github.com/azureology/kitti-velo2cam)

  @inproceedings{liu2023bevfusion,
    title={BEVFusion: Multi-Task Multi-Sensor Fusion with Unified BEV Representation},
    author={Liu, Zhijian and Tang, Haotian and Amini, Alexander and Yang, Xingyu and Mao, Huizi and Rus, Daniela and Han, Song},
    booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
    pages={2774--2781},
    year={2023}
  }

  @inproceedings{sindagi2019mvx,
    title={MVX-Net: Multimodal VoxelNet for 3D Object Detection},
    author={Sindagi, Vishwanath A and Zhou, Yin and Tuzel, Oncel},
    booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
    pages={7276--7282},
    year={2019}
  }

  @inproceedings{wu2023virtual,
    title={Virtual Sparse Convolution for Multimodal 3D Object Detection},
    author={Wu, Hai and Deng, Chenglu and Rao, Jintao and Zhao, Jie and Lu, Jiwen and Zhou, Jie},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages={21653--21662},
    year={2023}
  }
  

  @misc{mmdet3d2020,
    title={{MMDetection3D}: OpenMMLab next-generation platform for general 3D object detection},
    author={MMDetection3D Contributors},
    howpublished={\url{https://github.com/open-mmlab/mmdetection3d}},
    year={2020}
  }
  

  @misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Cloud},
    author={OpenPCDet Development Team},
    howpublished={\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
  }
  ```
