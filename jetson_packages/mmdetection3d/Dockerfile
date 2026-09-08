#---
# name: mmdetection3d
# alias: mmdet3d
# depends: [l4t-pytorch, numba]
#---

    ARG BASE_IMAGE
    FROM ${BASE_IMAGE}

    #ENV GITHUB_TOKEN=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

    # Jeston arquitecture required by cumm (e.g. 8.7 for Jetson Orin NX)
    ENV CUMM_CUDA_ARCH_LIST=${CUDA_ARCH_LIST}  

    # Install  required packages
    RUN apt-get update \
        && apt-get install -y ffmpeg libsm6 libxext6 git libglib2.0-0 libsm6 libxrender-dev libxext6 \
        && apt-get remove -y python3-blinker \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

    # Install spconv and cumm to use it as default on mmdetection3d
    # will take up less GPU memory than using the default mmcv spconv
    COPY spconv/pyproject.toml pyproject.toml 
    COPY spconv/setup.py setup.py
    COPY spconv/install_spconv.sh install_spconv.sh 

    RUN bash install_spconv.sh 
  
    # Install MMEngine, MMCV and MMDetection
    RUN pip3 install --upgrade pip && \
        pip3 install --no-cache-dir openmim seaborn pandas matplotlib jupyter jetson-stats setuptools scikit-learn && \
        mim install "mmengine>=0.7.1,<1.0.0" "mmcv>=2.0.0rc4,<2.2.0" "mmdet>=3.0.0,<3.3.0"
    
    # Install MMDetection3D
    RUN git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x /mmdetection3d \
        && cd /mmdetection3d \
        && git submodule update --init --recursive  \
        && pip3 install --no-cache-dir -e .
        
    # Add changes to the repository
    COPY work_dirs /mmdetection3d/work_dirs
    
    COPY benchmark/kitti_metric.py /mmdetection3d/mmdet3d/evaluation/metrics/kitti_metric.py
    COPY benchmark/eval.py /mmdetection3d/mmdet3d/evaluation/functional/kitti_utils/eval.py
    COPY benchmark/benchmark.py /mmdetection3d/tools/analysis_tools/benchmark.py 
    COPY benchmark/confusion_matrix.py /mmdetection3d/tools/analysis_tools/confusion_matrix.py 
    COPY benchmark/test_multiple_ckpt.py /mmdetection3d/tools/test_multiple_ckpt.py
   
    COPY models/bevfusion/setup.py /mmdetection3d/projects/BEVFusion/setup.py 
    COPY models/bevfusion/bevfusion_kitti3_lidar-cam.py /mmdetection3d/projects/BEVFusion/configs/bevfusion_kitti3_lidar-cam.py 
    COPY models/bevfusion/bevfusion_kitti3_lidar.py /mmdetection3d/projects/BEVFusion/configs/bevfusion_kitti3_lidar.py 
    COPY models/bevfusion/loading.py /mmdetection3d/projects/BEVFusion/bevfusion/loading.py 
    
    COPY models/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py /mmdetection3d/configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py
    COPY models/mvxnet/cosine.py /mmdetection3d/configs/_base_/schedules/cosine.py
    
    WORKDIR /mmdetection3d
    RUN python3 projects/BEVFusion/setup.py develop


