#---
# name: VirConv
# alias: virconv
# depends: [l4t-pytorch, numba]
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Jeston arquitecture required by cumm (e.g. 8.7 for Jetson Orin NX)
ENV CUMM_CUDA_ARCH_LIST=${CUDA_ARCH_LIST}

# Uninstall conflicting dependencies
RUN apt-get remove -y python3-blinker \
    && apt-get autoremove -y

RUN pip3 install --upgrade pip \
    && pip3 uninstall -y docutils readme-renderer

# Install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir easydict pyyaml scikit-image tqdm scipy pyyaml fire shapely matplotlib addict pyquaternion awscli open3d pandas future pybind11 Cython prefetch-generator keyring readme-renderer jupyter jetson-stats scikit-learn

# Install spconv and cumm
COPY spconv/pyproject.toml pyproject.toml 
COPY spconv/setup.py setup.py
COPY spconv/install_spconv.sh install_spconv.sh 

RUN bash install_spconv.sh 

# Setup VirConv 
COPY ./VirConv /VirConv

WORKDIR /VirConv

RUN python3 setup.py develop
