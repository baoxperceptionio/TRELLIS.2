FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG INSTALL_FLASH_ATTN=0
ARG INSTALL_NVDIFFRAST=0
ARG INSTALL_NVDIFFREC=0
ARG INSTALL_CUMESH=0
ARG INSTALL_FLEXGEMM=0
ARG INSTALL_OVOXEL=0

ENV CUDA_HOME=/usr/local/cuda \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST=9.0 \
    FLASH_ATTN_CUDA_ARCHS=90 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    NVCC_THREADS=4 \
    MAX_JOBS=8 \
    FORCE_CUDA=1 \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libjpeg-dev \
        libegl-dev \
        libgl-dev \
        libsm6 \
        libx11-dev \
        libxext6 \
        libxrender1 \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python3 -m venv ${VIRTUAL_ENV} \
    && python -m pip install --upgrade pip setuptools wheel packaging

RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${PYTORCH_INDEX_URL}

RUN pip install \
        imageio \
        imageio-ffmpeg \
        tqdm \
        easydict \
        opencv-python-headless \
        ninja \
        trimesh \
        transformers \
        gradio==6.0.1 \
        tensorboard \
        pandas \
        lpips \
        zstandard \
        kornia \
        timm \
        plyfile \
        psutil \
        numpy \
        scipy \
        scikit-image \
        huggingface-hub \
    && (pip install pillow-simd || pip install pillow) \
    && pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

RUN if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then \
        pip install flash-attn==2.7.3 --no-build-isolation; \
    fi

RUN if [ "${INSTALL_NVDIFFRAST}" = "1" ]; then \
        git clone -b v0.4.0 --depth=1 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast \
        && pip install /tmp/nvdiffrast --no-build-isolation \
        && rm -rf /tmp/nvdiffrast; \
    fi

RUN if [ "${INSTALL_NVDIFFREC}" = "1" ]; then \
        git clone -b renderutils --depth=1 https://github.com/JeffreyXiang/nvdiffrec.git /tmp/nvdiffrec \
        && pip install /tmp/nvdiffrec --no-build-isolation \
        && rm -rf /tmp/nvdiffrec; \
    fi

RUN if [ "${INSTALL_CUMESH}" = "1" ]; then \
        git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh \
        && pip install /tmp/CuMesh --no-build-isolation \
        && rm -rf /tmp/CuMesh; \
    fi

RUN if [ "${INSTALL_FLEXGEMM}" = "1" ]; then \
        git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM \
        && pip install /tmp/FlexGEMM --no-build-isolation \
        && rm -rf /tmp/FlexGEMM; \
    fi

COPY o-voxel ./o-voxel

RUN if [ "${INSTALL_OVOXEL}" = "1" ]; then \
        if [ ! -f o-voxel/third_party/eigen/CMakeLists.txt ]; then \
            rm -rf o-voxel/third_party/eigen \
            && git clone --depth=1 https://gitlab.com/libeigen/eigen.git o-voxel/third_party/eigen; \
        fi \
        && pip install ./o-voxel --no-build-isolation; \
    fi

RUN pip install \
        transformers==4.57.1 \
        huggingface-hub==0.36.2 \
        requests==2.34.2

COPY . .

EXPOSE 7860

CMD ["bash"]
