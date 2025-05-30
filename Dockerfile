ARG CUDA_VERSION=12.4.1
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.5.1
ARG FC_VERSION=HEAD
ARG BUILD_GTEST=OFF  # Default value for GTest (can be changed during build)

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Asia/Shanghai

##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}
ENV WORK_DIR=/workspace
RUN mkdir -p ${WORK_DIR}
##############################################################################
# Install Basic Utilities
##############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    echo "N" | apt-get install -y --no-install-recommends pkg-config && \
    apt-get install -y --no-install-recommends \
        autotools-dev \
        build-essential \
        ca-certificates ccache cmake curl \
        gcc git g++ \
        htop \
        iftop iotop \
        libcairo2-dev libfontconfig-dev libibverbs1 libibverbs-dev libnuma-dev libx11-dev lsb-release \
        net-tools nfs-common ninja-build \
        openssh-server openssh-client \
        pdsh psmisc \
        rsync \
        screen software-properties-common sudo \
        tmux tzdata \
        unzip \
        vim \
        openssh-server openssh-client \
        wget && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd && \

    echo "Port 3217" >> /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    echo "UseDNS no" >> /etc/ssh/sshd_config && \
    echo "Port 3217" >> /etc/ssh/ssh_config && \
    echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config && \
    echo "UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    ssh-keygen -A && \
    mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh && \
    ssh-keygen -t rsa -f /root/.ssh/id_rsa -q -N "" && \
    cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys && \
    echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    echo '#!/bin/sh\n/etc/init.d/ssh start\n/bin/bash' > /start.sh && \
    chmod +x /start.sh

EXPOSE 3217

CMD ["/start.sh"]

##############################################################################
# Install Python
##############################################################################
RUN add-apt-repository ppa:deadsnakes/ppa && \
        apt-get update -y && \
        apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv && \
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
        update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} && \
        ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config && \
        ln -s /usr/bin/python3 /usr/bin/python && \
        apt install python3-distutils python3-pip -y && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

##############################################################################
# Install CUDNN
##############################################################################
ENV NV_CUDNN_VERSION 9.1.0.70-1
ENV NV_CUDNN_PACKAGE_NAME libcudnn9-cuda-12
ENV NV_CUDNN_PACKAGE libcudnn9-cuda-12=${NV_CUDNN_VERSION}
ENV NV_CUDNN_PACKAGE_DEV libcudnn9-dev-cuda-12=${NV_CUDNN_VERSION}
RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*

##############################################################################
# SCCACHE
##############################################################################
ENV SCCACHE_BUCKET_NAME=flagscale-build-sccache
ENV SCCACHE_REGION_NAME=cn-north-1
ENV SCCACHE_S3_NO_CREDENTIALS=0
RUN cd ${STAGE_DIR} && \
        echo "Installing sccache..." && \
        curl -L -o sccache.tar.gz https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-x86_64-unknown-linux-musl.tar.gz && \
        tar -xzf sccache.tar.gz && \
        mv sccache-v0.8.1-x86_64-unknown-linux-musl/sccache /usr/bin/sccache && \
        rm -rf sccache.tar.gz sccache-v0.8.1-x86_64-unknown-linux-musl

##############################################################################
# OPENMPI
##############################################################################
ENV OPENMPI_BASEVERSION=4.1
ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.6
RUN cd ${STAGE_DIR} && \
        wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION}.tar.gz | tar xzf - && \
        cd openmpi-${OPENMPI_VERSION} && \
        ./configure --prefix=/usr/local/openmpi-${OPENMPI_VERSION} && \
        make -j"$(nproc)" install && \
        ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi && \
        # Sanity check:
        test -f /usr/local/mpi/bin/mpic++ && \
        cd .. && \
        rm -r openmpi-${OPENMPI_VERSION}
ENV PATH=/usr/local/mpi/bin:${PATH} \
        LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}
# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
        echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
        echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
        chmod a+x /usr/local/mpi/bin/mpirun

##############################################################################
# Install Miniconda
##############################################################################
RUN cd ${STAGE_DIR} && \
        mkdir -p ~/miniconda3 && \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
        rm -rf ~/miniconda3/miniconda.sh && \
        ~/miniconda3/bin/conda init bash && \
        ~/miniconda3/bin/conda config --set auto_activate_base false && \
        ~/miniconda3/bin/conda config --set default_python ${PYTHON_VERSION}

##############################################################################
# Clone FlagCX and Build with USE_NVIDIA=1 and GTEST
##############################################################################
RUN cd ${WORK_DIR} && \
    git clone https://github.com/FlagOpen/FlagCX.git && \
    cd FlagCX && \
    git checkout $FC_VERSION && \
    git submodule update --init --recursive && \
    make USE_NVIDIA=1

##############################################################################
# Final Cleanup
##############################################################################
RUN rm -rf ${STAGE_DIR}/*
