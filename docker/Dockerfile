FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Just in case we need it
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends git curl wget \
      git tmux vim g++ net-tools iputils-ping openssh-server lsb-release

# ==========> INSTALL ROS noetic <=============
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
    && curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add - \
    && echo "deb http://packages.ros.org/ros/ubuntu focal main" | tee /etc/apt/sources.list.d/ros-focal.list

RUN apt-get update && apt-get install -y ros-noetic-desktop ros-noetic-ros-numpy \
	ros-noetic-vision-msgs ros-noetic-catkin python3-catkin-tools \
	python3-rosdep python3-rosinstall python3-rosinstall-generator \
	python3-wstool build-essential clang clang-tidy libbenchmark-dev \
	libgoogle-glog-dev libblas-dev libboost-serialization-dev libboost-filesystem-dev \
       	libboost-iostreams-dev libboost-system-dev libflann-dev libglew-dev \
       	libgtest-dev libopenni-dev libopenni2-dev libpcap-dev libproj-dev \
       	libqhull-dev libqt5opengl5-dev libusb-1.0-0-dev lsb-release wget \
       	python3-pip python3-setuptools libpcl-dev ros-noetic-octomap ros-noetic-octomap-ros \
       	ros-noetic-eigen-conversions ros-noetic-tf-conversions ros-noetic-tf2-bullet\
       	unzip git ros-noetic-grid-map-core ros-noetic-grid-map-ros  \
       	ros-noetic-grid-map-rviz-plugin ros-noetic-rviz ros-noetic-pcl-ros \
       	ros-noetic-costmap-2d ros-noetic-navfn ros-noetic-base-local-planner \
       	ros-noetic-clear-costmap-recovery ros-noetic-move-base-msgs \
       	ros-noetic-rotate-recovery 

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc \
    && echo "alias sds="source devel/setup.bash"" >> ~/.bashrc

# 设置全局索引 URL 为阿里云的 PyPI 镜像
RUN pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip3 config set install.trusted-host mirrors.aliyun.com

RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 \ 
    && pip3 install scipy open3d scikit-image alpha-transform

ARG DEBIAN_FRONTEND=noninteractive

##############################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install dependencies
RUN apt-get update && \
    apt-get install -y git ninja-build cmake build-essential libopenblas-dev xterm gedit \ 
    xauth openssh-server tmux wget mate-desktop-environment-core \
    && python3 -m pip install --upgrade pip && apt remove python3-pip -y

# For faster build, use more jobs.
ENV MAX_JOBS=4
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && cd MinkowskiEngine && python setup.py install --force_cuda --blas=openblas \
    && pip install --ignore-installed blinker

COPY requirements_all.txt requirements_all.txt
RUN pip install -r requirements_all.txt

WORKDIR /home/RangePlace
ENV PYTHONPATH="/home/RangePlace"
