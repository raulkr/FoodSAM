# Use a Python 3.7 base image
FROM python:3.7-slim-buster

# Set the working directory in the container
WORKDIR /FoodSAM

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the FoodSAM directory contents
COPY . /FoodSAM

# Install PyTorch and torchvision (CPU versions)
RUN pip install torch==1.8.1 torchvision==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install MMCV (CPU version)
RUN pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html

# Install SAM
RUN pip install git+https://github.com/facebookresearch/segment-anything.git@6fdee8f

# Install MMSegmentation (CPU version)
RUN pip install mmsegmentation==0.18.0 -f https://download.openmmlab.com/mmsegmentation/dist/index.html

# Install other requirements
RUN pip install -r requirement.txt

# Set the default command to python
CMD ["python"]
