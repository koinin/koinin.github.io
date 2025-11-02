---
title: "Docker Cuda"
date: "2025-03-19T13:24:47+08:00"
draft: false
---

没想到这么久以后又遇到了cuda的问题，我再次记录一下我的理解

```sh
export CUDA_VISIBLE_DEVICES=0,1
CUDA_VISIBLE_DEVICES=0,1 python test.py
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple some-package

python -c "import torch; print(torch.__version__)"
```

# 宿主机cuda

首先你的宿主机的cuda结构应该是 cuda-driver -> cuda toolkit -> pytorch-cuda

第一层

一般服务器是装好了的

第二层

[CUDA Toolkit 11.8 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local)

```sh
nvcc -V
# 一般来说安装在/usr/local/下
# 可以搜索一下，然后把目标加入环境变量
```



第三层

[PyTorch](https://pytorch.org/)

# Docker-cuda

起因是想解决centos编译库太老旧的问题，把一些编译环境放在docker里面

编译这个flash_attn要编译好久好久



```sh
sudo docker run -it -d \
  --gpus all \
  --name my_cuda \
  --network host \
  -v /home/sunyatao_B:/root/sunyatao_B \
  --shm-size=1g \
  nvidia/cuda:11.8.0-runtime-ubuntu22.04 bash
  
sudo docker exec -it my_cuda bash

```

