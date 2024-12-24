---
title: "Mamba2 pre-trained model使用"
date: "2024-08-12"
---

> 参考 https://github.com/vasqu/mamba2-torch > https://huggingface.co/AntonV/mamba2-130m-av

这里用的是作者自己转换的mamba2参数模型（从原始版本转换成hf的transfromer兼容版本），并提供了一个本地包，但是安装包的时候出现一点问题，版本不符合 需要把requirements.txt中的torch triton==2.2.0 改成你现在的版本，我测试没什么问题
