---
title: "CodeAgent"
date: 2026-05-09
draft: true
---

# CodeAgent

很久没写Blog了，这两天使用的所谓的Agent比较复杂，决定还是记录一下，就好像之前使用的糊里糊涂的。

其实agent本身就很简单，一个循环调用大模型大模型进行观察，决策。

```python
# 伪代码
state = 初始化()
while not state.任务完成:
    观察 = 获取环境信息(state)
    决策 = 大模型(观察)
    行动结果 = 执行决策(决策)
    state = 更新状态(state, 行动结果)


```

其他所有的cli，gui底层逻辑都是在维护这个循环
这其中存在很多的优化措施，比如上下文优化，通过subagent去做上下文分割。
缓存，ds v4的缓存就做得很好，导致他的成本极低，1m的长对话，最多能到99的缓存命中。
还有harness engineering. 为了提升模型的输出稳定性

像claude code，opencode , cursor , copilot这类都是在做这个，包括不同provider的抽象，上下文的管理，错误重试，权限管理

> opencode 通过 oh-my-openagent添加了一层套索，指定不同模型去做不同的事，可以解决token，提升能容纳的上下文。

其次opencode和claude code这类基于cli的工具由于没有自带lsp服务器，需要自己去安装，pyright。

## Opencode

这其中分享一下我的opencode使用经验，包括omo的配置，lsp配置，都可以交给oc完成，只要给他prompt就行了

omo通过一个主模型sisyphus决策，通过分析用户意图，或者explore模式路由到其他模型上。还有就是对他一些模型能力的适配，cursor就说过，一个模型的能力是有特点的，所以他们针对一个模型会有不同的优化措施，不到万不得已，不要去在对话内切换模型，一方面，调用方式会导致整个上下文管理的重构，一方面，模型的特性不同，会让模式崩溃（虽然我没什么感受就是了）。

PS:
MoE到底在做什么，在训练和推理的时候节约了哪部分显存，和dense模型的区别在哪？