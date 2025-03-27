+++
date = '2025-03-17T15:55:29+08:00'
draft = true
title = 'Nanogpt'

+++

# Nanogpt

https://github.com/karpathy/nanoGPT.git

## 分词器

本质上就是一个双向字典

```python
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
```

tiktoken.get_encoding

```python
# return Rncoding(class)
def get_encoding(encoding_name: str) -> Encoding:
    if encoding_name in ENCODINGS:
        return ENCODINGS[encoding_name]

    with _lock:
        if encoding_name in ENCODINGS:
            return ENCODINGS[encoding_name]

        if ENCODING_CONSTRUCTORS is None:
            _find_constructors()
            assert ENCODING_CONSTRUCTORS is not None

        if encoding_name not in ENCODING_CONSTRUCTORS:
            raise ValueError(
                f"Unknown encoding {encoding_name}. Plugins found: {_available_plugin_modules()}"
            )

        constructor = ENCODING_CONSTRUCTORS[encoding_name]
        enc = Encoding(**constructor())
        ENCODINGS[encoding_name] = enc
        return enc

```

## 梯度累计

```python
# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20

# 模拟大的batch_size
for i, (inputs, labels) in enumerate(trainloader):
    outputs = net(inputs)                   # 正向传播
    loss = criterion(outputs, labels)       # 计算损失函数
    loss = loss / accumulation_steps        # 梯度均值，损失标准化
    loss.backward()                         # 梯度均值累加，反向传播，计算梯度
    
	# 累加到指定的 steps 后再更新参数
	if (i+1) % accumulation_steps == 0:     
        optimizer.step()                    # 更新参数
        optimizer.zero_grad()               # 梯度清零
        if (i+1) % evaluation_steps == 0:
            evaluate_model()

```

## 模型加载

```python
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line


```

## 优化器和GradScalar

```python
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
optimizer.step()  # 在代码中通过 scaler.step(optimizer) 调用
optimizer.zore_grad(set_to_none= True)

权重衰减adamw里面可以 ~ L2正则化

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
# 缩放损失值

```

## Get_lr()

余弦退火调度学习率

## 上下文管理器

```python
with ctx:
            logits, loss = model(X, Y)
        # 前向，返回logits（预测）, loss（损失）， 
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
		# pytorch自动计算图中每个参数相对于损失的梯度
# 后续的梯度处理
if grad_clip != 0.0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
	# 限制梯度全局范数，防止梯度爆炸
# 优化器步骤
# 等同于条件性optimizer.step()
# 此处optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# 也就是optimize是model的一个工具
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
```

