## 1.Megatron相比于DeepSpeed的特性

![img](https://pic3.zhimg.com/80/v2-3f70e4435d0a7e2d92ea8eca29d82b82_720w.webp)

### Tensor并行

把一个神经网络层Tensor切成了多个小的Tensor，每个tensor放在不同的gpu。主要就是列并行、行并行。

![img](https://pic1.zhimg.com/80/v2-6ea73fd9877c26c2e37a0d08fdbf0854_720w.webp)

### Fused CUDA Kernels

![img](https://pic1.zhimg.com/80/v2-be77c1446ba060be9e010bd3fd9b99d4_720w.webp)

$x'$和$y'$需要临时保存在内存中，优化后不需要

### DataLoader

提前做tokenize、shuffle