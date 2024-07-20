<div>
<font color=Blue size=6>Chapter 2 JAX Numpy</font> 
</div>

`jax.numpy` 是一个提供 NumPy API 的库，但它与 NumPy 有一些关键的区别。`jax` 是 Google 开发的一个开源库，用于自动微分和机器学习的研究。`jax.numpy` 是 `jax` 库的一个模块，它提供了与 NumPy 相似的接口，但是可以生成自动微分的函数。
以下是 `jax.numpy` 的一些特点：
1. **自动微分**：`jax` 提供了自动微分的功能，这意味着你可以轻松地计算函数的导数。这对于优化问题、机器学习模型和其他需要梯度的应用非常有用。
2. **JIT编译**：`jax` 允许你使用 `@jit` 装饰器来编译函数，以提高代码的运行速度。这特别适用于计算量大的函数，如循环或递归函数。
3. **向量化和批处理**：`jax` 提供了 `vmap` 函数，允许你自动地将函数向量化和批处理，这在处理大规模数据时非常有用。
4. **兼容性**：`jax.numpy` 尽可能地与 NumPy 的接口保持一致，以便用户可以轻松地从 NumPy 切换到 `jax.numpy`。
5. **并行计算**：`jax` 支持自动并行计算，这意味着它可以在多个 CPU 和 GPU 上运行代码，而不需要用户进行复杂的配置。
要使用 `jax.numpy`，首先需要安装 `jax` 库。可以使用以下命令进行安装：
```
pip install jax
```
然后，你可以像使用 NumPy 一样使用 `jax.numpy`：
```python
import jax.numpy as jnp
x = jnp.array([1, 2, 3])
y = jnp.array([4, 5, 6])
z = x + y
print(z)
```
这将输出：
```
[5 7 9]
```
总的来说，`jax.numpy` 是一个强大的库，它提供了 NumPy 的接口和自动微分的功能，同时还支持 JIT 编译和并行计算。这使得它成为机器学习、优化和其他科学计算任务的理想选择。
