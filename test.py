import matplotlib.pyplot as plt
import Grid
import Methods

# 假设 sinogram 是一个已定义的对象
sinogram = Grid(180, 256, 1.0)

# 运行 ramlak_filter 函数并获取滤波后的 sinogram 和 ramlak_kernel
filtered_sinogram, ramlak_kernel = Methods.ramlak_filter(sinogram, 1.0)

# 绘制 ramlak_kernel 的形状
plt.plot(ramlak_kernel)
plt.title("Ram-Lak Filter Kernel")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()