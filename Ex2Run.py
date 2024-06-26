
import Methods
from Grid import Grid
from phantom import phantom
import time
import matplotlib.pyplot as plt
shepp = phantom(n=64, p_type='Modified Shepp-Logan', ellipses=None)
grid_size = [64, 64]
grid_spacing = [0.5, 0.5]
sheppGrid = Grid(grid_size[0], grid_size[1], grid_spacing)
sheppGrid.set_buffer(shepp)

# sinogram parameters
number_of_projections = 180
detector_spacing = 0.5
detector_size = 96
scan_range_in_degree = 180

# create sinogram
t0 = time.time()
sinogram = Methods.create_sinogram(sheppGrid, number_of_projections, detector_spacing, detector_size, scan_range_in_degree)
t1 = time.time()
print('sinogram:', t1-t0)
plt.subplot(231)
plt.imshow(sinogram.buffer)
plt.gray()
plt.title('original sinogram')

# creat original reco
t0 = time.time()
reco = Methods.backproject(sinogram, grid_size[0], grid_size[1], grid_spacing)
t1 = time.time()
print('backprojection:', t1-t0)
plt.subplot(234)
plt.imshow(reco.buffer)
plt.gray()
plt.title('original reco')

# ramp filter sinogram
t0 = time.time()
sinogram_filtered = Methods.ramp_filter(sinogram, sinogram.get_spacing()[1])
t1 = time.time()
print('ramp filter:', t1-t0)
plt.subplot(232)
plt.imshow(sinogram_filtered.buffer)
plt.gray()
plt.title('ramp filtered sinogram')

# create ramp filtered reco
t0 = time.time()
reco_filtered = Methods.backproject(sinogram_filtered, grid_size[0], grid_size[1], grid_spacing)
t1 = time.time()
print('backprojection:', t1-t0)
plt.subplot(235)
plt.imshow(reco_filtered.buffer)
plt.gray()
plt.title('ramp filtered reco')

# ramlak filter sinogram
t0 = time.time()
ramlak_sinogram_filtered = Methods.ramlak_filter(sinogram, sinogram.get_spacing()[1])
t1 = time.time()
print('ramlak filter:', t1-t0)
plt.subplot(233)
plt.imshow(ramlak_sinogram_filtered.buffer)
plt.gray()
plt.title('ramlak filtered sinogram')

# create ramlak filtered reco
t0 = time.time()
reco_filtered = Methods.backproject(ramlak_sinogram_filtered, grid_size[0], grid_size[1], grid_spacing)
t1 = time.time()
print('backprojection:', t1-t0)
plt.subplot(236)
plt.imshow(reco_filtered.buffer)
plt.gray()
plt.title('ramlak filtered reco')

plt.tight_layout()
plt.show()

