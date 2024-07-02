from Grid import Grid
import pyconrad as pyc
import math
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt


def create_sinogram(phantom, number_of_projections, detector_spacing, detector_size, scan_range):
    #计算角度增量
    angular_increment = scan_range / number_of_projections

    #创建sinogram
    sinogram = Grid(number_of_projections, detector_size,[angular_increment, detector_spacing])

    #设置 sinogram origin
    sinogram.set_origin([0, (detector_size - 1) * detector_spacing / 2])

    #for loop over angular range
    for projection_idx in range(number_of_projections):
        #当前旋转后角度
        angle = projection_idx * angular_increment

        #for loop over detector size
        for detector_idx in range(detector_size):

            # Point on the line parallel to the detector through the origin of the phantom
            detector_position = (detector_idx - (detector_size - 1) / 2) * detector_spacing

            #计算采样点的位置坐标
            x_center =detector_position * np.cos(angle * np.pi / 180 + np.pi / 2)
            y_center = detector_position * np.sin(angle * np.pi / 180 + np.pi / 2)

            #采样长度
            sample_length = int(np.sqrt(np.power(phantom.width, 2) + np.power(phantom.height, 2))*0.5)
            ray_sum = 0.0

            # 根据探测器间距定义采样数(delta_t)
            delta_t = 0.5 * detector_spacing

            # for loop to sample along the X ray with a sampling distance delta_t
            for sample_idx in np.arange(-sample_length, sample_length, delta_t):
                # 获取插值值
                x = x_center - sample_idx * np.sin(angle * np.pi / 180 + np.pi / 2)
                y = y_center + sample_idx * np.cos(angle * np.pi / 180 + np.pi / 2)
                sample_value = phantom.get_at_physical(x, y)

                # integral is a sum over samples, afterwards the sum need to be multiplied by delta_t
                ray_sum += sample_value * delta_t

            # set the value to the sinogram
            sinogram.set_at_index(projection_idx, detector_idx, ray_sum)
    return sinogram


def backproject(sinogram, size_x, size_y, grid_spacing):
    reco = Grid(size_x, size_y, grid_spacing)
    #set origin
    sinogram.set_origin([0, -(sinogram.width - 1) * sinogram.spacing[1] / 2])
    num_projections = sinogram.height

    for i in range(size_x):
        for j in range(size_y):
            #calculate word coordinate (x,y)
            physical_ccordinate = reco.index_to_physical(i, j)
            bp_sum = 0.0

            for k in range(0, num_projections + 1):
                #calculate rotation angle
                angle = k * sinogram.spacing[0]
                #calculate physical detector position s
                detector_position = physical_ccordinate[0] * np.cos(np.deg2rad(angle) + np.pi / 2) + physical_ccordinate[1] * np.sin(np.deg2rad(angle) + np.pi / 2)
                # read the corresponding sinogram value and add the value to the reconstruction pixel value at x_i, y_j
                value = sinogram.get_at_physical(angle, detector_position)

                bp_sum += value
            reco.set_at_index(i, j, bp_sum)
    return reco


def ramp_filter(sinogram, detector_spacing):
    num_projections, detector_size = sinogram.height, sinogram.width

    #zero padding后应得的信号全长
    full_length_signal= next_power_of_two(detector_size)

    #initialize ramp kernel
    ramp_kernel = np.zeros(full_length_signal)

    #calculate frequency spacing
    frequency_spacing = 1 / (detector_spacing * full_length_signal)

    #first half ramp filter，Initialize in Fourier domain
    for i in range(full_length_signal//2 + 1):
        frequency = i * frequency_spacing
        ramp_kernel[i] = 2 * np.abs(frequency)

    #latter half ramp filter(mirror the first half)
    ramp_kernel[full_length_signal//2 +1 :] = ramp_kernel[1 : full_length_signal//2][::-1]

    #初始化滤波后的sinogram
    filtered_sinogram = Grid(sinogram.height, sinogram.width, sinogram.spacing)

    #获取sinogram的结构信息
    sinogram_array = sinogram.get_buffer()

    for i in range(num_projections):
        #Zero padding
        zero_padded_signal = np.pad(sinogram_array[i], (0, full_length_signal - detector_size), 'constant')

        #Fourier transform of the zero padded signal
        projection_fft = np.fft.fft(zero_padded_signal)

        #Filtering in Fourier domain
        filtered_projection_fft = projection_fft * ramp_kernel

        #Inverse Fourier transform to get back to spatial domain
        filtered_projection = np.fft.ifft(filtered_projection_fft).real

        #store the filtered projection back to the sinogram
        filtered_sinogram.get_buffer()[i] = filtered_projection[:detector_size]

    return filtered_sinogram


def ramlak_filter(sinogram, detector_spacing):
    num_projections, detector_size = sinogram.height, sinogram.width

    # zero padding后应得的信号全长
    full_length_signal = next_power_of_two(detector_size)

    # 初始化滤波后的sinogram
    filtered_sinogram = Grid(sinogram.height, sinogram.width, sinogram.spacing)

    # initialize ramp kernel
    ramlak_kernel = np.zeros(full_length_signal)


    # Initialize the Ram-Lak filter in spatial domain
    for n in range(-full_length_signal//2, full_length_signal//2 + 1):
        if n == 0:
            ramlak_kernel[n] = 1 / (4 * detector_spacing ** 2)
        elif n % 2 != 0:
            ramlak_kernel[n] = -1 / (np.pi * n * detector_spacing) ** 2
        else:
            ramlak_kernel[n] = 0

    # 获取sinogram的结构信息
    sinogram_array = sinogram.get_buffer()

    for i in range(num_projections):
        # Zero-padding using np.pad
        zero_padded_signal = np.pad(sinogram_array[i], (0, full_length_signal - detector_size), 'constant')

        # Convolution in spatial domain using FFT
        ramlak_kernel_fft = np.fft.fft(ramlak_kernel)
        projection_fft = np.fft.fft(zero_padded_signal)

        filtered_projection_fft = projection_fft * ramlak_kernel_fft

        # Inverse Fourier transform to get back to spatial domain
        filtered_projection = np.fft.ifft(filtered_projection_fft).real

        # Store the filtered projection back to the sinogram
        filtered_sinogram.get_buffer()[i] = filtered_projection[:detector_size]

    return filtered_sinogram


def next_power_of_two(value):
    if is_power_of_two(value):
        return value * 2
    else:
        i = 2
        while i <= value:
            i *= 2
        return i * 2


def is_power_of_two(k):
    return k and not k & (k - 1)


def create_fanogram(phantom, number_of_projections, detector_spacing, detector_size, angular_increment, d_si, d_sd):

    fanogram = Grid(number_of_projections,detector_size, [angular_increment, detector_spacing])

    fanogram.set_origin([0, -(detector_size - 1) * detector_spacing / 2])

    d_id = d_sd - d_si

    for beta_idx in range(0, number_of_projections):

        beta = beta_idx * angular_increment
        sin_beta = np.sin(beta * np.pi / 180 + np.pi / 2)
        cos_beta = np.cos(beta * np.pi / 180 + np.pi / 2)

        s = [-d_si * sin_beta, d_si * cos_beta]
        m = [d_id * sin_beta, -d_id * cos_beta]

        for t in range(0, detector_size):
            MP = [(t - (detector_size - 1)/2) * detector_spacing * cos_beta, (t - (detector_size - 1)/2) * detector_spacing * sin_beta]

            P = [m[0] + MP[0], m[1] + MP[1]]
            SP = [P[0] - s[0], P[1] - s[1]]
            SP_length = np.sqrt(SP[0] ** 2 + SP[1] ** 2)

            step_size = detector_spacing
            number_of_steps = math.ceil(SP_length / step_size)

            ray_sum = 0
            for i in range(0, number_of_steps):
                curr_point = [s[0] + i * step_size * SP[0] / SP_length, s[1] + i * step_size * SP[1] / SP_length]
                val = phantom.get_at_physical(curr_point[0], curr_point[1])
                ray_sum += val * step_size

            fanogram.set_at_index(beta_idx, t, ray_sum)

    return fanogram

def rebinning(fanogram, d_si, d_sd):
    detector_spacing = 0.5
    detector_size = 128
    angular_increment = 1
    number_of_projections = 180

    sinogram = Grid(number_of_projections, detector_size, [angular_increment, detector_spacing]) #create an 180-degree sinogram
    sinogram.set_origin([0, (detector_size - 1) * detector_spacing / 2])

    for p in range(0,number_of_projections):
        theta = p * angular_increment

        for s in range(0, detector_size):
            s_world = (s - (detector_size - 1)/2) * detector_spacing
            gamma = np.arcsin(s_world / d_si) / (np.pi / 180)
            t = np.tan(gamma * np.pi/180) * d_sd
            beta = theta - gamma

            # 180-360
            if beta < 0:
                gamma = -gamma
                beta = beta + 2 * gamma + 180
                t = -t

                val = fanogram.get_at_physical(beta, t)
            else:
                val = fanogram.get_at_physical(beta, t)

            sinogram.set_at_index(p, s, val)

    return sinogram

def backproject_fanbeam(fanogram, size_x, size_y, image_spacing, d_si, d_sd):
    reco = Grid(size_x, size_y, [image_spacing, image_spacing])
    reco.set_origin([-(size_x- 1) * image_spacing / 2, -(size_y - 1) * image_spacing / 2])
    d_id = d_sd - d_si
    angular_range = 360
    angular_increment = angular_range / fanogram.get_size()[0]
    print('angular increment: ' + str(angular_increment))

    # #Cosine weighting
    for i in range(0, fanogram.get_size()[0]):
        for j in range(0, fanogram.get_size()[1]):
            [beta, t] = fanogram.index_to_physical(i, j)
            cos_weight = d_sd / np.sqrt(d_sd**2 + t**2)
            fanogram.set_at_index(i, j, fanogram.get_at_index(i, j) * cos_weight)

    for i_x in range(0, reco.get_size()[0]):
        for i_y in range(0, reco.get_size()[1]):
            [x, y] = reco.index_to_physical(i_x, i_y)
            bp_sum = 0.0
            for beta_index in range(0, fanogram.get_size()[0]):
                beta_degree = beta_index * angular_increment
                beta = np.deg2rad(beta_degree)
                source =np.array([-d_si * np.cos(beta), d_si * np.sin(beta)])

                SX = np.array([x, y]) - source
                SQ_length = SX[0] * (-np.cos(beta)) + SX[1] * np.sin(beta) # |SQ| = SX*(sinβ, −cosβ)
                SM_length = d_sd
                ratio_alpha = SM_length / SQ_length
                SP = ratio_alpha * SX
                t = SP[0] * np.sin(beta) + SP[1] * np.cos(beta)

                value = fanogram.get_at_physical(beta_degree, t)
                U = np.abs(SQ_length)/d_si
                value /= U**2 # value after distance weight
                bp_sum += value
            reco.set_at_index(i_x, i_y, bp_sum/2) #update value
    return reco


# get platforms (i.e. GPU/CPU)
platform = cl.get_platforms()
# use the GPU which is in 2nd pos
GPU = platform[0].get_devices()
# create context (with GPU)
ctx = cl.Context(GPU)
# create commandqueue to communicate between host and device
queue = cl.CommandQueue(ctx)
# load .cl file
kernel = open("OpenCLKernel.cl", "r", encoding="utf-8").read()
# Build .cl file
prg = cl.Program(ctx, kernel).build()

def addGrids_normal(grid1, grid2):
        grid1 = np.array(grid1.buffer, dtype=np.float32)
        grid2 = np.array(grid2.buffer, dtype=np.float32)
        # 创建 OpenCL 缓冲区
        mf = cl.mem_flags
        grid1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=grid1)
        grid2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=grid2)
        result_buf = cl.Buffer(ctx, mf.WRITE_ONLY, grid1.nbytes)

        # 设置内核参数并执行
        global_size = (grid1.shape[1], grid1.shape[0])
        prg.add_grids_normal(queue, global_size, None, grid1_buf, grid2_buf, result_buf, np.int32(grid1.shape[1]), np.int32(grid1.shape[0]))

        # 读取结果
        result = np.empty_like(grid1)
        cl.enqueue_copy(queue, result, result_buf).wait()
        return Grid(grid1.shape[0], grid1.shape[1], result)#将numpy再转换成Grid，以便调用get_at_index


def addGrids_texture(grid1, grid2):
        grid1 = np.array(grid1.buffer, dtype=np.float32)
        grid2 = np.array(grid2.buffer, dtype=np.float32)

        # 创建 OpenCL 图像对象
        image_format = cl.ImageFormat(cl.channel_order.INTENSITY, cl.channel_type.FLOAT)
        grid1_img = cl.image_from_array(ctx, grid1, 1, mode='r', norm_int=False)
        grid2_img = cl.image_from_array(ctx, grid2, 1, mode='r', norm_int=False)
        result_img = cl.Image(ctx, cl.mem_flags.WRITE_ONLY, image_format, shape=(grid1.shape[1], grid1.shape[0]))

        # 设置内核参数并执行
        global_size = (grid1.shape[1], grid1.shape[0])
        prg.add_grids_texture(queue, global_size, None, grid1_img, grid2_img, result_img)
        
        # 读取结果
        result = np.empty_like(grid1)
        cl.enqueue_copy(queue, result, result_img, origin=(0, 0), region=(grid1.shape[1], grid1.shape[0])).wait()
        return Grid(grid1.shape[0], grid1.shape[1], result)#将numpy再转换成Grid，以便调用get_at_index


def backprojectOpenCL(sinogram, size_x, size_y, spacing):

    # 初始化
    reco_grid = Grid(size_y, size_x, spacing)
    reco_grid_buffer = np.zeros((size_y, size_x), dtype=np.float32)
    reco_grid.set_buffer(reco_grid_buffer)
    
    result_array = np.zeros((size_y, size_x), dtype=np.float32)

    # 设置原点
    [origin_x, origin_y] = (0, -(sinogram.width - 1) * sinogram.spacing[1] / 2)
    sinogram.set_origin([origin_x, origin_y])

    # texture buffer
    texture_sinogram = cl.image_from_array(ctx, sinogram.get_buffer().astype(np.float32),  mode="r")
    texture_reco = cl.image_from_array(ctx, reco_grid.get_buffer().astype(np.float32), mode="w")

    # 参数
    num_projections = np.int32(sinogram.height)
    detector_size = np.int32(sinogram.width)
    angular_increment_degree = np.float32(sinogram.spacing[0])
    detector_spacing = np.float32(sinogram.spacing[1])
    detector_origin = np.float32(-(detector_size - 1) * detector_spacing / 2)

    reco_sizeX = np.int32(size_x)
    reco_sizeY = np.int32(size_y)

    reco_originX = np.float32(-(size_x - 1) * spacing[1] / 2)
    reco_originY = np.float32(-(size_y - 1) * spacing[0] / 2)

    reco_spacingX = np.float32(spacing[1])
    reco_spacingY = np.float32(spacing[0])

    prg.backproject(queue, (size_y,size_x), None, texture_sinogram,texture_reco,num_projections,
        detector_size,angular_increment_degree,detector_spacing,
        detector_origin,reco_sizeX,reco_sizeY,
        reco_originX,reco_originY,reco_spacingX,
        reco_spacingY)


    # 从gpu复制到cpu
    cl.enqueue_copy(queue, result_array, texture_reco, origin=(0, 0), region=texture_reco.shape)

    reco_grid.set_buffer(result_array)

    return reco_grid