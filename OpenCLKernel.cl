kernel void add_grids_normal(__global const float* grid1, __global const float* grid2, __global float* result, int width, int height) {
    int i = get_global_id(0);  // 获取 x 方向的全局 ID
    int j = get_global_id(1);  // 获取 y 方向的全局 ID

    if (i < width && j < height) {
        int index = j * width + i;  // 计算一维索引
        result[index] = grid1[index] + grid2[index];
    }
}

kernel void add_grids_texture(read_only image2d_t grid1, read_only image2d_t grid2, write_only image2d_t result) {
    int i = get_global_id(0);  // 获取 x 方向的全局 ID
    int j = get_global_id(1);  // 获取 y 方向的全局 ID
    int height = get_image_height(grid1);
    int width = get_image_width(grid1);
    
    int2 coords = (int2)(i, j);  // 构建二维坐标

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    if(coords.x < width && coords.y < height){
        float4 val1 = read_imagef(grid1, sampler, coords);  // 从 grid1 读取值
        float4 val2 = read_imagef(grid2, sampler, coords);  // 从 grid2 读取值
        float4 res = val1 + val2;  // 相加操作
    
    write_imagef(result, coords, res);  // 将结果写入 result 图像对象
    }
}


kernel void backproject(read_only image2d_t sinogram, write_only image2d_t reco, int num_projections, 
        int detector_size, float angular_increment_degree, float detector_spacing, 
        float detector_origin, int reco_sizeX, int reco_sizeY,
        float reco_originX, float reco_originY, float reco_spacingX, 
        float reco_spacingY) {

    //定义sampler
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    
    // 重建图像像素的坐标
    int i = get_global_id(0);
    int j = get_global_id(1);

    int2 pixelCoords = (int2)(i, j);

    float sum = 0.0;
    for(int k = 0; k < num_projections; k++){
        //求当前角度
        float angle = (k + 0.5f) * angular_increment_degree * M_PI/180;

        //得到（i,j)的物理坐标
        float physical_x = i * reco_spacingX + reco_originX;
        float physical_y = j * reco_spacingY + reco_originY;
        

        //计算s
        float s = physical_y * cos(angle + M_PI/2) + physical_x * sin(angle + M_PI/2);

        //获取对应值的坐标
        float angle_index = k;
        float detector_index = (s - detector_origin) / detector_spacing;

        //获得对应值
        float2 sinogram_index = (float2) (detector_index + 0.5f, angle_index + 0.5f);
        float value = read_imagef(sinogram, sampler, sinogram_index).x;

        sum += value;
    }

    write_imagef(reco, pixelCoords, sum);
}