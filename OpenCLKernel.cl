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
    int height = get_image_height(grid1)
    int width = get_image_width(grid1)
    
    int2 coords = (int2)(i, j);  // 构建二维坐标

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    if(coords.x < width && coords.y < height){
    float4 val1 = read_imagef(grid1, sampler, coords);  // 从 grid1 读取值
    float4 val2 = read_imagef(grid2, sampler, coords);  // 从 grid2 读取值
    float4 res = val1 + val2;  // 相加操作
    
    write_imagef(result, coords, res);  // 将结果写入 result 图像对象
    }
}


kernel void backproject(read only image2d t sinogram, write only image2d t reco, int num projections, 
        int detector size, float angular increment degree, float detector spacing, 
        float detector origin, int reco sizeX, int reco sizeY,
        float reco originX, float reco originY, float reco spacingX, 
        float reco spacingY) {

    //定义sampler
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    
}