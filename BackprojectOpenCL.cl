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