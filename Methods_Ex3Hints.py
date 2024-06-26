# from Grid import Grid
# import pyconrad as pyc
# import math
# import numpy as np
# import pyopencl as cl
# import matplotlib.pyplot as plt
#
#
# def create_sinogram(phantom, number_of_projections, detector_spacing, detector_size, scan_range):
#
#     # Your create_sinogram from Ex2
#
#     return sinogram
#
#
# def backproject(sinogram, size_x, size_y, grid_spacing):
#     # Your backproject from Ex2
#
#     return reco
#
#
# def ramp_filter(sinogram, detector_spacing):
#     # Your ramp_filter from Ex2
#     return result
#
#
# def ramlak_filter(sinogram, detector_spacing):
#     # Your ramlak_filter from Ex2
#
#     return result
#
#
# def create_fanogram(phantom, number_of_projections, detector_spacing, detector_size, angular_increment, d_si, d_sd):
#     fanogram = Grid(...)
#     fanogram.set_origin(...)
#
#
#     for beta_idx in range(0, number_of_projections):
#
#         beta =
#
#         source_position =
#         source_direction =
#         detector_direction =
#
#         for s in range(0, detector_size):
#             s_world =
#
#             point_on_detector =
#
#             ray_SP =
#             SP_length =
#             ray_direction =
#
#             step_size = 0.5
#             number_of_steps = math.ceil(SP_length / step_size)
#
#             ray_sum =
#             for i in range(0, number_of_steps):
#                 curr_point =
#                 val =
#                 ray_sum =
#
#             fanogram #set new value to fanogram
#
#     return fanogram
#
#
# def backproject_fanbeam(fanogram, size_x, size_y, image_spacing, d_si, d_sd):
#     reco = Grid(...)
#     reco.set_origin(...)
#
#     d_id =
#     angular_range =
#     angular_increment =
#     print('angular increment: ' + str(angular_increment))
#     # #Cosine weighting
#     for i in range(0, fanogram.?):
#         for j in range(0, fanogram.?):
#             [beta, t] =
#             cos_weight =
#             fanogram #apply cosine weight
#
#     for i_x in range(0, reco.get_size()[0]):
#         for i_y in range(0, reco.get_size()[1]):
#             [x, y] =
#
#             for beta_index in range(0, fanogram.get_size()[?]):
#                 beta_degree =
#                 beta =
#                 source =
#                 SX =
#                 SQ =
#                 ratio_alpha =
#                 SP =
#
#                 t =
#                 value =
#                 U =
#                 value = # value after distance weight
#                 reco.set_at_index(i_x, i_y, ?) #update value
#
#     return reco
#
# def rebinning(fanogram, d_si, d_sd):
#     sinogram = Grid(...) #create an 180-degree sinogram
#     sinogram.set_origin()
#
#     for p in range(0,sinogram.?):
#         theta_degree =
#         theta =
#
#         for s in range(0, sinogram.?):
#             s_world =
#             gamma =
#             beta =
#
#             if beta < 0:
#                 #redundancy equation
#
#             s_fan_world = d
#             beta_degrees =
#             val =
#
#             sinogram.? #set value to sinogram
#
#     return sinogram
#
