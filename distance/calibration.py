import numpy as np
from numpy import sin, cos, tan
from operator import itemgetter

# internal parameters are known
fov = 86 # horizontal
pw = 1280
ph = 720

coord_set_for_estimation = [
    # uvz(screen cood), xyz(unrealengine coord)
    ([490, 260, 460], [20, 130, 220]),
    ([615, 270, 460], [120, 130, 220]),
    ([790, 270, 460], [270, 130, 220]),
    ([775, 255, 360], [220, 130, 120]),
    ([740, 270, 510], [220, 130, 270]),
    ([610, 225, 260], [120, 130, 20]),
    ([760, 275, 560], [270, 130, 320]),
    ([570, 270, 600], [70, 130, 360]),
]

# calc internal parameter matrix
fx = 1.0 / (2.0 * tan(np.radians(fov) / 2.0)) * pw
fy = fx
cx = pw / 2.0
cy = ph / 2.0

K = np.asarray([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1],
])

# estimate external parameter matrix
screen_coords = list(map(itemgetter(0), coord_set_for_estimation))
world_coords = list(map(itemgetter(1), coord_set_for_estimation))

## convert screen coord to camera coord
K_inv = np.linalg.inv(K)
def conv_cs_to_cc(cs):
    u, v, z = cs
    cs_ = np.array([u, v, 1])
    cc = np.dot(K_inv, cs_) * z
    return [cc[0], cc[1], cc[2], 1]
# lambda cs: np.dot(K_inv, [cs[0], cs[1], 1] * cs[2]
camera_coords = list(map(conv_cs_to_cc, screen_coords))

## split X to R,t
R = [[ 1.19935047e+00,  6.83878520e-01,  8.87610162e-02],
     [ 4.24421102e-17, -2.72004641e-15, -4.16333634e-16],
     [-2.98463872e-16,  5.10702591e-15,  1.00000000e+00]]

t = [143.4804488403023, 129.9999999999999, -240.0]

#print(R)
#print(t)

def calibration_pos(pos_list):

    calibration_pos_list = []
    
    for cs in pos_list:
        us, vs, zs = cs

        K_inv = np.linalg.inv(K)

        # camera coord
        cc = np.dot(K_inv, zs * np.asarray([us, vs, 1]))

        # world coord (= unrealengine coord)
        cw = np.dot(R, cc) + t
        
        calibration_pos_list.append(cw)
        
    return calibration_pos_list

'''
        print(cs)
        print(cw)
        print()
'''

'''
if __name__ == "__main__":
    # check conversion with estimated matrix
    coord_set = [
        # uvz(screen cood), xyz(unrealengine coord)
       [900, 275, 560],
       [610, 250, 360]
    ]
    
    calibration_pos(coord_set)
'''