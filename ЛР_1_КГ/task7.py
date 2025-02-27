import math
import random as rn

import numpy as np
from PIL import Image, ImageOps

img_mat = np.zeros((2000, 2000, 3), dtype = np.uint8)

z_buf = [[float('inf') for i in range(img_mat.shape[0])] for j in range(img_mat.shape[1])]

def barycentric(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

def draw_triangl(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, color):
    xmax = math.ceil(max(x0, x1, x2))
    if xmax > img_mat.shape[1]: xmax = img_mat.shape[1]
    xmin = math.floor(min(x0, x1, x2))
    if xmin < 0: xmin = 0
    ymax = math.ceil(max(y0, y1, y2))
    if ymax > img_mat.shape[0]: ymax = img_mat.shape[0]
    ymin = math.floor(min(y0, y1, y2))
    if ymin < 0: ymin = 0

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            l0, l1, l2 = barycentric(x,y, x0, y0, x1, y1, x2, y2)
            if(l0>=0 and l1>=0 and l2>=0):
                z = l0*z0 + l1*z1 + l2*z2
                if z < z_buf[y][x]:
                    img_mat[y, x] = color
                    z_buf[y][x] = z

def norm(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    x = (y1 - y2)*(z1 - z0) - (y1 - y0)*(z1 - z2)
    y = -((x1 - x2)*(z1 - z0) - (x1 - x0)*(z1 - z2))
    z = (x1 - x2)*(y1 - y0) - (x1 - x0)*(y1 - y2)
    return x,y,z

def cos_light(x,y,z):
    return z / math.sqrt(x**2 + y**2 + z**2)


f = open('model_1.obj')

list_v = []
list_f = []
for s in f:
    spl = s.split()
    if(spl[0]=='v'): list_v.append([float(x) for x in spl[1:]])
    if(spl[0]=='f'): list_f.append([int(x.split('/')[0]) for x in spl[1:]])


for i in list_v:
    img_mat[int(10000*i[1])+1000, int(10000*i[0])+1000] = (255, 117, 20)


for face in list_f:
    x0 = 10000*list_v[face[0]-1][0]+1000
    y0 = 10000*list_v[face[0]-1][1]+1000
    z0 = 10000*list_v[face[0]-1][2]+1000
    x1 = 10000*list_v[face[1]-1][0]+1000
    y1 = 10000*list_v[face[1]-1][1]+1000
    z1 = 10000*list_v[face[1]-1][2]+1000
    x2 = 10000*list_v[face[2]-1][0]+1000
    y2 = 10000*list_v[face[2]-1][1]+1000
    z2 = 10000*list_v[face[2]-1][2]+1000
    xn, yn, zn = norm(x0, y0, z0, x1, y1, z1, x2, y2, z2)


    cosl = cos_light(xn,yn,zn)
    if cosl < 0 :draw_triangl(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, (-255*cosl, -100*cosl ,-20*cosl))





img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img7.png')