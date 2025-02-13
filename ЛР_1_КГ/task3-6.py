import math
from os.path import split

import numpy as np
from PIL import Image, ImageOps

img_mat = np.zeros((2000, 2000, 3), dtype = np.uint8)

def draw_line_dy(img_mat, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2*abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0-t)*y0 + t*y1)
        if (xchange):
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color

        derror += dy
        if (derror > (x1 -x0)):
            derror -= 2*(x1 -x0)
            y += y_update


f = open('model_1.obj')

"""
for s in f:
    if(s[0]=='v' and s[1]==" "): print(s)
"""
list_v = []
list_f = []
for s in f:
    spl = s.split()
    if(spl[0]=='v'): list_v.append([float(x) for x in spl[1:]])
    if(spl[0]=='f'): list_f.append([int(x.split('/')[0]) for x in spl[1:]])

for i in list_v:
    img_mat[int(10000*i[1])+1000, int(10000*i[0])+1000] = (255, 117, 20)

for face in list_f:
    x0 = int(10000*list_v[face[0]-1][0])+1000
    y0 = int(10000*list_v[face[0]-1][1])+1000
    x1 = int(10000*list_v[face[1]-1][0])+1000
    y1 = int(10000*list_v[face[1]-1][1])+1000
    x2 = int(10000*list_v[face[2]-1][0])+1000
    y2 = int(10000*list_v[face[2]-1][1])+1000

    draw_line_dy(img_mat, x0,y0,x1,y1, (255, 117, 20))
    draw_line_dy(img_mat, x1,y1,x2,y2, (255, 117, 20))
    draw_line_dy(img_mat, x0,y0,x2,y2, (255, 117, 20))


img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img3.png')