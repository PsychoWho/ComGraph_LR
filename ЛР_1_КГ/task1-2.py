import math

import numpy as np
from PIL import Image

img_mat = np.zeros((200, 200, 3), dtype = np.uint8)

#img_mat[0:600, 400:800, 0] = 50
#img_mat[0:600, 0:400, 1] = 255

"""
for i in range(600):
    for j in range(800):
        if i == j: img_mat[i,j] = 255;
"""

def draw_line(img_mat, x0, y0, x1, y1, color):
    step = 1.0/100
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        img_mat[y, x] = color

def draw_line_x(img_mat, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0-t)*y0 + t*y1)
        if (xchange):
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color

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




for i in range(13):

    x0 = 100
    y0 = 100
    x1 = int(100+95*math.cos(i*2*math.pi/13))
    y1 = int(100+95*math.sin(i*2*math.pi/13))

    #draw_line(img_mat, x0, y0, x1, y1, 255)
    #draw_line_x(img_mat, x0, y0, x1, y1, 255)
    #draw_line_dy(img_mat, x0, y0, x1, y1, (255, 117, 20))

img = Image.fromarray(img_mat, mode='RGB')

img.save('img.png')