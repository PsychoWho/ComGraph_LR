import math
import numpy as np
from PIL import Image, ImageOps

f = open('model_1.obj')
texture_jpg = Image.open('bunny-atlas.jpg')
texture_jpg = ImageOps.flip(texture_jpg)
texture_np = np.asarray(texture_jpg)

texture_width = texture_np.shape[1]
texture_height = texture_np.shape[0]

img_mat = np.zeros((2000, 2000, 3), dtype = np.uint8)

Tz = 1
A = 10000 * Tz

z_buf = [[float('inf') for i in range(img_mat.shape[1])] for j in range(img_mat.shape[0])]

def barycentric(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

def draw_triangl(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, I, W, H):
    u0 = A * x0 / z0 + img_mat.shape[1]/2
    v0 = A * y0 / z0 + img_mat.shape[0]/2
    u1 = A * x1 / z1 + img_mat.shape[1]/2
    v1 = A * y1 / z1 + img_mat.shape[0]/2
    u2 = A * x2 / z2 + img_mat.shape[1]/2
    v2 = A * y2 / z2 + img_mat.shape[0]/2

    xmax = math.ceil(max(u0, u1, u2))
    if xmax > img_mat.shape[1]: xmax = img_mat.shape[1]
    xmin = math.floor(min(u0, u1, u2))
    if xmin < 0: xmin = 0
    ymax = math.ceil(max(v0, v1, v2))
    if ymax > img_mat.shape[0]: ymax = img_mat.shape[0]
    ymin = math.floor(min(v0, v1, v2))
    if ymin < 0: ymin = 0

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            l0, l1, l2 = barycentric(x,y, u0, v0, u1, v1, u2, v2)
            if(l0>=0 and l1>=0 and l2>=0):
                z = l0*z0 + l1*z1 + l2*z2
                In = (l0*I[0] + l1*I[1] + l2*I[2])
                w_t = int(texture_width * (l0*W[0] + l1*W[1] + l2*W[2]))
                h_t = int(texture_height * (l0*H[0] + l1*H[1] + l2*H[2]))
                if z < z_buf[y][x] and In <= 0:
                    img_mat[y, x] = texture_np[h_t, w_t] * -In
                    z_buf[y][x] = z

def norm(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    x = (y1 - y2)*(z1 - z0) - (y1 - y0)*(z1 - z2)
    y = -((x1 - x2)*(z1 - z0) - (x1 - x0)*(z1 - z2))
    z = (x1 - x2)*(y1 - y0) - (x1 - x0)*(y1 - y2)
    return x,y,z

def cos_light(x,y,z):
    return z / math.sqrt(x**2 + y**2 + z**2)

def rotate(x,y,z,a,b,g):
    Rx = np.array([[1,0,0],[0, np.cos(a),np.sin(a)],[0, -np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0,1,0], [-np.sin(b),0,np.cos(b)]])
    Rz = np.array([[np.cos(g),np.sin(g),0],[-np.sin(g), np.cos(g),0],[0,0,1]])
    res = Rx @ Ry @ Rz @ [x,y,z] + [0,-0.03,Tz]
    return res[0], res[1], res[2]

list_v = []
list_f = []
list_vt = []
list_coord = []

for s in f:
    spl = s.split()
    if(spl[0]=='v'): list_v.append([float(x) for x in spl[1:]])
    if(spl[0]=='f'): list_f.append([int(x.split('/')[0]) for x in spl[1:]])
    if(spl[0]=='f'): list_coord.append([int(x.split('/')[1]) for x in spl[1:]])
    if(spl[0]=='vt'): list_vt.append([float(x) for x in spl[1:]])


for i in list_v:
    i[0],i[1],i[2] = rotate(i[0],i[1],i[2], 0,-np.pi,0)

vn_calc = [[0,0,0] for i in range(len(list_v))]


for face in list_f:
    x0 = list_v[face[0]-1][0]
    y0 = list_v[face[0]-1][1]
    z0 = list_v[face[0]-1][2]
    x1 = list_v[face[1]-1][0]
    y1 = list_v[face[1]-1][1]
    z1 = list_v[face[1]-1][2]
    x2 = list_v[face[2]-1][0]
    y2 = list_v[face[2]-1][1]
    z2 = list_v[face[2]-1][2]

    xn, yn, zn = norm(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    n = math.sqrt(xn**2 + yn**2 + zn**2)

    for i in range(3):
        vn_calc[face[i]-1][0] += xn/n
        vn_calc[face[i]-1][1] += yn/n
        vn_calc[face[i]-1][2] += zn/n

for vec in vn_calc:
    n = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    for i in range(3):
        vec[i] /= n


for (face, texture) in zip(list_f, list_coord):
    x0 = list_v[face[0] - 1][0]
    y0 = list_v[face[0] - 1][1]
    z0 = list_v[face[0] - 1][2]
    x1 = list_v[face[1] - 1][0]
    y1 = list_v[face[1] - 1][1]
    z1 = list_v[face[1] - 1][2]
    x2 = list_v[face[2] - 1][0]
    y2 = list_v[face[2] - 1][1]
    z2 = list_v[face[2] - 1][2]

    w0 = list_vt[texture[0] - 1][0]
    h0 = list_vt[texture[0] - 1][1]
    w1 = list_vt[texture[1] - 1][0]
    h1 = list_vt[texture[1] - 1][1]
    w2 = list_vt[texture[2] - 1][0]
    h2 = list_vt[texture[2] - 1][1]

    n0 = vn_calc[face[0]-1]
    n1 = vn_calc[face[1]-1]
    n2 = vn_calc[face[2]-1]

    I0 = cos_light(n0[0], n0[1], n0[2])
    I1 = cos_light(n1[0], n1[1], n1[2])
    I2 = cos_light(n2[0], n2[1], n2[2])

    xn, yn, zn = norm(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    cosl = cos_light(xn, yn, zn)

    if cosl < 0 :draw_triangl(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, (I0, I1, I2), (w0,w1,w2), (h0,h1,h2))


img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img.png')