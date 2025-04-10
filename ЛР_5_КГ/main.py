import math
import numpy as np
from PIL import Image, ImageOps

img_mat = np.zeros((2000, 2000, 3), dtype = np.uint8)

z_buf = [[float('inf') for i in range(img_mat.shape[1])] for j in range(img_mat.shape[0])]

def q_mul(a,b):
    return np.array([a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
                     a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
                     a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
                     a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]])

def q_conj(a):
    return np.array([a[0], -a[1], -a[2], -a[3]])

def q_rotate(x, y, z, ux, uy, uz, tetta, T):
    q = np.array([np.cos(tetta/2), ux*np.sin(tetta/2), uy*np.sin(tetta/2), uz*np.sin(tetta/2)])
    rot = q_mul(q_mul(q,np.array([0,x,y,z])), q_conj(q)) + [0, T[0], T[1], T[2]]
    return rot[1], rot[2], rot[3]


def barycentric(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

def draw_triangl(img_mat, V0, V1, V2, A, I, WH, texture, w_h):
    u0 = A * V0[0] / V0[2] + img_mat.shape[1]/2
    v0 = A * V0[1] / V0[2] + img_mat.shape[0]/2
    u1 = A * V1[0] / V1[2] + img_mat.shape[1]/2
    v1 = A * V1[1] / V1[2] + img_mat.shape[0]/2
    u2 = A * V2[0] / V2[2] + img_mat.shape[1]/2
    v2 = A * V2[1] / V2[2] + img_mat.shape[0]/2

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
                z = l0*V0[2] + l1*V1[2] + l2*V2[2]
                In = (l0*I[0] + l1*I[1] + l2*I[2])
                w_t = int(w_h[0] * (l0*WH[0][0] + l1*WH[1][0] + l2*WH[2][0]))
                h_t = int(w_h[1] * (l0*WH[0][1] + l1*WH[1][1] + l2*WH[2][1]))
                color = texture[h_t, w_t] * -In
                #color = (255*-In, 100*-In ,20*-In)
                if z < z_buf[y][x] and In <= 0:
                    img_mat[y, x] = color
                    z_buf[y][x] = z

def norm(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    x = (y1 - y2)*(z1 - z0) - (y1 - y0)*(z1 - z2)
    y = -((x1 - x2)*(z1 - z0) - (x1 - x0)*(z1 - z2))
    z = (x1 - x2)*(y1 - y0) - (x1 - x0)*(y1 - y2)
    return x,y,z

def cos_light(x,y,z):
    return z / math.sqrt(x**2 + y**2 + z**2)

def rotate(x,y,z,a,b,g,T):
    Rx = np.array([[1,0,0],[0, np.cos(a),np.sin(a)],[0, -np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0,1,0], [-np.sin(b),0,np.cos(b)]])
    Rz = np.array([[np.cos(g),np.sin(g),0],[-np.sin(g), np.cos(g),0],[0,0,1]])
    res = Rx @ Ry @ Rz @ [x,y,z] + T
    return res[0], res[1], res[2]

def draw_img(name_model,A, Rot, T, name_texture):
    texture_jpg = Image.open(name_texture)
    texture_jpg = ImageOps.flip(texture_jpg)
    texture_np = np.asarray(texture_jpg)

    texture_width = texture_np.shape[1]
    texture_height = texture_np.shape[0]

    f = open(name_model)
    list_v = []
    list_f = []
    list_vt = []
    list_coord = []

    for s in f:
        spl = s.split()
        if(len(spl) != 0):
            if(spl[0]=='v'): list_v.append([float(x) for x in spl[1:]])
            if(spl[0]=='f'): list_f.append([int(x.split('/')[0]) for x in spl[1:]])
            if(spl[0]=='f'): list_coord.append([int(x.split('/')[1]) for x in spl[1:]])
            if(spl[0]=='vt'): list_vt.append([float(x) for x in spl[1:]])

    for i in list_v:
        #i[0],i[1],i[2] = rotate(i[0],i[1],i[2], Rot[0],Rot[1],Rot[2], T)
        i[0],i[1],i[2] = q_rotate(i[0],i[1],i[2], 0,0,1, Rot, T)

    vn_calc = [[0,0,0] for i in range(len(list_v))]

    for face in list_f:
        vertex = []
        for i in range(len(face)):
            x = list_v[face[i]-1][0]
            y = list_v[face[i]-1][1]
            z = list_v[face[i]-1][2]
            vertex.append([x,y,z])

        for v in range(len(vertex)-2):
            xn, yn, zn = norm(vertex[0][0], vertex[0][1], vertex[0][2],
                              vertex[v+1][0], vertex[v+1][1], vertex[v+1][2],
                              vertex[v+2][0], vertex[v+2][1], vertex[v+2][2],)
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
        vertex = []
        for i in range(len(face)):
            x = list_v[face[i] - 1][0]
            y = list_v[face[i] - 1][1]
            z = list_v[face[i] - 1][2]
            vertex.append([x, y, z])

        coord = []
        for i in range(len(texture)):
            w = list_vt[texture[i] - 1][0]
            h = list_vt[texture[i] - 1][1]
            coord.append([w,h])

        for v in range(len(face) - 2):
            n0 = vn_calc[face[0] - 1]
            n1 = vn_calc[face[v+1] - 1]
            n2 = vn_calc[face[v+2] - 1]

            I0 = cos_light(n0[0], n0[1], n0[2])
            I1 = cos_light(n1[0], n1[1], n1[2])
            I2 = cos_light(n2[0], n2[1], n2[2])

            xn, yn, zn = norm(vertex[0][0], vertex[0][1], vertex[0][2],
                              vertex[v + 1][0], vertex[v + 1][1], vertex[v + 1][2],
                              vertex[v + 2][0], vertex[v + 2][1], vertex[v + 2][2], )
            cosl = cos_light(xn, yn, zn)

            WH = [coord[0], coord[v+1], coord[v+2]]

            if cosl < 0 :draw_triangl(img_mat, vertex[0], vertex[v+1], vertex[v+2],
                                      A,(I0, I1, I2), WH, texture_np, (texture_width, texture_height))


#draw_img('model.obj',1000,(0,5*math.pi/6,0),(-1,1.5,3), 'model.bmp')
#draw_img('model.obj',1000,(0,-5*math.pi/6,0),(1,1.5,3), 'model.bmp')
#draw_img('cotik.obj', 1000, (-math.pi/2, math.pi, math.pi/2), (-25,-30,60), 'koshk.jpg')
#draw_img('cotik.obj', 1000, (-math.pi/2, math.pi, -math.pi/2), (25  ,-30,60), 'koshk.jpg')

draw_img('cotik.obj', 1000, math.pi, (0,-10,60), 'koshk.jpg')
#draw_img('cotik.obj', 1000, (math.pi,0,0), (0,-20,60), 'koshk.jpg')

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img5.png')