import numpy as np
import cv2 as cv

HEIGHT = 400
WIDTH = 400
final_lon = 256

def collect_edges(start, end, u_d, v_d, u, v, up, down, left, right):
    if (v_d == start):
        up.append([u, v])
    if (v_d == end - 1):
        down.append([u, v])
    if (u_d == start):
        left.append([u, v])
    if (u_d == end - 1):
        right.append([u, v])
    return up, down, left, right

def inter_x(start, end):
    list = []
    num = end[0] - start[0] - 1
    deta = end[1] - start[1]
    each = float(deta)/float(num + 1)
    for i in range(1, num + 1):
        list.append([start[0] + i, int(start[1] + i * each)])

    list.insert(0, start)
    list.insert(len(list), end)
    return list

def compen_row(src, name):
    result = []

    if(name == 'up'):
        left = [0, 0]
        right = [WIDTH - 1, 0]

        if (src[0][0] != 0):
            src.insert(0, left)
        if (src[len(src) - 1][0] != WIDTH - 1):
            src.insert(len(src), right)

    if (name == 'down'):
        left = [0, WIDTH - 1]
        right = [WIDTH - 1, WIDTH - 1]

        if (src[0][0] != 0):
            src.insert(0, left)
        if (src[len(src) - 1][0] != WIDTH - 1):
            src.insert(len(src), right)

    for i in range(len(src) - 1):
        add_list = inter_x(src[i], src[i + 1])
        if(i != 0):
            add_list.pop(0)
        result[len(result):len(result)] = add_list
    return result

def inter_y(start, end):
    list = []
    num = end[1] - start[1] - 1
    deta = end[0] - start[0]
    each = float(deta)/float(num + 1)
    for i in range(1, num + 1):
        list.append([int(start[0] + i * each), start[1] + i])

    list.insert(0, start)
    list.insert(len(list), end)
    return list

def compen_col(src, name):
    result = []

    if(name == 'left'):
        up = [0, 0]
        down = [0, HEIGHT - 1]

        if (src[0][1] != 0):
            src.insert(0, up)
        if (src[len(src) - 1][1] != HEIGHT - 1):
            src.insert(len(src), down)

    if (name == 'right'):
        up = [WIDTH - 1, 0]
        down = [WIDTH - 1, WIDTH - 1]

        if (src[0][1] != 0):
            src.insert(0, up)
        if (src[len(src) - 1][1] != WIDTH - 1):
            src.insert(len(src), down)

    for i in range(len(src) - 1):
        add_list = inter_y(src[i], src[i + 1])
        if(i != 0):
            add_list.pop(0)
        result[len(result):len(result)] = add_list
    return result

def compen_edges(up, down, left, right):
    new_up = compen_row(up, 'up')
    new_down = compen_row(down,  'down')
    new_left = compen_col(left,  'left')
    new_right = compen_col(right,  'right')
    return new_up, new_down, new_left, new_right

def cut(input, cor, name):
    # print(cor)
    dstImg = np.zeros([HEIGHT, WIDTH, 3], np.uint8)
    if(name == 'up'):
        for i in range(0, WIDTH):
            for j in range(0, HEIGHT):
                if (j >= cor[i][1]):
                    dstImg[j, i, 0] = input[j, i, 0]
                    dstImg[j, i, 1] = input[j, i, 1]
                    dstImg[j, i, 2] = input[j, i, 2]
    if(name == 'down'):
        for i in range(0, WIDTH):
            for j in range(0, HEIGHT):
                if (j <= cor[i][1]):
                    dstImg[j, i, 0] = input[j, i, 0]
                    dstImg[j, i, 1] = input[j, i, 1]
                    dstImg[j, i, 2] = input[j, i, 2]
    if(name == 'left'):
        for i in range(0, HEIGHT):
            for j in range(0, WIDTH):
                if (j >= cor[i][0]):
                    dstImg[i, j, 0] = input[i, j, 0]
                    dstImg[i, j, 1] = input[i, j, 1]
                    dstImg[i, j, 2] = input[i, j, 2]
    if(name == 'right'):
        for i in range(0, HEIGHT):
            for j in range(0, WIDTH):
                if (j <= cor[i][0]):
                    dstImg[i, j, 0] = input[i, j, 0]
                    dstImg[i, j, 1] = input[i, j, 1]
                    dstImg[i, j, 2] = input[i, j, 2]
    return dstImg

def cut_one_c(input, cor, name):
    dstImg = np.zeros([HEIGHT, WIDTH], np.float) + 500
    if(name == 'up'):
        for i in range(0, WIDTH):
            for j in range(0, HEIGHT):
                if (j >= cor[i][1]):
                    dstImg[j, i] = input[j, i]
    if(name == 'down'):
        for i in range(0, WIDTH):
            for j in range(0, HEIGHT):
                if (j <= cor[i][1]):
                    dstImg[j, i] = input[j, i]
    if(name == 'left'):
        for i in range(0, HEIGHT):
            for j in range(0, WIDTH):
                if (j >= cor[i][0]):
                    dstImg[i, j] = input[i, j]
    if(name == 'right'):
        for i in range(0, HEIGHT):
            for j in range(0, WIDTH):
                if (j <= cor[i][0]):
                    dstImg[i, j] = input[i, j]
    return dstImg


def get_new_gt2(srcImg, det_u, det_v, up, down, left, right):
    new_up, new_down, new_left, new_right = compen_edges(up, down, left, right)
    src1 = cut(srcImg, new_up, 'up')
    src2 = cut(src1, new_down, 'down')
    src3 = cut(src2, new_left, 'left')
    src4 = cut(src3, new_right, 'right')

    det_u1 = cut_one_c(det_u, new_up, 'up')
    det_u2 = cut_one_c(det_u1, new_down, 'down')
    det_u3 = cut_one_c(det_u2, new_left, 'left')
    det_u4 = cut_one_c(det_u3, new_right, 'right')

    det_v1 = cut_one_c(det_v, new_up, 'up')
    det_v2 = cut_one_c(det_v1, new_down, 'down')
    det_v3 = cut_one_c(det_v2, new_left, 'left')
    det_v4 = cut_one_c(det_v3, new_right, 'right')
    return src4, det_u4, det_v4