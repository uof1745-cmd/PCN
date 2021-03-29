import numpy as np
import cv2 as cv
import math

HEIGHT = 400
WIDTH = 400
break_pix = 500
size = 3

def getpix(cut):
    cor = []
    value = []
    for i in range(0, size + size):
        for j in range(0, size + size):
            if(cut[i, j] != break_pix):
                cor.append([i,j])
                value.append(cut[i,j])
    return cor, value

def getnear(cors, values, now_cor):
    min = 100
    value = break_pix
    for i in range(0, len(cors)):
        cor = cors[i]
        leng = math.sqrt(pow(cor[0] - now_cor[0], 2) + pow(cor[1] - now_cor[1], 2))
        if(leng < min):
            min = leng
            value = values[i]
    return value

def compensate(srcImg):
    dstImg = np.zeros([HEIGHT, WIDTH], np.float32)
    srcImg = cv.copyMakeBorder(srcImg, size, size, size, size, cv.BORDER_CONSTANT, value=0)
    for i in range(size, HEIGHT + size):
        for j in range(size, WIDTH + size):
            if (srcImg[i, j] != break_pix):
                dstImg[i - size, j - size] = srcImg[i, j]
            else:
                if(i < 3 * size or i > WIDTH - (2 * size)):
                    if(i < 3 * size):
                        w_start = size
                        w_end = 3 * size
                        now_x = i - size
                    else:
                        w_start = WIDTH - (2 * size)
                        w_end = WIDTH
                        now_x = i - (WIDTH - (2 * size))
                else:
                    w_start = i - size
                    w_end = i + size
                    now_x = size

                if (j < 3 * size or j > HEIGHT - (2 * size)):
                    if (j < 3 * size):
                        h_start = size
                        h_end = 3 * size
                        now_y = j - size
                    else:
                        h_start = WIDTH - (2 * size)
                        h_end = WIDTH
                        now_y = j - (WIDTH - (2 * size))
                else:
                    h_start = j - size
                    h_end = j + size
                    now_y = size

                cut = srcImg[w_start:w_end, h_start:h_end]
                # cut = srcImg[i - size:i + size, j - size:j + size]
                # print(cut)
                cors, value = getpix(cut)
                rgb = getnear(cors, value, [now_x, now_y])
                dstImg[i - size, j - size] = rgb
    return dstImg