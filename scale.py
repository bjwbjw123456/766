from PIL import Image
import numpy as np
from pylab import *
from numba import jit

@jit
def scale(garray, h, w):
    height,width = garray.shape
    H = int(height-h)
    W = int(width-w)
    m = float(H)/height
    n = float(W)/width
    #size = (W,H)
    r = np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            x1 = int(i/m)
            x2 = int((i+1)/m)
            y1 = int(j/n)
            y2 = int((j+1)/n)
            sum = 0
            for k in range(x1,x2):
                for l in range(y1,y2):
                    sum = sum+garray[k,l]
            num = (x2-x1)*(y2-y1)
            r[i,j] = sum/num
    return uint8(r)


def scaleColor(img, h ,w):
    height,width,d = img.shape
    #print a.shape
    r = np.zeros((height-h,width-w,d))
    for i in range(d):
        r[:,:,i] = scale(img[:,:,i], h, w)
    return uint8(r)



if __name__ == '__main__':
    imgname = 'castle.jpg'
    oriImg = Image.open(imgname)
    a = array(oriImg)
    tmp = scaleColor(a, 300, 200)
    tmp = Image.fromarray(tmp)
    tmp.save("tmp.jpg")

    '''imgname = 'castle.jpg'
    oriImg = Image.open(imgname)
    grayImg = oriImg.convert('L')
    im2 = array(grayImg)


    tmp = scale(im2, 200, 100)
    tmp = Image.fromarray(tmp)
    tmp.save("tmp.jpg")'''

