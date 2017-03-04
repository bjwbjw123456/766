import numpy as np
import sys
from PIL import Image
from scipy.ndimage import filters
from pylab import *
import sys
import time
import multiprocessing


def worker(imarray1,imarray2,i,s,t,patchsize):
    #print "in worker"
    part1 = A_DTW(imarray1,imarray2,i,s,t,patchsize)
    part2 = A_DTW(imarray2,imarray1,i,t,s,patchsize)
    ans = 1.0/s*part1 +1.0/t* part2
    return ans





def BDW(imarray1, imarray2, patchsize):
    step = patchsize
    height1, width1 = imarray1.shape
    height2, width2 = imarray2.shape

    imarray1 = np.lib.pad(imarray1, ((0, (patchsize - height1%patchsize)%patchsize),\
                                     (0, (patchsize - width1%patchsize)%patchsize)),\
                          'constant', constant_values=(0,0))
    imarray2 = np.lib.pad(imarray2, ((0, (patchsize - height2%patchsize)%patchsize),\
                                     (0, (patchsize - width2%patchsize)%patchsize)),\
                          'constant', constant_values=(0,0))
    height1, width1 = imarray1.shape
    height2, width2 = imarray2.shape
    #print height1, height2

    h = height1/ patchsize
    s = width1 / patchsize
    t = width2 / patchsize
    ans = 0
    #m1 = 0
    #m2 = 0
    pool = multiprocessing.Pool(processes=4)
    #q = multiprocessing.Queue(h+1)
    result = []
    ans = 0
    for i in range(0,h):
        result.append(pool.apply_async(worker, args=(imarray1,imarray2,i,s,t,patchsize,)))
        #part1 = A_DTW(imarray1,imarray2,i,s,t,patchsize)
        #part2 = A_DTW(imarray2,imarray1,i,t,s,patchsize)
        #m1 = max(m1,part1)
        #m2 = max(m2, part2)

        #ans += 1.0/s*part1 +1.0/t* part2
    pool.close()
    pool.join()
    for res in result:
        tmpp = res.get()
     #   print tmpp
        ans+=tmpp


    return ans*1.0/h

def A_DTW(imarray1,imarray2, h, s, t, patchsize):
    m = np.zeros((s+1,t+1))
    m[0,0] = 0
    for i in range(1,s+1):
        m[i,0] = sys.maxint
    for j in range(1,t+1):
        m[0,j] = 0
    for i in range(1,s+1):
        for j in range(1,t+1):
            dist = distance(imarray1,imarray2,h,i,j,patchsize)
            m[i,j] = min(m[i-1,j-1] + dist,m[i,j-1],m[i-1,j] + dist)
    return m[s,t]

def distance(imarray1, imarray2, h, i,j,patchsize):
    #print 'right'
    sub1 = imarray1[h*patchsize:(h+1)*patchsize,(i-1)*patchsize:(i)*patchsize]
    sub2 = imarray2[h*patchsize:(h+1)*patchsize,(j-1)*patchsize:(j)*patchsize]
    val = sum(abs(sub1 * sub1 - sub2 * sub2))

    return val



if __name__ == '__main__':
    imgname = 'castle.jpg'
    oriImg = Image.open(imgname)
    grayImg = oriImg.convert('L')
    im2 = array(grayImg)


    imgname = 'tmp.jpg'
    oriImg = Image.open(imgname)
    grayImg = oriImg.convert('L')
    im1 = array(grayImg)

    t1 = time.time()
    ans = BDW(im1,im2,16)
    t2 = time.time()

    print t2-t1
    print ans

    #p = multiprocessing.Process(target=worker, args=(1,))
    #p.start()
