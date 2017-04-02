from PIL import Image
import numpy as np
from scipy.ndimage import filters
from pylab import *
import sys
import time
from numba import jit
import copy


def computeEnergy(im):
    imx = zeros(im.shape)
    filters.sobel(im,1,imx)

    imy = zeros(im.shape)
    filters.sobel(im,0,imy)

    magnitude = sqrt(imx**2 + imy**2)
    return magnitude

'''@jit
def minParent(x,y,dp, width):
    if x==0:
        return 0
    if y==0:
        return min(dp[x-1,y], dp[x-1,y+1])
    if y==width-1:
        return min(dp[x-1,y-1], dp[x-1,y])
    return min(dp[x-1,y-1], dp[x-1,y], dp[x-1,y+1])'''

'''@jit
def dynamicP(magnitude):
    height, width = magnitude.shape
    #print width, height

    #dp = [[0 for col in range(width)] for row in range(height)]
    dp = np.zeros((height,width))

    Min = 0
    bestj = 0;
    for i in range(height):
        for j in range(width):
            tmp = 0
            if i==0:
                tmp = 0
            elif j==0:
                tmp = min(dp[i-1,j], dp[i-1,j+1])
            elif j==width-1:
                tmp = min(dp[i-1,j-1], dp[i-1,j])
            else:
                tmp = min(dp[i-1,j-1], dp[i-1,j], dp[i-1,j+1])

            dp[i,j] = magnitude[i,j] + tmp
    for j in range(width):
        if j==0:
            Min=dp[i,j]
        else:
            if dp[i,j] < Min:
                bestj = j
                Min = dp[i,j]
    return dp, Min, bestj'''

#Todo: boosting

@jit
def find_vertical_seam(magnitude):
    height, width = magnitude.shape
    #dp, Min, bestj = dynamicP(magnitude)
    dp = np.zeros((height,width))

    Min = 0
    bestj = 0;
    for i in range(height):
        for j in range(width):
            tmp = 0
            if i==0:
                tmp = 0
            elif j==0:
                tmp = min(dp[i-1,j], dp[i-1,j+1])
            elif j==width-1:
                tmp = min(dp[i-1,j-1], dp[i-1,j])
            else:
                tmp = min(dp[i-1,j-1], dp[i-1,j], dp[i-1,j+1])

            dp[i,j] = magnitude[i,j] + tmp
            '''if i==height-1:
                if j==0:
                    Min=dp[i,j]
                else:
                    if dp[i,j] < Min:
                        bestj = j
                        Min = dp[i,j]'''
    for j in range(width):
        if j==0:
            Min=dp[i,j]
        else:
            if dp[i,j] < Min:
                bestj = j
                Min = dp[i,j]

    r = [(height-1, bestj)]
    for i in range(height-1,0,-1):
        target = dp[i,r[-1][1]]-magnitude[i,r[-1][1]]
        if target == dp[i-1,r[-1][1]]:
            r.append((i-1,r[-1][1]))
            continue
        if r[-1][1] > 0 and target == dp[i-1,r[-1][1]-1]:
            r.append((i-1,r[-1][1]-1))
            continue
        if r[-1][1] < width-1 and target == dp[i-1,r[-1][1]+1]:
            r.append((i-1,r[-1][1]+1))
            continue
        #print '--------'
        #print target
        #print dp[i-1][r[-1][1]], dp[i-1][r[-1][1]-1], dp[i-1][r[-1][1]+1]
        #break
    return r


    #return dp

def delete_verticle_seam(seam,im,ori_img):
    # delete seam from gray_scale img array and return a new img
    #new_im = im.reshape(1,im.shape(0) * im.shape(1))
    row,col = im.shape
    seam_index = []
    for i in range(len(seam)):
        index = (seam[i][0])*col + seam[i][1]
        seam_index.append(index)
    new_im = np.delete(im,seam_index)
    new_ori_img = np.zeros((row,col-1,3))
    new_ori_img[:,:,0] = np.delete(ori_img[:,:,0],seam_index).reshape(row,col-1)
    new_ori_img[:,:,1] = np.delete(ori_img[:,:,1],seam_index).reshape(row,col-1)
    new_ori_img[:,:,2] = np.delete(ori_img[:,:,2],seam_index).reshape(row,col-1)
    im = new_im.reshape(row,col-1)
    ori_img = new_ori_img
    return im,ori_img

def find_horizontal_seam(magnitude):
    return find_vertical_seam(magnitude.transpose())

def delete_horizontal_seam(seam, im, ori_img):
    new_im = im.transpose()
    row,col = new_im.shape

    seam_index = []
    for i in range(len(seam)):
        index = (seam[i][0])*col + seam[i][1]
        seam_index.append(index)
    new_im = np.delete(new_im,seam_index)
    new_ori_img = np.zeros((col-1,row,3))
    new_ori_img[:,:,0] = np.delete(ori_img[:,:,0].transpose(),seam_index).reshape(row,col-1).transpose()
    new_ori_img[:,:,1] = np.delete(ori_img[:,:,1].transpose(),seam_index).reshape(row,col-1).transpose()
    new_ori_img[:,:,2] = np.delete(ori_img[:,:,2].transpose(),seam_index).reshape(row,col-1).transpose()
    im = new_im.reshape(row,col-1).transpose()
    ori_img = new_ori_img
    return im,ori_img


def carvColor(color_Img, grayImg, wp, hp):
    for i in range(0,wp):
        #print i
        magnitude = computeEnergy(grayImg)
        best_seam = find_vertical_seam(uint32(magnitude))
        grayImg,color_Img = delete_verticle_seam(best_seam,grayImg,color_Img)


    for i in range(0,hp):
        #print i
        magnitude = computeEnergy(grayImg)
        best_seam = find_horizontal_seam(uint32(magnitude))
        grayImg,color_Img = delete_horizontal_seam(best_seam,grayImg,color_Img)

    return uint8(color_Img), uint8(grayImg)

def delete_verticle_gray(seam,im):
    # delete seam from gray_scale img array and return a new img
    #new_im = im.reshape(1,im.shape(0) * im.shape(1))
    row,col = im.shape
    seam_index = []
    for i in range(len(seam)):
        index = (seam[i][0])*col + seam[i][1]
        seam_index.append(index)
    new_im = np.delete(im,seam_index)
    im = new_im.reshape(row,col-1)

    return im

def delete_horizontal_gray(seam, im):
    new_im = im.transpose()
    row,col = new_im.shape

    seam_index = []
    for i in range(len(seam)):
        index = (seam[i][0])*col + seam[i][1]
        seam_index.append(index)
    new_im = np.delete(new_im,seam_index)
    im = new_im.reshape(row,col-1).transpose()
    return im


def carvGray(grayImg, wp, hp):
    #garray = grayImg[:,:]
    for i in range(0,wp):
        #print i
        magnitude = computeEnergy(grayImg)
        best_seam = find_vertical_seam(uint32(magnitude))
        grayImg = delete_verticle_gray(best_seam,grayImg)


    for i in range(0,hp):
        #print i
        magnitude = computeEnergy(grayImg)
        best_seam = find_horizontal_seam(uint32(magnitude))
        grayImg = delete_horizontal_gray(best_seam,grayImg)

    return uint8(grayImg)


def carving(imgname, wp, hp):
    oriImg = Image.open(imgname)
    color_Img = array(oriImg)
    grayImg = oriImg.convert('L')

    #calculate steps
    if wp < hp:
        wstep = 10
        hstep = hp/wp * 10
    else:
        hstep = 10
        wstep = wp/hp * 10


    tE = 0.0
    tF = 0.0
    tD = 0.0
    tf,tb = 0.0,0.0

    im = array(grayImg)
    best_seam = 0

    ite = min(wp/wstep, hp/hstep)
    for i in range(0,ite):
        wp -= wstep
        hp -= hstep
        for m in range(0,wstep):
            magnitude = computeEnergy(im)
            best_seam = find_vertical_seam(uint32(magnitude))
            im,color_Img = delete_verticle_seam(best_seam,im,color_Img)
        for n in range(0,hstep):
            magnitude = computeEnergy(im)
            best_seam = find_horizontal_seam(uint32(magnitude))
            im,color_Img = delete_horizontal_seam(best_seam,im,color_Img)
    for m in range(0,wp):
        magnitude = computeEnergy(im)
        best_seam = find_vertical_seam(uint32(magnitude))
        im,color_Img = delete_verticle_seam(best_seam,im,color_Img)
    for n in range(0,hp):
        magnitude = computeEnergy(im)
        best_seam = find_horizontal_seam(uint32(magnitude))
        im,color_Img = delete_horizontal_seam(best_seam,im,color_Img)


    pil_im = Image.fromarray(im)
    pil_im.save('out.tiff')
    pil_color = Image.fromarray(uint8(color_Img))
    pil_color.save('output.jpg')




    #ori = Image.fromarray()
    '''
    for i in range(0,wp):
        print i
        t1 = time.time()
        magnitude = computeEnergy(im)

        t2 = time.time()
        best_seam = find_vertical_seam(uint32(magnitude))
        t3 = time.time()
        im,color_Img = delete_verticle_seam(best_seam,im,color_Img)
        t4 = time.time()
        tE += t2-t1;
        tF += t3-t2;
        tD += t4-t3;


    for i in range(0,hp):
        print i
        magnitude = computeEnergy(im)
        best_seam = find_horizontal_seam(uint32(magnitude))
        im,color_Img = delete_horizontal_seam(best_seam,im,color_Img)

    #print tF, tf,tb

    pil_im = Image.fromarray(im)
    pil_im.save('out.tiff')
    pil_color = Image.fromarray(uint8(color_Img))
    pil_color.save('output.jpg')
    #ori = Image.fromarray()

    #print best_seam
    image = array(oriImg)
    for i in range(len(best_seam)):
        curPoint = best_seam[i]
        plot(curPoint[0],curPoint[1],'r*')

    imshow(image)
    show()
    '''






if __name__ == '__main__':
    t1 = time.time()
    #carving('castle.jpg',20,1)
    oriImg = Image.open('empire.jpg')
    color_Img = array(oriImg)
    grayImg = oriImg.convert('L')
    im = array(grayImg)

    tmp,gimg = carvColor(color_Img, im, 500,0)
    #tmp = carvGray(im,200,0)
    tmp = Image.fromarray(tmp)
    tmp.save('tmp.png')

    t2 = time.time()

    #testArray = np.array([1,1,2,3,3,5,7,1,3])
    #testArray = testArray.reshape(3,3)
    #find_vertical_seam(testArray)
    print t2-t1
