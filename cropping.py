from PIL import Image
from numpy import *
from scipy.ndimage import filters
from pylab import *
import sys
#from numba import jit

def croppingColor(color_img,im,w,h):
    imx = zeros(im.shape)
    filters.sobel(im,1,imx)

    imy = zeros(im.shape)
    filters.sobel(im,0,imy)

    magnitude = sqrt(imx**2 + imy**2)

    i = 0
    height,width = im.shape
    if w > 0:
        l, r = 0, width-1
        while i < w:
            i += 1
            l_sum = sum(magnitude[:,l])
            r_sum = sum(magnitude[:,r])
            if l_sum < r_sum:
                l = l + 1
            else:
                r = r -1
        #tmp = array(oriImg)
        list = range(l,r+1)
        result = color_img[:,list]
        grayimg = im[:,list]
        resultImg = Image.fromarray(result)
        resultImg.save('tmp.jpg')
    else:
        l,r  = 0,height-1
        while i < h:
            i+= 1
            l_sum = sum(magnitude[l,:])
            r_sum = sum(magnitude[r,:])
            if l_sum < r_sum:
                l += 1
            else:
                r -= 1
        #tmp = array(oriImg)
        list = range(l,r+1)
        result = color_img[list,:]
        grayimg = im[list,:]
        resultImg = Image.fromarray(result)
        resultImg.save('tmp.jpg')
    return result,grayimg



def cropping(im, w, h):
    #print nHeight, nWidth
    imx = zeros(im.shape)
    filters.sobel(im,1,imx)
    imy = zeros(im.shape)
    filters.sobel(im,0,imy)

    magnitude = sqrt(imx**2 + imy**2)
    i = 0
    height,width = im.shape
    if w > 0:
        l, r = 0, width-1
        while i < w:
            i += 1
            l_sum = sum(magnitude[:,l])
            r_sum = sum(magnitude[:,r])
            if l_sum < r_sum:
                l = l + 1
            else:
                r = r -1

        list = range(l,r+1)
        result = im[:,list]
    else:
        l,r  = 0,height-1
        while i < h:
            i+= 1
            l_sum = sum(magnitude[l,:])
            r_sum = sum(magnitude[r,:])
            if l_sum < r_sum:
                l += 1
            else:
                r -= 1
        list = range(l,r+1)
        result = im[list,:]
    return result



    '''
    Max = 0
    besti = 0
    bestj = 0
    width, height = size(grayImg)
    dp = [[0 for col in range(width - nWidth + 1)] for row in range(height - nHeight + 1)]

    for i in range(nHeight):
        for j in range(nWidth):
            dp[0][0] += magnitude[i][j]
    Max = dp[0][0]

    for i in range(1,len(dp)):
        neg = 0
        pos = 0

        for x in range(nWidth):
            neg += magnitude[i-1][x]
            pos += magnitude[i-1+nHeight][x]

        dp[i][0] = dp[i-1][0] - neg + pos
        if dp[i][0] > Max:
            Max,besti, bestj = dp[i][0],i,0


    for j in range(1,len(dp[0])):
        neg = 0
        pos = 0

        for x in range(nHeight):
            neg += magnitude[x][j-1]
            pos += magnitude[x][j-1+nWidth]

        dp[0][j] = dp[0][j-1] - neg + pos
        if dp[0][j] > Max:
            Max,besti,bestj = dp[0][j], 0, j


    for i in range(1, len(dp)):
        for j in range(1,len(dp[0])):
            dp[i][j] = dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1] + magnitude[i+nHeight-1][j+nWidth-1] \
                        + magnitude[i-1][j-1]  - magnitude[i+nHeight-1][j-1] - magnitude[i-1][j+nWidth-1]
            if dp[i][j] > Max:
                Max,besti,bestj = dp[i][j], i, j


    tmp = array(oriImg)
    result = tmp[besti:besti+nHeight,bestj:bestj+nWidth]
    resultImg = Image.fromarray(result)
    resultImg.save('tmp.jpg')
    '''

if __name__ == '__main__':
    fileName = sys.argv[1]
    #verticle_crop = int(sys.argv[2])
    #crop_size = int(sys.argv[3])

    oriImg = Image.open(fileName);
    grayImg = oriImg.convert('L')
    im = array(grayImg)
    color_img = array(oriImg)

    croppingColor(color_img,im,500,0)



    #cropping(fileName, verticle_crop, crop_size)


    '''oriImg = Image.open('test2.jpg')
    grayImg = oriImg.convert('L')

    im = array(grayImg)

    imx = zeros(im.shape)
    filters.sobel(im,1,imx)

    imy = zeros(im.shape)
    filters.sobel(im,0,imy)

    magnitude = sqrt(imx**2 + imy**2)

    max = 0
    height, width = size(grayImg)


    dp = [[0 for col in range(width)] for row in range(height)]



    pil_im = Image.fromarray(uint8(magnitude))
    result = pil_im.convert('RGB')




    #print type(magnitude[0][0])

    imshow(magnitude[:, :])
    tarray = array(oriImg)
    print magnitude[:,:].mean()

    show()
    #pil_im.save('out.tiff')'''
