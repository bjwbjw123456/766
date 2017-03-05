__author__ = 'jiaan'
import sys
from diff2 import BDW
from PIL import Image
import numpy as np
from scipy.ndimage import filters
from pylab import *
from cropping import cropping,croppingColor
from scale import scale,scaleColor
from cair import carvColor,carvGray



def dp(c_size,im,patchsize,verticle,step,ope1,ope2):

    dp = []
    parent = []
    resid = 0
    if c_size % step != 0:
        resid = c_size % step
        dp.append([])
        parent.append([])
    c_size = c_size/step
    c_size += 1
    for i in range (c_size):
        dp.append([])
        parent.append([])

    #t = (im,(-1,-1))
   # print len(dp),c_size
    dp[0].append(im)
    #parent.append()
    #print verticle
    if verticle: # for removing verticle seam , decrease width
        for i in range(1,c_size):

            print i
            ones = 0
            twos = 1
            for j in range(0,i+1):
                if j == 0:
                    img = ope1(dp[i-1][0],step,0)
                    dp[i].append(img)
                    parent[i].append(1)
                elif j < i:
                    #print i,j
                    img1 = ope1(dp[i-1][j],step,0)
                    img2 = ope2(dp[i-1][j-1],step,0)
                    diff1 = BDW(im,img1,patchsize)
                    diff2 = BDW(im,img2,patchsize)
                    print 'diff1',diff1
                    print 'diff2', diff2
                    if(diff1 < diff2):
                        #ones += 1
                        dp[i].append(img1)
                        parent[i].append(1)
                    else :
                        #twos += 1
                        dp[i].append(img2)
                        parent[i].append(2)
                else:
                    img = ope2(dp[i-1][i-1],step,0)
                    dp[i].append(img)
                    parent[i].append(2)
            dp[i-1]  = []

        i = c_size-1
        min = BDW(im,dp[i][1],patchsize)
        index = 0

        for j in range(2,len(dp[i])-1):
            diff = BDW(dp[i][j],im,patchsize)
            if diff < min:
                min = diff
                index = j
        print 'min',min ,'index',index
    else:
        for i in range(1,c_size):
            print i
            for j in range(0,i+1):
                if j == 0:
                    img = ope1(dp[i-1][0],0,step)
                    dp[i].append(img)
                    parent[i].append(1)
                elif j < i:
                    #print i,j
                    img1 = ope1(dp[i-1][j],0,step)
                    img2 = ope2(dp[i-1][j-1],0,step)
                    #print img1.shape, im.shape
                    diff1 = BDW(im.transpose(),img1.transpose(),patchsize)
                    diff2 = BDW(im.transpose(),img2.transpose(),patchsize)

                    if(diff1 < diff2):
                        dp[i].append(img1)
                        parent[i].append(1)
                    else :
                        dp[i].append(img2)
                        parent[i].append(2)
                else:
                    img = ope2(dp[i-1][i-1],0,step)
                    dp[i].append(img)
                    parent[i].append(2)
            dp[i-1]  = []
        i = c_size-1
        #print im.shape ,dp[i][0].shape
        min = BDW(im.transpose(),dp[i][0].transpose(),patchsize)
        index = 0
        for j in range(1,len(dp[i])):
            diff = BDW(dp[i][j].transpose(),im.transpose(),patchsize)
            if diff < min:
                min = diff
                index = j

    return parent,index

def multi_OP(filename, w , h, patchsize, step, opegray, opecolor):
    oriImg = Image.open(filename)
    grayImg = oriImg.convert('L')
    color_img = array(oriImg)
    im = array(grayImg)
    #operation = [croppingColor, scaleColor, carveColor]
    if w > 0:
        parent ,index = dp(w,im,patchsize,1,step,opegray[0],opegray[1])
        opes = []
        i = len(parent) -1
        while i > 0:
            val = parent[i][index]
            opes.append(val)
            i-= 1
            if val == 2:
                index -= 1


        opes.reverse()
        print opes
        frequent = []
        prev = opes[0]
        cnt = 0
        for ope in opes:
            if ope == prev:
                cnt+= 1
            else:
                frequent.append((prev-1,cnt))
                cnt = 1
                prev = ope
        frequent.append((prev-1,cnt))

        for tup in frequent:
            print 'tup0',tup[0]
            color_img,im = opecolor[tup[0]](color_img,im,tup[1]*step,0)



    if h > 0:
        frequent = []
        parent ,index = dp(h,im,patchsize,0,opegray[0],opegray[1])
        opes = []
        i = len(parent) -1
        while i > 0:
            val = parent[i][index]
            opes.append(val)
            i-= 1
            if val == 2:
                index -= 1
        opes.reverse()
        print opes
        prev = opes[0]
        cnt = 0
        for ope in opes:
            if ope == prev:
                cnt+= 1
            else:
                frequent.append((prev-1,cnt))
                cnt = 1
                prev = ope
        frequent.append((prev-1,cnt))
        for tup in frequent:
            color_img,im = opecolor[tup[0]](color_img,im,0,tup[1]*step)

    pil_color = Image.fromarray(uint8(color_img))
    pil_color.save('output.jpg')



if __name__ == '__main__':
    '''
    filename = sys.argv[1]
    h_carved_size = sys.argv[2]
    v_carved_size = sys.argv[3]
    oriImg = Image.open(filename)
    grayImg = oriImg.convert('L')
    im = array(grayImg)
    '''
    opegray = [ carvGray, scale]
    opecolor = [carvColor,scaleColor]
    multi_OP('human.jpg',50,0,16,10,opegray,opecolor)
