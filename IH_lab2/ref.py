# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt

#N = 2448*3264

def generate_ref():
    for i in range(40):
        
        ref = np.random.randint(0,255,size=(512,512))
        #ref = np.zeros((512,512))
        #ref = ref/255.0
        #text = "Hello world!"
        #cv2.putText(ref,text,(0,100),cv2.FONT_HERSHEY_COMPLEX, 2.0, 128, 5)
        
        #ref = cv2.cvtColor(ref,cv2.COLOR_GRAY2BGR)
        cv2.imwrite("D:\\IH_lab2\\reference_new\\reference_255_"+str(i)+".BMP",ref)
    
    
    
def embedding(Wa_ref,Cover):
    print(str(Wa_ref.shape)+"  "+str(Cover.shape))
    # cv2.imshow("Cover",Cover)
    #print(Cover[0])
    Cover_w = cv2.add(Cover,Wa_ref,dtype = cv2.CV_32F)
    Cover_w = Cover_w/255.0
    #print(Cover_w[0])
    # cv2.imshow("Cover_w",Cover_w)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return Cover_w

def decode(Cover_w,ref,size):
    zlc = []
    for i in range(8):
        zlc.append(np.sum(Cover_w*ref[i]/size))
    
    #print(zlc)
    return zlc

def Wa_ref_cal(ref,msg,alpha):
    temp_bits = 1#用于按位与运算
    Wa_ref = np.zeros((512,512))
    for i in range(8):
        is_1 = temp_bits & msg
        if is_1 == 0 :
            Wa_ref -= ref[i]
        else:
            Wa_ref += ref[i]
        temp_bits*=2
    #Wa_ref = (2*msg-1)*alpha*ref
    Wa_ref /= 8.0
    Wa_ref *= alpha/math.sqrt(8)
    print(Wa_ref[0])
    return Wa_ref

generate_ref()

msg = 255
alpha = math.sqrt(8)
parts = 80

Cover = cv2.imread("D:\\IH_lab2\\data\\lena512.bmp")
Cover = cv2.cvtColor(Cover,cv2.COLOR_BGR2GRAY)
path = "D:\\IH_lab2\\reference"
files = os.listdir(path)

#m = 0
threshold = 0.7
fn = 0
fp = 0
for j in range(32):
    ref = []
    zlclist_0 = []
    zlclist_ref = []
    for i in range(8):
        ref_temp = cv2.imread("D:\\IH_lab2\\reference_new\\reference_255_"+str(i+j)+".BMP")
        ref_temp = cv2.cvtColor(ref_temp,cv2.COLOR_BGR2GRAY)
        ref.append(ref_temp)
    
    # Cover = cv2.imread(path+'/'+f)
    # if Cover is None: #or not Cover.shape == (2448,3264,3):
    #     continue
    # print("The picture "+f+" is processing...")
    # Cover = cv2.cvtColor(Cover,cv2.COLOR_BGR2GRAY)
    Wa_ref = Wa_ref_cal(ref,msg,alpha)
    Wa_ref = np.resize(Wa_ref,Cover.shape)
    Cover_w = embedding(Wa_ref,Cover)
    for i in range(8):
        ref[i] = np.resize(ref[i],Cover.shape)
    zlclist_0 = decode(Cover_w,ref,ref[0].shape[0]*ref[0].shape[1])
    zlclist_ref = decode(Cover,ref,ref[0].shape[0]*ref[0].shape[1])
    print("The result of "+str(j)+" is:"+str(zlclist_0))
    print("The result of "+str(j)+" is:"+str(zlclist_ref))
    count_0 = 0
    count_1 = 0
    count_no = 0
    for i in range(8):
        zlclist_0[i] -= zlclist_ref[i]
        if zlclist_0[i]>= threshold:
            count_0+=1
        elif zlclist_0[i]<= -threshold:
            count_1+=1
        else:
            count_no+=1
    if count_0>4:
        fp = fp
    elif count_1>4 or count_no>4:
        fp += 1
    print("The correctness of the decoding is:"+str(count_0/8))
    count_0 = 0
    count_1 = 0
    count_no = 0
    for i in range(8):
        if zlclist_ref[i]>= threshold:
            count_0+=1
        elif zlclist_ref[i]<= -threshold:
            count_1+=1
        else:
            count_no+=1
    if count_0>4 or count_1>4:
        fn+=1
#print(zlclist_0)   



#print(zlclist_no)
#average = np.mean(zlclist_no)
# print(average)
# zlclist_0 = (zlclist_0-average)/2
# zlclist_1 = (zlclist_1-average)/2
# zlclist_no = (zlclist_no-average)/2

# threshold = 0.7
# fp = 0
# fn = 0
# for i in range(200):
#     if zlclist_0[i]>-0.7:
#         fp+=1
#     if zlclist_1[i]<0.7:
#         fp+=1
#     if zlclist_no[i]<-0.7 or zlclist_no[i]>0.7:
#         fn+=1
fp_possi = fp/400.0
fn_possi = fn/200.0

print("False positive:"+str(fp_possi))
print("False negative:"+str(fn_possi))

# boxlist_m_0 = [0]*4*parts
# for zlclist_0_one in zlclist_0:
#     boxnumber = (int(zlclist_0_one*parts))//1+2*parts
#     #print(boxnumber)
#     boxlist_m_0[boxnumber]+=1
# #print(boxlist_m_0)
