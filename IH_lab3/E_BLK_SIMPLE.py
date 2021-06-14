# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 12:13:27 2021

@author: 小天才
"""

import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from Crypto.Hash import SHA256

ALPHA = 2.0
password = "Hello world!"
msg = "10110110"
ref = []
path = "D:\\IH_lab2\\data_new"
threshold = 0.2
#N = 2448*3264

def generate_ref(password,shape):
    hash = SHA256.new()
    hash.update(password.encode('utf-8'))
    seed = int.from_bytes(hash.digest(),"little")%2 **32
    np.random.seed(seed)
    ref = np.random.randint(256,size=shape).astype('float32')
    ref -= np.mean(ref)#0均值
    ref /= np.std(ref)#单位化长度
    
    cv2.imwrite("D:\\IH_lab2\\reference_new\\referenceSHA256.BMP",ref)
    return ref
    
def Wa_ref_cal(ref,msg):
    #E_SIMPLE_8中的msg作用在ref上的结果
    Wm_i = [ref[i] if msg[i] == "1" else -ref[i] for i in range(len(msg))]
    Wm_i = np.array(Wm_i)
    temp = np.sum(Wm_i,axis=0)
    #均值的归一化
    Wm = temp/temp.std()
    Wa = (ALPHA*Wm)
    return Wa    


def embedding(Wa_ref,Cover):
    d1 = Cover.shape[0] - Cover.shape[0]%8
    d2 = Cover.shape[1] - Cover.shape[1]%8

    Vo = np.zeros((8,8)).astype('float32')
    for i in range(8):
        for j in range(8):
            Vo[i, j] = np.mean(Cover[:d1,:d2][i::8,j::8])

    Vw = Vo+Wa_ref
    Cover_w = Cover[:d1,:d2]+np.tile(Vw-Vo,(d1//8,d2//8))
    Cover_w[Cover_w>255] = 255
    Cover_w[Cover_w<0] = 0
    
    return np.round(Cover_w).astype('uint8')

def decode_zcc(Cover_w,ref):
    zcc = []
    v = np.zeros((8,8)).astype('float32')
    for i in range(8):
        for j in range(8):
            v[i, j] = np.mean(Cover_w[i::8,j::8])
    
    for i in range(8):
        V_nor = (v-np.mean(v))/(np.std(v)*8)
        ref_nor = (ref[i]-np.mean(ref[i]))/(np.std(ref[i])*8)
        zcc_temp = np.sum(V_nor*ref_nor)
        zcc.append(zcc_temp)
    return zcc
def decode_zlc(Cover_w,ref):
    zlc = []
    v = np.zeros((8,8)).astype('float32')
    for i in range(8):
        for j in range(8):
            v[i, j] = np.mean(Cover_w[i::8,j::8])
    for i in range(8):
        zlc.append(np.mean(Cover_w*ref[i]))
    return zlc

#主函数入口
if __name__ == "__main__":
    
    #读取所有的files文件
    files = os.listdir(path)

#生成当前的reference集合，大小暂时设定为512*512，之后会随着实际进行修改
    for i in range(len(msg)):
        ref_temp = generate_ref(password+":"+str(i),(8,8));
        ref.append(ref_temp)
    
    fn = 0
    fp = 0
    total_count = 0
    for j in range(40):
        f = files[j]
        #embedding
        Cover = cv2.imread(path+'/'+f)
        if Cover is None: #or not Cover.shape == (2448,3264,3):
            continue
        print("The picture "+f+" is processing...")
        Cover = cv2.cvtColor(Cover,cv2.COLOR_BGR2GRAY)
        # for i in range(len(msg)):
        #     ref[i] = np.resize(ref[i],Cover.shape)
        Wa_ref = Wa_ref_cal(ref,msg)
        Cover_w = embedding(Wa_ref,Cover)
        
        
        #decoding
        zlclist = []
        zlclist_no_watermark = []
        result = ""
        zlclist = decode_zcc(Cover_w,ref)
        zlclist_no_watermark = decode_zcc(Cover,ref)
        
        #计算fp
        #print("The result of "+f+" is:"+str(zlclist_no_watermark))
        count_0 = 0
        count_1 = 0
        count_no = 0
        #计算解码正确率
        correct_count = 0
        
        for i in range(8):
            #zlclist_0[i] -= zlclist_ref[0]
            if zlclist[i]>= threshold:
                result+="1"
                count_0+=1
            elif zlclist[i]<= -threshold:
                result+="0"
                count_1+=1
            else:
                #print("No watermark!")
                count_no+=1
            if len(result) and result[len(result)-1]==msg[i]:
                correct_count+=1
                total_count+=1
        if count_0>4:
            fp = fp
        elif count_1>4 or count_no>4:
            fp += 1
        
        print(result)       
        print("The correctness of the decoding is:"+str(correct_count/8))
        #计算fn
        count_0 = 0
        count_1 = 0
        count_no = 0
        for i in range(8):
            if zlclist_no_watermark[i]>= threshold:
                count_0+=1
            elif zlclist_no_watermark[i]<= -threshold:
                count_1+=1
            else:
                count_no+=1
        if count_0>4 or count_1>4:
            fn+=1
    print("The total correctness of the decoding is:"+str(total_count/(len(msg)*40)))
    fp_possi = fp/40.0
    fn_possi = fn/40.0

print("False positive:"+str(fp_possi))
print("False negative:"+str(fn_possi))