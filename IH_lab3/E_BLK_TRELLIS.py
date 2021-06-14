# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from Crypto.Hash import SHA256


#状态机转换
STATES = {
    "A": ["A", "B"],
    "B": ["C", "D"],
    "C": ["E", "F"],
    "D": ["G", "H"],
    "E": ["A", "B"],
    "F": ["C", "D"],
    "G": ["E", "F"],
    "H": ["G", "H"]
    }


password = "Hello world!"
msg = "1011011000"
ALPHA = math.sqrt(len(msg))
ref = []
filepath = "D:\\IH_lab2\\data_new"
threshold = 0
total_correct = 0
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
    
def Wa_ref_cal(password,msg,shape):
    state = "A"
    
    
    #E_TRELLIS_8中的msg作用在ref上的结果
    Wm_i = []
    for i in range(len(msg)):
        #状态机转换，计算下一个状态
        next_state = STATES[state][int(msg[i])]
        password_state = next_state+state
        #根据两个状态计算最后的结果
        ref = generate_ref(password+":"+password_state+str(i),(8,8))
        #ref = np.resize(ref,shape)
        Wm_i.append(ref)
        state = next_state
    
    #Wm_i = [ref[i] if msg[i] == "1" else -ref[i] for i in range(len(msg))]
    Wm_i = np.array(Wm_i)
    temp = np.sum(Wm_i,axis=0)
    #均值的归一化
    Wm = temp/temp.std()
    Wa = (ALPHA*Wm)
    return Wa    


def embedding(Wa_ref,Cover):
    #print(str(Wa_ref.shape)+"  "+str(Cover.shape))
    d1 = Cover.shape[0] - Cover.shape[0]%8
    d2 = Cover.shape[1] - Cover.shape[1]%8

    Vo = np.zeros((8,8)).astype('float32')
    for i in range(8):
        for j in range(8):
            Vo[i, j] = np.mean(Cover[:d1,:d2][i::8,j::8])
    
    #Cover_w = Wa_ref+Cover
    Vw = Vo+Wa_ref
    Cover_w = Cover[:d1,:d2]+np.tile(Vw-Vo,(d1//8,d2//8))
    Cover_w[Cover_w>255] = 255
    Cover_w[Cover_w<0] = 0
    
    return np.round(Cover_w).astype('uint8')

def decode(Cover_w,ref,length):
    
    v = np.zeros((8,8)).astype('float32')
    for i in range(8):
        for j in range(8):
            v[i, j] = np.mean(Cover_w[i::8,j::8])
    
    #初始化每一个状态下的path和cost
    msg = ""
    cost = {state: 0.0 for state in STATES}
    path = {state:[] for state in STATES}
    
    path["A"] = ["A"]
    for i in range (length):
        #当前路径中所有的可能的状态
        vstates = [s for s in path if len(path[s])]
        new_path = {}
        new_cost = {state:0.0 for state in STATES}
        #对于每一个状态读取path和cost
        for state in vstates:
            #对于每个0，1，读取cost和path，计算下一个状态
            for msg_bit in range(2):
                next_state = STATES[state][msg_bit]
                password_state = next_state+state
                #根据两个状态计算最后的结果
                ref = generate_ref(password+":"+password_state+str(i),(8,8))
                #ref = np.resize(ref,Cover_w.shape)
                V_nor = (v-np.mean(v))/(np.std(v)*8)
                ref_nor = (ref-np.mean(ref))/(np.std(ref)*8)
                zcc = np.sum(V_nor*ref_nor)
                #遍历更新cost和path
                if not next_state in new_path or new_cost[next_state]<zcc+cost[state]:
                    new_cost[next_state] = cost[state]+zcc
                    new_path[next_state] = path[state]+[next_state]                        
        path = dict(new_path)
        cost = dict(new_cost)
        
    best_path = max(cost,key = cost.get)
    # print(best_path)
    # print(path)
    # print(len(path[best_path]))
    
    for i in range(len(path[best_path])-1):
        if STATES[path[best_path][i]][0]==path[best_path][i+1]:
            msg+="0"
        elif STATES[path[best_path][i]][1]==path[best_path][i+1]:
            msg+="1"
        else:
            print("ERROR:An unexpected state")
            sys.exit(0)
    return msg
                
                
#主函数入口
if __name__ == "__main__":
    
    #读取所有的files文件
    files = os.listdir(filepath)
    
    fn = 0
    fp = 0
    for f in files:
        #embedding
        Cover = cv2.imread(filepath+'/'+f)
        if Cover is None: #or not Cover.shape == (2448,3264,3):
            continue
        print("The picture "+f+" is processing...")
        Cover = cv2.cvtColor(Cover,cv2.COLOR_BGR2GRAY)
        Wa_ref = Wa_ref_cal(password,msg,Cover.shape)
        Cover_w = embedding(Wa_ref,Cover)
        
        
        #decoding
        result = ""
        result = decode(Cover_w,ref,len(msg))
        correct_count = 0
        print(result)
        for i in range(len(msg)):
            if result[i] == msg[i]:
                correct_count+=1
                total_correct+=1
        print("The correctness of the decoding is:"+str(correct_count/len(msg)))
    print("Total correctness:"+str(total_correct/(len(files)*len(msg))))
    print("This message is:"+msg)