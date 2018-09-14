import matplotlib.pyplot as plt
import numpy as np
import neurolab
#from tmgsimple import TmgSimple
import sys
from scipy.io import loadmat
import xlrd
import nltk
from scipy.linalg import svd

from sklearn import svm

import math

def print3dTensor(tensor, numberOfMatrices): # 4x4xnumberOfMatrices
    print("****************************************************************************************")
    for k in range(0, numberOfMatrices):
        print("---------------------------------------- k = {0} -----------------------------------------".format(k))
        print("| {0} | {1} | {2} | {3} |".format(tensor[0][0][k], tensor[0][1][k], tensor[0][2][k], tensor[0][3][k]))
        print("| {0} | {1} | {2} | {3} |".format(tensor[1][0][k], tensor[1][1][k], tensor[1][2][k], tensor[1][3][k]))
        print("| {0} | {1} | {2} | {3} |".format(tensor[2][0][k], tensor[2][1][k], tensor[2][2][k], tensor[2][3][k]))
        print("| {0} | {1} | {2} | {3} |".format(tensor[3][0][k], tensor[3][1][k], tensor[3][2][k], tensor[3][3][k]))
        print("----------------------------------------------------------------------------------------")
    print("****************************************************************************************")

def load3dTensors(dataSetPath, paticipant, sVersion): # sVersion = 1 for train and 2 for test
    dataType = ""
    if (sVersion == 1):
        dataType = "train"
    elif (sVersion == 2):
        dataType = "test"
    else:
        print("sVersion has to be 1 (train) or 2 (test).")
        return
    
    if not(33 <= paticipant <= 102):
        print("No data for participant {0}.".format(paticipant))
        return
    
    print("Loading {0} data for participant {1}...".format(dataType, paticipant))
    dataPath_brushingX = "{0}/Participant_{1}/p{1}_a3_s{2}_brushingX.lvm".format(dataSetPath, paticipant, sVersion)
    dataPath_slidingX = "{0}/Participant_{1}/p{1}_a5_s{2}_slidingX.lvm".format(dataSetPath, paticipant, sVersion)
    dataPath_rollingX = "{0}/Participant_{1}/p{1}_a7_s{2}_rollingX.lvm".format(dataSetPath, paticipant, sVersion)
    dNames = ("Time", "TX11", "TX12", "TX13", "TX14", "TX21", "TX22", "TX23", "TX24", "TX31", "TX32", "TX33", "TX34", "TX41", "TX42", "TX43", "TX44")
    dFormats = ("f",) * 17
    dtype1 = {'names': dNames, 'formats': dFormats}
    converters1 = {}
    for i in range(0, 17):
        converters1.update({i: lambda s: float(s.strip().replace(b',', b'.'))})
    
    data_brushing = np.loadtxt(dataPath_brushingX, dtype=dtype1, delimiter='\t', skiprows=0, converters=converters1)
    data_sliding = np.loadtxt(dataPath_slidingX, dtype=dtype1, delimiter='\t', skiprows=0, converters=converters1)
    data_rolling = np.loadtxt(dataPath_rollingX, dtype=dtype1, delimiter='\t', skiprows=0, converters=converters1)
    times_brushing = np.array([col[0] for col in data_brushing])
    times_sliding = np.array([col[0] for col in data_sliding])
    times_rolling = np.array([col[0] for col in data_rolling])
    tensor3d_brushing = np.zeros([4, 4, len(data_brushing)])
    tensor3d_sliding = np.zeros([4, 4, len(data_sliding)])
    tensor3d_rolling = np.zeros([4, 4, len(data_rolling)])
    classNames = ["Brushing a paintbrush", "Sliding a finger", "Rolling a washer"]
    cnt = 1
    for k in range(0, len(data_brushing)):
        for i in range(0, 4):
            for j in range(0, 4):
                tensor3d_brushing[i][j][k] = data_brushing[k][cnt]
                cnt += 1
        cnt  = 1
        
    for k in range(0, len(data_sliding)):
        for i in range(0, 4):
            for j in range(0, 4):
                tensor3d_sliding[i][j][k] = data_sliding[k][cnt]
                cnt += 1
        cnt  = 1
        
    for k in range(0, len(data_rolling)):
        for i in range(0, 4):
            for j in range(0, 4):
                tensor3d_rolling[i][j][k] = data_rolling[k][cnt]
                cnt += 1
        cnt  = 1        
    
    print("Loaded {0} data for participant {1}.".format(dataType, paticipant))
    #return tensor2d, targets, times_brushing, times_sliding, times_rolling, data_brushing, data_sliding, data_rolling
    return tensor3d_brushing, tensor3d_sliding, tensor3d_rolling

# Plot settings
np.set_printoptions(linewidth=10000, suppress=True, formatter={'float': '{: 0.10f}'.format}, edgeitems=500)

dataSetPath = "/Dataset"
paticipant = 33
sVersion = 2

# Load data:
participant_tensor3d_brushing_train_raw, participant_tensor3d_sliding_train_raw, participant_tensor3d_rolling_train_raw = load3dTensors(dataSetPath, paticipant, sVersion)

#print(participant_tensor3d_brushing_train_raw)
#print(len(participant_tensor3d_brushing_train_raw))
#print(np.shape(participant_tensor3d_brushing_train_raw))

tensorFormatted = np.zeros([4*4*len(participant_tensor3d_brushing_train_raw[0][0])])
cnt = 1
cntI = 0
for k in range(0, len(participant_tensor3d_brushing_train_raw[0][0])):
    for i in range(0, 4):
        for j in range(0, 4):
            tensorFormatted[cntI] = participant_tensor3d_brushing_train_raw[i][j][k]
            cntI += 1
            cnt += 1
    cnt  = 1 

print(tensorFormatted)
    
#print(np.shape(tensorFormatted))
#print3dTensor(tensorFormatted, numberOfMatrices)