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
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

import math
from _operator import indexOf


def load3dTensors(dataSetPath, paticipant): # sVersion = 1 for train and 2 for test
        
    if not(33 <= paticipant <= 102):
        print("No data for participant {0}.".format(paticipant))
        return
    
    #print("Loading data for participant {0}...".format(paticipant))
    
    tensorNames = ["a3_s1_brushingX", "a5_s1_slidingX", "a7_s1_rollingX", "a4_s1_brushingY", "a6_s1_slidingY", "a8_s1_rollingY", "a3_s2_brushingX", "a5_s2_slidingX", "a7_s2_rollingX", "a4_s2_brushingY", "a6_s2_slidingY", "a8_s2_rollingY"]        
    dNames = ("Time", "TX11", "TX12", "TX13", "TX14", "TX21", "TX22", "TX23", "TX24", "TX31", "TX32", "TX33", "TX34", "TX41", "TX42", "TX43", "TX44")
    dFormats = ("f",) * 17
    dtype1 = {'names': dNames, 'formats': dFormats}
    converters1 = {}
    for i in range(0, 17):
        converters1.update({i: lambda s: float(s.strip().replace(b',', b'.'))})
        
    tensors = []
    cnt = 1
    for t in range(0, numberOfTensorsPerParticipant):
        dataPath = "{0}/Participant_{1}/p{1}_{2}.lvm".format(dataSetPath, paticipant, tensorNames[t])
        data = np.loadtxt(dataPath, dtype=dtype1, delimiter='\t', skiprows=0, converters=converters1)
        tensors.append(np.zeros([4, 4, len(data)]))
        for k in range(0, len(data)):
            for i in range(0, 4):
                for j in range(0, 4):
                    tensors[t][i][j][k] = data[k][cnt]
                    cnt += 1
            cnt  = 1
    
    #print("Loaded data for participant {0}.".format(paticipant))
    return tensors


def getWtensor(raw3dTensor): # Returns a smaller tensor
    #print("Getting data window for triggerValue = +-{0}...".format(threshold))
    startindex = 0
    endindex = len(raw3dTensor[0][0])-1
    for k in range(0, len(raw3dTensor[0][0]), Wstep):
        if (startindex == 0):
            for i in range(0, 4):
                for j in range(0, 4):
                    if (raw3dTensor[i][j][k] < mean-threshold or raw3dTensor[i][j][k] > mean+threshold):
                        startindex = k
    
    for k in range(len(raw3dTensor[0][0])-1, startindex, -1*Wstep):
        if (endindex == len(raw3dTensor[0][0])-1):
            for i in range(0, 4):
                for j in range(0, 4):
                    if (raw3dTensor[i][j][k] < mean-threshold or raw3dTensor[i][j][k] > mean+threshold):
                        endindex = k      
     
    wTensor = np.zeros([4, 4, (endindex-startindex)]) 
    for k in range(0, endindex-startindex):
        for i in range(0, 4):
            for j in range(0, 4):
                wTensor[i][j][k] = raw3dTensor[i][j][k+startindex]
        
    #print("Got data window [{0} : {1}], length = {2}.".format(startindex, endindex, len(wTensor[0][0])))
    return wTensor


def getDtensor(w3dTensor, D): # Returns sub-sampled tensor
    #print("Getting sub-sampled tensor for D = {0}...".format(D))
    subStep = int(math.floor(len(w3dTensor[0][0])/D))
    dTensor = np.zeros([4, 4, D]) 
    cnt = 0
    for k in range(0, D):
        for i in range(0, 4):
            for j in range(0, 4):
                dTensor[i][j][k] = w3dTensor[i][j][cnt]
        cnt += subStep
        
    #print("Got sub-sampled tensor with length = {0} for D = {1}.".format(len(dTensor[0][0]), D))
    return dTensor


def unfold(tensor, n):
    return np.rollaxis(tensor, n, 0).reshape(tensor.shape[n], -1)


def unfold3dTensor(tensor): # Return the three mode-n unfolding for a 3d tensor
    X1 = unfold(tensor, 0)
    X2 = unfold(tensor, 1)
    X3 = unfold(tensor, 2)
    return X1, X2, X3


def doSVD(Xmatrix):
    U, S, V = np.linalg.svd(Xmatrix, full_matrices=True)
    return U, S, V


def preprocessParticipant(dataSetPath, paticipant, D): # Returns Vx1, Vx2, and Vx3 for each tensor for the participant
    # Load data:
    participant_tensors_raw = load3dTensors(dataSetPath, paticipant)
    
    Vx1Matrices = []
    Vx2Matrices = []
    Vx3Matrices = []
    for i in range(0, len(participant_tensors_raw)):
        # Get data window (W) tensor:
        participant_tensor_W = getWtensor(participant_tensors_raw[i])
        # sub-sample:
        participant_tensor_D = getDtensor(participant_tensor_W, D)
        # Normalize to [-1, 1]:
        participant_tensor_D_min = participant_tensor_D.min(axis=(0, 1, 2))
        participant_tensor_D_max = participant_tensor_D.max(axis=(0, 1, 2))
        participant_tensor_D_norm = ((2*(participant_tensor_D - participant_tensor_D_min))/(participant_tensor_D_max - participant_tensor_D_min)) - 1
        # Tensor unfolding:
        if normalizeData:
            X1, X2, X3 = unfold3dTensor(participant_tensor_D_norm)
        else:
            X1, X2, X3 = unfold3dTensor(participant_tensor_D)
        # Singular value decomposition:
        Ux1, Sx1, Vx1 = doSVD(X1)
        Ux2, Sx2, Vx2 = doSVD(X2)
        Ux3, Sx3, Vx3 = doSVD(X3)
        Vx1Matrices.append(Vx1)
        Vx2Matrices.append(Vx2)
        Vx3Matrices.append(Vx3)

    return Vx1Matrices, Vx2Matrices, Vx3Matrices

def kernelCompute(Vx1, Vx2, Vx3, Vy1, Vy2, Vy3, sigma, D): # V's are matrices
    V1x = (Vx1*Vx1.T)
    V1y = (Vy1*Vy1.T)
    V1 = V1x - V1y
    k1_b = math.exp((-1/(2*sigma**2)) * np.linalg.norm(V1, ord="fro")**2)
    V2x = (Vx2*Vx2.T)
    V2y = (Vy2*Vy2.T)
    V2 = V2x - V2y
    k2_b = math.exp((-1/(2*sigma**2)) * np.linalg.norm(V2, ord="fro")**2)
    V3x = (Vx3*Vx3.T)
    V3y = (Vy3*Vy3.T)
    V3 = V3x - V3y
    k3_b = math.exp((-1/(2*sigma**2)) * np.linalg.norm(V3, ord="fro")**2)
    
    K_b = k1_b*k2_b*k3_b
    
    #print("K_a = {0} and K_b = {1}".format(0, K_b))
    
    return K_b
    

######### SETTINGS #########

# Dataset location:
dataSetPath = "D:/Google Drive/Dokumenter/DTU/Bachelorproject - FPGA Implementation of Machine Learning/TactileDataset/Dataset"  

# Print settings
np.set_printoptions(linewidth=10000, suppress=False, formatter={'float': '{: 0.10f}'.format})

# Define participants to use, ouliers, and train/test split
allParticipants = range(33, 103) # Entire dataset = range(33, 103)
outliers = [] # Participants to skip
usedParticipants = [participant for participant in allParticipants if participant not in outliers]
#trainingParticipants = [participant for participant in range(33, 84) if participant not in outliers]  # range(33, 84)
#testParticipants = [participant for participant in usedParticipants if participant not in trainingParticipants]
numberOfTensorsPerParticipant = 12 
numberOfTensorsPerClassPerParticipant = 4

# Run setting:
loadVsFromTxts = True # If True the V matrices are loaded from txt files, else the preprocessing will execute (executing the preprocessing will store the resulting V matrices as txt)
onlyPreProcess = False  # If true the program will only preprocess and save V matrices to txt for all tensors for all participants with the below defined values for D and then quit

# Preprocess parameters:
Wstep = 100 # Steps to jump in window detection
mean = 1.645 # Mean of signal idle noise
threshold = 0.03 # Threshold for signal window detection (mean +/- threshold)

# Learning parameters:
learnClasses = [1, 2] # Class encoding: Brushing a paintbrush = 1, Sliding a finger = 2, Rolling a washer = 3
normalizeData = False # The tensorSVM approach does not require normalization
DsToTry = [20, 50] # Make sure all V matrices are available as txt if loadVsFromTxts = True. If loadVsFromTxts = False make sure folders are prepared
CsToTry = [10, 1]
#sigmasToTry = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
sigmasToTry = [8, 9, 10]
K1 = 4
K2 = 4

# .txt store settings:
txtStoreDecimals = 20
txtStoreDelimiter = " "

######### ######### #########

#print("Training participants: {0}".format(trainingParticipants))
#print("Test participants:     {0}".format(testParticipants))

class participant:
    Id = 0
    Vx1Matrices = [] # All Vx1 matrices for the 12 tensors
    Vx2Matrices = [] # All Vx2 matrices for the 12 tensors
    Vx3Matrices = [] # All Vx3 matrices for the 12 tensors
    
    def __init__(self, Id, Vx1Matrices, Vx2Matrices, Vx3Matrices):
        self.Id = Id 
        self.Vx1Matrices = Vx1Matrices
        self.Vx2Matrices = Vx2Matrices
        self.Vx3Matrices = Vx3Matrices
        
        
class binarySVMmodel:
    C = 0
    sigma = 0
    validationErrors = []
    
    def __init__(self, C, sigma):
        self.C = C
        self.sigma = sigma


# Define all models
binarySVMmodels = []
for C in CsToTry:
    for sigma in sigmasToTry:
        binarySVMmodels.append(C, sigma)
            

# Count total iterations to be executed
totalIterations = len(DsToTry)*K1*K2*len(binarySVMmodels)
    
### Learning folds: ###

EgenFinals = []
bestModels = []
for D in DsToTry:  
    # Preprocessing phase:
    processedParticipants = []
    if not loadVsFromTxts: # Then do preprocess to obtain V matrices (and save them as txt files)
        print("******* Preprocessing *********")
        for i in allParticipants:
            if i in usedParticipants:
                normalize = ("True" if normalizeData else "False")
                print("Preprocessing participant {0} for [Normalize = {1}, Wstep = {2}, mean = {3}, threshold = {4}, D = {5}]...".format(i, normalize, Wstep, mean, threshold, D))
                Vx1Matrices, Vx2Matrices, Vx3Matrices = preprocessParticipant(dataSetPath, i, D)
                processedParticipants.append(participant(i, Vx1Matrices, Vx2Matrices, Vx3Matrices))
                tensorNames = ["s1_brushingX", "s1_slidingX", "s1_rollingX", "s1_brushingY", "s1_slidingY", "s1_rollingY", "s2_brushingX", "s2_slidingX", "s2_rollingX", "s2_brushingY", "s2_slidingY", "s2_rollingY"]
                folder = ("Normalized" if normalizeData else "Unnormalized")
                text_file = open("ProcessedParticipants/{0}/ProcessedParticipant_{1}/Id.txt".format(folder, i), "w+")
                text_file.write(str(i))
                text_file.close()
                for t in range(0, numberOfTensorsPerParticipant):
                    np.savetxt("ProcessedParticipants/{0}/ProcessedParticipant_{1}/D_{2}/Vx1_{3}.txt".format(folder, i, D, tensorNames[t]), Vx1Matrices[t], fmt='%.{0}f'.format(txtStoreDecimals), delimiter=txtStoreDelimiter)
                    np.savetxt("ProcessedParticipants/{0}/ProcessedParticipant_{1}/D_{2}/Vx2_{3}.txt".format(folder, i, D, tensorNames[t]), Vx2Matrices[t], fmt='%.{0}f'.format(txtStoreDecimals), delimiter=txtStoreDelimiter)
                    np.savetxt("ProcessedParticipants/{0}/ProcessedParticipant_{1}/D_{2}/Vx3_{3}.txt".format(folder, i, D, tensorNames[t]), Vx3Matrices[t], fmt='%.{0}f'.format(txtStoreDecimals), delimiter=txtStoreDelimiter)
                
            else:
                processedParticipants.append(None)
    else: # Load V matrices from txt files
        print("******* Loading V matrices *********")
        for i in allParticipants:
            if i in usedParticipants:
                normalize = ("True" if normalizeData else "False")
                print("Loading V matrices for participant {0} for [Normalize = {1}, Wstep = {2}, mean = {3}, threshold = {4}, D = {5}]...".format(i, normalize, Wstep, mean, threshold, D))
                tensorNames = ["s1_brushingX", "s1_slidingX", "s1_rollingX", "s1_brushingY", "s1_slidingY", "s1_rollingY", "s2_brushingX", "s2_slidingX", "s2_rollingX", "s2_brushingY", "s2_slidingY", "s2_rollingY"]
                folder = ("Normalized" if normalizeData else "Unnormalized")
                text_file = open("ProcessedParticipants/{0}/ProcessedParticipant_{1}/Id.txt".format(folder, i), "r")
                Id = int(text_file.read())
                text_file.close()
                Vx1Matrices = []
                Vx2Matrices = []
                Vx3Matrices = []
                for t in range(0, numberOfTensorsPerParticipant):
                    Vx1Matrices.append(np.loadtxt("ProcessedParticipants/{0}/ProcessedParticipant_{1}/D_{2}/Vx1_{3}.txt".format(folder, i, D, tensorNames[t])))
                    Vx2Matrices.append(np.loadtxt("ProcessedParticipants/{0}/ProcessedParticipant_{1}/D_{2}/Vx2_{3}.txt".format(folder, i, D, tensorNames[t])))
                    Vx3Matrices.append(np.loadtxt("ProcessedParticipants/{0}/ProcessedParticipant_{1}/D_{2}/Vx3_{3}.txt".format(folder, i, D, tensorNames[t])))
                  
                processedParticipants.append(participant(Id, Vx1Matrices, Vx2Matrices, Vx3Matrices))    
            else:
                processedParticipants.append(None)
        
     
    if not onlyPreProcess:
        
        # Assemble sample matrix for all samples N
        cnt = 0
        allSamples = [] # One sample is one set of V matrices (Vx1, Vx2, and Vx3)
        alltargets = [] # 1 = Class 1, -1 = Class 2
        classIndexes = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
        for participant in processedParticipants:
            for i in classIndexes[learnClasses[0]-1]:
                allSamples.append([participant.Vx1Matrices[i], participant.Vx2Matrices[i], participant.Vx3Matrices[i]])
                alltargets.append(1) # Class 1
            
            for i in classIndexes[learnClasses[1]-1]:
                allSamples.append([participant.Vx1Matrices[i], participant.Vx2Matrices[i], participant.Vx3Matrices[i]])
                alltargets.append(-1) # Class 2    


        # Two-level CV starts:
        iterationcnt = 0
        N = numberOfTensorsPerClassPerParticipant*len(usedParticipants)*2
        
        # Prepare to be able to do splitting in the outer CV loop
        outerSplitSize = int(N/K1)
        outerSplits_test = []
        outerSplitStartIndex = 0
        outerSplitEndIndex = outerSplitSize-1
        for i in range(0, K1):
            outerSplits_test.append(range(outerSplitStartIndex, outerSplitEndIndex))
            outerSplitStartIndex += outerSplitSize
            outerSplitEndIndex += outerSplitSize
           
        testErrors = []  
        Di_test = []
        for cv_k1 in range(0, K1): # The outer CV loop 
            print("Started executing outer CV loop {0}/{1} for D = {2}".format(cv_k1+1, K1, D))
            Di_test = [sample for sample in allSamples if allSamples.indexOf(sample) in outerSplits_test[cv_k1]]
            Di_part = [sample for sample in allSamples if sample not in Di_test]
            Di_test_targets = [target for target in alltargets if alltargets.indexOf(target) in outerSplits_test[cv_k1]]
            Di_part_targets = [target for target in alltargets if target not in Di_test_targets]
            
            Di_part_T = [[Di_part[j][i] for j in range(len(Di_part))] for i in range(len(Di_part[0]))]
        
            # Prepare to be able to do splitting in the inner CV loop
            innerSplitSize = int(len(Di_part)/K2)
            innerSplits_test = []
            innerSplitStartIndex = 0
            innerSplitEndIndex = innerSplitSize-1
            for i in range(0, K2):
                innerSplits_test.append(range(innerSplitStartIndex, innerSplitEndIndex))
                innerSplitStartIndex += innerSplitSize
                innerSplitEndIndex += innerSplitSize
             
            Di_val = []   
            for cv_k2 in range(0, K2): # The inner CV loop
                print("Started executing inner CV loop {0}/{1} in outer CV loop {2}/{3}".format(cv_k2+1, K2, cv_k1+1, K1))
                Di_val = [sample for sample in Di_part if Di_part.indexOf(sample) in innerSplits_test[cv_k2]]
                Di_train = [sample for sample in Di_part if sample not in Di_val]
                Di_val_targets = [target for target in Di_part_targets if Di_part_targets.indexOf(target) in innerSplits_test[cv_k2]]
                Di_train_targets = [target for target in Di_part_targets if target not in Di_val_targets]
                
                Di_train_T = [[Di_train[j][i] for j in range(len(Di_train))] for i in range(len(Di_train[0]))]
                
                modelcnt = 0
                for model in binarySVMmodels: # The 'for every model' loop
                    modelcnt += 1
                    iterationcnt += 1
                    svc = svm.SVC(C=model.C, kernel='precomputed')
                    
                    # Compute all kernel values for training
                    kernel_train = np.zeros([len(Di_train), len(Di_train)])
                    for i in range(0, len(Di_train)):
                        for j in range(0, len(Di_train)):
                            kernel_train[i][j] = kernelCompute(Di_train[i][0], Di_train[i][1], Di_train[i][2], Di_train_T[0][j], Di_train_T[1][j], Di_train_T[2][j], model.sigma, D)
                
                    
                    svc.fit(kernel_train, Di_train_targets) 
        
                    # Compute all kernel values for testing
                    kernel_test = np.zeros([len(Di_val), len(Di_train)])
                    for i in range(0, len(Di_val)):
                        for j in range(0, len(Di_train)):
                            kernel_test[i][j] = kernelCompute(Di_val[i][0], Di_val[i][1], Di_val[i][2], Di_train_T[0][j], Di_train_T[1][j], Di_train_T[2][j], model.sigma, D)
                
                    
                    targets_pred = svc.predict(kernel_test)
                    
                    validationError = 1-accuracy_score(Di_val_targets, targets_pred)
                    model.validationErrors.append(validationError)
                    
                    print("Got validation error (Eval) for model {0}/{1} in CV iteration {2}/{3}: accuracy score (Eval): {4:.4f} with [D = {5}, C = {6:.1f}, sigma = {7:.3f}]".format(modelcnt, len(binarySVMmodels), iterationcnt, totalIterations, validationError, D, model.C, model.sigma))
        
                print("Done executing inner CV loop {0}/{1} in outer CV loop {2}/{3}".format(cv_k2+1, K2, cv_k1+1, K1))
                
            
            # Compute generalization error for each model based on their validation error found doing the inner CV looping
            Egens = []
            for model in binarySVMmodels:
                Egen_model = 0
                for i in range(0, K2):
                    Egen_model += int(len(Di_val)/len(Di_part))*model.validationErrors[i]
                
                Egens.append(Egen_model)
                
            # The best model (in this fold) is the one with the lowest generalization error (Egen)
            bestModel = binarySVMmodels[Egens.index(min(Egens))]
            
            # Training the best model on Di_part
            print("Training the best model in outer fold {0}/{1} [Generalization error = {2:.4f}, D = {3}, C = {4:.1f}, sigma = {5:.3f}] on Di_part...".format(cv_k1+1, K1, min(Egens), D, bestModel.C, bestModel.sigma))
            svc = svm.SVC(C=bestModel.C, kernel='precomputed')
                    
            # Compute all kernel values for training
            kernel_train = np.zeros([len(Di_part), len(Di_part)])
            for i in range(0, len(Di_part)):
                for j in range(0, len(Di_part)):
                    kernel_train[i][j] = kernelCompute(Di_part[i][0], Di_part[i][1], Di_part[i][2], Di_part_T[0][j], Di_part_T[1][j], Di_part_T[2][j], bestModel.sigma, D)
        
            
            svc.fit(kernel_train, Di_part_targets) 

            # Compute all kernel values for testing
            kernel_test = np.zeros([len(Di_test), len(Di_part)])
            for i in range(0, len(Di_test)):
                for j in range(0, len(Di_part)):
                    kernel_test[i][j] = kernelCompute(Di_test[i][0], Di_test[i][1], Di_test[i][2], Di_part_T[0][j], Di_part_T[1][j], Di_part_T[2][j], bestModel.sigma, D)
        
            
            targets_pred = svc.predict(kernel_test)
            
            testError = 1-accuracy_score(Di_test_targets, targets_pred)
            testErrors.append(testError)

            print("Done executing outer CV loop {0}/{1} for D = {2}".format(cv_k1+1, K1, D))
        
        # Compute the final estimate of the generalization error
        EgenFinal = 0
        for i in range(i, K1):
            EgenFinal += int(len(Di_test)/N)*testErrors[i]
            
        
        print("Done executing two-level CV for D = {0}, {1}/{2} -> Final generalization error = {3}".format(D, DsToTry.index(D)+1, len(DsToTry), EgenFinal))
            
        
   
### Results: *** 
if not onlyPreProcess:
    print("\nBest Model: Generalization error = {0:.4f} with [D = {1}, C = {2:.1f}, sigma = {3:.3f}]".format(EgenFinal, D, bestModel.C, bestModel.sigma))
    print("Learning done -> Program ended.")
else:
    print("Preprocessing done -> Program ended.")
