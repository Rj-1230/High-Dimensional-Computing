import numpy as np;
import random;
import math;
from scipy import stats as st;

def genRandomHV(D):
    randomHV = [0] *D
    if D%2!=0:
        print('Dimension is odd!!')
    else:
        randomIndex = np.random.permutation(D)
        d=int(D/2);
        rand = randomIndex[0:d]
        for j in rand:
            randomHV[j]=1;
    return randomHV;


def initItemMemories(D, MAXL, channels):
    
#     INPUTS:
# %   D           : Dimension of vectors
# %   MAXL        : Maximum amplitude of EMG signal
# %   channels    : Number of acquisition channels
    CiM = {}
    iM = {}
#     rng.default in MATLAB used Mersenne Twister, Python also by default uses the same
#     random.seed(1)
    for i in range(channels):
        iM[i] = genRandomHV(D)
    
    initHV = genRandomHV(D)
    currentHV = initHV
    randomIndex = np.random.permutation(D)
    
    for i in range(MAXL):
        CiM[i]=currentHV
        SP = math.floor(D/2/MAXL)
        startInx = (i*SP) + 1
        endInx = ((i+1)*SP) + 1
        rand = randomIndex[startInx:endInx]
        for j in rand:
            if currentHV[j]==0:
                currentHV[j]=1
            else:
                currentHV[j]=0
        
    return CiM,iM;


def downSampling(data, labels, downSampRate):
    
#     % DESCRIPTION   : apply a downsampling to get rid of redundancy in signals 
# %
# % INPUTS:
# %   data        : input data
# %   labels      : input labels
# %   donwSampRate: the rate or stride of downsampling
# % OUTPUTS:
# %   downSampledData: downsampled data
# %   downSampledLabels: downsampled labels
# %    
    downSampledData = [0]*int(math.ceil(len(data)/downSampRate))
    downSampledLabels = [0]*int(math.ceil(len(data)/downSampRate))
    j=0
#     print(len(downSampledData), len(downSampledLabels))
#     print(len(data), downSampRate)
    for i in range(0,len(data),downSampRate):
        downSampledData [j] = data[i]
        downSampledLabels[j] = labels[i]
        j = j + 1
#         print(i,j)
    np.transpose(downSampledLabels)
    return downSampledData, downSampledLabels;



# Yha pe ek ayga function 
# Leave space for it 


def projBRandomHV( D, F ,q):
    
#     %   D: dim
# %   F: number of features
# %   q: sparsity
    
    proj_m = np.zeros((F, D)) 
    if D%2!=0:
        print('Dimension is odd!!')
    else:
        F_D=F*D
#         probM=rand(F,D);  FxD ka ek matrix with random values
        probM = np.zeros((F, D))
#         print(len(probM))
#         print(len(probM[0]))
        for i in range(F):
            for j in range(D):
                probM[i][j]=random.random()
        p_n1=(1-q)/2
        p_p1=p_n1
        
        for k in range(F):
            for i in range(D):
                if probM[k][i]<p_n1:
                    proj_m[k][i]=-1
                elif (p_n1<=probM[k,i]) and (probM[k,i]<(q+p_n1)):
                    proj_m[k][i]=0
                else:
                    proj_m[k][i]=1;
    return proj_m;


def lookupItemMemeory(itemMemory, rawKey, precision):

# %
# % DESCRIPTION   : recalls a vector from item Memory based on inputs
# %
# % INPUTS:
# %   itemMemory  : item memory
# %   rawKey      : the input key
# %   D           : Dimension of vectors
# %   precision   : precision used in quantization of input EMG signals
# %
# % OUTPUTS:
# %   randomHV    : return the related vector
    key = rawKey * precision
    if key in itemMemory:
        randomHV = itemMemory[key]
    else:
        print('CANNOT FIND THIS KEY :',key)
    return randomHV;


def projItemMemeory (projM, voffeature,ioffeature):
    
# %
# % INPUTS:
# %   projM	: random vector with {-1,0,+1}
# %   voffeature	: value of a feature
# %   ioffeature	: index of a feature
# % OUTPUTS:
# %   randomHV    : return the related vector
    projV=projM[ioffeature]
    h= np.multiply(voffeature,projV)
    randomHV=np.zeros(len(h))
    for i in range(len(h)):
        if h[i]>0:
            randomHV[i]=1
        else:
            randomHV[i]=0;
    return randomHV;


def genTrainData (data, labels, trainingFrac, order):
    
#     % DESCRIPTION   : generates a dataset to train the algorithm using a fraction of the input data 
# %
# % INPUTS:
# %   data        : input data
# %   labels      : input labels
# %   trainingFrac: the fraction of data we should use to output a training dataset
# %   order       : whether preserve the order of inputs (inorder) or randomly select
# %   donwSampRate: the rate or stride of downsampling
# % OUTPUTS:
# %   SAMPL_DATA  : dataset for training
# %   L_SAMPL_DATA: corresponding labels
# %    

    #Note , here labels is a column vector 
    L1 = []
    for i in range(len(labels)):
        if labels[i]==1:
            L1.append(i)
    
    L2 = []
    for i in range(len(labels)):
        if labels[i]==2:
            L2.append(i)
    L3 = []
    for i in range(len(labels)):
        if labels[i]==3:
            L3.append(i)
    L4 = []
    for i in range(len(labels)):
        if labels[i]==4:
            L4.append(i)
    L5 = []
    for i in range(len(labels)):
        if labels[i]==5:
            L5.append(i)
    L6 = []
    for i in range(len(labels)):
        if labels[i]==6:
            L6.append(i)
    
    L7 = []
    for i in range(len(labels)):
        if labels[i]==7:
            L7.append(i)
    
    
    L1 = L1 [0 : math.floor(len(L1) * trainingFrac)]
    L2 = L2 [0 : math.floor(len(L2) * trainingFrac)]
    L3 = L3 [0 : math.floor(len(L3) * trainingFrac)]
    L4 = L4 [0 : math.floor(len(L4) * trainingFrac)]
    L5 = L5 [0 : math.floor(len(L5) * trainingFrac)]
    L6 = L6 [0 : math.floor(len(L6) * trainingFrac)]
    L7 = L7 [0 : math.floor(len(L7) * trainingFrac)]
    
    if order == 'inorder':
        Inx1 = list(range(len(L1)))
        Inx2 = list(range(len(L2)))
        Inx3 = list(range(len(L3)))
        Inx4 = list(range(len(L4)))
        Inx5 = list(range(len(L5)))
        Inx6 = list(range(len(L6)))
        Inx7 = list(range(len(L7)))
    else:
        Inx1 = np.random.permutation(len(L1))
        Inx2 = np.random.permutation(len(L2))
        Inx3 = np.random.permutation(len(L3))
        Inx4 = np.random.permutation(len(L4))
        Inx5 = np.random.permutation(len(L5))
        Inx6 = np.random.permutation(len(L6))
        Inx7 = np.random.permutation(len(L7))
    
#     L_SAMPL_DATA = labels[L1(Inx1)];
    L_SAMPL_DATA=[]
    
    for i in Inx1:
        L_SAMPL_DATA.append(labels[L1[i]])
    for i in Inx2:
        L_SAMPL_DATA.append(labels[L2[i]])
    for i in Inx3:
        L_SAMPL_DATA.append(labels[L3[i]])
    for i in Inx4:
        L_SAMPL_DATA.append(labels[L4[i]])
    for i in Inx5:
        L_SAMPL_DATA.append(labels[L5[i]])
    for i in Inx6:
        L_SAMPL_DATA.append(labels[L6[i]])
    for i in Inx7:
        L_SAMPL_DATA.append(labels[L7[i]])
    
    SAMPL_DATA=[]
    for i in Inx1:
        SAMPL_DATA.append(data[L1[i]])
    for i in Inx2:
        SAMPL_DATA.append(data[L2[i]])
    for i in Inx3:
        SAMPL_DATA.append(data[L3[i]])
    for i in Inx4:
        SAMPL_DATA.append(data[L4[i]])
    for i in Inx5:
        SAMPL_DATA.append(data[L5[i]])
    for i in Inx6:
        SAMPL_DATA.append(data[L6[i]])
    for i in Inx7:
        SAMPL_DATA.append(data[L7[i]])
    
    return L_SAMPL_DATA, SAMPL_DATA;




def hdctrainproj (labelTrainSet, trainSet, CiM, iM, D, N, precision, channels,projM ):
    
#     %
# % DESCRIPTION   : train an associative memory based on input training data
# %
# % INPUTS:
# %   labelTrainSet : training labels
# %   trainSet    : training data
# %   CiM         : cont. item memory (no use)
# %   iM          : item memory
# %   D           : Dimension of vectors
# %   N           : size of n-gram, i.e., window size
# %   precision   : precision used in quantization (no use)
# %
# % OUTPUTS:
# %   AM          : Trained associative memory
# %   numPat      : Number of stored patterns for each class of AM
# %

    AM = {}
    numPat= {}
    for label in range(math.max(labelTrainSet)):
        AM[label] = np.zeros(D)
        numPat[label]= 0
    trainVecList=np.zeros(D)
    i = 1
    label = labelTrainSet[1]
    
    while i < len(labelTrainSet)-N+1:
        j=0
        if labelTrainSet(i) == label:
            for i in range(i,i+N-1):
                new_trainSet[j,:]=trainSet[i,:]
                j=j+1
            ngram = computeNgramproj(new_trainSet, CiM, N, precision, iM, channels,projM);
            trainVecList = trainVecList + ngram
            numPat [labelTrainSet[i+N-1]] = numPat[labelTrainSet[i+N-1]] + 1
            i = i + 1;
        else:
            trainVecList[1 , :] = 3
            AM[label] = st.mode (np.array(trainVecList))
            label = labelTrainSet[i]
            numPat [label] = 0;
            trainVecList=np.zeros(1,D)
            
    l=math.floor(i+(N/2))
    if l > len(labelTrainSet):
        l= length(labelTrainSet)
    AM [labelTrainSet[l]] = st.mode (np.array(trainVecList))   
    for label in range(max(labelTrainSet)):
        print('Class = '+label+'\t sum ='+ np.sum(AM[label])+'\t created \n')
    
    return numPat, AM;








# Function :- computeNgramproj

# Inputs
# buffer = N-1 x 31 here 3x31 matrix
# CiM = Dictionary of 2x10000
# N=4
# precision = 20
# iM =Dicitionary of 31x10000
# channels = 31 
# projM = Vector of vector 31x10000


# def computeNgramproj(buffer, CiM, N, precision, iM, channels,projM):
# # % 	DESCRIPTION: computes the N-gram
# # % 
# # % 	INPUTS:
# # % 	buffer   :  data input
# # % 	iM       :  Item Memory for IDs of the channels
# # %   N        :  dimension of the N-gram
# # %   precision:  precision used in quantization (no use)
# # % 	CiM      :  Continious Item Memory for the values of a channel (no use)
# # %   channels :  numeber of features
# # % 	OUTPUTS:
# # % 	Ngram    :  query hypervector
#     chHV = projItemMemeory(projM, buffer[0, 0],0)
# #     print(type(chHV))

    
# #     print(chHV)
               
                   
# #     print(iM[0]) 10000 values
#     new_chHV = list(chHV)
#     new_iM = list(iM[0])
    
#     new_iM = [int(item) for item in new_iM]
#     new_chHV = [int(item) for item in new_chHV]
    
# #     for i in range(len(chHV)):
# #         chHV[i] = int(chHV[i])
    
# #     for i in range(len(iM[0])):
# #         iM[0][i] = int(iM[0][i])   
    
# #     print(type(new_chHV))
# #     print(type(new_iM))   List
    
# #     print(type(new_chHV[0]))
#     chHV = [0]*len(new_iM)
#     for i in range(len(new_iM)):
#         chHV[i] = new_iM[i]^new_chHV[i]
    
# #     chHV = list(a^b for a,b in zip(list(chHV),list(iM[0])))
# #     print(chHV)
#     v=[]
#     v.append(chHV)
#     if channels>1:
#         for i in range(1,channels):
#             chHV = projItemMemeory (projM, buffer[0, i], i)
#             new_chHV = list(chHV)
#             new_iM = list(iM[i])
#             new_iM = [int(item) for item in new_iM]
#             new_chHV = [int(item) for item in new_chHV]
            
#             chHV = [0]*len(new_iM)
#             for j in range(len(new_iM)):
#                 chHV[j] = new_iM[j]^new_chHV[j]
            
#             if i == 1:
#                 ch2HV=chHV
#             v.append(chHV)
        
#         new_chHV = list(chHV)
# #         print(ch2HV)
#         new_ch2HV = list(ch2HV)
#         new_ch2HV = [int(item) for item in new_ch2HV]
#         new_chHV = [int(item) for item in new_chHV]
        
#         chHV = [0]*len(new_ch2HV)
#         for i in range(len(new_ch2HV)):
#             chHV[i] = new_ch2HV[i]^new_chHV[i]
#         v.append(chHV)
    
#     if channels==1:
#         Ngram = v;
#     else:
#         v_arr = np.array(v)
#         Ngram = st.mode(v_arr)
    
#     v=[]
#     for i in range(1,N):
#         chHV = projItemMemeory (projM, buffer[i, 0], 0)
# #         print(chHV)
# #         print(iM[0])
#         new_chHV = list(chHV)
#         new_iM = list(iM[0])
    
#         new_iM = [int(item) for item in new_iM]
#         new_chHV = [int(item) for item in new_chHV]
#         chHV = [0]*len(new_iM)
#         for j in range(len(new_iM)):
#             chHV[j] = new_iM[j]^new_chHV[j]
# #         chHV = chHV ^ iM[0]
#         ch1HV = chHV
#         v.append(chHV)
#         if channels>1:  
#             for j in range(1,channels):
#                 chHV = projItemMemeory (projM, buffer[i, j], j)
#                 new_chHV = list(chHV)
#                 new_iM = list(iM[j])
#                 new_iM = [int(item) for item in new_iM]
#                 new_chHV = [int(item) for item in new_chHV]
            
#                 chHV = [0]*len(new_iM)
#                 for k in range(len(new_iM)):
#                     chHV[k] = new_iM[k]^new_chHV[k]
# #                 chHV = chHV ^ iM[j]
#                 if j == 1:
#                     ch2HV=chHV; 
#                 v.append(chHV)
            
# #             chHV = xor(chHV , ch2HV);
# #             v = v+chHV; 
#             new_chHV = list(chHV)
# #         print(ch2HV)
#             new_ch2HV = list(ch2HV)
#             new_ch2HV = [int(item) for item in new_ch2HV]
#             new_chHV = [int(item) for item in new_chHV]
        
#             chHV = [0]*len(new_ch2HV)
#             for k in range(len(new_ch2HV)):
#                 chHV[k] = new_ch2HV[k]^new_chHV[k]
#             v.append(chHV)
           
#         if channels==1:
#             record = v;          
#         else:
#             v_arr = np.array(v)
#             record = st.mode(v_arr)
           
# #         print(Ngram)  ModeResult(mode=array([[1, 0, 1, ..., 0, 0, 1]]), count=array([[19, 17, 21, ..., 19, 20, 19]]))

#         print(type(Ngram))
#         print(len(Ngram))
#         print(len(Ngram[0]))
        
# #         print((Ngram[0][0])) [1 0 0 ... 1 1 0]
# #         print(Ngram[1][0])  [18 20 18 ... 19 20 16]

# #         for i in range(len(Ngram)):
# #             Ngram[i] = np.roll(Ngram[i], 1)
# #         for i in range(len(Ngram[0])):
# #             Ngram[:,i] = np.roll(Ngram[:,i], 1)
# #         new_chHV = list(Ngram)
        
        
# # #         print(ch2HV)
# #         new_ch2HV = list(record)
# #         new_ch2HV = [int(item) for item in new_ch2HV]
# #         new_chHV = [int(item) for item in new_chHV]        
# #         Ngram = [0]*len(new_chHV)
# #         for k in range(len(new_chHV)):
# #             Ngram[k] = new_ch2HV[k]^new_chHV[k]
# # #         Ngram = 
# # #         Ngram = (circshift (Ngram, [1,1])) ^ record 
# # #         cicrcular shift can be done using np.roll but I didn't get how Ngram is 2D
    
# #     ye circular shift kyu hua, reason?????
    
#     return Ngram;
 
def computeNgramproj(buffer, CiM, N, precision, iM, channels,projM):
# % 	DESCRIPTION: computes the N-gram
# % 
# % 	INPUTS:
# % 	buffer   :  data input
# % 	iM       :  Item Memory for IDs of the channels
# %   N        :  dimension of the N-gram
# %   precision:  precision used in quantization (no use)
# % 	CiM      :  Continious Item Memory for the values of a channel (no use)
# %   channels :  numeber of features
# % 	OUTPUTS:
# % 	Ngram    :  query hypervector
#     print("Line 0 crossed")

#     chHV = projItemMemeory(projM, buffer[0, 0],0)
# #     print("Line 1 crossed")
#     chHV = chHV^iM[0]
#     v=[]
#     v=v+chHV
#     if channels>1:
#         for i in range(1,channels):
#             chHV = projItemMemeory (projM, buffer[0, i], i)
#             chHV = chHV ^ iM[i]
#             if i == 1:
#                 ch2HV=chHV
#             v = v+chHV
    
#         chHV = chHV ^ ch2HV
#         v = v+chHV
    
#     if channels==1:
#         Ngram = v;
#     else:
#         v_arr = np.array(v)
#         Ngram = st.mode(v_arr)
    
#     v=[]
#     for i in range(1,N):
#         chHV = projItemMemeory (projM, buffer[i, 0], 0)
        
#         chHV = chHV ^ iM[0]
#         ch1HV = chHV
#         v = v+chHV
#         if channels>1:  
#             for j in range(1,channels):
#                 chHV = projItemMemeory (projM, buffer[i, j], j)
#                 chHV = chHV ^ iM[j]
#                 if j == 1:
#                     ch2HV=chHV; 
#                 v = v+chHV  
#             chHV = xor(chHV , ch2HV);
#             v = v+chHV; 
#         if channels==1:
#             record = v;          
#         else:
#             v_arr = np.array(v)
#             record = st.mode(v_arr)
            
#         Ngram = (circshift (Ngram, [1,1])) ^ record 
# #         cicrcular shift can be done using np.roll but I didn't get how Ngram is 2D

    chHV = projItemMemeory(projM, buffer[0, 0],0)
#             print("Line 1 crossed")
#             print(len(chHV ))
#             print(len(iM[0]))
    k=0
    for k in range(len(chHV)):
            chHV[k]=chHV[k]*iM[0][k]
#             chHV = chHV^iM[0]
    v=[]
    v.append(chHV)
    if channels>1:
        for i in range(1,channels):
            chHV = projItemMemeory (projM, buffer[0, i], i)
            k=0
            for k in range(len(chHV)):
                chHV[k]=chHV[k]*iM[i][k]
#                     chHV = chHV ^ iM[i]
            if i == 1:
                ch2HV=chHV
            v = v+chHV

        k=0
        for k in range(len(chHV)):
            chHV[k]=chHV[k]*ch2HV[k]
#                 chHV = chHV ^ ch2HV
        v = v+chHV

    if channels==1:
        Ngram = v;
    else:
        v_arr = np.array(v)
        Ngram = st.mode(v_arr)

    v=[]
#             print(type(v))
    for i in range(1,N):
#                 print(i)
        chHV = projItemMemeory (projM, buffer[i, 0], 0)
#                 print(len(chHV ))
#                 print(len(iM[0]))
        k=0
        for k in range(len(chHV)):
            chHV[k]=chHV[k]*iM[0][k]
#                 chHV = chHV ^ iM[0]
        ch1HV = chHV
#                 print(type(v))
        v.append(chHV)
#                 print(channels)
        if channels>1:  
            for j in range(1,channels):
                chHV = projItemMemeory (projM, buffer[i, j], j)
                k=0
                for k in range(len(chHV)):
                    chHV[k]=chHV[k]*iM[j][k]
#                         chHV = chHV ^ iM[j]
                if j == 1:
                    ch2HV=chHV; 

#                         print(type(v))

                v = v+chHV  
#                     chHV = xor(chHV , ch2HV);
            k=0
            for k in range(len(chHV)):
                chHV[k]=chHV[k]*ch2HV[k]
#                     print(type(v))

            v = v+chHV;
#                     print(type(v))

        if channels==1:
            record = v;          
        else:
#                     v_arr=v
            v_arr = np.array(v)
#                     print(type(v))
            record = st.mode(v_arr)
#                 print(type(Ngram))
        v=v.tolist()
#                 print("Here hu mai")
        record_arr = record[0][0]
        Ngram_arr = Ngram[0][0]


#                 print(Ngram_arr)
#         for p in range(0, 1):
        last_elem = Ngram_arr[len(Ngram_arr)-1]
        for j in range(len(Ngram_arr)-1,-1,-1):
            Ngram_arr[j] = Ngram_arr[j-1]
        Ngram_arr[0] = last_elem

#                 print(Ngram_arr)

        for x in range(len(Ngram_arr)):
            Ngram_arr[x]=int(Ngram_arr[x])^int(record_arr[x])


#         print(Ngram_arr)
#                 print(len(record_arr))
#                 print(len(Ngram_arr))

#                 print(record[0][0])
#                 Ngram = (circshift (Ngram, [1,1])) ^ record 
            
#         cicrcular shift can be done using np.roll but I didn't get how Ngram is 2D 
    
    return Ngram_arr;
 
    
    
    
def hdctrainproj (labelTrainSet, trainSet, CiM, iM, D, N, precision, channels,projM ):
    
#     %
# % DESCRIPTION   : train an associative memory based on input training data
# %
# % INPUTS:
# %   labelTrainSet : training labels
# %   trainSet    : training data
# %   CiM         : cont. item memory (no use)
# %   iM          : item memory
# %   D           : Dimension of vectors
# %   N           : size of n-gram, i.e., window size
# %   precision   : precision used in quantization (no use)
# %
# % OUTPUTS:
# %   AM          : Trained associative memory
# %   numPat      : Number of stored patterns for each class of AM
# %

    AM = {}
    numPat= {}
#     print(max(labelTrainSet))
    for label in range(int(max(labelTrainSet))):
        AM[label] = np.zeros(D)
        numPat[label]= 0
#     print(AM)   {0: array([0., 0., 0., ..., 0., 0., 0.]), 1: array([0., 0., 0., ..., 0., 0., 0.])}
#     print(numPat)   {0: 0, 1: 0}
    trainVecList=np.zeros(D)
    z = 0
    label = labelTrainSet[0]
    
    print(len(labelTrainSet))    #94
    while z < len(labelTrainSet)-N+1:     #91
#         print(len(labelTrainSet)-N+1)
        print('Value of z is : ',z)
#         print(len(trainSet))
        if labelTrainSet[z] == label:
            W = len(trainSet[0])
            new_trainSet = np.zeros((N, W))
            j=0
            p=z
            for p in range(p,p+N-1):
#                 print(i)
                new_trainSet[j,:]=trainSet[p]
                j=j+1
        
            ngram = computeNgramproj(new_trainSet, CiM, N, precision, iM, channels,projM);
#             print("Hello")
            trainVecList = trainVecList + ngram
#             print(labelTrainSet[z+N-1])
            numPat [int(labelTrainSet[z+N-1])-1] = int(numPat[labelTrainSet[z+N-1]-1]) + 1
            z = z + 1;
#             print(z)
        else:
            trainVecList[0] = 3
            AM[label] = st.mode (np.array(trainVecList))
            label = labelTrainSet[z]
            numPat [label] = 0
            trainVecList=np.zeros(D)
            
    l=math.floor(z+(N/2))
    if l > len(labelTrainSet):
        l= length(labelTrainSet)
    AM [labelTrainSet[l]] = st.mode (np.array(trainVecList))   
    for label in range(int(max(labelTrainSet))+1):
        print('Class = ',label,'\t sum =', np.sum(AM[label]),'\t created \n')
    
    return numPat, AM;




def hdcpredictproj(labelTestSet1, testSet1, labelTestSet2, testSet2,labelTestSet3, testSet3,AM, CiM, iM, D, N, precision, classes, channels1, channels2, channels3,projM1, projM2, projM3):
    
#     %
# % DESCRIPTION   : test accuracy based on input testing data
# %
# % INPUTS:
# %   labelTestSet: testing labels
# %   testSet     : EMG test data
# %   AM          : Trained associative memory
# %   CiM         : Cont. item memory (no use)
# %   iM          : item memory
# %   D           : Dimension of vectors
# %   N           : size of n-gram, i.e., window size 
# %   precision   : precision used in quantization (no use)
# %
# % OUTPUTS:
# %   accuracy    : classification accuracy for all situations
# %   accExcTrnz  : classification accuracy excluding the transitions between gestutes
# %
    correct = 0
    numTests = 0
    tranzError = 0

    for i in range(len(testSet1)-N+1):
        numTests = numTests + 1
    #     print(numTests)    1
        #actualLabel[i : i+N-1,:] = mode(labelTestSet1 (i : i+N-1));
        W = len(testSet1[0]) 
    #     print(W)  31
        new_testSet1 = np.zeros((N-1, W))
        j=0
        p=i
        print(i)
        for p in range(p,p+N-1):
            new_testSet1[j,:]=testSet1[p]
            j=j+1
        print(i)
        sigHV1 = functions.computeNgramproj (new_testSet1, CiM, N, precision, iM, channels1,projM1);
        W = len(testSet2[0])
        new_testSet2 = np.zeros((N-1, W))
        j=0
        p=i
        for p in range(p,p+N-1):
    #                 print(i)
            new_testSet2[j,:]=testSet2[p]
            j=j+1
        print(i)
        print(new_testSet2)
        sigHV2 = functions.computeNgramproj (new_testSet2, CiM, N, precision, iM, channels2,projM2);
        W = len(testSet3[0])
        new_testSet3 = np.zeros((N-1, W))
        j=0
        p=i
        for p in range(p,p+N-1):
    #                 print(i)
            new_testSet3[j,:]=testSet3[p]
            j=j+1
        sigHV3 = functions.computeNgramproj (new_testSet3, CiM, N, precision, iM, channels3,projM3);

        sigHV=st.mode(sigHV1+sigHV2+sigHV3)
        predict_hamm = scipy.spatial.distance.hamming(sigHV, AM, classes);
        predicLabel[i : i+N-1] = predict_hamm;

        if predict_hamm == actualLabel[i]:
            correct = correct + 1;
        elif labelTestSet1 (i) != labelTestSet1(i+N-1):
            tranzError = tranzError + 1;
            
        accuracy = correct / numTests;
        accExcTrnz = (correct + tranzError) / numTests;
    
    return accExcTrnz, accuracy, predicLabel, actualLabel
