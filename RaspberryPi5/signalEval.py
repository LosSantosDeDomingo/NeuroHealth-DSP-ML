##############################################################################################
# Background Information
##############################################################################################
# Publisher(s): Jose Caraballo
# School: 		Florida Atlantic University
# Professor: 	Dr. Hanqi Zhuang
# Sponsor:		Dr. Sree Ranjani Rajendran
##############################################################################################
# References
##############################################################################################
# https://docs.scipy.org/doc/scipy/tutorial/io.html
# https://numpy.org/doc/stable/user/absolute_beginners.html
##############################################################################################
# Purpose of Program
##############################################################################################
#
#
##############################################################################################
# Improvement Status
##############################################################################################
# Immediate Improvement for Current Version:
# --------------------------------------------
# (1) 
#
# Possible Improvement for Later Version:
# --------------------------------------------
# (1)
#
##############################################################################################
# Version Info
##############################################################################################
# Version: 1
# Date Created:  09/13/2025
# Last Revision: 09/13/2025
##############################################################################################
# Packages
##############################################################################################
import scipy.io as sio
import numpy as np
##############################################################################################
# Functions
##############################################################################################
# MATLAB Functions
def getModel(filePath):
    content = sio.loadmat(filePath)
    viewer 	= sio.whosmat(filePath)
    parameters 	= content['compactStruct']
    
    return content, viewer, parameters

def unwrapMetadata(metaldataFile, category):
    unwrappedMetadata = metaldataFile[0][0]
    unwrappedCategory = unwrappedMetadata[category][0][0]
    # unwrappedVariable = unwrappedCategory[variable]
    
    return unwrappedCategory

def convertFloat32(variable):
    return np.array(variable, dtype=np.float32)

# Reassemble Polynomial SVM
def standardizeInput(inputData, mean, std):
    if std != 0:
        standardize = (inputData - mean) / std
    else:
        standardize = 0
        
    return standardize


##############################################################################################
# Driving Code
##############################################################################################
# MATLAB File
modelFilePath = '/home/team5/Desktop/miniModelPolySVM.mat'
modelContent, modelViewer, modelMetadata = getModel(modelFilePath)

# Retrieve Categories
dataSummary 	 = unwrapMetadata(modelMetadata, 'DataSummary')
impl 			 = unwrapMetadata(modelMetadata, 'Impl')
kernelParameters = impl['KernelParameters'][0][0]
classSummary 	 = unwrapMetadata(modelMetadata, 'ClassSummary')

# Retrieve Variables
predNames 			= convertFloat32(dataSummary['PredictorNames'])
numPredictors 		= convertFloat32(dataSummary['NumPredictors'])
mu 					= convertFloat32(impl['Mu']) 	# Mean
sigma 				= convertFloat32(impl['Sigma']) # STD
supportVectors 		= convertFloat32(impl['SupportVectors'])
alpha 				= convertFloat32(impl['Alpha'])
supportVectorLabels = convertFloat32(impl['SupportVectorLabels'])
bias 				= convertFloat32(impl['Bias'])
polyOrder 			= convertFloat32(kernelParameters['PolyOrder'])
scale 				= convertFloat32(kernelParameters['Scale'])
offset 				= convertFloat32(kernelParameters['Offset'])
classNames 			= convertFloat32(classSummary['ClassNames'])

print(predNames)
print(predNames.dtype)
##############################################################################################
# Test Code
##############################################################################################



    



