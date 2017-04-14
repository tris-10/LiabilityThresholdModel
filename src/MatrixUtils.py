# -------------------------------------------------------------------------------
# Name:        MatrixUtils.py
# Purpose:
#
# Author:      Tristan Hayeck
#
# Created:     12/2016
# Copyright:   (c) Hayeck 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import argparse,math, sys, os,random, datetime
import numpy as np
import pandas as pd
from scipy import linalg
from scipy import optimize
from scipy import stats

from sklearn.decomposition import RandomizedPCA
from numpy import linalg as LA

import Utility






def divideMatrixRowByVectorElement(theMatrix, theVector):
    """
    Wrapper function to
    adapted from http://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element


    data = np.array([[1,1,1],[2,2,2],[3,3,3]])
    vector = np.array([1,2,3])
    gives:
    div_result = [[1,1,1], [1,1,1], [1,1,1]]

    :param theMatrix: The matrix being operated on
    :param theVector: The vector to divide by each element
    :return:
    """
    return (theMatrix.T / theVector).T


def multMatrices(A,B,transA=False,transB=False,DEBUG=True):
    """
    Wrapper function for linalg.fblas.dgemm
    :param A: First Matrix
    :param B: Second Matrix
    :return: multiplied matrix
    """
    try:
        linalg.blas
    except AttributeError:
        Utility.PROGRESS("Warning, slow processing, cannot use linalg.blas")
        return np.dot(A,B)
    return linalg.blas.dgemm(alpha=1.,a=A,b=B,trans_a=transA,trans_b=transB)


def removeTopPCs(GRM,numberPcs,testing=False):
    """
    Removes the top set of PC from a genetic relationship matrix

    Take the top K PC vectors (V matrix) top K eigen values in a diagonal matrix (D matrix)

    GRM adjusted = GRM - V[:,0:K]D[0:K,0:K]trans(V[:,0:K])

    :param GRM: Genetic relationship matrix (or whatever symetric matrix) to remove PCs from
    :param numberPcs: number of PCs to remove from GRM
    :param testing: flag to print more details while testing
    :return:
    """
    eigenValues, eigenVectors = linalg.eig(GRM)
    #now sort by top eigen values
    idx = eigenValues.argsort()[::-1]
    eigenValuesSorted = eigenValues[idx]
    eigenVectorsSorted = eigenVectors[:, idx]
    if testing:
        print "Eigen Values: %s" % (np.diag(eigenValuesSorted)[0:numberPcs, 0:numberPcs])
        print "Eigen Vectors: \n%s" % (eigenVectorsSorted[:, 0:numberPcs])
    intermediateMat = multMatrices(eigenVectorsSorted[:, 0:numberPcs],np.diag(eigenValuesSorted)[0:numberPcs, 0:numberPcs])
    grmAdjusted = GRM - multMatrices(intermediateMat,  eigenVectorsSorted[:, 0:numberPcs], False, True)
    return grmAdjusted, eigenVectorsSorted[:, 0:numberPcs]


def binomVar(p, DEBUG=False):
    """
    calculates 2*p*(1-p) with some error checking
    :param p:
    :param DEBUG:
    :return:
    """
    if p == 0 or p == 1:
        return np.nan
    elif p>1 or p < 0:
        err = "Getting invalid allele freq of: %s" % p
        raise Utility.MyRuntimeException(err)
    else:
        return (2*p*(1-p))

def standardizeGenoRow(x,DEBUG=True):
    """
    Takes an array corresponding to a single SNP and returns the standardized value
    :param x: array of geno type values
    :param DEBUG:
    :return: (x-2p)/sqrt(2*p(1-p))
    """
    allealFreq = np.nanmean(x)/2
    var = binomVar(allealFreq)
    xNoNull = x
    xNoNull = np.nan_to_num(xNoNull)
    if np.isnan(var):
        Utility.PROGRESS("Warning, cannot get valid SNP variance. Allele freq 0 or 1",DEBUG)
        return np.zeros(x.size)
    else:
        xNorm = (xNoNull-2*allealFreq)/np.sqrt(var)
        return xNorm

def convertNullsToNan(X,nullValue=9, DEBUG=False):
    """
    Replaces the nullValue in (frequently coded as 9) the matrix and explicitly
    sets it to np.nan
    :param X: np.array to modify
    :param nullValue: typically something like 9 or -1 null coded values
    :param DEBUG:
    :return: none, should be passed by reference
    """
    Utility.PROGRESS("Removing nulls with values %s"%nullValue)
    np.place(X,X==nullValue,np.nan)

def standardizeGeno(X,DEBUG):
    """
     Standardizes the geno matrix to look more like:
        (X-2p)/sqrt(2*p(1-p))
    This also removes null values (9) or np.nan values

    :param X: genotype matrix
    :param DEBUG:
    :return: Xnorm = (X-2p)/sqrt(2*p(1-p))
    """
    convertNullsToNan(X)
    Xnorm =  np.apply_along_axis( standardizeGenoRow, 1,X)
    return Xnorm


def calculateGRM(X,alreadyNormalized=True,DEBUG=True):
    '''
    Calculate the genetic relationship matrix 1/M * X.T x X
    :param X:
    :param alreadyNormalized:
    :param DEBUG:
    :return:
    '''
    GRM = np.nan
    if not alreadyNormalized:
        Xnorm = standardizeGeno(X,DEBUG)
        GRM = multMatrices(Xnorm.T,Xnorm)/float(Xnorm.shape[0])
    else:
        GRM = multMatrices(X.T,X)/float(X.shape[0])
    grmConditionNum=np.nan
    if DEBUG:
        grmConditionNum = np.linalg.cond(GRM)

    Utility.PROGRESS("The condition number (relative error) for the GRM is %s with dimensions %s"
                     %(grmConditionNum,GRM.shape))
    return GRM

def calcHEregressionHerit(GRM, phenoNorm,DEBUG=False):
    """
    Calculate the HE heritability estimate

    h2 = sum i!=j GRM_ij *Y_i*Y_j / sum i!=j GRM_ij^2

    :param GRM: the genetic relationship matrix 2D-np.array
    :param phenoNorm: normalized phenotype np.array
    :return: the HE heritability estimate
    """
    N = GRM.shape[0]
    h2_HE_num = 0
    h2_HE_denom = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                h2_HE_num += GRM[i,j]*phenoNorm[i]*phenoNorm[j]
                h2_HE_denom += GRM[i,j]*GRM[i,j]
                if DEBUG:
                    print "Individual i and j have cor %s and phenos %s and %s" % \
                          (GRM[i,j],phenoNorm[i], phenoNorm[j])
                    print "Resulting in h2 numerator term of %s and denominator term %s " % \
                          (h2_HE_num, h2_HE_denom)
    h2_HE = h2_HE_num/h2_HE_denom
    return  h2_HE



def testCalcHEreg(cor,pheno):
    GRM = np.array([[1,cor,0,cor,1,0,0,0,1]]).reshape(3,3)
    phenoNorm = standardizePhenoRow(np.array(pheno))
    h2_HE = calcHEregressionHerit(GRM,phenoNorm,True)
    print "The GRM is: "
    print GRM
    print "With pheno: "
    print phenoNorm
    print "resulting in a heritability of %s" % h2_HE


def standardizePhenoRow(x):
    '''
    Standardizes np.array to mean zero variance 1
    :param x: np.array to standardize
    :return: standardized array
    '''
    if x.var()==0:
        raise Utility.MyRuntimeException("There's an issue with the phenotype, appears to have no variance: %s" % x)
    else:
        return (x-x.mean())/np.sqrt(x.var())

def standardizePheno(pheno,DEBUG):
    '''
    Standardized the repeated measurement phenotypes to mean zero and variance 1
    :param pheno: 2d-np.array with the phenotype measurements
    :param DEBUG: standardized 2d np.array with the phentoype measurements
    :return:
    '''
    phenoNorm = np.apply_along_axis(standardizePhenoRow,1,pheno)
    Utility.PROGRESS("Finished standardizing the phenotypes")
    return phenoNorm


def standGeno_V2(X):
    '''
    second attempt, appears slower
    :param X:
    :return:
    '''
    M,N = X.shape
    for i in range(M):
        rowMean = X[i,True - np.isnan(X[i,:])].mean()
        X[i,np.isnan(X[i,:])] = rowMean
        vr = rowMean*(1-rowMean)
        if vr == 0:
            X[:,i] = np.zeros(M)
        X[i,:] = (X[i,:] - rowMean) / np.sqrt(vr)
    return X


