# cython: boundscheck=False
# cython: wraparound=False

#-------------------------------------------------------------------------------
# Name:       LTMstats.pyx
# Purpose: Calculates LTMLM and LT-Fam statistics
#
# Author:      Tristan J. Hayeck
#
# Acknowledgements:
# LTMLM and this sampler was written by Tristan J. Hayeck, Samuela Pollack, and Alkes Price
# ----------------------------
# Created:     1/2017
# Copyright:
# SOFTWARE COPYRIGHT NOTICE AGREEMENT
# This software and its documentation are copyright (2010) by Harvard University
# and The Broad Institute. All rights are reserved. This software is supplied
# without any warranty or guaranteed support whatsoever. Neither Harvard
# University nor The Broad Institute can be responsible for its use, misuse, or
# functionality. The software may be freely copied for non-commercial purposes,
# provided this copyright notice is retained.
#
#-------------------------------------------------------------------------------

import cython
    #, Utility
import numpy as np
import scipy as sp

cimport gibbsSampler, MatrixFuncs
#from gibbsSampler cimport truncatedMVN

from libc.math cimport log, exp
from scipy import linalg, stats

#TODO user should make sure these match up properly in their file (or change them here)
cdef int CASE = 2
cdef int CONTROL = 1


# cdef double calcHEregressionHerit(double[:,:] GRM, double[:] phenoNorm,bint DEBUG=False):
#     """
#     Calculate the HE heritability estimate
#
#     h2 = sum i!=j GRM_ij *Y_i*Y_j / sum i!=j GRM_ij^2
#
#     :param GRM: the genetic relationship matrix 2D-np.array
#     :param phenoNorm: normalized phenotype np.array
#     :return: the HE heritability estimate
#     """
#
#     cdef:
#         int N,i,j
#         double h2_HE_num,h2_HE_denom, h2_HE
#     N = GRM.shape[0]
#     h2_HE_num = 0
#     h2_HE_denom = 0
#     for i in range(N):
#         for j in range(N):
#             if i != j:
#                 h2_HE_num += GRM[i,j] * phenoNorm[i] * phenoNorm[j]
#                 h2_HE_denom += GRM[i,j] * GRM[i,j]
#                 if DEBUG:
#                     print "Individual i and j have cor %s and phenos %s and %s" % \
#                           (GRM[i,j],phenoNorm[i], phenoNorm[j])
#                     print "Resulting in h2 numerator term of %s and denominator term %s " % \
#                           (h2_HE_num, h2_HE_denom)
#     h2_HE = h2_HE_num/h2_HE_denom
#     return  h2_HE
#
# cdef standardizeVector(double[:] x):
#     '''
#     Standardizes np.array to mean zero variance 1
#     :param x: vector to standardize
#     :return: standardized array
#     '''
#     cdef:
#         double varOfX
#     varOfX = np.var(x)
#     if varOfX == 0:
#         raise ValueError("There's an issue with the memory view, appears to have no variance: %s" % varOfX)
#     else:
#         return (x-np.mean(x))/np.sqrt(varOfX)


#cdef class statManager(object):


cdef double[:] calc(double[:,:] geno, double[:,:] GRM, int[:] caseControlStatuses, double[:] thresholds, int burnIn, int numIters, double maxValue=100, bint useRaoBlackwell=True, bint DEBUG=False):
    '''
    Calculates

    :param geno: genotype data in form of 0,1,2
    :param invCov: the inverse of the covariance matrix
    :param caseControlStatuses: numpy array containing the case control statuses (coding specified at top of script)
    :param thresholds: numpy array of the per individual thresholds
    :param burnIn: int for the number of burn in iterations to take
    :param numIters: int for total number of iterations to take
    :param maxValue: bsolute value of the space to search for the truncated normal ie,
     defaulted to look at a normal truncated to range (-100,100)
    :param useRaoBlackwell: using Rao Blackwell efficiency boost, ie using the average over the random number generated
        to get faster convergence
    :param DEBUG:
    :return:

    '''
    cdef:
        double[:,:] invCov
        double[:] phenoNorm, pheno, outputSamples
        double h2_HE
        int numIndiv,i
    numIndiv = caseControlStatuses.shape[0]
    pheno = np.empty(numIndiv,dtype=np.double)
    pheno =  MatrixFuncs.castIntsToDoubles( caseControlStatuses)
    phenoNorm = MatrixFuncs.standardizeVector(pheno)
    h2_HE = MatrixFuncs.calcHEregressionHerit(GRM, phenoNorm,DEBUG)
    invCov = linalg.inv(GRM)
    if True:

        print "pheno, phenoNorm: "
        for i in range(numIndiv):
            print "%.2f, %.2f" % (pheno[i], phenoNorm[i])
        print "HE regression heritability is: %s" % h2_HE

    outputSamples = gibbsSampler.truncatedMVN(invCov=invCov, caseControlStatuses=caseControlStatuses, \
                                                  thresholds=thresholds, burnIn=burnIn, \
                                                  numIters=numIters, maxValue=maxValue, \
                                                  useRaoBlackwell=useRaoBlackwell, DEBUG=False)


    return outputSamples

def calcstats(double[:,:] geno, double[:,:] GRM, int[:] caseControlStatuses, double[:] thresholds, int burnIn, int numIters, double maxValue=100, bint useRaoBlackwell=True, bint DEBUG=False):
    cdef:
        double[:] phenoNorm, outputSamples
    outputSamples = calc(geno, GRM, caseControlStatuses, thresholds, burnIn, numIters, maxValue, useRaoBlackwell, DEBUG)
    return np.asarray(outputSamples)