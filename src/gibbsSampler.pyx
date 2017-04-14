# cython: boundscheck=False
# cython: wraparound=False

#-------------------------------------------------------------------------------
# Name:       gibbsSampler.pyx
# Purpose: Samples from a multivariate truncated normal.
#
# Author:      Tristan J. Hayeck
#
# Acknowledgements:
# LTMLM and this sampler was written by Tristan J. Hayeck, Samuela Pollack, and Alkes Price
# ----------------------------
# Created:     22/05/2015
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
import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats
from libc.math cimport log, exp

#TODO user should make sure these match up properly in their file (or change them here)
cdef int CASE = 2
cdef int CONTROL = 1


cdef double sampleTruncNorm(int caseControlStatus, double mean, double standardDeviation, double threshold, double maxValue, bint DEBUG):
    """
    Random variable sampled from a univariate truncated normal distribution, where cases are taken from
    above the threshold and controls below the threshold.

    Also, note this is a wrapper function for stats.truncnorm.rvs and based on the documentation
    http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.truncnorm.html
    The standard form of this distribution is a standard normal truncated to the range [a, b]
    notice that a and b are defined over the domain of the standard normal. To convert clip values
    for a specific mean and standard deviation, use:

    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

    :param caseControlStatus: Case Control status indicated sampling above or below the
    threshold
    :param threshold: threshold for the truncation
    :param maxValue: absolute value of the space to search for the truncated normal ie,
     defaulted to look at a normal truncated to range (-100,100)
    :param DEBUG:
    :return: numpy array sample of rancom variables taken from the truncated normal
    """
    cdef:
        double a,b, value
    #value = np.NaN
    try:
        if caseControlStatus == CASE:
            a, b = (threshold - mean) / standardDeviation, (maxValue - mean) / standardDeviation
            value = stats.truncnorm.rvs(a, b, loc=mean, scale=standardDeviation)
        elif caseControlStatus == CONTROL:
            a, b = (-1.0 * maxValue - mean) / standardDeviation, (threshold - mean) / standardDeviation
            value = stats.truncnorm.rvs(a, b, loc=mean, scale=standardDeviation)
        else:
            if DEBUG:
                print "Warning case control status not set for individual"
            value = standardDeviation * np.random.randn() + mean
    except ValueError:
            print "Error values for range of truncated normal causing problems %s and %s" % (a,b)
            print "with inputs caseControlStatus=%s,  mean=%s, standardDeviation=%s, threshold=%s, MAX =%s" \
                  %(caseControlStatus, mean, standardDeviation, threshold, maxValue)
            print ValueError
    return value

cdef double[:] truncatedMVN(double[:,:] invCov, int[:] caseControlStatuses, double[:] thresholds, int burnIn, int numIters, double maxValue, bint useRaoBlackwell, bint DEBUG):
    '''
    Calculates the expected values from a multivariate truncated normal used to liability threshold analysis.

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
    :return: numpy array of the per individual the expected values from a multivariate truncated normal
    used to liability threshold analysis.
    '''
    cdef:
        int numIndiv
        int i,j,k = 0
        double runningSum, conditionalMean, conditionalSigma
        double[:] phi, newPhi, phiOld,phiSum,rbphi,outputSamples,rbPhiSum

    numIndiv = invCov.shape[0]
    #Gibbs sampler terms for new and old iterations
    phi = np.empty((numIndiv),dtype=np.double)
    newPhi = np.empty((numIndiv),dtype=np.double)
    phiOld = np.empty((numIndiv),dtype=np.double)
    phiSum = np.empty((numIndiv),dtype=np.double)
    phiSum[:]  = 0
    #Rao-Blackwell terms for faster convergence
    rbphi = np.empty((numIndiv),dtype=np.double)
    rbPhiSum = np.empty((numIndiv),dtype=np.double)
    rbPhiSum[:]  = 0
    #output
    outputSamples = np.empty((numIndiv),dtype=np.double)

    if DEBUG:
        print "About to run truncatedMVN with terms initialized"
    for i in range(numIndiv):
        #Setting initial values for indiiduals, set to marginal truncated normal value for phi to start
        phi[i] = sampleTruncNorm(caseControlStatus=caseControlStatuses[i], mean=0.0, standardDeviation=1.0, \
                                 threshold=thresholds[i], maxValue=maxValue, DEBUG=True)
    if DEBUG:
        print "finished initializing to univarite sample, now to run sampler"
    for i in range(numIters+burnIn):
        #TODO consider converting to matrix operations
        for j in range(numIndiv):
            runningSum = 0.0
            for k in range(numIndiv):
                if k != j:
                    runningSum += invCov[j, k] * phi[k]
            #helpful identity to understand derivation: http://grizzly.la.psu.edu/~hbierens/PARTITIONED_MATRIX.PDF
            conditionalMean = -runningSum / invCov[j, j]
            conditionalSigma  = 1.0/np.sqrt(invCov[j, j])
            newPhi[j] = sampleTruncNorm(caseControlStatuses[j], conditionalMean, conditionalSigma, thresholds[j], maxValue, DEBUG)
            rbphi[j] = sampleTruncNorm(caseControlStatuses[j], conditionalMean, conditionalSigma, thresholds[j], maxValue, DEBUG)
        #end of looping through individuals
        phiOld = np.copy(phi)
        phi = np.copy(newPhi)
        #only store after the burn in
        if (i < burnIn) :
            continue
        # update phisum and meansum
        for j in range(numIndiv):
            phiSum[j] += phi[j]
            rbPhiSum[j] += rbphi[j]
    if DEBUG:
        print "done iterating, now to calculate the posterior means"
    if(useRaoBlackwell):
        for j in range(numIndiv):
            outputSamples[j] = rbPhiSum[j]/numIters
    else:
        for j in range(numIndiv):
            outputSamples[j]= phi[j]
            if(j == numIndiv - 1):
                outputSamples[j]= phiOld[j]
    if DEBUG:
        print "Done with runmcmc"
    return outputSamples


def calcTruncatedMVN(double[:,:] invCov, int[:] caseControlStatuses, double[:] thresholds, int burnIn, int numIters, double maxValue=100, bint useRaoBlackwell=True, bint DEBUG=False):
    '''
    Calculates the expected values from a multivariate truncated normal used to liability threshold analysis.

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
    :return: numpy array of the per individual the expected values from a multivariate truncated normal
    used to liability threshold analysis.
    '''
    cdef:
        double[:] posteriorMeans
        int numIndiv,i

    if DEBUG:
        print "running in debug mode for calcTruncatedMVN"
    posteriorMeans =truncatedMVN(invCov, caseControlStatuses, thresholds, burnIn, numIters, maxValue, useRaoBlackwell, DEBUG)
    if DEBUG:
        numIndiv = invCov.shape[0]
        print "status posterior "
        for i in range(numIndiv):
            if caseControlStatuses[i] == CASE:
                print "Case, %s" % posteriorMeans[i]
            elif caseControlStatuses[i] == CONTROL:
                print "Control, %s" % posteriorMeans[i]
            else:
                print "Unknown, %s" % posteriorMeans[i]
    return np.asarray(posteriorMeans)