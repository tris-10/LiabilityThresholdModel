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

import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats
import argparse
import cython
import pyximport
pyximport.install()
import gibbsSampler, LTMstats

#TODO user should make sure these match up properly in their file (or change them here)
CASE = 2
CONTROL = 1

#TODO consider allowing this to be an option
useRaoBlackwell = True


def runToyExample(DEBUG=False):
    """
    Toy, Trio,
        Mother and Child are controls h2 = 0.5, Father and Denis are Cases
        covMat = np.array([ [1,0,0.25,0],
                            [0,1,.25,0],
                            [.25,.25,1,0],
                            [0,0,0,1]])
        threshold = 1
        Mother = -0.36
        Father = 1.50
        Child = -0.08
        Denis = 1.53

        threshold = 2
        Mother =-0.09
        Father = 2.37
        Child = 0.44
        Denis = 2.37

        threshold = 3
        Mother =0.00
        Father =3.28
        Child = 0.80
        Denis = 3.28
    :return:
    """
    covMat = np.array([[1,0,0.25,0],[0,1,.25,0],[.25,.25,1,0],[0,0,0,1]])
    print "Inverting pheno cov inside of runmcmc"
    print covMat
    Tinv = sp.linalg.inv(covMat)
    print "Done inverting pheno cov inside of runmcmc"
    print Tinv
    P = np.array([CONTROL,CASE,CONTROL,CASE],dtype=np.int32)
    print "About to run truncated MVN"
    numIters=10000
    burnIn=1000
    print "With case control status being: "
    print P
    #--------------- Test ---------------#
    maxValue = 100.0
    useRaoBlackwell = True
    pml_threshold_1 = gibbsSampler.calcTruncatedMVN(Tinv, P, np.ones(4,dtype=np.double),burnIn,numIters, maxValue, \
                                                    useRaoBlackwell, DEBUG)
    print "At a threshold of 1: \n%.2f, %.2f, %.2f, %.2f " % \
          (pml_threshold_1[0],pml_threshold_1[1],pml_threshold_1[2],pml_threshold_1[3])
    print"Expecting close to: \n-0.36,  1.50,  -0.08,  1.53\n"
    #--------------- Test ---------------#
    pml_threshold_2 = gibbsSampler.calcTruncatedMVN(Tinv, P, 2*np.ones(4),burnIn, numIters, maxValue, \
                                                    useRaoBlackwell, DEBUG)
    print "At a threshold of 2: \n%.2f, %.2f, %.2f, %.2f " % \
          (pml_threshold_2[0], pml_threshold_2[1], pml_threshold_2[2], pml_threshold_2[3])
    print"Expecting close to: \n-0.09,  2.37,  0.44, 2.37\n"
    #--------------- Test ---------------#
    pml_threshold_3 = gibbsSampler.calcTruncatedMVN(Tinv, P, 3*np.ones(4),burnIn,numIters, \
                                                    maxValue,useRaoBlackwell, DEBUG)
    print "At a threshold of 3: \n%.2f, %.2f, %.2f, %.2f " % \
          (pml_threshold_3[0], pml_threshold_3[1], pml_threshold_3[2], pml_threshold_3[3])
    print"Expecting close to: \n0.00, 3.28, 0.80, 3.28\n"
    #now try varrying thresholds
    #--------------- Test ---------------#
    pml_threshold_1113 = gibbsSampler.calcTruncatedMVN(Tinv, P, np.array([1,1,1,3], dtype=np.double),burnIn, numIters, \
                                                       maxValue, useRaoBlackwell, DEBUG)
    print "At a thresholds of 1,1,1,3: \n%.2f, %.2f, %.2f, %.2f " % \
          (pml_threshold_1113[0], pml_threshold_1113[1], pml_threshold_1113[2], pml_threshold_1113[3])
    print"Expecting close to: \n-0.36,  1.50,  -0.08,  3.28\n"
    #--------------- Test ---------------#
    pml_threshold_1133 = gibbsSampler.calcTruncatedMVN(Tinv, P,np.array([1,1,3,3], dtype=np.double), burnIn, numIters, \
                                                       maxValue, useRaoBlackwell, DEBUG)
    print "At a thresholds of 1,1,3,3: \n%.2f, %.2f, %.2f, %.2f " % \
          (pml_threshold_1133[0], pml_threshold_1133[1], pml_threshold_1133[2], pml_threshold_1133[3])
    print"Expecting close to: \n[-0.30,  1.52, 0.29, 3.29\n"
    print "Finished internal toy tests"



def runToyStatsExample(DEBUG=False):
    """
    Toy, Trio,
        Mother and Child are controls h2 = 0.5, Father and Denis are Cases
        covMat = np.array([ [1,0,0.25,0],
                            [0,1,.25,0],
                            [.25,.25,1,0],
                            [0,0,0,1]])
        threshold = 1
        Mother = -0.36
        Father = 1.50
        Child = -0.08
        Denis = 1.53

        threshold = 2
        Mother =-0.09
        Father = 2.37
        Child = 0.44
        Denis = 2.37

        threshold = 3
        Mother =0.00
        Father =3.28
        Child = 0.80
        Denis = 3.28
    :return:
    """
    covMat = np.array([[1,0,0.25,0],[0,1,.25,0],[.25,.25,1,0],[0,0,0,1]])
    print "Inverting pheno cov inside of runmcmc"
    print covMat
    Tinv = sp.linalg.inv(covMat)
    print "Done inverting pheno cov inside of runmcmc"
    print Tinv
    P = np.array([CONTROL,CASE,CONTROL,CASE],dtype=np.int32)
    print "About to run truncated MVN"
    numIters=10000
    burnIn=1000
    print "With case control status being: "
    print P
    #--------------- Test ---------------#

    pml_threshold_1 = LTMstats.calcstats(geno=covMat, GRM=covMat, caseControlStatuses=P, \
                                    thresholds=np.ones(4,dtype=np.double),burnIn=burnIn,\
                                   numIters=numIters,  useRaoBlackwell=True, DEBUG=DEBUG)


    print "At a threshold of 1: \n%.2f, %.2f, %.2f, %.2f " % \
          (pml_threshold_1[0],pml_threshold_1[1],pml_threshold_1[2],pml_threshold_1[3])
    print"Expecting close to: \n-0.36,  1.50,  -0.08,  1.53\n"

    print "Finished internal toy tests"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action="store", dest="invCovFile", type=str, default="toyInvCov.txt",\
                        help="File containing the inverse covariance matrix")
    parser.add_argument('-c', action="store", dest="caseControlStatusFile", type=str, default="toy.pheno",
                        help="File containing the case control status")
    parser.add_argument('-t', action="store", dest="thresholdFile", type=str, default="toyThresh.txt",\
                        help="File containing threshold for each individual")
    parser.add_argument('-m', action="store", dest="maxValue", type=float, default=10,\
                help="The maximum abosolute value for the support, generating truncated normals ranging from [-max,max]")
    parser.add_argument('-b', action="store", dest="burnIn", type=int, default=1000,\
                        help="number of samples for the burn in")
    parser.add_argument('-n', action="store", dest="numIter", type=int, default=1000,\
                        help="number of iterations after the burn in")
    parser.add_argument('-o', action="store", dest="outputFile", type=str, default="toyOut.txt",\
                        help="Output file with the expected values")
    args = parser.parse_args()
    #TODO to run testing uncomment this method
    runToyExample()

    runToyStatsExample()

    #TODO implement something like this properly
    # print "Reading in inputs"
    # invCov = pd.read_csv(args.invCovFile,header=None).as_matrix()
    # indivData = pd.read_csv(args.caseControlStatusFile,header=None,sep="\t")
    # #caseControlStatuses = np.array(indivData.shape[1], dtype=np.int32)
    # caseControlStatuses = indivData[3].as_matrix()
    # caseControlStatuses = caseControlStatuses.astype(np.int32)
    # thresholds = (pd.read_csv(args.thresholdFile,header=None)[0]).as_matrix()
    # thresholds = thresholds.astype(np.double)
    # print "Running truncated MVN"
    # outputSamples = gibbsSampler.calcTruncatedMVN(invCov=invCov, caseControlStatuses=caseControlStatuses, \
    #                                               thresholds=thresholds, burnIn=args.burnIn, \
    #                                                   numIters=args.numIter, maxValue=args.maxValue, \
    #                                               useRaoBlackwell=useRaoBlackwell, DEBUG=False)
    #
    #
    # print "Outputing results"
    # print outputSamples
    # outputDF = pd.concat([indivData,pd.DataFrame(outputSamples)], axis=1)
    # outputDF.to_csv(args.outputFile,sep="\t",index=None, header=None)
    print "Finished."

if __name__ == '__main__':
    main()


