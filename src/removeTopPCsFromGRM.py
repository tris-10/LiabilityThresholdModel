# -------------------------------------------------------------------------------
# Name:        removeTopPCsFromGRM.py
# Purpose:    Removes the top set of PC from a genetic relationship matrix
# Take the top K PC vectors (V matrix) top K eigen values in a diagonal matrix (D matrix)
#
# GRM adjusted = GRM - V[:,0:K]D[0:K,0:K]trans(V[:,0:K])
#
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

from numpy import linalg as LA

import Utility, MatrixUtils

# cd /groups/price/tris/LTFam/Revision/RealPheno/Unperturbed/src/
# module load dev/python/2.7.6
# bsub -q priority -W 12:0 -o consoleRemoveTopPCs.txt python removeTopPCsFromGRM.py
#bsub -q priority -W 12:0 -o consoleRemoveTopPCs.txt python removeTopPCsFromGRM.py -n 1 -o /groups/price/tris/LTFam/Revision/RealPheno/Unperturbed/Data/CARE_PC1_adj.cov



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', action="store", dest="grmFile", type=str,\
                        default="exampleGCTA.cov", \
                        help="file with the GRM")
    parser.add_argument('-o', action="store", dest="grmOutputFile", type=str, \
                        default="examplePCadj.cov", \
                        help="output file with adjusted GRM")
    parser.add_argument('-n', action="store", dest="numPCs", type=int, default=5, \
                        help="number of PCs being adjusted for")
    parser.add_argument('-p', action="store", dest="topPCsOutputFile", type=str, \
                        default="exampleTopPCs.pc", \
                        help="output file with top PCs")

    ##
    args = parser.parse_args()
    covDf = pd.read_csv(args.grmFile, sep=",", header=None, dtype=float)
    GRM = covDf.as_matrix()
    adjustedGRM, topPCs = MatrixUtils.removeTopPCs(GRM,args.numPCs)
    np.savetxt(args.grmOutputFile, adjustedGRM, delimiter=',')
    np.savetxt(args.topPCsOutputFile, topPCs, delimiter=',')

if __name__ == '__main__':
    main()

