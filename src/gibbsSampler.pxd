cdef double[:] truncatedMVN(double[:,:] invCov, int[:] caseControlStatuses, double[:] thresholds, int burnIn, int numIters, double maxValue, bint useRaoBlackwell, bint DEBUG)
cdef double sampleTruncNorm(int caseControlStatus, double mean, double standardDeviation, double threshold, double maxValue, bint DEBUG)
