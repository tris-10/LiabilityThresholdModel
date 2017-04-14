
cdef double calcHEregressionHerit(double[:,:] GRM, double[:] phenoNorm,bint DEBUG)
cdef standardizeVector(double[:] x)
cdef castIntsToDoubles(int[:]x)
cdef double[:,:] standardizeGeno(double[:,:] GRM)