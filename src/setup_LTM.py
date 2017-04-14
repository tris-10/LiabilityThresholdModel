from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
#from Cython.Build import cythonize
import numpy


#python setup_LTM.py build

exts = [
        Extension("gibbsSampler", ["gibbsSampler.pyx"],\
                  "LTMstats", ["LTMstats.pyx"],\
                    "MatrixFuncs", ["MatrixFuncs.pyx"],\
                  include_dirs=[numpy.get_include()]),
        ]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = exts,
)
