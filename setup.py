try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy
# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# pykdtree (kd tree)
pykdtree = Extension(
    'model.im2mesh.utils.libkdtree.pykdtree.kdtree',
    sources=[
        'model/im2mesh/utils/libkdtree/pykdtree/kdtree.c',
        'model/im2mesh/utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
    include_dirs=[numpy_include_dir]
)


# Gather all extension modules
ext_modules = [
    pykdtree
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy_include_dir],
    cmdclass={
        'build_ext': BuildExtension
    }
)
