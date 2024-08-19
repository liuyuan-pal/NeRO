import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

conda_prefix = os.environ.get('CONDA_PREFIX')
eigen_include = os.path.join(conda_prefix, 'include', 'eigen3') if conda_prefix else None

if eigen_include and os.path.exists(eigen_include):
    print(f"Found Eigen at: {eigen_include}")
else:
    raise RuntimeError("Eigen headers not found. Please ensure Eigen is installed in your conda environment.")

_src_path = os.path.dirname(os.path.abspath(__file__))

# ref: https://github.com/sxyu/sdf/blob/master/setup.py
def find_eigen(min_ver=(3, 3, 0)):
    """Helper to find or download the Eigen C++ library"""
    import re, os
    try_paths = [
        '/home/zeus/miniconda3/envs/cloudspace/include/eigen3',
        '/usr/include/eigen3',
        '/usr/local/include/eigen3',
        os.path.expanduser('~/.local/include/eigen3'),
        'C:/Program Files/eigen3',
        'C:/Program Files (x86)/eigen3',
    ]
    WORLD_VER_STR = "#define EIGEN_WORLD_VERSION"
    MAJOR_VER_STR = "#define EIGEN_MAJOR_VERSION"
    MINOR_VER_STR = "#define EIGEN_MINOR_VERSION"
    EIGEN_WEB_URL = 'https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2'
    TMP_EIGEN_FILE = 'tmp_eigen.tar.bz2'
    TMP_EIGEN_DIR = 'eigen-3.3.7'
    min_ver_str = '.'.join(map(str, min_ver))

    eigen_path = None
    for path in try_paths:
        macros_path = os.path.join(path, 'Eigen/src/Core/util/Macros.h')
        if os.path.exists(macros_path):
            macros = open(macros_path, 'r').read().split('\n')
            world_ver, major_ver, minor_ver = None, None, None
            for line in macros:
                if line.startswith(WORLD_VER_STR):
                    world_ver = int(line[len(WORLD_VER_STR):])
                elif line.startswith(MAJOR_VER_STR):
                    major_ver = int(line[len(MAJOR_VER_STR):])
                elif line.startswith(MINOR_VER_STR):
                    minor_ver = int(line[len(MAJOR_VER_STR):])
            if not world_ver or not major_ver or not minor_ver:
                print('Failed to parse macros file', macros_path)
            else:
                ver = (world_ver, major_ver, minor_ver)
                ver_str = '.'.join(map(str, ver))
                if ver < min_ver:
                    print('Found unsuitable Eigen version', ver_str, 'at',
                          path, '(need >= ' + min_ver_str + ')')
                else:
                    print('Found Eigen version', ver_str, 'at', path)
                    eigen_path = path
                    break

    if eigen_path is None:
        try:
            import urllib.request
            print("Couldn't find Eigen locally, downloading...")
            req = urllib.request.Request(
                EIGEN_WEB_URL,
                data=None,
                headers={
                    'User-Agent':
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'
                })

            with urllib.request.urlopen(req) as resp,\
                  open(TMP_EIGEN_FILE, 'wb') as file:
                data = resp.read()
                file.write(data)
            import tarfile
            tar = tarfile.open(TMP_EIGEN_FILE)
            tar.extractall()
            tar.close()

            eigen_path = TMP_EIGEN_DIR
            os.remove(TMP_EIGEN_FILE)
        except:
            print('Download failed, failed to find Eigen')

    if eigen_path is not None:
        print('Found eigen at', eigen_path)

    return eigen_path

nvcc_flags = [
    '-O3',
        '-std=c++17',
        '--expt-extended-lambda',
        '--expt-relaxed-constexpr',
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_HALF2_OPERATORS__',
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-gencode=arch=compute_80,code=sm_80',
]

c_flags = ['-O3', '-std=c++17']


'''
Usage:
python setup.py build_ext --inplace # build extensions locally, do not install (only can be used from the parent directory)
python setup.py install # build extensions and install (copy) to PATH.
pip install . # ditto but better (e.g., dependency & metadata handling)
python setup.py develop # build extensions and install (symbolic) to PATH.
pip install -e . # ditto but better (e.g., dependency & metadata handling)
'''
setup(
    name='raytracing', # package name, import this to use python API
    version='0.1.0',
    description='CUDA RayTracer with BVH acceleration',
    url='https://github.com/ashawkey/raytracing',
    author='kiui',
    author_email='ashawkey1999@gmail.com',
    ext_modules=[
        CUDAExtension(
            name='_raytracing', # extension name, import this to use CUDA API
            sources=[os.path.join(_src_path, 'src', f) for f in [
                'bvh.cu',
                'raytracer.cu',
                'bindings.cpp',
            ]],
            include_dirs=[
                os.path.join(_src_path, 'include'),
                find_eigen(),
                eigen_include,
                torch.utils.cpp_extension.include_paths(cuda=True)[0],
            ],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
    install_requires=[
        'ninja',
        'trimesh',
        'opencv-python',
        'torch',
        'numpy ',
        'tqdm',
        'matplotlib',
        'dearpygui',
    ],
)