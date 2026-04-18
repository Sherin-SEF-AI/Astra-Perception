from setuptools import setup, Extension
import pybind11
import subprocess

def get_opencv_flags():
    try:
        cflags = subprocess.check_output(['pkg-config', '--cflags', 'opencv4']).decode().split()
        libs = subprocess.check_output(['pkg-config', '--libs', 'opencv4']).decode().split()
        return cflags, libs
    except:
        return [], ['-lopencv_core', '-lopencv_imgproc', '-lopencv_imgcodecs']

opencv_cflags, opencv_libs = get_opencv_flags()

# Extract include dirs from cflags
include_dirs = [pybind11.get_include()]
for flag in opencv_cflags:
    if flag.startswith('-I'):
        include_dirs.append(flag[2:])

ext_modules = [
    Extension(
        'astra_core',
        sources=['src/controllers.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-O3', '-std=c++11'],
    ),
    Extension(
        'astra_vision',
        sources=['src/vision_core.cpp'],
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=['-O3', '-std=c++11'],
        extra_link_args=opencv_libs,
    ),
]

setup(
    name='astra_extensions',
    version='1.0',
    description='Native C++ core for Astra Perception',
    ext_modules=ext_modules,
)
