import os
import sys
from glob import glob

import cpuinfo
from setuptools import Extension, setup
from setuptools.errors import CCompilerError, ExecError, PlatformError

# determine CPU support for SSE2 and AVX2
cpu_info = cpuinfo.get_cpu_info()
flags = cpu_info.get('flags', [])
have_sse2 = 'sse2' in flags
have_avx2 = 'avx2' in flags
disable_sse2 = 'DISABLE_NUMCODECS_SSE2' in os.environ
disable_avx2 = 'DISABLE_NUMCODECS_AVX2' in os.environ

# setup common compile arguments
have_cflags = 'CFLAGS' in os.environ
base_compile_args = []
if have_cflags:
    # respect compiler options set by user
    pass
elif os.name == 'posix':
    if disable_sse2:
        base_compile_args.append('-mno-sse2')
    elif have_sse2:
        base_compile_args.append('-msse2')
    if disable_avx2:
        base_compile_args.append('-mno-avx2')
    elif have_avx2:
        base_compile_args.append('-mavx2')
# On macOS, force libc++ in case the system tries to use `stdlibc++`.
# The latter is often absent from modern macOS systems.
if sys.platform == 'darwin':
    base_compile_args.append('-stdlib=libc++')


def info(*msg):
    kwargs = dict(file=sys.stdout)
    print('[numcodecs]', *msg, **kwargs)


def error(*msg):
    kwargs = dict(file=sys.stderr)
    print('[numcodecs]', *msg, **kwargs)


def vlen_extension():
    info('setting up vlen extension')

    extra_compile_args = base_compile_args.copy()
    define_macros = []

    # setup sources
    include_dirs = ['numcodecs']
    # define_macros += [('CYTHON_TRACE', '1')]

    sources = ['numcodecs/vlen.pyx']

    # define extension module
    extensions = [
        Extension('numcodecs.vlen',
                  sources=sources,
                  include_dirs=include_dirs,
                  define_macros=define_macros,
                  extra_compile_args=extra_compile_args,
                  ),
    ]

    return extensions


def fletcher_extension():
    info('setting up fletcher32 extension')

    extra_compile_args = base_compile_args.copy()
    define_macros = []

    # setup sources
    include_dirs = ['numcodecs']
    # define_macros += [('CYTHON_TRACE', '1')]

    sources = ['numcodecs/fletcher32.pyx']

    # define extension module
    extensions = [
        Extension('numcodecs.fletcher32',
                  sources=sources,
                  include_dirs=include_dirs,
                  define_macros=define_macros,
                  extra_compile_args=extra_compile_args,
                  ),
    ]

    return extensions


def jenkins_extension():
    info('setting up jenkins extension')

    extra_compile_args = base_compile_args.copy()
    define_macros = []

    # setup sources
    include_dirs = ['numcodecs']
    define_macros += [('CYTHON_TRACE', '1')]

    sources = ['numcodecs/jenkins.pyx']

    # define extension module
    extensions = [
        Extension('numcodecs.jenkins',
                  sources=sources,
                  include_dirs=include_dirs,
                  define_macros=define_macros,
                  extra_compile_args=extra_compile_args,
                  ),
    ]

    return extensions


def compat_extension():
    info('setting up compat extension')

    extra_compile_args = base_compile_args.copy()

    sources = ['numcodecs/compat_ext.pyx']

    # define extension module
    extensions = [
        Extension('numcodecs.compat_ext',
                  sources=sources,
                  extra_compile_args=extra_compile_args),
    ]

    return extensions


def shuffle_extension():
    info('setting up shuffle extension')

    extra_compile_args = base_compile_args.copy()

    sources = ['numcodecs/_shuffle.pyx']

    # define extension module
    extensions = [
        Extension('numcodecs._shuffle',
                  sources=sources,
                  extra_compile_args=extra_compile_args),
    ]

    return extensions


if sys.platform == 'win32':
    ext_errors = (CCompilerError, ExecError, PlatformError,
                  IOError, ValueError)
else:
    ext_errors = (CCompilerError, ExecError, PlatformError)


class BuildFailed(Exception):
    pass


def run_setup(with_extensions):

    cmdclass = {}
    if with_extensions:
        ext_modules = (compat_extension() + shuffle_extension() + vlen_extension() +
                       fletcher_extension() + jenkins_extension())
    else:
        ext_modules = []


    setup(
        ext_modules=ext_modules,
        cmdclass=cmdclass,
    )


if __name__ == '__main__':
    is_pypy = hasattr(sys, 'pypy_translation_info')
    with_extensions = not is_pypy and 'DISABLE_NUMCODECS_CEXT' not in os.environ
    run_setup(with_extensions)
