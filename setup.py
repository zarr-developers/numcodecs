# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from glob import glob
import os
from setuptools import setup, Extension
import cpuinfo
import sys
from distutils.command.build_ext import build_ext
from distutils.errors import CCompilerError, DistutilsExecError, \
    DistutilsPlatformError


try:
    from Cython.Build import cythonize
except ImportError:
    have_cython = False
else:
    have_cython = True


PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3


# determine CPU support for SSE2 and AVX2
cpu_info = cpuinfo.get_cpu_info()
have_sse2 = 'sse2' in cpu_info['flags']
have_avx2 = 'avx2' in cpu_info['flags']
disable_sse2 = 'DISABLE_NUMCODECS_SSE2' in os.environ
disable_avx2 = 'DISABLE_NUMCODECS_AVX2' in os.environ


# setup common compile arguments
have_cflags = 'CFLAGS' in os.environ
base_compile_args = list()
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
# workaround lack of support for "inline" in MSVC when building for Python 2.7 64-bit
if PY2 and os.name == 'nt':
    base_compile_args.append('-Dinline=__inline')


def info(*msg):
    kwargs = dict(file=sys.stdout)
    print('[numcodecs]', *msg, **kwargs)


def error(*msg):
    kwargs = dict(file=sys.stderr)
    print('[numcodecs]', *msg, **kwargs)


def blosc_extension():
    info('setting up Blosc extension')

    extra_compile_args = list(base_compile_args)
    define_macros = []

    # setup blosc sources
    blosc_sources = [f for f in glob('c-blosc/blosc/*.c')
                     if 'avx2' not in f and 'sse2' not in f]
    include_dirs = [os.path.join('c-blosc', 'blosc')]

    # add internal complibs
    blosc_sources += glob('c-blosc/internal-complibs/lz4*/*.c')
    blosc_sources += glob('c-blosc/internal-complibs/snappy*/*.cc')
    blosc_sources += glob('c-blosc/internal-complibs/zlib*/*.c')
    blosc_sources += glob('c-blosc/internal-complibs/zstd*/common/*.c')
    blosc_sources += glob('c-blosc/internal-complibs/zstd*/compress/*.c')
    blosc_sources += glob('c-blosc/internal-complibs/zstd*/decompress/*.c')
    blosc_sources += glob('c-blosc/internal-complibs/zstd*/dictBuilder/*.c')
    include_dirs += [d for d in glob('c-blosc/internal-complibs/*')
                     if os.path.isdir(d)]
    include_dirs += [d for d in glob('c-blosc/internal-complibs/*/*')
                     if os.path.isdir(d)]
    define_macros += [('HAVE_LZ4', 1),
                      ('HAVE_SNAPPY', 1),
                      ('HAVE_ZLIB', 1),
                      ('HAVE_ZSTD', 1)]
    # define_macros += [('CYTHON_TRACE', '1')]

    # SSE2
    if have_sse2 and not disable_sse2:
        info('compiling Blosc extension with SSE2 support')
        extra_compile_args.append('-DSHUFFLE_SSE2_ENABLED')
        blosc_sources += [f for f in glob('c-blosc/blosc/*.c') if 'sse2' in f]
        if os.name == 'nt':
            define_macros += [('__SSE2__', 1)]
    else:
        info('compiling Blosc extension without SSE2 support')

    # AVX2
    if have_avx2 and not disable_avx2:
        info('compiling Blosc extension with AVX2 support')
        extra_compile_args.append('-DSHUFFLE_AVX2_ENABLED')
        blosc_sources += [f for f in glob('c-blosc/blosc/*.c') if 'avx2' in f]
        if os.name == 'nt':
            define_macros += [('__AVX2__', 1)]
    else:
        info('compiling Blosc extension without AVX2 support')

    if have_cython:
        sources = ['numcodecs/blosc.pyx']
    else:
        sources = ['numcodecs/blosc.c']

    # define extension module
    extensions = [
        Extension('numcodecs.blosc',
                  sources=sources + blosc_sources,
                  include_dirs=include_dirs,
                  define_macros=define_macros,
                  extra_compile_args=extra_compile_args,
                  ),
    ]

    if have_cython:
        extensions = cythonize(extensions)

    return extensions


def zstd_extension():
    info('setting up Zstandard extension')

    zstd_sources = []
    extra_compile_args = list(base_compile_args)
    include_dirs = []
    define_macros = []

    # setup sources - use zstd bundled in blosc
    zstd_sources += glob('c-blosc/internal-complibs/zstd*/common/*.c')
    zstd_sources += glob('c-blosc/internal-complibs/zstd*/compress/*.c')
    zstd_sources += glob('c-blosc/internal-complibs/zstd*/decompress/*.c')
    zstd_sources += glob('c-blosc/internal-complibs/zstd*/dictBuilder/*.c')
    include_dirs += [d for d in glob('c-blosc/internal-complibs/zstd*')
                     if os.path.isdir(d)]
    include_dirs += [d for d in glob('c-blosc/internal-complibs/zstd*/*')
                     if os.path.isdir(d)]
    # define_macros += [('CYTHON_TRACE', '1')]

    if have_cython:
        sources = ['numcodecs/zstd.pyx']
    else:
        sources = ['numcodecs/zstd.c']

    # define extension module
    extensions = [
        Extension('numcodecs.zstd',
                  sources=sources + zstd_sources,
                  include_dirs=include_dirs,
                  define_macros=define_macros,
                  extra_compile_args=extra_compile_args,
                  ),
    ]

    if have_cython:
        extensions = cythonize(extensions)

    return extensions


def lz4_extension():
    info('setting up LZ4 extension')

    extra_compile_args = list(base_compile_args)
    define_macros = []

    # setup sources - use LZ4 bundled in blosc
    lz4_sources = glob('c-blosc/internal-complibs/lz4*/*.c')
    include_dirs = [d for d in glob('c-blosc/internal-complibs/lz4*') if os.path.isdir(d)]
    include_dirs += ['numcodecs']
    # define_macros += [('CYTHON_TRACE', '1')]

    if have_cython:
        sources = ['numcodecs/lz4.pyx']
    else:
        sources = ['numcodecs/lz4.c']

    # define extension module
    extensions = [
        Extension('numcodecs.lz4',
                  sources=sources + lz4_sources,
                  include_dirs=include_dirs,
                  define_macros=define_macros,
                  extra_compile_args=extra_compile_args,
                  ),
    ]

    if have_cython:
        extensions = cythonize(extensions)

    return extensions


def vlen_extension():
    info('setting up vlen extension')

    extra_compile_args = list(base_compile_args)
    define_macros = []

    # setup sources
    include_dirs = ['numcodecs']
    # define_macros += [('CYTHON_TRACE', '1')]

    if have_cython:
        sources = ['numcodecs/vlen.pyx']
    else:
        sources = ['numcodecs/vlen.c']

    # define extension module
    extensions = [
        Extension('numcodecs.vlen',
                  sources=sources,
                  include_dirs=include_dirs,
                  define_macros=define_macros,
                  extra_compile_args=extra_compile_args,
                  ),
    ]

    if have_cython:
        extensions = cythonize(extensions)

    return extensions


def compat_extension():
    info('setting up compat extension')

    extra_compile_args = list(base_compile_args)

    if have_cython:
        sources = ['numcodecs/compat_ext.pyx']
    else:
        sources = ['numcodecs/compat_ext.c']

    # define extension module
    extensions = [
        Extension('numcodecs.compat_ext',
                  sources=sources,
                  extra_compile_args=extra_compile_args),
    ]

    if have_cython:
        extensions = cythonize(extensions)

    return extensions


if sys.platform == 'win32':
    ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError,
                  IOError, ValueError)
else:
    ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError)


class BuildFailed(Exception):
    pass


class ve_build_ext(build_ext):
    # This class allows C extension building to fail.

    def run(self):
        try:
            build_ext.run(self)
        except DistutilsPlatformError as e:
            error(e)
            raise BuildFailed()

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except ext_errors as e:
            error(e)
            raise BuildFailed()


DESCRIPTION = ("A Python package providing buffer compression and "
               "transformation codecs for use in data storage and "
               "communication applications.")

with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()


def run_setup(with_extensions):

    if with_extensions:
        ext_modules = (blosc_extension() + zstd_extension() + lz4_extension() +
                       compat_extension() + vlen_extension())
        cmdclass = dict(build_ext=ve_build_ext)
    else:
        ext_modules = []
        cmdclass = dict()

    setup(
        name='numcodecs',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        use_scm_version={
            'version_scheme': 'guess-next-dev',
            'local_scheme': 'dirty-tag',
            'write_to': 'numcodecs/version.py'
        },
        setup_requires=[
            'setuptools>18.0',
            'setuptools-scm>1.5.4'
        ],
        install_requires=[
            'numpy>=1.7',
        ],
        extras_require={
            'msgpack':  ["msgpack-python"],
        },
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        package_dir={'': '.'},
        packages=['numcodecs', 'numcodecs.tests'],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Information Technology',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Operating System :: Unix',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
        ],
        author='Alistair Miles',
        author_email='alimanfoo@googlemail.com',
        maintainer='Alistair Miles',
        maintainer_email='alimanfoo@googlemail.com',
        url='https://github.com/alimanfoo/numcodecs',
        license='MIT',
    )


if __name__ == '__main__':
    is_pypy = hasattr(sys, 'pypy_translation_info')
    with_extensions = not is_pypy and 'DISABLE_NUMCODECS_CEXT' not in os.environ

    try:
        run_setup(with_extensions)
    except BuildFailed:
        error('*' * 75)
        error('compilation of C extensions failed.')
        error('retrying installation without C extensions...')
        error('*' * 75)
        run_setup(False)
