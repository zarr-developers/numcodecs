import os
import sys
from glob import glob

import cpuinfo
import platform
from Cython.Distutils.build_ext import new_build_ext as build_ext
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
elif platform.machine() == 'aarch64':
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


def blosc_extension():
    info('setting up Blosc extension')

    extra_compile_args = base_compile_args.copy()
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
    include_dirs += [d for d in glob('c-blosc/internal-complibs/*/*/*')
                     if os.path.isdir(d)]
    define_macros += [('HAVE_LZ4', 1),
                      # ('HAVE_SNAPPY', 1),
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

    sources = ['numcodecs/blosc.pyx']

    # define extension module
    extensions = [
        Extension('numcodecs.blosc',
                  sources=sources + blosc_sources,
                  include_dirs=include_dirs,
                  define_macros=define_macros,
                  extra_compile_args=extra_compile_args,
                  ),
    ]

    return extensions


def zstd_extension():
    info('setting up Zstandard extension')

    zstd_sources = []
    extra_compile_args = base_compile_args.copy()
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

    sources = ['numcodecs/zstd.pyx']

    # define extension module
    extensions = [
        Extension('numcodecs.zstd',
                  sources=sources + zstd_sources,
                  include_dirs=include_dirs,
                  define_macros=define_macros,
                  extra_compile_args=extra_compile_args,
                  ),
    ]

    return extensions


def lz4_extension():
    info('setting up LZ4 extension')

    extra_compile_args = base_compile_args.copy()
    define_macros = []

    # setup sources - use LZ4 bundled in blosc
    lz4_sources = glob('c-blosc/internal-complibs/lz4*/*.c')
    include_dirs = [d for d in glob('c-blosc/internal-complibs/lz4*') if os.path.isdir(d)]
    include_dirs += ['numcodecs']
    # define_macros += [('CYTHON_TRACE', '1')]

    sources = ['numcodecs/lz4.pyx']

    # define extension module
    extensions = [
        Extension('numcodecs.lz4',
                  sources=sources + lz4_sources,
                  include_dirs=include_dirs,
                  define_macros=define_macros,
                  extra_compile_args=extra_compile_args,
                  ),
    ]

    return extensions


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


class ve_build_ext(build_ext):
    # This class allows C extension building to fail.

    def run(self):
        try:
            build_ext.run(self)
        except PlatformError as e:
            error(e)
            raise BuildFailed()

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except ext_errors as e:
            error(e)
            raise BuildFailed()


def run_setup(with_extensions):

    if with_extensions:
        ext_modules = (blosc_extension() + zstd_extension() + lz4_extension() +
                       compat_extension() + shuffle_extension() + vlen_extension() +
                       fletcher_extension())

        cmdclass = dict(build_ext=ve_build_ext)
    else:
        ext_modules = []
        cmdclass = {}

    setup(
        ext_modules=ext_modules,
        cmdclass=cmdclass,
    )


if __name__ == '__main__':
    is_pypy = hasattr(sys, 'pypy_translation_info')
    with_extensions = not is_pypy and 'DISABLE_NUMCODECS_CEXT' not in os.environ
    run_setup(with_extensions)
