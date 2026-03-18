# --- Compiler flags ---
c_args = []
link_args = []

if host_machine.system() != 'windows'
  c_args += ['-pthread']
  link_args += ['-pthread']
endif

# --- SIMD detection ---
# Detected via compiler capability, NOT runtime CPU flags.
# Correct for cross-compilation: checks what the *target* supports.
have_sse2 = false
have_avx2 = false

if host_machine.cpu_family() == 'x86_64'
  # Honor legacy environment variables for backwards compatibility with
  # existing build scripts and packaging recipes.
  disable_sse2_env = run_command(
    py, '-c', 'import os; print(os.environ.get("DISABLE_NUMCODECS_SSE2", ""))',
    check: true,
  ).stdout().strip() != ''
  disable_avx2_env = run_command(
    py, '-c', 'import os; print(os.environ.get("DISABLE_NUMCODECS_AVX2", ""))',
    check: true,
  ).stdout().strip() != ''

  if get_option('sse2').disabled() or disable_sse2_env
    have_sse2 = false
  else
    have_sse2 = cc.has_argument('-msse2')
  endif

  if get_option('avx2').disabled() or disable_avx2_env
    have_avx2 = false
  else
    have_avx2 = cc.has_argument('-mavx2')
  endif
endif

# --- Vendored zstd ---
zstd_dep = dependency('libzstd', required: get_option('system_zstd'))

if not zstd_dep.found()
  zstd_sources = files(
    # common/
    '../../c-blosc/internal-complibs/zstd-1.5.6/common/debug.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/common/entropy_common.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/common/error_private.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/common/fse_decompress.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/common/pool.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/common/threading.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/common/xxhash.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/common/zstd_common.c',
    # compress/
    '../../c-blosc/internal-complibs/zstd-1.5.6/compress/fse_compress.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/compress/hist.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/compress/huf_compress.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/compress/zstd_compress.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/compress/zstd_compress_literals.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/compress/zstd_compress_sequences.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/compress/zstd_compress_superblock.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/compress/zstd_double_fast.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/compress/zstd_fast.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/compress/zstd_lazy.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/compress/zstd_ldm.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/compress/zstd_opt.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/compress/zstdmt_compress.c',
    # decompress/
    '../../c-blosc/internal-complibs/zstd-1.5.6/decompress/huf_decompress.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/decompress/zstd_ddict.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/decompress/zstd_decompress.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/decompress/zstd_decompress_block.c',
    # dictBuilder/
    '../../c-blosc/internal-complibs/zstd-1.5.6/dictBuilder/cover.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/dictBuilder/divsufsort.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/dictBuilder/fastcover.c',
    '../../c-blosc/internal-complibs/zstd-1.5.6/dictBuilder/zdict.c',
  )

  # Assembly files — meson handles .S natively
  if host_machine.cpu_family() == 'x86_64'
    zstd_sources += files(
      '../../c-blosc/internal-complibs/zstd-1.5.6/decompress/huf_decompress_amd64.S',
    )
  endif

  zstd_inc = include_directories(
    '../../c-blosc/internal-complibs/zstd-1.5.6',
    '../../c-blosc/internal-complibs/zstd-1.5.6/common',
    '../../c-blosc/internal-complibs/zstd-1.5.6/compress',
    '../../c-blosc/internal-complibs/zstd-1.5.6/decompress',
    '../../c-blosc/internal-complibs/zstd-1.5.6/dictBuilder',
  )

  zstd_lib = static_library('zstd_vendored', zstd_sources,
    include_directories: zstd_inc,
    c_args: c_args,
  )
  zstd_dep = declare_dependency(
    link_with: zstd_lib,
    include_directories: zstd_inc,
  )
endif

# --- Vendored lz4 ---
lz4_dep = dependency('liblz4', required: get_option('system_lz4'))

if not lz4_dep.found()
  lz4_sources = files(
    '../../c-blosc/internal-complibs/lz4-1.10.0/lz4.c',
    '../../c-blosc/internal-complibs/lz4-1.10.0/lz4hc.c',
  )
  lz4_inc = include_directories('../../c-blosc/internal-complibs/lz4-1.10.0')

  lz4_lib = static_library('lz4_vendored', lz4_sources,
    include_directories: lz4_inc,
    c_args: c_args,
  )
  lz4_dep = declare_dependency(
    link_with: lz4_lib,
    include_directories: lz4_inc,
  )
endif

# --- Vendored zlib (always vendored, used by blosc) ---
zlib_sources = files(
  '../../c-blosc/internal-complibs/zlib-1.3.1/adler32.c',
  '../../c-blosc/internal-complibs/zlib-1.3.1/compress.c',
  '../../c-blosc/internal-complibs/zlib-1.3.1/crc32.c',
  '../../c-blosc/internal-complibs/zlib-1.3.1/deflate.c',
  '../../c-blosc/internal-complibs/zlib-1.3.1/gzclose.c',
  '../../c-blosc/internal-complibs/zlib-1.3.1/gzlib.c',
  '../../c-blosc/internal-complibs/zlib-1.3.1/gzread.c',
  '../../c-blosc/internal-complibs/zlib-1.3.1/gzwrite.c',
  '../../c-blosc/internal-complibs/zlib-1.3.1/infback.c',
  '../../c-blosc/internal-complibs/zlib-1.3.1/inffast.c',
  '../../c-blosc/internal-complibs/zlib-1.3.1/inflate.c',
  '../../c-blosc/internal-complibs/zlib-1.3.1/inftrees.c',
  '../../c-blosc/internal-complibs/zlib-1.3.1/trees.c',
  '../../c-blosc/internal-complibs/zlib-1.3.1/uncompr.c',
  '../../c-blosc/internal-complibs/zlib-1.3.1/zutil.c',
)
zlib_inc = include_directories('../../c-blosc/internal-complibs/zlib-1.3.1')

zlib_lib = static_library('zlib_vendored', zlib_sources,
  include_directories: zlib_inc,
  c_args: c_args,
)
zlib_dep = declare_dependency(
  link_with: zlib_lib,
  include_directories: zlib_inc,
)

# --- Vendored blosc ---
blosc_dep = dependency('blosc', required: get_option('system_blosc'))

if not blosc_dep.found()
  blosc_sources = files(
    '../../c-blosc/blosc/blosc.c',
    '../../c-blosc/blosc/blosclz.c',
    '../../c-blosc/blosc/fastcopy.c',
    '../../c-blosc/blosc/shuffle.c',
    '../../c-blosc/blosc/shuffle-generic.c',
    '../../c-blosc/blosc/bitshuffle-generic.c',
  )

  blosc_c_args = c_args + ['-DHAVE_LZ4', '-DHAVE_ZLIB', '-DHAVE_ZSTD']

  if have_sse2
    blosc_sources += files(
      '../../c-blosc/blosc/shuffle-sse2.c',
      '../../c-blosc/blosc/bitshuffle-sse2.c',
    )
    blosc_c_args += ['-DSHUFFLE_SSE2_ENABLED', '-msse2']
    if host_machine.system() == 'windows'
      blosc_c_args += ['-D__SSE2__']
    endif
  endif

  if have_avx2
    blosc_sources += files(
      '../../c-blosc/blosc/shuffle-avx2.c',
      '../../c-blosc/blosc/bitshuffle-avx2.c',
    )
    blosc_c_args += ['-DSHUFFLE_AVX2_ENABLED', '-mavx2']
    if host_machine.system() == 'windows'
      blosc_c_args += ['-D__AVX2__']
    endif
  endif

  blosc_inc = include_directories('../../c-blosc/blosc')

  blosc_lib = static_library('blosc_vendored', blosc_sources,
    include_directories: blosc_inc,
    dependencies: [lz4_dep, zlib_dep, zstd_dep],
    c_args: blosc_c_args,
  )
  blosc_dep = declare_dependency(
    link_with: blosc_lib,
    include_directories: blosc_inc,
  )
endif

# --- Cython extension modules ---

# Extensions with C library dependencies
py.extension_module('blosc',
  'blosc.pyx',
  dependencies: [blosc_dep, lz4_dep, zlib_dep, zstd_dep],
  c_args: c_args,
  link_args: link_args,
  install: true,
  subdir: 'numcodecs',
)

py.extension_module('zstd',
  'zstd.pyx',
  dependencies: [zstd_dep],
  c_args: c_args,
  link_args: link_args,
  install: true,
  subdir: 'numcodecs',
)

py.extension_module('lz4',
  'lz4.pyx',
  dependencies: [lz4_dep],
  include_directories: include_directories('.'),
  c_args: c_args,
  link_args: link_args,
  install: true,
  subdir: 'numcodecs',
)

py.extension_module('vlen',
  'vlen.pyx',
  dependencies: [numpy_dep],
  include_directories: include_directories('.'),
  c_args: c_args,
  install: true,
  subdir: 'numcodecs',
)

# Pure-Cython extensions (no C library deps)
foreach ext : ['fletcher32', 'jenkins', 'compat_ext', '_shuffle']
  py.extension_module(ext,
    ext + '.pyx',
    include_directories: include_directories('.'),
    c_args: c_args,
    install: true,
    subdir: 'numcodecs',
  )
endforeach

# --- Pure Python sources ---
py.install_sources(
  '__init__.py',
  'abc.py',
  'astype.py',
  'base64.py',
  'bitround.py',
  'bz2.py',
  'categorize.py',
  'checksum32.py',
  'compat.py',
  'delta.py',
  'errors.py',
  'fixedscaleoffset.py',
  'gzip.py',
  'json.py',
  'lzma.py',
  'msgpacks.py',
  'ndarray_like.py',
  'packbits.py',
  'pcodec.py',
  'pickles.py',
  'quantize.py',
  'registry.py',
  'shuffle.py',
  'version.py',
  'zarr3.py',
  'zfpy.py',
  'zlib.py',
  subdir: 'numcodecs',
)
