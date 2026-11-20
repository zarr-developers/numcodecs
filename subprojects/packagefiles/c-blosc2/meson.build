# Meson build for c-blosc2, which upstream only builds with CMake.
# Only the core library plus the built-in codec/filter/tuner plugins are built
# (no ZFP, tests, benchmarks or examples). Used via subprojects/c-blosc2.wrap.
project(
  'c-blosc2', 'c',
  version: '3.3.6.dev',
  meson_version: '>=1.6.0',
  default_options: ['c_std=gnu11', 'warning_level=0'],
)

cc = meson.get_compiler('c')
host_system = host_machine.system()
cpu = host_machine.cpu_family()

blosc2_args = ['-DBUILD_STATIC']
if host_system != 'windows'
  blosc2_args += ['-D_GNU_SOURCE']
else
  blosc2_args += ['-D_CRT_SECURE_NO_WARNINGS']
endif

deps = [dependency('threads')]
if host_system != 'windows'
  deps += cc.find_library('m', required: false)
  deps += cc.find_library('dl', required: false)
endif

# --- compression libraries (resolved by the parent project or WrapDB) ---
lz4_dep = dependency('liblz4', fallback: ['lz4', 'liblz4_dep'],
                      default_options: ['default_library=static'])
deps += lz4_dep
blosc2_args += ['-DHAVE_LZ4']

zstd_dep = dependency('libzstd', required: get_option('zstd'),
                      fallback: ['zstd', 'libzstd_dep'],
                      default_options: ['default_library=static'])
if zstd_dep.found()
  deps += zstd_dep
  blosc2_args += ['-DHAVE_ZSTD']
endif

zlib_dep = dependency('zlib', required: get_option('zlib'),
                      fallback: ['zlib', 'zlib_dep'],
                      default_options: ['default_library=static'])
if zlib_dep.found()
  deps += zlib_dep
  blosc2_args += ['-DHAVE_ZLIB']
endif

sources = files(
  'blosc/blosc2.c', 'blosc/blosclz.c', 'blosc/fastcopy.c', 'blosc/schunk.c',
  'blosc/frame.c', 'blosc/stune.c', 'blosc/delta.c', 'blosc/shuffle.c',
  'blosc/shuffle-generic.c', 'blosc/bitshuffle-generic.c',
  'blosc/trunc-prec.c', 'blosc/timestamp.c', 'blosc/sframe.c',
  'blosc/directories.c', 'blosc/blosc2-stdio.c', 'blosc/b2nd.c',
  'blosc/b2nd_utils.c',
  # plugins: registries and the plugins they register (ZFP excluded)
  'plugins/codecs/codecs-registry.c',
  'plugins/codecs/ndlz/ndlz.c', 'plugins/codecs/ndlz/ndlz4x4.c',
  'plugins/codecs/ndlz/ndlz8x8.c', 'plugins/codecs/ndlz/xxhash.c',
  'plugins/filters/filters-registry.c',
  'plugins/filters/bytedelta/bytedelta.c',
  'plugins/filters/int_trunc/int_trunc.c',
  'plugins/filters/ndcell/ndcell.c', 'plugins/filters/ndmean/ndmean.c',
  'plugins/tuners/tuners-registry.c',
)
if host_system == 'windows'
  sources += files('blosc/win32/threading.c')
endif

# --- SIMD kernels: detected by compiler capability (cross-compile safe) ---
# Each kernel file gets its ISA flag; shuffle.c only learns that it may
# dispatch to them at run time (SHUFFLE_*_ENABLED) and is built portably.
shuffle_defs = []
simd_libs = []
msvc = cc.get_argument_syntax() == 'msvc'

if cpu in ['x86', 'x86_64']
  if not get_option('sse2').disabled() and cc.has_argument(msvc ? '/arch:SSE2' : '-msse2')
    simd_libs += static_library('blosc2_sse2',
      files('blosc/shuffle-sse2.c', 'blosc/bitshuffle-sse2.c'),
      include_directories: include_directories('include', 'blosc'),
      c_args: blosc2_args + (msvc ? [] : ['-msse2']),
      dependencies: deps)
    shuffle_defs += '-DSHUFFLE_SSE2_ENABLED'
  endif
  if not get_option('avx2').disabled() and cc.has_argument(msvc ? '/arch:AVX2' : '-mavx2')
    simd_libs += static_library('blosc2_avx2',
      files('blosc/shuffle-avx2.c', 'blosc/bitshuffle-avx2.c'),
      include_directories: include_directories('include', 'blosc'),
      c_args: blosc2_args + (msvc ? ['/arch:AVX2'] : ['-mavx2']),
      dependencies: deps)
    shuffle_defs += '-DSHUFFLE_AVX2_ENABLED'
  endif
  if not get_option('avx512').disabled() and not msvc and cc.has_argument('-mavx512bw')
    simd_libs += static_library('blosc2_avx512',
      files('blosc/bitshuffle-avx512.c'),
      include_directories: include_directories('include', 'blosc'),
      c_args: blosc2_args + ['-mavx512f', '-mavx512bw'],
      dependencies: deps)
    shuffle_defs += '-DSHUFFLE_AVX512_ENABLED'
  endif
endif

if cpu == 'aarch64' and not get_option('neon').disabled()
  simd_libs += static_library('blosc2_neon',
    files('blosc/shuffle-neon.c'),
    include_directories: include_directories('include', 'blosc'),
    c_args: blosc2_args + ['-flax-vector-conversions'],
    dependencies: deps)
  shuffle_defs += '-DSHUFFLE_NEON_ENABLED'
endif

blosc2_inc = include_directories('include')

blosc2_lib = static_library('blosc2', sources,
  include_directories: [blosc2_inc, include_directories('blosc')],
  c_args: blosc2_args + shuffle_defs,
  dependencies: deps,
  link_whole: simd_libs,
  pic: true,
)

blosc2_dep = declare_dependency(
  link_with: blosc2_lib,
  include_directories: blosc2_inc,
  dependencies: deps,
  compile_args: ['-DBUILD_STATIC'],
)
meson.override_dependency('blosc2', blosc2_dep)
