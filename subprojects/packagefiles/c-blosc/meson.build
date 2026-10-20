# Meson build for c-blosc 1.x, maintained by numcodecs (upstream ships CMake only).
# Builds a static libblosc with lz4, zlib and zstd support. The codec libraries are
# resolved with dependency(), so they are shared with the parent project: either the
# same system libraries or the same WrapDB subprojects.
project('c-blosc', 'c',
  default_options: ['c_std=gnu11', 'warning_level=1', 'default_library=static'],
  meson_version: '>=1.6.0',
)

cc = meson.get_compiler('c')

lz4_dep = dependency('liblz4')
zlib_dep = dependency('zlib')
zstd_dep = dependency('libzstd')

c_args = ['-DHAVE_LZ4', '-DHAVE_ZLIB', '-DHAVE_ZSTD']
if host_machine.system() != 'windows'
  c_args += ['-pthread']
endif

sources = files(
  'blosc/blosc.c',
  'blosc/blosclz.c',
  'blosc/fastcopy.c',
  'blosc/shuffle.c',
  'blosc/bitshuffle-generic.c',
)

# --- SIMD shuffle ---
# x86: c-blosc selects SSE2/AVX2 at runtime via CPUID; these only control
# whether the kernels are compiled in.
have_sse2 = false
have_avx2 = false
have_neon = false

if host_machine.cpu_family() == 'x86_64'
  have_sse2 = not get_option('sse2').disabled() and cc.has_argument('-msse2')
  have_avx2 = not get_option('avx2').disabled() and cc.has_argument('-mavx2')
endif

# aarch64: c-blosc 1.x has no NEON shuffle, so shuffle-generic.c is replaced by a
# NEON implementation of the same symbols (kernels from c-blosc2). NEON is
# mandatory on aarch64, so no runtime detection is needed.
if host_machine.cpu_family() == 'aarch64' and not get_option('neon').disabled()
  have_neon = cc.compiles('''
    #if !defined(__ARM_NEON) && !defined(_M_ARM64)
    #error "no NEON"
    #endif
    #include <arm_neon.h>
    int main(void) { uint8x16_t v = vdupq_n_u8(0); return vgetq_lane_u8(v, 0); }
  ''', name: 'NEON intrinsics')
endif

if have_neon
  sources += files('blosc/shuffle-generic-neon.c')
else
  sources += files('blosc/shuffle-generic.c')
endif

if have_sse2
  sources += files('blosc/shuffle-sse2.c', 'blosc/bitshuffle-sse2.c')
  c_args += ['-DSHUFFLE_SSE2_ENABLED', '-msse2']
  if host_machine.system() == 'windows'
    c_args += ['-D__SSE2__']
  endif
endif

if have_avx2
  sources += files('blosc/shuffle-avx2.c', 'blosc/bitshuffle-avx2.c')
  c_args += ['-DSHUFFLE_AVX2_ENABLED', '-mavx2']
  if host_machine.system() == 'windows'
    c_args += ['-D__AVX2__']
  endif
endif

blosc_inc = include_directories('blosc')

blosc_lib = static_library('blosc', sources,
  include_directories: blosc_inc,
  dependencies: [lz4_dep, zlib_dep, zstd_dep],
  c_args: c_args,
)

blosc_dep = declare_dependency(
  link_with: blosc_lib,
  include_directories: blosc_inc,
)
meson.override_dependency('blosc', blosc_dep)
