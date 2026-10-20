# --- Compiler flags ---
c_args = []
link_args = []

if host_machine.system() != 'windows'
  c_args += ['-pthread']
  link_args += ['-pthread']
endif

# --- C library dependencies ---
# Each codec library is either found on the system (-Dsystem_<lib>=enabled/auto)
# or built as a static subproject from subprojects/<lib>.wrap. Once a subproject
# has been configured, every later dependency('<name>') lookup -- including those
# inside the c-blosc subproject -- resolves to it, so numcodecs.zstd and blosc
# always link the same zstd.
subproject_opts = ['default_library=static', 'warning_level=0']

zstd_dep = dependency('libzstd', required: get_option('system_zstd'), allow_fallback: false)
if not zstd_dep.found()
  subproject('zstd', default_options: subproject_opts + [
    'bin_programs=false', 'multi_thread=disabled', 'legacy_level=0',
    'zlib=disabled', 'lzma=disabled', 'lz4=disabled',
  ])
  zstd_dep = dependency('libzstd')
endif

lz4_dep = dependency('liblz4', required: get_option('system_lz4'), allow_fallback: false)
if not lz4_dep.found()
  subproject('lz4', default_options: subproject_opts + ['ossfuzz=false', 'programs=false'])
  lz4_dep = dependency('liblz4')
endif

blosc_dep = dependency('blosc', required: get_option('system_blosc'), allow_fallback: false)
if not blosc_dep.found()
  # zlib is only needed by blosc; it is not built when using a system blosc.
  zlib_dep = dependency('zlib', required: get_option('system_zlib'), allow_fallback: false)
  if not zlib_dep.found()
    subproject('zlib', default_options: subproject_opts)
  endif

  # Honor legacy environment variables for backwards compatibility with
  # existing build scripts and packaging recipes.
  env_disabled = {}
  foreach simd : ['sse2', 'avx2']
    env_disabled += {simd: run_command(
      py, '-c',
      'import os; print(os.environ.get("DISABLE_NUMCODECS_@0@", ""))'.format(simd.to_upper()),
      check: true,
    ).stdout().strip() != ''}
  endforeach

  simd_opts = []
  foreach simd : ['sse2', 'avx2', 'neon']
    f = get_option(simd)
    if f.disabled() or env_disabled.get(simd, false)
      state = 'disabled'
    elif f.enabled()
      state = 'enabled'
    else
      state = 'auto'
    endif
    simd_opts += '@0@=@1@'.format(simd, state)
  endforeach

  subproject('c-blosc', default_options: subproject_opts + simd_opts)
  blosc_dep = dependency('blosc')
endif

# --- Cython extension modules ---

# Extensions with C library dependencies
py.extension_module('blosc',
  'blosc.pyx',
  dependencies: [blosc_dep],
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
