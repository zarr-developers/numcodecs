project(
  'numcodecs',
  'c', 'cython',
  version: run_command(
    ['src/numcodecs/_build_utils/gitversion.py'], check: true,
  ).stdout().strip(),
  default_options: [
    'c_std=gnu11',
    'warning_level=1',
  ],
  meson_version: '>=1.6.0',
)

py = import('python').find_installation(pure: false)
cc = meson.get_compiler('c')

# NumPy include directory (needed for vlen extension)
# When cross-compiling for Emscripten/WASM, numpy-config is not available on
# the target because of https://github.com/pyodide/pyodide-build/pull/21, so
# we have tofall back to finding headers.
if host_machine.cpu_family() == 'wasm32'
  _numpy_inc = run_command(
    find_program('python', native: true),
    ['-c', 'import numpy; print(numpy.get_include())'],
    check: true,
  ).stdout().strip()
  numpy_dep = declare_dependency(include_directories: include_directories(_numpy_inc))
else
  numpy_dep = dependency('numpy')
endif

subdir('src/numcodecs')
