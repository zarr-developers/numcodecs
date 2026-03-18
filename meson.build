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
numpy_dep = dependency('numpy')

subdir('src/numcodecs')
