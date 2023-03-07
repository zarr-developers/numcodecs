SZIP
====

This codec is a numcodecs translation of the `hdf5 implementation`_. The binary driver is
available in the open source package ``libaec``, which must be installed for use.

SZIP is used, for instance, by `NASA-EOSS`_. By including this codec, we
make such data readable in zarr by using kerchunk. As such, the
compression parameters are provided in the HDF metadata.

.. _hdf5 implementation: https://portal.hdfgroup.org/display/HDF5/Szip+Compression+in+HDF+Products

.. _NASA-EOSS: https://www.earthdata.nasa.gov/esdis/esco/standards-and-practices/hdf-eos5

If you wish to compress new data with this codec, you will need to
find reasonable values for the parameters, see `this section of HDF code`_


.. _this section of HDF code: https://github.com/HDFGroup/hdf5/blob/7b833f04b5146bdad339ff10d42aadc416fb2f00/src/H5Zszip.c#L106-L244)
