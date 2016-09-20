Release notes
=============

.. _release_0.0.1:

0.0.1
-----

Fixed project description in setup.py.

.. _release_0.0.0:

0.0.0
-----

First release. This version is a port of the ``codecs`` module from `Zarr
<http://zarr.readthedocs.io>`_ 2.1.0. The following changes have been made from
the original Zarr module:

* Codec classes have been re-organized into separate modules, mostly one per
  codec class, for ease of maintenance.
* Two new codec classes have been added based on 32-bit checksums: CRC32 and
  Adler32.
* The Blosc extension has been refactored to remove code duplications related
  to handling of buffer compatibility.
