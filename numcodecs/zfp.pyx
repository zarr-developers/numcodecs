from __future__ import absolute_import, print_function, division
#cimport numpy
cimport cython
#from cpython cimport array
import numpy as np
import sys
#import array

from cpython.buffer cimport PyBUF_ANY_CONTIGUOUS, PyBUF_WRITEABLE
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING

from .compat_ext cimport Buffer
from .compat_ext import Buffer
from numcodecs.compat import ensure_contiguous_ndarray,ensure_ndarray
from numcodecs.abc import Codec


cdef extern from "bitstream.h":
   ctypedef struct bitstream:
       pass

   bitstream* stream_open(void*,size_t)
   bitstream* stream_close(void*)
   bitstream* zfp_stream_bit_stream(const zfp_stream*)

cdef extern from "zfp.h":

   cdef enum:
       ZFP_VERSION_STRING
       ZFP_HEADER_FULL
       ZFP_MIN_BITS
       ZFP_MAX_BITS
       ZFP_MAX_PREC
       ZFP_MIN_EXP
   ctypedef enum zfp_type:
       zfp_type_none = 0
       zfp_type_int32 = 1
       zfp_type_int64 = 2
       zfp_type_float = 3
       zfp_type_double = 4
 
   ctypedef enum zfp_exec_policy:
       zfp_exec_serial = 0
       zfp_exec_omp = 1
       zfp_exec_cuda = 2
 
   ctypedef struct zfp_exec_params_omp:
       unsigned int threads
       unsigned int chunk_size

   ctypedef union zfp_exec_params:
       zfp_exec_params_omp omp

   ctypedef struct zfp_execution:
       zfp_exec_policy policy
       zfp_exec_params params
   
   ctypedef struct zfp_stream:
       unsigned int minbits
       unsigned int maxbits
       unsigned int maxprec
       int minexp
       bitstream* stream
       zfp_execution _exec 
   
   ctypedef struct zfp_field:
       zfp_type _type
       unsigned int nx,ny,nz,nw
       int sx,sy,sz,sw
       void* data


   zfp_stream* zfp_stream_open(zfp_stream*)
   void zfp_stream_close(zfp_stream*)
   void zfp_stream_rewind(zfp_stream*)
   zfp_type zfp_field_set_type(zfp_field*, zfp_type)
   void zfp_field_set_pointer(zfp_field*,void*) 
   void zfp_field_set_size_1d(zfp_field*,unsigned int nx)
   void zfp_field_set_size_2d(zfp_field*,unsigned int nx,unsigned int nx)
   void zfp_field_set_size_3d(zfp_field*,unsigned int nx,unsigned int nx,unsigned int nz)
   void zfp_field_set_size_4d(zfp_field*,unsigned int nx,unsigned int nx,unsigned int nz,unsigned int nw)
   zfp_field* zfp_field_alloc()
   void zfp_field_free(zfp_field* field)
   size_t zfp_stream_maximum_size(const zfp_stream*,const zfp_field*)
   void zfp_stream_set_bit_stream(zfp_stream*,bitstream*)
   double zfp_stream_set_rate(zfp_stream*,double,zfp_type,unsigned int,int)
   unsigned int zfp_stream_set_precision(zfp_stream*,unsigned int)
   double zfp_stream_set_accuracy(zfp_stream*,double)
   int zfp_stream_set_params(zfp_stream*,unsigned int,unsigned int,unsigned int,int)
   size_t zfp_compress(zfp_stream*,const zfp_field*)
   int zfp_decompress(zfp_stream*,zfp_field*)
   int zfp_stream_set_execution(zfp_stream* stream, zfp_exec_policy policy)
   size_t zfp_write_header(zfp_stream* stream, const zfp_field* field, unsigned int mask)
   size_t zfp_read_header(zfp_stream* stream, zfp_field* field, unsigned int mask)      

VERSION_STRING = <char *> ZFP_VERSION_STRING
__version__ = VERSION_STRING


def compress(input_array,method):
  '''Compress data. 
  
  Parameters
  ----------
  input_array : numpy array
     Data to be compressed. zfp can compress data better with arrays that have
     larger number of dimensions (1-4). So we want to reserve the dimension of the data

  method : an object has compression mode and its relate compression parameters
  
  Returns
  -------
  dest : bytes
     Compressed data

  '''
  cdef zfp_stream* zfp
  cdef bitstream* stream
  cdef zfp_field* field
  cdef size_t bufsize

  cdef:
     char *source_ptr
     char *dest_ptr
     Buffer source_buffer
     int dest_size,compressed_size
     bytes dest

  # get compression mode
  zfpmode = method.zfpmode

  # allocate an object to store all parameters
  field = zfp_field_alloc()

  # setup source buffer
  source_buffer = Buffer(input_array,PyBUF_ANY_CONTIGUOUS)
  source_ptr = source_buffer.ptr

  # determine type
  if input_array.dtype == np.float32:
     the_type = zfp_type_float
  elif input_array.dtype == np.float64:
     the_type = zfp_type_double
  elif input_array.dtype == np.int32:
     the_type = zfp_type_int32
  elif input_array.dtype == np.int64:
     the_type = zfp_type_int64
  else:
     print("The type of data should be int32, int64, float or double")
  zfp_field_set_type(field,the_type)
  zfp_field_set_pointer(field,source_ptr)
  print(input_array.shape)

  # determine the dimensions
  if input_array.ndim == 1:
     zfp_field_set_size_1d(field,input_array.shape[0])
  elif input_array.ndim == 2:
     zfp_field_set_size_2d(field,input_array.shape[1],input_array.shape[0])
  elif input_array.ndim == 3:
     zfp_field_set_size_3d(field,input_array.shape[2],input_array.shape[1],input_array.shape[0])
  elif input_array.ndim == 4:
     print(input_array.shape)
     zfp_field_set_size_4d(field,input_array.shape[3],input_array.shape[2],input_array.shape[1],input_array.shape[0])
  else:
     print("The dimension of data should be equal or less than 4, please reshape")

  # setup compression mode
  zfp=zfp_stream_open(NULL)
  if zfpmode == 'a':
     zfp_stream_set_accuracy(zfp,method.tolerance)
  elif zfpmode == 'p':
     zfp_stream_set_precision(zfp,method.precision)
  elif zfpmode == 'r':
     zfp_stream_set_rate(zfp,method.rate,the_type,input_array.ndim,0)
  elif zfpmode =='c':
     zfp_stream_set_params(zfp,method.minbits,method.maxbits,method.maxprec,method.minexp)
  else:
     print('Must specify zfp mode')

  # setup destination
  dest_size = zfp_stream_maximum_size(zfp,field)
  dest = PyBytes_FromStringAndSize(NULL,dest_size)
  dest_ptr = PyBytes_AS_STRING(dest)
  stream = stream_open(dest_ptr,dest_size)
  zfp_stream_set_bit_stream(zfp,stream)
  # currently only use serial execution
  zfp_stream_set_execution(zfp,zfp_exec_serial)

  # perform compression
  ret = zfp_write_header(zfp,field,ZFP_HEADER_FULL)
  if not ret:
     print("zfp_write_header failed")
  compressed_size = zfp_compress(zfp,field)
  if not compressed_size:
     print("zfp_compress is failed")

  # release buffers
  source_buffer.release()
  zfp_field_free(field)
  zfp_stream_close(zfp)
  stream_close(stream)
 
  return dest[:compressed_size]

def decompress(source):
 '''Decompress data.

 Parameters
 ----------
 source : bytes-like
     Compressed data, including zfp header. Can be any object supporting the
     buffer protocol
 
 Returns 
 -------
 dest : array-like
     Object containing decompressed data
 
 '''
 cdef:
     zfp_stream* zfp
     bitstream* stream
     zfp_field* field
     Buffer source_buffer
     Buffer dest_buffer 
     char *source_ptr
     char *dest_ptr
     int source_size
     

 # setup source buffer
 source_buffer = Buffer(source, PyBUF_ANY_CONTIGUOUS)
 source_ptr = source_buffer.ptr
 source_size = source_buffer.nbytes

 # setup zfp object
 field = zfp_field_alloc()
 zfp = zfp_stream_open(NULL)
 stream = stream_open(source_ptr,source_size)
 zfp_stream_set_bit_stream(zfp,stream)
 zfp_stream_rewind(zfp)
 print(source_size)

 # read zfp header
 ret = zfp_read_header(zfp,field,ZFP_HEADER_FULL);
 if not ret:
    print("zfp_read_header failed")

 # determine destination data type 
 if field._type == zfp_type_float:
    the_type = np.float32 
    type_size = 4 
 elif field._type == zfp_type_double:
    the_type = np.float64 
    type_size = 8 
 elif field._type == zfp_type_int32:
    the_type = np.int32
    type_size = 4 
 else:
    the_type = np.int64
    type_size = 8 

 # determine data dimension
 nx,ny,nz,nw = field.nx,field.ny,field.nz,field.nw
 datashape = (nx if nx != 0 else 1)* (ny if ny != 0 else 1) \
           *(nz if nz != 0 else 1)* (nw if nw != 0 else 1)

 # currently only serial exeution available
 zfp_stream_set_execution(zfp,zfp_exec_serial)

 # setup destination buffer
 dest = PyBytes_FromStringAndSize(NULL,datashape*type_size)
 dest_ptr = PyBytes_AS_STRING(dest)
 zfp_field_set_pointer(field,dest_ptr)

 # perform decompression
 ret = zfp_decompress(zfp,field)
 if not ret: 
    print("decompress failed")

 # release buffers
 source_buffer.release()
 zfp_field_free(field)
 zfp_stream_close(zfp)
 stream_close(stream)

 # reshape the buffer
 if not ny:
   buf_shape=(nx)
 elif ny and not nz:
   buf_shape=(ny,nx)
 elif ny and nz and not nw:
   buf_shape=(nz,ny,nx)
 else:
   buf_shape=(nw,nz,ny,nx)
 print(buf_shape)
 dest = ensure_ndarray(dest).view(the_type)
 dest = dest.reshape(buf_shape)

 return dest

class CompressMethod:
 '''
 Create a CompressMethod object to store zfp mode and 
 its parameters in Zfp class
 
 '''

 # Initilize with defaults
 def __init__(self):
    self.zfpmode='a'
    self.tolerance=0.01

 def set_mode(self,mode):
     self.zfpmode=mode
 def set_tolerance(self,tol):
     self.tolerance=tol
 def set_precision(self,prec):
     self.precision=prec
 def set_rate(self,rate):
     self.rate=rate
 def set_minbits(self,minbits):
     self.minbits=minbits
 def set_maxbits(self,maxbits):
     self.maxbits=maxbits
 def set_maxprec(self,maxprec):
     self.maxprec=maxprec
 def set_minexp(self,minexp):
     self.minexp=minexp
     
class Zfp(Codec):
 '''Codec providing compression using zfp via the Python standard library.
 
 Parameters
 ----------
 mode : a
 tolerance : float (absolue error tolerance), default 0.01
 
 mode : p
 precision : int (uncompressed bits per value), can be 0-64 for double precision

 mode : r
 rate : int (compressed bits per floating-point value), can be 0-64 for double precision

 mode : c
 minbits : int (min bits per 4^d values in d dimensions)
 maxbits : int (max bits per 4^d values in d dimensions), 0 for unlimited
 minexp : int (min bit plane : -1074 for all bit planes)
 maxprec : int (max bits of precision per value), 0 for full
 '''
 codec_id = 'zfp'

 def __init__(self, mode = 'a',tol = 0.01, prec = 32, rate = 16, minbits = ZFP_MIN_BITS, maxbits = ZFP_MAX_BITS, minexp = ZFP_MIN_EXP, maxprec = ZFP_MAX_PREC):
    self.mode=mode
    the_method=CompressMethod()
    
    if mode == 'a':
       self.tolerance=tol
       the_method.set_mode(mode)
       the_method.set_tolerance(tol)
    elif mode == 'p':
       self.maxprec = prec
       the_method.set_mode(mode)
       the_method.set_precision(prec)
    elif mode == 'r':
       self.rate = rate
       the_method.set_mode(mode)
       the_method.set_rate(rate)
    elif mode == 'c':
       self.minbits = minbits
       self.maxbits = maxbits
       self.maxprec = maxprec
       self.minexp  = minexp
       the_method.set_mode(mode)
       the_method.set_minbits(minbits)
       the_method.set_maxbits(maxbits)
       the_method.set_minexp(minexp)
       the_method.set_maxprec(maxprec)
    else:
         print("Wrong mode, please set mode to 'a', 'p', 'r' or 'c'")
    self.the_method=the_method

 def encode(self,buf):
      #buf = ensure_contiguous_ndarray(buf)
      return compress(buf,self.the_method)

 def decode(self,buf):
     buf=ensure_contiguous_ndarray(buf)
     return decompress(buf)

 def __repr__(self):
     if self.mode == 'a':
        r= '%r(mode=%r,minexp=%s)' % (type(self).__name__, self.mode,self.minexp)
     elif self.mode == 'p':
        r= '%r(mode=%r,maxprec=%s)' % (type(self).__name__,self.mode,self.maxprec)
     elif self.mode == 'r':
        r= '%r(mode=%r,rate=%s)' % (type(self).__name__,self.mode,self.rate)
     elif self.mode == 'c':
        r= '%r(mode=%r,minbits=%s,maxbits=%s,maxprec=%s,minexp=%s)' % (type(self).__name__,self.mode,self.minbits,self.maxbits,self.maxprec,self.minexp)
     else:
        r="WRONG MODE"
     return r
      
