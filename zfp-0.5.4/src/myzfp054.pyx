cimport numpy
cimport cython
from cpython cimport array
import numpy
import sys
import array
from numcodecs.compat import ensure_contiguous_ndarray
from numcodecs.abc import Codec


# Cython header file of myzfp054.pyx
from libc.stdlib cimport malloc,free

__VERSION__ = '0.5.4'

cdef extern from "bitstream.h":
   ctypedef struct bitstream:
       pass

   bitstream* stream_open(void*,size_t)
   bitstream* stream_close(void*)
   bitstream* zfp_stream_bit_stream(const zfp_stream*)

cdef extern from "zfp.h":
   cdef int _ZFP_HEADER_FULL "ZFP_HEADER_FULL"
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
   zfp_field* zfp_field_alloc()
   void zfp_field_free(zfp_field* field)
   void zfp_stream_params(const zfp_stream*,unsigned int*,unsigned int*,unsigned int*,int*)
   size_t zfp_stream_compressed_size(const zfp_stream*)
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

cpdef allocate(typecode, ct):
   cdef array.array array_template = array.array(chr(typecode),[])
   return array.clone(array_template, ct, zero=True)

def compress(input_array,method):
  cdef zfp_stream* zfp
  cdef bitstream* stream
  cdef zfp_field* field
  cdef size_t bufsize
  cdef numpy.ndarray[numpy.int8_t,ndim=1,mode="c"] out_stream
  cdef numpy.ndarray[numpy.float32_t,ndim=1,mode="c"] arrf1
  cdef numpy.ndarray[numpy.float32_t,ndim=2,mode="c"] arrf2
  cdef numpy.ndarray[numpy.float32_t,ndim=3,mode="c"] arrf3
  cdef numpy.ndarray[numpy.float64_t,ndim=1,mode="c"] arrd1
  cdef numpy.ndarray[numpy.float64_t,ndim=2,mode="c"] arrd2
  cdef numpy.ndarray[numpy.float64_t,ndim=3,mode="c"] arrd3

  zfpmode=method.zfpmode
  field=zfp_field_alloc()

  if input_array.dtype == numpy.float32:
     zfp_field_set_type(field,zfp_type_float)
     if input_array.ndim == 1:
        arrf1=numpy.ascontiguousarray(input_array,dtype=numpy.float32)
        zfp_field_set_pointer(field,&arrf1[0])
        zfp_field_set_size_1d(field,input_array.shape[0])
     elif input_array.ndim == 2:
        arrf2=numpy.ascontiguousarray(input_array,dtype=numpy.float32)
        zfp_field_set_pointer(field,&arrf2[0,0])
        zfp_field_set_size_2d(field,input_array.shape[0],input_array.shape[1])
     elif input_array.ndim == 3:
        arrf3=numpy.ascontiguousarray(input_array,dtype=numpy.float32)
        zfp_field_set_pointer(field,&arrf3[0,0,0])
        zfp_field_set_size_3d(field,input_array.shape[0],input_array.shape[1],input_array.shape[2])
  elif input_array.dtype == numpy.float64:
     zfp_field_set_type(field,zfp_type_double)
     if input_array.ndim == 1:
        arrd1=numpy.ascontiguousarray(input_array,dtype=numpy.float64)
        zfp_field_set_pointer(field,&arrd1[0])
        zfp_field_set_size_1d(field,input_array.shape[0])
     elif input_array.ndim == 2:
        arrd2=numpy.ascontiguousarray(input_array,dtype=numpy.float64)
        zfp_field_set_pointer(field,&arrd2[0,0])
        zfp_field_set_size_2d(field,input_array.shape[0],input_array.shape[1])
     elif input_array.ndim == 3:
        arrd3=numpy.ascontiguousarray(input_array,dtype=numpy.float64)
        zfp_field_set_pointer(field,&arrd3[0,0,0])
        zfp_field_set_size_3d(field,input_array.shape[0],input_array.shape[1],input_array.shape[2])
  else:
     print("data type is not float or double, cannot be compressed")
     sys.exit(1)

  zfp=zfp_stream_open(NULL)
  if zfpmode == 'a':
     zfp_stream_set_accuracy(zfp,method.arg3)
  elif zfpmode == 'p':
     zfp_stream_set_precision(zfp,method.arg2)
  elif zfpmode == 'r':
     zfp_stream_set_rate(zfp,method.arg2,zfp_type_float,input_array.ndim,0)
  elif zfpmode =='c':
     #p.minbits = method.arg4
     #p.maxbits = method.arg5
     #p.maxprec = method.arg2
     #p.minexp = method.arg3 
     print zfpmode
  else:
     print 'Must specify compression parameters'
     sys.exit(1)

  bufsize=zfp_stream_maximum_size(zfp,field)
  buffer=numpy.empty(bufsize,dtype=numpy.int8)
  out_stream=numpy.ascontiguousarray(buffer,dtype=numpy.int8)
  stream=stream_open(<void *>(&out_stream[0]),bufsize)
  zfp_stream_set_bit_stream(zfp,stream)
  zfp_stream_set_execution(zfp,zfp_exec_serial)

  ret=zfp_write_header(zfp,field,_ZFP_HEADER_FULL)
  if not ret:
     print("zfp_write_header failed")
     sys.exit(1)

  outsize=zfp_compress(zfp,field)
 
  if not outsize:
     print "zfp_compress is failed"
     sys.exit(1)

  zfp_field_free(field)
  zfp_stream_close(zfp)
  stream_close(stream)
 
  return outsize,out_stream[:outsize]

def decompress(buf_stream):
 cdef zfp_stream* zfp
 cdef bitstream* stream
 cdef zfp_field* field
 cdef numpy.ndarray[numpy.int8_t,ndim=1,mode="c"] char_arr 

 field=zfp_field_alloc()
 zfp=zfp_stream_open(NULL)
 char_arr=numpy.ascontiguousarray(buf_stream,dtype=numpy.int8)
 stream=stream_open(<void *>&char_arr[0],len(buf_stream))
 zfp_stream_set_bit_stream(zfp,stream)
 zfp_stream_rewind(zfp)
 ret=zfp_read_header(zfp,field,_ZFP_HEADER_FULL);
 if not ret:
    print("zfp_read_header failed")
    sys.exit(1)

 cdef char ztype=b'f' if field._type == zfp_type_float else b'd'
 nx,ny,nz,nw=field.nx,field.ny,field.nz,field.nw
 datashape=(field.nx if field.nx != 0 else 1)* (field.ny if field.ny != 0 else 1) \
           *(field.nz if field.nz != 0 else 1)* (field.nw if field.nw != 0 else 1)

 zfp_stream_set_execution(zfp,zfp_exec_serial)
 cdef array.array output_buf=allocate(ztype,datashape)

 if ztype == b'f':
     zfp_field_set_pointer(field,output_buf.data.as_floats)
 elif ztype == b'd':
     zfp_field_set_pointer(field,output_buf.data.as_doubles)
 ret=zfp_decompress(zfp,field)
 if not ret: 
    print "decompress failed"
    sys.exit(1)

 zfp_field_free(field)
 zfp_stream_close(zfp)
 stream_close(stream)

 dtype = numpy.float32 if ztype == b'f' else numpy.float64
 if not ny:
   buf_shape=(nx)
 elif ny and not nz:
   buf_shape=(nx,ny)
 elif ny and nz and not nw:
   buf_shape=(nx,ny,nz)
 else:
   buf_shape=(nx,ny,nz,nw)

 return numpy.frombuffer(output_buf,dtype=dtype).reshape(buf_shape,order='C')


class Zfp(Codec):
 """Codec providing compression using zfp via the Python standard library.
 
 Parameters
 ----------
 mode : a
 level : float
 
 mode : p
 precision : int

 mode : r
 rate : int

 mode : c
 minbits : int
 maxbits : int
 minexps : float
 precision : int
 """
 codec_id = 'zfp'

 def __int__(self, mode = 'a', minbits = 0, maxbits = 10 , maxprec=32, minexp= 0.1, rate=0.1):
    self.mode=mode
    if mode == 'a':
       self.minexp=minexp
    elif mode == 'p':
       self.maxprec = maxprec
    elif mode == 'r':
       self.rate = rate
    elif mode == 'c':
       self.minbits = minbits
       self.maxbits = maxbits
       self.maxprec = maxprec
       self.minexp  = minexp
    else:
         "Wrong mode"
         sys.exit(1)

 def encode(self,buf):
      buf = ensure_contiguous_ndarray(buf)
      return compress(buf)

 def decode(self,buf,out=None):
     buf=ensure_contiguous_ndarray(buf)
     return decompress(buf)

 def __repr__(self):
     if self.mode == 'a':
        r= '%s(minexp=%r)' % ("accuracy",self.minexp)
     elif self.mode == 'p':
        r= '%s(maxprec=%r)' % ("precision",self.maxprec)
     elif self.mode == 'r':
        r= '%s(rate=%r)' % ("rate",self.rate)
     elif self.mode == 'c':
        r= '%s(minbits=%r,maxbits=%r,maxprec=%r,minexp=%r)' % ("rate",self.minbits,self.maxbits,self.maxprec,self.minexp)
     else:
        r="WRONG MODE"
     return r
      
