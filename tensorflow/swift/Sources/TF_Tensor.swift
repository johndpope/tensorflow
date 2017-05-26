
/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/

import Darwin.C.stddef
import Darwin.C.stdint
import CTensorFlow
import protoTensorFlow


// --------------------------------------------------------------------------
// TF_Tensor holds a multi-dimensional array of elements of a single data type.
// For all types other than TF_STRING, the data buffer stores elements
// in row major order.  E.g. if data is treated as a vector of TF_DataType:
//
//   element 0:   index (0, ..., 0)
//   element 1:   index (0, ..., 1)
//   ...
//
// The format for TF_STRING tensors is:
//   start_offset: array[uint64]
//   data:         byte[...]
//
//   The string length (as a varint), followed by the contents of the string
//   is encoded at data[start_offset[i]]]. TF_StringEncode and TF_StringDecode
//   facilitate this encoding.

// Return a new tensor that holds the bytes data[0,len-1].
//
// The data will be deallocated by a subsequent call to TF_DeleteTensor via:
//      (*deallocator)(data, len, deallocator_arg)
// Clients must provide a custom deallocator function so they can pass in
// memory managed by something like numpy.
public func tfNewTensor(dt: TF_DataType, _ dims: UnsafePointer<Int64>!, _ num_dims: Int32, _ data: UnsafeMutableRawPointer!, _ len: Int, _ deallocator: (@convention(c) (UnsafeMutableRawPointer?, Int, UnsafeMutableRawPointer?) -> Swift.Void)!, _ deallocator_arg: UnsafeMutableRawPointer!) -> TF_Tensor!{
    return TF_NewTensor(dt,dims, num_dims, data, len, deallocator, deallocator_arg)
  
}



// Allocate and return a new Tensor.
//
// This function is an alternative to TF_NewTensor and should be used when
// memory is allocated to pass the Tensor to the C API. The allocated memory
// satisfies TensorFlow's memory alignment preferences and should be preferred
// over calling malloc and free.
//
// The caller must set the Tensor values by writing them to the pointer returned
// by TF_TensorData with length TF_TensorByteSize.
public func tfAllocateTensor(dt: TF_DataType, _ dims: UnsafePointer<Int64>!, _ num_dims: Int32, _ len: Int) -> TF_Tensor!{
    return TF_AllocateTensor(dt,dims,num_dims,len)
}

// Destroy a tensor.
public func tfDeleteTensor(_ pointer:TF_Tensor!){
    return TF_DeleteTensor(pointer)
}

// Return the type of a tensor element.
public func tfTensorType(_ pointer:TF_Tensor!) -> TF_DataType{
    return TF_TensorType(pointer)
}

// Return the number of dimensions that the tensor has.
public func tfNumDims(_ pointer: TF_Tensor!) -> Int32{
    return TF_NumDims(pointer)
}

// Return the length of the tensor in the "dim_index" dimension.
// REQUIRES: 0 <= dim_index < TF_NumDims(tensor)
public func tfDim(_ tensor: TF_Tensor!, _ dim_index: Int32) -> Int64{
    return TF_Dim(tensor,dim_index)
}

// Return the size of the underlying data in bytes.
public func tfTensorByteSize(_ pointer: TF_Tensor!) -> Int{
    return  TF_TensorByteSize(pointer)
}

// Return a pointer to the underlying data buffer.
public func tfTensorData(_ pointer: TF_Tensor!) -> UnsafeMutableRawPointer!{
    return TF_TensorData(pointer)
}


