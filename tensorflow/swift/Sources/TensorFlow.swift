
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
// C API for TensorFlow.
//
// The API leans towards simplicity and uniformity instead of convenience
// since most usage will be by language specific wrappers.
//
// Conventions:
// * We use the prefix TF_ for everything in the API.
// * Objects are always passed around as pointers to opaque structs
//   and these structs are allocated/deallocated via the API.
// * TF_Status holds error information.  It is an object type
//   and therefore is passed around as a pointer to an opaque
//   struct as mentioned above.
// * Every call that has a TF_Status* argument clears it on success
//   and fills it with error info on failure.
// * unsigned char is used for booleans (instead of the 'bool' type).
//   In C++ bool is a keyword while in C99 bool is a macro defined
//   in stdbool.h. It is possible for the two to be inconsistent.
//   For example, neither the C99 nor the C++11 standard force a byte
//   size on the bool type, so the macro defined in stdbool.h could
//   be inconsistent with the bool keyword in C++. Thus, the use
//   of stdbool.h is avoided and unsigned char is used instead.
// * size_t is used to represent byte sizes of objects that are
//   materialized in the address space of the calling process.
// * int is used as an index into arrays.
//
// Questions left to address:
// * Might at some point need a way for callers to provide their own Env.
// * Maybe add TF_TensorShape that encapsulates dimension info.
//
// Design decisions made:
// * Backing store for tensor memory has an associated deallocation
//   function.  This deallocation function will point to client code
//   for tensors populated by the client.  So the client can do things
//   like shadowing a numpy array.
// * We do not provide TF_OK since it is not strictly necessary and we
//   are not optimizing for convenience.
// * We make assumption that one session has one graph.  This should be
//   fine since we have the ability to run sub-graphs.
// * We could allow NULL for some arguments (e.g., NULL options arg).
//   However since convenience is not a primary goal, we don't do this.
// * Devices are not in this API.  Instead, they are created/used internally
//   and the API just provides high level controls over the number of
//   devices of each type.

public typealias TF_Session = OpaquePointer
public typealias TF_Tensor = OpaquePointer
public typealias TF_Status = OpaquePointer
public typealias TF_SessionOptions = OpaquePointer
public typealias TF_Graph = OpaquePointer
public typealias TF_Library = OpaquePointer
public typealias TF_DeprecatedSession = OpaquePointer
public typealias TF_OperationDescription = OpaquePointer
public typealias TF_Operation = OpaquePointer


//public typealias TF_Buffer = OpaquePointer


// --------------------------------------------------------------------------
// TF_Version returns a string describing version information of the
// TensorFlow library. TensorFlow using semantic versioning.
public func tfVersion() ->String{
    let str:UnsafePointer<Int8> = TF_Version()
    return  String(cString:str)

}

// --------------------------------------------------------------------------
// TF_DataType holds the type for a scalar value.  E.g., one slot in a tensor.
// The enum values here are identical to corresponding values in types.proto.
/*public struct TF_DataType : RawRepresentable, Equatable {

    public init(_ rawValue: UInt32){
        self = TF_DataType(rawValue)
    }

    public init(rawValue: UInt32){
        self = TF_DataType(rawValue)
    }

    public var rawValue: UInt32
}*/

// Int32 tensors are always in 'host' memory.

// Single-precision complex
// Old identifier kept for API backwards compatibility

// Quantized int8
// Quantized uint8
// Quantized int32
// Float32 truncated to 16 bits.  Only for cast ops.
// Quantized int16
// Quantized uint16

// Double-precision complex

// TF_DataTypeSize returns the sizeof() for the underlying type corresponding
// to the given TF_DataType enum value. Returns 0 for variable length types
// (eg. TF_STRING) or on failure.
public func tfDataTypeSize(_ dt: TF_DataType) -> Int{
    let typeSize =  TF_DataTypeSize(dt) as Int
    return typeSize
}

// --------------------------------------------------------------------------
// TF_Code holds an error code.  The enum values here are identical to
// corresponding values in error_codes.proto.
/*public struct TF_Code : RawRepresentable, Equatable {

    public init(_ rawValue: UInt32){
        self = TF_Code(rawValue)
    }

    public init(rawValue: UInt32){
        self = TF_Code(rawValue)
    }

    public var rawValue: UInt32
}*/


// --------------------------------------------------------------------------
// TF_Buffer holds a pointer to a block of data and its associated length.
// Typically, the data consists of a serialized protocol buffer, but other data
// may also be held in a buffer.
//
// By default, TF_Buffer itself does not do any memory management of the
// pointed-to block.  If need be, users of this struct should specify how to
// deallocate the block by setting the `data_deallocator` function pointer.
/*public struct tfBuffer {

    var c:TF_Buffer
    
    public var data: UnsafeRawPointer{
        get { return self.c.data }
    }

    public var length: Int{
        get {return self.c.length}
    }

    public var data_deallocator: (@convention(c) (UnsafeMutableRawPointer?, Int) -> Swift.Void)!{
        get {return self.c.data_deallocator}
    }

    

    public init(data: UnsafeRawPointer!, length: Int, data_deallocator: (@convention(c) (UnsafeMutableRawPointer?, Int) -> Swift.Void)!){
      self = tfBuffer(data:nil,length:0,data_deallocator:nil)
        self.c = TF_Buffer(data:data,length:length,data_deallocator:data_deallocator)
    }
}*/




