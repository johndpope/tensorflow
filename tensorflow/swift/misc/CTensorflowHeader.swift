
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

// --------------------------------------------------------------------------
// TF_Version returns a string describing version information of the
// TensorFlow library. TensorFlow using semantic versioning.
public func TF_Version() -> UnsafePointer<Int8>!

// --------------------------------------------------------------------------
// TF_DataType holds the type for a scalar value.  E.g., one slot in a tensor.
// The enum values here are identical to corresponding values in types.proto.
public struct TF_DataType : RawRepresentable, Equatable {

    public init(_ rawValue: UInt32)

    public init(rawValue: UInt32)

    public var rawValue: UInt32
}

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
public func TF_DataTypeSize(_ dt: TF_DataType) -> Int

// --------------------------------------------------------------------------
// TF_Code holds an error code.  The enum values here are identical to
// corresponding values in error_codes.proto.
public struct TF_Code : RawRepresentable, Equatable {

    public init(_ rawValue: UInt32)

    public init(rawValue: UInt32)

    public var rawValue: UInt32
}

// --------------------------------------------------------------------------
// TF_Status holds error information.  It either has an OK code, or
// else an error code with an associated error message.

// Return a new status object.
public func TF_NewStatus() -> OpaquePointer!

// Delete a previously created status object.
public func TF_DeleteStatus(_: OpaquePointer!)

// Record <code, msg> in *s.  Any previous information is lost.
// A common use is to clear a status: TF_SetStatus(s, TF_OK, "");
public func TF_SetStatus(_ s: OpaquePointer!, _ code: TF_Code, _ msg: UnsafePointer<Int8>!)

// Return the code record in *s.
public func TF_GetCode(_ s: OpaquePointer!) -> TF_Code

// Return a pointer to the (null-terminated) error message in *s.  The
// return value points to memory that is only usable until the next
// mutation to *s.  Always returns an empty string if TF_GetCode(s) is
// TF_OK.
public func TF_Message(_ s: OpaquePointer!) -> UnsafePointer<Int8>!

// --------------------------------------------------------------------------
// TF_Buffer holds a pointer to a block of data and its associated length.
// Typically, the data consists of a serialized protocol buffer, but other data
// may also be held in a buffer.
//
// By default, TF_Buffer itself does not do any memory management of the
// pointed-to block.  If need be, users of this struct should specify how to
// deallocate the block by setting the `data_deallocator` function pointer.
public struct TF_Buffer {

    public var data: UnsafeRawPointer!

    public var length: Int

    public var data_deallocator: (@convention(c) (UnsafeMutableRawPointer?, Int) -> Swift.Void)!

    public init()

    public init(data: UnsafeRawPointer!, length: Int, data_deallocator: (@convention(c) (UnsafeMutableRawPointer?, Int) -> Swift.Void)!)
}

// Makes a copy of the input and sets an appropriate deallocator.  Useful for
// passing in read-only, input protobufs.
public func TF_NewBufferFromString(_ proto: UnsafeRawPointer!, _ proto_len: Int) -> UnsafeMutablePointer<TF_Buffer>!

// Useful for passing *out* a protobuf.
public func TF_NewBuffer() -> UnsafeMutablePointer<TF_Buffer>!

public func TF_DeleteBuffer(_: UnsafeMutablePointer<TF_Buffer>!)

public func TF_GetBuffer(_ buffer: UnsafeMutablePointer<TF_Buffer>!) -> TF_Buffer

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
public func TF_NewTensor(_: TF_DataType, _ dims: UnsafePointer<Int64>!, _ num_dims: Int32, _ data: UnsafeMutableRawPointer!, _ len: Int, _ deallocator: (@convention(c) (UnsafeMutableRawPointer?, Int, UnsafeMutableRawPointer?) -> Swift.Void)!, _ deallocator_arg: UnsafeMutableRawPointer!) -> OpaquePointer!

// Allocate and return a new Tensor.
//
// This function is an alternative to TF_NewTensor and should be used when
// memory is allocated to pass the Tensor to the C API. The allocated memory
// satisfies TensorFlow's memory alignment preferences and should be preferred
// over calling malloc and free.
//
// The caller must set the Tensor values by writing them to the pointer returned
// by TF_TensorData with length TF_TensorByteSize.
public func TF_AllocateTensor(_: TF_DataType, _ dims: UnsafePointer<Int64>!, _ num_dims: Int32, _ len: Int) -> OpaquePointer!

// Destroy a tensor.
public func TF_DeleteTensor(_: OpaquePointer!)

// Return the type of a tensor element.
public func TF_TensorType(_: OpaquePointer!) -> TF_DataType

// Return the number of dimensions that the tensor has.
public func TF_NumDims(_: OpaquePointer!) -> Int32

// Return the length of the tensor in the "dim_index" dimension.
// REQUIRES: 0 <= dim_index < TF_NumDims(tensor)
public func TF_Dim(_ tensor: OpaquePointer!, _ dim_index: Int32) -> Int64

// Return the size of the underlying data in bytes.
public func TF_TensorByteSize(_: OpaquePointer!) -> Int

// Return a pointer to the underlying data buffer.
public func TF_TensorData(_: OpaquePointer!) -> UnsafeMutableRawPointer!

// --------------------------------------------------------------------------
// Encode the string `src` (`src_len` bytes long) into `dst` in the format
// required by TF_STRING tensors. Does not write to memory more than `dst_len`
// bytes beyond `*dst`. `dst_len` should be at least
// TF_StringEncodedSize(src_len).
//
// On success returns the size in bytes of the encoded string.
// Returns an error into `status` otherwise.
public func TF_StringEncode(_ src: UnsafePointer<Int8>!, _ src_len: Int, _ dst: UnsafeMutablePointer<Int8>!, _ dst_len: Int, _ status: OpaquePointer!) -> Int

// Decode a string encoded using TF_StringEncode.
//
// On success, sets `*dst` to the start of the decoded string and `*dst_len` to
// its length. Returns the number of bytes starting at `src` consumed while
// decoding. `*dst` points to memory within the encoded buffer.  On failure,
// `*dst` and `*dst_len` are undefined and an error is set in `status`.
//
// Does not read memory more than `src_len` bytes beyond `src`.
public func TF_StringDecode(_ src: UnsafePointer<Int8>!, _ src_len: Int, _ dst: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ dst_len: UnsafeMutablePointer<Int>!, _ status: OpaquePointer!) -> Int

// Return the size in bytes required to encode a string `len` bytes long into a
// TF_STRING tensor.
public func TF_StringEncodedSize(_ len: Int) -> Int

// --------------------------------------------------------------------------
// TF_SessionOptions holds options that can be passed during session creation.

// Return a new options object.
public func TF_NewSessionOptions() -> OpaquePointer!

// Set the target in TF_SessionOptions.options.
// target can be empty, a single entry, or a comma separated list of entries.
// Each entry is in one of the following formats :
// "local"
// ip:port
// host:port
public func TF_SetTarget(_ options: OpaquePointer!, _ target: UnsafePointer<Int8>!)

// Set the config in TF_SessionOptions.options.
// config should be a serialized tensorflow.ConfigProto proto.
// If config was not parsed successfully as a ConfigProto, record the
// error information in *status.
public func TF_SetConfig(_ options: OpaquePointer!, _ proto: UnsafeRawPointer!, _ proto_len: Int, _ status: OpaquePointer!)

// Destroy an options object.
public func TF_DeleteSessionOptions(_: OpaquePointer!)

// TODO(jeff,sanjay):
// - export functions to set Config fields

// --------------------------------------------------------------------------
// The new graph construction API, still under development.

// Represents a computation graph.  Graphs may be shared between sessions.
// Graphs are thread-safe when used as directed below.

// Return a new graph object.
public func TF_NewGraph() -> OpaquePointer!

// Destroy an options object.  Graph will be deleted once no more
// TFSession's are referencing it.
public func TF_DeleteGraph(_: OpaquePointer!)

// Operation being built. The underlying graph must outlive this.

// Operation that has been added to the graph. Valid until the graph is
// deleted -- in particular adding a new operation to the graph does not
// invalidate old TF_Operation* pointers.

// Represents a specific input of an operation.
public struct TF_Input {

    public var oper: OpaquePointer!

    public var index: Int32 // The index of the input within oper.

    public init()

    public init(oper: OpaquePointer!, index: Int32)
}

// Represents a specific output of an operation.
public struct TF_Output {

    public var oper: OpaquePointer!

    public var index: Int32 // The index of the output within oper.

    public init()

    public init(oper: OpaquePointer!, index: Int32)
}

// Sets the shape of the Tensor referenced by `output` in `graph` to
// the shape described by `dims` and `num_dims`.
//
// If the number of dimensions is unknown, `num_dims` must be
// set to -1 and dims can be null. If a dimension is unknown,
// the corresponding entry in the `dims` array must be -1.
//
// This does not overwrite the existing shape associated with `output`,
// but merges the input shape with the existing shape.  For example,
// setting a shape of [-1, 2] with an existing shape [2, -1] would set
// a final shape of [2, 2] based on shape merging semantics.
//
// Returns an error into `status` if:
//   * `output` is not in `graph`.
//   * An invalid shape is being set (e.g., the shape being set
//     is incompatible with the existing shape).
public func TF_GraphSetTensorShape(_ graph: OpaquePointer!, _ output: TF_Output, _ dims: UnsafePointer<Int64>!, _ num_dims: Int32, _ status: OpaquePointer!)

// Returns the number of dimensions of the Tensor referenced by `output`
// in `graph`.
//
// If the number of dimensions in the shape is unknown, returns -1.
//
// Returns an error into `status` if:
//   * `output` is not in `graph`.
public func TF_GraphGetTensorNumDims(_ graph: OpaquePointer!, _ output: TF_Output, _ status: OpaquePointer!) -> Int32

// Returns the shape of the Tensor referenced by `output` in `graph`
// into `dims`. `dims` must be an array large enough to hold `num_dims`
// entries (e.g., the return value of TF_GraphGetTensorNumDims).
//
// If the number of dimensions in the shape is unknown or the shape is
// a scalar, `dims` will remain untouched. Otherwise, each element of
// `dims` will be set corresponding to the size of the dimension. An
// unknown dimension is represented by `-1`.
//
// Returns an error into `status` if:
//   * `output` is not in `graph`.
//   * `num_dims` does not match the actual number of dimensions.
public func TF_GraphGetTensorShape(_ graph: OpaquePointer!, _ output: TF_Output, _ dims: UnsafeMutablePointer<Int64>!, _ num_dims: Int32, _ status: OpaquePointer!)

// Operation will only be added to *graph when TF_FinishOperation() is
// called (assuming TF_FinishOperation() does not return an error).
// *graph must not be deleted until after TF_FinishOperation() is
// called.
public func TF_NewOperation(_ graph: OpaquePointer!, _ op_type: UnsafePointer<Int8>!, _ oper_name: UnsafePointer<Int8>!) -> OpaquePointer!

// Specify the device for `desc`.  Defaults to empty, meaning unconstrained.
public func TF_SetDevice(_ desc: OpaquePointer!, _ device: UnsafePointer<Int8>!)

// The calls to TF_AddInput and TF_AddInputList must match (in number,
// order, and type) the op declaration.  For example, the "Concat" op
// has registration:
//   REGISTER_OP("Concat")
//       .Input("concat_dim: int32")
//       .Input("values: N * T")
//       .Output("output: T")
//       .Attr("N: int >= 2")
//       .Attr("T: type");
// that defines two inputs, "concat_dim" and "values" (in that order).
// You must use TF_AddInput() for the first input (since it takes a
// single tensor), and TF_AddInputList() for the second input (since
// it takes a list, even if you were to pass a list with a single
// tensor), as in:
//   TF_OperationDescription* desc = TF_NewOperation(graph, "Concat", "c");
//   TF_Output concat_dim_input = {...};
//   TF_AddInput(desc, concat_dim_input);
//   TF_Output values_inputs[5] = {{...}, ..., {...}};
//   TF_AddInputList(desc, values_inputs, 5);

// For inputs that take a single tensor.
public func TF_AddInput(_ desc: OpaquePointer!, _ input: TF_Output)

// For inputs that take a list of tensors.
// inputs must point to TF_Output[num_inputs].
public func TF_AddInputList(_ desc: OpaquePointer!, _ inputs: UnsafePointer<TF_Output>!, _ num_inputs: Int32)

// Call once per control input to `desc`.
public func TF_AddControlInput(_ desc: OpaquePointer!, _ input: OpaquePointer!)

// Request that `desc` be co-located on the device where `op`
// is placed.
//
// Use of this is discouraged since the implementation of device placement is
// subject to change. Primarily intended for internal libraries
public func TF_ColocateWith(_ desc: OpaquePointer!, _ op: OpaquePointer!)

// Call some TF_SetAttr*() function for every attr that is not
// inferred from an input and doesn't have a default value you wish to
// keep.

// `value` must point to a string of length `length` bytes.
public func TF_SetAttrString(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeRawPointer!, _ length: Int)
// `values` and `lengths` each must have lengths `num_values`.
// `values[i]` must point to a string of length `lengths[i]` bytes.
public func TF_SetAttrStringList(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafePointer<UnsafeRawPointer?>!, _ lengths: UnsafePointer<Int>!, _ num_values: Int32)
public func TF_SetAttrInt(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: Int64)
public func TF_SetAttrIntList(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafePointer<Int64>!, _ num_values: Int32)
public func TF_SetAttrFloat(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: Float)
public func TF_SetAttrFloatList(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafePointer<Float>!, _ num_values: Int32)
public func TF_SetAttrBool(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UInt8)
public func TF_SetAttrBoolList(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafePointer<UInt8>!, _ num_values: Int32)
public func TF_SetAttrType(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: TF_DataType)
public func TF_SetAttrTypeList(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafePointer<TF_DataType>!, _ num_values: Int32)

// Set `num_dims` to -1 to represent "unknown rank".  Otherwise,
// `dims` points to an array of length `num_dims`.  `dims[i]` must be
// >= -1, with -1 meaning "unknown dimension".
public func TF_SetAttrShape(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ dims: UnsafePointer<Int64>!, _ num_dims: Int32)
// `dims` and `num_dims` must point to arrays of length `num_shapes`.
// Set `num_dims[i]` to -1 to represent "unknown rank".  Otherwise,
// `dims[i]` points to an array of length `num_dims[i]`.  `dims[i][j]`
// must be >= -1, with -1 meaning "unknown dimension".
public func TF_SetAttrShapeList(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ dims: UnsafePointer<UnsafePointer<Int64>?>!, _ num_dims: UnsafePointer<Int32>!, _ num_shapes: Int32)
// `proto` must point to an array of `proto_len` bytes representing a
// binary-serialized TensorShapeProto.
public func TF_SetAttrTensorShapeProto(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ proto: UnsafeRawPointer!, _ proto_len: Int, _ status: OpaquePointer!)
// `protos` and `proto_lens` must point to arrays of length `num_shapes`.
// `protos[i]` must point to an array of `proto_lens[i]` bytes
// representing a binary-serialized TensorShapeProto.
public func TF_SetAttrTensorShapeProtoList(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ protos: UnsafePointer<UnsafeRawPointer?>!, _ proto_lens: UnsafePointer<Int>!, _ num_shapes: Int32, _ status: OpaquePointer!)

public func TF_SetAttrTensor(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: OpaquePointer!, _ status: OpaquePointer!)
public func TF_SetAttrTensorList(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafePointer<OpaquePointer?>!, _ num_values: Int32, _ status: OpaquePointer!)

// `proto` should point to a sequence of bytes of length `proto_len`
// representing a binary serialization of an AttrValue protocol
// buffer.
public func TF_SetAttrValueProto(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ proto: UnsafeRawPointer!, _ proto_len: Int, _ status: OpaquePointer!)

// If this function succeeds:
//   * *status is set to an OK value,
//   * a TF_Operation is added to the graph,
//   * a non-null value pointing to the added operation is returned --
//     this value is valid until the underlying graph is deleted.
// Otherwise:
//   * *status is set to a non-OK value,
//   * the graph is not modified,
//   * a null value is returned.
// In either case, it deletes `desc`.
public func TF_FinishOperation(_ desc: OpaquePointer!, _ status: OpaquePointer!) -> OpaquePointer!

// TF_Operation functions.  Operations are immutable once created, so
// these are all query functions.

public func TF_OperationName(_ oper: OpaquePointer!) -> UnsafePointer<Int8>!
public func TF_OperationOpType(_ oper: OpaquePointer!) -> UnsafePointer<Int8>!
public func TF_OperationDevice(_ oper: OpaquePointer!) -> UnsafePointer<Int8>!

public func TF_OperationNumOutputs(_ oper: OpaquePointer!) -> Int32
public func TF_OperationOutputType(_ oper_out: TF_Output) -> TF_DataType
public func TF_OperationOutputListLength(_ oper: OpaquePointer!, _ arg_name: UnsafePointer<Int8>!, _ status: OpaquePointer!) -> Int32

public func TF_OperationNumInputs(_ oper: OpaquePointer!) -> Int32
public func TF_OperationInputType(_ oper_in: TF_Input) -> TF_DataType
public func TF_OperationInputListLength(_ oper: OpaquePointer!, _ arg_name: UnsafePointer<Int8>!, _ status: OpaquePointer!) -> Int32

// In this code:
//   TF_Output producer = TF_OperationInput(consumer);
// There is an edge from producer.oper's output (given by
// producer.index) to consumer.oper's input (given by consumer.index).
public func TF_OperationInput(_ oper_in: TF_Input) -> TF_Output

// Get the number of current consumers of a specific output of an
// operation.  Note that this number can change when new operations
// are added to the graph.
public func TF_OperationOutputNumConsumers(_ oper_out: TF_Output) -> Int32

// Get list of all current consumers of a specific output of an
// operation.  `consumers` must point to an array of length at least
// `max_consumers` (ideally set to
// TF_OperationOutputNumConsumers(oper_out)).  Beware that a concurrent
// modification of the graph can increase the number of consumers of
// an operation.  Returns the number of output consumers (should match
// TF_OperationOutputNumConsumers(oper_out)).
public func TF_OperationOutputConsumers(_ oper_out: TF_Output, _ consumers: UnsafeMutablePointer<TF_Input>!, _ max_consumers: Int32) -> Int32

// Get the number of control inputs to an operation.
public func TF_OperationNumControlInputs(_ oper: OpaquePointer!) -> Int32

// Get list of all control inputs to an operation.  `control_inputs` must
// point to an array of length `max_control_inputs` (ideally set to
// TF_OperationNumControlInputs(oper)).  Returns the number of control
// inputs (should match TF_OperationNumControlInputs(oper)).
public func TF_OperationGetControlInputs(_ oper: OpaquePointer!, _ control_inputs: UnsafeMutablePointer<OpaquePointer?>!, _ max_control_inputs: Int32) -> Int32

// Get the number of operations that have `*oper` as a control input.
// Note that this number can change when new operations are added to
// the graph.
public func TF_OperationNumControlOutputs(_ oper: OpaquePointer!) -> Int32

// Get the list of operations that have `*oper` as a control input.
// `control_outputs` must point to an array of length at least
// `max_control_outputs` (ideally set to
// TF_OperationNumControlOutputs(oper)). Beware that a concurrent
// modification of the graph can increase the number of control
// outputs.  Returns the number of control outputs (should match
// TF_OperationNumControlOutputs(oper)).
public func TF_OperationGetControlOutputs(_ oper: OpaquePointer!, _ control_outputs: UnsafeMutablePointer<OpaquePointer?>!, _ max_control_outputs: Int32) -> Int32

// TF_AttrType describes the type of the value of an attribute on an operation.
public struct TF_AttrType : RawRepresentable, Equatable {

    public init(_ rawValue: UInt32)

    public init(rawValue: UInt32)

    public var rawValue: UInt32
}

// TF_AttrMetadata describes the value of an attribute on an operation.
public struct TF_AttrMetadata {

    // A boolean: 1 if the attribute value is a list, 0 otherwise.
    public var is_list: UInt8

    
    // Length of the list if is_list is true. Undefined otherwise.
    public var list_size: Int64

    
    // Type of elements of the list if is_list != 0.
    // Type of the single value stored in the attribute if is_list == 0.
    public var type: TF_AttrType

    
    // Total size the attribute value.
    // The units of total_size depend on is_list and type.
    // (1) If type == TF_ATTR_STRING and is_list == 0
    //     then total_size is the byte size of the string
    //     valued attribute.
    // (2) If type == TF_ATTR_STRING and is_list == 1
    //     then total_size is the cumulative byte size
    //     of all the strings in the list.
    // (3) If type == TF_ATTR_SHAPE and is_list == 0
    //     then total_size is the number of dimensions
    //     of the shape valued attribute, or -1
    //     if its rank is unknown.
    // (4) If type == TF_ATTR_SHAPE and is_list == 1
    //     then total_size is the cumulative number
    //     of dimensions of all shapes in the list.
    // (5) Otherwise, total_size is undefined.
    public var total_size: Int64

    public init()

    public init(is_list: UInt8, list_size: Int64, type: TF_AttrType, total_size: Int64)
}

// Returns metadata about the value of the attribute `attr_name` of `oper`.
public func TF_OperationGetAttrMetadata(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ status: OpaquePointer!) -> TF_AttrMetadata

// Fills in `value` with the value of the attribute `attr_name`.  `value` must
// point to an array of length at least `max_length` (ideally set to
// TF_AttrMetadata.total_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
public func TF_OperationGetAttrString(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeMutableRawPointer!, _ max_length: Int, _ status: OpaquePointer!)

// Get the list of strings in the value of the attribute `attr_name`.  Fills in
// `values` and `lengths`, each of which must point to an array of length at
// least `max_values`.
//
// The elements of values will point to addresses in `storage` which must be at
// least `storage_size` bytes in length.  Ideally, max_values would be set to
// TF_AttrMetadata.list_size and `storage` would be at least
// TF_AttrMetadata.total_size, obtained from TF_OperationGetAttrMetadata(oper,
// attr_name).
//
// Fails if storage_size is too small to hold the requested number of strings.
public func TF_OperationGetAttrStringList(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafeMutablePointer<UnsafeMutableRawPointer?>!, _ lengths: UnsafeMutablePointer<Int>!, _ max_values: Int32, _ storage: UnsafeMutableRawPointer!, _ storage_size: Int, _ status: OpaquePointer!)

public func TF_OperationGetAttrInt(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeMutablePointer<Int64>!, _ status: OpaquePointer!)

// Fills in `values` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `max_values` (ideally set
// TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
public func TF_OperationGetAttrIntList(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafeMutablePointer<Int64>!, _ max_values: Int32, _ status: OpaquePointer!)

public func TF_OperationGetAttrFloat(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeMutablePointer<Float>!, _ status: OpaquePointer!)

// Fills in `values` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `max_values` (ideally set
// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
public func TF_OperationGetAttrFloatList(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafeMutablePointer<Float>!, _ max_values: Int32, _ status: OpaquePointer!)

public func TF_OperationGetAttrBool(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeMutablePointer<UInt8>!, _ status: OpaquePointer!)

// Fills in `values` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `max_values` (ideally set
// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
public func TF_OperationGetAttrBoolList(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafeMutablePointer<UInt8>!, _ max_values: Int32, _ status: OpaquePointer!)

public func TF_OperationGetAttrType(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeMutablePointer<TF_DataType>!, _ status: OpaquePointer!)

// Fills in `values` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `max_values` (ideally set
// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
public func TF_OperationGetAttrTypeList(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafeMutablePointer<TF_DataType>!, _ max_values: Int32, _ status: OpaquePointer!)

// Fills in `value` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `num_dims` (ideally set to
// TF_Attr_Meta.size from TF_OperationGetAttrMetadata(oper, attr_name)).
public func TF_OperationGetAttrShape(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeMutablePointer<Int64>!, _ num_dims: Int32, _ status: OpaquePointer!)

// Fills in `dims` with the list of shapes in the attribute `attr_name` of
// `oper` and `num_dims` with the corresponding number of dimensions. On return,
// for every i where `num_dims[i]` > 0, `dims[i]` will be an array of
// `num_dims[i]` elements. A value of -1 for `num_dims[i]` indicates that the
// i-th shape in the list is unknown.
//
// The elements of `dims` will point to addresses in `storage` which must be
// large enough to hold at least `storage_size` int64_ts.  Ideally, `num_shapes`
// would be set to TF_AttrMetadata.list_size and `storage_size` would be set to
// TF_AttrMetadata.total_size from TF_OperationGetAttrMetadata(oper,
// attr_name).
//
// Fails if storage_size is insufficient to hold the requested shapes.
public func TF_OperationGetAttrShapeList(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ dims: UnsafeMutablePointer<UnsafeMutablePointer<Int64>?>!, _ num_dims: UnsafeMutablePointer<Int32>!, _ num_shapes: Int32, _ storage: UnsafeMutablePointer<Int64>!, _ storage_size: Int32, _ status: OpaquePointer!)

// Sets `value` to the binary-serialized TensorShapeProto of the value of
// `attr_name` attribute of `oper`'.
public func TF_OperationGetAttrTensorShapeProto(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeMutablePointer<TF_Buffer>!, _ status: OpaquePointer!)

// Fills in `values` with binary-serialized TensorShapeProto values of the
// attribute `attr_name` of `oper`. `values` must point to an array of length at
// least `num_values` (ideally set to TF_AttrMetadata.list_size from
// TF_OperationGetAttrMetadata(oper, attr_name)).
public func TF_OperationGetAttrTensorShapeProtoList(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafeMutablePointer<UnsafeMutablePointer<TF_Buffer>?>!, _ max_values: Int32, _ status: OpaquePointer!)

// Gets the TF_Tensor valued attribute of `attr_name` of `oper`.
//
// Allocates a new TF_Tensor which the caller is expected to take
// ownership of (and can deallocate using TF_DeleteTensor).
public func TF_OperationGetAttrTensor(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeMutablePointer<OpaquePointer?>!, _ status: OpaquePointer!)

// Fills in `values` with the TF_Tensor values of the attribute `attr_name` of
// `oper`. `values` must point to an array of TF_Tensor* of length at least
// `max_values` (ideally set to TF_AttrMetadata.list_size from
// TF_OperationGetAttrMetadata(oper, attr_name)).
//
// The caller takes ownership of all the non-null TF_Tensor* entries in `values`
// (which can be deleted using TF_DeleteTensor(values[i])).
public func TF_OperationGetAttrTensorList(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafeMutablePointer<OpaquePointer?>!, _ max_values: Int32, _ status: OpaquePointer!)

// Sets `output_attr_value` to the binary-serialized AttrValue proto
// representation of the value of the `attr_name` attr of `oper`.
public func TF_OperationGetAttrValueProto(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ output_attr_value: UnsafeMutablePointer<TF_Buffer>!, _ status: OpaquePointer!)

// Returns the operation in the graph with `oper_name`. Returns nullptr if
// no operation found.
public func TF_GraphOperationByName(_ graph: OpaquePointer!, _ oper_name: UnsafePointer<Int8>!) -> OpaquePointer!

// Iterate through the operations of a graph.  To use:
// size_t pos = 0;
// TF_Operation* oper;
// while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
//   DoSomethingWithOperation(oper);
// }
public func TF_GraphNextOperation(_ graph: OpaquePointer!, _ pos: UnsafeMutablePointer<Int>!) -> OpaquePointer!

// Write out a serialized representation of `graph` (as a GraphDef protocol
// message) to `output_graph_def` (allocated by TF_NewBuffer()).
// `output_graph_def`'s underlying buffer will be freed when TF_DeleteBuffer()
// is called.
//
// May fail on very large graphs in the future.
public func TF_GraphToGraphDef(_ graph: OpaquePointer!, _ output_graph_def: UnsafeMutablePointer<TF_Buffer>!, _ status: OpaquePointer!)

// TF_ImportGraphDefOptions holds options that can be passed to
// TF_GraphImportGraphDef.

public func TF_NewImportGraphDefOptions() -> OpaquePointer!
public func TF_DeleteImportGraphDefOptions(_ opts: OpaquePointer!)

// Set the prefix to be prepended to the names of nodes in `graph_def` that will
// be imported into `graph`.
public func TF_ImportGraphDefOptionsSetPrefix(_ opts: OpaquePointer!, _ prefix: UnsafePointer<Int8>!)

// Set any imported nodes with input `src_name:src_index` to have that input
// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
// `dst` references a node already existing in the graph being imported into.
public func TF_ImportGraphDefOptionsAddInputMapping(_ opts: OpaquePointer!, _ src_name: UnsafePointer<Int8>!, _ src_index: Int32, _ dst: TF_Output)

// Set any imported nodes with control input `src_name` to have that input
// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
// `dst` references an operation already existing in the graph being imported
// into.
public func TF_GraphImportGraphDefOptionsRemapControlDependency(_ opts: OpaquePointer!, _ src_name: UnsafePointer<Int8>!, _ dst: OpaquePointer!)

// Cause the imported graph to have a control dependency on `oper`. `oper`
// should exist in the graph being imported into.
public func TF_ImportGraphDefOptionsAddControlDependency(_ opts: OpaquePointer!, _ oper: OpaquePointer!)

// Add an output in `graph_def` to be returned via the `return_outputs` output
// parameter of TF_GraphImportGraphDef(). If the output is remapped via an input
// mapping, the corresponding existing tensor in `graph` will be returned.
public func TF_ImportGraphDefOptionsAddReturnOutput(_ opts: OpaquePointer!, _ oper_name: UnsafePointer<Int8>!, _ index: Int32)

// Returns the number of return outputs added via
// TF_ImportGraphDefOptionsAddReturnOutput().
public func TF_ImportGraphDefOptionsNumReturnOutputs(_ opts: OpaquePointer!) -> Int32

// Import the graph serialized in `graph_def` into `graph`.
//
// `num_return_outputs` must be the number of return outputs added (i.e. the
// result of TF_ImportGraphDefOptionsNumReturnOutputs()).  If
// `num_return_outputs` is non-zero, `return_outputs` must be of length
// `num_return_outputs`. Otherwise it can be null.
public func TF_GraphImportGraphDefWithReturnOutputs(_ graph: OpaquePointer!, _ graph_def: UnsafePointer<TF_Buffer>!, _ options: OpaquePointer!, _ return_outputs: UnsafeMutablePointer<TF_Output>!, _ num_return_outputs: Int32, _ status: OpaquePointer!)

// Import the graph serialized in `graph_def` into `graph`.
// Convenience function for when no return outputs have been added.
public func TF_GraphImportGraphDef(_ graph: OpaquePointer!, _ graph_def: UnsafePointer<TF_Buffer>!, _ options: OpaquePointer!, _ status: OpaquePointer!)

// Note: The following function may fail on very large protos in the future.

public func TF_OperationToNodeDef(_ oper: OpaquePointer!, _ output_node_def: UnsafeMutablePointer<TF_Buffer>!, _ status: OpaquePointer!)

public struct TF_WhileParams {

    // The number of inputs to the while loop, i.e. the number of loop variables.
    // This is the size of cond_inputs, body_inputs, and body_outputs.
    public var ninputs: Int32

    
    // The while condition graph. The inputs are the current values of the loop
    // variables. The output should be a scalar boolean.
    public var cond_graph: OpaquePointer!

    public var cond_inputs: UnsafePointer<TF_Output>!

    public var cond_output: TF_Output

    
    // The loop body graph. The inputs are the current values of the loop
    // variables. The outputs are the updated values of the loop variables.
    public var body_graph: OpaquePointer!

    public var body_inputs: UnsafePointer<TF_Output>!

    public var body_outputs: UnsafeMutablePointer<TF_Output>!

    
    // Unique null-terminated name for this while loop. This is used as a prefix
    // for created operations.
    public var name: UnsafePointer<Int8>!

    public init()

    public init(ninputs: Int32, cond_graph: OpaquePointer!, cond_inputs: UnsafePointer<TF_Output>!, cond_output: TF_Output, body_graph: OpaquePointer!, body_inputs: UnsafePointer<TF_Output>!, body_outputs: UnsafeMutablePointer<TF_Output>!, name: UnsafePointer<Int8>!)
}

// Creates a TF_WhileParams for creating a while loop in `g`. `inputs` are
// outputs that already exist in `g` used as initial values for the loop
// variables.
//
// The returned TF_WhileParams will have all fields initialized except
// `cond_output`, `body_outputs`, and `name`. The `body_outputs` buffer will be
// allocated to size `ninputs`. The caller should build `cond_graph` and
// `body_graph` starting from the inputs, and store the final outputs in
// `cond_output` and `body_outputs`.
//
// If `status` is OK, the caller must call either TF_FinishWhile or
// TF_AbortWhile on the returned TF_WhileParams. If `status` isn't OK, the
// returned TF_WhileParams is not valid, and the caller should not call
// TF_FinishWhile() or TF_AbortWhile().
//
// Missing functionality (TODO):
// - Gradients (not yet implmented for any ops)
// - Reference-type inputs
// - Directly referencing external tensors from the cond/body graphs (this is
//   possible in the Python API)
public func TF_NewWhile(_ g: OpaquePointer!, _ inputs: UnsafeMutablePointer<TF_Output>!, _ ninputs: Int32, _ status: OpaquePointer!) -> TF_WhileParams

// Builds the while loop specified by `params` and returns the output tensors of
// the while loop in `outputs`. `outputs` should be allocated to size
// `params.ninputs`.
//
// `params` is no longer valid once this returns.
//
// Either this or TF_AbortWhile() must be called after a successful
// TF_NewWhile() call.
public func TF_FinishWhile(_ params: UnsafePointer<TF_WhileParams>!, _ status: OpaquePointer!, _ outputs: UnsafeMutablePointer<TF_Output>!)

// Frees `params`s resources without building a while loop. `params` is no
// longer valid after this returns. Either this or TF_FinishWhile() must be
// called after a successful TF_NewWhile() call.
public func TF_AbortWhile(_ params: UnsafePointer<TF_WhileParams>!)

// TODO(andydavis): Function to add gradients to a graph.

// TODO(josh11b): Register OpDef, available to all operations added
// to this graph.

// The following two may both benefit from a subgraph-definition API
// that re-uses most of the graph-definition API.
// TODO(andydavis): Add functions to a graph.

// --------------------------------------------------------------------------
// API for driving Graph execution.

// Return a new execution session with the associated graph, or NULL on error.
//
// *graph must be a valid graph (not deleted or nullptr).  This function will
// prevent the graph from being deleted until TF_DeleteSession() is called.
// Does not take ownership of opts.
public func TF_NewSession(_ graph: OpaquePointer!, _ opts: OpaquePointer!, _ status: OpaquePointer!) -> OpaquePointer!

// This function creates a new TF_Session (which is created on success) using
// `session_options`, and then initializes state (restoring tensors and other
// assets) using `run_options`.
//
// Any NULL and non-NULL value combinations for (`run_options, `meta_graph_def`)
// are valid.
//
// - `export_dir` must be set to the path of the exported SavedModel.
// - `tags` must include the set of tags used to identify one MetaGraphDef in
//    the SavedModel.
// - `graph` must be a graph newly allocated with TF_NewGraph().
//
// If successful, populates `graph` with the contents of the Graph and
// `meta_graph_def` with the MetaGraphDef of the loaded model.
public func TF_LoadSessionFromSavedModel(_ session_options: OpaquePointer!, _ run_options: UnsafePointer<TF_Buffer>!, _ export_dir: UnsafePointer<Int8>!, _ tags: UnsafePointer<UnsafePointer<Int8>?>!, _ tags_len: Int32, _ graph: OpaquePointer!, _ meta_graph_def: UnsafeMutablePointer<TF_Buffer>!, _ status: OpaquePointer!) -> OpaquePointer!

// Close a session.
//
// Contacts any other processes associated with the session, if applicable.
// May not be called after TF_DeleteSession().
public func TF_CloseSession(_: OpaquePointer!, _ status: OpaquePointer!)

// Destroy a session object.
//
// Even if error information is recorded in *status, this call discards all
// local resources associated with the session.  The session may not be used
// during or after this call (and the session drops its reference to the
// corresponding graph).
public func TF_DeleteSession(_: OpaquePointer!, _ status: OpaquePointer!)

// Run the graph associated with the session starting with the supplied inputs
// (inputs[0,ninputs-1] with corresponding values in input_values[0,ninputs-1]).
//
// Any NULL and non-NULL value combinations for (`run_options`,
// `run_metadata`) are valid.
//
//    - `run_options` may be NULL, in which case it will be ignored; or
//      non-NULL, in which case it must point to a `TF_Buffer` containing the
//      serialized representation of a `RunOptions` protocol buffer.
//    - `run_metadata` may be NULL, in which case it will be ignored; or
//      non-NULL, in which case it must point to an empty, freshly allocated
//      `TF_Buffer` that may be updated to contain the serialized representation
//      of a `RunMetadata` protocol buffer.
//
// The caller retains ownership of `input_values` (which can be deleted using
// TF_DeleteTensor). The caller also retains ownership of `run_options` and/or
// `run_metadata` (when not NULL) and should manually call TF_DeleteBuffer on
// them.
//
// On success, the tensors corresponding to outputs[0,noutputs-1] are placed in
// output_values[]. Ownership of the elements of output_values[] is transferred
// to the caller, which must eventually call TF_DeleteTensor on them.
//
// On failure, output_values[] contains NULLs.
public func TF_SessionRun(_ session: OpaquePointer!, _ run_options: UnsafePointer<TF_Buffer>!, _ inputs: UnsafePointer<TF_Output>!, _ input_values: UnsafePointer<OpaquePointer?>!, _ ninputs: Int32, _ outputs: UnsafePointer<TF_Output>!, _ output_values: UnsafeMutablePointer<OpaquePointer?>!, _ noutputs: Int32, _ target_opers: UnsafePointer<OpaquePointer?>!, _ ntargets: Int32, _ run_metadata: UnsafeMutablePointer<TF_Buffer>!, _: OpaquePointer!)

// RunOptions

// Input tensors

// Output tensors

// Target operations

// RunMetadata

// Output status

// Set up the graph with the intended feeds (inputs) and fetches (outputs) for a
// sequence of partial run calls.
//
// On success, returns a handle that is used for subsequent PRun calls. The
// handle should be deleted with TF_DeletePRunHandle when it is no longer
// needed.
//
// On failure, out_status contains a tensorflow::Status with an error
// message.
// NOTE: This is EXPERIMENTAL and subject to change.
public func TF_SessionPRunSetup(_: OpaquePointer!, _ inputs: UnsafePointer<TF_Output>!, _ ninputs: Int32, _ outputs: UnsafePointer<TF_Output>!, _ noutputs: Int32, _ target_opers: UnsafePointer<OpaquePointer?>!, _ ntargets: Int32, _ handle: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _: OpaquePointer!)

// Input names

// Output names

// Target operations

// Output handle

// Output status

// Continue to run the graph with additional feeds and fetches. The
// execution state is uniquely identified by the handle.
// NOTE: This is EXPERIMENTAL and subject to change.
public func TF_SessionPRun(_: OpaquePointer!, _ handle: UnsafePointer<Int8>!, _ inputs: UnsafePointer<TF_Output>!, _ input_values: UnsafePointer<OpaquePointer?>!, _ ninputs: Int32, _ outputs: UnsafePointer<TF_Output>!, _ output_values: UnsafeMutablePointer<OpaquePointer?>!, _ noutputs: Int32, _ target_opers: UnsafePointer<OpaquePointer?>!, _ ntargets: Int32, _: OpaquePointer!)

// Input tensors

// Output tensors

// Target operations

// Output status

// Deletes a handle allocated by TF_SessionPRunSetup.
// Once called, no more calls to TF_SessionPRun should be made.
public func TF_DeletePRunHandle(_ handle: UnsafePointer<Int8>!)

// --------------------------------------------------------------------------
// The deprecated session API.  Please switch to the above instead of
// TF_ExtendGraph(). This deprecated API can be removed at any time without
// notice.

public func TF_NewDeprecatedSession(_: OpaquePointer!, _ status: OpaquePointer!) -> OpaquePointer!
public func TF_CloseDeprecatedSession(_: OpaquePointer!, _ status: OpaquePointer!)
public func TF_DeleteDeprecatedSession(_: OpaquePointer!, _ status: OpaquePointer!)
public func TF_Reset(_ opt: OpaquePointer!, _ containers: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ ncontainers: Int32, _ status: OpaquePointer!)
// Treat the bytes proto[0,proto_len-1] as a serialized GraphDef and
// add the nodes in that GraphDef to the graph for the session.
//
// Prefer use of TF_Session and TF_GraphImportGraphDef over this.
public func TF_ExtendGraph(_: OpaquePointer!, _ proto: UnsafeRawPointer!, _ proto_len: Int, _: OpaquePointer!)

// See TF_SessionRun() above.
public func TF_Run(_: OpaquePointer!, _ run_options: UnsafePointer<TF_Buffer>!, _ input_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ inputs: UnsafeMutablePointer<OpaquePointer?>!, _ ninputs: Int32, _ output_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ outputs: UnsafeMutablePointer<OpaquePointer?>!, _ noutputs: Int32, _ target_oper_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ ntargets: Int32, _ run_metadata: UnsafeMutablePointer<TF_Buffer>!, _: OpaquePointer!)

// See TF_SessionPRunSetup() above.
public func TF_PRunSetup(_: OpaquePointer!, _ input_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ ninputs: Int32, _ output_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ noutputs: Int32, _ target_oper_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ ntargets: Int32, _ handle: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _: OpaquePointer!)

// See TF_SessionPRun above.
public func TF_PRun(_: OpaquePointer!, _ handle: UnsafePointer<Int8>!, _ input_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ inputs: UnsafeMutablePointer<OpaquePointer?>!, _ ninputs: Int32, _ output_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ outputs: UnsafeMutablePointer<OpaquePointer?>!, _ noutputs: Int32, _ target_oper_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ ntargets: Int32, _: OpaquePointer!)

// --------------------------------------------------------------------------
// Load plugins containing custom ops and kernels

// TF_Library holds information about dynamically loaded TensorFlow plugins.

// Load the library specified by library_filename and register the ops and
// kernels present in that library.
//
// Pass "library_filename" to a platform-specific mechanism for dynamically
// loading a library. The rules for determining the exact location of the
// library are platform-specific and are not documented here.
//
// On success, place OK in status and return the newly created library handle.
// The caller owns the library handle.
//
// On failure, place an error status in status and return NULL.
public func TF_LoadLibrary(_ library_filename: UnsafePointer<Int8>!, _ status: OpaquePointer!) -> OpaquePointer!

// Get the OpList of OpDefs defined in the library pointed by lib_handle.
//
// Returns a TF_Buffer. The memory pointed to by the result is owned by
// lib_handle. The data in the buffer will be the serialized OpList proto for
// ops defined in the library.
public func TF_GetOpList(_ lib_handle: OpaquePointer!) -> TF_Buffer

// Frees the memory associated with the library handle.
// Does NOT unload the library.
public func TF_DeleteLibraryHandle(_ lib_handle: OpaquePointer!)

// Get the OpList of all OpDefs defined in this address space.
// Returns a TF_Buffer, ownership of which is transferred to the caller
// (and can be freed using TF_DeleteBuffer).
//
// The data in the buffer will be the serialized OpList proto for ops registered
// in this address space.
public func TF_GetAllOpList() -> UnsafeMutablePointer<TF_Buffer>!

/* end extern "C" */

// TENSORFLOW_C_C_API_H_


