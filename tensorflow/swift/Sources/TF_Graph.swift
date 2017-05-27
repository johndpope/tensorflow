
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
// The new graph construction API, still under development.

// Represents a computation graph.  Graphs may be shared between sessions.
// Graphs are thread-safe when used as directed below.

// Return a new graph object.
public func tfNewGraph() -> TF_Graph!{
    return TF_NewGraph()
}

// Destroy an options object.  Graph will be deleted once no more
// TFSession's are referencing it.
public func tfDeleteGraph(_ pointer:TF_Graph!){
    return TF_DeleteGraph(pointer)
}

// Operation being built. The underlying graph must outlive this.

// Operation that has been added to the graph. Valid until the graph is
// deleted -- in particular adding a new operation to the graph does not
// invalidate old TF_Operation* pointers.

// Represents a specific input of an operation.
/*public struct TF_Input {

    public var oper: OpaquePointer!

    public var index: Int32 // The index of the input within oper.

    public init(){
        self = TF_Input()
    }

    public init(oper: OpaquePointer!, index: Int32){
      self = TF_Input(oper,index)
    }
}*/

// Represents a specific output of an operation.
/*public struct TF_Output {

    public var oper: OpaquePointer!

    public var index: Int32 // The index of the output within oper.

    public init(){
        self = TF_Output
    }

    public init(oper: OpaquePointer!, index: Int32){
        self = TF_Output(oper,index)
    }
}*/

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
public func tfGraphSetTensorShape(_ graph: TF_Graph!, _ output: TF_Output, _ dims: UnsafePointer<Int64>!, _ num_dims: Int32, _ status: TF_Status!){
     TF_GraphSetTensorShape(graph,output,dims,num_dims,status)
}

// Returns the number of dimensions of the Tensor referenced by `output`
// in `graph`.
//
// If the number of dimensions in the shape is unknown, returns -1.
//
// Returns an error into `status` if:
//   * `output` is not in `graph`.
public func tfGraphGetTensorNumDims(_ graph: TF_Graph!, _ output: TF_Output, _ status: TF_Status!) -> Int32{
    return TF_GraphGetTensorNumDims(graph,output,status)
}

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
public func tfGraphGetTensorShape(_ graph: TF_Graph!, _ output: TF_Output, _ dims: UnsafeMutablePointer<Int64>!, _ num_dims: Int32, _ status: TF_Status!){
    return TF_GraphGetTensorShape(graph,output,dims,num_dims,status)
}

// Operation will only be added to *graph when TF_FinishOperation() is
// called (assuming TF_FinishOperation() does not return an error).
// *graph must not be deleted until after TF_FinishOperation() is
// called.
public func tfNewOperation(_ graph: TF_Graph!, _ op_type: UnsafePointer<Int8>!, _ oper_name: UnsafePointer<Int8>!) -> OpaquePointer!{
    return TF_NewOperation(graph,op_type,oper_name)
}

// Specify the device for `desc`.  Defaults to empty, meaning unconstrained.
public func tfSetDevice(_ desc: TF_OperationDescription!, _ device: UnsafePointer<Int8>!){
    TF_SetDevice(desc,device)
}

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
public func tfAddInput(_ desc: TF_OperationDescription!, _ input: TF_Output){
    TF_AddInput(desc,input)
}

// For inputs that take a list of tensors.
// inputs must point to TF_Output[num_inputs].
public func tfAddInputList(_ desc: TF_OperationDescription!, _ inputs: UnsafePointer<TF_Output>!, _ num_inputs: Int32){
    TF_AddInputList(desc,inputs,num_inputs)
}

// Call once per control input to `desc`.
public func tfAddControlInput(_ desc: TF_OperationDescription!, _ input: OpaquePointer!){
    TF_AddControlInput(desc,input)
}

// Request that `desc` be co-located on the device where `op`
// is placed.
//
// Use of this is discouraged since the implementation of device placement is
// subject to change. Primarily intended for internal libraries
public func tfColocateWith(_ desc: OpaquePointer!, _ op: OpaquePointer!){
    TF_ColocateWith(desc,op)
}

// Call some TF_SetAttr*() function for every attr that is not
// inferred from an input and doesn't have a default value you wish to
// keep.

// `value` must point to a string of length `length` bytes.
public func tfSetAttrString(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeRawPointer!, _ length: Int){
    TF_SetAttrString(desc,attr_name,value,length)
}
// `values` and `lengths` each must have lengths `num_values`.
// `values[i]` must point to a string of length `lengths[i]` bytes.
public func tfSetAttrStringList(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafePointer<UnsafeRawPointer?>!, _ lengths: UnsafePointer<Int>!, _ num_values: Int32){
    TF_SetAttrStringList(desc,attr_name,values,lengths,num_values)
}
public func tfSetAttrInt(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: Int64){
    TF_SetAttrInt(desc,attr_name,value)
}
public func tfSetAttrIntList(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafePointer<Int64>!, _ num_values: Int32){
    TF_SetAttrIntList(desc,attr_name,values,num_values)
}
public func tfSetAttrFloat(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: Float){
    TF_SetAttrFloat(desc,attr_name,value)
}
public func tfSetAttrFloatList(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafePointer<Float>!, _ num_values: Int32){
    TF_SetAttrFloatList(desc,attr_name,values,num_values)
}
public func tfSetAttrBool(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UInt8){
    TF_SetAttrBool(desc,attr_name,value)
}
public func tfSetAttrBoolList(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafePointer<UInt8>!, _ num_values: Int32){
    TF_SetAttrBoolList(desc,attr_name,values,num_values)
}
public func tfSetAttrType(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: TF_DataType){
    TF_SetAttrType(desc,attr_name,value)
}
public func tfSetAttrTypeList(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafePointer<TF_DataType>!, _ num_values: Int32){
    TF_SetAttrTypeList(desc,attr_name,values,num_values)
}

// Set `num_dims` to -1 to represent "unknown rank".  Otherwise,
// `dims` points to an array of length `num_dims`.  `dims[i]` must be
// >= -1, with -1 meaning "unknown dimension".
public func tfSetAttrShape(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ dims: UnsafePointer<Int64>!, _ num_dims: Int32){
     TF_SetAttrShape(desc,attr_name,dims,num_dims)
}
// `dims` and `num_dims` must point to arrays of length `num_shapes`.
// Set `num_dims[i]` to -1 to represent "unknown rank".  Otherwise,
// `dims[i]` points to an array of length `num_dims[i]`.  `dims[i][j]`
// must be >= -1, with -1 meaning "unknown dimension".
public func tfSetAttrShapeList(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ dims: UnsafePointer<UnsafePointer<Int64>?>!, _ num_dims: UnsafePointer<Int32>!, _ num_shapes: Int32){
    TF_SetAttrShapeList(desc,attr_name,dims,num_dims,num_shapes)
}
// `proto` must point to an array of `proto_len` bytes representing a
// binary-serialized TensorShapeProto.
public func tfSetAttrTensorShapeProto(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ proto: UnsafeRawPointer!, _ proto_len: Int, _ status: TF_Status!){
  TF_SetAttrTensorShapeProto(desc,attr_name,proto,proto_len,status)
  print("status:",status)
}
// `protos` and `proto_lens` must point to arrays of length `num_shapes`.
// `protos[i]` must point to an array of `proto_lens[i]` bytes
// representing a binary-serialized TensorShapeProto.
public func tfSetAttrTensorShapeProtoList(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ protos: UnsafePointer<UnsafeRawPointer?>!, _ proto_lens: UnsafePointer<Int>!, _ num_shapes: Int32, _ status: TF_Status!){
   TF_SetAttrTensorShapeProtoList(desc,attr_name,protos,proto_lens,num_shapes,status)
}

public func tfSetAttrTensor(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: OpaquePointer!, _ status: TF_Status!){
  TF_SetAttrTensor(desc,attr_name,value,status)
}
public func tfSetAttrTensorList(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafePointer<OpaquePointer?>!, _ num_values: Int32, _ status: TF_Status!){
 print("TO IMPLEMENT")
}

// `proto` should point to a sequence of bytes of length `proto_len`
// representing a binary serialization of an AttrValue protocol
// buffer.
public func tfSetAttrValueProto(_ desc: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ proto: UnsafeRawPointer!, _ proto_len: Int, _ status: TF_Status!){
 print("TO IMPLEMENT")
}

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
public func tfFinishOperation(_ desc: OpaquePointer!, _ status: TF_Status!) -> OpaquePointer!{
    return TF_FinishOperation(desc,status)
}

// TF_Operation functions.  Operations are immutable once created, so
// these are all query functions.

public func tfOperationName(_ oper: OpaquePointer!) -> String{
    let str:UnsafePointer<Int8> = TF_OperationName(oper)
    return  String(cString:str)

}
public func tfOperationOpType(_ oper: OpaquePointer!) -> String{
    let str:UnsafePointer<Int8> = TF_OperationOpType(oper)
    return  String(cString:str)

}
public func tfOperationDevice(_ oper: OpaquePointer!) -> String{
    let str:UnsafePointer<Int8> = TF_OperationDevice(oper)
    return  String(cString:str)
}

public func tfOperationNumOutputs(_ oper: OpaquePointer!) -> Int32{
       return  TF_OperationNumOutputs(oper)
}
public func tfOperationOutputType(_ oper_out: TF_Output) -> TF_DataType{
       return  TF_OperationOutputType(oper_out)
}

public func tfOperationOutputListLength(_ oper: OpaquePointer!, _ arg_name: UnsafePointer<Int8>!, _ status: TF_Status!) -> Int32{
       return  TF_OperationOutputListLength(oper,arg_name,status)
}

public func tfOperationNumInputs(_ oper: OpaquePointer!) -> Int32{
       return  TF_OperationNumInputs(oper)
}
public func tfOperationInputType(_ oper_in: TF_Input) -> TF_DataType{
       return  TF_OperationInputType(oper_in)
}
public func tfOperationInputListLength(_ oper: OpaquePointer!, _ arg_name: UnsafePointer<Int8>!, _ status: TF_Status!) -> Int32{
       return  TF_OperationInputListLength(oper,arg_name,status)
}

// In this code:
//   TF_Output producer = TF_OperationInput(consumer);
// There is an edge from producer.oper's output (given by
// producer.index) to consumer.oper's input (given by consumer.index).
public func tfOperationInput(_ oper_in: TF_Input) -> TF_Output{
    return TF_OperationInput(oper_in)
}

// Get the number of current consumers of a specific output of an
// operation.  Note that this number can change when new operations
// are added to the graph.
public func tfOperationOutputNumConsumers(_ oper_out: TF_Output) -> Int32{
    return TF_OperationOutputNumConsumers(oper_out)
}

// Get list of all current consumers of a specific output of an
// operation.  `consumers` must point to an array of length at least
// `max_consumers` (ideally set to
// TF_OperationOutputNumConsumers(oper_out)).  Beware that a concurrent
// modification of the graph can increase the number of consumers of
// an operation.  Returns the number of output consumers (should match
// TF_OperationOutputNumConsumers(oper_out)).
public func tfOperationOutputConsumers(_ oper_out: TF_Output, _ consumers: UnsafeMutablePointer<TF_Input>!, _ max_consumers: Int32) -> Int32{
    return TF_OperationOutputConsumers(oper_out,consumers,max_consumers)
    
}

// Get the number of control inputs to an operation.
public func tfOperationNumControlInputs(_ oper: OpaquePointer!) -> Int32{
    return TF_OperationNumControlInputs(oper)
}

// Get list of all control inputs to an operation.  `control_inputs` must
// point to an array of length `max_control_inputs` (ideally set to
// TF_OperationNumControlInputs(oper)).  Returns the number of control
// inputs (should match TF_OperationNumControlInputs(oper)).
public func tfOperationGetControlInputs(_ oper: OpaquePointer!, _ control_inputs: UnsafeMutablePointer<OpaquePointer?>!, _ max_control_inputs: Int32) -> Int32{
    return TF_OperationGetControlInputs(oper,control_inputs,max_control_inputs)
}

// Get the number of operations that have `*oper` as a control input.
// Note that this number can change when new operations are added to
// the graph.
public func tfOperationNumControlOutputs(_ oper: OpaquePointer!) -> Int32{
    return TF_OperationNumControlOutputs(oper)
}

// Get the list of operations that have `*oper` as a control input.
// `control_outputs` must point to an array of length at least
// `max_control_outputs` (ideally set to
// TF_OperationNumControlOutputs(oper)). Beware that a concurrent
// modification of the graph can increase the number of control
// outputs.  Returns the number of control outputs (should match
// TF_OperationNumControlOutputs(oper)).
public func tfOperationGetControlOutputs(_ oper: OpaquePointer!, _ control_outputs: UnsafeMutablePointer<OpaquePointer?>!, _ max_control_outputs: Int32) -> Int32{
    return TF_OperationGetControlOutputs(oper,control_outputs,max_control_outputs)
}

// TF_AttrType describes the type of the value of an attribute on an operation.
/*public struct TF_AttrType : RawRepresentable, Equatable {

    public init(_ rawValue: UInt32){
        self = TF_AttrType(rawValue)
    }

    public init(rawValue: UInt32){
        self =  TF_AttrType(rawValue)
    }

    public var rawValue: UInt32
}*/

// TF_AttrMetadata describes the value of an attribute on an operation.
/*public struct TF_AttrMetadata {

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

    public init(){
        self = TF_AttrMetadata()
    }

    public init(is_list: UInt8, list_size: Int64, type: TF_AttrType, total_size: Int64){
        self = TF_AttrMetadata.init(is_list,list_size,type,total_size)
    }
}*/

// Returns metadata about the value of the attribute `attr_name` of `oper`.
public func tfOperationGetAttrMetadata(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ status: TF_Status!) -> TF_AttrMetadata{
    return TF_OperationGetAttrMetadata(oper,attr_name,status)
}

// Fills in `value` with the value of the attribute `attr_name`.  `value` must
// point to an array of length at least `max_length` (ideally set to
// TF_AttrMetadata.total_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
public func tfOperationGetAttrString(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeMutableRawPointer!, _ max_length: Int, _ status: TF_Status!){
    
     TF_OperationGetAttrString(oper,attr_name,value,max_length,status)
     print("status:",status)
}

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
public func tfOperationGetAttrStringList(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafeMutablePointer<UnsafeMutableRawPointer?>!, _ lengths: UnsafeMutablePointer<Int>!, _ max_values: Int32, _ storage: UnsafeMutableRawPointer!, _ storage_size: Int, _ status: TF_Status!){
    TF_OperationGetAttrStringList(oper,attr_name,values,lengths,max_values,storage,storage_size,status)
}

public func tfOperationGetAttrInt(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeMutablePointer<Int64>!, _ status: TF_Status!){
    TF_OperationGetAttrInt(oper,attr_name,value,status)
    print("status:",status)
}

// Fills in `values` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `max_values` (ideally set
// TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
public func tfOperationGetAttrIntList(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafeMutablePointer<Int64>!, _ max_values: Int32, _ status: TF_Status!){
 print("TO IMPLEMENT")
}

public func tfOperationGetAttrFloat(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeMutablePointer<Float>!, _ status: TF_Status!){
 print("TO IMPLEMENT")
}

// Fills in `values` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `max_values` (ideally set
// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
public func tfOperationGetAttrFloatList(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafeMutablePointer<Float>!, _ max_values: Int32, _ status: TF_Status!){
 print("TO IMPLEMENT")
}

public func tfOperationGetAttrBool(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeMutablePointer<UInt8>!, _ status: TF_Status!){
 print("TO IMPLEMENT")
}

// Fills in `values` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `max_values` (ideally set
// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
public func tfOperationGetAttrBoolList(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafeMutablePointer<UInt8>!, _ max_values: Int32, _ status: TF_Status!){
 print("TO IMPLEMENT")
}

public func tfOperationGetAttrType(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeMutablePointer<TF_DataType>!, _ status: TF_Status!){
 print("TO IMPLEMENT")
}

// Fills in `values` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `max_values` (ideally set
// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
// attr_name)).
public func tfOperationGetAttrTypeList(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafeMutablePointer<TF_DataType>!, _ max_values: Int32, _ status: TF_Status!){
 print("TO IMPLEMENT")
}

// Fills in `value` with the value of the attribute `attr_name` of `oper`.
// `values` must point to an array of length at least `num_dims` (ideally set to
// TF_Attr_Meta.size from TF_OperationGetAttrMetadata(oper, attr_name)).
public func tfOperationGetAttrShape(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeMutablePointer<Int64>!, _ num_dims: Int32, _ status: TF_Status!){

}

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
public func tfOperationGetAttrShapeList(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ dims: UnsafeMutablePointer<UnsafeMutablePointer<Int64>?>!, _ num_dims: UnsafeMutablePointer<Int32>!, _ num_shapes: Int32, _ storage: UnsafeMutablePointer<Int64>!, _ storage_size: Int32, _ status: TF_Status!){
 print("TO IMPLEMENT")
}

// Sets `value` to the binary-serialized TensorShapeProto of the value of
// `attr_name` attribute of `oper`'.
public func tfOperationGetAttrTensorShapeProto(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeMutablePointer<TF_Buffer>!, _ status: TF_Status!){
 print("TO IMPLEMENT")
}

// Fills in `values` with binary-serialized TensorShapeProto values of the
// attribute `attr_name` of `oper`. `values` must point to an array of length at
// least `num_values` (ideally set to TF_AttrMetadata.list_size from
// TF_OperationGetAttrMetadata(oper, attr_name)).
public func tfOperationGetAttrTensorShapeProtoList(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafeMutablePointer<UnsafeMutablePointer<TF_Buffer>?>!, _ max_values: Int32, _ status: TF_Status!){
 print("TO IMPLEMENT")
}

// Gets the TF_Tensor valued attribute of `attr_name` of `oper`.
//
// Allocates a new TF_Tensor which the caller is expected to take
// ownership of (and can deallocate using TF_DeleteTensor).
public func tfOperationGetAttrTensor(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ value: UnsafeMutablePointer<OpaquePointer?>!, _ status: TF_Status!){
 print("TO IMPLEMENT")
}

// Fills in `values` with the TF_Tensor values of the attribute `attr_name` of
// `oper`. `values` must point to an array of TF_Tensor* of length at least
// `max_values` (ideally set to TF_AttrMetadata.list_size from
// TF_OperationGetAttrMetadata(oper, attr_name)).
//
// The caller takes ownership of all the non-null TF_Tensor* entries in `values`
// (which can be deleted using TF_DeleteTensor(values[i])).
public func tfOperationGetAttrTensorList(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ values: UnsafeMutablePointer<OpaquePointer?>!, _ max_values: Int32, _ status: TF_Status!){
 print("TO IMPLEMENT")
}

// Sets `output_attr_value` to the binary-serialized AttrValue proto
// representation of the value of the `attr_name` attr of `oper`.
public func tfOperationGetAttrValueProto(_ oper: OpaquePointer!, _ attr_name: UnsafePointer<Int8>!, _ output_attr_value: UnsafeMutablePointer<TF_Buffer>!, _ status: TF_Status!){
    print("TO IMPLEMENT") 
}

// Returns the operation in the graph with `oper_name`. Returns nullptr if
// no operation found.
public func tfGraphOperationByName(_ graph: TF_Graph!, _ oper_name: UnsafePointer<Int8>!) -> OpaquePointer!{
    return TF_GraphOperationByName(graph,oper_name)
}

// Iterate through the operations of a graph.  To use:
// size_t pos = 0;
// TF_Operation* oper;
// while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
//   DoSomethingWithOperation(oper);
// }
public func tfGraphNextOperation(_ graph: TF_Graph!, _ pos: UnsafeMutablePointer<Int>!) -> OpaquePointer!{
    return TF_GraphNextOperation(graph,pos)
}

// Write out a serialized representation of `graph` (as a GraphDef protocol
// message) to `output_graph_def` (allocated by TF_NewBuffer()).
// `output_graph_def`'s underlying buffer will be freed when TF_DeleteBuffer()
// is called.
//
// May fail on very large graphs in the future.
public func tfGraphToGraphDef(_ graph: TF_Graph!, _ output_graph_def: UnsafeMutablePointer<TF_Buffer>!, _ status: TF_Status!){
 print("TO IMPLEMENT")
    TF_GraphToGraphDef(graph,output_graph_def,status)
}

// TF_ImportGraphDefOptions holds options that can be passed to
// TF_GraphImportGraphDef.

public func tfNewImportGraphDefOptions() -> OpaquePointer!{
    return TF_NewImportGraphDefOptions()
}
public func tfDeleteImportGraphDefOptions(_ opts: OpaquePointer!){
    TF_DeleteImportGraphDefOptions(opts)
}

// Set the prefix to be prepended to the names of nodes in `graph_def` that will
// be imported into `graph`.
public func tfImportGraphDefOptionsSetPrefix(_ opts: OpaquePointer!, _ prefix: String){
     let cPrefix = UnsafePointer<Int8>(prefix)
    TF_ImportGraphDefOptionsSetPrefix(opts,cPrefix)
}

// Set any imported nodes with input `src_name:src_index` to have that input
// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
// `dst` references a node already existing in the graph being imported into.
public func tfImportGraphDefOptionsAddInputMapping(_ opts: OpaquePointer!, _ src_name: String, _ src_index: Int32, _ dst: TF_Output){
    let cSrcName = UnsafePointer<Int8>(src_name)
    TF_ImportGraphDefOptionsAddInputMapping(opts,cSrcName,src_index,dst)
}

// Set any imported nodes with control input `src_name` to have that input
// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
// `dst` references an operation already existing in the graph being imported
// into.
public func tfGraphImportGraphDefOptionsRemapControlDependency(_ opts: OpaquePointer!, _ src_name: String, _ dst: OpaquePointer!){
    let cSrcName = UnsafePointer<Int8>(src_name)
  //  TF_GraphImportGraphDefOptionsRemapControlDependency(opts, cSrcName, dst)
}

// Cause the imported graph to have a control dependency on `oper`. `oper`
// should exist in the graph being imported into.
public func tfImportGraphDefOptionsAddControlDependency(_ opts: OpaquePointer!, _ oper: OpaquePointer!){
 print("TO IMPLEMENT")
}

// Add an output in `graph_def` to be returned via the `return_outputs` output
// parameter of TF_GraphImportGraphDef(). If the output is remapped via an input
// mapping, the corresponding existing tensor in `graph` will be returned.
public func tfImportGraphDefOptionsAddReturnOutput(_ opts: OpaquePointer!, _ oper_name: UnsafePointer<Int8>!, _ index: Int32){
 print("TO IMPLEMENT")
}

// Returns the number of return outputs added via
// TF_ImportGraphDefOptionsAddReturnOutput().
public func tfImportGraphDefOptionsNumReturnOutputs(_ opts: OpaquePointer!) -> Int32{
    return TF_ImportGraphDefOptionsNumReturnOutputs(opts)
}

// Import the graph serialized in `graph_def` into `graph`.
//
// `num_return_outputs` must be the number of return outputs added (i.e. the
// result of TF_ImportGraphDefOptionsNumReturnOutputs()).  If
// `num_return_outputs` is non-zero, `return_outputs` must be of length
// `num_return_outputs`. Otherwise it can be null.
public func tfGraphImportGraphDefWithReturnOutputs(_ graph: TF_Graph!, _ graph_def: UnsafePointer<TF_Buffer>!, _ options: TF_SessionOptions!, _ return_outputs: UnsafeMutablePointer<TF_Output>!, _ num_return_outputs: Int32, _ status: TF_Status!){
 print("TO IMPLEMENT")
}

// Import the graph serialized in `graph_def` into `graph`.
// Convenience function for when no return outputs have been added.
public func tfGraphImportGraphDef(_ graph: TF_Graph!, _ graph_def: UnsafePointer<TF_Buffer>!, _ options: TF_SessionOptions!, _ status: TF_Status!){
 print("TO IMPLEMENT")
}

// Note: The following function may fail on very large protos in the future.

public func tfOperationToNodeDef(_ oper: OpaquePointer!, _ output_node_def: UnsafeMutablePointer<TF_Buffer>!, _ status: TF_Status!){
 print("TO IMPLEMENT")
}


