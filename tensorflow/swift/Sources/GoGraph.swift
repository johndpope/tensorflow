/*
 Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 
 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/graph.go
 */
import CTensorFlow
import Foundation
import IOSwift
import func Darwin.C.stdlib.malloc
import func Darwin.C.stdlib.posix_memalign
import func Darwin.C.stdlib.free
import func Darwin.C.string.memset
import func Darwin.C.string.memcpy
import func Darwin.malloc.malloc_size
import protoTensorFlow

typealias Byte = UInt8

// Graph represents a computation graph. Graphs may be shared between sessions.
class Graph  {
    var c:TF_Graph!
    
    // TODO work out how to use  runtime.SetFinalizer(g, (*Graph).finalizer) on struct in swift
    deinit {
        finalizer(g:self)
    }
}

func newGraph()-> Graph{
    let graph:OpaquePointer = tfNewGraph()
    let g = Graph()
    g.c = graph
    return g
}
/*
// NewGraph returns a new Graph.
func NewGraph() *Graph {
    g = &Graph{TF_NewGraph()}
    runtime.SetFinalizer(g, (*Graph).finalizer)
    return g
}*/


func  finalizer(g :Graph) {
    tfDeleteGraph(g.c)
}

// WriteTo writes out a serialized representation of g to w.
//
// Implements the io.WriterTo interface.
func  writeTo(g:Graph, w:Writer)-> (Int, NSError?) {

    if let buffer =  tfNewBuffer(){
        var status = newStatus()
        
        defer {
            TF_DeleteStatus(status.c)
            TF_DeleteBuffer(buffer)
        }
        
        tfGraphToGraphDef(g.c, buffer, status.c)

        if let msg = status.errorMessage(){
            return (0, NSError.newIoError(msg, code: 111))
        }
        
        if buffer.pointee.length > (1 << 30) {
            // For very large graphs, the writes can be chunked.
            // Punt on that for now.
            return (0, NSError.newIoError("Graph is too large to write out, Graph.WriteTo needs to be updated", code: 111))
        }
        // A []byte slice backed by C memory.
        // See: https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices

        return w.write(data: buffer.pointee.data.load(as: NSData.self))
    }else{
        return (0,NSError.newIoError("couldn't access buffer", code: 111))
    }
    
}


// Import imports the nodes and edges from a serialized representation of
// another Graph into g.
//
// Names of imported nodes will be prefixed with prefix.

func importGraph(g:Graph,def: [Byte], prefix:String)-> NSError? {
    let cprefix = prefix.cString(using: .utf8)
    
//    defer{
//        free(cprefix)
//    }
    
    let opts = TF_NewImportGraphDefOptions()
    
    defer {
        TF_DeleteImportGraphDefOptions(opts)
        TF_ImportGraphDefOptionsSetPrefix(opts, cprefix)
    }
    
    
    if let buffer = tfNewBuffer(){
        
        defer{
            TF_DeleteBuffer(buffer)
        }
        // Would have preferred to use C.CBytes, but that does not play well
        // with "go vet" till https://github.com/golang/go/issues/17201 is
        // resolved.
        buffer.pointee.length = size_t(def.count)
//        buffer.pointee.data =  malloc(4)
        if buffer.pointee.data == nil {
            return NSError.newIoError("unable to allocate memory", code: 123)
        }
        defer {
            free(buffer)
        }
//        memcpy(buffer.pointee.data, &def[0], buffer.pointee.length)
//        C.memcpy(buf.data, unsafe.Pointer(&def[0]), buf.length)
        
        let status = newStatus()
        tfGraphImportGraphDef(g.c, buffer, opts, status.c)
        if let error = status.error() {
            return error
        }
    }
    
    return NSError.newIoError("couldn't allocate buffer", code: 123)
    

    
}

// Operation returns the Operation named name in the Graph, or nil if no such
// operation is present.
/*func (g *Graph) Operation(name string) *Operation {
    cname = C.CString(name)
    defer C.free(unsafe.Pointer(cname))
    cop = TF_GraphOperationByName(g.c, cname)
    if cop == nil {
        return nil
    }
    return &Operation{cop, g}
}*/

func operation(g:Graph,name:String) -> GoOperation?{
    let cname = name.cString(using: .utf8)
    defer{
//        free(cname)
    }
    let cOperation = tfGraphOperationByName(g.c, cname)
    if cOperation == nil{
        return nil
    }
    return GoOperation.init(c:cOperation!,g:g)
    
}

// OpSpec is the specification of an Operation to be added to a Graph
// (using Graph.AddOperation).
struct OpSpec  {
    // Type of the operation (e.g., "Add", "MatMul").
    var OpType:String
    
    // Name by which the added operation will be referred to in the Graph.
    // If omitted, defaults to Type.
    var Name:String
    
    // Inputs to this operation, which in turn must be outputs
    // of other operations already added to the Graph.
    //
    // An operation may have multiple inputs with individual inputs being
    // either a single tensor produced by another operation or a list of
    // tensors produced by multiple operations. For example, the "Concat"
    // operation takes two inputs: (1) the dimension along which to
    // concatenate and (2) a list of tensors to concatenate. Thus, for
    // Concat, len(Input) must be 2, with the first element being an Output
    // and the second being an OutputList.
    var Input:[TF_Input]
    
    // Map from attribute name to its value that will be attached to this
    // operation.
    var Attrs: Dictionary<String,Tensorflow_AttrValue> = [:]
    
    // Other possible fields: Device, ColocateWith, ControlInputs.
}

// AddOperation adds an operation to g.
func addOperation (g: Graph,  args:OpSpec)-> (GoOperation?, NSError?) {

    let cOperationDesc = tfNewOperation(g.c, args.OpType, args.Name)

    for input in  args.Input {
        
//        switch input. = in.(type) {
//            case Output:
//            TF_AddInput(cOperationDesc, in.c())
//            case OutputList:
//            size = len(in)
//            list = make([]TF_Output, size)
//            for i, v = range in {
//            list[i] = v.c()
//            }
//            if size > 0 {
//            TF_AddInputList(cOperationDesc, &list[0], C.int(size))
//            } else {
//            TF_AddInputList(cOperationDesc, nil, 0)
//            }
//        }
    }
    var status = newStatus()
    for (name, value) in args.Attrs {
        
        if let err = setAttr(cOperationDesc, status.c, name, value) {
            // Memory leak here as the TF_OperationDescription
            // object will not be cleaned up. At the time of this
            // writing, this was next to impossible since it
            // required value to be a string tensor with
            // incorrectly encoded strings. Given this rarity, live
            // with the memory leak.  If it becomes a real problem,
            // consider adding a TF_DeleteOperationDescription
            // function to the C API.
            return (nil, NSError.newIoError(" (memory will be leaked)", code: 444))
        }
    }
    var op = GoOperation(
        c: TF_FinishOperation(cOperationDesc, status.c),
        g: g
    )
    return (op, status.error())
}

/*
 
 TODO - review Tensorflow_AttrValue in proto library to simplify this mess.
 
func setAttr(_ cDesc:TF_OperationDescription?,_ status:TF_Status,_ name:String,_ value:Tensorflow_AttrValue) ->  NSError? {
    
    switch value.type {
        case .dtBfloat16:
        print("test")
        
        default:
        print("default")
    }
    
    switch value = value.(type) {
        case string:
        cstr = C.CString(value)
        TF_SetAttrString(cDesc, cAttrName, unsafe.Pointer(cstr), C.size_t(len(value)))
        C.free(unsafe.Pointer(cstr))
        case []string:
        size = len(value)
        list = make([]unsafe.Pointer, size)
        lens = make([]C.size_t, size)
        for i, s = range value {
            list[i] = unsafe.Pointer(C.CString(s))
            lens[i] = C.size_t(len(s))
        }
        if size > 0 {
            TF_SetAttrStringList(cdesc, cAttrName, &list[0], &lens[0], C.int(size))
        } else {
            TF_SetAttrStringList(cdesc, cAttrName, nil, nil, 0)
        }
        for _, s = range list {
            C.free(s)
        }
        case int64:
        TF_SetAttrInt(cdesc, cAttrName, C.int64_t(value))
        case []int64:
        size = len(value)
        list = make([]C.int64_t, size)
        for i, v = range value {
            list[i] = C.int64_t(v)
        }
        if size > 0 {
            TF_SetAttrIntList(cdesc, cAttrName, &list[0], C.int(size))
        } else {
            TF_SetAttrIntList(cdesc, cAttrName, nil, 0)
        }
        case float32:
        TF_SetAttrFloat(cdesc, cAttrName, C.float(value))
        case []float32:
        size = len(value)
        list = make([]C.float, size)
        for i, v = range value {
            list[i] = C.float(v)
        }
        if size > 0 {
            TF_SetAttrFloatList(cdesc, cAttrName, &list[0], C.int(size))
        } else {
            TF_SetAttrFloatList(cdesc, cAttrName, nil, 0)
        }
        case bool:
        v = C.uchar(0)
        if value {
        v = 1
        }
        TF_SetAttrBool(cdesc, cAttrName, v)
        case []bool:
        size = len(value)
        list = make([]C.uchar, size)
        for i, v = range value {
            if v {
                list[i] = 1
            }
        }
        if size > 0 {
            TF_SetAttrBoolList(cdesc, cAttrName, &list[0], C.int(size))
        } else {
            TF_SetAttrBoolList(cdesc, cAttrName, nil, 0)
        }
        case DataType:
        TF_SetAttrType(cdesc, cAttrName, TF_DataType(value))
        case []DataType:
        var list *TF_DataType
        if len(value) > 0 {
            list = (*TF_DataType)(&value[0])
        }
        TF_SetAttrTypeList(cdesc, cAttrName, list, C.int(len(value)))
        case *Tensor:
        TF_SetAttrTensor(cdesc, cAttrName, value.c, status.c)
        if err = status.Err(); err != nil {
            return fmt.Errorf("bad value for attribute %q: %v", name, err)
        }
        case []*Tensor:
        size = len(value)
        list = make([]*TF_Tensor, size)
        for i, v = range value {
            list[i] = v.c
        }
        var plist **TF_Tensor
        if size > 0 {
            plist = &list[0]
        }
        TF_SetAttrTensorList(cdesc, cAttrName, plist, C.int(size), status.c)
        if err = status.Err(); err != nil {
            return fmt.Errorf("bad value for attribute %q: %v", name, err)
        }
        case Shape:
        ndims, dims = cshape(value)
        var dimsp *C.int64_t
        if ndims > 0 {
            dimsp = &dims[0]
        }
        TF_SetAttrShape(cdesc, cAttrName, dimsp, ndims)
        case []Shape:
        ndims = make([]C.int, len(value))
        dims = make([][]C.int64_t, len(value))
        dimsp = make([]*C.int64_t, len(value))
        for i, s = range value {
            ndims[i], dims[i] = cshape(s)
            if ndims[i] > 0 {
                dimsp[i] = &dims[i][0]
            }
        }
        if len(value) > 0 {
            TF_SetAttrShapeList(cdesc, cAttrName, &dimsp[0], &ndims[0], C.int(len(value)))
        } else {
            TF_SetAttrShapeList(cdesc, cAttrName, nil, nil, 0)
        }
        default:
        return fmt.Errorf("attribute %q has a type (%T) which is not valid for operation attributes", name, value)
    }
    return nil
}

func cshape(s Shape) (C.int, []C.int64_t) {
    ndims = C.int(s.NumDimensions())
    if ndims < 0 {
        return -1, nil
    }
    dims = make([]C.int64_t, ndims)
    for i, s = range s.dims {
        dims[i] = C.int64_t(s)
    }
    return ndims, dims
}
*/
