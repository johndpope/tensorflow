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
 
 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/operation.go
 */
import CTensorFlow
import Foundation

// Operation that has been added to the graph.
struct GoOperation  {
    var c:TF_Operation
    // A reference to the Graph to prevent it from
    // being GCed while the Operation is still alive.
    var g:Graph
    public init(_ c:TF_Operation,_ g:Graph){
        self.c = c
        self.g = g
    }
}

// Name returns the name of the operation.
extension GoOperation{
    func Name()-> String  {
        return  tf.OperationName(self.c)
    }
    // Type returns the name of the operator used by this operation.
    func Type()-> String {
        return  tf.OperationOpType(self.c)
    }
    // NumOutputs returns the number of outputs of op.
    func NumOutputs() -> Int32 {
        return  tf.OperationNumOutputs(self.c)
    }
    // OutputListSize returns the size of the list of Outputs that is produced by a
    // named output of op.
    //
    // An Operation has multiple named outputs, each of which produces either
    // a single tensor or a list of tensors. This method returns the size of
    // the list of tensors for a specific output of the operation, identified
    // by its name.
    func  OutputListSize(output:String)-> (Int32, NSError?) {
        let cname = output.cString(using: .utf8)
        defer{
            //            free(cname)
        }
        
        let status = newStatus()
        let n =  tf.OperationOutputListLength(self.c, cname, status.c)
        return (n, status.error())
    }
    
    
    
}

// Output represents one of the outputs of an operation in the graph. Has a
// DataType (and eventually a Shape).  May be passed as an input argument to a
// function for adding operations to a graph, or to a Session's Run() method to
// fetch that output as a tensor.
struct Output  {
    
    init( Op:GoOperation, Index:Int) {
        self.Op = Op
        self.Index = Index
    }
    
    // Op is the Operation that produces this Output.
    var Op :GoOperation
    
    // Index specifies the index of the output within the Operation.
    var Index:Int
    
    // DataType returns the type of elements in the tensor produced by p.
    func DataType()-> TF_DataType {
        
        return   TF_OperationOutputType(self.c())
    }
    
    // Shape returns the (possibly incomplete) shape of the tensor produced p.*/
    
    func  Shape()-> Shape {
        let status = newStatus()
        let port = self.c()
        let ndims = tf.GraphGetTensorNumDims(self.Op.g.c, port, status.c)
        if let err = status.error() {
            // This should not be possible since an error only occurs if
            // the operation does not belong to the graph.  It should not
            // be possible to construct such an Operation object.
            return Shape()
        }
        if ndims < 0 {
            return Shape()
        }
        if ndims == 0 {
            return ScalarShape()
        }
        var dims:[Int64] = []
        tf.GraphGetTensorShape(self.Op.g.c, port, &dims[0], ndims, status.c)
        if let err = status.error() {
            // Same as above, should not be possible.
            return Shape()
        }
        var ret = Shape()
        //    for dim in ndims{
        //
        //    }
        //    for i = 0; i < int(ndims); i++ {
        //        ret.dims[i] = int64(dims[i])
        //    }
        return ret
    }
    
    func c() -> TF_Output {
        return TF_Output(oper: self.Op.c, index: CInt(self.Index))
    }
    
    func canBeAnInput() {}
}

extension GoOperation{
    // Output returns the i-th output of op.
    func output(_ i:Int)-> Output {
        return Output(Op: self, Index: i)
    }
}
/*
 // Input is the interface for specifying inputs to an operation being added to
 // a Graph.
 //
 // Operations can have multiple inputs, each of which could be either a tensor
 // produced by another operation (an Output object), or a list of tensors
 // produced by other operations (an OutputList). Thus, this interface is
 // implemented by both Output and OutputList.
 //
 // See OpSpec.Input for more information.
 type Input interface {
 // Unexported to preclude implementations outside this package.
 canBeAnInput()
 }
 
 // OutputList represents a list of Outputs that can be provided as input to
 // another operation.
 type OutputList []Output
 
 func (l OutputList) canBeAnInput() {}
 */
