import Foundation

/*
 protoTensorFlow.Tensorflow_OpDef:
 name: "Add"
 input_arg {
 name: "x"
 type_attr: "T"
 }
 input_arg {
 name: "y"
 type_attr: "T"
 }
 output_arg {
 name: "z"
 type_attr: "T"
 }
 attr {
 name: "T"
 type: "type"
 allowed_values {
 list {
 type: [DT_HALF, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128, DT_STRING]
 }
 }
 }
 summary: "Returns x + y element-wise."
 description: "*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting\n[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)"
 */


protocol Numeric1 { }

extension Int32: Numeric1 {} /// DT_HALF
extension Float: Numeric1 {} /// DT_FLOAT
extension Double: Numeric1 {} //DT_DOUBLE
extension Int32: Numeric1 {} /// DT_INT32, DT_INT16, DT_INT8, DT_UINT8.
extension Data: Numeric1 {} /// DT_STRING
extension Float: Numeric1 {} /// DT_COMPLEX64
extension Int64: Numeric1 {} /// DT_INT64
extension Bool: Numeric1 {} /// DT_BOOL
extension Double: Numeric1 {} /// DT_COMPLEX128
extension Tensorflow_ResourceHandle: Numeric1 {} //DT_RESOURCE




func add( scope:Scope,x: Numeric1, y: Numeric1 )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }
    
    var attrs:Dictionary<String,Any> = [:]
    attrs["T"] = t
    
    let opspec = OpSpec(
        OpType: "Add",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}


/*
 // Returns x + y element-wise.
 //
 // *NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
 // [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
 func Add(scope *Scope, x tf.Output, y tf.Output) (z tf.Output) {
	if scope.Err() != nil {
 return
	}
	opspec := tf.OpSpec{
 Type: "Add",
 Input: []tf.Input{
 x, y,
 },
	}
	op := scope.AddOperation(opspec)
	return op.Output(0)
 }
 */
