// This file is generated automatically, DO NOT EDIT
//  tensorflow/core/framework/op_def.proto
//  tensorflow/core/framework/op_def.pb.swift

import Foundation
import Darwin.C.stddef
import Darwin.C.stdint
import CTensorFlow
import protoTensorFlow

/*
var _inputArg: [Tensorflow_OpDef.ArgDef] = []
var _outputArg: [Tensorflow_OpDef.ArgDef] = []
var _attr: [Tensorflow_OpDef.AttrDef] = []
var _deprecation: Tensorflow_OpDeprecation? = nil
var _summary: String = String()
var _description_p: String = String()
var _isCommutative: Bool = false
var _isAggregate: Bool = false
var _isStateful: Bool = false
var _allowsUninitializedInput: Bool = false
*/


/*
Raise a exception to abort the process when called.

If exit_without_error is true, the process will exit normally,
otherwise it will exit with a SIGABORT signal.
Returns nothing but an exception.

_inputArg
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Abort(scope:Scope, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Abort",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Computes the absolute value of a tensor.

Given a tensor `x`, this operation returns a tensor containing the absolute
value of each element in `x`. For example, if x is an input element and y is
an output element, this operation computes \\(y = |x|\\).

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Abs(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Abs",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Applies a gradient to a given accumulator.

Does not add if local_step is lesser than the accumulator's global_step.

_inputArg
 handle:: .  local_step:: .  gradient:: .dtype 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AccumulatorApplyGradient(scope:Scope,  handle:tf.Output,  local_step:tf.Output,  gradient:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AccumulatorApplyGradient",
        Input: [  handle,  local_step,  gradient, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Returns the number of gradients aggregated in the given accumulators.


_inputArg
 handle:: . 
_outputArg

_outputArg
 num_accumulated:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AccumulatorNumAccumulated(scope:Scope,  handle:tf.Output, ) -> ( num_accumulated:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AccumulatorNumAccumulated",
        Input: [  handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) num_accumulated )

}

/*
Updates the accumulator with a new value for global_step.

Logs warning if the accumulator's value is already higher than
new_global_step.

_inputArg
 handle:: .  new_global_step:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AccumulatorSetGlobalStep(scope:Scope,  handle:tf.Output,  new_global_step:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AccumulatorSetGlobalStep",
        Input: [  handle,  new_global_step, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Extracts the average gradient in the given ConditionalAccumulator.

The op blocks until sufficient (i.e., more than num_required)
gradients have been accumulated.  If the accumulator has already
aggregated more than num_required gradients, it returns the average of
the accumulated gradients.  Also automatically increments the recorded
global_step in the accumulator by 1, and resets the aggregate to 0.

_inputArg
 handle:: .  num_required:: . 
_outputArg

_outputArg
 average::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AccumulatorTakeGradient(scope:Scope,  handle:tf.Output,  num_required:tf.Output, ) -> ( average:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AccumulatorTakeGradient",
        Input: [  handle,  num_required, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) average )

}

/*
Computes acos of x element-wise.


_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Acos(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Acos",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Returns x + y element-wise.

*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Add(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Add",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Add an `N`-minibatch `SparseTensor` to a `SparseTensorsMap`, return `N` handles.

A `SparseTensor` of rank `R` is represented by three tensors: `sparse_indices`,
`sparse_values`, and `sparse_shape`, where
```sparse_indices.shape[1] == sparse_shape.shape[0] == R```
An `N`-minibatch of `SparseTensor` objects is represented as a `SparseTensor`
having a first `sparse_indices` column taking values between `[0, N)`, where
the minibatch size `N == sparse_shape[0]`.
The input `SparseTensor` must have rank `R` greater than 1, and the first
dimension is treated as the minibatch dimension.  Elements of the `SparseTensor`
must be sorted in increasing order of this first dimension.  The stored
`SparseTensor` objects pointed to by each row of the output `sparse_handles`
will have rank `R-1`.
The `SparseTensor` values can then be read out as part of a minibatch by passing
the given keys as vector elements to `TakeManySparseFromTensorsMap`.  To ensure
the correct `SparseTensorsMap` is accessed, ensure that the same
`container` and `shared_name` are passed to that Op.  If no `shared_name`
is provided here, instead use the *name* of the Operation created by calling
`AddManySparseToTensorsMap` as the `shared_name` passed to
`TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.

_inputArg
 sparse_indices:: .  sparse_values:: .T  sparse_shape:: . 
_outputArg

_outputArg
 sparse_handles:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AddManySparseToTensorsMap(scope:Scope,  sparse_indices:tf.Output,  sparse_values:tf.Output,  sparse_shape:tf.Output, ) -> ( sparse_handles:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AddManySparseToTensorsMap",
        Input: [  sparse_indices,  sparse_values,  sparse_shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) sparse_handles )

}

/*
Add all input tensors element wise.


_inputArg
 inputs:: .T 
_outputArg

_outputArg
 sum::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AddN(scope:Scope,  inputs:tf.Output, ) -> ( sum:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AddN",
        Input: [  inputs, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) sum )

}

/*
Add a `SparseTensor` to a `SparseTensorsMap` return its handle.

A `SparseTensor` is represented by three tensors: `sparse_indices`,
`sparse_values`, and `sparse_shape`.
This operator takes the given `SparseTensor` and adds it to a container
object (a `SparseTensorsMap`).  A unique key within this container is generated
in the form of an `int64`, and this is the value that is returned.
The `SparseTensor` can then be read out as part of a minibatch by passing
the key as a vector element to `TakeManySparseFromTensorsMap`.  To ensure
the correct `SparseTensorsMap` is accessed, ensure that the same
`container` and `shared_name` are passed to that Op.  If no `shared_name`
is provided here, instead use the *name* of the Operation created by calling
`AddSparseToTensorsMap` as the `shared_name` passed to
`TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.

_inputArg
 sparse_indices:: .  sparse_values:: .T  sparse_shape:: . 
_outputArg

_outputArg
 sparse_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AddSparseToTensorsMap(scope:Scope,  sparse_indices:tf.Output,  sparse_values:tf.Output,  sparse_shape:tf.Output, ) -> ( sparse_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AddSparseToTensorsMap",
        Input: [  sparse_indices,  sparse_values,  sparse_shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) sparse_handle )

}

/*
Deprecated. Disallowed in GraphDef version >= 2.


_inputArg
 images:: .T  contrast_factor:: .  min_value:: .  max_value:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AdjustContrast(scope:Scope,  images:tf.Output,  contrast_factor:tf.Output,  min_value:tf.Output,  max_value:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AdjustContrast",
        Input: [  images,  contrast_factor,  min_value,  max_value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Adjust the contrast of one or more images.

`images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
interpreted as `[height, width, channels]`.  The other dimensions only
represent a collection of images, such as `[batch, height, width, channels].`
Contrast is adjusted independently for each channel of each image.
For each channel, the Op first computes the mean of the image pixels in the
channel and then adjusts each component of each pixel to
`(x - mean) * contrast_factor + mean`.

_inputArg
 images:: .  contrast_factor:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AdjustContrastv2(scope:Scope,  images:tf.Output,  contrast_factor:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AdjustContrastv2",
        Input: [  images,  contrast_factor, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Adjust the hue of one or more images.

`images` is a tensor of at least 3 dimensions.  The last dimension is
interpretted as channels, and must be three.
The input image is considered in the RGB colorspace. Conceptually, the RGB
colors are first mapped into HSV. A delta is then applied all the hue values,
and then remapped back to RGB colorspace.

_inputArg
 images:: .  delta:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AdjustHue(scope:Scope,  images:tf.Output,  delta:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AdjustHue",
        Input: [  images,  delta, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Adjust the saturation of one or more images.

`images` is a tensor of at least 3 dimensions.  The last dimension is
interpretted as channels, and must be three.
The input image is considered in the RGB colorspace. Conceptually, the RGB
colors are first mapped into HSV. A scale is then applied all the saturation
values, and then remapped back to RGB colorspace.

_inputArg
 images:: .  scale:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AdjustSaturation(scope:Scope,  images:tf.Output,  scale:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AdjustSaturation",
        Input: [  images,  scale, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the "logical and" of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

_inputArg
 input:: .  reduction_indices:: .Tidx 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func All(scope:Scope,  input:tf.Output,  reduction_indices:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "All",
        Input: [  input,  reduction_indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Generates labels for candidate sampling with a learned unigram distribution.

See explanations of candidate sampling and the data formats at
go/candidate-sampling.
For each batch, this op picks a single set of sampled candidate labels.
The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

_inputArg
 true_classes:: . 
_outputArg

_outputArg
 sampled_candidates::  true_expected_count::  sampled_expected_count:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AllCandidateSampler(scope:Scope,  true_classes:tf.Output, ) -> ( sampled_candidates:tf.Output,  true_expected_count:tf.Output,  sampled_expected_count:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AllCandidateSampler",
        Input: [  true_classes, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) sampled_candidates  op.Output(*) true_expected_count  op.Output(*) sampled_expected_count )

}

/*
Computes the "logical or" of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

_inputArg
 input:: .  reduction_indices:: .Tidx 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Any(scope:Scope,  input:tf.Output,  reduction_indices:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Any",
        Input: [  input,  reduction_indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Update '*var' according to the adadelta scheme.

accum = rho() * accum + (1 - rho()) * grad.square();
update = (update_accum + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad;
update_accum = rho() * update_accum + (1 - rho()) * update.square();
var -= update;

_inputArg
 var:: .T  accum:: .T  accum_update:: .T  lr:: .T  rho:: .T  epsilon:: .T  grad:: .T 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ApplyAdadelta(scope:Scope,  var:tf.Output,  accum:tf.Output,  accum_update:tf.Output,  lr:tf.Output,  rho:tf.Output,  epsilon:tf.Output,  grad:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ApplyAdadelta",
        Input: [  var,  accum,  accum_update,  lr,  rho,  epsilon,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Update '*var' according to the adagrad scheme.

accum += grad * grad
var -= lr * grad * (1 / sqrt(accum))

_inputArg
 var:: .T  accum:: .T  lr:: .T  grad:: .T 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ApplyAdagrad(scope:Scope,  var:tf.Output,  accum:tf.Output,  lr:tf.Output,  grad:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ApplyAdagrad",
        Input: [  var,  accum,  lr,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Update '*var' according to the proximal adagrad scheme.


_inputArg
 var:: .T  gradient_accumulator:: .T  gradient_squared_accumulator:: .T  grad:: .T  lr:: .T  l1:: .T  l2:: .T  global_step:: . 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ApplyAdagradDA(scope:Scope,  var:tf.Output,  gradient_accumulator:tf.Output,  gradient_squared_accumulator:tf.Output,  grad:tf.Output,  lr:tf.Output,  l1:tf.Output,  l2:tf.Output,  global_step:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ApplyAdagradDA",
        Input: [  var,  gradient_accumulator,  gradient_squared_accumulator,  grad,  lr,  l1,  l2,  global_step, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Update '*var' according to the Adam algorithm.

lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)

_inputArg
 var:: .T  m:: .T  v:: .T  beta1_power:: .T  beta2_power:: .T  lr:: .T  beta1:: .T  beta2:: .T  epsilon:: .T  grad:: .T 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ApplyAdam(scope:Scope,  var:tf.Output,  m:tf.Output,  v:tf.Output,  beta1_power:tf.Output,  beta2_power:tf.Output,  lr:tf.Output,  beta1:tf.Output,  beta2:tf.Output,  epsilon:tf.Output,  grad:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ApplyAdam",
        Input: [  var,  m,  v,  beta1_power,  beta2_power,  lr,  beta1,  beta2,  epsilon,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Update '*var' according to the centered RMSProp algorithm.

The centered RMSProp algorithm uses an estimate of the centered second moment
(i.e., the variance) for normalization, as opposed to regular RMSProp, which
uses the (uncentered) second moment. This often helps with training, but is
slightly more expensive in terms of computation and memory.
Note that in dense implementation of this algorithm, mg, ms, and mom will
update even if the grad is zero, but in this sparse implementation, mg, ms,
and mom will not update in iterations during which the grad is zero.
mean_square = decay * mean_square + (1-decay) * gradient ** 2
mean_grad = decay * mean_grad + (1-decay) * gradient
Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)
mg <- rho * mg_{t-1} + (1-rho) * grad
ms <- rho * ms_{t-1} + (1-rho) * grad * grad
mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms - mg * mg + epsilon)
var <- var - mom

_inputArg
 var:: .T  mg:: .T  ms:: .T  mom:: .T  lr:: .T  rho:: .T  momentum:: .T  epsilon:: .T  grad:: .T 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ApplyCenteredRMSProp(scope:Scope,  var:tf.Output,  mg:tf.Output,  ms:tf.Output,  mom:tf.Output,  lr:tf.Output,  rho:tf.Output,  momentum:tf.Output,  epsilon:tf.Output,  grad:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ApplyCenteredRMSProp",
        Input: [  var,  mg,  ms,  mom,  lr,  rho,  momentum,  epsilon,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Update '*var' according to the Ftrl-proximal scheme.

accum_new = accum + grad * grad
linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new

_inputArg
 var:: .T  accum:: .T  linear:: .T  grad:: .T  lr:: .T  l1:: .T  l2:: .T  lr_power:: .T 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ApplyFtrl(scope:Scope,  var:tf.Output,  accum:tf.Output,  linear:tf.Output,  grad:tf.Output,  lr:tf.Output,  l1:tf.Output,  l2:tf.Output,  lr_power:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ApplyFtrl",
        Input: [  var,  accum,  linear,  grad,  lr,  l1,  l2,  lr_power, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Update '*var' by subtracting 'alpha' * 'delta' from it.


_inputArg
 var:: .T  alpha:: .T  delta:: .T 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ApplyGradientDescent(scope:Scope,  var:tf.Output,  alpha:tf.Output,  delta:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ApplyGradientDescent",
        Input: [  var,  alpha,  delta, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Update '*var' according to the momentum scheme. Set use_nesterov = True if you

want to use Nesterov momentum.
accum = accum * momentum + grad
var -= lr * accum

_inputArg
 var:: .T  accum:: .T  lr:: .T  grad:: .T  momentum:: .T 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ApplyMomentum(scope:Scope,  var:tf.Output,  accum:tf.Output,  lr:tf.Output,  grad:tf.Output,  momentum:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ApplyMomentum",
        Input: [  var,  accum,  lr,  grad,  momentum, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.

accum += grad * grad
prox_v = var - lr * grad * (1 / sqrt(accum))
var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}

_inputArg
 var:: .T  accum:: .T  lr:: .T  l1:: .T  l2:: .T  grad:: .T 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ApplyProximalAdagrad(scope:Scope,  var:tf.Output,  accum:tf.Output,  lr:tf.Output,  l1:tf.Output,  l2:tf.Output,  grad:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ApplyProximalAdagrad",
        Input: [  var,  accum,  lr,  l1,  l2,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Update '*var' as FOBOS algorithm with fixed learning rate.

prox_v = var - alpha * delta
var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

_inputArg
 var:: .T  alpha:: .T  l1:: .T  l2:: .T  delta:: .T 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ApplyProximalGradientDescent(scope:Scope,  var:tf.Output,  alpha:tf.Output,  l1:tf.Output,  l2:tf.Output,  delta:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ApplyProximalGradientDescent",
        Input: [  var,  alpha,  l1,  l2,  delta, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Update '*var' according to the RMSProp algorithm.

Note that in dense implementation of this algorithm, ms and mom will
update even if the grad is zero, but in this sparse implementation, ms
and mom will not update in iterations during which the grad is zero.
mean_square = decay * mean_square + (1-decay) * gradient ** 2
Delta = learning_rate * gradient / sqrt(mean_square + epsilon)
ms <- rho * ms_{t-1} + (1-rho) * grad * grad
mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
var <- var - mom

_inputArg
 var:: .T  ms:: .T  mom:: .T  lr:: .T  rho:: .T  momentum:: .T  epsilon:: .T  grad:: .T 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ApplyRMSProp(scope:Scope,  var:tf.Output,  ms:tf.Output,  mom:tf.Output,  lr:tf.Output,  rho:tf.Output,  momentum:tf.Output,  epsilon:tf.Output,  grad:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ApplyRMSProp",
        Input: [  var,  ms,  mom,  lr,  rho,  momentum,  epsilon,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Returns the truth value of abs(x-y) < tolerance element-wise.


_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ApproximateEqual(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ApproximateEqual",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Returns the index with the largest value across dimensions of a tensor.

Note that in case of ties the identity of the return value is not guaranteed.

_inputArg
 input:: .T  dimension:: .Tidx 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ArgMax(scope:Scope,  input:tf.Output,  dimension:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ArgMax",
        Input: [  input,  dimension, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Returns the index with the smallest value across dimensions of a tensor.

Note that in case of ties the identity of the return value is not guaranteed.

_inputArg
 input:: .T  dimension:: .Tidx 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ArgMin(scope:Scope,  input:tf.Output,  dimension:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ArgMin",
        Input: [  input,  dimension, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Converts each entry in the given tensor to strings.  Supports many numeric

types and boolean.

_inputArg
 input:: .T 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AsString(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AsString",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes asin of x element-wise.


_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Asin(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Asin",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Asserts that the given condition is true.

If `condition` evaluates to false, print the list of tensors in `data`.
`summarize` determines how many entries of the tensors to print.

_inputArg
 condition:: .  data:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Assert(scope:Scope,  condition:tf.Output,  data:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Assert",
        Input: [  condition,  data, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Update 'ref' by assigning 'value' to it.

This operation outputs "ref" after the assignment is done.
This makes it easier to chain operations that need to use the reset value.

_inputArg
 ref:: .T  value:: .T 
_outputArg

_outputArg
 output_ref::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Assign(scope:Scope,  ref:tf.Output,  value:tf.Output, ) -> ( output_ref:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Assign",
        Input: [  ref,  value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_ref )

}

/*
Update 'ref' by adding 'value' to it.

This operation outputs "ref" after the update is done.
This makes it easier to chain operations that need to use the reset value.

_inputArg
 ref:: .T  value:: .T 
_outputArg

_outputArg
 output_ref::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AssignAdd(scope:Scope,  ref:tf.Output,  value:tf.Output, ) -> ( output_ref:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AssignAdd",
        Input: [  ref,  value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_ref )

}

/*
Update 'ref' by subtracting 'value' from it.

This operation outputs "ref" after the update is done.
This makes it easier to chain operations that need to use the reset value.

_inputArg
 ref:: .T  value:: .T 
_outputArg

_outputArg
 output_ref::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AssignSub(scope:Scope,  ref:tf.Output,  value:tf.Output, ) -> ( output_ref:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AssignSub",
        Input: [  ref,  value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_ref )

}

/*
Computes atan of x element-wise.


_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Atan(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Atan",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Computes arctangent of `y/x` element-wise, respecting signs of the arguments.

This is the angle \( \theta \in [-\pi, \pi] \) such that
\[ x = r \cos(\theta) \]
and
\[ y = r \sin(\theta) \]
where \(r = \sqrt(x^2 + y^2) \).

_inputArg
 y:: .T  x:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Atan2(scope:Scope,  y:tf.Output,  x:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Atan2",
        Input: [  y,  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Produces a visualization of audio data over time.

Spectrograms are a standard way of representing audio information as a series of
slices of frequency information, one slice for each window of time. By joining
these together into a sequence, they form a distinctive fingerprint of the sound
over time.
This op expects to receive audio data as an input, stored as floats in the range
-1 to 1, together with a window width in samples, and a stride specifying how
far to move the window between slices. From this it generates a three
dimensional output. The lowest dimension has an amplitude value for each
frequency during that time slice. The next dimension is time, with successive
frequency slices. The final dimension is for the channels in the input, so a
stereo audio input would have two here for example.
This means the layout when converted and saved as an image is rotated 90 degrees
clockwise from a typical spectrogram. Time is descending down the Y axis, and
the frequency decreases from left to right.
Each value in the result represents the square root of the sum of the real and
imaginary parts of an FFT on the current window of samples. In this way, the
lowest dimension represents the power of each frequency in the current window,
and adjacent windows are concatenated in the next dimension.
To get a more intuitive and visual look at what this operation does, you can run
tensorflow/examples/wav_to_spectrogram to read in an audio file and save out the
resulting spectrogram as a PNG image.

_inputArg
 input:: . 
_outputArg

_outputArg
 spectrogram:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AudioSpectrogram(scope:Scope,  input:tf.Output, ) -> ( spectrogram:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AudioSpectrogram",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) spectrogram )

}

/*
Outputs a `Summary` protocol buffer with audio.

The summary has up to `max_outputs` summary values containing audio. The
audio is built from `tensor` which must be 3-D with shape `[batch_size,
frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.
The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:
*  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
*  If `max_outputs` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.

_inputArg
 tag:: .  tensor:: . 
_outputArg

_outputArg
 summary:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AudioSummary(scope:Scope,  tag:tf.Output,  tensor:tf.Output, ) -> ( summary:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AudioSummary",
        Input: [  tag,  tensor, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) summary )

}

/*
Outputs a `Summary` protocol buffer with audio.

The summary has up to `max_outputs` summary values containing audio. The
audio is built from `tensor` which must be 3-D with shape `[batch_size,
frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.
The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:
*  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
*  If `max_outputs` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.

_inputArg
 tag:: .  tensor:: .  sample_rate:: . 
_outputArg

_outputArg
 summary:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AudioSummaryV2(scope:Scope,  tag:tf.Output,  tensor:tf.Output,  sample_rate:tf.Output, ) -> ( summary:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AudioSummaryV2",
        Input: [  tag,  tensor,  sample_rate, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) summary )

}

/*
Performs average pooling on the input.

Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.

_inputArg
 value:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AvgPool(scope:Scope,  value:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AvgPool",
        Input: [  value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Performs 3D average pooling on the input.


_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AvgPool3D(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AvgPool3D",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes gradients of average pooling function.


_inputArg
 orig_input_shape:: .  grad:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AvgPool3DGrad(scope:Scope,  orig_input_shape:tf.Output,  grad:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AvgPool3DGrad",
        Input: [  orig_input_shape,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes gradients of the average pooling function.


_inputArg
 orig_input_shape:: .  grad:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func AvgPoolGrad(scope:Scope,  orig_input_shape:tf.Output,  grad:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "AvgPoolGrad",
        Input: [  orig_input_shape,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Defines a barrier that persists across different graph executions.

A barrier represents a key-value map, where each key is a string, and
each value is a tuple of tensors.
At runtime, the barrier contains 'complete' and 'incomplete'
elements. A complete element has defined tensors for all components of
its value tuple, and may be accessed using BarrierTakeMany. An
incomplete element has some undefined components in its value tuple,
and may be updated using BarrierInsertMany.

_inputArg
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Barrier(scope:Scope, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Barrier",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Closes the given barrier.

This operation signals that no more new elements will be inserted in the
given barrier. Subsequent InsertMany that try to introduce a new key will fail.
Subsequent InsertMany operations that just add missing components to already
existing elements will continue to succeed. Subsequent TakeMany operations will
continue to succeed if sufficient completed elements remain in the barrier.
Subsequent TakeMany operations that would block will fail immediately.

_inputArg
 handle:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BarrierClose(scope:Scope,  handle:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BarrierClose",
        Input: [  handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Computes the number of incomplete elements in the given barrier.


_inputArg
 handle:: . 
_outputArg

_outputArg
 size:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BarrierIncompleteSize(scope:Scope,  handle:tf.Output, ) -> ( size:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BarrierIncompleteSize",
        Input: [  handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) size )

}

/*
For each key, assigns the respective value to the specified component.

If a key is not found in the barrier, this operation will create a new
incomplete element. If a key is found in the barrier, and the element
already has a value at component_index, this operation will fail with
INVALID_ARGUMENT, and leave the barrier in an undefined state.

_inputArg
 handle:: .  keys:: .  values:: .T 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BarrierInsertMany(scope:Scope,  handle:tf.Output,  keys:tf.Output,  values:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BarrierInsertMany",
        Input: [  handle,  keys,  values, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Computes the number of complete elements in the given barrier.


_inputArg
 handle:: . 
_outputArg

_outputArg
 size:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BarrierReadySize(scope:Scope,  handle:tf.Output, ) -> ( size:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BarrierReadySize",
        Input: [  handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) size )

}

/*
Takes the given number of completed elements from a barrier.

This operation concatenates completed-element component tensors along
the 0th dimension to make a single component tensor.
Elements come out of the barrier when they are complete, and in the order
in which they were placed into the barrier.  The indices output provides
information about the batch in which each element was originally inserted
into the barrier.

_inputArg
 handle:: .  num_elements:: . 
_outputArg

_outputArg
 indices::  keys::  values:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BarrierTakeMany(scope:Scope,  handle:tf.Output,  num_elements:tf.Output, ) -> ( indices:tf.Output,  keys:tf.Output,  values:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BarrierTakeMany",
        Input: [  handle,  num_elements, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) indices  op.Output(*) keys  op.Output(*) values )

}

/*


_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchCholesky(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchCholesky",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*


_inputArg
 l:: .T  grad:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchCholeskyGrad(scope:Scope,  l:tf.Output,  grad:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchCholeskyGrad",
        Input: [  l,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Creates a dataset that batches `batch_size` elements from `input_dataset`.


_inputArg
 input_dataset:: .  batch_size:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchDataset(scope:Scope,  input_dataset:tf.Output,  batch_size:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchDataset",
        Input: [  input_dataset,  batch_size, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*


_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchFFT(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchFFT",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*


_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchFFT2D(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchFFT2D",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*


_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchFFT3D(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchFFT3D",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*


_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchIFFT(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchIFFT",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*


_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchIFFT2D(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchIFFT2D",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*


_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchIFFT3D(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchIFFT3D",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Multiplies slices of two tensors in batches.

Multiplies all slices of `Tensor` `x` and `y` (each slice can be
viewed as an element of a batch), and arranges the individual results
in a single output tensor of the same batch size. Each of the
individual slices can optionally be adjointed (to adjoint a matrix
means to transpose and conjugate it) before multiplication by setting
the `adj_x` or `adj_y` flag to `True`, which are by default `False`.
The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
and `[..., r_y, c_y]`.
The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:
    r_o = c_x if adj_x else r_x
    c_o = r_y if adj_y else c_y
It is computed as:
    output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchMatMul(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchMatMul",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*


_inputArg
 input:: .T  num_lower:: .  num_upper:: . 
_outputArg

_outputArg
 band::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchMatrixBandPart(scope:Scope,  input:tf.Output,  num_lower:tf.Output,  num_upper:tf.Output, ) -> ( band:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchMatrixBandPart",
        Input: [  input,  num_lower,  num_upper, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) band )

}

/*


_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchMatrixDeterminant(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchMatrixDeterminant",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*


_inputArg
 diagonal:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchMatrixDiag(scope:Scope,  diagonal:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchMatrixDiag",
        Input: [  diagonal, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*


_inputArg
 input:: .T 
_outputArg

_outputArg
 diagonal::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchMatrixDiagPart(scope:Scope,  input:tf.Output, ) -> ( diagonal:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchMatrixDiagPart",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) diagonal )

}

/*


_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchMatrixInverse(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchMatrixInverse",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*


_inputArg
 input:: .T  diagonal:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchMatrixSetDiag(scope:Scope,  input:tf.Output,  diagonal:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchMatrixSetDiag",
        Input: [  input,  diagonal, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*


_inputArg
 matrix:: .T  rhs:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchMatrixSolve(scope:Scope,  matrix:tf.Output,  rhs:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchMatrixSolve",
        Input: [  matrix,  rhs, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*


_inputArg
 matrix:: .T  rhs:: .T  l2_regularizer:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchMatrixSolveLs(scope:Scope,  matrix:tf.Output,  rhs:tf.Output,  l2_regularizer:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchMatrixSolveLs",
        Input: [  matrix,  rhs,  l2_regularizer, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*


_inputArg
 matrix:: .T  rhs:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchMatrixTriangularSolve(scope:Scope,  matrix:tf.Output,  rhs:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchMatrixTriangularSolve",
        Input: [  matrix,  rhs, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Batch normalization.

This op is deprecated. Prefer `tf.nn.batch_normalization`.

_inputArg
 t:: .T  m:: .T  v:: .T  beta:: .T  gamma:: .T 
_outputArg

_outputArg
 result::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchNormWithGlobalNormalization(scope:Scope,  t:tf.Output,  m:tf.Output,  v:tf.Output,  beta:tf.Output,  gamma:tf.Output, ) -> ( result:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchNormWithGlobalNormalization",
        Input: [  t,  m,  v,  beta,  gamma, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) result )

}

/*
Gradients for batch normalization.

This op is deprecated. See `tf.nn.batch_normalization`.

_inputArg
 t:: .T  m:: .T  v:: .T  gamma:: .T  backprop:: .T 
_outputArg

_outputArg
 dx::T  dm::T  dv::T  db::T  dg::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchNormWithGlobalNormalizationGrad(scope:Scope,  t:tf.Output,  m:tf.Output,  v:tf.Output,  gamma:tf.Output,  backprop:tf.Output, ) -> ( dx:tf.Output,  dm:tf.Output,  dv:tf.Output,  db:tf.Output,  dg:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchNormWithGlobalNormalizationGrad",
        Input: [  t,  m,  v,  gamma,  backprop, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) dx  op.Output(*) dm  op.Output(*) dv  op.Output(*) db  op.Output(*) dg )

}

/*


_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchSelfAdjointEig(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchSelfAdjointEig",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*


_inputArg
 input:: .T 
_outputArg

_outputArg
 e::T  v::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchSelfAdjointEigV2(scope:Scope,  input:tf.Output, ) -> ( e:tf.Output,  v:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchSelfAdjointEigV2",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) e  op.Output(*) v )

}

/*


_inputArg
 input:: .T 
_outputArg

_outputArg
 s::T  u::T  v::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchSvd(scope:Scope,  input:tf.Output, ) -> ( s:tf.Output,  u:tf.Output,  v:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchSvd",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) s  op.Output(*) u  op.Output(*) v )

}

/*
BatchToSpace for 4-D tensors of type T.

This is a legacy version of the more general BatchToSpaceND.
Rearranges (permutes) data from batch into blocks of spatial data, followed by
cropping. This is the reverse transformation of SpaceToBatch. More specifically,
this op outputs a copy of the input tensor where values from the `batch`
dimension are moved in spatial blocks to the `height` and `width` dimensions,
followed by cropping along the `height` and `width` dimensions.

_inputArg
 input:: .T  crops:: .Tidx 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchToSpace(scope:Scope,  input:tf.Output,  crops:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchToSpace",
        Input: [  input,  crops, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
BatchToSpace for N-D tensors of type T.

This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of shape
`block_shape + [batch]`, interleaves these blocks back into the grid defined by
the spatial dimensions `[1, ..., M]`, to obtain a result with the same rank as
the input.  The spatial dimensions of this intermediate result are then
optionally cropped according to `crops` to produce the output.  This is the
reverse of SpaceToBatch.  See below for a precise description.

_inputArg
 input:: .T  block_shape:: .Tblock_shape  crops:: .Tcrops 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BatchToSpaceND(scope:Scope,  input:tf.Output,  block_shape:tf.Output,  crops:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BatchToSpaceND",
        Input: [  input,  block_shape,  crops, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Compute the regularized incomplete beta integral \\(I_x(a, b)\\).

The regularized incomplete beta integral is defined as:
\\(I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}\\)
where
\\(B(x; a, b) = \int_0^x t^{a-1} (1 - t)^{b-1} dt\\)
is the incomplete beta function and \\(B(a, b)\\) is the *complete*
beta function.

_inputArg
 a:: .T  b:: .T  x:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Betainc(scope:Scope,  a:tf.Output,  b:tf.Output,  x:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Betainc",
        Input: [  a,  b,  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Adds `bias` to `value`.

This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions.

_inputArg
 value:: .T  bias:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BiasAdd(scope:Scope,  value:tf.Output,  bias:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BiasAdd",
        Input: [  value,  bias, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
The backward operation for "BiasAdd" on the "bias" tensor.

It accumulates all the values from out_backprop into the feature dimension.
For NHWC data format, the feature dimension is the last. For NCHW data format,
the feature dimension is the third-to-last.

_inputArg
 out_backprop:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BiasAddGrad(scope:Scope,  out_backprop:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BiasAddGrad",
        Input: [  out_backprop, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Adds `bias` to `value`.

This is a deprecated version of BiasAdd and will be soon removed.
This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions.

_inputArg
 value:: .T  bias:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BiasAddV1(scope:Scope,  value:tf.Output,  bias:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BiasAddV1",
        Input: [  value,  bias, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Counts the number of occurrences of each value in an integer array.

Outputs a vector with length `size` and the same dtype as `weights`. If
`weights` are empty, then index `i` stores the number of times the value `i` is
counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
the value in `weights` at each index where the corresponding value in `arr` is
`i`.
Values in `arr` outside of the range [0, size) are ignored.

_inputArg
 arr:: .  size:: .  weights:: .T 
_outputArg

_outputArg
 bins::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Bincount(scope:Scope,  arr:tf.Output,  size:tf.Output,  weights:tf.Output, ) -> ( bins:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Bincount",
        Input: [  arr,  size,  weights, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) bins )

}

/*
Bitcasts a tensor from one type to another without copying data.

Given a tensor `input`, this operation returns a tensor that has the same buffer
data as `input` with datatype `type`.
If the input datatype `T` is larger than the output datatype `type` then the
shape changes from [...] to [..., sizeof(`T`)/sizeof(`type`)].
If `T` is smaller than `type`, the operator requires that the rightmost
dimension be equal to sizeof(`type`)/sizeof(`T`). The shape then goes from
[..., sizeof(`type`)/sizeof(`T`)] to [...].
*NOTE*: Bitcast is implemented as a low-level cast, so machines with different
endian orderings will give different results.

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::type 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Bitcast(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Bitcast",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Return the shape of s0 op s1 with broadcast.

Given `s0` and `s1`, tensors that represent shapes, compute `r0`, the
broadcasted shape. `s0`, `s1` and `r0` are all integer vectors.

_inputArg
 s0:: .T  s1:: .T 
_outputArg

_outputArg
 r0::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BroadcastArgs(scope:Scope,  s0:tf.Output,  s1:tf.Output, ) -> ( r0:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BroadcastArgs",
        Input: [  s0,  s1, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) r0 )

}

/*
Return the reduction indices for computing gradients of s0 op s1 with broadcast.

This is typically used by gradient computations for a broadcasting operation.

_inputArg
 s0:: .T  s1:: .T 
_outputArg

_outputArg
 r0::T  r1::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func BroadcastGradientArgs(scope:Scope,  s0:tf.Output,  s1:tf.Output, ) -> ( r0:tf.Output,  r1:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "BroadcastGradientArgs",
        Input: [  s0,  s1, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) r0  op.Output(*) r1 )

}

/*
Bucketizes 'input' based on 'boundaries'.

For example, if the inputs are
    boundaries = [0, 10, 100]
    input = [[-5, 10000]
             [150,   10]
             [5,    100]]
then the output will be
    output = [[0, 3]
              [3, 2]
              [1, 3]]

_inputArg
 input:: .T 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Bucketize(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Bucketize",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Performs beam search decoding on the logits given in input.

A note about the attribute merge_repeated: For the beam search decoder,
this means that if consecutive entries in a beam are the same, only
the first of these is emitted.  That is, when the top path is "A B B B B",
"A B" is returned if merge_repeated = True but "A B B B B" is
returned if merge_repeated = False.

_inputArg
 inputs:: .  sequence_length:: . 
_outputArg

_outputArg
 decoded_indices::  decoded_values::  decoded_shape::  log_probability:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func CTCBeamSearchDecoder(scope:Scope,  inputs:tf.Output,  sequence_length:tf.Output, ) -> ( decoded_indices:tf.Output,  decoded_values:tf.Output,  decoded_shape:tf.Output,  log_probability:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "CTCBeamSearchDecoder",
        Input: [  inputs,  sequence_length, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) decoded_indices  op.Output(*) decoded_values  op.Output(*) decoded_shape  op.Output(*) log_probability )

}

/*
Performs greedy decoding on the logits given in inputs.

A note about the attribute merge_repeated: if enabled, when
consecutive logits' maximum indices are the same, only the first of
these is emitted.  Labeling the blank '*', the sequence "A B B * B B"
becomes "A B B" if merge_repeated = True and "A B B B B" if
merge_repeated = False.
Regardless of the value of merge_repeated, if the maximum index of a given
time and batch corresponds to the blank, index `(num_classes - 1)`, no new
element is emitted.

_inputArg
 inputs:: .  sequence_length:: . 
_outputArg

_outputArg
 decoded_indices::  decoded_values::  decoded_shape::  log_probability:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func CTCGreedyDecoder(scope:Scope,  inputs:tf.Output,  sequence_length:tf.Output, ) -> ( decoded_indices:tf.Output,  decoded_values:tf.Output,  decoded_shape:tf.Output,  log_probability:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "CTCGreedyDecoder",
        Input: [  inputs,  sequence_length, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) decoded_indices  op.Output(*) decoded_values  op.Output(*) decoded_shape  op.Output(*) log_probability )

}

/*
Calculates the CTC Loss (log probability) for each batch entry.  Also calculates

the gradient.  This class performs the softmax operation for you, so inputs
should be e.g. linear projections of outputs by an LSTM.

_inputArg
 inputs:: .  labels_indices:: .  labels_values:: .  sequence_length:: . 
_outputArg

_outputArg
 loss::  gradient:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func CTCLoss(scope:Scope,  inputs:tf.Output,  labels_indices:tf.Output,  labels_values:tf.Output,  sequence_length:tf.Output, ) -> ( loss:tf.Output,  gradient:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "CTCLoss",
        Input: [  inputs,  labels_indices,  labels_values,  sequence_length, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) loss  op.Output(*) gradient )

}

/*
Cast x of type SrcT to y of DstT.


_inputArg
 x:: .SrcT 
_outputArg

_outputArg
 y::DstT 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Cast(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Cast",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Returns element-wise smallest integer in not less than x.


_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Ceil(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Ceil",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Checks a tensor for NaN and Inf values.

When run, reports an `InvalidArgument` error if `tensor` has any values
that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.

_inputArg
 tensor:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func CheckNumerics(scope:Scope,  tensor:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "CheckNumerics",
        Input: [  tensor, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the Cholesky decomposition of one or more square matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices, with the same constraints as the single matrix Cholesky
decomposition above. The output is a tensor of the same shape as the input
containing the Cholesky decompositions for all input submatrices `[..., :, :]`.

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Cholesky(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Cholesky",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the reverse mode backpropagated gradient of the Cholesky algorithm.

For an explanation see "Differentiation of the Cholesky algorithm" by
Iain Murray http://arxiv.org/abs/1602.07527.

_inputArg
 l:: .T  grad:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func CholeskyGrad(scope:Scope,  l:tf.Output,  grad:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "CholeskyGrad",
        Input: [  l,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Converts two real numbers to a complex number.

Given a tensor `real` representing the real part of a complex number, and a
tensor `imag` representing the imaginary part of a complex number, this
operation returns complex numbers elementwise of the form \\(a + bj\\), where
*a* represents the `real` part and *b* represents the `imag` part.
The input tensors `real` and `imag` must have the same shape.
For example:
```
# tensor 'real' is [2.25, 3.25]
# tensor `imag` is [4.75, 5.75]
tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
```

_inputArg
 real:: .T  imag:: .T 
_outputArg

_outputArg
 out::Tout 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Complex(scope:Scope,  real:tf.Output,  imag:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Complex",
        Input: [  real,  imag, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Computes the complex absolute value of a tensor.

Given a tensor `x` of complex numbers, this operation returns a tensor of type
`float` or `double` that is the absolute value of each element in `x`. All
elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
value is computed as \\( \sqrt{a^2 + b^2}\\).

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::Tout 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ComplexAbs(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ComplexAbs",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Computes the ids of the positions in sampled_candidates that match true_labels.

When doing log-odds NCE, the result of this op should be passed through a
SparseToDense op, then added to the logits of the sampled candidates. This has
the effect of 'removing' the sampled labels that match the true labels by
making the classifier sure that they are sampled labels.

_inputArg
 true_classes:: .  sampled_candidates:: . 
_outputArg

_outputArg
 indices::  ids::  weights:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ComputeAccidentalHits(scope:Scope,  true_classes:tf.Output,  sampled_candidates:tf.Output, ) -> ( indices:tf.Output,  ids:tf.Output,  weights:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ComputeAccidentalHits",
        Input: [  true_classes,  sampled_candidates, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) indices  op.Output(*) ids  op.Output(*) weights )

}

/*
Concatenates tensors along one dimension.


_inputArg
 concat_dim:: .  values:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Concat(scope:Scope,  concat_dim:tf.Output,  values:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Concat",
        Input: [  concat_dim,  values, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes offsets of concat inputs within its output.

For example:
```
# 'x' is [2, 2, 7]
# 'y' is [2, 3, 7]
# 'z' is [2, 5, 7]
concat_offset(2, [x, y, z]) => [0, 0, 0], [0, 2, 0], [0, 5, 0]
```
This is typically used by gradient computations for a concat operation.

_inputArg
 concat_dim:: .  shape:: . 
_outputArg

_outputArg
 offset:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ConcatOffset(scope:Scope,  concat_dim:tf.Output,  shape:tf.Output, ) -> ( offset:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ConcatOffset",
        Input: [  concat_dim,  shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) offset )

}

/*
Concatenates tensors along one dimension.


_inputArg
 values:: .T  axis:: .Tidx 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ConcatV2(scope:Scope,  values:tf.Output,  axis:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ConcatV2",
        Input: [  values,  axis, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
A conditional accumulator for aggregating gradients.

The accumulator accepts gradients marked with local_step greater or
equal to the most recent global_step known to the accumulator. The
average can be extracted from the accumulator, provided sufficient
gradients have been accumulated. Extracting the average automatically
resets the aggregate to 0, and increments the global_step recorded by
the accumulator.

_inputArg
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ConditionalAccumulator(scope:Scope, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ConditionalAccumulator",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Returns the complex conjugate of a complex number.

Given a tensor `input` of complex numbers, this operation returns a tensor of
complex numbers that are the complex conjugate of each element in `input`. The
complex numbers in `input` must be of the form \\(a + bj\\), where *a* is the
real part and *b* is the imaginary part.
The complex conjugate returned by this operation is of the form \\(a - bj\\).
For example:
```
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
```

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Conj(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Conj",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Returns a constant tensor.


_inputArg
_outputArg

_outputArg
 output::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Const(scope:Scope, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Const",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Does nothing. Serves as a control trigger for scheduling.

Only useful as a placeholder for control edges.

_inputArg
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ControlTrigger(scope:Scope, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ControlTrigger",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Computes a 2-D convolution given 4-D `input` and `filter` tensors.

Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, out_channels]`, this op
performs the following:
1. Flattens the filter to a 2-D matrix with shape
   `[filter_height * filter_width * in_channels, output_channels]`.
2. Extracts image patches from the input tensor to form a *virtual*
   tensor of shape `[batch, out_height, out_width,
   filter_height * filter_width * in_channels]`.
3. For each patch, right-multiplies the filter matrix and the image patch
   vector.
In detail, with the default NHWC format,
    output[b, i, j, k] =
        sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                        filter[di, dj, q, k]
Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

_inputArg
 input:: .T  filter:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Conv2D(scope:Scope,  input:tf.Output,  filter:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Conv2D",
        Input: [  input,  filter, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the gradients of convolution with respect to the filter.


_inputArg
 input:: .T  filter_sizes:: .  out_backprop:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Conv2DBackpropFilter(scope:Scope,  input:tf.Output,  filter_sizes:tf.Output,  out_backprop:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Conv2DBackpropFilter",
        Input: [  input,  filter_sizes,  out_backprop, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the gradients of convolution with respect to the input.


_inputArg
 input_sizes:: .  filter:: .T  out_backprop:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Conv2DBackpropInput(scope:Scope,  input_sizes:tf.Output,  filter:tf.Output,  out_backprop:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Conv2DBackpropInput",
        Input: [  input_sizes,  filter,  out_backprop, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes a 3-D convolution given 5-D `input` and `filter` tensors.

In signal processing, cross-correlation is a measure of similarity of
two waveforms as a function of a time-lag applied to one of them. This
is also known as a sliding dot product or sliding inner-product.
Our Conv3D implements a form of cross-correlation.

_inputArg
 input:: .T  filter:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Conv3D(scope:Scope,  input:tf.Output,  filter:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Conv3D",
        Input: [  input,  filter, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the gradients of 3-D convolution with respect to the filter.


_inputArg
 input:: .T  filter:: .T  out_backprop:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Conv3DBackpropFilter(scope:Scope,  input:tf.Output,  filter:tf.Output,  out_backprop:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Conv3DBackpropFilter",
        Input: [  input,  filter,  out_backprop, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the gradients of 3-D convolution with respect to the filter.


_inputArg
 input:: .T  filter_sizes:: .  out_backprop:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Conv3DBackpropFilterV2(scope:Scope,  input:tf.Output,  filter_sizes:tf.Output,  out_backprop:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Conv3DBackpropFilterV2",
        Input: [  input,  filter_sizes,  out_backprop, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the gradients of 3-D convolution with respect to the input.


_inputArg
 input:: .T  filter:: .T  out_backprop:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Conv3DBackpropInput(scope:Scope,  input:tf.Output,  filter:tf.Output,  out_backprop:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Conv3DBackpropInput",
        Input: [  input,  filter,  out_backprop, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the gradients of 3-D convolution with respect to the input.


_inputArg
 input_sizes:: .  filter:: .T  out_backprop:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Conv3DBackpropInputV2(scope:Scope,  input_sizes:tf.Output,  filter:tf.Output,  out_backprop:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Conv3DBackpropInputV2",
        Input: [  input_sizes,  filter,  out_backprop, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Copy Op.

Performs CPU-to-CPU or GPU-to-GPU deep-copying of tensor, depending on the
device on which the tensor is allocated.
N.B.: If the all downstream attached debug ops are disabled given the current
gRPC gating status, the output will simply forward the input tensor without
deep-copying. See the documentation of Debug* ops for more details.
Unlike the CopyHost Op, this op does not have HostMemory constraint on its
input or output.

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Copy(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Copy",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Copy Host Op.

Performs CPU-to-CPU deep-copying of tensor.
N.B.: If the all downstream attached debug ops are disabled given the current
gRPC gating status, the output will simply forward the input tensor without
deep-copying. See the documentation of Debug* ops for more details.
Unlike the Copy Op, this op has HostMemory constraint on its input or output.

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func CopyHost(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "CopyHost",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes cos of x element-wise.


_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Cos(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Cos",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Increments 'ref' until it reaches 'limit'.


_inputArg
 ref:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func CountUpTo(scope:Scope,  ref:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "CountUpTo",
        Input: [  ref, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Extracts crops from the input image tensor and bilinearly resizes them (possibly

with aspect ratio change) to a common output size specified by `crop_size`. This
is more general than the `crop_to_bounding_box` op which extracts a fixed size
slice from the input image and does not allow resizing or aspect ratio change.
Returns a tensor with `crops` from the input `image` at positions defined at the
bounding box locations in `boxes`. The cropped boxes are all resized (with
bilinear interpolation) to a fixed `size = [crop_height, crop_width]`. The
result is a 4-D tensor `[num_boxes, crop_height, crop_width, depth]`.

_inputArg
 image:: .T  boxes:: .  box_ind:: .  crop_size:: . 
_outputArg

_outputArg
 crops:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func CropAndResize(scope:Scope,  image:tf.Output,  boxes:tf.Output,  box_ind:tf.Output,  crop_size:tf.Output, ) -> ( crops:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "CropAndResize",
        Input: [  image,  boxes,  box_ind,  crop_size, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) crops )

}

/*
Computes the gradient of the crop_and_resize op wrt the input boxes tensor.


_inputArg
 grads:: .  image:: .T  boxes:: .  box_ind:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func CropAndResizeGradBoxes(scope:Scope,  grads:tf.Output,  image:tf.Output,  boxes:tf.Output,  box_ind:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "CropAndResizeGradBoxes",
        Input: [  grads,  image,  boxes,  box_ind, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the gradient of the crop_and_resize op wrt the input image tensor.


_inputArg
 grads:: .  boxes:: .  box_ind:: .  image_size:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func CropAndResizeGradImage(scope:Scope,  grads:tf.Output,  boxes:tf.Output,  box_ind:tf.Output,  image_size:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "CropAndResizeGradImage",
        Input: [  grads,  boxes,  box_ind,  image_size, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Compute the pairwise cross product.

`a` and `b` must be the same shape; they can either be simple 3-element vectors,
or any shape where the innermost dimension is 3. In the latter case, each pair
of corresponding 3-element vectors is cross-multiplied independently.

_inputArg
 a:: .T  b:: .T 
_outputArg

_outputArg
 product::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Cross(scope:Scope,  a:tf.Output,  b:tf.Output, ) -> ( product:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Cross",
        Input: [  a,  b, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) product )

}

/*
Compute the cumulative product of the tensor `x` along `axis`.

By default, this op performs an inclusive cumprod, which means that the first
element of the input is identical to the first element of the output:
```python
tf.cumprod([a, b, c])  # => [a, a * b, a * b * c]
```
By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
performed instead:
```python
tf.cumprod([a, b, c], exclusive=True)  # => [1, a, a * b]
```
By setting the `reverse` kwarg to `True`, the cumprod is performed in the
opposite direction:
```python
tf.cumprod([a, b, c], reverse=True)  # => [a * b * c, b * c, c]
```
This is more efficient than using separate `tf.reverse` ops.
The `reverse` and `exclusive` kwargs can also be combined:
```python
tf.cumprod([a, b, c], exclusive=True, reverse=True)  # => [b * c, c, 1]
```

_inputArg
 x:: .T  axis:: .Tidx 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Cumprod(scope:Scope,  x:tf.Output,  axis:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Cumprod",
        Input: [  x,  axis, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Compute the cumulative sum of the tensor `x` along `axis`.

By default, this op performs an inclusive cumsum, which means that the first
element of the input is identical to the first element of the output:
```python
tf.cumsum([a, b, c])  # => [a, a + b, a + b + c]
```
By setting the `exclusive` kwarg to `True`, an exclusive cumsum is
performed instead:
```python
tf.cumsum([a, b, c], exclusive=True)  # => [0, a, a + b]
```
By setting the `reverse` kwarg to `True`, the cumsum is performed in the
opposite direction:
```python
tf.cumsum([a, b, c], reverse=True)  # => [a + b + c, b + c, c]
```
This is more efficient than using separate `tf.reverse` ops.
The `reverse` and `exclusive` kwargs can also be combined:
```python
tf.cumsum([a, b, c], exclusive=True, reverse=True)  # => [b + c, c, 0]
```

_inputArg
 x:: .T  axis:: .Tidx 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Cumsum(scope:Scope,  x:tf.Output,  axis:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Cumsum",
        Input: [  x,  axis, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Debug Identity Op.

Provides an identity mapping of the non-Ref type input tensor for debugging.

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DebugIdentity(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DebugIdentity",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Debug NaN Value Counter Op

Counts number of NaNs in the input tensor, for debugging.

_inputArg
 input:: .T 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DebugNanCount(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DebugNanCount",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Debug Numeric Summary Op.

Provide a basic summary of numeric value types, range and distribution.

_inputArg
 input:: .T 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DebugNumericSummary(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DebugNumericSummary",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Decode web-safe base64-encoded strings.

Input may or may not have padding at the end. See EncodeBase64 for padding.
Web-safe means that input must use - and _ instead of + and /.

_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DecodeBase64(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DecodeBase64",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Decode the first frame of a BMP-encoded image to a uint8 tensor.

The attr `channels` indicates the desired number of color channels for the
decoded image.
Accepted values are:
*   0: Use the number of channels in the BMP-encoded image.
*   3: output an RGB image.
*   4: output an RGBA image.

_inputArg
 contents:: . 
_outputArg

_outputArg
 image:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DecodeBmp(scope:Scope,  contents:tf.Output, ) -> ( image:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DecodeBmp",
        Input: [  contents, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) image )

}

/*
Convert CSV records to tensors. Each column maps to one tensor.

RFC 4180 format is expected for the CSV records.
(https://tools.ietf.org/html/rfc4180)
Note that we allow leading and trailing spaces with int or float field.

_inputArg
 records:: .  record_defaults:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DecodeCSV(scope:Scope,  records:tf.Output,  record_defaults:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DecodeCSV",
        Input: [  records,  record_defaults, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Decode the first frame of a GIF-encoded image to a uint8 tensor.

GIF with frame or transparency compression are not supported
convert animated GIF from compressed to uncompressed by:
    convert $src.gif -coalesce $dst.gif
This op also supports decoding JPEGs and PNGs, though it is cleaner to use
`tf.image.decode_image`.

_inputArg
 contents:: . 
_outputArg

_outputArg
 image:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DecodeGif(scope:Scope,  contents:tf.Output, ) -> ( image:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DecodeGif",
        Input: [  contents, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) image )

}

/*
Convert JSON-encoded Example records to binary protocol buffer strings.

This op translates a tensor containing Example records, encoded using
the [standard JSON
mapping](https://developers.google.com/protocol-buffers/docs/proto3#json),
into a tensor containing the same records encoded as binary protocol
buffers. The resulting tensor can then be fed to any of the other
Example-parsing ops.

_inputArg
 json_examples:: . 
_outputArg

_outputArg
 binary_examples:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DecodeJSONExample(scope:Scope,  json_examples:tf.Output, ) -> ( binary_examples:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DecodeJSONExample",
        Input: [  json_examples, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) binary_examples )

}

/*
Decode a JPEG-encoded image to a uint8 tensor.

The attr `channels` indicates the desired number of color channels for the
decoded image.
Accepted values are:
*   0: Use the number of channels in the JPEG-encoded image.
*   1: output a grayscale image.
*   3: output an RGB image.
If needed, the JPEG-encoded image is transformed to match the requested number
of color channels.
The attr `ratio` allows downscaling the image by an integer factor during
decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
downscaling the image later.
This op also supports decoding PNGs and non-animated GIFs since the interface is
the same, though it is cleaner to use `tf.image.decode_image`.

_inputArg
 contents:: . 
_outputArg

_outputArg
 image:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DecodeJpeg(scope:Scope,  contents:tf.Output, ) -> ( image:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DecodeJpeg",
        Input: [  contents, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) image )

}

/*
Decode a PNG-encoded image to a uint8 or uint16 tensor.

The attr `channels` indicates the desired number of color channels for the
decoded image.
Accepted values are:
*   0: Use the number of channels in the PNG-encoded image.
*   1: output a grayscale image.
*   3: output an RGB image.
*   4: output an RGBA image.
If needed, the PNG-encoded image is transformed to match the requested number
of color channels.
This op also supports decoding JPEGs and non-animated GIFs since the interface
is the same, though it is cleaner to use `tf.image.decode_image`.

_inputArg
 contents:: . 
_outputArg

_outputArg
 image::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DecodePng(scope:Scope,  contents:tf.Output, ) -> ( image:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DecodePng",
        Input: [  contents, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) image )

}

/*
Reinterpret the bytes of a string as a vector of numbers.


_inputArg
 bytes:: . 
_outputArg

_outputArg
 output::out_type 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DecodeRaw(scope:Scope,  bytes:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DecodeRaw",
        Input: [  bytes, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Decode a 16-bit PCM WAV file to a float tensor.

The -32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float.
When desired_channels is set, if the input contains fewer channels than this
then the last channel will be duplicated to give the requested number, else if
the input has more channels than requested then the additional channels will be
ignored.
If desired_samples is set, then the audio will be cropped or padded with zeroes
to the requested length.
The first output contains a Tensor with the content of the audio samples. The
lowest dimension will be the number of channels, and the second will be the
number of samples. For example, a ten-sample-long stereo WAV file should give an
output shape of [10, 2].

_inputArg
 contents:: . 
_outputArg

_outputArg
 audio::  sample_rate:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DecodeWav(scope:Scope,  contents:tf.Output, ) -> ( audio:tf.Output,  sample_rate:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DecodeWav",
        Input: [  contents, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) audio  op.Output(*) sample_rate )

}

/*
Delete the tensor specified by its handle in the session.


_inputArg
 handle:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DeleteSessionTensor(scope:Scope,  handle:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DeleteSessionTensor",
        Input: [  handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Applies set operation along last dimension of 2 `Tensor` inputs.

See SetOperationOp::SetOperationFromContext for values of `set_operation`.
Output `result` is a `SparseTensor` represented by `result_indices`,
`result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
dimension contains the result of `set_operation` applied to the corresponding
`[0...n-1]` dimension of `set`.

_inputArg
 set1:: .T  set2:: .T 
_outputArg

_outputArg
 result_indices::  result_values::T  result_shape:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DenseToDenseSetOperation(scope:Scope,  set1:tf.Output,  set2:tf.Output, ) -> ( result_indices:tf.Output,  result_values:tf.Output,  result_shape:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DenseToDenseSetOperation",
        Input: [  set1,  set2, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) result_indices  op.Output(*) result_values  op.Output(*) result_shape )

}

/*
Creates a dataset that yields a SparseTensor for each element of the input.


_inputArg
 input_dataset:: .  batch_size:: .  row_shape:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DenseToSparseBatchDataset(scope:Scope,  input_dataset:tf.Output,  batch_size:tf.Output,  row_shape:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DenseToSparseBatchDataset",
        Input: [  input_dataset,  batch_size,  row_shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Applies set operation along last dimension of `Tensor` and `SparseTensor`.

See SetOperationOp::SetOperationFromContext for values of `set_operation`.
Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
ignored.
If `validate_indices` is `True`, this op validates the order and range of `set2`
indices.
Output `result` is a `SparseTensor` represented by `result_indices`,
`result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
dimension contains the result of `set_operation` applied to the corresponding
`[0...n-1]` dimension of `set`.

_inputArg
 set1:: .T  set2_indices:: .  set2_values:: .T  set2_shape:: . 
_outputArg

_outputArg
 result_indices::  result_values::T  result_shape:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DenseToSparseSetOperation(scope:Scope,  set1:tf.Output,  set2_indices:tf.Output,  set2_values:tf.Output,  set2_shape:tf.Output, ) -> ( result_indices:tf.Output,  result_values:tf.Output,  result_shape:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DenseToSparseSetOperation",
        Input: [  set1,  set2_indices,  set2_values,  set2_shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) result_indices  op.Output(*) result_values  op.Output(*) result_shape )

}

/*
DepthToSpace for tensors of type T.

Rearranges data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically,
this op outputs a copy of the input tensor where values from the `depth`
dimension are moved in spatial blocks to the `height` and `width` dimensions.
The attr `block_size` indicates the input block size and how the data is moved.
  * Chunks of data of size `block_size * block_size` from depth are rearranged
    into non-overlapping blocks of size `block_size x block_size`
  * The width the output tensor is `input_depth * block_size`, whereas the
    height is `input_height * block_size`.
  * The depth of the input tensor must be divisible by
    `block_size * block_size`.
That is, assuming the input is in the shape:
`[batch, height, width, depth]`,
the shape of the output will be:
`[batch, height*block_size, width*block_size, depth/(block_size*block_size)]`
This operation requires that the input tensor be of rank 4, and that
`block_size` be >=1 and that `block_size * block_size` be a divisor of the
input depth.
This operation is useful for resizing the activations between convolutions
(but keeping all data), e.g. instead of pooling. It is also useful for training
purely convolutional models.
For example, given this input of shape `[1, 1, 1, 4]`, and a block size of 2:
```
x = [[[[1, 2, 3, 4]]]]
```
This operation will output a tensor of shape `[1, 2, 2, 1]`:
```
   [[[[1], [2]],
     [[3], [4]]]]
```
Here, the input has a batch of 1 and each batch element has shape `[1, 1, 4]`,
the corresponding output will have 2x2 elements and will have a depth of
1 channel (1 = `4 / (block_size * block_size)`).
The output element shape is `[2, 2, 1]`.
For an input tensor with larger depth, here of shape `[1, 1, 1, 12]`, e.g.
```
x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
```
This operation, for block size of 2, will return the following tensor of shape
`[1, 2, 2, 3]`
```
   [[[[1, 2, 3], [4, 5, 6]],
     [[7, 8, 9], [10, 11, 12]]]]
```
Similarly, for the following input of shape `[1 2 2 4]`, and a block size of 2:
```
x =  [[[[1, 2, 3, 4],
       [5, 6, 7, 8]],
      [[9, 10, 11, 12],
       [13, 14, 15, 16]]]]
```
the operator will return the following tensor of shape `[1 4 4 1]`:
```
x = [[ [1],   [2],  [5],  [6]],
     [ [3],   [4],  [7],  [8]],
     [ [9],  [10], [13],  [14]],
     [ [11], [12], [15],  [16]]]
```

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DepthToSpace(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DepthToSpace",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.

Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, channel_multiplier]`, containing
`in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
a different filter to each input channel (expanding from 1 channel to
`channel_multiplier` channels for each), then concatenates the results
together. Thus, the output has `in_channels * channel_multiplier` channels.
for k in 0..in_channels-1
  for q in 0..channel_multiplier-1
    output[b, i, j, k * channel_multiplier + q] =
      sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                        filter[di, dj, k, q]
Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

_inputArg
 input:: .T  filter:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DepthwiseConv2dNative(scope:Scope,  input:tf.Output,  filter:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DepthwiseConv2dNative",
        Input: [  input,  filter, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the gradients of depthwise convolution with respect to the filter.


_inputArg
 input:: .T  filter_sizes:: .  out_backprop:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DepthwiseConv2dNativeBackpropFilter(scope:Scope,  input:tf.Output,  filter_sizes:tf.Output,  out_backprop:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DepthwiseConv2dNativeBackpropFilter",
        Input: [  input,  filter_sizes,  out_backprop, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the gradients of depthwise convolution with respect to the input.


_inputArg
 input_sizes:: .  filter:: .T  out_backprop:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DepthwiseConv2dNativeBackpropInput(scope:Scope,  input_sizes:tf.Output,  filter:tf.Output,  out_backprop:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DepthwiseConv2dNativeBackpropInput",
        Input: [  input_sizes,  filter,  out_backprop, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Dequantize the 'input' tensor into a float Tensor.

[min_range, max_range] are scalar floats that specify the range for
the 'input' data. The 'mode' attribute controls exactly which calculations are
used to convert the float values to their quantized equivalents.
In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:
```
if T == qint8, in[i] += (range(T) + 1)/ 2.0
out[i] = min_range + (in[i]* (max_range - min_range) / range(T))
```
here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`
*MIN_COMBINED Mode Example*
If the input comes from a QuantizedRelu6, the output type is
quint8 (range of 0-255) but the possible range of QuantizedRelu6 is
0-6.  The min_range and max_range values are therefore 0.0 and 6.0.
Dequantize on quint8 will take each value, cast to float, and multiply
by 6 / 255.
Note that if quantizedtype is qint8, the operation will additionally add
each value by 128 prior to casting.
If the mode is 'MIN_FIRST', then this approach is used:
```c++
number_of_steps = 1 << (# of bits in T)
range_adjust = number_of_steps / (number_of_steps - 1)
range = (range_max - range_min) * range_adjust
range_scale = range / number_of_steps
const double offset_input = static_cast<double>(input) - lowest_quantized;
result = range_min + ((input - numeric_limits<T>::min()) * range_scale)
```

_inputArg
 input:: .T  min_range:: .  max_range:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Dequantize(scope:Scope,  input:tf.Output,  min_range:tf.Output,  max_range:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Dequantize",
        Input: [  input,  min_range,  max_range, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Deserialize and concatenate `SparseTensors` from a serialized minibatch.

The input `serialized_sparse` must be a string matrix of shape `[N x 3]` where
`N` is the minibatch size and the rows correspond to packed outputs of
`SerializeSparse`.  The ranks of the original `SparseTensor` objects
must all match.  When the final `SparseTensor` is created, it has rank one
higher than the ranks of the incoming `SparseTensor` objects
(they have been concatenated along a new row dimension).
The output `SparseTensor` object's shape values for all dimensions but the
first are the max across the input `SparseTensor` objects' shape values
for the corresponding dimensions.  Its first shape value is `N`, the minibatch
size.
The input `SparseTensor` objects' indices are assumed ordered in
standard lexicographic order.  If this is not the case, after this
step run `SparseReorder` to restore index ordering.
For example, if the serialized input is a `[2 x 3]` matrix representing two
original `SparseTensor` objects:
    index = [ 0]
            [10]
            [20]
    values = [1, 2, 3]
    shape = [50]
and
    index = [ 2]
            [10]
    values = [4, 5]
    shape = [30]
then the final deserialized `SparseTensor` will be:
    index = [0  0]
            [0 10]
            [0 20]
            [1  2]
            [1 10]
    values = [1, 2, 3, 4, 5]
    shape = [2 50]

_inputArg
 serialized_sparse:: . 
_outputArg

_outputArg
 sparse_indices::  sparse_values::dtype  sparse_shape:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DeserializeManySparse(scope:Scope,  serialized_sparse:tf.Output, ) -> ( sparse_indices:tf.Output,  sparse_values:tf.Output,  sparse_shape:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DeserializeManySparse",
        Input: [  serialized_sparse, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) sparse_indices  op.Output(*) sparse_values  op.Output(*) sparse_shape )

}

/*
Destroys the temporary variable and returns its final value.

Sets output to the value of the Tensor pointed to by 'ref', then destroys
the temporary variable called 'var_name'.
All other uses of 'ref' *must* have executed before this op.
This is typically achieved by chaining the ref through each assign op, or by
using control dependencies.
Outputs the final value of the tensor pointed to by 'ref'.

_inputArg
 ref:: .T 
_outputArg

_outputArg
 value::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DestroyTemporaryVariable(scope:Scope,  ref:tf.Output, ) -> ( value:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DestroyTemporaryVariable",
        Input: [  ref, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) value )

}

/*
Returns a diagonal tensor with a given diagonal values.

Given a `diagonal`, this operation returns a tensor with the `diagonal` and
everything else padded with zeros. The diagonal is computed as follows:
Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:
`output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.
For example:
```
# 'diagonal' is [1, 2, 3, 4]
tf.diag(diagonal) ==> [[1, 0, 0, 0]
                       [0, 2, 0, 0]
                       [0, 0, 3, 0]
                       [0, 0, 0, 4]]
```

_inputArg
 diagonal:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Diag(scope:Scope,  diagonal:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Diag",
        Input: [  diagonal, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Returns the diagonal part of the tensor.

This operation returns a tensor with the `diagonal` part
of the `input`. The `diagonal` part is computed as follows:
Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
tensor of rank `k` with dimensions `[D1,..., Dk]` where:
`diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.
For example:
```
# 'input' is [[1, 0, 0, 0]
              [0, 2, 0, 0]
              [0, 0, 3, 0]
              [0, 0, 0, 4]]
tf.diag_part(input) ==> [1, 2, 3, 4]
```

_inputArg
 input:: .T 
_outputArg

_outputArg
 diagonal::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DiagPart(scope:Scope,  input:tf.Output, ) -> ( diagonal:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DiagPart",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) diagonal )

}

/*
Computes Psi, the derivative of Lgamma (the log of the absolute value of

`Gamma(x)`), element-wise.

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Digamma(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Digamma",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors.

The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
`filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
input channel is processed independently of the others with its own structuring
function. The `output` tensor has shape
`[batch, out_height, out_width, depth]`. The spatial dimensions of the output
tensor depend on the `padding` algorithm. We currently only support the default
"NHWC" `data_format`.
In detail, the grayscale morphological 2-D dilation is the max-sum correlation
(for consistency with `conv2d`, we use unmirrored filters):
    output[b, y, x, c] =
       max_{dy, dx} input[b,
                          strides[1] * y + rates[1] * dy,
                          strides[2] * x + rates[2] * dx,
                          c] +
                    filter[dy, dx, c]
Max-pooling is a special case when the filter has size equal to the pooling
kernel size and contains all zeros.
Note on duality: The dilation of `input` by the `filter` is equal to the
negation of the erosion of `-input` by the reflected `filter`.

_inputArg
 input:: .T  filter:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Dilation2D(scope:Scope,  input:tf.Output,  filter:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Dilation2D",
        Input: [  input,  filter, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the gradient of morphological 2-D dilation with respect to the filter.


_inputArg
 input:: .T  filter:: .T  out_backprop:: .T 
_outputArg

_outputArg
 filter_backprop::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Dilation2DBackpropFilter(scope:Scope,  input:tf.Output,  filter:tf.Output,  out_backprop:tf.Output, ) -> ( filter_backprop:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Dilation2DBackpropFilter",
        Input: [  input,  filter,  out_backprop, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) filter_backprop )

}

/*
Computes the gradient of morphological 2-D dilation with respect to the input.


_inputArg
 input:: .T  filter:: .T  out_backprop:: .T 
_outputArg

_outputArg
 in_backprop::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Dilation2DBackpropInput(scope:Scope,  input:tf.Output,  filter:tf.Output,  out_backprop:tf.Output, ) -> ( in_backprop:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Dilation2DBackpropInput",
        Input: [  input,  filter,  out_backprop, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) in_backprop )

}

/*
Returns x / y element-wise.

*NOTE*: `Div` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Div(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Div",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Draw bounding boxes on a batch of images.

Outputs a copy of `images` but draws on top of the pixels zero or more bounding
boxes specified by the locations in `boxes`. The coordinates of the each
bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`. The
bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
height of the underlying image.
For example, if an image is 100 x 200 pixels and the bounding box is
`[0.1, 0.2, 0.5, 0.9]`, the bottom-left and upper-right coordinates of the
bounding box will be `(10, 40)` to `(50, 180)`.
Parts of the bounding box may fall outside the image.

_inputArg
 images:: .T  boxes:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DrawBoundingBoxes(scope:Scope,  images:tf.Output,  boxes:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DrawBoundingBoxes",
        Input: [  images,  boxes, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Partitions `data` into `num_partitions` tensors using indices from `partitions`.

For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
are placed in `outputs[i]` in lexicographic order of `js`, and the first
dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
In detail,
```python
    outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]
    outputs[i] = pack([data[js, ...] for js if partitions[js] == i])
```
`data.shape` must start with `partitions.shape`.
For example:
```python
    # Scalar partitions.
    partitions = 1
    num_partitions = 2
    data = [10, 20]
    outputs[0] = []  # Empty with shape [0, 2]
    outputs[1] = [[10, 20]]
    # Vector partitions.
    partitions = [0, 0, 1, 1, 0]
    num_partitions = 2
    data = [10, 20, 30, 40, 50]
    outputs[0] = [10, 20, 50]
    outputs[1] = [30, 40]
```
See `dynamic_stitch` for an example on how to merge partitions back.
<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/DynamicPartition.png" alt>
</div>

_inputArg
 data:: .T  partitions:: . 
_outputArg

_outputArg
 outputs::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DynamicPartition(scope:Scope,  data:tf.Output,  partitions:tf.Output, ) -> ( outputs:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DynamicPartition",
        Input: [  data,  partitions, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) outputs )

}

/*
Interleave the values from the `data` tensors into a single tensor.

Builds a merged tensor such that
```python
    merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
```
For example, if each `indices[m]` is scalar or vector, we have
```python
    # Scalar indices:
    merged[indices[m], ...] = data[m][...]
    # Vector indices:
    merged[indices[m][i], ...] = data[m][i, ...]
```
Each `data[i].shape` must start with the corresponding `indices[i].shape`,
and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
must have `data[i].shape = indices[i].shape + constant`.  In terms of this
`constant`, the output shape is
    merged.shape = [max(indices)] + constant
Values are merged in order, so if an index appears in both `indices[m][i]` and
`indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
merged result.
For example:
```python
    indices[0] = 6
    indices[1] = [4, 1]
    indices[2] = [[5, 2], [0, 3]]
    data[0] = [61, 62]
    data[1] = [[41, 42], [11, 12]]
    data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
    merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
              [51, 52], [61, 62]]
```
This method can be used to merge partitions created by `dynamic_partition`
as illustrated on the following example:
```python
    # Apply function (increments x_i) on elements for which a certain condition
    # apply (x_i != -1 in this example).
    x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
    condition_mask=tf.not_equal(x,tf.constant(-1.))
    partitioned_data = tf.dynamic_partition(
        x, tf.cast(condition_mask, tf.int32) , 2)
    partitioned_data[1] = partitioned_data[1] + 1.0
    condition_indices = tf.dynamic_partition(
        tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
    x = tf.dynamic_stitch(condition_indices, partitioned_data)
    # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
    # unchanged.
```
<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/DynamicStitch.png" alt>
</div>

_inputArg
 indices:: .  data:: .T 
_outputArg

_outputArg
 merged::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func DynamicStitch(scope:Scope,  indices:tf.Output,  data:tf.Output, ) -> ( merged:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "DynamicStitch",
        Input: [  indices,  data, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) merged )

}

/*
Computes the (possibly normalized) Levenshtein Edit Distance.

The inputs are variable-length sequences provided by SparseTensors
  (hypothesis_indices, hypothesis_values, hypothesis_shape)
and
  (truth_indices, truth_values, truth_shape).
The inputs are:

_inputArg
 hypothesis_indices:: .  hypothesis_values:: .T  hypothesis_shape:: .  truth_indices:: .  truth_values:: .T  truth_shape:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func EditDistance(scope:Scope,  hypothesis_indices:tf.Output,  hypothesis_values:tf.Output,  hypothesis_shape:tf.Output,  truth_indices:tf.Output,  truth_values:tf.Output,  truth_shape:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "EditDistance",
        Input: [  hypothesis_indices,  hypothesis_values,  hypothesis_shape,  truth_indices,  truth_values,  truth_shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.

See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
](http://arxiv.org/abs/1511.07289)

_inputArg
 features:: .T 
_outputArg

_outputArg
 activations::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Elu(scope:Scope,  features:tf.Output, ) -> ( activations:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Elu",
        Input: [  features, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) activations )

}

/*
Computes gradients for the exponential linear (Elu) operation.


_inputArg
 gradients:: .T  outputs:: .T 
_outputArg

_outputArg
 backprops::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func EluGrad(scope:Scope,  gradients:tf.Output,  outputs:tf.Output, ) -> ( backprops:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "EluGrad",
        Input: [  gradients,  outputs, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) backprops )

}

/*
Encode strings into web-safe base64 format.

Refer to the following article for more information on base64 format:
en.wikipedia.org/wiki/Base64. Base64 strings may have padding with '=' at the
end so that the encoded has length multiple of 4. See Padding section of the
link above.
Web-safe means that the encoder uses - and _ instead of + and /.

_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func EncodeBase64(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "EncodeBase64",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
JPEG-encode an image.

`image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.
The attr `format` can be used to override the color format of the encoded
output.  Values can be:
*   `''`: Use a default format based on the number of channels in the image.
*   `grayscale`: Output a grayscale JPEG image.  The `channels` dimension
    of `image` must be 1.
*   `rgb`: Output an RGB JPEG image. The `channels` dimension
    of `image` must be 3.
If `format` is not specified or is the empty string, a default format is picked
in function of the number of channels in `image`:
*   1: Output a grayscale image.
*   3: Output an RGB image.

_inputArg
 image:: . 
_outputArg

_outputArg
 contents:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func EncodeJpeg(scope:Scope,  image:tf.Output, ) -> ( contents:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "EncodeJpeg",
        Input: [  image, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) contents )

}

/*
PNG-encode an image.

`image` is a 3-D uint8 or uint16 Tensor of shape `[height, width, channels]`
where `channels` is:
*   1: for grayscale.
*   2: for grayscale + alpha.
*   3: for RGB.
*   4: for RGBA.
The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
default or a value from 0 to 9.  9 is the highest compression level, generating
the smallest output, but is slower.

_inputArg
 image:: .T 
_outputArg

_outputArg
 contents:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func EncodePng(scope:Scope,  image:tf.Output, ) -> ( contents:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "EncodePng",
        Input: [  image, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) contents )

}

/*
Encode audio data using the WAV file format.

This operation will generate a string suitable to be saved out to create a .wav
audio file. It will be encoded in the 16-bit PCM format. It takes in float
values in the range -1.0f to 1.0f, and any outside that value will be clamped to
that range.
`audio` is a 2-D float Tensor of shape `[length, channels]`.
`sample_rate` is a scalar Tensor holding the rate to use (e.g. 44100).

_inputArg
 audio:: .  sample_rate:: . 
_outputArg

_outputArg
 contents:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func EncodeWav(scope:Scope,  audio:tf.Output,  sample_rate:tf.Output, ) -> ( contents:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "EncodeWav",
        Input: [  audio,  sample_rate, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) contents )

}

/*
Creates or finds a child frame, and makes `data` available to the child frame.

This op is used together with `Exit` to create loops in the graph.
The unique `frame_name` is used by the `Executor` to identify frames. If
`is_constant` is true, `output` is a constant in the child frame; otherwise
it may be changed in the child frame. At most `parallel_iterations` iterations
are run in parallel in the child frame.

_inputArg
 data:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Enter(scope:Scope,  data:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Enter",
        Input: [  data, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Returns the truth value of (x == y) element-wise.

*NOTE*: `Equal` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Equal(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Equal",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Computes the Gauss error function of `x` element-wise.


_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Erf(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Erf",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Computes the complementary error function of `x` element-wise.


_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Erfc(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Erfc",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Exits the current frame to its parent frame.

Exit makes its input `data` available to the parent frame.

_inputArg
 data:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Exit(scope:Scope,  data:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Exit",
        Input: [  data, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes exponential of x element-wise.  \\(y = e^x\\).


_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Exp(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Exp",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Inserts a dimension of 1 into a tensor's shape.

Given a tensor `input`, this operation inserts a dimension of 1 at the
dimension index `dim` of `input`'s shape. The dimension index `dim` starts at
zero; if you specify a negative number for `dim` it is counted backward from
the end.
This operation is useful if you want to add a batch dimension to a single
element. For example, if you have a single image of shape `[height, width,
channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
which will make the shape `[1, height, width, channels]`.
Other examples:
```
# 't' is a tensor of shape [2]
shape(expand_dims(t, 0)) ==> [1, 2]
shape(expand_dims(t, 1)) ==> [2, 1]
shape(expand_dims(t, -1)) ==> [2, 1]
# 't2' is a tensor of shape [2, 3, 5]
shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
```
This operation requires that:
`-1-input.dims() <= dim <= input.dims()`
This operation is related to `squeeze()`, which removes dimensions of
size 1.

_inputArg
 input:: .T  dim:: .Tdim 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ExpandDims(scope:Scope,  input:tf.Output,  dim:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ExpandDims",
        Input: [  input,  dim, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes exponential of x - 1 element-wise.

I.e., \\(y = (\exp x) - 1\\).

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Expm1(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Expm1",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Extracts a glimpse from the input tensor.

Returns a set of windows called glimpses extracted at location
`offsets` from the input tensor. If the windows only partially
overlaps the inputs, the non overlapping areas will be filled with
random noise.
The result is a 4-D tensor of shape `[batch_size, glimpse_height,
glimpse_width, channels]`. The channels and batch dimensions are the
same as that of the input tensor. The height and width of the output
windows are specified in the `size` parameter.
The argument `normalized` and `centered` controls how the windows are built:
* If the coordinates are normalized but not centered, 0.0 and 1.0
  correspond to the minimum and maximum of each height and width
  dimension.
* If the coordinates are both normalized and centered, they range from
  -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper
  left corner, the lower right corner is located at (1.0, 1.0) and the
  center is at (0, 0).
* If the coordinates are not normalized they are interpreted as
  numbers of pixels.

_inputArg
 input:: .  size:: .  offsets:: . 
_outputArg

_outputArg
 glimpse:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ExtractGlimpse(scope:Scope,  input:tf.Output,  size:tf.Output,  offsets:tf.Output, ) -> ( glimpse:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ExtractGlimpse",
        Input: [  input,  size,  offsets, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) glimpse )

}

/*
Extract `patches` from `images` and put them in the "depth" output dimension.


_inputArg
 images:: .T 
_outputArg

_outputArg
 patches::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ExtractImagePatches(scope:Scope,  images:tf.Output, ) -> ( patches:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ExtractImagePatches",
        Input: [  images, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) patches )

}

/*
Fast Fourier transform.

Computes the 1-dimensional discrete Fourier transform over the inner-most
dimension of `input`.

_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FFT(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FFT",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
2D fast Fourier transform.

Computes the 2-dimensional discrete Fourier transform over the inner-most
2 dimensions of `input`.

_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FFT2D(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FFT2D",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
3D fast Fourier transform.

Computes the 3-dimensional discrete Fourier transform over the inner-most 3
dimensions of `input`.

_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FFT3D(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FFT3D",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
A queue that produces elements in first-in first-out order.


_inputArg
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FIFOQueue(scope:Scope, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FIFOQueue",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
A queue that produces elements in first-in first-out order.


_inputArg
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FIFOQueueV2(scope:Scope, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FIFOQueueV2",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Output a fact about factorials.


_inputArg
_outputArg

_outputArg
 fact:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Fact(scope:Scope, ) -> ( fact:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Fact",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) fact )

}

/*
Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same type.

Attributes [min; max] define the clamping range for the 'inputs' data.  Op
divides this range into 255 steps (total of 256 values), then replaces each
'inputs' value with the closest of the quantized step values.
'num_bits' is the bitwidth of the quantization; between 2 and 8, inclusive.
Quantization is called fake since the output is still in floating point.

_inputArg
 inputs:: . 
_outputArg

_outputArg
 outputs:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FakeQuantWithMinMaxArgs(scope:Scope,  inputs:tf.Output, ) -> ( outputs:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FakeQuantWithMinMaxArgs",
        Input: [  inputs, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) outputs )

}

/*
Compute gradients for a FakeQuantWithMinMaxArgs operation.


_inputArg
 gradients:: .  inputs:: . 
_outputArg

_outputArg
 backprops:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FakeQuantWithMinMaxArgsGradient(scope:Scope,  gradients:tf.Output,  inputs:tf.Output, ) -> ( backprops:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FakeQuantWithMinMaxArgsGradient",
        Input: [  gradients,  inputs, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) backprops )

}

/*
Fake-quantize the 'inputs' tensor of type float via global float scalars `min`

and `max` to 'outputs' tensor of same shape as `inputs`.
[min; max] is the clamping range for the 'inputs' data.  Op divides this range
into 255 steps (total of 256 values), then replaces each 'inputs' value with the
closest of the quantized step values.
'num_bits' is the bitwidth of the quantization; between 2 and 8, inclusive.
This operation has a gradient and thus allows for training `min` and `max` values.

_inputArg
 inputs:: .  min:: .  max:: . 
_outputArg

_outputArg
 outputs:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FakeQuantWithMinMaxVars(scope:Scope,  inputs:tf.Output,  min:tf.Output,  max:tf.Output, ) -> ( outputs:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FakeQuantWithMinMaxVars",
        Input: [  inputs,  min,  max, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) outputs )

}

/*
Compute gradients for a FakeQuantWithMinMaxVars operation.


_inputArg
 gradients:: .  inputs:: .  min:: .  max:: . 
_outputArg

_outputArg
 backprops_wrt_input::  backprop_wrt_min::  backprop_wrt_max:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FakeQuantWithMinMaxVarsGradient(scope:Scope,  gradients:tf.Output,  inputs:tf.Output,  min:tf.Output,  max:tf.Output, ) -> ( backprops_wrt_input:tf.Output,  backprop_wrt_min:tf.Output,  backprop_wrt_max:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FakeQuantWithMinMaxVarsGradient",
        Input: [  gradients,  inputs,  min,  max, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) backprops_wrt_input  op.Output(*) backprop_wrt_min  op.Output(*) backprop_wrt_max )

}

/*
Fake-quantize the 'inputs' tensor of type float and one of the shapes: `[d]`,

`[b, d]` `[b, h, w, d]` via per-channel floats `min` and `max` of shape `[d]`
to 'outputs' tensor of same shape as `inputs`.
[min; max] is the clamping range for the 'inputs' data in the corresponding
depth channel.  Op divides this range into 255 steps (total of 256 values), then
replaces each 'inputs' value with the closest of the quantized step values.
'num_bits' is the bitwidth of the quantization; between 2 and 8, inclusive.
This operation has a gradient and thus allows for training `min` and `max` values.

_inputArg
 inputs:: .  min:: .  max:: . 
_outputArg

_outputArg
 outputs:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FakeQuantWithMinMaxVarsPerChannel(scope:Scope,  inputs:tf.Output,  min:tf.Output,  max:tf.Output, ) -> ( outputs:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FakeQuantWithMinMaxVarsPerChannel",
        Input: [  inputs,  min,  max, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) outputs )

}

/*
Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation.


_inputArg
 gradients:: .  inputs:: .  min:: .  max:: . 
_outputArg

_outputArg
 backprops_wrt_input::  backprop_wrt_min::  backprop_wrt_max:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FakeQuantWithMinMaxVarsPerChannelGradient(scope:Scope,  gradients:tf.Output,  inputs:tf.Output,  min:tf.Output,  max:tf.Output, ) -> ( backprops_wrt_input:tf.Output,  backprop_wrt_min:tf.Output,  backprop_wrt_max:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FakeQuantWithMinMaxVarsPerChannelGradient",
        Input: [  gradients,  inputs,  min,  max, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) backprops_wrt_input  op.Output(*) backprop_wrt_min  op.Output(*) backprop_wrt_max )

}

/*
Deprecated. Do not use.


_inputArg
 resource:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FakeQueue(scope:Scope,  resource:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FakeQueue",
        Input: [  resource, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Creates a tensor filled with a scalar value.

This operation creates a tensor of shape `dims` and fills it with `value`.
For example:
```
# Output tensor has shape [2, 3].
fill([2, 3], 9) ==> [[9, 9, 9]
                     [9, 9, 9]]
```

_inputArg
 dims:: .  value:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Fill(scope:Scope,  dims:tf.Output,  value:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Fill",
        Input: [  dims,  value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Creates a dataset containing elements of `input_dataset` matching `predicate`.

The `predicate` function must return a scalar boolean and accept the
following arguments:
* One tensor for each component of an element of `input_dataset`.
* One tensor for each value in `other_arguments`.

_inputArg
 input_dataset:: .  other_arguments:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FilterDataset(scope:Scope,  input_dataset:tf.Output,  other_arguments:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FilterDataset",
        Input: [  input_dataset,  other_arguments, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Creates a dataset that emits the records from one or more binary files.


_inputArg
 filenames:: .  header_bytes:: .  record_bytes:: .  footer_bytes:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FixedLengthRecordDataset(scope:Scope,  filenames:tf.Output,  header_bytes:tf.Output,  record_bytes:tf.Output,  footer_bytes:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FixedLengthRecordDataset",
        Input: [  filenames,  header_bytes,  record_bytes,  footer_bytes, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
A Reader that outputs fixed-length records from a file.


_inputArg
_outputArg

_outputArg
 reader_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FixedLengthRecordReader(scope:Scope, ) -> ( reader_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FixedLengthRecordReader",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) reader_handle )

}

/*
A Reader that outputs fixed-length records from a file.


_inputArg
_outputArg

_outputArg
 reader_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FixedLengthRecordReaderV2(scope:Scope, ) -> ( reader_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FixedLengthRecordReaderV2",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) reader_handle )

}

/*
Generates labels for candidate sampling with a learned unigram distribution.

A unigram sampler could use a fixed unigram distribution read from a
file or passed in as an in-memory array instead of building up the distribution
from data on the fly. There is also an option to skew the distribution by
applying a distortion power to the weights.
The vocabulary file should be in CSV-like format, with the last field
being the weight associated with the word.
For each batch, this op picks a single set of sampled candidate labels.
The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

_inputArg
 true_classes:: . 
_outputArg

_outputArg
 sampled_candidates::  true_expected_count::  sampled_expected_count:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FixedUnigramCandidateSampler(scope:Scope,  true_classes:tf.Output, ) -> ( sampled_candidates:tf.Output,  true_expected_count:tf.Output,  sampled_expected_count:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FixedUnigramCandidateSampler",
        Input: [  true_classes, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) sampled_candidates  op.Output(*) true_expected_count  op.Output(*) sampled_expected_count )

}

/*
Creates a dataset that applies `f` to the outputs of `input_dataset`.

Unlike MapDataset, the `f` in FlatMapDataset is expected to return a
Dataset resource, and FlatMapDataset will flatten successive results
into a single Dataset.

_inputArg
 input_dataset:: .  other_arguments:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FlatMapDataset(scope:Scope,  input_dataset:tf.Output,  other_arguments:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FlatMapDataset",
        Input: [  input_dataset,  other_arguments, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Returns element-wise largest integer not greater than x.


_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Floor(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Floor",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Returns x // y element-wise.

*NOTE*: `FloorDiv` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FloorDiv(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FloorDiv",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Returns element-wise remainder of division. When `x < 0` xor `y < 0` is

true, this follows Python semantics in that the result here is consistent
with a flooring divide. E.g. `floor(x / y) * y + mod(x, y) = x`.
*NOTE*: `FloorMod` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FloorMod(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FloorMod",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Performs fractional average pooling on the input.

Fractional average pooling is similar to Fractional max pooling in the pooling
region generation step. The only difference is that after pooling regions are
generated, a mean operation is performed instead of a max operation in each
pooling region.

_inputArg
 value:: .T 
_outputArg

_outputArg
 output::T  row_pooling_sequence::  col_pooling_sequence:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FractionalAvgPool(scope:Scope,  value:tf.Output, ) -> ( output:tf.Output,  row_pooling_sequence:tf.Output,  col_pooling_sequence:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FractionalAvgPool",
        Input: [  value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output  op.Output(*) row_pooling_sequence  op.Output(*) col_pooling_sequence )

}

/*
Computes gradient of the FractionalAvgPool function.

Unlike FractionalMaxPoolGrad, we don't need to find arg_max for
FractionalAvgPoolGrad, we just need to evenly back-propagate each element of
out_backprop to those indices that form the same pooling cell. Therefore, we
just need to know the shape of original input tensor, instead of the whole
tensor.

_inputArg
 orig_input_tensor_shape:: .  out_backprop:: .T  row_pooling_sequence:: .  col_pooling_sequence:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FractionalAvgPoolGrad(scope:Scope,  orig_input_tensor_shape:tf.Output,  out_backprop:tf.Output,  row_pooling_sequence:tf.Output,  col_pooling_sequence:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FractionalAvgPoolGrad",
        Input: [  orig_input_tensor_shape,  out_backprop,  row_pooling_sequence,  col_pooling_sequence, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Performs fractional max pooling on the input.

Fractional max pooling is slightly different than regular max pooling.  In
regular max pooling, you downsize an input set by taking the maximum value of
smaller N x N subsections of the set (often 2x2), and try to reduce the set by
a factor of N, where N is an integer.  Fractional max pooling, as you might
expect from the word "fractional", means that the overall reduction ratio N
does not have to be an integer.
The sizes of the pooling regions are generated randomly but are fairly uniform.
For example, let's look at the height dimension, and the constraints on the
list of rows that will be pool boundaries.
First we define the following:
1.  input_row_length : the number of rows from the input set
2.  output_row_length : which will be smaller than the input
3.  alpha = input_row_length / output_row_length : our reduction ratio
4.  K = floor(alpha)
5.  row_pooling_sequence : this is the result list of pool boundary rows
Then, row_pooling_sequence should satisfy:
1.  a[0] = 0 : the first value of the sequence is 0
2.  a[end] = input_row_length : the last value of the sequence is the size
3.  K <= (a[i+1] - a[i]) <= K+1 : all intervals are K or K+1 size
4.  length(row_pooling_sequence) = output_row_length+1
For more details on fractional max pooling, see this paper:
[Benjamin Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071)

_inputArg
 value:: .T 
_outputArg

_outputArg
 output::T  row_pooling_sequence::  col_pooling_sequence:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FractionalMaxPool(scope:Scope,  value:tf.Output, ) -> ( output:tf.Output,  row_pooling_sequence:tf.Output,  col_pooling_sequence:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FractionalMaxPool",
        Input: [  value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output  op.Output(*) row_pooling_sequence  op.Output(*) col_pooling_sequence )

}

/*
Computes gradient of the FractionalMaxPool function.


_inputArg
 orig_input:: .T  orig_output:: .T  out_backprop:: .T  row_pooling_sequence:: .  col_pooling_sequence:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FractionalMaxPoolGrad(scope:Scope,  orig_input:tf.Output,  orig_output:tf.Output,  out_backprop:tf.Output,  row_pooling_sequence:tf.Output,  col_pooling_sequence:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FractionalMaxPoolGrad",
        Input: [  orig_input,  orig_output,  out_backprop,  row_pooling_sequence,  col_pooling_sequence, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Batch normalization.

Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.

_inputArg
 x:: .T  scale:: .T  offset:: .T  mean:: .T  variance:: .T 
_outputArg

_outputArg
 y::T  batch_mean::T  batch_variance::T  reserve_space_1::T  reserve_space_2::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FusedBatchNorm(scope:Scope,  x:tf.Output,  scale:tf.Output,  offset:tf.Output,  mean:tf.Output,  variance:tf.Output, ) -> ( y:tf.Output,  batch_mean:tf.Output,  batch_variance:tf.Output,  reserve_space_1:tf.Output,  reserve_space_2:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FusedBatchNorm",
        Input: [  x,  scale,  offset,  mean,  variance, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y  op.Output(*) batch_mean  op.Output(*) batch_variance  op.Output(*) reserve_space_1  op.Output(*) reserve_space_2 )

}

/*
Gradient for batch normalization.

Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.

_inputArg
 y_backprop:: .T  x:: .T  scale:: .T  reserve_space_1:: .T  reserve_space_2:: .T 
_outputArg

_outputArg
 x_backprop::T  scale_backprop::T  offset_backprop::T  reserve_space_3::T  reserve_space_4::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FusedBatchNormGrad(scope:Scope,  y_backprop:tf.Output,  x:tf.Output,  scale:tf.Output,  reserve_space_1:tf.Output,  reserve_space_2:tf.Output, ) -> ( x_backprop:tf.Output,  scale_backprop:tf.Output,  offset_backprop:tf.Output,  reserve_space_3:tf.Output,  reserve_space_4:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FusedBatchNormGrad",
        Input: [  y_backprop,  x,  scale,  reserve_space_1,  reserve_space_2, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) x_backprop  op.Output(*) scale_backprop  op.Output(*) offset_backprop  op.Output(*) reserve_space_3  op.Output(*) reserve_space_4 )

}

/*
Performs a padding as a preprocess during a convolution.

Similar to FusedResizeAndPadConv2d, this op allows for an optimized
implementation where the spatial padding transformation stage is fused with the
im2col lookup, but in this case without the bilinear filtering required for
resizing. Fusing the padding prevents the need to write out the intermediate
results as whole tensors, reducing memory pressure, and we can get some latency
gains by merging the transformation calculations.
The data_format attribute for Conv2D isn't supported by this op, and 'NHWC'
order is used instead.
Internally this op uses a single per-graph scratch buffer, which means that it
will block if multiple versions are being run in parallel. This is because this
operator is primarily an optimization to minimize memory usage.

_inputArg
 input:: .T  paddings:: .  filter:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FusedPadConv2D(scope:Scope,  input:tf.Output,  paddings:tf.Output,  filter:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FusedPadConv2D",
        Input: [  input,  paddings,  filter, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Performs a resize and padding as a preprocess during a convolution.

It's often possible to do spatial transformations more efficiently as part of
the packing stage of a convolution, so this op allows for an optimized
implementation where these stages are fused together. This prevents the need to
write out the intermediate results as whole tensors, reducing memory pressure,
and we can get some latency gains by merging the transformation calculations.
The data_format attribute for Conv2D isn't supported by this op, and defaults to
'NHWC' order.
Internally this op uses a single per-graph scratch buffer, which means that it
will block if multiple versions are being run in parallel. This is because this
operator is primarily an optimization to minimize memory usage.

_inputArg
 input:: .T  size:: .  paddings:: .  filter:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func FusedResizeAndPadConv2D(scope:Scope,  input:tf.Output,  size:tf.Output,  paddings:tf.Output,  filter:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "FusedResizeAndPadConv2D",
        Input: [  input,  size,  paddings,  filter, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Gather slices from `params` according to `indices`.

`indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
Produces an output tensor with shape `indices.shape + params.shape[1:]` where:
```python
    # Scalar indices
    output[:, ..., :] = params[indices, :, ... :]
    # Vector indices
    output[i, :, ..., :] = params[indices[i], :, ... :]
    # Higher rank indices
    output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
```
If `indices` is a permutation and `len(indices) == params.shape[0]` then
this operation will permute `params` accordingly.
`validate_indices`: DEPRECATED. If this operation is assigned to CPU, values in
`indices` are always validated to be within range. If assigned to GPU,
out-of-bound indices result in safe but unspecified behavior, which may include
raising an error.
<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
</div>

_inputArg
 params:: .Tparams  indices:: .Tindices 
_outputArg

_outputArg
 output::Tparams 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Gather(scope:Scope,  params:tf.Output,  indices:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Gather",
        Input: [  params,  indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Gather values or slices from `params` according to `indices`.

`indices` is an integer tensor containing indices into `params`.  The last
dimension of `indices` can be at most the rank of `params`:
    indices.shape[-1] <= params.rank
The last dimension of `indices` corresponds to elements
(if `indices.shape[-1] = params.rank`) or slices
(if `indices.shape[-1] < params.rank`) along dimension `indices.shape[-1]`
of `params`.  The output tensor has shape
    indices.shape[:-1] + params.shape[indices.shape[-1]:]
Some examples below.
Simple indexing into a matrix:
```python
    indices = [[0, 0], [1, 1]]
    params = [['a', 'b'], ['c', 'd']]
    output = ['a', 'd']
```
Slice indexing into a matrix:
```python
    indices = [[1], [0]]
    params = [['a', 'b'], ['c', 'd']]
    output = [['c', 'd'], ['a', 'b']]
```
Indexing into a 3-tensor:
```python
    indices = [[1]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[['a1', 'b1'], ['c1', 'd1']]]
    indices = [[0, 1], [1, 0]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [['c0', 'd0'], ['a1', 'b1']]
    indices = [[0, 0, 1], [1, 0, 1]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = ['b0', 'b1']
```
Batched indexing into a matrix:
```python
    indices = [[[0, 0]], [[0, 1]]]
    params = [['a', 'b'], ['c', 'd']]
    output = [['a'], ['b']]
```
Batched slice indexing into a matrix:
```python
    indices = [[[1]], [[0]]]
    params = [['a', 'b'], ['c', 'd']]
    output = [[['c', 'd']], [['a', 'b']]]
```
Batched indexing into a 3-tensor:
```python
    indices = [[[1]], [[0]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[[['a1', 'b1'], ['c1', 'd1']]],
              [[['a0', 'b0'], ['c0', 'd0']]]]
    indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[['c0', 'd0'], ['a1', 'b1']],
              [['a0', 'b0'], ['c1', 'd1']]]
    indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [['b0', 'b1'], ['d0', 'c1']]
```

_inputArg
 params:: .Tparams  indices:: .Tindices 
_outputArg

_outputArg
 output::Tparams 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func GatherNd(scope:Scope,  params:tf.Output,  indices:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "GatherNd",
        Input: [  params,  indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Store the input tensor in the state of the current session.


_inputArg
 value:: .T 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func GetSessionHandle(scope:Scope,  value:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "GetSessionHandle",
        Input: [  value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Store the input tensor in the state of the current session.


_inputArg
 value:: .T 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func GetSessionHandleV2(scope:Scope,  value:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "GetSessionHandleV2",
        Input: [  value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Get the value of the tensor specified by its handle.


_inputArg
 handle:: . 
_outputArg

_outputArg
 value::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func GetSessionTensor(scope:Scope,  handle:tf.Output, ) -> ( value:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "GetSessionTensor",
        Input: [  handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) value )

}

/*
Returns the truth value of (x > y) element-wise.

*NOTE*: `Greater` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Greater(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Greater",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Returns the truth value of (x >= y) element-wise.

*NOTE*: `GreaterEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func GreaterEqual(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "GreaterEqual",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Creates a dataset that computes a windowed group-by on `input_dataset`.

// TODO(mrry): Support non-int64 keys.

_inputArg
 input_dataset:: .  key_func_other_arguments:: .  reduce_func_other_arguments:: .  window_size:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func GroupByWindowDataset(scope:Scope,  input_dataset:tf.Output,  key_func_other_arguments:tf.Output,  reduce_func_other_arguments:tf.Output,  window_size:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "GroupByWindowDataset",
        Input: [  input_dataset,  key_func_other_arguments,  reduce_func_other_arguments,  window_size, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Convert one or more images from HSV to RGB.

Outputs a tensor of the same shape as the `images` tensor, containing the RGB
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.
See `rgb_to_hsv` for a description of the HSV encoding.

_inputArg
 images:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func HSVToRGB(scope:Scope,  images:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "HSVToRGB",
        Input: [  images, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Creates a non-initialized hash table.

This op creates a hash table, specifying the type of its keys and values.
Before using the table you will have to initialize it.  After initialization the
table will be immutable.

_inputArg
_outputArg

_outputArg
 table_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func HashTable(scope:Scope, ) -> ( table_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "HashTable",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) table_handle )

}

/*
Creates a non-initialized hash table.

This op creates a hash table, specifying the type of its keys and values.
Before using the table you will have to initialize it.  After initialization the
table will be immutable.

_inputArg
_outputArg

_outputArg
 table_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func HashTableV2(scope:Scope, ) -> ( table_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "HashTableV2",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) table_handle )

}

/*
Outputs a `Summary` protocol buffer with a histogram.

The generated
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
has one summary value containing a histogram for `values`.
This op reports an `InvalidArgument` error if any value is not finite.

_inputArg
 tag:: .  values:: .T 
_outputArg

_outputArg
 summary:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func HistogramSummary(scope:Scope,  tag:tf.Output,  values:tf.Output, ) -> ( summary:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "HistogramSummary",
        Input: [  tag,  values, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) summary )

}

/*
Inverse fast Fourier transform.

Computes the inverse 1-dimensional discrete Fourier transform over the
inner-most dimension of `input`.

_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func IFFT(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "IFFT",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Inverse 2D fast Fourier transform.

Computes the inverse 2-dimensional discrete Fourier transform over the
inner-most 2 dimensions of `input`.

_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func IFFT2D(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "IFFT2D",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Inverse 3D fast Fourier transform.

Computes the inverse 3-dimensional discrete Fourier transform over the
inner-most 3 dimensions of `input`.

_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func IFFT3D(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "IFFT3D",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Inverse real-valued fast Fourier transform.

Computes the inverse 1-dimensional discrete Fourier transform of a real-valued
signal over the inner-most dimension of `input`.
The inner-most dimension of `input` is assumed to be the result of `RFFT`: the
`fft_length / 2 + 1` unique components of the DFT of a real-valued signal. If
`fft_length` is not provided, it is computed from the size of the inner-most
dimension of `input` (`fft_length = 2 * (inner - 1)`). If the FFT length used to
compute `input` is odd, it should be provided since it cannot be inferred
properly.

_inputArg
 input:: .  fft_length:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func IRFFT(scope:Scope,  input:tf.Output,  fft_length:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "IRFFT",
        Input: [  input,  fft_length, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Inverse 2D real-valued fast Fourier transform.

Computes the inverse 2-dimensional discrete Fourier transform of a real-valued
signal over the inner-most 2 dimensions of `input`.
The inner-most 2 dimensions of `input` are assumed to be the result of `RFFT2D`:
The inner-most dimension contains the `fft_length / 2 + 1` unique components of
the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
from the size of the inner-most 2 dimensions of `input`. If the FFT length used
to compute `input` is odd, it should be provided since it cannot be inferred
properly.

_inputArg
 input:: .  fft_length:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func IRFFT2D(scope:Scope,  input:tf.Output,  fft_length:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "IRFFT2D",
        Input: [  input,  fft_length, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Inverse 3D real-valued fast Fourier transform.

Computes the inverse 3-dimensional discrete Fourier transform of a real-valued
signal over the inner-most 3 dimensions of `input`.
The inner-most 3 dimensions of `input` are assumed to be the result of `RFFT3D`:
The inner-most dimension contains the `fft_length / 2 + 1` unique components of
the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
from the size of the inner-most 3 dimensions of `input`. If the FFT length used
to compute `input` is odd, it should be provided since it cannot be inferred
properly.

_inputArg
 input:: .  fft_length:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func IRFFT3D(scope:Scope,  input:tf.Output,  fft_length:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "IRFFT3D",
        Input: [  input,  fft_length, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Return a tensor with the same shape and contents as the input tensor or value.


_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Identity(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Identity",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
A Reader that outputs the queued work as both the key and value.

To use, enqueue strings in a Queue.  ReaderRead will take the front
work string and output (work, work).

_inputArg
_outputArg

_outputArg
 reader_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func IdentityReader(scope:Scope, ) -> ( reader_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "IdentityReader",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) reader_handle )

}

/*
A Reader that outputs the queued work as both the key and value.

To use, enqueue strings in a Queue.  ReaderRead will take the front
work string and output (work, work).

_inputArg
_outputArg

_outputArg
 reader_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func IdentityReaderV2(scope:Scope, ) -> ( reader_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "IdentityReaderV2",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) reader_handle )

}

/*
Compute the lower regularized incomplete Gamma function `Q(a, x)`.

The lower regularized incomplete Gamma function is defined as:
\\(P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)\\)
where
\\(gamma(a, x) = int_{0}^{x} t^{a-1} exp(-t) dt\\)
is the lower incomplete Gamma function.
Note, above `Q(a, x)` (`Igammac`) is the upper regularized complete
Gamma function.

_inputArg
 a:: .T  x:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Igamma(scope:Scope,  a:tf.Output,  x:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Igamma",
        Input: [  a,  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Compute the upper regularized incomplete Gamma function `Q(a, x)`.

The upper regularized incomplete Gamma function is defined as:
\\(Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)\\)
where
\\(Gamma(a, x) = int_{x}^{\infty} t^{a-1} exp(-t) dt\\)
is the upper incomplete Gama function.
Note, above `P(a, x)` (`Igamma`) is the lower regularized complete
Gamma function.

_inputArg
 a:: .T  x:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Igammac(scope:Scope,  a:tf.Output,  x:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Igammac",
        Input: [  a,  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Returns the imaginary part of a complex number.

Given a tensor `input` of complex numbers, this operation returns a tensor of
type `float` that is the imaginary part of each element in `input`. All
elements in `input` must be complex numbers of the form \\(a + bj\\), where *a*
is the real part and *b* is the imaginary part returned by this operation.
For example:
```
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.imag(input) ==> [4.75, 5.75]
```

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::Tout 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Imag(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Imag",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Outputs a `Summary` protocol buffer with images.

The summary has up to `max_images` summary values containing images. The
images are built from `tensor` which must be 4-D with shape `[batch_size,
height, width, channels]` and where `channels` can be:
*  1: `tensor` is interpreted as Grayscale.
*  3: `tensor` is interpreted as RGB.
*  4: `tensor` is interpreted as RGBA.
The images have the same number of channels as the input tensor. For float
input, the values are normalized one image at a time to fit in the range
`[0, 255]`.  `uint8` values are unchanged.  The op uses two different
normalization algorithms:
*  If the input values are all positive, they are rescaled so the largest one
   is 255.
*  If any input value is negative, the values are shifted so input value 0.0
   is at 127.  They are then rescaled so that either the smallest value is 0,
   or the largest one is 255.
The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:
*  If `max_images` is 1, the summary value tag is '*tag*/image'.
*  If `max_images` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.
The `bad_color` argument is the color to use in the generated images for
non-finite input values.  It is a `unit8` 1-D tensor of length `channels`.
Each element must be in the range `[0, 255]` (It represents the value of a
pixel in the output image).  Non-finite values in the input tensor are
replaced by this tensor in the output image.  The default value is the color
red.

_inputArg
 tag:: .  tensor:: .T 
_outputArg

_outputArg
 summary:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ImageSummary(scope:Scope,  tag:tf.Output,  tensor:tf.Output, ) -> ( summary:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ImageSummary",
        Input: [  tag,  tensor, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) summary )

}

/*
Returns immutable tensor from memory region.

The current implementation memmaps the tensor from a file.

_inputArg
_outputArg

_outputArg
 tensor::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ImmutableConst(scope:Scope, ) -> ( tensor:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ImmutableConst",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) tensor )

}

/*
Says whether the targets are in the top `K` predictions.

This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
prediction for the target class is among the top `k` predictions among
all predictions for example `i`. Note that the behavior of `InTopK` differs
from the `TopK` op in its handling of ties; if multiple classes have the
same prediction value and straddle the top-`k` boundary, all of those
classes are considered to be in the top `k`.
More formally, let
  \\(predictions_i\\) be the predictions for all classes for example `i`,
  \\(targets_i\\) be the target class for example `i`,
  \\(out_i\\) be the output for example `i`,
$$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$

_inputArg
 predictions:: .  targets:: .T 
_outputArg

_outputArg
 precision:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func InTopK(scope:Scope,  predictions:tf.Output,  targets:tf.Output, ) -> ( precision:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "InTopK",
        Input: [  predictions,  targets, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) precision )

}

/*
Table initializer that takes two tensors for keys and values respectively.


_inputArg
 table_handle:: .  keys:: .Tkey  values:: .Tval 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func InitializeTable(scope:Scope,  table_handle:tf.Output,  keys:tf.Output,  values:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "InitializeTable",
        Input: [  table_handle,  keys,  values, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Initializes a table from a text file.

It inserts one key-value pair into the table for each line of the file.
The key and value is extracted from the whole line content, elements from the
split line based on `delimiter` or the line number (starting from zero).
Where to extract the key and value from a line is specified by `key_index` and
`value_index`.
- A value of -1 means use the line number(starting from zero), expects `int64`.
- A value of -2 means use the whole line content, expects `string`.
- A value >= 0 means use the index (starting at zero) of the split line based
  on `delimiter`.

_inputArg
 table_handle:: .  filename:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func InitializeTableFromTextFile(scope:Scope,  table_handle:tf.Output,  filename:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "InitializeTableFromTextFile",
        Input: [  table_handle,  filename, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Initializes a table from a text file.

It inserts one key-value pair into the table for each line of the file.
The key and value is extracted from the whole line content, elements from the
split line based on `delimiter` or the line number (starting from zero).
Where to extract the key and value from a line is specified by `key_index` and
`value_index`.
- A value of -1 means use the line number(starting from zero), expects `int64`.
- A value of -2 means use the whole line content, expects `string`.
- A value >= 0 means use the index (starting at zero) of the split line based
  on `delimiter`.

_inputArg
 table_handle:: .  filename:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func InitializeTableFromTextFileV2(scope:Scope,  table_handle:tf.Output,  filename:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "InitializeTableFromTextFileV2",
        Input: [  table_handle,  filename, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Table initializer that takes two tensors for keys and values respectively.


_inputArg
 table_handle:: .  keys:: .Tkey  values:: .Tval 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func InitializeTableV2(scope:Scope,  table_handle:tf.Output,  keys:tf.Output,  values:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "InitializeTableV2",
        Input: [  table_handle,  keys,  values, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Computes the reciprocal of x element-wise.

I.e., \\(y = 1 / x\\).

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Inv(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Inv",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Computes the gradient for the inverse of `x` wrt its input.

Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
is the corresponding input gradient.

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func InvGrad(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "InvGrad",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Computes the inverse permutation of a tensor.

This operation computes the inverse of an index permutation. It takes a 1-D
integer tensor `x`, which represents the indices of a zero-based array, and
swaps each value with its index position. In other words, for an output tensor
`y` and an input tensor `x`, this operation computes the following:
`y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`
The values must include 0. There can be no duplicate values or negative values.
For example:
```
# tensor `x` is [3, 4, 0, 2, 1]
invert_permutation(x) ==> [2, 4, 3, 0, 1]
```

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func InvertPermutation(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "InvertPermutation",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Returns which elements of x are finite.

@compatibility(numpy)
Equivalent to np.isfinite
@end_compatibility

_inputArg
 x:: .T 
_outputArg

_outputArg
 y:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func IsFinite(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "IsFinite",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Returns which elements of x are Inf.

@compatibility(numpy)
Equivalent to np.isinf
@end_compatibility

_inputArg
 x:: .T 
_outputArg

_outputArg
 y:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func IsInf(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "IsInf",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Returns which elements of x are NaN.

@compatibility(numpy)
Equivalent to np.isnan
@end_compatibility

_inputArg
 x:: .T 
_outputArg

_outputArg
 y:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func IsNan(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "IsNan",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Checks whether a tensor has been initialized.

Outputs boolean scalar indicating whether the tensor has been initialized.

_inputArg
 ref:: .dtype 
_outputArg

_outputArg
 is_initialized:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func IsVariableInitialized(scope:Scope,  ref:tf.Output, ) -> ( is_initialized:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "IsVariableInitialized",
        Input: [  ref, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) is_initialized )

}

/*
A container for an iterator resource.


_inputArg
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Iterator(scope:Scope, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Iterator",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Releases any resources used by the given iterator.


_inputArg
 iterator:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func IteratorDispose(scope:Scope,  iterator:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "IteratorDispose",
        Input: [  iterator, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Gets the next output from the given iterator.


_inputArg
 iterator:: . 
_outputArg

_outputArg
 components:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func IteratorGetNext(scope:Scope,  iterator:tf.Output, ) -> ( components:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "IteratorGetNext",
        Input: [  iterator, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) components )

}

/*
L2 Loss.

Computes half the L2 norm of a tensor without the `sqrt`:
    output = sum(t ** 2) / 2

_inputArg
 t:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func L2Loss(scope:Scope,  t:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "L2Loss",
        Input: [  t, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Local Response Normalization.

The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
dimension), and each vector is normalized independently.  Within a given vector,
each component is divided by the weighted, squared sum of inputs within
`depth_radius`.  In detail,
    sqr_sum[a, b, c, d] =
        sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
    output = input / (bias + alpha * sqr_sum) ** beta
For details, see [Krizhevsky et al., ImageNet classification with deep
convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LRN(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LRN",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Gradients for Local Response Normalization.


_inputArg
 input_grads:: .T  input_image:: .T  output_image:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LRNGrad(scope:Scope,  input_grads:tf.Output,  input_image:tf.Output,  output_image:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LRNGrad",
        Input: [  input_grads,  input_image,  output_image, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Generates labels for candidate sampling with a learned unigram distribution.

See explanations of candidate sampling and the data formats at
go/candidate-sampling.
For each batch, this op picks a single set of sampled candidate labels.
The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

_inputArg
 true_classes:: . 
_outputArg

_outputArg
 sampled_candidates::  true_expected_count::  sampled_expected_count:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LearnedUnigramCandidateSampler(scope:Scope,  true_classes:tf.Output, ) -> ( sampled_candidates:tf.Output,  true_expected_count:tf.Output,  sampled_expected_count:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LearnedUnigramCandidateSampler",
        Input: [  true_classes, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) sampled_candidates  op.Output(*) true_expected_count  op.Output(*) sampled_expected_count )

}

/*
Returns the truth value of (x < y) element-wise.

*NOTE*: `Less` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Less(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Less",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Returns the truth value of (x <= y) element-wise.

*NOTE*: `LessEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LessEqual(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LessEqual",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Computes the log of the absolute value of `Gamma(x)` element-wise.


_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Lgamma(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Lgamma",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Generates values in an interval.

A sequence of `num` evenly-spaced values are generated beginning at `start`.
If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
so that the last one is exactly `stop`.
For example:
```
tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
```

_inputArg
 start:: .T  stop:: .T  num:: .Tidx 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LinSpace(scope:Scope,  start:tf.Output,  stop:tf.Output,  num:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LinSpace",
        Input: [  start,  stop,  num, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the difference between two lists of numbers or strings.

Given a list `x` and a list `y`, this operation returns a list `out` that
represents all values that are in `x` but not in `y`. The returned list `out`
is sorted in the same order that the numbers appear in `x` (duplicates are
preserved). This operation also returns a list `idx` that represents the
position of each `out` element in `x`. In other words:
`out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`
For example, given this input:
```
x = [1, 2, 3, 4, 5, 6]
y = [1, 3, 5]
```
This operation would return:
```
out ==> [2, 4, 6]
idx ==> [1, 3, 5]
```

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 out::T  idx::out_idx 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ListDiff(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( out:tf.Output,  idx:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ListDiff",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out  op.Output(*) idx )

}

/*
Computes natural logarithm of x element-wise.

I.e., \\(y = \log_e x\\).

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Log(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Log",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Computes natural logarithm of (1 + x) element-wise.

I.e., \\(y = \log_e (1 + x)\\).

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Log1p(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Log1p",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Computes log softmax activations.

For each batch `i` and class `j` we have
    logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))

_inputArg
 logits:: .T 
_outputArg

_outputArg
 logsoftmax::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LogSoftmax(scope:Scope,  logits:tf.Output, ) -> ( logsoftmax:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LogSoftmax",
        Input: [  logits, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) logsoftmax )

}

/*
Generates labels for candidate sampling with a log-uniform distribution.

See explanations of candidate sampling and the data formats at
go/candidate-sampling.
For each batch, this op picks a single set of sampled candidate labels.
The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

_inputArg
 true_classes:: . 
_outputArg

_outputArg
 sampled_candidates::  true_expected_count::  sampled_expected_count:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LogUniformCandidateSampler(scope:Scope,  true_classes:tf.Output, ) -> ( sampled_candidates:tf.Output,  true_expected_count:tf.Output,  sampled_expected_count:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LogUniformCandidateSampler",
        Input: [  true_classes, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) sampled_candidates  op.Output(*) true_expected_count  op.Output(*) sampled_expected_count )

}

/*
Returns the truth value of x AND y element-wise.

*NOTE*: `LogicalAnd` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .  y:: . 
_outputArg

_outputArg
 z:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LogicalAnd(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LogicalAnd",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Returns the truth value of NOT x element-wise.


_inputArg
 x:: . 
_outputArg

_outputArg
 y:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LogicalNot(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LogicalNot",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Returns the truth value of x OR y element-wise.

*NOTE*: `LogicalOr` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .  y:: . 
_outputArg

_outputArg
 z:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LogicalOr(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LogicalOr",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Outputs all keys and values in the table.


_inputArg
 table_handle:: . 
_outputArg

_outputArg
 keys::Tkeys  values::Tvalues 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LookupTableExport(scope:Scope,  table_handle:tf.Output, ) -> ( keys:tf.Output,  values:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LookupTableExport",
        Input: [  table_handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) keys  op.Output(*) values )

}

/*
Outputs all keys and values in the table.


_inputArg
 table_handle:: . 
_outputArg

_outputArg
 keys::Tkeys  values::Tvalues 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LookupTableExportV2(scope:Scope,  table_handle:tf.Output, ) -> ( keys:tf.Output,  values:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LookupTableExportV2",
        Input: [  table_handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) keys  op.Output(*) values )

}

/*
Looks up keys in a table, outputs the corresponding values.

The tensor `keys` must of the same type as the keys of the table.
The output `values` is of the type of the table values.
The scalar `default_value` is the value output for keys not present in the
table. It must also be of the same type as the table values.

_inputArg
 table_handle:: .  keys:: .Tin  default_value:: .Tout 
_outputArg

_outputArg
 values::Tout 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LookupTableFind(scope:Scope,  table_handle:tf.Output,  keys:tf.Output,  default_value:tf.Output, ) -> ( values:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LookupTableFind",
        Input: [  table_handle,  keys,  default_value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) values )

}

/*
Looks up keys in a table, outputs the corresponding values.

The tensor `keys` must of the same type as the keys of the table.
The output `values` is of the type of the table values.
The scalar `default_value` is the value output for keys not present in the
table. It must also be of the same type as the table values.

_inputArg
 table_handle:: .  keys:: .Tin  default_value:: .Tout 
_outputArg

_outputArg
 values::Tout 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LookupTableFindV2(scope:Scope,  table_handle:tf.Output,  keys:tf.Output,  default_value:tf.Output, ) -> ( values:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LookupTableFindV2",
        Input: [  table_handle,  keys,  default_value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) values )

}

/*
Replaces the contents of the table with the specified keys and values.

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

_inputArg
 table_handle:: .  keys:: .Tin  values:: .Tout 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LookupTableImport(scope:Scope,  table_handle:tf.Output,  keys:tf.Output,  values:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LookupTableImport",
        Input: [  table_handle,  keys,  values, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Replaces the contents of the table with the specified keys and values.

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

_inputArg
 table_handle:: .  keys:: .Tin  values:: .Tout 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LookupTableImportV2(scope:Scope,  table_handle:tf.Output,  keys:tf.Output,  values:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LookupTableImportV2",
        Input: [  table_handle,  keys,  values, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Updates the table to associates keys with values.

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

_inputArg
 table_handle:: .  keys:: .Tin  values:: .Tout 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LookupTableInsert(scope:Scope,  table_handle:tf.Output,  keys:tf.Output,  values:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LookupTableInsert",
        Input: [  table_handle,  keys,  values, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Updates the table to associates keys with values.

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

_inputArg
 table_handle:: .  keys:: .Tin  values:: .Tout 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LookupTableInsertV2(scope:Scope,  table_handle:tf.Output,  keys:tf.Output,  values:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LookupTableInsertV2",
        Input: [  table_handle,  keys,  values, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Computes the number of elements in the given table.


_inputArg
 table_handle:: . 
_outputArg

_outputArg
 size:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LookupTableSize(scope:Scope,  table_handle:tf.Output, ) -> ( size:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LookupTableSize",
        Input: [  table_handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) size )

}

/*
Computes the number of elements in the given table.


_inputArg
 table_handle:: . 
_outputArg

_outputArg
 size:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LookupTableSizeV2(scope:Scope,  table_handle:tf.Output, ) -> ( size:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LookupTableSizeV2",
        Input: [  table_handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) size )

}

/*
Forwards the input to the output.

This operator represents the loop termination condition used by the
"pivot" switches of a loop.

_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func LoopCond(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "LoopCond",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Makes a new iterator from the given `dataset` and stores it in `iterator`.

This operation may be executed multiple times. Each execution will reset the
iterator in `iterator` to the first element of `dataset`.

_inputArg
 dataset:: .  iterator:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MakeIterator(scope:Scope,  dataset:tf.Output,  iterator:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MakeIterator",
        Input: [  dataset,  iterator, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Creates a dataset that applies `f` to the outputs of `input_dataset`.


_inputArg
 input_dataset:: .  other_arguments:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MapDataset(scope:Scope,  input_dataset:tf.Output,  other_arguments:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MapDataset",
        Input: [  input_dataset,  other_arguments, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Multiply the matrix "a" by the matrix "b".

The inputs must be two-dimensional matrices and the inner dimension of
"a" (after being transposed if transpose_a is true) must match the
outer dimension of "b" (after being transposed if transposed_b is
true).
*Note*: The default kernel implementation for MatMul on GPUs uses
cublas.

_inputArg
 a:: .T  b:: .T 
_outputArg

_outputArg
 product::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MatMul(scope:Scope,  a:tf.Output,  b:tf.Output, ) -> ( product:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MatMul",
        Input: [  a,  b, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) product )

}

/*
Returns the set of files matching one or more glob patterns.

Note that this routine only supports wildcard characters in the
basename portion of the pattern, not in the directory portion.

_inputArg
 pattern:: . 
_outputArg

_outputArg
 filenames:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MatchingFiles(scope:Scope,  pattern:tf.Output, ) -> ( filenames:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MatchingFiles",
        Input: [  pattern, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) filenames )

}

/*
Copy a tensor setting everything outside a central band in each innermost matrix

to zero.
The `band` part is computed as follows:
Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
tensor with the same shape where
`band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.
The indicator function
`in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) &&
                 (num_upper < 0 || (n-m) <= num_upper)`.
For example:
```
# if 'input' is [[ 0,  1,  2, 3]
                 [-1,  0,  1, 2]
                 [-2, -1,  0, 1]
                 [-3, -2, -1, 0]],
tf.matrix_band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
                                       [-1,  0,  1, 2]
                                       [ 0, -1,  0, 1]
                                       [ 0,  0, -1, 0]],
tf.matrix_band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
                                      [-1,  0,  1, 0]
                                      [-2, -1,  0, 1]
                                      [ 0, -2, -1, 0]]
```
Useful special cases:
```
 tf.matrix_band_part(input, 0, -1) ==> Upper triangular part.
 tf.matrix_band_part(input, -1, 0) ==> Lower triangular part.
 tf.matrix_band_part(input, 0, 0) ==> Diagonal.
```

_inputArg
 input:: .T  num_lower:: .  num_upper:: . 
_outputArg

_outputArg
 band::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MatrixBandPart(scope:Scope,  input:tf.Output,  num_lower:tf.Output,  num_upper:tf.Output, ) -> ( band:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MatrixBandPart",
        Input: [  input,  num_lower,  num_upper, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) band )

}

/*
Computes the determinant of one ore more square matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor containing the determinants
for all input submatrices `[..., :, :]`.

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MatrixDeterminant(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MatrixDeterminant",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Returns a batched diagonal tensor with a given batched diagonal values.

Given a `diagonal`, this operation returns a tensor with the `diagonal` and
everything else padded with zeros. The diagonal is computed as follows:
Assume `diagonal` has `k` dimensions `[I, J, K, ..., N]`, then the output is a
tensor of rank `k+1` with dimensions [I, J, K, ..., N, N]` where:
`output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n]`.
For example:
```
# 'diagonal' is [[1, 2, 3, 4], [5, 6, 7, 8]]
and diagonal.shape = (2, 4)
tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0]
                                     [0, 2, 0, 0]
                                     [0, 0, 3, 0]
                                     [0, 0, 0, 4]],
                                    [[5, 0, 0, 0]
                                     [0, 6, 0, 0]
                                     [0, 0, 7, 0]
                                     [0, 0, 0, 8]]]
which has shape (2, 4, 4)
```

_inputArg
 diagonal:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MatrixDiag(scope:Scope,  diagonal:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MatrixDiag",
        Input: [  diagonal, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Returns the batched diagonal part of a batched tensor.

This operation returns a tensor with the `diagonal` part
of the batched `input`. The `diagonal` part is computed as follows:
Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
tensor of rank `k - 1` with dimensions `[I, J, K, ..., min(M, N)]` where:
`diagonal[i, j, k, ..., n] = input[i, j, k, ..., n, n]`.
The input must be at least a matrix.
For example:
```
# 'input' is [[[1, 0, 0, 0]
               [0, 2, 0, 0]
               [0, 0, 3, 0]
               [0, 0, 0, 4]],
              [[5, 0, 0, 0]
               [0, 6, 0, 0]
               [0, 0, 7, 0]
               [0, 0, 0, 8]]]
and input.shape = (2, 4, 4)
tf.matrix_diag_part(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]
which has shape (2, 4)
```

_inputArg
 input:: .T 
_outputArg

_outputArg
 diagonal::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MatrixDiagPart(scope:Scope,  input:tf.Output, ) -> ( diagonal:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MatrixDiagPart",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) diagonal )

}

/*
Computes the inverse of one or more square invertible matrices or their

adjoints (conjugate transposes).
The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor of the same shape as the input
containing the inverse for all input submatrices `[..., :, :]`.
The op uses LU decomposition with partial pivoting to compute the inverses.
If a matrix is not invertible there is no guarantee what the op does. It
may detect the condition and raise an exception or it may simply return a
garbage result.

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MatrixInverse(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MatrixInverse",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Returns a batched matrix tensor with new batched diagonal values.

Given `input` and `diagonal`, this operation returns a tensor with the
same shape and values as `input`, except for the main diagonal of the
innermost matrices.  These will be overwritten by the values in `diagonal`.
The output is computed as follows:
Assume `input` has `k+1` dimensions `[I, J, K, ..., M, N]` and `diagonal` has
`k` dimensions `[I, J, K, ..., min(M, N)]`.  Then the output is a
tensor of rank `k+1` with dimensions `[I, J, K, ..., M, N]` where:
  * `output[i, j, k, ..., m, n] = diagonal[i, j, k, ..., n]` for `m == n`.
  * `output[i, j, k, ..., m, n] = input[i, j, k, ..., m, n]` for `m != n`.

_inputArg
 input:: .T  diagonal:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MatrixSetDiag(scope:Scope,  input:tf.Output,  diagonal:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MatrixSetDiag",
        Input: [  input,  diagonal, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Solves systems of linear equations.

`Matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. `Rhs` is a tensor of shape `[..., M, K]`. The `output` is
a tensor shape `[..., M, K]`.  If `adjoint` is `False` then each output matrix
satisfies `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
If `adjoint` is `True` then each output matrix satisfies
`adjoint(matrix[..., :, :]) * output[..., :, :] = rhs[..., :, :]`.

_inputArg
 matrix:: .T  rhs:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MatrixSolve(scope:Scope,  matrix:tf.Output,  rhs:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MatrixSolve",
        Input: [  matrix,  rhs, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Solves one or more linear least-squares problems.

`matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
form matrices of size `[M, N]`. Rhs is a tensor of shape `[..., M, K]`.
The output is a tensor shape `[..., N, K]` where each output matrix solves
each of the equations matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]
in the least squares sense.
matrix and right-hand sides in the batch:
`matrix`=\\(A \in \Re^{m \times n}\\),
`rhs`=\\(B  \in \Re^{m \times k}\\),
`output`=\\(X  \in \Re^{n \times k}\\),
`l2_regularizer`=\\(\lambda\\).
If `fast` is `True`, then the solution is computed by solving the normal
equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
\\(X = (A^T A + \lambda I)^{-1} A^T B\\), which solves the least-squares
problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k} } ||A Z - B||_F^2 +
\lambda ||Z||_F^2\\). If \\(m \lt n\\) then `output` is computed as
\\(X = A^T (A A^T + \lambda I)^{-1} B\\), which (for \\(\lambda = 0\\)) is the
minimum-norm solution to the under-determined linear system, i.e.
\\(X = \mathrm{argmin}_{Z \in \Re^{n \times k} } ||Z||_F^2 \\), subject to
\\(A Z = B\\). Notice that the fast path is only numerically stable when
\\(A\\) is numerically full rank and has a condition number
\\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach} } }\\) or\\(\lambda\\) is
sufficiently large.
If `fast` is `False` an algorithm based on the numerically robust complete
orthogonal decomposition is used. This computes the minimum-norm
least-squares solution, even when \\(A\\) is rank deficient. This path is
typically 6-7 times slower than the fast path. If `fast` is `False` then
`l2_regularizer` is ignored.

_inputArg
 matrix:: .T  rhs:: .T  l2_regularizer:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MatrixSolveLs(scope:Scope,  matrix:tf.Output,  rhs:tf.Output,  l2_regularizer:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MatrixSolveLs",
        Input: [  matrix,  rhs,  l2_regularizer, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Solves systems of linear equations with upper or lower triangular matrices by

backsubstitution.
`matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
square matrices. If `lower` is `True` then the strictly upper triangular part
of each inner-most matrix is assumed to be zero and not accessed.
If `lower` is False then the strictly lower triangular part of each inner-most
matrix is assumed to be zero and not accessed.
`rhs` is a tensor of shape `[..., M, K]`.
The output is a tensor of shape `[..., M, K]`. If `adjoint` is
`True` then the innermost matrices in output` satisfy matrix equations
`matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
If `adjoint` is `False` then the strictly then the  innermost matrices in
`output` satisfy matrix equations
`adjoint(matrix[..., i, k]) * output[..., k, j] = rhs[..., i, j]`.

_inputArg
 matrix:: .T  rhs:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MatrixTriangularSolve(scope:Scope,  matrix:tf.Output,  rhs:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MatrixTriangularSolve",
        Input: [  matrix,  rhs, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the maximum of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

_inputArg
 input:: .T  reduction_indices:: .Tidx 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Max(scope:Scope,  input:tf.Output,  reduction_indices:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Max",
        Input: [  input,  reduction_indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Performs max pooling on the input.


_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MaxPool(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MaxPool",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Performs 3D max pooling on the input.


_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MaxPool3D(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MaxPool3D",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes gradients of max pooling function.


_inputArg
 orig_input:: .TInput  orig_output:: .TInput  grad:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MaxPool3DGrad(scope:Scope,  orig_input:tf.Output,  orig_output:tf.Output,  grad:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MaxPool3DGrad",
        Input: [  orig_input,  orig_output,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes second-order gradients of the maxpooling function.


_inputArg
 orig_input:: .T  orig_output:: .T  grad:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MaxPool3DGradGrad(scope:Scope,  orig_input:tf.Output,  orig_output:tf.Output,  grad:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MaxPool3DGradGrad",
        Input: [  orig_input,  orig_output,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes gradients of the maxpooling function.


_inputArg
 orig_input:: .T  orig_output:: .T  grad:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MaxPoolGrad(scope:Scope,  orig_input:tf.Output,  orig_output:tf.Output,  grad:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MaxPoolGrad",
        Input: [  orig_input,  orig_output,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes second-order gradients of the maxpooling function.


_inputArg
 orig_input:: .T  orig_output:: .T  grad:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MaxPoolGradGrad(scope:Scope,  orig_input:tf.Output,  orig_output:tf.Output,  grad:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MaxPoolGradGrad",
        Input: [  orig_input,  orig_output,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes second-order gradients of the maxpooling function.


_inputArg
 input:: .T  grad:: .T  argmax:: .Targmax 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MaxPoolGradGradWithArgmax(scope:Scope,  input:tf.Output,  grad:tf.Output,  argmax:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MaxPoolGradGradWithArgmax",
        Input: [  input,  grad,  argmax, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes gradients of the maxpooling function.


_inputArg
 input:: .T  grad:: .T  argmax:: .Targmax 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MaxPoolGradWithArgmax(scope:Scope,  input:tf.Output,  grad:tf.Output,  argmax:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MaxPoolGradWithArgmax",
        Input: [  input,  grad,  argmax, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Performs max pooling on the input and outputs both max values and indices.

The indices in `argmax` are flattened, so that a maximum value at position
`[b, y, x, c]` becomes flattened index
`((b * height + y) * width + x) * channels + c`.

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T  argmax::Targmax 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MaxPoolWithArgmax(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output,  argmax:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MaxPoolWithArgmax",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output  op.Output(*) argmax )

}

/*
Returns the max of x and y (i.e. x > y ? x : y) element-wise.

*NOTE*: `Maximum` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Maximum(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Maximum",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Computes the mean of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

_inputArg
 input:: .T  reduction_indices:: .Tidx 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Mean(scope:Scope,  input:tf.Output,  reduction_indices:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Mean",
        Input: [  input,  reduction_indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Forwards the value of an available tensor from `inputs` to `output`.

`Merge` waits for at least one of the tensors in `inputs` to become available.
It is usually combined with `Switch` to implement branching.
`Merge` forwards the first tensor to become available to `output`, and sets
`value_index` to its index in `inputs`.

_inputArg
 inputs:: .T 
_outputArg

_outputArg
 output::T  value_index:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Merge(scope:Scope,  inputs:tf.Output, ) -> ( output:tf.Output,  value_index:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Merge",
        Input: [  inputs, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output  op.Output(*) value_index )

}

/*
Merges summaries.

This op creates a
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
protocol buffer that contains the union of all the values in the input
summaries.
When the Op is run, it reports an `InvalidArgument` error if multiple values
in the summaries to merge use the same tag.

_inputArg
 inputs:: . 
_outputArg

_outputArg
 summary:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MergeSummary(scope:Scope,  inputs:tf.Output, ) -> ( summary:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MergeSummary",
        Input: [  inputs, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) summary )

}

/*
V2 format specific: merges the metadata files of sharded checkpoints.  The

result is one logical checkpoint, with one physical metadata file and renamed
data files.
Intended for "grouping" multiple checkpoints in a sharded checkpoint setup.
If delete_old_dirs is true, attempts to delete recursively the dirname of each
path in the input checkpoint_prefixes.  This is useful when those paths are non
user-facing temporary locations.

_inputArg
 checkpoint_prefixes:: .  destination_prefix:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MergeV2Checkpoints(scope:Scope,  checkpoint_prefixes:tf.Output,  destination_prefix:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MergeV2Checkpoints",
        Input: [  checkpoint_prefixes,  destination_prefix, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Transforms a spectrogram into a form that's useful for speech recognition.

Mel Frequency Cepstral Coefficients are a way of representing audio data that's
been effective as an input feature for machine learning. They are created by
taking the spectrum of a spectrogram (a 'cepstrum'), and discarding some of the
higher frequencies that are less significant to the human ear. They have a long
history in the speech recognition world, and https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
is a good resource to learn more.

_inputArg
 spectrogram:: .  sample_rate:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Mfcc(scope:Scope,  spectrogram:tf.Output,  sample_rate:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Mfcc",
        Input: [  spectrogram,  sample_rate, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the minimum of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

_inputArg
 input:: .T  reduction_indices:: .Tidx 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Min(scope:Scope,  input:tf.Output,  reduction_indices:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Min",
        Input: [  input,  reduction_indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Returns the min of x and y (i.e. x < y ? x : y) element-wise.

*NOTE*: `Minimum` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Minimum(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Minimum",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Pads a tensor with mirrored values.

This operation pads a `input` with mirrored values according to the `paddings`
you specify. `paddings` is an integer tensor with shape `[n, 2]`, where n is
the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
how many values to add before the contents of `input` in that dimension, and
`paddings[D, 1]` indicates how many values to add after the contents of `input`
in that dimension. Both `paddings[D, 0]` and `paddings[D, 1]` must be no greater
than `input.dim_size(D)` (or `input.dim_size(D) - 1`) if `copy_border` is true
(if false, respectively).
The padded size of each dimension D of the output is:
`paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
For example:
```
# 't' is [[1, 2, 3], [4, 5, 6]].
# 'paddings' is [[1, 1]], [2, 2]].
# 'mode' is SYMMETRIC.
# rank of 't' is 2.
pad(t, paddings) ==> [[2, 1, 1, 2, 3, 3, 2]
                      [2, 1, 1, 2, 3, 3, 2]
                      [5, 4, 4, 5, 6, 6, 5]
                      [5, 4, 4, 5, 6, 6, 5]]
```

_inputArg
 input:: .T  paddings:: .Tpaddings 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MirrorPad(scope:Scope,  input:tf.Output,  paddings:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MirrorPad",
        Input: [  input,  paddings, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Gradient op for `MirrorPad` op. This op folds a mirror-padded tensor.

This operation folds the padded areas of `input` by `MirrorPad` according to the
`paddings` you specify. `paddings` must be the same as `paddings` argument
given to the corresponding `MirrorPad` op.
The folded size of each dimension D of the output is:
`input.dim_size(D) - paddings(D, 0) - paddings(D, 1)`
For example:
```
# 't' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
# 'paddings' is [[0, 1]], [0, 1]].
# 'mode' is SYMMETRIC.
# rank of 't' is 2.
pad(t, paddings) ==> [[ 1,  5]
                      [11, 28]]
```

_inputArg
 input:: .T  paddings:: .Tpaddings 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MirrorPadGrad(scope:Scope,  input:tf.Output,  paddings:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MirrorPadGrad",
        Input: [  input,  paddings, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Returns element-wise remainder of division. This emulates C semantics in that

the result here is consistent with a truncating divide. E.g. `truncate(x / y) *
y + truncate_mod(x, y) = x`.
*NOTE*: `Mod` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Mod(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Mod",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Returns x * y element-wise.

*NOTE*: `Mul` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Mul(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Mul",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Draws samples from a multinomial distribution.


_inputArg
 logits:: .T  num_samples:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Multinomial(scope:Scope,  logits:tf.Output,  num_samples:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Multinomial",
        Input: [  logits,  num_samples, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Creates an empty hash table that uses tensors as the backing store.

It uses "open addressing" with quadratic reprobing to resolve
collisions.
This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

_inputArg
 empty_key:: .key_dtype 
_outputArg

_outputArg
 table_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MutableDenseHashTable(scope:Scope,  empty_key:tf.Output, ) -> ( table_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MutableDenseHashTable",
        Input: [  empty_key, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) table_handle )

}

/*
Creates an empty hash table that uses tensors as the backing store.

It uses "open addressing" with quadratic reprobing to resolve
collisions.
This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

_inputArg
 empty_key:: .key_dtype 
_outputArg

_outputArg
 table_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MutableDenseHashTableV2(scope:Scope,  empty_key:tf.Output, ) -> ( table_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MutableDenseHashTableV2",
        Input: [  empty_key, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) table_handle )

}

/*
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

_inputArg
_outputArg

_outputArg
 table_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MutableHashTable(scope:Scope, ) -> ( table_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MutableHashTable",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) table_handle )

}

/*
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a vector. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

_inputArg
_outputArg

_outputArg
 table_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MutableHashTableOfTensors(scope:Scope, ) -> ( table_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MutableHashTableOfTensors",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) table_handle )

}

/*
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a vector. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

_inputArg
_outputArg

_outputArg
 table_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MutableHashTableOfTensorsV2(scope:Scope, ) -> ( table_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MutableHashTableOfTensorsV2",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) table_handle )

}

/*
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

_inputArg
_outputArg

_outputArg
 table_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func MutableHashTableV2(scope:Scope, ) -> ( table_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "MutableHashTableV2",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) table_handle )

}

/*
Computes numerical negative value element-wise.

I.e., \\(y = -x\\).

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Neg(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Neg",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Training via negative sampling.


_inputArg
 w_in:: .  w_out:: .  examples:: .  labels:: .  lr:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func NegTrain(scope:Scope,  w_in:tf.Output,  w_out:tf.Output,  examples:tf.Output,  labels:tf.Output,  lr:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "NegTrain",
        Input: [  w_in,  w_out,  examples,  labels,  lr, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Makes its input available to the next iteration.


_inputArg
 data:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func NextIteration(scope:Scope,  data:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "NextIteration",
        Input: [  data, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Does nothing. Only useful as a placeholder for control edges.


_inputArg
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func NoOp(scope:Scope, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "NoOp",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Greedily selects a subset of bounding boxes in descending order of score,

pruning away boxes that have high intersection-over-union (IOU) overlap
with previously selected boxes.  Bounding boxes are supplied as
[y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
diagonal pair of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
is agnostic to where the origin is in the coordinate system.  Note that this
algorithm is invariant to orthogonal transformations and translations
of the coordinate system; thus translating or reflections of the coordinate
system result in the same boxes being selected by the algorithm.
The output of this operation is a set of integers indexing into the input
collection of bounding boxes representing the selected boxes.  The bounding
box coordinates corresponding to the selected indices can then be obtained
using the `tf.gather operation`.  For example:
  selected_indices = tf.image.non_max_suppression(
      boxes, scores, max_output_size, iou_threshold)
  selected_boxes = tf.gather(boxes, selected_indices)

_inputArg
 boxes:: .  scores:: .  max_output_size:: . 
_outputArg

_outputArg
 selected_indices:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func NonMaxSuppression(scope:Scope,  boxes:tf.Output,  scores:tf.Output,  max_output_size:tf.Output, ) -> ( selected_indices:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "NonMaxSuppression",
        Input: [  boxes,  scores,  max_output_size, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) selected_indices )

}

/*
Greedily selects a subset of bounding boxes in descending order of score,

pruning away boxes that have high intersection-over-union (IOU) overlap
with previously selected boxes.  Bounding boxes are supplied as
[y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
diagonal pair of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
is agnostic to where the origin is in the coordinate system.  Note that this
algorithm is invariant to orthogonal transformations and translations
of the coordinate system; thus translating or reflections of the coordinate
system result in the same boxes being selected by the algorithm.
The output of this operation is a set of integers indexing into the input
collection of bounding boxes representing the selected boxes.  The bounding
box coordinates corresponding to the selected indices can then be obtained
using the `tf.gather operation`.  For example:
  selected_indices = tf.image.non_max_suppression_v2(
      boxes, scores, max_output_size, iou_threshold)
  selected_boxes = tf.gather(boxes, selected_indices)

_inputArg
 boxes:: .  scores:: .  max_output_size:: .  iou_threshold:: . 
_outputArg

_outputArg
 selected_indices:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func NonMaxSuppressionV2(scope:Scope,  boxes:tf.Output,  scores:tf.Output,  max_output_size:tf.Output,  iou_threshold:tf.Output, ) -> ( selected_indices:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "NonMaxSuppressionV2",
        Input: [  boxes,  scores,  max_output_size,  iou_threshold, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) selected_indices )

}

/*
Returns the truth value of (x != y) element-wise.

*NOTE*: `NotEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func NotEqual(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "NotEqual",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Returns a one-hot tensor.

The locations represented by indices in `indices` take value `on_value`,
while all other locations take value `off_value`.
If the input `indices` is rank `N`, the output will have rank `N+1`,
The new axis is created at dimension `axis` (default: the new axis is
appended at the end).
If `indices` is a scalar the output shape will be a vector of length `depth`.
If `indices` is a vector of length `features`, the output shape will be:
```
  features x depth if axis == -1
  depth x features if axis == 0
```
If `indices` is a matrix (batch) with shape `[batch, features]`,
the output shape will be:
```
  batch x features x depth if axis == -1
  batch x depth x features if axis == 1
  depth x batch x features if axis == 0
```
Examples
=========
Suppose that
```
  indices = [0, 2, -1, 1]
  depth = 3
  on_value = 5.0
  off_value = 0.0
  axis = -1
```
Then output is `[4 x 3]`:
    ```output =
      [5.0 0.0 0.0]  // one_hot(0)
      [0.0 0.0 5.0]  // one_hot(2)
      [0.0 0.0 0.0]  // one_hot(-1)
      [0.0 5.0 0.0]  // one_hot(1)
    ```
Suppose that
```
  indices = [0, 2, -1, 1]
  depth = 3
  on_value = 0.0
  off_value = 3.0
  axis = 0
```
Then output is `[3 x 4]`:
    ```output =
      [0.0 3.0 3.0 3.0]
      [3.0 3.0 3.0 0.0]
      [3.0 3.0 3.0 3.0]
      [3.0 0.0 3.0 3.0]
    //  ^                one_hot(0)
    //      ^            one_hot(2)
    //          ^        one_hot(-1)
    //              ^    one_hot(1)
    ```
Suppose that
```
  indices = [[0, 2], [1, -1]]
  depth = 3
  on_value = 1.0
  off_value = 0.0
  axis = -1
```
Then output is `[2 x 2 x 3]`:
    ```output =
      [
        [1.0, 0.0, 0.0]  // one_hot(0)
        [0.0, 0.0, 1.0]  // one_hot(2)
      ][
        [0.0, 1.0, 0.0]  // one_hot(1)
        [0.0, 0.0, 0.0]  // one_hot(-1)
      ]```

_inputArg
 indices:: .TI  depth:: .  on_value:: .T  off_value:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func OneHot(scope:Scope,  indices:tf.Output,  depth:tf.Output,  on_value:tf.Output,  off_value:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "OneHot",
        Input: [  indices,  depth,  on_value,  off_value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Makes a "one-shot" iterator that can be iterated only once.

A one-shot iterator bundles the logic for defining the dataset and
the state of the iterator in a single op, which allows simple input
pipelines to be defined without an additional initialization
("MakeIterator") step.
One-shot iterators have the following limitations:
* They do not support parameterization: all logic for creating the underlying
  dataset must be bundled in the `dataset_factory` function.
* They are not resettable. Once a one-shot iterator reaches the end of its
  underlying dataset, subsequent "IteratorGetNext" operations on that
  iterator will always produce an `OutOfRange` error.
For greater flexibility, use "Iterator" and "MakeIterator" to define
an iterator using an arbitrary subgraph, which may capture tensors
(including fed values) as parameters, and which may be reset multiple
times by rerunning "MakeIterator".

_inputArg
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func OneShotIterator(scope:Scope, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "OneShotIterator",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Returns a tensor of ones with the same shape and type as x.


_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func OnesLike(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "OnesLike",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor.

Packs the `N` tensors in `values` into a tensor with rank one higher than each
tensor in `values`, by packing them along the `axis` dimension.
Given a list of tensors of shape `(A, B, C)`;
if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
Etc.
For example:
```
# 'x' is [1, 4]
# 'y' is [2, 5]
# 'z' is [3, 6]
pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
```
This is the opposite of `unpack`.

_inputArg
 values:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Pack(scope:Scope,  values:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Pack",
        Input: [  values, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Pads a tensor with zeros.

This operation pads a `input` with zeros according to the `paddings` you
specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
how many zeros to add before the contents of `input` in that dimension, and
`paddings[D, 1]` indicates how many zeros to add after the contents of `input`
in that dimension.
The padded size of each dimension D of the output is:
`paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
For example:
```
# 't' is [[1, 1], [2, 2]]
# 'paddings' is [[1, 1], [2, 2]]
# rank of 't' is 2
pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
                      [0, 0, 1, 1, 0, 0]
                      [0, 0, 2, 2, 0, 0]
                      [0, 0, 0, 0, 0, 0]]
```

_inputArg
 input:: .T  paddings:: .Tpaddings 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Pad(scope:Scope,  input:tf.Output,  paddings:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Pad",
        Input: [  input,  paddings, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Creates a dataset that batches and pads `batch_size` elements from the input.


_inputArg
 input_dataset:: .  batch_size:: .  padded_shapes:: .  padding_values:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func PaddedBatchDataset(scope:Scope,  input_dataset:tf.Output,  batch_size:tf.Output,  padded_shapes:tf.Output,  padding_values:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "PaddedBatchDataset",
        Input: [  input_dataset,  batch_size,  padded_shapes,  padding_values, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
A queue that produces elements in first-in first-out order.

Variable-size shapes are allowed by setting the corresponding shape dimensions
to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
size of any given element in the minibatch.  See below for details.

_inputArg
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func PaddingFIFOQueue(scope:Scope, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "PaddingFIFOQueue",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
A queue that produces elements in first-in first-out order.

Variable-size shapes are allowed by setting the corresponding shape dimensions
to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
size of any given element in the minibatch.  See below for details.

_inputArg
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func PaddingFIFOQueueV2(scope:Scope, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "PaddingFIFOQueueV2",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Concatenates a list of `N` tensors along the first dimension.

The input tensors are all required to have size 1 in the first dimension.
For example:
```
# 'x' is [[1, 4]]
# 'y' is [[2, 5]]
# 'z' is [[3, 6]]
parallel_concat([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
```
The difference between concat and parallel_concat is that concat requires all
of the inputs be computed before the operation will begin but doesn't require
that the input shapes be known during graph construction.  Parallel concat
will copy pieces of the input into the output as they become available, in
some situations this can provide a performance benefit.

_inputArg
 values:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ParallelConcat(scope:Scope,  values:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ParallelConcat",
        Input: [  values, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Creates a dataset that applies `f` to the outputs of `input_dataset`.

Unlike a "MapDataset", which applies `f` sequentially, this dataset uses
up to `num_threads` threads to process elements from `input_dataset`
in parallel.

_inputArg
 input_dataset:: .  other_arguments:: .  num_threads:: .  output_buffer_size:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ParallelMapDataset(scope:Scope,  input_dataset:tf.Output,  other_arguments:tf.Output,  num_threads:tf.Output,  output_buffer_size:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ParallelMapDataset",
        Input: [  input_dataset,  other_arguments,  num_threads,  output_buffer_size, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Outputs random values from a normal distribution. The parameters may each be a

scalar which applies to the entire output, or a vector of length shape[0] which
stores the parameters for each batch.

_inputArg
 shape:: .T  means:: .dtype  stdevs:: .dtype  minvals:: .dtype  maxvals:: .dtype 
_outputArg

_outputArg
 output::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ParameterizedTruncatedNormal(scope:Scope,  shape:tf.Output,  means:tf.Output,  stdevs:tf.Output,  minvals:tf.Output,  maxvals:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ParameterizedTruncatedNormal",
        Input: [  shape,  means,  stdevs,  minvals,  maxvals, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Transforms a vector of brain.Example protos (as strings) into typed tensors.


_inputArg
 serialized:: .  names:: .  sparse_keys:: .  dense_keys:: .  dense_defaults:: . 
_outputArg

_outputArg
 sparse_indices::  sparse_values::  sparse_shapes::  dense_values:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ParseExample(scope:Scope,  serialized:tf.Output,  names:tf.Output,  sparse_keys:tf.Output,  dense_keys:tf.Output,  dense_defaults:tf.Output, ) -> ( sparse_indices:tf.Output,  sparse_values:tf.Output,  sparse_shapes:tf.Output,  dense_values:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ParseExample",
        Input: [  serialized,  names,  sparse_keys,  dense_keys,  dense_defaults, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) sparse_indices  op.Output(*) sparse_values  op.Output(*) sparse_shapes  op.Output(*) dense_values )

}

/*
Transforms a scalar brain.SequenceExample proto (as strings) into typed tensors.


_inputArg
 serialized:: .  feature_list_dense_missing_assumed_empty:: .  context_sparse_keys:: .  context_dense_keys:: .  feature_list_sparse_keys:: .  feature_list_dense_keys:: .  context_dense_defaults:: .  debug_name:: . 
_outputArg

_outputArg
 context_sparse_indices::  context_sparse_values::  context_sparse_shapes::  context_dense_values::  feature_list_sparse_indices::  feature_list_sparse_values::  feature_list_sparse_shapes::  feature_list_dense_values:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ParseSingleSequenceExample(scope:Scope,  serialized:tf.Output,  feature_list_dense_missing_assumed_empty:tf.Output,  context_sparse_keys:tf.Output,  context_dense_keys:tf.Output,  feature_list_sparse_keys:tf.Output,  feature_list_dense_keys:tf.Output,  context_dense_defaults:tf.Output,  debug_name:tf.Output, ) -> ( context_sparse_indices:tf.Output,  context_sparse_values:tf.Output,  context_sparse_shapes:tf.Output,  context_dense_values:tf.Output,  feature_list_sparse_indices:tf.Output,  feature_list_sparse_values:tf.Output,  feature_list_sparse_shapes:tf.Output,  feature_list_dense_values:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ParseSingleSequenceExample",
        Input: [  serialized,  feature_list_dense_missing_assumed_empty,  context_sparse_keys,  context_dense_keys,  feature_list_sparse_keys,  feature_list_dense_keys,  context_dense_defaults,  debug_name, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) context_sparse_indices  op.Output(*) context_sparse_values  op.Output(*) context_sparse_shapes  op.Output(*) context_dense_values  op.Output(*) feature_list_sparse_indices  op.Output(*) feature_list_sparse_values  op.Output(*) feature_list_sparse_shapes  op.Output(*) feature_list_dense_values )

}

/*
Transforms a serialized tensorflow.TensorProto proto into a Tensor.


_inputArg
 serialized:: . 
_outputArg

_outputArg
 output::out_type 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ParseTensor(scope:Scope,  serialized:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ParseTensor",
        Input: [  serialized, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
A placeholder op for a value that will be fed into the computation.

N.B. This operation will fail with an error if it is executed. It is
intended as a way to represent a value that will always be fed, and to
provide attrs that enable the fed value to be checked at runtime.

_inputArg
_outputArg

_outputArg
 output::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Placeholder(scope:Scope, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Placeholder",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
A placeholder op for a value that will be fed into the computation.

N.B. This operation will fail with an error if it is executed. It is
intended as a way to represent a value that will always be fed, and to
provide attrs that enable the fed value to be checked at runtime.

_inputArg
_outputArg

_outputArg
 output::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func PlaceholderV2(scope:Scope, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "PlaceholderV2",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
A placeholder op that passes through `input` when its output is not fed.


_inputArg
 input:: .dtype 
_outputArg

_outputArg
 output::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func PlaceholderWithDefault(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "PlaceholderWithDefault",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Compute the polygamma function \\(\psi^{(n)}(x)\\).

The polygamma function is defined as:
\\(\psi^{(n)}(x) = \frac{d^n}{dx^n} \psi(x)\\)
where \\(\psi(x)\\) is the digamma function.

_inputArg
 a:: .T  x:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Polygamma(scope:Scope,  a:tf.Output,  x:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Polygamma",
        Input: [  a,  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Computes the power of one value to another.

Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
corresponding elements in `x` and `y`. For example:
```
# tensor 'x' is [[2, 2]], [3, 3]]
# tensor 'y' is [[8, 16], [2, 3]]
tf.pow(x, y) ==> [[256, 65536], [9, 27]]
```

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Pow(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Pow",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
An identity op that triggers an error if a gradient is requested.

When executed in a graph, this op outputs its input tensor as-is.
When building ops to compute gradients, the TensorFlow gradient system
will return an error when trying to lookup the gradient of this op,
because no gradient must ever be registered for this function.  This
op exists to prevent subtle bugs from silently returning unimplemented
gradients in some corner cases.

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func PreventGradient(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "PreventGradient",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Prints a list of tensors.

Passes `input` through to `output` and prints `data` when evaluating.

_inputArg
 input:: .T  data:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Print(scope:Scope,  input:tf.Output,  data:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Print",
        Input: [  input,  data, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
A queue that produces elements sorted by the first component value.

Note that the PriorityQueue requires the first component of any element
to be a scalar int64, in addition to the other elements declared by
component_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
entry in their input (resp. output) lists.

_inputArg
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func PriorityQueue(scope:Scope, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "PriorityQueue",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
A queue that produces elements sorted by the first component value.

Note that the PriorityQueue requires the first component of any element
to be a scalar int64, in addition to the other elements declared by
component_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
entry in their input (resp. output) lists.

_inputArg
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func PriorityQueueV2(scope:Scope, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "PriorityQueueV2",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Computes the product of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

_inputArg
 input:: .T  reduction_indices:: .Tidx 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Prod(scope:Scope,  input:tf.Output,  reduction_indices:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Prod",
        Input: [  input,  reduction_indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Invokes a python function to compute func(input)->output.

This operation is considered stateful. For a stateless version, see
PyFuncStateless.

_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func PyFunc(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "PyFunc",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
A stateless version of PyFunc.


_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func PyFuncStateless(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "PyFuncStateless",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the QR decompositions of one or more matrices.

Computes the QR decomposition of each inner matrix in `tensor` such that
`tensor[..., :, :] = q[..., :, :] * r[..., :,:])`
```python
# a is a tensor.
# q is a tensor of orthonormal matrices.
# r is a tensor of upper triangular matrices.
q, r = qr(a)
q_full, r_full = qr(a, full_matrices=True)
```

_inputArg
 input:: .T 
_outputArg

_outputArg
 q::T  r::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Qr(scope:Scope,  input:tf.Output, ) -> ( q:tf.Output,  r:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Qr",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) q  op.Output(*) r )

}

/*
Use QuantizeAndDequantizeV2 instead.


_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizeAndDequantize(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizeAndDequantize",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Quantizes then dequantizes a tensor.

This op simulates the precision loss from the quantized forward pass by:
1. Quantizing the tensor to fixed point numbers, which should match the target
   quantization method when it is used in inference.
2. Dequantizing it back to floating point numbers for the following ops, most
   likely matmul.
There are different ways to quantize. This version does not use the full range
of the output type, choosing to elide the lowest possible value for symmetry
(e.g., output range is -127 to 127, not -128 to 127 for signed 8 bit
quantization), so that 0.0 maps to 0.
To perform this op, we first find the range of values in our tensor. The range
we use is always centered on 0, so we find m such that
1. m = max(abs(input_min), abs(input_max)) if range_given is true,
2. m = max(abs(min_elem(input)), abs(max_elem(input))) otherwise.
Our input tensor range is then [-m, m].
Next, we choose our fixed-point quantization buckets, [min_fixed, max_fixed].
If signed_input is true, this is
  [min_fixed, max_fixed ] =
      [-(1 << (num_bits - 1) - 1), (1 << (num_bits - 1)) - 1].
Otherwise, if signed_input is false, the fixed-point range is
  [min_fixed, max_fixed] = [0, (1 << num_bits) - 1].
From this we compute our scaling factor, s:
  s = (max_fixed - min_fixed) / (2 * m).
Now we can quantize and dequantize the elements of our tensor.  An element e
is transformed into e':
  e' = (e * s).round_to_nearest() / s.
Note that we have a different number of buckets in the signed vs. unsigned
cases.  For example, if num_bits == 8, we get 254 buckets in the signed case
vs. 255 in the unsigned case.
For example, suppose num_bits = 8 and m = 1.  Then
  [min_fixed, max_fixed] = [-127, 127], and
  s = (127 + 127) / 2 = 127.
Given the vector {-1, -0.5, 0, 0.3}, this is quantized to
{-127, -63, 0, 38}, and dequantized to {-1, -63.0/127, 0, 38.0/127}.

_inputArg
 input:: .T  input_min:: .T  input_max:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizeAndDequantizeV2(scope:Scope,  input:tf.Output,  input_min:tf.Output,  input_max:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizeAndDequantizeV2",
        Input: [  input,  input_min,  input_max, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Convert the quantized 'input' tensor into a lower-precision 'output', using the

actual distribution of the values to maximize the usage of the lower bit depth
and adjusting the output min and max ranges accordingly.
[input_min, input_max] are scalar floats that specify the range for the float
interpretation of the 'input' data. For example, if input_min is -1.0f and
input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.
This operator tries to squeeze as much precision as possible into an output with
a lower bit depth by calculating the actual min and max values found in the
data. For example, maybe that quint16 input has no values lower than 16,384 and
none higher than 49,152. That means only half the range is actually needed, all
the float interpretations are between -0.5f and 0.5f, so if we want to compress
the data into a quint8 output, we can use that range rather than the theoretical
-1.0f to 1.0f that is suggested by the input min and max.
In practice, this is most useful for taking output from operations like
QuantizedMatMul that can produce higher bit-depth outputs than their inputs and
may have large potential output ranges, but in practice have a distribution of
input values that only uses a small fraction of the possible range. By feeding
that output into this operator, we can reduce it from 32 bits down to 8 with
minimal loss of accuracy.

_inputArg
 input:: .Tinput  input_min:: .  input_max:: . 
_outputArg

_outputArg
 output::out_type  output_min::  output_max:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizeDownAndShrinkRange(scope:Scope,  input:tf.Output,  input_min:tf.Output,  input_max:tf.Output, ) -> ( output:tf.Output,  output_min:tf.Output,  output_max:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizeDownAndShrinkRange",
        Input: [  input,  input_min,  input_max, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output  op.Output(*) output_min  op.Output(*) output_max )

}

/*
Quantize the 'input' tensor of type float to 'output' tensor of type 'T'.

[min_range, max_range] are scalar floats that specify the range for
the 'input' data. The 'mode' attribute controls exactly which calculations are
used to convert the float values to their quantized equivalents.
In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:
```
out[i] = (in[i] - min_range) * range(T) / (max_range - min_range)
if T == qint8, out[i] -= (range(T) + 1) / 2.0
```
here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`
*MIN_COMBINED Mode Example*
Assume the input is type float and has a possible range of [0.0, 6.0] and the
output type is quint8 ([0, 255]). The min_range and max_range values should be
specified as 0.0 and 6.0. Quantizing from float to quint8 will multiply each
value of the input by 255/6 and cast to quint8.
If the output type was qint8 ([-128, 127]), the operation will additionally
subtract each value by 128 prior to casting, so that the range of values aligns
with the range of qint8.
If the mode is 'MIN_FIRST', then this approach is used:
```
number_of_steps = 1 << (# of bits in T)
range_adjust = number_of_steps / (number_of_steps - 1)
range = (range_max - range_min) * range_adjust
range_scale = number_of_steps / range
quantized = round(input * range_scale) - round(range_min * range_scale) +
  numeric_limits<T>::min()
quantized = max(quantized, numeric_limits<T>::min())
quantized = min(quantized, numeric_limits<T>::max())
```
The biggest difference between this and MIN_COMBINED is that the minimum range
is rounded first, before it's subtracted from the rounded value. With
MIN_COMBINED, a small bias is introduced where repeated iterations of quantizing
and dequantizing will introduce a larger and larger error.
One thing to watch out for is that the operator may choose to adjust the
requested minimum and maximum values slightly during the quantization process,
so you should always use the output ports as the range for further calculations.
For example, if the requested minimum and maximum values are close to equal,
they will be separated by a small epsilon value to prevent ill-formed quantized
buffers from being created. Otherwise, you can end up with buffers where all the
quantized values map to the same float value, which causes problems for
operations that have to perform further calculations on them.

_inputArg
 input:: .  min_range:: .  max_range:: . 
_outputArg

_outputArg
 output::T  output_min::  output_max:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizeV2(scope:Scope,  input:tf.Output,  min_range:tf.Output,  max_range:tf.Output, ) -> ( output:tf.Output,  output_min:tf.Output,  output_max:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizeV2",
        Input: [  input,  min_range,  max_range, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output  op.Output(*) output_min  op.Output(*) output_max )

}

/*
Returns x + y element-wise, working on quantized buffers.


_inputArg
 x:: .T1  y:: .T2  min_x:: .  max_x:: .  min_y:: .  max_y:: . 
_outputArg

_outputArg
 z::Toutput  min_z::  max_z:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizedAdd(scope:Scope,  x:tf.Output,  y:tf.Output,  min_x:tf.Output,  max_x:tf.Output,  min_y:tf.Output,  max_y:tf.Output, ) -> ( z:tf.Output,  min_z:tf.Output,  max_z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizedAdd",
        Input: [  x,  y,  min_x,  max_x,  min_y,  max_y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z  op.Output(*) min_z  op.Output(*) max_z )

}

/*
Produces the average pool of the input tensor for quantized types.


_inputArg
 input:: .T  min_input:: .  max_input:: . 
_outputArg

_outputArg
 output::T  min_output::  max_output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizedAvgPool(scope:Scope,  input:tf.Output,  min_input:tf.Output,  max_input:tf.Output, ) -> ( output:tf.Output,  min_output:tf.Output,  max_output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizedAvgPool",
        Input: [  input,  min_input,  max_input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output  op.Output(*) min_output  op.Output(*) max_output )

}

/*
Quantized Batch normalization.

This op is deprecated and will be removed in the future. Prefer
`tf.nn.batch_normalization`.

_inputArg
 t:: .Tinput  t_min:: .  t_max:: .  m:: .Tinput  m_min:: .  m_max:: .  v:: .Tinput  v_min:: .  v_max:: .  beta:: .Tinput  beta_min:: .  beta_max:: .  gamma:: .Tinput  gamma_min:: .  gamma_max:: . 
_outputArg

_outputArg
 result::out_type  result_min::  result_max:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizedBatchNormWithGlobalNormalization(scope:Scope,  t:tf.Output,  t_min:tf.Output,  t_max:tf.Output,  m:tf.Output,  m_min:tf.Output,  m_max:tf.Output,  v:tf.Output,  v_min:tf.Output,  v_max:tf.Output,  beta:tf.Output,  beta_min:tf.Output,  beta_max:tf.Output,  gamma:tf.Output,  gamma_min:tf.Output,  gamma_max:tf.Output, ) -> ( result:tf.Output,  result_min:tf.Output,  result_max:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizedBatchNormWithGlobalNormalization",
        Input: [  t,  t_min,  t_max,  m,  m_min,  m_max,  v,  v_min,  v_max,  beta,  beta_min,  beta_max,  gamma,  gamma_min,  gamma_max, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) result  op.Output(*) result_min  op.Output(*) result_max )

}

/*
Adds Tensor 'bias' to Tensor 'input' for Quantized types.

Broadcasts the values of bias on dimensions 0..N-2 of 'input'.

_inputArg
 input:: .T1  bias:: .T2  min_input:: .  max_input:: .  min_bias:: .  max_bias:: . 
_outputArg

_outputArg
 output::out_type  min_out::  max_out:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizedBiasAdd(scope:Scope,  input:tf.Output,  bias:tf.Output,  min_input:tf.Output,  max_input:tf.Output,  min_bias:tf.Output,  max_bias:tf.Output, ) -> ( output:tf.Output,  min_out:tf.Output,  max_out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizedBiasAdd",
        Input: [  input,  bias,  min_input,  max_input,  min_bias,  max_bias, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output  op.Output(*) min_out  op.Output(*) max_out )

}

/*
Concatenates quantized tensors along one dimension.


_inputArg
 concat_dim:: .  values:: .T  input_mins:: .  input_maxes:: . 
_outputArg

_outputArg
 output::T  output_min::  output_max:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizedConcat(scope:Scope,  concat_dim:tf.Output,  values:tf.Output,  input_mins:tf.Output,  input_maxes:tf.Output, ) -> ( output:tf.Output,  output_min:tf.Output,  output_max:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizedConcat",
        Input: [  concat_dim,  values,  input_mins,  input_maxes, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output  op.Output(*) output_min  op.Output(*) output_max )

}

/*
Computes a 2D convolution given quantized 4D input and filter tensors.

The inputs are quantized tensors where the lowest value represents the real
number of the associated minimum, and the highest represents the maximum.
This means that you can only interpret the quantized output in the same way, by
taking the returned minimum and maximum values into account.

_inputArg
 input:: .Tinput  filter:: .Tfilter  min_input:: .  max_input:: .  min_filter:: .  max_filter:: . 
_outputArg

_outputArg
 output::out_type  min_output::  max_output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizedConv2D(scope:Scope,  input:tf.Output,  filter:tf.Output,  min_input:tf.Output,  max_input:tf.Output,  min_filter:tf.Output,  max_filter:tf.Output, ) -> ( output:tf.Output,  min_output:tf.Output,  max_output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizedConv2D",
        Input: [  input,  filter,  min_input,  max_input,  min_filter,  max_filter, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output  op.Output(*) min_output  op.Output(*) max_output )

}

/*
Quantized Instance normalization.


_inputArg
 x:: .T  x_min:: .  x_max:: . 
_outputArg

_outputArg
 y::T  y_min::  y_max:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizedInstanceNorm(scope:Scope,  x:tf.Output,  x_min:tf.Output,  x_max:tf.Output, ) -> ( y:tf.Output,  y_min:tf.Output,  y_max:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizedInstanceNorm",
        Input: [  x,  x_min,  x_max, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y  op.Output(*) y_min  op.Output(*) y_max )

}

/*
Perform a quantized matrix multiplication of  `a` by the matrix `b`.

The inputs must be two-dimensional matrices and the inner dimension of
`a` (after being transposed if `transpose_a` is non-zero) must match the
outer dimension of `b` (after being transposed if `transposed_b` is
non-zero).

_inputArg
 a:: .T1  b:: .T2  min_a:: .  max_a:: .  min_b:: .  max_b:: . 
_outputArg

_outputArg
 out::Toutput  min_out::  max_out:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizedMatMul(scope:Scope,  a:tf.Output,  b:tf.Output,  min_a:tf.Output,  max_a:tf.Output,  min_b:tf.Output,  max_b:tf.Output, ) -> ( out:tf.Output,  min_out:tf.Output,  max_out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizedMatMul",
        Input: [  a,  b,  min_a,  max_a,  min_b,  max_b, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out  op.Output(*) min_out  op.Output(*) max_out )

}

/*
Produces the max pool of the input tensor for quantized types.


_inputArg
 input:: .T  min_input:: .  max_input:: . 
_outputArg

_outputArg
 output::T  min_output::  max_output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizedMaxPool(scope:Scope,  input:tf.Output,  min_input:tf.Output,  max_input:tf.Output, ) -> ( output:tf.Output,  min_output:tf.Output,  max_output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizedMaxPool",
        Input: [  input,  min_input,  max_input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output  op.Output(*) min_output  op.Output(*) max_output )

}

/*
Returns x * y element-wise, working on quantized buffers.


_inputArg
 x:: .T1  y:: .T2  min_x:: .  max_x:: .  min_y:: .  max_y:: . 
_outputArg

_outputArg
 z::Toutput  min_z::  max_z:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizedMul(scope:Scope,  x:tf.Output,  y:tf.Output,  min_x:tf.Output,  max_x:tf.Output,  min_y:tf.Output,  max_y:tf.Output, ) -> ( z:tf.Output,  min_z:tf.Output,  max_z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizedMul",
        Input: [  x,  y,  min_x,  max_x,  min_y,  max_y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z  op.Output(*) min_z  op.Output(*) max_z )

}

/*
Computes Quantized Rectified Linear: `max(features, 0)`


_inputArg
 features:: .Tinput  min_features:: .  max_features:: . 
_outputArg

_outputArg
 activations::out_type  min_activations::  max_activations:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizedRelu(scope:Scope,  features:tf.Output,  min_features:tf.Output,  max_features:tf.Output, ) -> ( activations:tf.Output,  min_activations:tf.Output,  max_activations:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizedRelu",
        Input: [  features,  min_features,  max_features, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) activations  op.Output(*) min_activations  op.Output(*) max_activations )

}

/*
Computes Quantized Rectified Linear 6: `min(max(features, 0), 6)`


_inputArg
 features:: .Tinput  min_features:: .  max_features:: . 
_outputArg

_outputArg
 activations::out_type  min_activations::  max_activations:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizedRelu6(scope:Scope,  features:tf.Output,  min_features:tf.Output,  max_features:tf.Output, ) -> ( activations:tf.Output,  min_activations:tf.Output,  max_activations:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizedRelu6",
        Input: [  features,  min_features,  max_features, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) activations  op.Output(*) min_activations  op.Output(*) max_activations )

}

/*
Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`


_inputArg
 features:: .Tinput  max_value:: .  min_features:: .  max_features:: . 
_outputArg

_outputArg
 activations::out_type  min_activations::  max_activations:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizedReluX(scope:Scope,  features:tf.Output,  max_value:tf.Output,  min_features:tf.Output,  max_features:tf.Output, ) -> ( activations:tf.Output,  min_activations:tf.Output,  max_activations:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizedReluX",
        Input: [  features,  max_value,  min_features,  max_features, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) activations  op.Output(*) min_activations  op.Output(*) max_activations )

}

/*
Reshapes a quantized tensor as per the Reshape op.

```

_inputArg
 tensor:: .T  shape:: .Tshape  input_min:: .  input_max:: . 
_outputArg

_outputArg
 output::T  output_min::  output_max:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QuantizedReshape(scope:Scope,  tensor:tf.Output,  shape:tf.Output,  input_min:tf.Output,  input_max:tf.Output, ) -> ( output:tf.Output,  output_min:tf.Output,  output_max:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QuantizedReshape",
        Input: [  tensor,  shape,  input_min,  input_max, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output  op.Output(*) output_min  op.Output(*) output_max )

}

/*
Closes the given queue.

This operation signals that no more elements will be enqueued in the
given queue. Subsequent Enqueue(Many) operations will fail.
Subsequent Dequeue(Many) operations will continue to succeed if
sufficient elements remain in the queue. Subsequent Dequeue(Many)
operations that would block will fail immediately.

_inputArg
 handle:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QueueClose(scope:Scope,  handle:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QueueClose",
        Input: [  handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Closes the given queue.

This operation signals that no more elements will be enqueued in the
given queue. Subsequent Enqueue(Many) operations will fail.
Subsequent Dequeue(Many) operations will continue to succeed if
sufficient elements remain in the queue. Subsequent Dequeue(Many)
operations that would block will fail immediately.

_inputArg
 handle:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QueueCloseV2(scope:Scope,  handle:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QueueCloseV2",
        Input: [  handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Dequeues a tuple of one or more tensors from the given queue.

This operation has k outputs, where k is the number of components
in the tuples stored in the given queue, and output i is the ith
component of the dequeued tuple.
N.B. If the queue is empty, this operation will block until an element
has been dequeued (or 'timeout_ms' elapses, if specified).

_inputArg
 handle:: . 
_outputArg

_outputArg
 components:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QueueDequeue(scope:Scope,  handle:tf.Output, ) -> ( components:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QueueDequeue",
        Input: [  handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) components )

}

/*
Dequeues `n` tuples of one or more tensors from the given queue.

If the queue is closed and there are fewer than `n` elements, then an
OutOfRange error is returned.
This operation concatenates queue-element component tensors along the
0th dimension to make a single component tensor.  All of the components
in the dequeued tuple will have size `n` in the 0th dimension.
This operation has `k` outputs, where `k` is the number of components in
the tuples stored in the given queue, and output `i` is the ith
component of the dequeued tuple.
N.B. If the queue is empty, this operation will block until `n` elements
have been dequeued (or 'timeout_ms' elapses, if specified).

_inputArg
 handle:: .  n:: . 
_outputArg

_outputArg
 components:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QueueDequeueMany(scope:Scope,  handle:tf.Output,  n:tf.Output, ) -> ( components:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QueueDequeueMany",
        Input: [  handle,  n, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) components )

}

/*
Dequeues `n` tuples of one or more tensors from the given queue.

If the queue is closed and there are fewer than `n` elements, then an
OutOfRange error is returned.
This operation concatenates queue-element component tensors along the
0th dimension to make a single component tensor.  All of the components
in the dequeued tuple will have size `n` in the 0th dimension.
This operation has `k` outputs, where `k` is the number of components in
the tuples stored in the given queue, and output `i` is the ith
component of the dequeued tuple.
N.B. If the queue is empty, this operation will block until `n` elements
have been dequeued (or 'timeout_ms' elapses, if specified).

_inputArg
 handle:: .  n:: . 
_outputArg

_outputArg
 components:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QueueDequeueManyV2(scope:Scope,  handle:tf.Output,  n:tf.Output, ) -> ( components:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QueueDequeueManyV2",
        Input: [  handle,  n, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) components )

}

/*
Dequeues `n` tuples of one or more tensors from the given queue.

This operation is not supported by all queues.  If a queue does not support
DequeueUpTo, then an Unimplemented error is returned.
If the queue is closed and there are more than 0 but less than `n`
elements remaining, then instead of returning an OutOfRange error like
QueueDequeueMany, less than `n` elements are returned immediately.  If
the queue is closed and there are 0 elements left in the queue, then
an OutOfRange error is returned just like in QueueDequeueMany.
Otherwise the behavior is identical to QueueDequeueMany:
This operation concatenates queue-element component tensors along the
0th dimension to make a single component tensor.  All of the components
in the dequeued tuple will have size `n` in the 0th dimension.
This operation has k outputs, where `k` is the number of components in
the tuples stored in the given queue, and output `i` is the ith
component of the dequeued tuple.

_inputArg
 handle:: .  n:: . 
_outputArg

_outputArg
 components:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QueueDequeueUpTo(scope:Scope,  handle:tf.Output,  n:tf.Output, ) -> ( components:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QueueDequeueUpTo",
        Input: [  handle,  n, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) components )

}

/*
Dequeues `n` tuples of one or more tensors from the given queue.

This operation is not supported by all queues.  If a queue does not support
DequeueUpTo, then an Unimplemented error is returned.
If the queue is closed and there are more than 0 but less than `n`
elements remaining, then instead of returning an OutOfRange error like
QueueDequeueMany, less than `n` elements are returned immediately.  If
the queue is closed and there are 0 elements left in the queue, then
an OutOfRange error is returned just like in QueueDequeueMany.
Otherwise the behavior is identical to QueueDequeueMany:
This operation concatenates queue-element component tensors along the
0th dimension to make a single component tensor.  All of the components
in the dequeued tuple will have size n in the 0th dimension.
This operation has `k` outputs, where `k` is the number of components in
the tuples stored in the given queue, and output `i` is the ith
component of the dequeued tuple.

_inputArg
 handle:: .  n:: . 
_outputArg

_outputArg
 components:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QueueDequeueUpToV2(scope:Scope,  handle:tf.Output,  n:tf.Output, ) -> ( components:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QueueDequeueUpToV2",
        Input: [  handle,  n, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) components )

}

/*
Dequeues a tuple of one or more tensors from the given queue.

This operation has k outputs, where k is the number of components
in the tuples stored in the given queue, and output i is the ith
component of the dequeued tuple.
N.B. If the queue is empty, this operation will block until an element
has been dequeued (or 'timeout_ms' elapses, if specified).

_inputArg
 handle:: . 
_outputArg

_outputArg
 components:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QueueDequeueV2(scope:Scope,  handle:tf.Output, ) -> ( components:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QueueDequeueV2",
        Input: [  handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) components )

}

/*
Enqueues a tuple of one or more tensors in the given queue.

The components input has k elements, which correspond to the components of
tuples stored in the given queue.
N.B. If the queue is full, this operation will block until the given
element has been enqueued (or 'timeout_ms' elapses, if specified).

_inputArg
 handle:: .  components:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QueueEnqueue(scope:Scope,  handle:tf.Output,  components:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QueueEnqueue",
        Input: [  handle,  components, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Enqueues zero or more tuples of one or more tensors in the given queue.

This operation slices each component tensor along the 0th dimension to
make multiple queue elements. All of the tuple components must have the
same size in the 0th dimension.
The components input has k elements, which correspond to the components of
tuples stored in the given queue.
N.B. If the queue is full, this operation will block until the given
elements have been enqueued (or 'timeout_ms' elapses, if specified).

_inputArg
 handle:: .  components:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QueueEnqueueMany(scope:Scope,  handle:tf.Output,  components:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QueueEnqueueMany",
        Input: [  handle,  components, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Enqueues zero or more tuples of one or more tensors in the given queue.

This operation slices each component tensor along the 0th dimension to
make multiple queue elements. All of the tuple components must have the
same size in the 0th dimension.
The components input has k elements, which correspond to the components of
tuples stored in the given queue.
N.B. If the queue is full, this operation will block until the given
elements have been enqueued (or 'timeout_ms' elapses, if specified).

_inputArg
 handle:: .  components:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QueueEnqueueManyV2(scope:Scope,  handle:tf.Output,  components:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QueueEnqueueManyV2",
        Input: [  handle,  components, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Enqueues a tuple of one or more tensors in the given queue.

The components input has k elements, which correspond to the components of
tuples stored in the given queue.
N.B. If the queue is full, this operation will block until the given
element has been enqueued (or 'timeout_ms' elapses, if specified).

_inputArg
 handle:: .  components:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QueueEnqueueV2(scope:Scope,  handle:tf.Output,  components:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QueueEnqueueV2",
        Input: [  handle,  components, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Computes the number of elements in the given queue.


_inputArg
 handle:: . 
_outputArg

_outputArg
 size:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QueueSize(scope:Scope,  handle:tf.Output, ) -> ( size:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QueueSize",
        Input: [  handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) size )

}

/*
Computes the number of elements in the given queue.


_inputArg
 handle:: . 
_outputArg

_outputArg
 size:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func QueueSizeV2(scope:Scope,  handle:tf.Output, ) -> ( size:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "QueueSizeV2",
        Input: [  handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) size )

}

/*
Real-valued fast Fourier transform.

Computes the 1-dimensional discrete Fourier transform of a real-valued signal
over the inner-most dimension of `input`.
Since the DFT of a real signal is Hermitian-symmetric, `RFFT` only returns the
`fft_length / 2 + 1` unique components of the FFT: the zero-frequency term,
followed by the `fft_length / 2` positive-frequency terms.

_inputArg
 input:: .  fft_length:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RFFT(scope:Scope,  input:tf.Output,  fft_length:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RFFT",
        Input: [  input,  fft_length, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
2D real-valued fast Fourier transform.

Computes the 2-dimensional discrete Fourier transform of a real-valued signal
over the inner-most 2 dimensions of `input`.
Since the DFT of a real signal is Hermitian-symmetric, `RFFT2D` only returns the
`fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
of `output`: the zero-frequency term, followed by the `fft_length / 2`
positive-frequency terms.

_inputArg
 input:: .  fft_length:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RFFT2D(scope:Scope,  input:tf.Output,  fft_length:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RFFT2D",
        Input: [  input,  fft_length, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
3D real-valued fast Fourier transform.

Computes the 3-dimensional discrete Fourier transform of a real-valued signal
over the inner-most 3 dimensions of `input`.
Since the DFT of a real signal is Hermitian-symmetric, `RFFT3D` only returns the
`fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
of `output`: the zero-frequency term, followed by the `fft_length / 2`
positive-frequency terms.

_inputArg
 input:: .  fft_length:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RFFT3D(scope:Scope,  input:tf.Output,  fft_length:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RFFT3D",
        Input: [  input,  fft_length, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Converts one or more images from RGB to HSV.

Outputs a tensor of the same shape as the `images` tensor, containing the HSV
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.
`output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
`output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.

_inputArg
 images:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RGBToHSV(scope:Scope,  images:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RGBToHSV",
        Input: [  images, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Randomly crop `image`.

`size` is a 1-D int64 tensor with 2 elements representing the crop height and
width.  The values must be non negative.
This Op picks a random location in `image` and crops a `height` by `width`
rectangle from that location.  The random location is picked so the cropped
area will fit inside the original image.

_inputArg
 image:: .T  size:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RandomCrop(scope:Scope,  image:tf.Output,  size:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RandomCrop",
        Input: [  image,  size, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Outputs random values from the Gamma distribution(s) described by alpha.

This op uses the algorithm by Marsaglia et al. to acquire samples via
transformation-rejection from pairs of uniform and normal random variables.
See http://dl.acm.org/citation.cfm?id=358414

_inputArg
 shape:: .S  alpha:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RandomGamma(scope:Scope,  shape:tf.Output,  alpha:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RandomGamma",
        Input: [  shape,  alpha, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Outputs random values from the Poisson distribution(s) described by rate.

This op uses two algorithms, depending on rate. If rate >= 10, then
the algorithm by Hormann is used to acquire samples via
transformation-rejection.
See http://www.sciencedirect.com/science/article/pii/0167668793909974.
Otherwise, Knuth's algorithm is used to acquire samples via multiplying uniform
random variables.
See Donald E. Knuth (1969). Seminumerical Algorithms. The Art of Computer
Programming, Volume 2. Addison Wesley

_inputArg
 shape:: .S  rate:: .dtype 
_outputArg

_outputArg
 output::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RandomPoisson(scope:Scope,  shape:tf.Output,  rate:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RandomPoisson",
        Input: [  shape,  rate, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Randomly shuffles a tensor along its first dimension.

  The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
  to one and only one `output[i]`. For example, a mapping that might occur for a
  3x2 tensor is:
```
[[1, 2],       [[5, 6],
 [3, 4],  ==>   [1, 2],
 [5, 6]]        [3, 4]]
```

_inputArg
 value:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RandomShuffle(scope:Scope,  value:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RandomShuffle",
        Input: [  value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
A queue that randomizes the order of elements.


_inputArg
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RandomShuffleQueue(scope:Scope, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RandomShuffleQueue",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
A queue that randomizes the order of elements.


_inputArg
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RandomShuffleQueueV2(scope:Scope, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RandomShuffleQueueV2",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Outputs random values from a normal distribution.

The generated values will have mean 0 and standard deviation 1.

_inputArg
 shape:: .T 
_outputArg

_outputArg
 output::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RandomStandardNormal(scope:Scope,  shape:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RandomStandardNormal",
        Input: [  shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Outputs random values from a uniform distribution.

The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.

_inputArg
 shape:: .T 
_outputArg

_outputArg
 output::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RandomUniform(scope:Scope,  shape:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RandomUniform",
        Input: [  shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Outputs random integers from a uniform distribution.

The generated values are uniform integers in the range `[minval, maxval)`.
The lower bound `minval` is included in the range, while the upper bound
`maxval` is excluded.
The random integers are slightly biased unless `maxval - minval` is an exact
power of two.  The bias is small for values of `maxval - minval` significantly
smaller than the range of the output (either `2^32` or `2^64`).

_inputArg
 shape:: .T  minval:: .Tout  maxval:: .Tout 
_outputArg

_outputArg
 output::Tout 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RandomUniformInt(scope:Scope,  shape:tf.Output,  minval:tf.Output,  maxval:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RandomUniformInt",
        Input: [  shape,  minval,  maxval, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Creates a sequence of numbers.

This operation creates a sequence of numbers that begins at `start` and
extends by increments of `delta` up to but not including `limit`.
For example:
```
# 'start' is 3
# 'limit' is 18
# 'delta' is 3
tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
```

_inputArg
 start:: .Tidx  limit:: .Tidx  delta:: .Tidx 
_outputArg

_outputArg
 output::Tidx 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Range(scope:Scope,  start:tf.Output,  limit:tf.Output,  delta:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Range",
        Input: [  start,  limit,  delta, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Creates a dataset with a range of values. Corresponds to python's xrange.


_inputArg
 start:: .  stop:: .  step:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RangeDataset(scope:Scope,  start:tf.Output,  stop:tf.Output,  step:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RangeDataset",
        Input: [  start,  stop,  step, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Returns the rank of a tensor.

This operation returns an integer representing the rank of `input`.
For example:
```
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
# shape of tensor 't' is [2, 2, 3]
rank(t) ==> 3
```
**Note**: The rank of a tensor is not the same as the rank of a matrix. The rank
of a tensor is the number of indices required to uniquely select each element
of the tensor. Rank is also known as "order", "degree", or "ndims."

_inputArg
 input:: .T 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Rank(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Rank",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Reads and outputs the entire contents of the input filename.


_inputArg
 filename:: . 
_outputArg

_outputArg
 contents:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReadFile(scope:Scope,  filename:tf.Output, ) -> ( contents:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReadFile",
        Input: [  filename, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) contents )

}

/*
Returns the number of records this Reader has produced.

This is the same as the number of ReaderRead executions that have
succeeded.

_inputArg
 reader_handle:: . 
_outputArg

_outputArg
 records_produced:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReaderNumRecordsProduced(scope:Scope,  reader_handle:tf.Output, ) -> ( records_produced:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReaderNumRecordsProduced",
        Input: [  reader_handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) records_produced )

}

/*
Returns the number of records this Reader has produced.

This is the same as the number of ReaderRead executions that have
succeeded.

_inputArg
 reader_handle:: . 
_outputArg

_outputArg
 records_produced:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReaderNumRecordsProducedV2(scope:Scope,  reader_handle:tf.Output, ) -> ( records_produced:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReaderNumRecordsProducedV2",
        Input: [  reader_handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) records_produced )

}

/*
Returns the number of work units this Reader has finished processing.


_inputArg
 reader_handle:: . 
_outputArg

_outputArg
 units_completed:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReaderNumWorkUnitsCompleted(scope:Scope,  reader_handle:tf.Output, ) -> ( units_completed:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReaderNumWorkUnitsCompleted",
        Input: [  reader_handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) units_completed )

}

/*
Returns the number of work units this Reader has finished processing.


_inputArg
 reader_handle:: . 
_outputArg

_outputArg
 units_completed:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReaderNumWorkUnitsCompletedV2(scope:Scope,  reader_handle:tf.Output, ) -> ( units_completed:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReaderNumWorkUnitsCompletedV2",
        Input: [  reader_handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) units_completed )

}

/*
Returns the next record (key, value pair) produced by a Reader.

Will dequeue from the input queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has finished
with the previous file).

_inputArg
 reader_handle:: .  queue_handle:: . 
_outputArg

_outputArg
 key::  value:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReaderRead(scope:Scope,  reader_handle:tf.Output,  queue_handle:tf.Output, ) -> ( key:tf.Output,  value:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReaderRead",
        Input: [  reader_handle,  queue_handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) key  op.Output(*) value )

}

/*
Returns up to `num_records` (key, value) pairs produced by a Reader.

Will dequeue from the input queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has finished
with the previous file).
It may return less than `num_records` even before the last batch.

_inputArg
 reader_handle:: .  queue_handle:: .  num_records:: . 
_outputArg

_outputArg
 keys::  values:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReaderReadUpTo(scope:Scope,  reader_handle:tf.Output,  queue_handle:tf.Output,  num_records:tf.Output, ) -> ( keys:tf.Output,  values:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReaderReadUpTo",
        Input: [  reader_handle,  queue_handle,  num_records, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) keys  op.Output(*) values )

}

/*
Returns up to `num_records` (key, value) pairs produced by a Reader.

Will dequeue from the input queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has finished
with the previous file).
It may return less than `num_records` even before the last batch.

_inputArg
 reader_handle:: .  queue_handle:: .  num_records:: . 
_outputArg

_outputArg
 keys::  values:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReaderReadUpToV2(scope:Scope,  reader_handle:tf.Output,  queue_handle:tf.Output,  num_records:tf.Output, ) -> ( keys:tf.Output,  values:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReaderReadUpToV2",
        Input: [  reader_handle,  queue_handle,  num_records, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) keys  op.Output(*) values )

}

/*
Returns the next record (key, value pair) produced by a Reader.

Will dequeue from the input queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has finished
with the previous file).

_inputArg
 reader_handle:: .  queue_handle:: . 
_outputArg

_outputArg
 key::  value:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReaderReadV2(scope:Scope,  reader_handle:tf.Output,  queue_handle:tf.Output, ) -> ( key:tf.Output,  value:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReaderReadV2",
        Input: [  reader_handle,  queue_handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) key  op.Output(*) value )

}

/*
Restore a Reader to its initial clean state.


_inputArg
 reader_handle:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReaderReset(scope:Scope,  reader_handle:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReaderReset",
        Input: [  reader_handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Restore a Reader to its initial clean state.


_inputArg
 reader_handle:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReaderResetV2(scope:Scope,  reader_handle:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReaderResetV2",
        Input: [  reader_handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Restore a reader to a previously saved state.

Not all Readers support being restored, so this can produce an
Unimplemented error.

_inputArg
 reader_handle:: .  state:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReaderRestoreState(scope:Scope,  reader_handle:tf.Output,  state:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReaderRestoreState",
        Input: [  reader_handle,  state, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Restore a reader to a previously saved state.

Not all Readers support being restored, so this can produce an
Unimplemented error.

_inputArg
 reader_handle:: .  state:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReaderRestoreStateV2(scope:Scope,  reader_handle:tf.Output,  state:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReaderRestoreStateV2",
        Input: [  reader_handle,  state, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Produce a string tensor that encodes the state of a Reader.

Not all Readers support being serialized, so this can produce an
Unimplemented error.

_inputArg
 reader_handle:: . 
_outputArg

_outputArg
 state:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReaderSerializeState(scope:Scope,  reader_handle:tf.Output, ) -> ( state:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReaderSerializeState",
        Input: [  reader_handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) state )

}

/*
Produce a string tensor that encodes the state of a Reader.

Not all Readers support being serialized, so this can produce an
Unimplemented error.

_inputArg
 reader_handle:: . 
_outputArg

_outputArg
 state:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReaderSerializeStateV2(scope:Scope,  reader_handle:tf.Output, ) -> ( state:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReaderSerializeStateV2",
        Input: [  reader_handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) state )

}

/*
Returns the real part of a complex number.

Given a tensor `input` of complex numbers, this operation returns a tensor of
type `float` that is the real part of each element in `input`. All elements in
`input` must be complex numbers of the form \\(a + bj\\), where *a* is the real
 part returned by this operation and *b* is the imaginary part.
For example:
```
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.real(input) ==> [-2.25, 3.25]
```

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::Tout 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Real(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Real",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Returns x / y element-wise for real types.

If `x` and `y` are reals, this will return the floating-point division.
*NOTE*: `Div` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RealDiv(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RealDiv",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Computes the reciprocal of x element-wise.

I.e., \\(y = 1 / x\\).

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Reciprocal(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Reciprocal",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Computes the gradient for the inverse of `x` wrt its input.

Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
is the corresponding input gradient.

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReciprocalGrad(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReciprocalGrad",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Emits randomized records.


_inputArg
_outputArg

_outputArg
 records:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RecordInput(scope:Scope, ) -> ( records:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RecordInput",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) records )

}

/*
Joins a string Tensor across the given dimensions.

Computes the string join across dimensions in the given string Tensor of shape
`[d_0, d_1, ..., d_n-1]`.  Returns a new Tensor created by joining the input
strings with the given separator (default: empty string).  Negative indices are
counted backwards from the end, with `-1` being equivalent to `n - 1`.
For example:
```python
# tensor `a` is [["a", "b"], ["c", "d"]]
tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
tf.reduce_join(a, [0, 1]) ==> ["acbd"]
tf.reduce_join(a, [1, 0]) ==> ["abcd"]
tf.reduce_join(a, []) ==> ["abcd"]
```

_inputArg
 inputs:: .  reduction_indices:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReduceJoin(scope:Scope,  inputs:tf.Output,  reduction_indices:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReduceJoin",
        Input: [  inputs,  reduction_indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Creates or finds a child frame, and makes `data` available to the child frame.

The unique `frame_name` is used by the `Executor` to identify frames. If
`is_constant` is true, `output` is a constant in the child frame; otherwise
it may be changed in the child frame. At most `parallel_iterations` iterations
are run in parallel in the child frame.

_inputArg
 data:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RefEnter(scope:Scope,  data:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RefEnter",
        Input: [  data, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Exits the current frame to its parent frame.

Exit makes its input `data` available to the parent frame.

_inputArg
 data:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RefExit(scope:Scope,  data:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RefExit",
        Input: [  data, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Return the same ref tensor as the input ref tensor.


_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RefIdentity(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RefIdentity",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Forwards the value of an available tensor from `inputs` to `output`.

`Merge` waits for at least one of the tensors in `inputs` to become available.
It is usually combined with `Switch` to implement branching.
`Merge` forwards the first tensor for become available to `output`, and sets
`value_index` to its index in `inputs`.

_inputArg
 inputs:: .T 
_outputArg

_outputArg
 output::T  value_index:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RefMerge(scope:Scope,  inputs:tf.Output, ) -> ( output:tf.Output,  value_index:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RefMerge",
        Input: [  inputs, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output  op.Output(*) value_index )

}

/*
Makes its input available to the next iteration.


_inputArg
 data:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RefNextIteration(scope:Scope,  data:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RefNextIteration",
        Input: [  data, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Forwards the `index`th element of `inputs` to `output`.


_inputArg
 index:: .  inputs:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RefSelect(scope:Scope,  index:tf.Output,  inputs:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RefSelect",
        Input: [  index,  inputs, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Forwards the ref tensor `data` to the output port determined by `pred`.

If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
the data goes to `output_false`.
See also `Switch` and `Merge`.

_inputArg
 data:: .T  pred:: . 
_outputArg

_outputArg
 output_false::T  output_true::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RefSwitch(scope:Scope,  data:tf.Output,  pred:tf.Output, ) -> ( output_false:tf.Output,  output_true:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RefSwitch",
        Input: [  data,  pred, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_false  op.Output(*) output_true )

}

/*
Computes rectified linear: `max(features, 0)`.


_inputArg
 features:: .T 
_outputArg

_outputArg
 activations::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Relu(scope:Scope,  features:tf.Output, ) -> ( activations:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Relu",
        Input: [  features, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) activations )

}

/*
Computes rectified linear 6: `min(max(features, 0), 6)`.


_inputArg
 features:: .T 
_outputArg

_outputArg
 activations::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Relu6(scope:Scope,  features:tf.Output, ) -> ( activations:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Relu6",
        Input: [  features, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) activations )

}

/*
Computes rectified linear 6 gradients for a Relu6 operation.


_inputArg
 gradients:: .T  features:: .T 
_outputArg

_outputArg
 backprops::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Relu6Grad(scope:Scope,  gradients:tf.Output,  features:tf.Output, ) -> ( backprops:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Relu6Grad",
        Input: [  gradients,  features, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) backprops )

}

/*
Computes rectified linear gradients for a Relu operation.


_inputArg
 gradients:: .T  features:: .T 
_outputArg

_outputArg
 backprops::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReluGrad(scope:Scope,  gradients:tf.Output,  features:tf.Output, ) -> ( backprops:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReluGrad",
        Input: [  gradients,  features, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) backprops )

}

/*
Creates a dataset that emits the outputs of `input_dataset` `count` times.


_inputArg
 input_dataset:: .  count:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RepeatDataset(scope:Scope,  input_dataset:tf.Output,  count:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RepeatDataset",
        Input: [  input_dataset,  count, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Given a quantized tensor described by (input, input_min, input_max), outputs a

range that covers the actual values present in that tensor.  This op is
typically used to produce the requested_output_min and requested_output_max for
Requantize.

_inputArg
 input:: .Tinput  input_min:: .  input_max:: . 
_outputArg

_outputArg
 output_min::  output_max:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RequantizationRange(scope:Scope,  input:tf.Output,  input_min:tf.Output,  input_max:tf.Output, ) -> ( output_min:tf.Output,  output_max:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RequantizationRange",
        Input: [  input,  input_min,  input_max, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_min  op.Output(*) output_max )

}

/*
Convert the quantized 'input' tensor into a lower-precision 'output', using the

output range specified with 'requested_output_min' and 'requested_output_max'.
[input_min, input_max] are scalar floats that specify the range for the float
interpretation of the 'input' data. For example, if input_min is -1.0f and
input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.

_inputArg
 input:: .Tinput  input_min:: .  input_max:: .  requested_output_min:: .  requested_output_max:: . 
_outputArg

_outputArg
 output::out_type  output_min::  output_max:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Requantize(scope:Scope,  input:tf.Output,  input_min:tf.Output,  input_max:tf.Output,  requested_output_min:tf.Output,  requested_output_max:tf.Output, ) -> ( output:tf.Output,  output_min:tf.Output,  output_max:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Requantize",
        Input: [  input,  input_min,  input_max,  requested_output_min,  requested_output_max, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output  op.Output(*) output_min  op.Output(*) output_max )

}

/*
Reshapes a tensor.

Given `tensor`, this operation returns a tensor that has the same values
as `tensor` with shape `shape`.
If one component of `shape` is the special value -1, the size of that dimension
is computed so that the total size remains constant.  In particular, a `shape`
of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.
If `shape` is 1-D or higher, then the operation returns a tensor with shape
`shape` filled with the values of `tensor`. In this case, the number of elements
implied by `shape` must be the same as the number of elements in `tensor`.
For example:
```
# tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
# tensor 't' has shape [9]
reshape(t, [3, 3]) ==> [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]
# tensor 't' is [[[1, 1], [2, 2]],
#                [[3, 3], [4, 4]]]
# tensor 't' has shape [2, 2, 2]
reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                        [3, 3, 4, 4]]
# tensor 't' is [[[1, 1, 1],
#                 [2, 2, 2]],
#                [[3, 3, 3],
#                 [4, 4, 4]],
#                [[5, 5, 5],
#                 [6, 6, 6]]]
# tensor 't' has shape [3, 2, 3]
# pass '[-1]' to flatten 't'
reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
# -1 can also be used to infer the shape
# -1 is inferred to be 9:
reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# -1 is inferred to be 2:
reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# -1 is inferred to be 3:
reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                              [2, 2, 2],
                              [3, 3, 3]],
                             [[4, 4, 4],
                              [5, 5, 5],
                              [6, 6, 6]]]
# tensor 't' is [7]
# shape `[]` reshapes to a scalar
reshape(t, []) ==> 7
```

_inputArg
 tensor:: .T  shape:: .Tshape 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Reshape(scope:Scope,  tensor:tf.Output,  shape:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Reshape",
        Input: [  tensor,  shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Resize `images` to `size` using area interpolation.

Input images can be of different types but output images are always float.

_inputArg
 images:: .T  size:: . 
_outputArg

_outputArg
 resized_images:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResizeArea(scope:Scope,  images:tf.Output,  size:tf.Output, ) -> ( resized_images:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResizeArea",
        Input: [  images,  size, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) resized_images )

}

/*
Resize `images` to `size` using bicubic interpolation.

Input images can be of different types but output images are always float.

_inputArg
 images:: .T  size:: . 
_outputArg

_outputArg
 resized_images:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResizeBicubic(scope:Scope,  images:tf.Output,  size:tf.Output, ) -> ( resized_images:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResizeBicubic",
        Input: [  images,  size, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) resized_images )

}

/*
Resize `images` to `size` using bilinear interpolation.

Input images can be of different types but output images are always float.

_inputArg
 images:: .T  size:: . 
_outputArg

_outputArg
 resized_images:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResizeBilinear(scope:Scope,  images:tf.Output,  size:tf.Output, ) -> ( resized_images:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResizeBilinear",
        Input: [  images,  size, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) resized_images )

}

/*
Computes the gradient of bilinear interpolation.


_inputArg
 grads:: .  original_image:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResizeBilinearGrad(scope:Scope,  grads:tf.Output,  original_image:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResizeBilinearGrad",
        Input: [  grads,  original_image, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Resize `images` to `size` using nearest neighbor interpolation.


_inputArg
 images:: .T  size:: . 
_outputArg

_outputArg
 resized_images::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResizeNearestNeighbor(scope:Scope,  images:tf.Output,  size:tf.Output, ) -> ( resized_images:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResizeNearestNeighbor",
        Input: [  images,  size, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) resized_images )

}

/*
Computes the gradient of nearest neighbor interpolation.


_inputArg
 grads:: .T  size:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResizeNearestNeighborGrad(scope:Scope,  grads:tf.Output,  size:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResizeNearestNeighborGrad",
        Input: [  grads,  size, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Update '*var' according to the adadelta scheme.

accum = rho() * accum + (1 - rho()) * grad.square();
update = (update_accum + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad;
update_accum = rho() * update_accum + (1 - rho()) * update.square();
var -= update;

_inputArg
 var:: .  accum:: .  accum_update:: .  lr:: .T  rho:: .T  epsilon:: .T  grad:: .T 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceApplyAdadelta(scope:Scope,  var:tf.Output,  accum:tf.Output,  accum_update:tf.Output,  lr:tf.Output,  rho:tf.Output,  epsilon:tf.Output,  grad:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceApplyAdadelta",
        Input: [  var,  accum,  accum_update,  lr,  rho,  epsilon,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Update '*var' according to the adagrad scheme.

accum += grad * grad
var -= lr * grad * (1 / sqrt(accum))

_inputArg
 var:: .  accum:: .  lr:: .T  grad:: .T 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceApplyAdagrad(scope:Scope,  var:tf.Output,  accum:tf.Output,  lr:tf.Output,  grad:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceApplyAdagrad",
        Input: [  var,  accum,  lr,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Update '*var' according to the proximal adagrad scheme.


_inputArg
 var:: .  gradient_accumulator:: .  gradient_squared_accumulator:: .  grad:: .T  lr:: .T  l1:: .T  l2:: .T  global_step:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceApplyAdagradDA(scope:Scope,  var:tf.Output,  gradient_accumulator:tf.Output,  gradient_squared_accumulator:tf.Output,  grad:tf.Output,  lr:tf.Output,  l1:tf.Output,  l2:tf.Output,  global_step:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceApplyAdagradDA",
        Input: [  var,  gradient_accumulator,  gradient_squared_accumulator,  grad,  lr,  l1,  l2,  global_step, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Update '*var' according to the Adam algorithm.

lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)

_inputArg
 var:: .  m:: .  v:: .  beta1_power:: .T  beta2_power:: .T  lr:: .T  beta1:: .T  beta2:: .T  epsilon:: .T  grad:: .T 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceApplyAdam(scope:Scope,  var:tf.Output,  m:tf.Output,  v:tf.Output,  beta1_power:tf.Output,  beta2_power:tf.Output,  lr:tf.Output,  beta1:tf.Output,  beta2:tf.Output,  epsilon:tf.Output,  grad:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceApplyAdam",
        Input: [  var,  m,  v,  beta1_power,  beta2_power,  lr,  beta1,  beta2,  epsilon,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Update '*var' according to the centered RMSProp algorithm.

The centered RMSProp algorithm uses an estimate of the centered second moment
(i.e., the variance) for normalization, as opposed to regular RMSProp, which
uses the (uncentered) second moment. This often helps with training, but is
slightly more expensive in terms of computation and memory.
Note that in dense implementation of this algorithm, mg, ms, and mom will
update even if the grad is zero, but in this sparse implementation, mg, ms,
and mom will not update in iterations during which the grad is zero.
mean_square = decay * mean_square + (1-decay) * gradient ** 2
mean_grad = decay * mean_grad + (1-decay) * gradient
Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)
mg <- rho * mg_{t-1} + (1-rho) * grad
ms <- rho * ms_{t-1} + (1-rho) * grad * grad
mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms - mg * mg + epsilon)
var <- var - mom

_inputArg
 var:: .  mg:: .  ms:: .  mom:: .  lr:: .T  rho:: .T  momentum:: .T  epsilon:: .T  grad:: .T 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceApplyCenteredRMSProp(scope:Scope,  var:tf.Output,  mg:tf.Output,  ms:tf.Output,  mom:tf.Output,  lr:tf.Output,  rho:tf.Output,  momentum:tf.Output,  epsilon:tf.Output,  grad:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceApplyCenteredRMSProp",
        Input: [  var,  mg,  ms,  mom,  lr,  rho,  momentum,  epsilon,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Update '*var' according to the Ftrl-proximal scheme.

accum_new = accum + grad * grad
linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new

_inputArg
 var:: .  accum:: .  linear:: .  grad:: .T  lr:: .T  l1:: .T  l2:: .T  lr_power:: .T 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceApplyFtrl(scope:Scope,  var:tf.Output,  accum:tf.Output,  linear:tf.Output,  grad:tf.Output,  lr:tf.Output,  l1:tf.Output,  l2:tf.Output,  lr_power:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceApplyFtrl",
        Input: [  var,  accum,  linear,  grad,  lr,  l1,  l2,  lr_power, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Update '*var' by subtracting 'alpha' * 'delta' from it.


_inputArg
 var:: .  alpha:: .T  delta:: .T 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceApplyGradientDescent(scope:Scope,  var:tf.Output,  alpha:tf.Output,  delta:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceApplyGradientDescent",
        Input: [  var,  alpha,  delta, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Update '*var' according to the momentum scheme. Set use_nesterov = True if you

want to use Nesterov momentum.
accum = accum * momentum + grad
var -= lr * accum

_inputArg
 var:: .  accum:: .  lr:: .T  grad:: .T  momentum:: .T 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceApplyMomentum(scope:Scope,  var:tf.Output,  accum:tf.Output,  lr:tf.Output,  grad:tf.Output,  momentum:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceApplyMomentum",
        Input: [  var,  accum,  lr,  grad,  momentum, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.

accum += grad * grad
prox_v = var - lr * grad * (1 / sqrt(accum))
var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}

_inputArg
 var:: .  accum:: .  lr:: .T  l1:: .T  l2:: .T  grad:: .T 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceApplyProximalAdagrad(scope:Scope,  var:tf.Output,  accum:tf.Output,  lr:tf.Output,  l1:tf.Output,  l2:tf.Output,  grad:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceApplyProximalAdagrad",
        Input: [  var,  accum,  lr,  l1,  l2,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Update '*var' as FOBOS algorithm with fixed learning rate.

prox_v = var - alpha * delta
var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

_inputArg
 var:: .  alpha:: .T  l1:: .T  l2:: .T  delta:: .T 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceApplyProximalGradientDescent(scope:Scope,  var:tf.Output,  alpha:tf.Output,  l1:tf.Output,  l2:tf.Output,  delta:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceApplyProximalGradientDescent",
        Input: [  var,  alpha,  l1,  l2,  delta, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Update '*var' according to the RMSProp algorithm.

Note that in dense implementation of this algorithm, ms and mom will
update even if the grad is zero, but in this sparse implementation, ms
and mom will not update in iterations during which the grad is zero.
mean_square = decay * mean_square + (1-decay) * gradient ** 2
Delta = learning_rate * gradient / sqrt(mean_square + epsilon)
ms <- rho * ms_{t-1} + (1-rho) * grad * grad
mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
var <- var - mom

_inputArg
 var:: .  ms:: .  mom:: .  lr:: .T  rho:: .T  momentum:: .T  epsilon:: .T  grad:: .T 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceApplyRMSProp(scope:Scope,  var:tf.Output,  ms:tf.Output,  mom:tf.Output,  lr:tf.Output,  rho:tf.Output,  momentum:tf.Output,  epsilon:tf.Output,  grad:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceApplyRMSProp",
        Input: [  var,  ms,  mom,  lr,  rho,  momentum,  epsilon,  grad, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
var: Should be from a Variable().


_inputArg
 var:: .  accum:: .  accum_update:: .  lr:: .T  rho:: .T  epsilon:: .T  grad:: .T  indices:: .Tindices 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceSparseApplyAdadelta(scope:Scope,  var:tf.Output,  accum:tf.Output,  accum_update:tf.Output,  lr:tf.Output,  rho:tf.Output,  epsilon:tf.Output,  grad:tf.Output,  indices:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceSparseApplyAdadelta",
        Input: [  var,  accum,  accum_update,  lr,  rho,  epsilon,  grad,  indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Update relevant entries in '*var' and '*accum' according to the adagrad scheme.

That is for rows we have grad for, we update var and accum as follows:
accum += grad * grad
var -= lr * grad * (1 / sqrt(accum))

_inputArg
 var:: .  accum:: .  lr:: .T  grad:: .T  indices:: .Tindices 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceSparseApplyAdagrad(scope:Scope,  var:tf.Output,  accum:tf.Output,  lr:tf.Output,  grad:tf.Output,  indices:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceSparseApplyAdagrad",
        Input: [  var,  accum,  lr,  grad,  indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Update entries in '*var' and '*accum' according to the proximal adagrad scheme.


_inputArg
 var:: .  gradient_accumulator:: .  gradient_squared_accumulator:: .  grad:: .T  indices:: .Tindices  lr:: .T  l1:: .T  l2:: .T  global_step:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceSparseApplyAdagradDA(scope:Scope,  var:tf.Output,  gradient_accumulator:tf.Output,  gradient_squared_accumulator:tf.Output,  grad:tf.Output,  indices:tf.Output,  lr:tf.Output,  l1:tf.Output,  l2:tf.Output,  global_step:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceSparseApplyAdagradDA",
        Input: [  var,  gradient_accumulator,  gradient_squared_accumulator,  grad,  indices,  lr,  l1,  l2,  global_step, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Update '*var' according to the centered RMSProp algorithm.

The centered RMSProp algorithm uses an estimate of the centered second moment
(i.e., the variance) for normalization, as opposed to regular RMSProp, which
uses the (uncentered) second moment. This often helps with training, but is
slightly more expensive in terms of computation and memory.
Note that in dense implementation of this algorithm, mg, ms, and mom will
update even if the grad is zero, but in this sparse implementation, mg, ms,
and mom will not update in iterations during which the grad is zero.
mean_square = decay * mean_square + (1-decay) * gradient ** 2
mean_grad = decay * mean_grad + (1-decay) * gradient
Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)
ms <- rho * ms_{t-1} + (1-rho) * grad * grad
mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
var <- var - mom

_inputArg
 var:: .  mg:: .  ms:: .  mom:: .  lr:: .T  rho:: .T  momentum:: .T  epsilon:: .T  grad:: .T  indices:: .Tindices 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceSparseApplyCenteredRMSProp(scope:Scope,  var:tf.Output,  mg:tf.Output,  ms:tf.Output,  mom:tf.Output,  lr:tf.Output,  rho:tf.Output,  momentum:tf.Output,  epsilon:tf.Output,  grad:tf.Output,  indices:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceSparseApplyCenteredRMSProp",
        Input: [  var,  mg,  ms,  mom,  lr,  rho,  momentum,  epsilon,  grad,  indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Update relevant entries in '*var' according to the Ftrl-proximal scheme.

That is for rows we have grad for, we update var, accum and linear as follows:
accum_new = accum + grad * grad
linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new

_inputArg
 var:: .  accum:: .  linear:: .  grad:: .T  indices:: .Tindices  lr:: .T  l1:: .T  l2:: .T  lr_power:: .T 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceSparseApplyFtrl(scope:Scope,  var:tf.Output,  accum:tf.Output,  linear:tf.Output,  grad:tf.Output,  indices:tf.Output,  lr:tf.Output,  l1:tf.Output,  l2:tf.Output,  lr_power:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceSparseApplyFtrl",
        Input: [  var,  accum,  linear,  grad,  indices,  lr,  l1,  l2,  lr_power, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Update relevant entries in '*var' and '*accum' according to the momentum scheme.

Set use_nesterov = True if you want to use Nesterov momentum.
That is for rows we have grad for, we update var and accum as follows:
accum = accum * momentum + grad
var -= lr * accum

_inputArg
 var:: .  accum:: .  lr:: .T  grad:: .T  indices:: .Tindices  momentum:: .T 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceSparseApplyMomentum(scope:Scope,  var:tf.Output,  accum:tf.Output,  lr:tf.Output,  grad:tf.Output,  indices:tf.Output,  momentum:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceSparseApplyMomentum",
        Input: [  var,  accum,  lr,  grad,  indices,  momentum, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Sparse update entries in '*var' and '*accum' according to FOBOS algorithm.

That is for rows we have grad for, we update var and accum as follows:
accum += grad * grad
prox_v = var
prox_v -= lr * grad * (1 / sqrt(accum))
var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}

_inputArg
 var:: .  accum:: .  lr:: .T  l1:: .T  l2:: .T  grad:: .T  indices:: .Tindices 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceSparseApplyProximalAdagrad(scope:Scope,  var:tf.Output,  accum:tf.Output,  lr:tf.Output,  l1:tf.Output,  l2:tf.Output,  grad:tf.Output,  indices:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceSparseApplyProximalAdagrad",
        Input: [  var,  accum,  lr,  l1,  l2,  grad,  indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Sparse update '*var' as FOBOS algorithm with fixed learning rate.

That is for rows we have grad for, we update var as follows:
prox_v = var - alpha * grad
var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

_inputArg
 var:: .  alpha:: .T  l1:: .T  l2:: .T  grad:: .T  indices:: .Tindices 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceSparseApplyProximalGradientDescent(scope:Scope,  var:tf.Output,  alpha:tf.Output,  l1:tf.Output,  l2:tf.Output,  grad:tf.Output,  indices:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceSparseApplyProximalGradientDescent",
        Input: [  var,  alpha,  l1,  l2,  grad,  indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Update '*var' according to the RMSProp algorithm.

Note that in dense implementation of this algorithm, ms and mom will
update even if the grad is zero, but in this sparse implementation, ms
and mom will not update in iterations during which the grad is zero.
mean_square = decay * mean_square + (1-decay) * gradient ** 2
Delta = learning_rate * gradient / sqrt(mean_square + epsilon)
ms <- rho * ms_{t-1} + (1-rho) * grad * grad
mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
var <- var - mom

_inputArg
 var:: .  ms:: .  mom:: .  lr:: .T  rho:: .T  momentum:: .T  epsilon:: .T  grad:: .T  indices:: .Tindices 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceSparseApplyRMSProp(scope:Scope,  var:tf.Output,  ms:tf.Output,  mom:tf.Output,  lr:tf.Output,  rho:tf.Output,  momentum:tf.Output,  epsilon:tf.Output,  grad:tf.Output,  indices:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceSparseApplyRMSProp",
        Input: [  var,  ms,  mom,  lr,  rho,  momentum,  epsilon,  grad,  indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Assign `value` to the sliced l-value reference of `ref`.

The values of `value` are assigned to the positions in the variable
`ref` that are selected by the slice parameters. The slice parameters
`begin, `end`, `strides`, etc. work exactly as in `StridedSlice`.
NOTE this op currently does not support broadcasting and so `value`'s
shape must be exactly the shape produced by the slice of `ref`.

_inputArg
 ref:: .  begin:: .Index  end:: .Index  strides:: .Index  value:: .T 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ResourceStridedSliceAssign(scope:Scope,  ref:tf.Output,  begin:tf.Output,  end:tf.Output,  strides:tf.Output,  value:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ResourceStridedSliceAssign",
        Input: [  ref,  begin,  end,  strides,  value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Restores a tensor from checkpoint files.

Reads a tensor stored in one or several files. If there are several files (for
instance because a tensor was saved as slices), `file_pattern` may contain
wildcard symbols (`*` and `?`) in the filename portion only, not in the
directory portion.
If a `file_pattern` matches several files, `preferred_shard` can be used to hint
in which file the requested tensor is likely to be found. This op will first
open the file at index `preferred_shard` in the list of matching files and try
to restore tensors from that file.  Only if some tensors or tensor slices are
not found in that first file, then the Op opens all the files. Setting
`preferred_shard` to match the value passed as the `shard` input
of a matching `Save` Op may speed up Restore.  This attribute only affects
performance, not correctness.  The default value -1 means files are processed in
order.
See also `RestoreSlice`.

_inputArg
 file_pattern:: .  tensor_name:: . 
_outputArg

_outputArg
 tensor::dt 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Restore(scope:Scope,  file_pattern:tf.Output,  tensor_name:tf.Output, ) -> ( tensor:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Restore",
        Input: [  file_pattern,  tensor_name, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) tensor )

}

/*
Restores a tensor from checkpoint files.

This is like `Restore` except that restored tensor can be listed as filling
only a slice of a larger tensor.  `shape_and_slice` specifies the shape of the
larger tensor and the slice that the restored tensor covers.
The `shape_and_slice` input has the same format as the
elements of the `shapes_and_slices` input of the `SaveSlices` op.

_inputArg
 file_pattern:: .  tensor_name:: .  shape_and_slice:: . 
_outputArg

_outputArg
 tensor::dt 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RestoreSlice(scope:Scope,  file_pattern:tf.Output,  tensor_name:tf.Output,  shape_and_slice:tf.Output, ) -> ( tensor:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RestoreSlice",
        Input: [  file_pattern,  tensor_name,  shape_and_slice, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) tensor )

}

/*
Restores tensors from a V2 checkpoint.

For backward compatibility with the V1 format, this Op currently allows
restoring from a V1 checkpoint as well:
  - This Op first attempts to find the V2 index file pointed to by "prefix", and
    if found proceed to read it as a V2 checkpoint;
  - Otherwise the V1 read path is invoked.
Relying on this behavior is not recommended, as the ability to fall back to read
V1 might be deprecated and eventually removed.
By default, restores the named tensors in full.  If the caller wishes to restore
specific slices of stored tensors, "shape_and_slices" should be non-empty
strings and correspondingly well-formed.
Callers must ensure all the named tensors are indeed stored in the checkpoint.

_inputArg
 prefix:: .  tensor_names:: .  shape_and_slices:: . 
_outputArg

_outputArg
 tensors:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RestoreV2(scope:Scope,  prefix:tf.Output,  tensor_names:tf.Output,  shape_and_slices:tf.Output, ) -> ( tensors:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RestoreV2",
        Input: [  prefix,  tensor_names,  shape_and_slices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) tensors )

}

/*
Reverses specific dimensions of a tensor.

Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
of `tensor`, this operation reverses each dimension i of `tensor` where
`dims[i]` is `True`.
`tensor` can have up to 8 dimensions. The number of dimensions
of `tensor` must equal the number of elements in `dims`. In other words:
`rank(tensor) = size(dims)`
For example:
```
# tensor 't' is [[[[ 0,  1,  2,  3],
#                  [ 4,  5,  6,  7],
#                  [ 8,  9, 10, 11]],
#                 [[12, 13, 14, 15],
#                  [16, 17, 18, 19],
#                  [20, 21, 22, 23]]]]
# tensor 't' shape is [1, 2, 3, 4]
# 'dims' is [False, False, False, True]
reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                        [ 7,  6,  5,  4],
                        [ 11, 10, 9, 8]],
                       [[15, 14, 13, 12],
                        [19, 18, 17, 16],
                        [23, 22, 21, 20]]]]
# 'dims' is [False, True, False, False]
reverse(t, dims) ==> [[[[12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23]
                       [[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11]]]]
# 'dims' is [False, False, True, False]
reverse(t, dims) ==> [[[[8, 9, 10, 11],
                        [4, 5, 6, 7],
                        [0, 1, 2, 3]]
                       [[20, 21, 22, 23],
                        [16, 17, 18, 19],
                        [12, 13, 14, 15]]]]
```

_inputArg
 tensor:: .T  dims:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Reverse(scope:Scope,  tensor:tf.Output,  dims:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Reverse",
        Input: [  tensor,  dims, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Reverses variable length slices.

This op first slices `input` along the dimension `batch_dim`, and for each
slice `i`, reverses the first `seq_lengths[i]` elements along
the dimension `seq_dim`.
The elements of `seq_lengths` must obey `seq_lengths[i] <= input.dims[seq_dim]`,
and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.
The output slice `i` along dimension `batch_dim` is then given by input
slice `i`, with the first `seq_lengths[i]` slices along dimension
`seq_dim` reversed.
For example:
```
# Given this:
batch_dim = 0
seq_dim = 1
input.dims = (4, 8, ...)
seq_lengths = [7, 2, 3, 5]
# then slices of input are reversed on seq_dim, but only up to seq_lengths:
output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]
# while entries past seq_lens are copied through:
output[0, 7:, :, ...] = input[0, 7:, :, ...]
output[1, 2:, :, ...] = input[1, 2:, :, ...]
output[2, 3:, :, ...] = input[2, 3:, :, ...]
output[3, 2:, :, ...] = input[3, 2:, :, ...]
```
In contrast, if:
```
# Given this:
batch_dim = 2
seq_dim = 0
input.dims = (8, ?, 4, ...)
seq_lengths = [7, 2, 3, 5]
# then slices of input are reversed on seq_dim, but only up to seq_lengths:
output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]
# while entries past seq_lens are copied through:
output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
```

_inputArg
 input:: .T  seq_lengths:: .Tlen 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReverseSequence(scope:Scope,  input:tf.Output,  seq_lengths:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReverseSequence",
        Input: [  input,  seq_lengths, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Reverses specific dimensions of a tensor.

NOTE `tf.reverse` has now changed behavior in preparation for 1.0.
`tf.reverse_v2` is currently an alias that will be deprecated before TF 1.0.
Given a `tensor`, and a `int32` tensor `axis` representing the set of
dimensions of `tensor` to reverse. This operation reverses each dimension
`i` for which there exists `j` s.t. `axis[j] == i`.
`tensor` can have up to 8 dimensions. The number of dimensions specified
in `axis` may be 0 or more entries. If an index is specified more than
once, a InvalidArgument error is raised.
For example:
```
# tensor 't' is [[[[ 0,  1,  2,  3],
#                  [ 4,  5,  6,  7],
#                  [ 8,  9, 10, 11]],
#                 [[12, 13, 14, 15],
#                  [16, 17, 18, 19],
#                  [20, 21, 22, 23]]]]
# tensor 't' shape is [1, 2, 3, 4]
# 'dims' is [3] or 'dims' is -1
reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                        [ 7,  6,  5,  4],
                        [ 11, 10, 9, 8]],
                       [[15, 14, 13, 12],
                        [19, 18, 17, 16],
                        [23, 22, 21, 20]]]]
# 'dims' is '[1]' (or 'dims' is '[-3]')
reverse(t, dims) ==> [[[[12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23]
                       [[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11]]]]
# 'dims' is '[2]' (or 'dims' is '[-2]')
reverse(t, dims) ==> [[[[8, 9, 10, 11],
                        [4, 5, 6, 7],
                        [0, 1, 2, 3]]
                       [[20, 21, 22, 23],
                        [16, 17, 18, 19],
                        [12, 13, 14, 15]]]]
```

_inputArg
 tensor:: .T  axis:: .Tidx 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ReverseV2(scope:Scope,  tensor:tf.Output,  axis:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ReverseV2",
        Input: [  tensor,  axis, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Returns element-wise integer closest to x.

If the result is midway between two representable values,
the even representable is chosen.
For example:
```
rint(-1.5) ==> -2.0
rint(0.5000001) ==> 1.0
rint([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
```

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Rint(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Rint",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Rounds the values of a tensor to the nearest integer, element-wise.

Rounds half to even.  Also known as bankers rounding. If you want to round
according to the current system rounding mode use std::cint.

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Round(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Round",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Computes reciprocal of square root of x element-wise.

I.e., \\(y = 1 / \sqrt{x}\\).

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Rsqrt(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Rsqrt",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Computes the gradient for the rsqrt of `x` wrt its input.

Specifically, `grad = dy * -0.5 * y^3`, where `y = rsqrt(x)`, and `dy`
is the corresponding input gradient.

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func RsqrtGrad(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "RsqrtGrad",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Generate a single randomly distorted bounding box for an image.

Bounding box annotations are often supplied in addition to ground-truth labels
in image recognition or object localization tasks. A common technique for
training such a system is to randomly distort an image while preserving
its content, i.e. *data augmentation*. This Op outputs a randomly distorted
localization of an object, i.e. bounding box, given an `image_size`,
`bounding_boxes` and a series of constraints.
The output of this Op is a single bounding box that may be used to crop the
original image. The output is returned as 3 tensors: `begin`, `size` and
`bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
image. The latter may be supplied to `tf.image.draw_bounding_boxes` to visualize
what the bounding box looks like.
Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
height of the underlying image.
For example,
```python
    # Generate a single distorted bounding box.
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bounding_boxes)
    # Draw the bounding box in an image summary.
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                  bbox_for_draw)
    tf.image_summary('images_with_box', image_with_box)
    # Employ the bounding box to distort the image.
    distorted_image = tf.slice(image, begin, size)
```
Note that if no bounding box information is available, setting
`use_image_if_no_bounding_boxes = true` will assume there is a single implicit
bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
false and no bounding boxes are supplied, an error is raised.

_inputArg
 image_size:: .T  bounding_boxes:: . 
_outputArg

_outputArg
 begin::T  size::T  bboxes:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SampleDistortedBoundingBox(scope:Scope,  image_size:tf.Output,  bounding_boxes:tf.Output, ) -> ( begin:tf.Output,  size:tf.Output,  bboxes:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SampleDistortedBoundingBox",
        Input: [  image_size,  bounding_boxes, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) begin  op.Output(*) size  op.Output(*) bboxes )

}

/*
Saves the input tensors to disk.

The size of `tensor_names` must match the number of tensors in `data`. `data[i]`
is written to `filename` with name `tensor_names[i]`.
See also `SaveSlices`.

_inputArg
 filename:: .  tensor_names:: .  data:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Save(scope:Scope,  filename:tf.Output,  tensor_names:tf.Output,  data:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Save",
        Input: [  filename,  tensor_names,  data, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Saves input tensors slices to disk.

This is like `Save` except that tensors can be listed in the saved file as being
a slice of a larger tensor.  `shapes_and_slices` specifies the shape of the
larger tensor and the slice that this tensor covers. `shapes_and_slices` must
have as many elements as `tensor_names`.
Elements of the `shapes_and_slices` input must either be:
*  The empty string, in which case the corresponding tensor is
   saved normally.
*  A string of the form `dim0 dim1 ... dimN-1 slice-spec` where the
   `dimI` are the dimensions of the larger tensor and `slice-spec`
   specifies what part is covered by the tensor to save.
`slice-spec` itself is a `:`-separated list: `slice0:slice1:...:sliceN-1`
where each `sliceI` is either:
*  The string `-` meaning that the slice covers all indices of this dimension
*  `start,length` where `start` and `length` are integers.  In that
   case the slice covers `length` indices starting at `start`.
See also `Save`.

_inputArg
 filename:: .  tensor_names:: .  shapes_and_slices:: .  data:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SaveSlices(scope:Scope,  filename:tf.Output,  tensor_names:tf.Output,  shapes_and_slices:tf.Output,  data:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SaveSlices",
        Input: [  filename,  tensor_names,  shapes_and_slices,  data, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Saves tensors in V2 checkpoint format.

By default, saves the named tensors in full.  If the caller wishes to save
specific slices of full tensors, "shape_and_slices" should be non-empty strings
and correspondingly well-formed.

_inputArg
 prefix:: .  tensor_names:: .  shape_and_slices:: .  tensors:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SaveV2(scope:Scope,  prefix:tf.Output,  tensor_names:tf.Output,  shape_and_slices:tf.Output,  tensors:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SaveV2",
        Input: [  prefix,  tensor_names,  shape_and_slices,  tensors, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Outputs a `Summary` protocol buffer with scalar values.

The input `tags` and `values` must have the same shape.  The generated summary
has a summary value for each tag-value pair in `tags` and `values`.

_inputArg
 tags:: .  values:: .T 
_outputArg

_outputArg
 summary:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ScalarSummary(scope:Scope,  tags:tf.Output,  values:tf.Output, ) -> ( summary:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ScalarSummary",
        Input: [  tags,  values, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) summary )

}

/*
Adds sparse updates to a variable reference.

This operation computes
    # Scalar indices
    ref[indices, ...] += updates[...]
    # Vector indices (for each i)
    ref[indices[i], ...] += updates[i, ...]
    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]
This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.
Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions add.
Requires `updates.shape = indices.shape + ref.shape[1:]`.
<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/ScatterAdd.png" alt>
</div>

_inputArg
 ref:: .T  indices:: .Tindices  updates:: .T 
_outputArg

_outputArg
 output_ref::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ScatterAdd(scope:Scope,  ref:tf.Output,  indices:tf.Output,  updates:tf.Output, ) -> ( output_ref:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ScatterAdd",
        Input: [  ref,  indices,  updates, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_ref )

}

/*
Divides a variable reference by sparse updates.

This operation computes
```python
    # Scalar indices
    ref[indices, ...] /= updates[...]
    # Vector indices (for each i)
    ref[indices[i], ...] /= updates[i, ...]
    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] /= updates[i, ..., j, ...]
```
This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.
Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions divide.
Requires `updates.shape = indices.shape + ref.shape[1:]`.

_inputArg
 ref:: .T  indices:: .Tindices  updates:: .T 
_outputArg

_outputArg
 output_ref::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ScatterDiv(scope:Scope,  ref:tf.Output,  indices:tf.Output,  updates:tf.Output, ) -> ( output_ref:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ScatterDiv",
        Input: [  ref,  indices,  updates, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_ref )

}

/*
Multiplies sparse updates into a variable reference.

This operation computes
```python
    # Scalar indices
    ref[indices, ...] *= updates[...]
    # Vector indices (for each i)
    ref[indices[i], ...] *= updates[i, ...]
    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] *= updates[i, ..., j, ...]
```
This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.
Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions multiply.
Requires `updates.shape = indices.shape + ref.shape[1:]`.

_inputArg
 ref:: .T  indices:: .Tindices  updates:: .T 
_outputArg

_outputArg
 output_ref::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ScatterMul(scope:Scope,  ref:tf.Output,  indices:tf.Output,  updates:tf.Output, ) -> ( output_ref:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ScatterMul",
        Input: [  ref,  indices,  updates, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_ref )

}

/*
Scatter `updates` into a new (initially zero) tensor according to `indices`.

Creates a new tensor by applying sparse `updates` to individual
values or slices within a zero tensor of the given `shape` according to
indices.  This operator is the inverse of the [tf.gather_nd](#gather_nd)
operator which extracts values or slices from a given tensor.
**WARNING**: The order in which updates are applied is nondeterministic, so the
output will be nondeterministic if `indices` contains duplicates.
`indices` is an integer tensor containing indices into a new tensor of shape
`shape`.  The last dimension of `indices` can be at most the rank of `shape`:
    indices.shape[-1] <= shape.rank
The last dimension of `indices` corresponds to indices into elements
(if `indices.shape[-1] = shape.rank`) or slices
(if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
`shape`.  `updates` is a tensor with shape
    indices.shape[:-1] + shape[indices.shape[-1]:]
The simplest form of scatter is to insert individual elements in a tensor by
index. For example, say we want to insert 4 scattered elements in a rank-1
tensor with 8 elements.
<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd1.png" alt>
</div>
In Python, this scatter operation would look like this:
```python
    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    shape = tf.constant([8])
    scatter = tf.scatter_nd(indices, updates, shape)
    with tf.Session() as sess:
      print(sess.run(scatter))
```
The resulting tensor would look like this:
    [0, 11, 0, 10, 9, 0, 0, 12]
We can also, insert entire slices of a higher rank tensor all at once. For
example, if we wanted to insert two slices in the first dimension of a
rank-3 tensor with two matrices of new values.
<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd2.png" alt>
</div>
In Python, this scatter operation would look like this:
```python
    indices = tf.constant([[0], [2]])
    updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]],
                           [[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]]])
    shape = tf.constant([4, 4, 4])
    scatter = tf.scatter_nd(indices, updates, shape)
    with tf.Session() as sess:
      print(sess.run(scatter))
```
The resulting tensor would look like this:
    [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

_inputArg
 indices:: .Tindices  updates:: .T  shape:: .Tindices 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ScatterNd(scope:Scope,  indices:tf.Output,  updates:tf.Output,  shape:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ScatterNd",
        Input: [  indices,  updates,  shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Applies sparse addition between `updates` and individual values or slices

within a given variable according to `indices`.
`ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
`indices` must be integer tensor, containing indices into `ref`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `ref`.
`updates` is `Tensor` of rank `Q-1+P-K` with shape:
```
[d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
```
For example, say we want to add 4 scattered elements to a rank-1 tensor to 8
elements. In Python, that addition would look like this:
    ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    add = tf.scatter_nd_add(ref, indices, updates)
    with tf.Session() as sess:
      print sess.run(add)
The resulting update to ref would look like this:
    [1, 13, 3, 14, 14, 6, 7, 20]
See [tf.scatter_nd](#scatter_nd) for more details about how to make updates to
slices.

_inputArg
 ref:: .T  indices:: .Tindices  updates:: .T 
_outputArg

_outputArg
 output_ref::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ScatterNdAdd(scope:Scope,  ref:tf.Output,  indices:tf.Output,  updates:tf.Output, ) -> ( output_ref:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ScatterNdAdd",
        Input: [  ref,  indices,  updates, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_ref )

}

/*
Applies sparse subtraction between `updates` and individual values or slices

within a given variable according to `indices`.
`ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
`indices` must be integer tensor, containing indices into `ref`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `ref`.
`updates` is `Tensor` of rank `Q-1+P-K` with shape:
```
[d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
```
For example, say we want to subtract 4 scattered elements from a rank-1 tensor
with 8 elements. In Python, that subtraction would look like this:
    ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    sub = tf.scatter_nd_sub(ref, indices, updates)
    with tf.Session() as sess:
      print sess.run(sub)
The resulting update to ref would look like this:
    [1, -9, 3, -6, -4, 6, 7, -4]
See [tf.scatter_nd](#scatter_nd) for more details about how to make updates to
slices.

_inputArg
 ref:: .T  indices:: .Tindices  updates:: .T 
_outputArg

_outputArg
 output_ref::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ScatterNdSub(scope:Scope,  ref:tf.Output,  indices:tf.Output,  updates:tf.Output, ) -> ( output_ref:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ScatterNdSub",
        Input: [  ref,  indices,  updates, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_ref )

}

/*
Applies sparse `updates` to individual values or slices within a given

variable according to `indices`.
`ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
`indices` must be integer tensor, containing indices into `ref`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `ref`.
`updates` is `Tensor` of rank `Q-1+P-K` with shape:
```
[d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
```
For example, say we want to update 4 scattered elements to a rank-1 tensor to
8 elements. In Python, that update would look like this:
```python
    ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
    indices = tf.constant([[4], [3], [1] ,[7]])
    updates = tf.constant([9, 10, 11, 12])
    update = tf.scatter_nd_update(ref, indices, updates)
    with tf.Session() as sess:
      print sess.run(update)
```
The resulting update to ref would look like this:
    [1, 11, 3, 10, 9, 6, 7, 12]
See [tf.scatter_nd](#scatter_nd) for more details about how to make updates to
slices.

_inputArg
 ref:: .T  indices:: .Tindices  updates:: .T 
_outputArg

_outputArg
 output_ref::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ScatterNdUpdate(scope:Scope,  ref:tf.Output,  indices:tf.Output,  updates:tf.Output, ) -> ( output_ref:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ScatterNdUpdate",
        Input: [  ref,  indices,  updates, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_ref )

}

/*
Subtracts sparse updates to a variable reference.

```python
    # Scalar indices
    ref[indices, ...] -= updates[...]
    # Vector indices (for each i)
    ref[indices[i], ...] -= updates[i, ...]
    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]
```
This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.
Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their (negated) contributions add.
Requires `updates.shape = indices.shape + ref.shape[1:]`.
<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/ScatterSub.png" alt>
</div>

_inputArg
 ref:: .T  indices:: .Tindices  updates:: .T 
_outputArg

_outputArg
 output_ref::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ScatterSub(scope:Scope,  ref:tf.Output,  indices:tf.Output,  updates:tf.Output, ) -> ( output_ref:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ScatterSub",
        Input: [  ref,  indices,  updates, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_ref )

}

/*
Applies sparse updates to a variable reference.

This operation computes
```python
    # Scalar indices
    ref[indices, ...] = updates[...]
    # Vector indices (for each i)
    ref[indices[i], ...] = updates[i, ...]
    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]
```
This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.
If values in `ref` is to be updated more than once, because there are
duplicate entries in `indices`, the order at which the updates happen
for each value is undefined.
Requires `updates.shape = indices.shape + ref.shape[1:]`.
<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/ScatterUpdate.png" alt>
</div>

_inputArg
 ref:: .T  indices:: .Tindices  updates:: .T 
_outputArg

_outputArg
 output_ref::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ScatterUpdate(scope:Scope,  ref:tf.Output,  indices:tf.Output,  updates:tf.Output, ) -> ( output_ref:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ScatterUpdate",
        Input: [  ref,  indices,  updates, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_ref )

}

/*
Computes fingerprints of the input strings.


_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SdcaFprint(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SdcaFprint",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Distributed version of Stochastic Dual Coordinate Ascent (SDCA) optimizer for

linear models with L1 + L2 regularization. As global optimization objective is
strongly-convex, the optimizer optimizes the dual objective at each step. The
optimizer applies each update one example at a time. Examples are sampled
uniformly, and the optimizer is learning rate free and enjoys linear convergence
rate.
[Proximal Stochastic Dual Coordinate Ascent](http://arxiv.org/pdf/1211.2717v1.pdf).<br>
Shai Shalev-Shwartz, Tong Zhang. 2012
$$Loss Objective = \sum f_{i} (wx_{i}) + (l2 / 2) * |w|^2 + l1 * |w|$$
[Adding vs. Averaging in Distributed Primal-Dual Optimization](http://arxiv.org/abs/1502.03508).<br>
Chenxin Ma, Virginia Smith, Martin Jaggi, Michael I. Jordan,
Peter Richtarik, Martin Takac. 2015
[Stochastic Dual Coordinate Ascent with Adaptive Probabilities](https://arxiv.org/abs/1502.08053).<br>
Dominik Csiba, Zheng Qu, Peter Richtarik. 2015

_inputArg
 sparse_example_indices:: .  sparse_feature_indices:: .  sparse_feature_values:: .  dense_features:: .  example_weights:: .  example_labels:: .  sparse_indices:: .  sparse_weights:: .  dense_weights:: .  example_state_data:: . 
_outputArg

_outputArg
 out_example_state_data::  out_delta_sparse_weights::  out_delta_dense_weights:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SdcaOptimizer(scope:Scope,  sparse_example_indices:tf.Output,  sparse_feature_indices:tf.Output,  sparse_feature_values:tf.Output,  dense_features:tf.Output,  example_weights:tf.Output,  example_labels:tf.Output,  sparse_indices:tf.Output,  sparse_weights:tf.Output,  dense_weights:tf.Output,  example_state_data:tf.Output, ) -> ( out_example_state_data:tf.Output,  out_delta_sparse_weights:tf.Output,  out_delta_dense_weights:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SdcaOptimizer",
        Input: [  sparse_example_indices,  sparse_feature_indices,  sparse_feature_values,  dense_features,  example_weights,  example_labels,  sparse_indices,  sparse_weights,  dense_weights,  example_state_data, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out_example_state_data  op.Output(*) out_delta_sparse_weights  op.Output(*) out_delta_dense_weights )

}

/*
Applies L1 regularization shrink step on the parameters.


_inputArg
 weights:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SdcaShrinkL1(scope:Scope,  weights:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SdcaShrinkL1",
        Input: [  weights, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Computes the maximum along segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
segments.
Computes a tensor such that
\\(output_i = \max_j(data_j)\\) where `max` is over `j` such
that `segment_ids[j] == i`.
If the max is empty for a given segment ID `i`, `output[i] = 0`.
<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/SegmentMax.png" alt>
</div>

_inputArg
 data:: .T  segment_ids:: .Tindices 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SegmentMax(scope:Scope,  data:tf.Output,  segment_ids:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SegmentMax",
        Input: [  data,  segment_ids, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the mean along segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
segments.
Computes a tensor such that
\\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
over `j` such that `segment_ids[j] == i` and `N` is the total number of
values summed.
If the mean is empty for a given segment ID `i`, `output[i] = 0`.
<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/SegmentMean.png" alt>
</div>

_inputArg
 data:: .T  segment_ids:: .Tindices 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SegmentMean(scope:Scope,  data:tf.Output,  segment_ids:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SegmentMean",
        Input: [  data,  segment_ids, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the minimum along segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
segments.
Computes a tensor such that
\\(output_i = \min_j(data_j)\\) where `min` is over `j` such
that `segment_ids[j] == i`.
If the min is empty for a given segment ID `i`, `output[i] = 0`.
<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/SegmentMin.png" alt>
</div>

_inputArg
 data:: .T  segment_ids:: .Tindices 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SegmentMin(scope:Scope,  data:tf.Output,  segment_ids:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SegmentMin",
        Input: [  data,  segment_ids, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the product along segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
segments.
Computes a tensor such that
\\(output_i = \prod_j data_j\\) where the product is over `j` such
that `segment_ids[j] == i`.
If the product is empty for a given segment ID `i`, `output[i] = 1`.
<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/SegmentProd.png" alt>
</div>

_inputArg
 data:: .T  segment_ids:: .Tindices 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SegmentProd(scope:Scope,  data:tf.Output,  segment_ids:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SegmentProd",
        Input: [  data,  segment_ids, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the sum along segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
segments.
Computes a tensor such that
\\(output_i = \sum_j data_j\\) where sum is over `j` such
that `segment_ids[j] == i`.
If the sum is empty for a given segment ID `i`, `output[i] = 0`.
<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/SegmentSum.png" alt>
</div>

_inputArg
 data:: .T  segment_ids:: .Tindices 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SegmentSum(scope:Scope,  data:tf.Output,  segment_ids:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SegmentSum",
        Input: [  data,  segment_ids, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Selects elements from `t` or `e`, depending on `condition`.

The `t`, and `e` tensors must all have the same shape, and the
output will also have that shape.
The `condition` tensor must be a scalar if `t` and `e` are scalars.
If `t` and `e` are vectors or higher rank, then `condition` must be either a
scalar, a vector with size matching the first dimension of `t`, or must have
the same shape as `t`.
The `condition` tensor acts as a mask that chooses, based on the value at each
element, whether the corresponding element / row in the output should be
taken from `t` (if true) or `e` (if false).
If `condition` is a vector and `t` and `e` are higher rank matrices, then
it chooses which row (outer dimension) to copy from `t` and `e`.
If `condition` has the same shape as `t` and `e`, then it chooses which
element to copy from `t` and `e`.
For example:
```python
# 'condition' tensor is [[True,  False]
#                        [False, True]]
# 't' is [[1, 2],
#         [3, 4]]
# 'e' is [[5, 6],
#         [7, 8]]
select(condition, t, e)  # => [[1, 6], [7, 4]]
# 'condition' tensor is [True, False]
# 't' is [[1, 2],
#         [3, 4]]
# 'e' is [[5, 6],
#         [7, 8]]
select(condition, t, e) ==> [[1, 2],
                             [7, 8]]
```

_inputArg
 condition:: .  t:: .T  e:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Select(scope:Scope,  condition:tf.Output,  t:tf.Output,  e:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Select",
        Input: [  condition,  t,  e, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the Eigen Decomposition of a batch of square self-adjoint matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices, with the same constraints as the single matrix
SelfAdjointEig.
The result is a [..., M+1, M] matrix with [..., 0,:] containing the
eigenvalues, and subsequent [...,1:, :] containing the eigenvectors.

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SelfAdjointEig(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SelfAdjointEig",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the eigen decomposition of one or more square self-adjoint matrices.

Computes the eigenvalues and (optionally) eigenvectors of each inner matrix in
`input` such that `input[..., :, :] = v[..., :, :] * diag(e[..., :])`.
```python
# a is a tensor.
# e is a tensor of eigenvalues.
# v is a tensor of eigenvectors.
e, v = self_adjoint_eig(a)
e = self_adjoint_eig(a, compute_v=False)
```

_inputArg
 input:: .T 
_outputArg

_outputArg
 e::T  v::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SelfAdjointEigV2(scope:Scope,  input:tf.Output, ) -> ( e:tf.Output,  v:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SelfAdjointEigV2",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) e  op.Output(*) v )

}

/*
Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` string `Tensor`.

The `SparseTensor` must have rank `R` greater than 1, and the first dimension
is treated as the minibatch dimension.  Elements of the `SparseTensor`
must be sorted in increasing order of this first dimension.  The serialized
`SparseTensor` objects going into each row of `serialized_sparse` will have
rank `R-1`.
The minibatch size `N` is extracted from `sparse_shape[0]`.

_inputArg
 sparse_indices:: .  sparse_values:: .T  sparse_shape:: . 
_outputArg

_outputArg
 serialized_sparse:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SerializeManySparse(scope:Scope,  sparse_indices:tf.Output,  sparse_values:tf.Output,  sparse_shape:tf.Output, ) -> ( serialized_sparse:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SerializeManySparse",
        Input: [  sparse_indices,  sparse_values,  sparse_shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) serialized_sparse )

}

/*
Serialize a `SparseTensor` into a string 3-vector (1-D `Tensor`) object.


_inputArg
 sparse_indices:: .  sparse_values:: .T  sparse_shape:: . 
_outputArg

_outputArg
 serialized_sparse:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SerializeSparse(scope:Scope,  sparse_indices:tf.Output,  sparse_values:tf.Output,  sparse_shape:tf.Output, ) -> ( serialized_sparse:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SerializeSparse",
        Input: [  sparse_indices,  sparse_values,  sparse_shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) serialized_sparse )

}

/*
Number of unique elements along last dimension of input `set`.

Input `set` is a `SparseTensor` represented by `set_indices`, `set_values`,
and `set_shape`. The last dimension contains values in a set, duplicates are
allowed but ignored.
If `validate_indices` is `True`, this op validates the order and range of `set`
indices.

_inputArg
 set_indices:: .  set_values:: .T  set_shape:: . 
_outputArg

_outputArg
 size:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SetSize(scope:Scope,  set_indices:tf.Output,  set_values:tf.Output,  set_shape:tf.Output, ) -> ( size:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SetSize",
        Input: [  set_indices,  set_values,  set_shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) size )

}

/*
Returns the shape of a tensor.

This operation returns a 1-D integer tensor representing the shape of `input`.
For example:
```
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
shape(t) ==> [2, 2, 3]
```

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::out_type 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Shape(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Shape",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Returns shape of tensors.

This operation returns N 1-D integer tensors representing shape of `input[i]s`.

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::out_type 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ShapeN(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ShapeN",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Generate a sharded filename. The filename is printf formatted as

   %s-%05d-of-%05d, basename, shard, num_shards.

_inputArg
 basename:: .  shard:: .  num_shards:: . 
_outputArg

_outputArg
 filename:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ShardedFilename(scope:Scope,  basename:tf.Output,  shard:tf.Output,  num_shards:tf.Output, ) -> ( filename:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ShardedFilename",
        Input: [  basename,  shard,  num_shards, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) filename )

}

/*
Generate a glob pattern matching all sharded file names.


_inputArg
 basename:: .  num_shards:: . 
_outputArg

_outputArg
 filename:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ShardedFilespec(scope:Scope,  basename:tf.Output,  num_shards:tf.Output, ) -> ( filename:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ShardedFilespec",
        Input: [  basename,  num_shards, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) filename )

}

/*
Creates a dataset that shuffles elements from `input_dataset` pseudorandomly.


_inputArg
 input_dataset:: .  buffer_size:: .  seed:: .  seed2:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ShuffleDataset(scope:Scope,  input_dataset:tf.Output,  buffer_size:tf.Output,  seed:tf.Output,  seed2:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ShuffleDataset",
        Input: [  input_dataset,  buffer_size,  seed,  seed2, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Computes sigmoid of `x` element-wise.

Specifically, `y = 1 / (1 + exp(-x))`.

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Sigmoid(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Sigmoid",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Computes the gradient of the sigmoid of `x` wrt its input.

Specifically, `grad = dy * y * (1 - y)`, where `y = sigmoid(x)`, and
`dy` is the corresponding input gradient.

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SigmoidGrad(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SigmoidGrad",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Returns an element-wise indication of the sign of a number.

`y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.
For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Sign(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Sign",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Computes sin of x element-wise.


_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Sin(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Sin",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Returns the size of a tensor.

This operation returns an integer representing the number of elements in
`input`.
For example:
```
# 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
size(t) ==> 12
```

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::out_type 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Size(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Size",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Creates a dataset that skips `count` elements from the `input_dataset`.


_inputArg
 input_dataset:: .  count:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SkipDataset(scope:Scope,  input_dataset:tf.Output,  count:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SkipDataset",
        Input: [  input_dataset,  count, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Parses a text file and creates a batch of examples.


_inputArg
_outputArg

_outputArg
 vocab_word::  vocab_freq::  words_per_epoch::  current_epoch::  total_words_processed::  examples::  labels:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Skipgram(scope:Scope, ) -> ( vocab_word:tf.Output,  vocab_freq:tf.Output,  words_per_epoch:tf.Output,  current_epoch:tf.Output,  total_words_processed:tf.Output,  examples:tf.Output,  labels:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Skipgram",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) vocab_word  op.Output(*) vocab_freq  op.Output(*) words_per_epoch  op.Output(*) current_epoch  op.Output(*) total_words_processed  op.Output(*) examples  op.Output(*) labels )

}

/*
Return a slice from 'input'.

The output tensor is a tensor with dimensions described by 'size'
whose values are extracted from 'input' starting at the offsets in
'begin'.
*Requirements*:
  0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)

_inputArg
 input:: .T  begin:: .Index  size:: .Index 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Slice(scope:Scope,  input:tf.Output,  begin:tf.Output,  size:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Slice",
        Input: [  input,  begin,  size, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes softmax activations.

For each batch `i` and class `j` we have
    softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))

_inputArg
 logits:: .T 
_outputArg

_outputArg
 softmax::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Softmax(scope:Scope,  logits:tf.Output, ) -> ( softmax:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Softmax",
        Input: [  logits, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) softmax )

}

/*
Computes softmax cross entropy cost and gradients to backpropagate.

Inputs are the logits, not probabilities.

_inputArg
 features:: .T  labels:: .T 
_outputArg

_outputArg
 loss::T  backprop::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SoftmaxCrossEntropyWithLogits(scope:Scope,  features:tf.Output,  labels:tf.Output, ) -> ( loss:tf.Output,  backprop:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SoftmaxCrossEntropyWithLogits",
        Input: [  features,  labels, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) loss  op.Output(*) backprop )

}

/*
Computes softplus: `log(exp(features) + 1)`.


_inputArg
 features:: .T 
_outputArg

_outputArg
 activations::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Softplus(scope:Scope,  features:tf.Output, ) -> ( activations:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Softplus",
        Input: [  features, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) activations )

}

/*
Computes softplus gradients for a softplus operation.


_inputArg
 gradients:: .T  features:: .T 
_outputArg

_outputArg
 backprops::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SoftplusGrad(scope:Scope,  gradients:tf.Output,  features:tf.Output, ) -> ( backprops:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SoftplusGrad",
        Input: [  gradients,  features, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) backprops )

}

/*
Computes softsign: `features / (abs(features) + 1)`.


_inputArg
 features:: .T 
_outputArg

_outputArg
 activations::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Softsign(scope:Scope,  features:tf.Output, ) -> ( activations:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Softsign",
        Input: [  features, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) activations )

}

/*
Computes softsign gradients for a softsign operation.


_inputArg
 gradients:: .T  features:: .T 
_outputArg

_outputArg
 backprops::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SoftsignGrad(scope:Scope,  gradients:tf.Output,  features:tf.Output, ) -> ( backprops:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SoftsignGrad",
        Input: [  gradients,  features, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) backprops )

}

/*
SpaceToBatch for 4-D tensors of type T.

This is a legacy version of the more general SpaceToBatchND.
Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
More specifically, this op outputs a copy of the input tensor where values from
the `height` and `width` dimensions are moved to the `batch` dimension. After
the zero-padding, both `height` and `width` of the input must be divisible by the
block size.

_inputArg
 input:: .T  paddings:: .Tpaddings 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SpaceToBatch(scope:Scope,  input:tf.Output,  paddings:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SpaceToBatch",
        Input: [  input,  paddings, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
SpaceToBatch for N-D tensors of type T.

This operation divides "spatial" dimensions `[1, ..., M]` of the input into a
grid of blocks of shape `block_shape`, and interleaves these blocks with the
"batch" dimension (0) such that in the output, the spatial dimensions
`[1, ..., M]` correspond to the position within the grid, and the batch
dimension combines both the position within a spatial block and the original
batch position.  Prior to division into blocks, the spatial dimensions of the
input are optionally zero padded according to `paddings`.  See below for a
precise description.

_inputArg
 input:: .T  block_shape:: .Tblock_shape  paddings:: .Tpaddings 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SpaceToBatchND(scope:Scope,  input:tf.Output,  block_shape:tf.Output,  paddings:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SpaceToBatchND",
        Input: [  input,  block_shape,  paddings, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
SpaceToDepth for tensors of type T.

Rearranges blocks of spatial data, into depth. More specifically,
this op outputs a copy of the input tensor where values from the `height`
and `width` dimensions are moved to the `depth` dimension.
The attr `block_size` indicates the input block size and how the data is moved.
  * Non-overlapping blocks of size `block_size x block size` are rearranged
    into depth at each location.
  * The depth of the output tensor is `input_depth * block_size * block_size`.
  * The input tensor's height and width must be divisible by block_size.
That is, assuming the input is in the shape:
`[batch, height, width, depth]`,
the shape of the output will be:
`[batch, height/block_size, width/block_size, depth*block_size*block_size]`
This operation requires that the input tensor be of rank 4, and that
`block_size` be >=1 and a divisor of both the input `height` and `width`.
This operation is useful for resizing the activations between convolutions
(but keeping all data), e.g. instead of pooling. It is also useful for training
purely convolutional models.
For example, given this input of shape `[1, 2, 2, 1]`, and block_size of 2:
```
x = [[[[1], [2]],
      [[3], [4]]]]
```
This operation will output a tensor of shape `[1, 1, 1, 4]`:
```
[[[[1, 2, 3, 4]]]]
```
Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
the corresponding output will have a single element (i.e. width and height are
both 1) and will have a depth of 4 channels (1 * block_size * block_size).
The output element shape is `[1, 1, 4]`.
For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.
```
x = [[[[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]]]]
```
This operation, for block_size of 2, will return the following tensor of shape
`[1, 1, 1, 12]`
```
[[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
```
Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:
```
x = [[[[1],   [2],  [5],  [6]],
      [[3],   [4],  [7],  [8]],
      [[9],  [10], [13],  [14]],
      [[11], [12], [15],  [16]]]]
```
the operator will return the following tensor of shape `[1 2 2 4]`:
```
x = [[[[1, 2, 3, 4],
       [5, 6, 7, 8]],
      [[9, 10, 11, 12],
       [13, 14, 15, 16]]]]
```

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SpaceToDepth(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SpaceToDepth",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Applies a sparse gradient to a given accumulator.

Does not add if local_step is smaller than the accumulator's
global_step.

_inputArg
 handle:: .  local_step:: .  gradient_indices:: .  gradient_values:: .dtype  gradient_shape:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseAccumulatorApplyGradient(scope:Scope,  handle:tf.Output,  local_step:tf.Output,  gradient_indices:tf.Output,  gradient_values:tf.Output,  gradient_shape:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseAccumulatorApplyGradient",
        Input: [  handle,  local_step,  gradient_indices,  gradient_values,  gradient_shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Extracts the average sparse gradient in a SparseConditionalAccumulator.

The op will blocks until sufficient (i.e., more than num_required)
gradients have been accumulated. If the accumulator has already
aggregated more than num_required gradients, it will return its
average of the accumulated gradients.  Also automatically increments
the recorded global_step in the accumulator by 1, and resets the
aggregate to 0.

_inputArg
 handle:: .  num_required:: . 
_outputArg

_outputArg
 indices::  values::dtype  shape:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseAccumulatorTakeGradient(scope:Scope,  handle:tf.Output,  num_required:tf.Output, ) -> ( indices:tf.Output,  values:tf.Output,  shape:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseAccumulatorTakeGradient",
        Input: [  handle,  num_required, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) indices  op.Output(*) values  op.Output(*) shape )

}

/*
Adds two `SparseTensor` objects to produce another `SparseTensor`.

The input `SparseTensor` objects' indices are assumed ordered in standard
lexicographic order.  If this is not the case, before this step run
`SparseReorder` to restore index ordering.
By default, if two values sum to zero at some index, the output `SparseTensor`
would still include that particular location in its index, storing a zero in the
corresponding value slot.  To override this, callers can specify `thresh`,
indicating that if the sum has a magnitude strictly smaller than `thresh`, its
corresponding value and index would then not be included.  In particular,
`thresh == 0` (default) means everything is kept and actual thresholding happens
only for a positive value.
In the following shapes, `nnz` is the count after taking `thresh` into account.

_inputArg
 a_indices:: .  a_values:: .T  a_shape:: .  b_indices:: .  b_values:: .T  b_shape:: .  thresh:: .Treal 
_outputArg

_outputArg
 sum_indices::  sum_values::T  sum_shape:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseAdd(scope:Scope,  a_indices:tf.Output,  a_values:tf.Output,  a_shape:tf.Output,  b_indices:tf.Output,  b_values:tf.Output,  b_shape:tf.Output,  thresh:tf.Output, ) -> ( sum_indices:tf.Output,  sum_values:tf.Output,  sum_shape:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseAdd",
        Input: [  a_indices,  a_values,  a_shape,  b_indices,  b_values,  b_shape,  thresh, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) sum_indices  op.Output(*) sum_values  op.Output(*) sum_shape )

}

/*
The gradient operator for the SparseAdd op.

The SparseAdd op calculates A + B, where A, B, and the sum are all represented
as `SparseTensor` objects.  This op takes in the upstream gradient w.r.t.
non-empty values of the sum, and outputs the gradients w.r.t. the non-empty
values of A and B.

_inputArg
 backprop_val_grad:: .T  a_indices:: .  b_indices:: .  sum_indices:: . 
_outputArg

_outputArg
 a_val_grad::T  b_val_grad::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseAddGrad(scope:Scope,  backprop_val_grad:tf.Output,  a_indices:tf.Output,  b_indices:tf.Output,  sum_indices:tf.Output, ) -> ( a_val_grad:tf.Output,  b_val_grad:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseAddGrad",
        Input: [  backprop_val_grad,  a_indices,  b_indices,  sum_indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) a_val_grad  op.Output(*) b_val_grad )

}

/*
var: Should be from a Variable().


_inputArg
 var:: .T  accum:: .T  accum_update:: .T  lr:: .T  rho:: .T  epsilon:: .T  grad:: .T  indices:: .Tindices 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseApplyAdadelta(scope:Scope,  var:tf.Output,  accum:tf.Output,  accum_update:tf.Output,  lr:tf.Output,  rho:tf.Output,  epsilon:tf.Output,  grad:tf.Output,  indices:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseApplyAdadelta",
        Input: [  var,  accum,  accum_update,  lr,  rho,  epsilon,  grad,  indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Update relevant entries in '*var' and '*accum' according to the adagrad scheme.

That is for rows we have grad for, we update var and accum as follows:
accum += grad * grad
var -= lr * grad * (1 / sqrt(accum))

_inputArg
 var:: .T  accum:: .T  lr:: .T  grad:: .T  indices:: .Tindices 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseApplyAdagrad(scope:Scope,  var:tf.Output,  accum:tf.Output,  lr:tf.Output,  grad:tf.Output,  indices:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseApplyAdagrad",
        Input: [  var,  accum,  lr,  grad,  indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Update entries in '*var' and '*accum' according to the proximal adagrad scheme.


_inputArg
 var:: .T  gradient_accumulator:: .T  gradient_squared_accumulator:: .T  grad:: .T  indices:: .Tindices  lr:: .T  l1:: .T  l2:: .T  global_step:: . 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseApplyAdagradDA(scope:Scope,  var:tf.Output,  gradient_accumulator:tf.Output,  gradient_squared_accumulator:tf.Output,  grad:tf.Output,  indices:tf.Output,  lr:tf.Output,  l1:tf.Output,  l2:tf.Output,  global_step:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseApplyAdagradDA",
        Input: [  var,  gradient_accumulator,  gradient_squared_accumulator,  grad,  indices,  lr,  l1,  l2,  global_step, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Update '*var' according to the centered RMSProp algorithm.

The centered RMSProp algorithm uses an estimate of the centered second moment
(i.e., the variance) for normalization, as opposed to regular RMSProp, which
uses the (uncentered) second moment. This often helps with training, but is
slightly more expensive in terms of computation and memory.
Note that in dense implementation of this algorithm, mg, ms, and mom will
update even if the grad is zero, but in this sparse implementation, mg, ms,
and mom will not update in iterations during which the grad is zero.
mean_square = decay * mean_square + (1-decay) * gradient ** 2
mean_grad = decay * mean_grad + (1-decay) * gradient
Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)
ms <- rho * ms_{t-1} + (1-rho) * grad * grad
mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
var <- var - mom

_inputArg
 var:: .T  mg:: .T  ms:: .T  mom:: .T  lr:: .T  rho:: .T  momentum:: .T  epsilon:: .T  grad:: .T  indices:: .Tindices 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseApplyCenteredRMSProp(scope:Scope,  var:tf.Output,  mg:tf.Output,  ms:tf.Output,  mom:tf.Output,  lr:tf.Output,  rho:tf.Output,  momentum:tf.Output,  epsilon:tf.Output,  grad:tf.Output,  indices:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseApplyCenteredRMSProp",
        Input: [  var,  mg,  ms,  mom,  lr,  rho,  momentum,  epsilon,  grad,  indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Update relevant entries in '*var' according to the Ftrl-proximal scheme.

That is for rows we have grad for, we update var, accum and linear as follows:
accum_new = accum + grad * grad
linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new

_inputArg
 var:: .T  accum:: .T  linear:: .T  grad:: .T  indices:: .Tindices  lr:: .T  l1:: .T  l2:: .T  lr_power:: .T 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseApplyFtrl(scope:Scope,  var:tf.Output,  accum:tf.Output,  linear:tf.Output,  grad:tf.Output,  indices:tf.Output,  lr:tf.Output,  l1:tf.Output,  l2:tf.Output,  lr_power:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseApplyFtrl",
        Input: [  var,  accum,  linear,  grad,  indices,  lr,  l1,  l2,  lr_power, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Update relevant entries in '*var' and '*accum' according to the momentum scheme.

Set use_nesterov = True if you want to use Nesterov momentum.
That is for rows we have grad for, we update var and accum as follows:
accum = accum * momentum + grad
var -= lr * accum

_inputArg
 var:: .T  accum:: .T  lr:: .T  grad:: .T  indices:: .Tindices  momentum:: .T 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseApplyMomentum(scope:Scope,  var:tf.Output,  accum:tf.Output,  lr:tf.Output,  grad:tf.Output,  indices:tf.Output,  momentum:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseApplyMomentum",
        Input: [  var,  accum,  lr,  grad,  indices,  momentum, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Sparse update entries in '*var' and '*accum' according to FOBOS algorithm.

That is for rows we have grad for, we update var and accum as follows:
accum += grad * grad
prox_v = var
prox_v -= lr * grad * (1 / sqrt(accum))
var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}

_inputArg
 var:: .T  accum:: .T  lr:: .T  l1:: .T  l2:: .T  grad:: .T  indices:: .Tindices 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseApplyProximalAdagrad(scope:Scope,  var:tf.Output,  accum:tf.Output,  lr:tf.Output,  l1:tf.Output,  l2:tf.Output,  grad:tf.Output,  indices:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseApplyProximalAdagrad",
        Input: [  var,  accum,  lr,  l1,  l2,  grad,  indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Sparse update '*var' as FOBOS algorithm with fixed learning rate.

That is for rows we have grad for, we update var as follows:
prox_v = var - alpha * grad
var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

_inputArg
 var:: .T  alpha:: .T  l1:: .T  l2:: .T  grad:: .T  indices:: .Tindices 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseApplyProximalGradientDescent(scope:Scope,  var:tf.Output,  alpha:tf.Output,  l1:tf.Output,  l2:tf.Output,  grad:tf.Output,  indices:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseApplyProximalGradientDescent",
        Input: [  var,  alpha,  l1,  l2,  grad,  indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Update '*var' according to the RMSProp algorithm.

Note that in dense implementation of this algorithm, ms and mom will
update even if the grad is zero, but in this sparse implementation, ms
and mom will not update in iterations during which the grad is zero.
mean_square = decay * mean_square + (1-decay) * gradient ** 2
Delta = learning_rate * gradient / sqrt(mean_square + epsilon)
ms <- rho * ms_{t-1} + (1-rho) * grad * grad
mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
var <- var - mom

_inputArg
 var:: .T  ms:: .T  mom:: .T  lr:: .T  rho:: .T  momentum:: .T  epsilon:: .T  grad:: .T  indices:: .Tindices 
_outputArg

_outputArg
 out::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseApplyRMSProp(scope:Scope,  var:tf.Output,  ms:tf.Output,  mom:tf.Output,  lr:tf.Output,  rho:tf.Output,  momentum:tf.Output,  epsilon:tf.Output,  grad:tf.Output,  indices:tf.Output, ) -> ( out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseApplyRMSProp",
        Input: [  var,  ms,  mom,  lr,  rho,  momentum,  epsilon,  grad,  indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) out )

}

/*
Concatenates a list of `SparseTensor` along the specified dimension.

Concatenation is with respect to the dense versions of these sparse tensors.
It is assumed that each input is a `SparseTensor` whose elements are ordered
along increasing dimension number.
All inputs' shapes must match, except for the concat dimension.  The
`indices`, `values`, and `shapes` lists must have the same length.
The output shape is identical to the inputs', except along the concat
dimension, where it is the sum of the inputs' sizes along that dimension.
The output elements will be resorted to preserve the sort order along
increasing dimension number.
This op runs in `O(M log M)` time, where `M` is the total number of non-empty
values across all inputs. This is due to the need for an internal sort in
order to concatenate efficiently across an arbitrary dimension.
For example, if `concat_dim = 1` and the inputs are
    sp_inputs[0]: shape = [2, 3]
    [0, 2]: "a"
    [1, 0]: "b"
    [1, 1]: "c"
    sp_inputs[1]: shape = [2, 4]
    [0, 1]: "d"
    [0, 2]: "e"
then the output will be
    shape = [2, 7]
    [0, 2]: "a"
    [0, 4]: "d"
    [0, 5]: "e"
    [1, 0]: "b"
    [1, 1]: "c"
Graphically this is equivalent to doing
    [    a] concat [  d e  ] = [    a   d e  ]
    [b c  ]        [       ]   [b c          ]

_inputArg
 indices:: .  values:: .T  shapes:: . 
_outputArg

_outputArg
 output_indices::  output_values::T  output_shape:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseConcat(scope:Scope,  indices:tf.Output,  values:tf.Output,  shapes:tf.Output, ) -> ( output_indices:tf.Output,  output_values:tf.Output,  output_shape:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseConcat",
        Input: [  indices,  values,  shapes, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_indices  op.Output(*) output_values  op.Output(*) output_shape )

}

/*
A conditional accumulator for aggregating sparse gradients.

The accumulator accepts gradients marked with local_step greater or
equal to the most recent global_step known to the accumulator. The
average can be extracted from the accumulator, provided sufficient
gradients have been accumulated. Extracting the average automatically
resets the aggregate to 0, and increments the global_step recorded by
the accumulator.

_inputArg
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseConditionalAccumulator(scope:Scope, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseConditionalAccumulator",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Generates sparse cross from a list of sparse and dense tensors.

The op takes two lists, one of 2D `SparseTensor` and one of 2D `Tensor`, each
representing features of one feature column. It outputs a 2D `SparseTensor` with
the batchwise crosses of these features.
For example, if the inputs are
    inputs[0]: SparseTensor with shape = [2, 2]
    [0, 0]: "a"
    [1, 0]: "b"
    [1, 1]: "c"
    inputs[1]: SparseTensor with shape = [2, 1]
    [0, 0]: "d"
    [1, 0]: "e"
    inputs[2]: Tensor [["f"], ["g"]]
then the output will be
    shape = [2, 2]
    [0, 0]: "a_X_d_X_f"
    [1, 0]: "b_X_e_X_g"
    [1, 1]: "c_X_e_X_g"
if hashed_output=true then the output will be
    shape = [2, 2]
    [0, 0]: FingerprintCat64(
                Fingerprint64("f"), FingerprintCat64(
                    Fingerprint64("d"), Fingerprint64("a")))
    [1, 0]: FingerprintCat64(
                Fingerprint64("g"), FingerprintCat64(
                    Fingerprint64("e"), Fingerprint64("b")))
    [1, 1]: FingerprintCat64(
                Fingerprint64("g"), FingerprintCat64(
                    Fingerprint64("e"), Fingerprint64("c")))

_inputArg
 indices:: .  values:: .  shapes:: .  dense_inputs:: . 
_outputArg

_outputArg
 output_indices::  output_values::out_type  output_shape:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseCross(scope:Scope,  indices:tf.Output,  values:tf.Output,  shapes:tf.Output,  dense_inputs:tf.Output, ) -> ( output_indices:tf.Output,  output_values:tf.Output,  output_shape:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseCross",
        Input: [  indices,  values,  shapes,  dense_inputs, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_indices  op.Output(*) output_values  op.Output(*) output_shape )

}

/*
Adds up a SparseTensor and a dense Tensor, using these special rules:

(1) Broadcasts the dense side to have the same shape as the sparse side, if
    eligible;
(2) Then, only the dense values pointed to by the indices of the SparseTensor
    participate in the cwise addition.
By these rules, the result is a logical SparseTensor with exactly the same
indices and shape, but possibly with different non-zero values.  The output of
this Op is the resultant non-zero values.

_inputArg
 sp_indices:: .  sp_values:: .T  sp_shape:: .  dense:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseDenseCwiseAdd(scope:Scope,  sp_indices:tf.Output,  sp_values:tf.Output,  sp_shape:tf.Output,  dense:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseDenseCwiseAdd",
        Input: [  sp_indices,  sp_values,  sp_shape,  dense, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Component-wise divides a SparseTensor by a dense Tensor.

*Limitation*: this Op only broadcasts the dense side to the sparse side, but not
the other direction.

_inputArg
 sp_indices:: .  sp_values:: .T  sp_shape:: .  dense:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseDenseCwiseDiv(scope:Scope,  sp_indices:tf.Output,  sp_values:tf.Output,  sp_shape:tf.Output,  dense:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseDenseCwiseDiv",
        Input: [  sp_indices,  sp_values,  sp_shape,  dense, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Component-wise multiplies a SparseTensor by a dense Tensor.

The output locations corresponding to the implicitly zero elements in the sparse
tensor will be zero (i.e., will not take up storage space), regardless of the
contents of the dense tensor (even if it's +/-INF and that INF*0 == NaN).
*Limitation*: this Op only broadcasts the dense side to the sparse side, but not
the other direction.

_inputArg
 sp_indices:: .  sp_values:: .T  sp_shape:: .  dense:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseDenseCwiseMul(scope:Scope,  sp_indices:tf.Output,  sp_values:tf.Output,  sp_shape:tf.Output,  dense:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseDenseCwiseMul",
        Input: [  sp_indices,  sp_values,  sp_shape,  dense, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Fills empty rows in the input 2-D `SparseTensor` with a default value.

The input `SparseTensor` is represented via the tuple of inputs
(`indices`, `values`, `dense_shape`).  The output `SparseTensor` has the
same `dense_shape` but with indices `output_indices` and values
`output_values`.
This op inserts a single entry for every row that doesn't have any values.
The index is created as `[row, 0, ..., 0]` and the inserted value
is `default_value`.
For example, suppose `sp_input` has shape `[5, 6]` and non-empty values:
    [0, 1]: a
    [0, 3]: b
    [2, 0]: c
    [3, 1]: d
Rows 1 and 4 are empty, so the output will be of shape `[5, 6]` with values:
    [0, 1]: a
    [0, 3]: b
    [1, 0]: default_value
    [2, 0]: c
    [3, 1]: d
    [4, 0]: default_value
The output `SparseTensor` will be in row-major order and will have the
same shape as the input.
This op also returns an indicator vector shaped `[dense_shape[0]]` such that
    empty_row_indicator[i] = True iff row i was an empty row.
And a reverse index map vector shaped `[indices.shape[0]]` that is used during
backpropagation,
    reverse_index_map[j] = out_j s.t. indices[j, :] == output_indices[out_j, :]

_inputArg
 indices:: .  values:: .T  dense_shape:: .  default_value:: .T 
_outputArg

_outputArg
 output_indices::  output_values::T  empty_row_indicator::  reverse_index_map:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseFillEmptyRows(scope:Scope,  indices:tf.Output,  values:tf.Output,  dense_shape:tf.Output,  default_value:tf.Output, ) -> ( output_indices:tf.Output,  output_values:tf.Output,  empty_row_indicator:tf.Output,  reverse_index_map:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseFillEmptyRows",
        Input: [  indices,  values,  dense_shape,  default_value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_indices  op.Output(*) output_values  op.Output(*) empty_row_indicator  op.Output(*) reverse_index_map )

}

/*
The gradient of SparseFillEmptyRows.

Takes vectors reverse_index_map, shaped `[N]`, and grad_values,
shaped `[N_full]`, where `N_full >= N` and copies data into either
`d_values` or `d_default_value`.  Here `d_values` is shaped `[N]` and
`d_default_value` is a scalar.
  d_values[j] = grad_values[reverse_index_map[j]]
  d_default_value = sum_{k : 0 .. N_full - 1} (
     grad_values[k] * 1{k not in reverse_index_map})

_inputArg
 reverse_index_map:: .  grad_values:: .T 
_outputArg

_outputArg
 d_values::T  d_default_value::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseFillEmptyRowsGrad(scope:Scope,  reverse_index_map:tf.Output,  grad_values:tf.Output, ) -> ( d_values:tf.Output,  d_default_value:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseFillEmptyRowsGrad",
        Input: [  reverse_index_map,  grad_values, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) d_values  op.Output(*) d_default_value )

}

/*
Multiply matrix "a" by matrix "b".

The inputs must be two-dimensional matrices and the inner dimension of "a" must
match the outer dimension of "b". This op is optimized for the case where at
least one of "a" or "b" is sparse. The breakeven for using this versus a dense
matrix multiply on one platform was 30% zero values in the sparse matrix.

_inputArg
 a:: .Ta  b:: .Tb 
_outputArg

_outputArg
 product:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseMatMul(scope:Scope,  a:tf.Output,  b:tf.Output, ) -> ( product:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseMatMul",
        Input: [  a,  b, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) product )

}

/*
Computes the sum of elements across dimensions of a SparseTensor.

This Op takes a SparseTensor and is the sparse counterpart to
`tf.reduce_sum()`.  In particular, this Op also returns a dense `Tensor`
instead of a sparse one.
Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
with length 1.
If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
with a single element is returned.  Additionally, the axes can be negative,
which are interpreted according to the indexing rules in Python.

_inputArg
 input_indices:: .  input_values:: .T  input_shape:: .  reduction_axes:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseReduceSum(scope:Scope,  input_indices:tf.Output,  input_values:tf.Output,  input_shape:tf.Output,  reduction_axes:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseReduceSum",
        Input: [  input_indices,  input_values,  input_shape,  reduction_axes, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the sum of elements across dimensions of a SparseTensor.

This Op takes a SparseTensor and is the sparse counterpart to
`tf.reduce_sum()`.  In contrast to SparseReduceSum, this Op returns a
SparseTensor.
Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
with length 1.
If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
with a single element is returned.  Additionally, the axes can be negative,
which are interpreted according to the indexing rules in Python.

_inputArg
 input_indices:: .  input_values:: .T  input_shape:: .  reduction_axes:: . 
_outputArg

_outputArg
 output_indices::  output_values::T  output_shape:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseReduceSumSparse(scope:Scope,  input_indices:tf.Output,  input_values:tf.Output,  input_shape:tf.Output,  reduction_axes:tf.Output, ) -> ( output_indices:tf.Output,  output_values:tf.Output,  output_shape:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseReduceSumSparse",
        Input: [  input_indices,  input_values,  input_shape,  reduction_axes, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_indices  op.Output(*) output_values  op.Output(*) output_shape )

}

/*
Reorders a SparseTensor into the canonical, row-major ordering.

Note that by convention, all sparse ops preserve the canonical ordering along
increasing dimension number. The only time ordering can be violated is during
manual manipulation of the indices and values vectors to add entries.
Reordering does not affect the shape of the SparseTensor.
If the tensor has rank `R` and `N` non-empty values, `input_indices` has
shape `[N, R]`, input_values has length `N`, and input_shape has length `R`.

_inputArg
 input_indices:: .  input_values:: .T  input_shape:: . 
_outputArg

_outputArg
 output_indices::  output_values::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseReorder(scope:Scope,  input_indices:tf.Output,  input_values:tf.Output,  input_shape:tf.Output, ) -> ( output_indices:tf.Output,  output_values:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseReorder",
        Input: [  input_indices,  input_values,  input_shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_indices  op.Output(*) output_values )

}

/*
Reshapes a SparseTensor to represent values in a new dense shape.

This operation has the same semantics as reshape on the represented dense
tensor.  The `input_indices` are recomputed based on the requested `new_shape`.
If one component of `new_shape` is the special value -1, the size of that
dimension is computed so that the total dense size remains constant.  At
most one component of `new_shape` can be -1.  The number of dense elements
implied by `new_shape` must be the same as the number of dense elements
originally implied by `input_shape`.
Reshaping does not affect the order of values in the SparseTensor.
If the input tensor has rank `R_in` and `N` non-empty values, and `new_shape`
has length `R_out`, then `input_indices` has shape `[N, R_in]`,
`input_shape` has length `R_in`, `output_indices` has shape `[N, R_out]`, and
`output_shape` has length `R_out`.

_inputArg
 input_indices:: .  input_shape:: .  new_shape:: . 
_outputArg

_outputArg
 output_indices::  output_shape:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseReshape(scope:Scope,  input_indices:tf.Output,  input_shape:tf.Output,  new_shape:tf.Output, ) -> ( output_indices:tf.Output,  output_shape:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseReshape",
        Input: [  input_indices,  input_shape,  new_shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_indices  op.Output(*) output_shape )

}

/*
Computes the mean along sparse segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
segments.
Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
dimension, selecting a subset of dimension 0, specified by `indices`.

_inputArg
 data:: .T  indices:: .Tidx  segment_ids:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseSegmentMean(scope:Scope,  data:tf.Output,  indices:tf.Output,  segment_ids:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseSegmentMean",
        Input: [  data,  indices,  segment_ids, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes gradients for SparseSegmentMean.

Returns tensor "output" with same shape as grad, except for dimension 0 whose
value is output_dim0.

_inputArg
 grad:: .T  indices:: .Tidx  segment_ids:: .  output_dim0:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseSegmentMeanGrad(scope:Scope,  grad:tf.Output,  indices:tf.Output,  segment_ids:tf.Output,  output_dim0:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseSegmentMeanGrad",
        Input: [  grad,  indices,  segment_ids,  output_dim0, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the sum along sparse segments of a tensor divided by the sqrt of N.

N is the size of the segment being reduced.
Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
segments.

_inputArg
 data:: .T  indices:: .Tidx  segment_ids:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseSegmentSqrtN(scope:Scope,  data:tf.Output,  indices:tf.Output,  segment_ids:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseSegmentSqrtN",
        Input: [  data,  indices,  segment_ids, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes gradients for SparseSegmentSqrtN.

Returns tensor "output" with same shape as grad, except for dimension 0 whose
value is output_dim0.

_inputArg
 grad:: .T  indices:: .Tidx  segment_ids:: .  output_dim0:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseSegmentSqrtNGrad(scope:Scope,  grad:tf.Output,  indices:tf.Output,  segment_ids:tf.Output,  output_dim0:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseSegmentSqrtNGrad",
        Input: [  grad,  indices,  segment_ids,  output_dim0, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the sum along sparse segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
segments.
Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
dimension, selecting a subset of dimension 0, specified by `indices`.
For example:
```python
c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
# Select two rows, one segment.
tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
# => [[0 0 0 0]]
# Select two rows, two segment.
tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
# => [[ 1  2  3  4]
#     [-1 -2 -3 -4]]
# Select all rows, two segments.
tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
# => [[0 0 0 0]
#     [5 6 7 8]]
# Which is equivalent to:
tf.segment_sum(c, tf.constant([0, 0, 1]))
```

_inputArg
 data:: .T  indices:: .Tidx  segment_ids:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseSegmentSum(scope:Scope,  data:tf.Output,  indices:tf.Output,  segment_ids:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseSegmentSum",
        Input: [  data,  indices,  segment_ids, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Applies softmax to a batched N-D `SparseTensor`.

The inputs represent an N-D SparseTensor  with logical shape `[..., B, C]`
(where `N >= 2`), and with indices sorted in the canonical lexicographic order.
This op is equivalent to applying the normal `tf.nn.softmax()` to each innermost
logical submatrix with shape `[B, C]`, but with the catch that *the implicitly
zero elements do not participate*.  Specifically, the algorithm is equivalent
to the following:
  (1) Applies `tf.nn.softmax()` to a densified view of each innermost submatrix
      with shape `[B, C]`, along the size-C dimension;
  (2) Masks out the original implicitly-zero locations;
  (3) Renormalizes the remaining elements.
Hence, the `SparseTensor` result has exactly the same non-zero indices and
shape.

_inputArg
 sp_indices:: .  sp_values:: .T  sp_shape:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseSoftmax(scope:Scope,  sp_indices:tf.Output,  sp_values:tf.Output,  sp_shape:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseSoftmax",
        Input: [  sp_indices,  sp_values,  sp_shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes softmax cross entropy cost and gradients to backpropagate.

Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
a matrix of label probabilities, but rather a single label per row
of features.  This label is considered to have probability 1.0 for the
given row.
Inputs are the logits, not probabilities.

_inputArg
 features:: .T  labels:: .Tlabels 
_outputArg

_outputArg
 loss::T  backprop::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseSoftmaxCrossEntropyWithLogits(scope:Scope,  features:tf.Output,  labels:tf.Output, ) -> ( loss:tf.Output,  backprop:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseSoftmaxCrossEntropyWithLogits",
        Input: [  features,  labels, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) loss  op.Output(*) backprop )

}

/*
Returns the element-wise max of two SparseTensors.

Assumes the two SparseTensors have the same shape, i.e., no broadcasting.

_inputArg
 a_indices:: .  a_values:: .T  a_shape:: .  b_indices:: .  b_values:: .T  b_shape:: . 
_outputArg

_outputArg
 output_indices::  output_values::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseSparseMaximum(scope:Scope,  a_indices:tf.Output,  a_values:tf.Output,  a_shape:tf.Output,  b_indices:tf.Output,  b_values:tf.Output,  b_shape:tf.Output, ) -> ( output_indices:tf.Output,  output_values:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseSparseMaximum",
        Input: [  a_indices,  a_values,  a_shape,  b_indices,  b_values,  b_shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_indices  op.Output(*) output_values )

}

/*
Returns the element-wise min of two SparseTensors.

Assumes the two SparseTensors have the same shape, i.e., no broadcasting.

_inputArg
 a_indices:: .  a_values:: .T  a_shape:: .  b_indices:: .  b_values:: .T  b_shape:: . 
_outputArg

_outputArg
 output_indices::  output_values::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseSparseMinimum(scope:Scope,  a_indices:tf.Output,  a_values:tf.Output,  a_shape:tf.Output,  b_indices:tf.Output,  b_values:tf.Output,  b_shape:tf.Output, ) -> ( output_indices:tf.Output,  output_values:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseSparseMinimum",
        Input: [  a_indices,  a_values,  a_shape,  b_indices,  b_values,  b_shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_indices  op.Output(*) output_values )

}

/*
Split a `SparseTensor` into `num_split` tensors along one dimension.

If the `shape[split_dim]` is not an integer multiple of `num_split`. Slices
`[0 : shape[split_dim] % num_split]` gets one extra dimension.
For example, if `split_dim = 1` and `num_split = 2` and the input is
    input_tensor = shape = [2, 7]
    [    a   d e  ]
    [b c          ]
Graphically the output tensors are:
    output_tensor[0] = shape = [2, 4]
    [    a  ]
    [b c    ]
    output_tensor[1] = shape = [2, 3]
    [ d e  ]
    [      ]

_inputArg
 split_dim:: .  indices:: .  values:: .T  shape:: . 
_outputArg

_outputArg
 output_indices::  output_values::T  output_shape:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseSplit(scope:Scope,  split_dim:tf.Output,  indices:tf.Output,  values:tf.Output,  shape:tf.Output, ) -> ( output_indices:tf.Output,  output_values:tf.Output,  output_shape:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseSplit",
        Input: [  split_dim,  indices,  values,  shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_indices  op.Output(*) output_values  op.Output(*) output_shape )

}

/*
Adds up a `SparseTensor` and a dense `Tensor`, producing a dense `Tensor`.

This Op does not require `a_indices` be sorted in standard lexicographic order.

_inputArg
 a_indices:: .Tindices  a_values:: .T  a_shape:: .Tindices  b:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseTensorDenseAdd(scope:Scope,  a_indices:tf.Output,  a_values:tf.Output,  a_shape:tf.Output,  b:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseTensorDenseAdd",
        Input: [  a_indices,  a_values,  a_shape,  b, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Multiply SparseTensor (of rank 2) "A" by dense matrix "B".

No validity checking is performed on the indices of A.  However, the following
input format is recommended for optimal behavior:
if adjoint_a == false:
  A should be sorted in lexicographically increasing order.  Use SparseReorder
  if you're not sure.
if adjoint_a == true:
  A should be sorted in order of increasing dimension 1 (i.e., "column major"
  order instead of "row major" order).

_inputArg
 a_indices:: .Tindices  a_values:: .T  a_shape:: .  b:: .T 
_outputArg

_outputArg
 product::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseTensorDenseMatMul(scope:Scope,  a_indices:tf.Output,  a_values:tf.Output,  a_shape:tf.Output,  b:tf.Output, ) -> ( product:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseTensorDenseMatMul",
        Input: [  a_indices,  a_values,  a_shape,  b, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) product )

}

/*
Creates a dataset that splits a SparseTensor into elements row-wise.


_inputArg
 indices:: .  values:: .Tvalues  dense_shape:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseTensorSliceDataset(scope:Scope,  indices:tf.Output,  values:tf.Output,  dense_shape:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseTensorSliceDataset",
        Input: [  indices,  values,  dense_shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Converts a sparse representation into a dense tensor.

Builds an array `dense` with shape `output_shape` such that
```
# If sparse_indices is scalar
dense[i] = (i == sparse_indices ? sparse_values : default_value)
# If sparse_indices is a vector, then for each i
dense[sparse_indices[i]] = sparse_values[i]
# If sparse_indices is an n by d matrix, then for each i in [0, n)
dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
```
All other values in `dense` are set to `default_value`.  If `sparse_values` is a
scalar, all sparse indices are set to this single value.
Indices should be sorted in lexicographic order, and indices must not
contain any repeats. If `validate_indices` is true, these properties
are checked during execution.

_inputArg
 sparse_indices:: .Tindices  output_shape:: .Tindices  sparse_values:: .T  default_value:: .T 
_outputArg

_outputArg
 dense::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseToDense(scope:Scope,  sparse_indices:tf.Output,  output_shape:tf.Output,  sparse_values:tf.Output,  default_value:tf.Output, ) -> ( dense:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseToDense",
        Input: [  sparse_indices,  output_shape,  sparse_values,  default_value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) dense )

}

/*
Applies set operation along last dimension of 2 `SparseTensor` inputs.

See SetOperationOp::SetOperationFromContext for values of `set_operation`.
If `validate_indices` is `True`, `SparseToSparseSetOperation` validates the
order and range of `set1` and `set2` indices.
Input `set1` is a `SparseTensor` represented by `set1_indices`, `set1_values`,
and `set1_shape`. For `set1` ranked `n`, 1st `n-1` dimensions must be the same
as `set2`. Dimension `n` contains values in a set, duplicates are allowed but
ignored.
Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
ignored.
If `validate_indices` is `True`, this op validates the order and range of `set1`
and `set2` indices.
Output `result` is a `SparseTensor` represented by `result_indices`,
`result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
dimension contains the result of `set_operation` applied to the corresponding
`[0...n-1]` dimension of `set`.

_inputArg
 set1_indices:: .  set1_values:: .T  set1_shape:: .  set2_indices:: .  set2_values:: .T  set2_shape:: . 
_outputArg

_outputArg
 result_indices::  result_values::T  result_shape:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SparseToSparseSetOperation(scope:Scope,  set1_indices:tf.Output,  set1_values:tf.Output,  set1_shape:tf.Output,  set2_indices:tf.Output,  set2_values:tf.Output,  set2_shape:tf.Output, ) -> ( result_indices:tf.Output,  result_values:tf.Output,  result_shape:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SparseToSparseSetOperation",
        Input: [  set1_indices,  set1_values,  set1_shape,  set2_indices,  set2_values,  set2_shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) result_indices  op.Output(*) result_values  op.Output(*) result_shape )

}

/*
Splits a tensor into `num_split` tensors along one dimension.


_inputArg
 split_dim:: .  value:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Split(scope:Scope,  split_dim:tf.Output,  value:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Split",
        Input: [  split_dim,  value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Splits a tensor into `num_split` tensors along one dimension.


_inputArg
 value:: .T  size_splits:: .Tlen  split_dim:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SplitV(scope:Scope,  value:tf.Output,  size_splits:tf.Output,  split_dim:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SplitV",
        Input: [  value,  size_splits,  split_dim, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes square root of x element-wise.

I.e., \\(y = \sqrt{x} = x^{1/2}\\).

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Sqrt(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Sqrt",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Computes the gradient for the sqrt of `x` wrt its input.

Specifically, `grad = dy * 0.5 / y`, where `y = sqrt(x)`, and `dy`
is the corresponding input gradient.

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SqrtGrad(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SqrtGrad",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Computes square of x element-wise.

I.e., \\(y = x * x = x^2\\).

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Square(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Square",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Returns (x - y)(x - y) element-wise.

*NOTE*: `SquaredDifference` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SquaredDifference(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SquaredDifference",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Removes dimensions of size 1 from the shape of a tensor.

Given a tensor `input`, this operation returns a tensor of the same type with
all dimensions of size 1 removed. If you don't want to remove all size 1
dimensions, you can remove specific size 1 dimensions by specifying
`squeeze_dims`.
For example:
```
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t)) ==> [2, 3]
```
Or, to remove specific size 1 dimensions:
```
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
```

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Squeeze(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Squeeze",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
A stack that produces elements in first-in last-out order.


_inputArg
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Stack(scope:Scope, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Stack",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Delete the stack from its resource container.


_inputArg
 handle:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func StackClose(scope:Scope,  handle:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "StackClose",
        Input: [  handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Pop the element at the top of the stack.


_inputArg
 handle:: . 
_outputArg

_outputArg
 elem::elem_type 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func StackPop(scope:Scope,  handle:tf.Output, ) -> ( elem:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "StackPop",
        Input: [  handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) elem )

}

/*
Push an element onto the stack.


_inputArg
 handle:: .  elem:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func StackPush(scope:Scope,  handle:tf.Output,  elem:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "StackPush",
        Input: [  handle,  elem, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Stage values similar to a lightweight Enqueue.

The basic functionality of this Op is similar to a queue with many
fewer capabilities and options.  This Op is optimized for performance.

_inputArg
 values:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Stage(scope:Scope,  values:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Stage",
        Input: [  values, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Outputs deterministic pseudorandom values from a normal distribution.

The generated values will have mean 0 and standard deviation 1.
The outputs are a deterministic function of `shape` and `seed`.

_inputArg
 shape:: .T  seed:: . 
_outputArg

_outputArg
 output::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func StatelessRandomNormal(scope:Scope,  shape:tf.Output,  seed:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "StatelessRandomNormal",
        Input: [  shape,  seed, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Outputs deterministic pseudorandom random values from a uniform distribution.

The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.
The outputs are a deterministic function of `shape` and `seed`.

_inputArg
 shape:: .T  seed:: . 
_outputArg

_outputArg
 output::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func StatelessRandomUniform(scope:Scope,  shape:tf.Output,  seed:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "StatelessRandomUniform",
        Input: [  shape,  seed, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Outputs deterministic pseudorandom values from a truncated normal distribution.

The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked.
The outputs are a deterministic function of `shape` and `seed`.

_inputArg
 shape:: .T  seed:: . 
_outputArg

_outputArg
 output::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func StatelessTruncatedNormal(scope:Scope,  shape:tf.Output,  seed:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "StatelessTruncatedNormal",
        Input: [  shape,  seed, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Stops gradient computation.

When executed in a graph, this op outputs its input tensor as-is.
When building ops to compute gradients, this op prevents the contribution of
its inputs to be taken into account.  Normally, the gradient generator adds ops
to a graph to compute the derivatives of a specified 'loss' by recursively
finding out inputs that contributed to its computation.  If you insert this op
in the graph it inputs are masked from the gradient generator.  They are not
taken into account for computing gradients.
This is useful any time you want to compute a value with TensorFlow but need
to pretend that the value was a constant. Some examples include:
*  The *EM* algorithm where the *M-step* should not involve backpropagation
   through the output of the *E-step*.
*  Contrastive divergence training of Boltzmann machines where, when
   differentiating the energy function, the training must not backpropagate
   through the graph that generated the samples from the model.
*  Adversarial training, where no backprop should happen through the adversarial
   example generation process.

_inputArg
 input:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func StopGradient(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "StopGradient",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Return a strided slice from `input`.

Note, most python users will want to use the Python `Tensor.__getitem__`
or `Variable.__getitem__` rather than this op directly.
The goal of this op is to produce a new tensor with a subset of
the elements from the `n` dimensional `input` tensor. The subset is chosen using
a sequence of `m` sparse range specifications encoded into the arguments
of this function. Note, in some cases
`m` could be equal to `n`, but this need not be the case. Each
range specification entry can be one of the following:
- An ellipsis (...). Ellipses are used to imply zero or more
  dimensions of full-dimension selection and are produced using
  `ellipsis_mask`. For example, `foo[...]` is the identity slice.
- A new axis. This is used to insert a new shape=1 dimension and is
  produced using `new_axis_mask`. For example, `foo[:, ...]` where
  `foo` is shape `(3, 4)` produces a `(1, 3, 4)` tensor.
- A range `begin:end:stride`. This is used to specify how much to choose from
  a given dimension. `stride` can be any integer but 0.  `begin` is an integer
  which represents the index of the first value to select while `end` represents
  the index of the last value to select. The number of values selected in each
  dimension is `end - begin` if `stride > 0` and `begin - end` if `stride < 0`.
  `begin` and `end` can be negative where `-1` is the last element, `-2` is
  the second to last. `begin_mask` controls whether to replace the explicitly
  given `begin` with an implicit effective value of `0` if `stride > 0` and
  `-1` if `stride < 0`. `end_mask` is analogous but produces the number
  required to create the largest open interval. For example, given a shape
  `(3,)` tensor `foo[:]`, the effective `begin` and `end` are `0` and `3`. Do
  not assume this is equivalent to `foo[0:-1]` which has an effective `begin`
  and `end` of `0` and `2`. Another example is `foo[-2::-1]` which reverses the
  first dimension of a tensor while dropping the last two (in the original
  order elements). For example `foo = [1,2,3,4]; foo[-2::-1]` is `[4,3]`.
- A single index. This is used to keep only elements that have a given
  index. For example (`foo[2, :]` on a shape `(5,6)` tensor produces a
  shape `(6,)` tensor. This is encoded in `begin` and `end` and
  `shrink_axis_mask`.
Each conceptual range specification is encoded in the op's argument. This
encoding is best understand by considering a non-trivial example. In
particular,
`foo[1, 2:4, None, ..., :-3:-1, :]` will be encoded as
```
begin = [1, 2, x, x, 0, x] # x denotes don't care (usually 0)
end = [2, 4, x, x, -3, x]
strides = [1, 1, x, x, -1, 1]
begin_mask = 1<<4 | 1 << 5 = 48
end_mask = 1<<5 = 32
ellipsis_mask = 1<<3 = 8
new_axis_mask = 1<<2 4
shrink_axis_mask = 1<<0
```
In this case if `foo.shape` is (5, 5, 5, 5, 5, 5) the final shape of
the slice becomes (2, 1, 5, 5, 2, 5).
Let us walk step by step through each argument specification.
1.  The first argument in the example slice is turned into `begin = 1` and
`end = begin + 1 = 2`. To disambiguate from the original spec `2:4` we
also set the appropriate bit in `shrink_axis_mask`.
2. `2:4` is contributes 2, 4, 1 to begin, end, and stride. All masks have
zero bits contributed.
3. None is a synonym for `tf.newaxis`. This means insert a dimension of size 1
dimension in the final shape. Dummy values are contributed to begin,
end and stride, while the new_axis_mask bit is set.
4. `...` grab the full ranges from as many dimensions as needed to
fully specify a slice for every dimension of the input shape.
5. `:-3:-1` shows the use of negative indices. A negative index `i` associated
with a dimension that has shape `s` is converted to a positive index
`s + i`. So `-1` becomes `s-1` (i.e. the last element). This conversion
is done internally so begin, end and strides receive x, -3, and -1.
The appropriate begin_mask bit is set to indicate the start range is the
full range (ignoring the x).
6. `:` indicates that the entire contents of the corresponding dimension
is selected. This is equivalent to `::` or `0::1`. begin, end, and strides
receive 0, 0, and 1, respectively. The appropriate bits in `begin_mask` and
`end_mask` are also set.
*Requirements*:
  `0 != strides[i] for i in [0, m)`
  `ellipsis_mask must be a power of two (only one ellipsis)`

_inputArg
 input:: .T  begin:: .Index  end:: .Index  strides:: .Index 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func StridedSlice(scope:Scope,  input:tf.Output,  begin:tf.Output,  end:tf.Output,  strides:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "StridedSlice",
        Input: [  input,  begin,  end,  strides, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Assign `value` to the sliced l-value reference of `ref`.

The values of `value` are assigned to the positions in the variable
`ref` that are selected by the slice parameters. The slice parameters
`begin, `end`, `strides`, etc. work exactly as in `StridedSlice`.
NOTE this op currently does not support broadcasting and so `value`'s
shape must be exactly the shape produced by the slice of `ref`.

_inputArg
 ref:: .T  begin:: .Index  end:: .Index  strides:: .Index  value:: .T 
_outputArg

_outputArg
 output_ref::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func StridedSliceAssign(scope:Scope,  ref:tf.Output,  begin:tf.Output,  end:tf.Output,  strides:tf.Output,  value:tf.Output, ) -> ( output_ref:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "StridedSliceAssign",
        Input: [  ref,  begin,  end,  strides,  value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_ref )

}

/*
Returns the gradient of `StridedSlice`.

Since `StridedSlice` cuts out pieces of its `input` which is size
`shape`, its gradient will have the same shape (which is passed here
as `shape`). The gradient will be zero in any element that the slice
does not select.
Arguments are the same as StridedSliceGrad with the exception that
`dy` is the input gradient to be propagated and `shape` is the
shape of `StridedSlice`'s `input`.

_inputArg
 shape:: .Index  begin:: .Index  end:: .Index  strides:: .Index  dy:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func StridedSliceGrad(scope:Scope,  shape:tf.Output,  begin:tf.Output,  end:tf.Output,  strides:tf.Output,  dy:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "StridedSliceGrad",
        Input: [  shape,  begin,  end,  strides,  dy, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Joins the strings in the given list of string tensors into one tensor;

with the given separator (default is an empty separator).

_inputArg
 inputs:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func StringJoin(scope:Scope,  inputs:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "StringJoin",
        Input: [  inputs, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Split elements of `input` based on `delimiter` into a `SparseTensor`.

Let N be the size of source (typically N will be the batch size). Split each
element of `input` based on `delimiter` and return a `SparseTensor`
containing the splitted tokens. Empty tokens are ignored.
`delimiter` can be empty, or a string of split characters. If `delimiter` is an
 empty string, each element of `input` is split into individual single-byte
 character strings, including splitting of UTF-8 multibyte sequences. Otherwise
 every character of `delimiter` is a potential split point.
For example:
  N = 2, input[0] is 'hello world' and input[1] is 'a b c', then the output
  will be
  indices = [0, 0;
             0, 1;
             1, 0;
             1, 1;
             1, 2]
  shape = [2, 3]
  values = ['hello', 'world', 'a', 'b', 'c']

_inputArg
 input:: .  delimiter:: . 
_outputArg

_outputArg
 indices::  values::  shape:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func StringSplit(scope:Scope,  input:tf.Output,  delimiter:tf.Output, ) -> ( indices:tf.Output,  values:tf.Output,  shape:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "StringSplit",
        Input: [  input,  delimiter, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) indices  op.Output(*) values  op.Output(*) shape )

}

/*
Converts each string in the input Tensor to its hash mod by a number of buckets.

The hash function is deterministic on the content of the string within the
process.
Note that the hash function may change from time to time.
This functionality will be deprecated and it's recommended to use
`tf.string_to_hash_bucket_fast()` or `tf.string_to_hash_bucket_strong()`.

_inputArg
 string_tensor:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func StringToHashBucket(scope:Scope,  string_tensor:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "StringToHashBucket",
        Input: [  string_tensor, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Converts each string in the input Tensor to its hash mod by a number of buckets.

The hash function is deterministic on the content of the string within the
process and will never change. However, it is not suitable for cryptography.
This function may be used when CPU time is scarce and inputs are trusted or
unimportant. There is a risk of adversaries constructing inputs that all hash
to the same bucket. To prevent this problem, use a strong hash function with
`tf.string_to_hash_bucket_strong`.

_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func StringToHashBucketFast(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "StringToHashBucketFast",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Converts each string in the input Tensor to its hash mod by a number of buckets.

The hash function is deterministic on the content of the string within the
process. The hash function is a keyed hash function, where attribute `key`
defines the key of the hash function. `key` is an array of 2 elements.
A strong hash is important when inputs may be malicious, e.g. URLs with
additional components. Adversaries could try to make their inputs hash to the
same bucket for a denial-of-service attack or to skew the results. A strong
hash prevents this by making it difficult, if not infeasible, to compute inputs
that hash to the same bucket. This comes at a cost of roughly 4x higher compute
time than `tf.string_to_hash_bucket_fast`.

_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func StringToHashBucketStrong(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "StringToHashBucketStrong",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Converts each string in the input Tensor to the specified numeric type.

(Note that int32 overflow results in an error while float overflow
results in a rounded value.)

_inputArg
 string_tensor:: . 
_outputArg

_outputArg
 output::out_type 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func StringToNumber(scope:Scope,  string_tensor:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "StringToNumber",
        Input: [  string_tensor, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Returns x - y element-wise.

*NOTE*: `Sub` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Sub(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Sub",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Return substrings from `Tensor` of strings.

For each string in the input `Tensor`, creates a substring starting at index
`pos` with a total length of `len`.
If `len` defines a substring that would extend beyond the length of the input
string, then as many characters as possible are used.
If `pos` is negative or specifies a character index larger than any of the input
strings, then an `InvalidArgumentError` is thrown.
`pos` and `len` must have the same shape, otherwise a `ValueError` is thrown on
Op creation.
*NOTE*: `Substr` supports broadcasting up to two dimensions. More about
broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
---
Examples
Using scalar `pos` and `len`:
```python
input = [b'Hello', b'World']
position = 1
length = 3
output = [b'ell', b'orl']
```
Using `pos` and `len` with same shape as `input`:
```python
input = [[b'ten', b'eleven', b'twelve'],
         [b'thirteen', b'fourteen', b'fifteen'],
         [b'sixteen', b'seventeen', b'eighteen']]
position = [[1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]]
length =   [[2, 3, 4],
            [4, 3, 2],
            [5, 5, 5]]
output = [[b'en', b'eve', b'lve'],
          [b'hirt', b'urt', b'te'],
          [b'ixtee', b'vente', b'hteen']]
```
Broadcasting `pos` and `len` onto `input`:
```
input = [[b'ten', b'eleven', b'twelve'],
         [b'thirteen', b'fourteen', b'fifteen'],
         [b'sixteen', b'seventeen', b'eighteen'],
         [b'nineteen', b'twenty', b'twentyone']]
position = [1, 2, 3]
length =   [1, 2, 3]
output = [[b'e', b'ev', b'lve'],
          [b'h', b'ur', b'tee'],
          [b'i', b've', b'hte'],
          [b'i', b'en', b'nty']]
```
Broadcasting `input` onto `pos` and `len`:
```
input = b'thirteen'
position = [1, 5, 7]
length =   [3, 2, 1]
output = [b'hir', b'ee', b'n"]
```

_inputArg
 input:: .  pos:: .T  len:: .T 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Substr(scope:Scope,  input:tf.Output,  pos:tf.Output,  len:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Substr",
        Input: [  input,  pos,  len, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the sum of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

_inputArg
 input:: .T  reduction_indices:: .Tidx 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Sum(scope:Scope,  input:tf.Output,  reduction_indices:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Sum",
        Input: [  input,  reduction_indices, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the singular value decompositions of one or more matrices.

Computes the SVD of each inner matrix in `input` such that
`input[..., :, :] = u[..., :, :] * diag(s[..., :, :]) * transpose(v[..., :, :])`
```python
# a is a tensor containing a batch of matrices.
# s is a tensor of singular values for each matrix.
# u is the tensor containing of left singular vectors for each matrix.
# v is the tensor containing of right singular vectors for each matrix.
s, u, v = svd(a)
s, _, _ = svd(a, compute_uv=False)
```

_inputArg
 input:: .T 
_outputArg

_outputArg
 s::T  u::T  v::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Svd(scope:Scope,  input:tf.Output, ) -> ( s:tf.Output,  u:tf.Output,  v:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Svd",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) s  op.Output(*) u  op.Output(*) v )

}

/*
Forwards `data` to the output port determined by `pred`.

If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
the data goes to `output_false`.
See also `RefSwitch` and `Merge`.

_inputArg
 data:: .T  pred:: . 
_outputArg

_outputArg
 output_false::T  output_true::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Switch(scope:Scope,  data:tf.Output,  pred:tf.Output, ) -> ( output_false:tf.Output,  output_true:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Switch",
        Input: [  data,  pred, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output_false  op.Output(*) output_true )

}

/*
Computes the gradient function for function f via backpropagation.


_inputArg
 input:: . 
_outputArg

_outputArg
 output:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func SymbolicGradient(scope:Scope,  input:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "SymbolicGradient",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Creates a dataset that emits the records from one or more TFRecord files.


_inputArg
 filenames:: .  compression_type:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TFRecordDataset(scope:Scope,  filenames:tf.Output,  compression_type:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TFRecordDataset",
        Input: [  filenames,  compression_type, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
A Reader that outputs the records from a TensorFlow Records file.


_inputArg
_outputArg

_outputArg
 reader_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TFRecordReader(scope:Scope, ) -> ( reader_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TFRecordReader",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) reader_handle )

}

/*
A Reader that outputs the records from a TensorFlow Records file.


_inputArg
_outputArg

_outputArg
 reader_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TFRecordReaderV2(scope:Scope, ) -> ( reader_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TFRecordReaderV2",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) reader_handle )

}

/*
Creates a dataset that contains `count` elements from the `input_dataset`.


_inputArg
 input_dataset:: .  count:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TakeDataset(scope:Scope,  input_dataset:tf.Output,  count:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TakeDataset",
        Input: [  input_dataset,  count, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Read `SparseTensors` from a `SparseTensorsMap` and concatenate them.

The input `sparse_handles` must be an `int64` matrix of shape `[N, 1]` where
`N` is the minibatch size and the rows correspond to the output handles of
`AddSparseToTensorsMap` or `AddManySparseToTensorsMap`.  The ranks of the
original `SparseTensor` objects that went into the given input ops must all
match.  When the final `SparseTensor` is created, it has rank one
higher than the ranks of the incoming `SparseTensor` objects
(they have been concatenated along a new row dimension on the left).
The output `SparseTensor` object's shape values for all dimensions but the
first are the max across the input `SparseTensor` objects' shape values
for the corresponding dimensions.  Its first shape value is `N`, the minibatch
size.
The input `SparseTensor` objects' indices are assumed ordered in
standard lexicographic order.  If this is not the case, after this
step run `SparseReorder` to restore index ordering.
For example, if the handles represent an input, which is a `[2, 3]` matrix
representing two original `SparseTensor` objects:
```
    index = [ 0]
            [10]
            [20]
    values = [1, 2, 3]
    shape = [50]
```
and
```
    index = [ 2]
            [10]
    values = [4, 5]
    shape = [30]
```
then the final `SparseTensor` will be:
```
    index = [0  0]
            [0 10]
            [0 20]
            [1  2]
            [1 10]
    values = [1, 2, 3, 4, 5]
    shape = [2 50]
```

_inputArg
 sparse_handles:: . 
_outputArg

_outputArg
 sparse_indices::  sparse_values::dtype  sparse_shape:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TakeManySparseFromTensorsMap(scope:Scope,  sparse_handles:tf.Output, ) -> ( sparse_indices:tf.Output,  sparse_values:tf.Output,  sparse_shape:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TakeManySparseFromTensorsMap",
        Input: [  sparse_handles, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) sparse_indices  op.Output(*) sparse_values  op.Output(*) sparse_shape )

}

/*
Computes tan of x element-wise.


_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Tan(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Tan",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Computes hyperbolic tangent of `x` element-wise.


_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Tanh(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Tanh",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Computes the gradient for the tanh of `x` wrt its input.

Specifically, `grad = dy * (1 - y*y)`, where `y = tanh(x)`, and `dy`
is the corresponding input gradient.

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TanhGrad(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TanhGrad",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Returns a tensor that may be mutated, but only persists within a single step.

This is an experimental op for internal use only and it is possible to use this
op in unsafe ways.  DO NOT USE unless you fully understand the risks.
It is the caller's responsibility to ensure that 'ref' is eventually passed to a
matching 'DestroyTemporaryVariable' op after all other uses have completed.
Outputs a ref to the tensor state so it may be read or modified.
  E.g.
      var = state_ops._temporary_variable([1, 2], types.float_)
      var_name = var.op.name
      var = state_ops.assign(var, [[4.0, 5.0]])
      var = state_ops.assign_add(var, [[6.0, 7.0]])
      final = state_ops._destroy_temporary_variable(var, var_name=var_name)

_inputArg
_outputArg

_outputArg
 ref::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TemporaryVariable(scope:Scope, ) -> ( ref:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TemporaryVariable",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) ref )

}

/*


_inputArg
 size:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArray(scope:Scope,  size:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArray",
        Input: [  size, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*


_inputArg
 handle:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayClose(scope:Scope,  handle:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayClose",
        Input: [  handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Deprecated. Use TensorArrayCloseV3


_inputArg
 handle:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayCloseV2(scope:Scope,  handle:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayCloseV2",
        Input: [  handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Delete the TensorArray from its resource container.

This enables the user to close and release the resource in the middle
of a step/run.

_inputArg
 handle:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayCloseV3(scope:Scope,  handle:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayCloseV3",
        Input: [  handle, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*


_inputArg
 handle:: .  flow_in:: . 
_outputArg

_outputArg
 value::dtype  lengths:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayConcat(scope:Scope,  handle:tf.Output,  flow_in:tf.Output, ) -> ( value:tf.Output,  lengths:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayConcat",
        Input: [  handle,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) value  op.Output(*) lengths )

}

/*
Deprecated. Use TensorArrayConcatV3


_inputArg
 handle:: .  flow_in:: . 
_outputArg

_outputArg
 value::dtype  lengths:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayConcatV2(scope:Scope,  handle:tf.Output,  flow_in:tf.Output, ) -> ( value:tf.Output,  lengths:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayConcatV2",
        Input: [  handle,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) value  op.Output(*) lengths )

}

/*
Concat the elements from the TensorArray into value `value`.

Takes `T` elements of shapes
  ```
  (n0 x d0 x d1 x ...), (n1 x d0 x d1 x ...), ..., (n(T-1) x d0 x d1 x ...)
  ```
and concatenates them into a Tensor of shape:
  ```(n0 + n1 + ... + n(T-1) x d0 x d1 x ...)```
All elements must have the same shape (excepting the first dimension).

_inputArg
 handle:: .  flow_in:: . 
_outputArg

_outputArg
 value::dtype  lengths:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayConcatV3(scope:Scope,  handle:tf.Output,  flow_in:tf.Output, ) -> ( value:tf.Output,  lengths:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayConcatV3",
        Input: [  handle,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) value  op.Output(*) lengths )

}

/*


_inputArg
 handle:: .  indices:: .  flow_in:: . 
_outputArg

_outputArg
 value::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayGather(scope:Scope,  handle:tf.Output,  indices:tf.Output,  flow_in:tf.Output, ) -> ( value:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayGather",
        Input: [  handle,  indices,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) value )

}

/*
Deprecated. Use TensorArrayGatherV3


_inputArg
 handle:: .  indices:: .  flow_in:: . 
_outputArg

_outputArg
 value::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayGatherV2(scope:Scope,  handle:tf.Output,  indices:tf.Output,  flow_in:tf.Output, ) -> ( value:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayGatherV2",
        Input: [  handle,  indices,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) value )

}

/*
Gather specific elements from the TensorArray into output `value`.

All elements selected by `indices` must have the same shape.

_inputArg
 handle:: .  indices:: .  flow_in:: . 
_outputArg

_outputArg
 value::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayGatherV3(scope:Scope,  handle:tf.Output,  indices:tf.Output,  flow_in:tf.Output, ) -> ( value:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayGatherV3",
        Input: [  handle,  indices,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) value )

}

/*


_inputArg
 handle:: .  flow_in:: . 
_outputArg

_outputArg
 grad_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayGrad(scope:Scope,  handle:tf.Output,  flow_in:tf.Output, ) -> ( grad_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayGrad",
        Input: [  handle,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) grad_handle )

}

/*
Deprecated. Use TensorArrayGradV3


_inputArg
 handle:: .  flow_in:: . 
_outputArg

_outputArg
 grad_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayGradV2(scope:Scope,  handle:tf.Output,  flow_in:tf.Output, ) -> ( grad_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayGradV2",
        Input: [  handle,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) grad_handle )

}

/*
Creates a TensorArray for storing the gradients of values in the given handle.

If the given TensorArray gradient already exists, returns a reference to it.
Locks the size of the original TensorArray by disabling its dynamic size flag.
**A note about the input flow_in:**
The handle flow_in forces the execution of the gradient lookup to occur
only after certain other operations have occurred.  For example, when
the forward TensorArray is dynamically sized, writes to this TensorArray
may resize the object.  The gradient TensorArray is statically sized based
on the size of the forward TensorArray when this operation executes.
Furthermore, the size of the forward TensorArray is frozen by this call.
As a result, the flow is used to ensure that the call to generate the gradient
TensorArray only happens after all writes are executed.
In the case of dynamically sized TensorArrays, gradient computation should
only be performed on read operations that have themselves been chained via
flow to occur only after all writes have executed. That way the final size
of the forward TensorArray is known when this operation is called.
**A note about the source attribute:**
TensorArray gradient calls use an accumulator TensorArray object.  If
multiple gradients are calculated and run in the same session, the multiple
gradient nodes may accidentally flow throuth the same accumulator TensorArray.
This double counts and generally breaks the TensorArray gradient flow.
The solution is to identify which gradient call this particular
TensorArray gradient is being called in.  This is performed by identifying
a unique string (e.g. "gradients", "gradients_1", ...) from the input
gradient Tensor's name.  This string is used as a suffix when creating
the TensorArray gradient object here (the attribute `source`).
The attribute `source` is added as a suffix to the forward TensorArray's
name when performing the creation / lookup, so that each separate gradient
calculation gets its own TensorArray accumulator.

_inputArg
 handle:: .  flow_in:: . 
_outputArg

_outputArg
 grad_handle::  flow_out:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayGradV3(scope:Scope,  handle:tf.Output,  flow_in:tf.Output, ) -> ( grad_handle:tf.Output,  flow_out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayGradV3",
        Input: [  handle,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) grad_handle  op.Output(*) flow_out )

}

/*


_inputArg
 handle:: .  flow_in:: . 
_outputArg

_outputArg
 value::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayPack(scope:Scope,  handle:tf.Output,  flow_in:tf.Output, ) -> ( value:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayPack",
        Input: [  handle,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) value )

}

/*


_inputArg
 handle:: .  index:: .  flow_in:: . 
_outputArg

_outputArg
 value::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayRead(scope:Scope,  handle:tf.Output,  index:tf.Output,  flow_in:tf.Output, ) -> ( value:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayRead",
        Input: [  handle,  index,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) value )

}

/*
Deprecated. Use TensorArrayReadV3


_inputArg
 handle:: .  index:: .  flow_in:: . 
_outputArg

_outputArg
 value::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayReadV2(scope:Scope,  handle:tf.Output,  index:tf.Output,  flow_in:tf.Output, ) -> ( value:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayReadV2",
        Input: [  handle,  index,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) value )

}

/*
Read an element from the TensorArray into output `value`.


_inputArg
 handle:: .  index:: .  flow_in:: . 
_outputArg

_outputArg
 value::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayReadV3(scope:Scope,  handle:tf.Output,  index:tf.Output,  flow_in:tf.Output, ) -> ( value:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayReadV3",
        Input: [  handle,  index,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) value )

}

/*


_inputArg
 handle:: .  indices:: .  value:: .T  flow_in:: . 
_outputArg

_outputArg
 flow_out:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayScatter(scope:Scope,  handle:tf.Output,  indices:tf.Output,  value:tf.Output,  flow_in:tf.Output, ) -> ( flow_out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayScatter",
        Input: [  handle,  indices,  value,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) flow_out )

}

/*
Deprecated. Use TensorArrayScatterV3


_inputArg
 handle:: .  indices:: .  value:: .T  flow_in:: . 
_outputArg

_outputArg
 flow_out:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayScatterV2(scope:Scope,  handle:tf.Output,  indices:tf.Output,  value:tf.Output,  flow_in:tf.Output, ) -> ( flow_out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayScatterV2",
        Input: [  handle,  indices,  value,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) flow_out )

}

/*
Scatter the data from the input value into specific TensorArray elements.

`indices` must be a vector, its length must match the first dim of `value`.

_inputArg
 handle:: .  indices:: .  value:: .T  flow_in:: . 
_outputArg

_outputArg
 flow_out:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayScatterV3(scope:Scope,  handle:tf.Output,  indices:tf.Output,  value:tf.Output,  flow_in:tf.Output, ) -> ( flow_out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayScatterV3",
        Input: [  handle,  indices,  value,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) flow_out )

}

/*


_inputArg
 handle:: .  flow_in:: . 
_outputArg

_outputArg
 size:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArraySize(scope:Scope,  handle:tf.Output,  flow_in:tf.Output, ) -> ( size:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArraySize",
        Input: [  handle,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) size )

}

/*
Deprecated. Use TensorArraySizeV3


_inputArg
 handle:: .  flow_in:: . 
_outputArg

_outputArg
 size:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArraySizeV2(scope:Scope,  handle:tf.Output,  flow_in:tf.Output, ) -> ( size:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArraySizeV2",
        Input: [  handle,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) size )

}

/*
Get the current size of the TensorArray.


_inputArg
 handle:: .  flow_in:: . 
_outputArg

_outputArg
 size:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArraySizeV3(scope:Scope,  handle:tf.Output,  flow_in:tf.Output, ) -> ( size:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArraySizeV3",
        Input: [  handle,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) size )

}

/*


_inputArg
 handle:: .  value:: .T  lengths:: .  flow_in:: . 
_outputArg

_outputArg
 flow_out:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArraySplit(scope:Scope,  handle:tf.Output,  value:tf.Output,  lengths:tf.Output,  flow_in:tf.Output, ) -> ( flow_out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArraySplit",
        Input: [  handle,  value,  lengths,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) flow_out )

}

/*
Deprecated. Use TensorArraySplitV3


_inputArg
 handle:: .  value:: .T  lengths:: .  flow_in:: . 
_outputArg

_outputArg
 flow_out:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArraySplitV2(scope:Scope,  handle:tf.Output,  value:tf.Output,  lengths:tf.Output,  flow_in:tf.Output, ) -> ( flow_out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArraySplitV2",
        Input: [  handle,  value,  lengths,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) flow_out )

}

/*
Split the data from the input value into TensorArray elements.

Assuming that `lengths` takes on values
  ```(n0, n1, ..., n(T-1))```
and that `value` has shape
  ```(n0 + n1 + ... + n(T-1) x d0 x d1 x ...)```,
this splits values into a TensorArray with T tensors.
TensorArray index t will be the subtensor of values with starting position
  ```(n0 + n1 + ... + n(t-1), 0, 0, ...)```
and having size
  ```nt x d0 x d1 x ...```

_inputArg
 handle:: .  value:: .T  lengths:: .  flow_in:: . 
_outputArg

_outputArg
 flow_out:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArraySplitV3(scope:Scope,  handle:tf.Output,  value:tf.Output,  lengths:tf.Output,  flow_in:tf.Output, ) -> ( flow_out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArraySplitV3",
        Input: [  handle,  value,  lengths,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) flow_out )

}

/*


_inputArg
 handle:: .  value:: .T  flow_in:: . 
_outputArg

_outputArg
 flow_out:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayUnpack(scope:Scope,  handle:tf.Output,  value:tf.Output,  flow_in:tf.Output, ) -> ( flow_out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayUnpack",
        Input: [  handle,  value,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) flow_out )

}

/*
Deprecated. Use TensorArrayV3


_inputArg
 size:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayV2(scope:Scope,  size:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayV2",
        Input: [  size, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
An array of Tensors of given size.

Write data via Write and read via Read or Pack.

_inputArg
 size:: . 
_outputArg

_outputArg
 handle::  flow:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayV3(scope:Scope,  size:tf.Output, ) -> ( handle:tf.Output,  flow:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayV3",
        Input: [  size, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle  op.Output(*) flow )

}

/*


_inputArg
 handle:: .  index:: .  value:: .T  flow_in:: . 
_outputArg

_outputArg
 flow_out:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayWrite(scope:Scope,  handle:tf.Output,  index:tf.Output,  value:tf.Output,  flow_in:tf.Output, ) -> ( flow_out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayWrite",
        Input: [  handle,  index,  value,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) flow_out )

}

/*
Deprecated. Use TensorArrayGradV3


_inputArg
 handle:: .  index:: .  value:: .T  flow_in:: . 
_outputArg

_outputArg
 flow_out:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayWriteV2(scope:Scope,  handle:tf.Output,  index:tf.Output,  value:tf.Output,  flow_in:tf.Output, ) -> ( flow_out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayWriteV2",
        Input: [  handle,  index,  value,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) flow_out )

}

/*
Push an element onto the tensor_array.


_inputArg
 handle:: .  index:: .  value:: .T  flow_in:: . 
_outputArg

_outputArg
 flow_out:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorArrayWriteV3(scope:Scope,  handle:tf.Output,  index:tf.Output,  value:tf.Output,  flow_in:tf.Output, ) -> ( flow_out:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorArrayWriteV3",
        Input: [  handle,  index,  value,  flow_in, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) flow_out )

}

/*
Creates a dataset that emits `components` as a tuple of tensors once.


_inputArg
 components:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorDataset(scope:Scope,  components:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorDataset",
        Input: [  components, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Creates a dataset that emits each dim-0 slice of `components` once.


_inputArg
 components:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorSliceDataset(scope:Scope,  components:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorSliceDataset",
        Input: [  components, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
Outputs a `Summary` protocol buffer with a tensor.


_inputArg
 tensor:: .T 
_outputArg

_outputArg
 summary:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TensorSummary(scope:Scope,  tensor:tf.Output, ) -> ( summary:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TensorSummary",
        Input: [  tensor, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) summary )

}

/*
Creates a dataset that emits the lines of one or more text files.


_inputArg
 filenames:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TextLineDataset(scope:Scope,  filenames:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TextLineDataset",
        Input: [  filenames, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

/*
A Reader that outputs the lines of a file delimited by '\n'.


_inputArg
_outputArg

_outputArg
 reader_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TextLineReader(scope:Scope, ) -> ( reader_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TextLineReader",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) reader_handle )

}

/*
A Reader that outputs the lines of a file delimited by '\n'.


_inputArg
_outputArg

_outputArg
 reader_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TextLineReaderV2(scope:Scope, ) -> ( reader_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TextLineReaderV2",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) reader_handle )

}

/*
Generates labels for candidate sampling with a learned unigram distribution.

See explanations of candidate sampling and the data formats at
go/candidate-sampling.
For each batch, this op picks a single set of sampled candidate labels.
The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

_inputArg
 true_classes:: . 
_outputArg

_outputArg
 sampled_candidates::  true_expected_count::  sampled_expected_count:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ThreadUnsafeUnigramCandidateSampler(scope:Scope,  true_classes:tf.Output, ) -> ( sampled_candidates:tf.Output,  true_expected_count:tf.Output,  sampled_expected_count:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ThreadUnsafeUnigramCandidateSampler",
        Input: [  true_classes, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) sampled_candidates  op.Output(*) true_expected_count  op.Output(*) sampled_expected_count )

}

/*
Constructs a tensor by tiling a given tensor.

This operation creates a new tensor by replicating `input` `multiples` times.
The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
and the values of `input` are replicated `multiples[i]` times along the 'i'th
dimension. For example, tiling `[a b c d]` by `[2]` produces
`[a b c d a b c d]`.

_inputArg
 input:: .T  multiples:: .Tmultiples 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Tile(scope:Scope,  input:tf.Output,  multiples:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Tile",
        Input: [  input,  multiples, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Returns the gradient of `Tile`.

Since `Tile` takes an input and repeats the input `multiples` times
along each dimension, `TileGrad` takes in `multiples` and aggregates
each repeated tile of `input` into `output`.

_inputArg
 input:: .T  multiples:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TileGrad(scope:Scope,  input:tf.Output,  multiples:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TileGrad",
        Input: [  input,  multiples, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Finds values and indices of the `k` largest elements for the last dimension.

If the input is a vector (rank-1), finds the `k` largest entries in the vector
and outputs their values and indices as vectors.  Thus `values[j]` is the
`j`-th largest entry in `input`, and its index is `indices[j]`.
For matrices (resp. higher rank input), computes the top `k` entries in each
row (resp. vector along the last dimension).  Thus,
    values.shape = indices.shape = input.shape[:-1] + [k]
If two elements are equal, the lower-index element appears first.
If `k` varies dynamically, use `TopKV2` below.

_inputArg
 input:: .T 
_outputArg

_outputArg
 values::T  indices:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TopK(scope:Scope,  input:tf.Output, ) -> ( values:tf.Output,  indices:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TopK",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) values  op.Output(*) indices )

}

/*
Finds values and indices of the `k` largest elements for the last dimension.

If the input is a vector (rank-1), finds the `k` largest entries in the vector
and outputs their values and indices as vectors.  Thus `values[j]` is the
`j`-th largest entry in `input`, and its index is `indices[j]`.
For matrices (resp. higher rank input), computes the top `k` entries in each
row (resp. vector along the last dimension).  Thus,
    values.shape = indices.shape = input.shape[:-1] + [k]
If two elements are equal, the lower-index element appears first.

_inputArg
 input:: .T  k:: . 
_outputArg

_outputArg
 values::T  indices:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TopKV2(scope:Scope,  input:tf.Output,  k:tf.Output, ) -> ( values:tf.Output,  indices:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TopKV2",
        Input: [  input,  k, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) values  op.Output(*) indices )

}

/*
Shuffle dimensions of x according to a permutation.

The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
  `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`

_inputArg
 x:: .T  perm:: .Tperm 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Transpose(scope:Scope,  x:tf.Output,  perm:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Transpose",
        Input: [  x,  perm, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Returns x / y element-wise for integer types.

Truncation designates that negative numbers will round fractional quantities
toward zero. I.e. -7 / 5 = 1. This matches C semantics but it is different
than Python semantics. See `FloorDiv` for a division function that matches
Python Semantics.
*NOTE*: `TruncateDiv` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TruncateDiv(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TruncateDiv",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Returns element-wise remainder of division. This emulates C semantics in that

the result here is consistent with a truncating divide. E.g. `truncate(x / y) *
y + truncate_mod(x, y) = x`.
*NOTE*: `TruncateMod` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

_inputArg
 x:: .T  y:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TruncateMod(scope:Scope,  x:tf.Output,  y:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TruncateMod",
        Input: [  x,  y, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Outputs random values from a truncated normal distribution.

The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked.

_inputArg
 shape:: .T 
_outputArg

_outputArg
 output::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func TruncatedNormal(scope:Scope,  shape:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "TruncatedNormal",
        Input: [  shape, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Generates labels for candidate sampling with a uniform distribution.

See explanations of candidate sampling and the data formats at
go/candidate-sampling.
For each batch, this op picks a single set of sampled candidate labels.
The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

_inputArg
 true_classes:: . 
_outputArg

_outputArg
 sampled_candidates::  true_expected_count::  sampled_expected_count:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func UniformCandidateSampler(scope:Scope,  true_classes:tf.Output, ) -> ( sampled_candidates:tf.Output,  true_expected_count:tf.Output,  sampled_expected_count:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "UniformCandidateSampler",
        Input: [  true_classes, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) sampled_candidates  op.Output(*) true_expected_count  op.Output(*) sampled_expected_count )

}

/*
Finds unique elements in a 1-D tensor.

This operation returns a tensor `y` containing all of the unique elements of `x`
sorted in the same order that they occur in `x`. This operation also returns a
tensor `idx` the same size as `x` that contains the index of each value of `x`
in the unique output `y`. In other words:
`y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
For example:
```
# tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
y, idx = unique(x)
y ==> [1, 2, 4, 7, 8]
idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
```

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T  idx::out_idx 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Unique(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output,  idx:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Unique",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y  op.Output(*) idx )

}

/*
Finds unique elements in a 1-D tensor.

This operation returns a tensor `y` containing all of the unique elements of `x`
sorted in the same order that they occur in `x`. This operation also returns a
tensor `idx` the same size as `x` that contains the index of each value of `x`
in the unique output `y`. Finally, it returns a third tensor `count` that
contains the count of each element of `y` in `x`. In other words:
`y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
For example:
```
# tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
y, idx, count = unique_with_counts(x)
y ==> [1, 2, 4, 7, 8]
idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
count ==> [2, 1, 3, 1, 2]
```

_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T  idx::out_idx  count::out_idx 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func UniqueWithCounts(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output,  idx:tf.Output,  count:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "UniqueWithCounts",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y  op.Output(*) idx  op.Output(*) count )

}

/*
Unpacks a given dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors.

Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
For example, given a tensor of shape `(A, B, C, D)`;
If `axis == 0` then the i'th tensor in `output` is the slice `value[i, :, :, :]`
  and each tensor in `output` will have shape `(B, C, D)`. (Note that the
  dimension unpacked along is gone, unlike `split`).
If `axis == 1` then the i'th tensor in `output` is the slice `value[:, i, :, :]`
  and each tensor in `output` will have shape `(A, C, D)`.
Etc.
This is the opposite of `pack`.

_inputArg
 value:: .T 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Unpack(scope:Scope,  value:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Unpack",
        Input: [  value, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the Max along segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
segments.
This operator is similar to the [unsorted segment sum operator](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
Instead of computing the sum over segments, it computes the maximum
such that:
\\(output_i = \max_j data_j\\) where max is over `j` such
that `segment_ids[j] == i`.
If the maximum is empty for a given segment ID `i`, it outputs the smallest possible value for specific numeric type,
 `output[i] = numeric_limits<T>::min()`.
<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentSum.png" alt>
</div>

_inputArg
 data:: .T  segment_ids:: .Tindices  num_segments:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func UnsortedSegmentMax(scope:Scope,  data:tf.Output,  segment_ids:tf.Output,  num_segments:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "UnsortedSegmentMax",
        Input: [  data,  segment_ids,  num_segments, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Computes the sum along segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
segments.
Computes a tensor such that
`(output[i] = sum_{j...} data[j...]` where the sum is over tuples `j...` such
that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
need not be sorted and need not cover all values in the full
range of valid values.
If the sum is empty for a given segment ID `i`, `output[i] = 0`.
`num_segments` should equal the number of distinct segment IDs.
<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentSum.png" alt>
</div>

_inputArg
 data:: .T  segment_ids:: .Tindices  num_segments:: . 
_outputArg

_outputArg
 output::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func UnsortedSegmentSum(scope:Scope,  data:tf.Output,  segment_ids:tf.Output,  num_segments:tf.Output, ) -> ( output:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "UnsortedSegmentSum",
        Input: [  data,  segment_ids,  num_segments, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) output )

}

/*
Op is similar to a lightweight Dequeue.

The basic funtionality is similar to dequeue with many fewer
capabilities and options.  This Op is optimized for performance.

_inputArg
_outputArg

_outputArg
 values:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Unstage(scope:Scope, ) -> ( values:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Unstage",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) values )

}

/*
Use VariableV2 instead.


_inputArg
_outputArg

_outputArg
 ref::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Variable(scope:Scope, ) -> ( ref:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Variable",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) ref )

}

/*
Holds state in the form of a tensor that persists across steps.

Outputs a ref to the tensor state so it may be read or modified.
TODO(zhifengc/mrry): Adds a pointer to a more detail document
about sharing states in tensorflow.

_inputArg
_outputArg

_outputArg
 ref::dtype 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func VariableV2(scope:Scope, ) -> ( ref:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "VariableV2",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) ref )

}

/*
Returns locations of true values in a boolean tensor.

This operation returns the coordinates of true elements in `input`. The
coordinates are returned in a 2-D tensor where the first dimension (rows)
represents the number of true elements, and the second dimension (columns)
represents the coordinates of the true elements. Keep in mind, the shape of
the output tensor can vary depending on how many true values there are in
`input`. Indices are output in row-major order.
For example:
```
# 'input' tensor is [[True, False]
#                    [True, False]]
# 'input' has two true values, so output has two coordinates.
# 'input' has rank of 2, so coordinates have two indices.
where(input) ==> [[0, 0],
                  [1, 0]]
# `input` tensor is [[[True, False]
#                     [True, False]]
#                    [[False, True]
#                     [False, True]]
#                    [[False, False]
#                     [False, True]]]
# 'input' has 5 true values, so output has 5 coordinates.
# 'input' has rank of 3, so coordinates have three indices.
where(input) ==> [[0, 0, 0],
                  [0, 1, 0],
                  [1, 0, 1],
                  [1, 1, 1],
                  [2, 1, 1]]
```

_inputArg
 input:: . 
_outputArg

_outputArg
 index:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Where(scope:Scope,  input:tf.Output, ) -> ( index:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Where",
        Input: [  input, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) index )

}

/*
A Reader that outputs the entire contents of a file as a value.

To use, enqueue filenames in a Queue.  The output of ReaderRead will
be a filename (key) and the contents of that file (value).

_inputArg
_outputArg

_outputArg
 reader_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func WholeFileReader(scope:Scope, ) -> ( reader_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "WholeFileReader",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) reader_handle )

}

/*
A Reader that outputs the entire contents of a file as a value.

To use, enqueue filenames in a Queue.  The output of ReaderRead will
be a filename (key) and the contents of that file (value).

_inputArg
_outputArg

_outputArg
 reader_handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func WholeFileReaderV2(scope:Scope, ) -> ( reader_handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "WholeFileReaderV2",
        Input: [ )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) reader_handle )

}

/*
Writes contents to the file at input filename. Creates file if not existing.


_inputArg
 filename:: .  contents:: . 
_outputArg

_outputArg

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func WriteFile(scope:Scope,  filename:tf.Output,  contents:tf.Output, ) -> () {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "WriteFile",
        Input: [  filename,  contents, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return ( )

}

/*
Returns a tensor of zeros with the same shape and type as x.


_inputArg
 x:: .T 
_outputArg

_outputArg
 y::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ZerosLike(scope:Scope,  x:tf.Output, ) -> ( y:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ZerosLike",
        Input: [  x, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) y )

}

/*
Compute the Hurwitz zeta function \\(\zeta(x, q)\\).

The Hurwitz zeta function is defined as:
\\(\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}\\)

_inputArg
 x:: .T  q:: .T 
_outputArg

_outputArg
 z::T 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func Zeta(scope:Scope,  x:tf.Output,  q:tf.Output, ) -> ( z:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "Zeta",
        Input: [  x,  q, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) z )

}

/*
Creates a dataset that zips together `input_datasets`.


_inputArg
 input_datasets:: . 
_outputArg

_outputArg
 handle:: 

*/


// TODO - detect if the output args = 0
// TODO - in the return parameters use an index Output(0) // output(1) etc
// TODO - work out a way to skip extroneous , eg. param,param, )
// TODO - Not all return types are output eg.
// func ResourceSparseApplyRMSProp(scope *Scope, var_ tf.Output, ms tf.Output, mom tf.Output, lr tf.Output, rho tf.Output, momentum tf.Output, epsilon tf.Output, grad tf.Output, indices tf.Output, optional ...ResourceSparseApplyRMSPropAttr) (o *tf.Operation)


func ZipDataset(scope:Scope,  input_datasets:tf.Output, ) -> ( handle:tf.Output, ) {
    if scope.error() != nil {
        return
    }


    let opspec = tf.OpSpec(
        Type: "ZipDataset",
        Input: [  input_datasets, )]
        Attrs: attrs
    )


    let op = scope.AddOperation(opspec)
    return (  op.Output(*) handle )

}

