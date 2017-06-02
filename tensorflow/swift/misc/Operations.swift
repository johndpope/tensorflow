// This file is generated automatically, DO NOT EDIT
//  tensorflow/core/framework/op_def.proto
//  tensorflow/core/framework/op_def.pb.swift
//  https://github.com/tensorflow/tensorflow/blob/c9428db01eec1c457d9451ef1ca550b634bddf60/tensorflow/go/op/wrappers.go
// https://github.com/SwiftGen/templates/blob/master/templates/strings/flat-swift3.stencil

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

*/




func abort(scope:Scope errorMsg :string  , exitWithoutError :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("error_msg":errorMsg)
    attr.append("exit_without_error":exitWithoutError)

    let opspec = tf.OpSpec(
        OpType: "Abort",
        Input: [ ],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Computes the absolute value of a tensor.

Given a tensor `x`, this operation returns a tensor containing the absolute
value of each element in `x`. For example, if x is an input element and y is
an output element, this operation computes \\(y = |x|\\).

*/




func abs(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Abs",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Applies a gradient to a given accumulator.

Does not add if local_step is lesser than the accumulator's global_step.

*/




func accumulatorApplyGradient(scope:Scope,handle: Tensorflow_DataType, localStep: Tensorflow_DataType, gradient: Tensorflow_DataType, dtype :type  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)

    let opspec = tf.OpSpec(
        OpType: "AccumulatorApplyGradient",
        Input: [ handle, localStep, gradient],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Returns the number of gradients aggregated in the given accumulators.


*/




func accumulatorNumAccumulated(scope:Scope,handle: Tensorflow_DataType )  ->(numAccumulated: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "AccumulatorNumAccumulated",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Updates the accumulator with a new value for global_step.

Logs warning if the accumulator's value is already higher than
new_global_step.

*/




func accumulatorSetGlobalStep(scope:Scope,handle: Tensorflow_DataType, newGlobalStep: Tensorflow_DataType )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "AccumulatorSetGlobalStep",
        Input: [ handle, newGlobalStep],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Extracts the average gradient in the given ConditionalAccumulator.

The op blocks until sufficient (i.e., more than num_required)
gradients have been accumulated.  If the accumulator has already
aggregated more than num_required gradients, it returns the average of
the accumulated gradients.  Also automatically increments the recorded
global_step in the accumulator by 1, and resets the aggregate to 0.

*/




func accumulatorTakeGradient(scope:Scope,handle: Tensorflow_DataType, numRequired: Tensorflow_DataType, dtype :type  )  ->(average: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)

    let opspec = tf.OpSpec(
        OpType: "AccumulatorTakeGradient",
        Input: [ handle, numRequired],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes acos of x element-wise.


*/




func acos(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Acos",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns x + y element-wise.

*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func add(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Add",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func addManySparseToTensorsMap(scope:Scope,sparseIndices: Tensorflow_DataType, sparseValues: Tensorflow_DataType, sparseShape: Tensorflow_DataType, t :type  , container :string  , sharedName :string  )  ->(sparseHandles: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "AddManySparseToTensorsMap",
        Input: [ sparseIndices, sparseValues, sparseShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Add all input tensors element wise.


*/




func addN(scope:Scope,inputs: Tensorflow_DataType, n :uInt8  , t :type  )  ->(sum: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("N":n)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "AddN",
        Input: [ inputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func addSparseToTensorsMap(scope:Scope,sparseIndices: Tensorflow_DataType, sparseValues: Tensorflow_DataType, sparseShape: Tensorflow_DataType, t :type  , container :string  , sharedName :string  )  ->(sparseHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "AddSparseToTensorsMap",
        Input: [ sparseIndices, sparseValues, sparseShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Deprecated. Disallowed in GraphDef version >= 2.


*/




func adjustContrast(scope:Scope,images: Tensorflow_DataType, contrastFactor: Tensorflow_DataType, minValue: Tensorflow_DataType, maxValue: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "AdjustContrast",
        Input: [ images, contrastFactor, minValue, maxValue],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func adjustContrastv2(scope:Scope,images: Tensorflow_DataType, contrastFactor: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "AdjustContrastv2",
        Input: [ images, contrastFactor],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Adjust the hue of one or more images.

`images` is a tensor of at least 3 dimensions.  The last dimension is
interpretted as channels, and must be three.
The input image is considered in the RGB colorspace. Conceptually, the RGB
colors are first mapped into HSV. A delta is then applied all the hue values,
and then remapped back to RGB colorspace.

*/




func adjustHue(scope:Scope,images: Tensorflow_DataType, delta: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "AdjustHue",
        Input: [ images, delta],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Adjust the saturation of one or more images.

`images` is a tensor of at least 3 dimensions.  The last dimension is
interpretted as channels, and must be three.
The input image is considered in the RGB colorspace. Conceptually, the RGB
colors are first mapped into HSV. A scale is then applied all the saturation
values, and then remapped back to RGB colorspace.

*/




func adjustSaturation(scope:Scope,images: Tensorflow_DataType, scale: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "AdjustSaturation",
        Input: [ images, scale],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the "logical and" of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

*/




func all(scope:Scope,input: Tensorflow_DataType, reductionIndices: Tensorflow_DataType, keepDims :bool  , tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("keep_dims":keepDims)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "All",
        Input: [ input, reductionIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func allCandidateSampler(scope:Scope,trueClasses: Tensorflow_DataType, numTrue :uInt8  , numSampled :uInt8  , unique :bool  , seed :uInt8  , seed2 :uInt8  )  ->(sampledCandidates: tf.Output, trueExpectedCount: tf.Output, sampledExpectedCount: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_true":numTrue)
    attr.append("num_sampled":numSampled)
    attr.append("unique":unique)
    attr.append("seed":seed)
    attr.append("seed2":seed2)

    let opspec = tf.OpSpec(
        OpType: "AllCandidateSampler",
        Input: [ trueClasses],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Computes the "logical or" of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

*/




func any(scope:Scope,input: Tensorflow_DataType, reductionIndices: Tensorflow_DataType, keepDims :bool  , tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("keep_dims":keepDims)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "Any",
        Input: [ input, reductionIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Update '*var' according to the adadelta scheme.

accum = rho() * accum + (1 - rho()) * grad.square();
update = (update_accum + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad;
update_accum = rho() * update_accum + (1 - rho()) * update.square();
var -= update;

*/




func applyAdadelta(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, accumUpdate: Tensorflow_DataType, lr: Tensorflow_DataType, rho: Tensorflow_DataType, epsilon: Tensorflow_DataType, grad: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ApplyAdadelta",
        Input: [ `var`, accum, accumUpdate, lr, rho, epsilon, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Update '*var' according to the adagrad scheme.

accum += grad * grad
var -= lr * grad * (1 / sqrt(accum))

*/




func applyAdagrad(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, lr: Tensorflow_DataType, grad: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ApplyAdagrad",
        Input: [ `var`, accum, lr, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Update '*var' according to the proximal adagrad scheme.


*/




func applyAdagradDA(scope:Scope,`var`: Tensorflow_DataType, gradientAccumulator: Tensorflow_DataType, gradientSquaredAccumulator: Tensorflow_DataType, grad: Tensorflow_DataType, lr: Tensorflow_DataType, l1: Tensorflow_DataType, l2: Tensorflow_DataType, globalStep: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ApplyAdagradDA",
        Input: [ `var`, gradientAccumulator, gradientSquaredAccumulator, grad, lr, l1, l2, globalStep],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Update '*var' according to the Adam algorithm.

lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)

*/




func applyAdam(scope:Scope,`var`: Tensorflow_DataType, m: Tensorflow_DataType, v: Tensorflow_DataType, beta1Power: Tensorflow_DataType, beta2Power: Tensorflow_DataType, lr: Tensorflow_DataType, beta1: Tensorflow_DataType, beta2: Tensorflow_DataType, epsilon: Tensorflow_DataType, grad: Tensorflow_DataType, t :type  , useLocking :bool  , useNesterov :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)
    attr.append("use_nesterov":useNesterov)

    let opspec = tf.OpSpec(
        OpType: "ApplyAdam",
        Input: [ `var`, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func applyCenteredRMSProp(scope:Scope,`var`: Tensorflow_DataType, mg: Tensorflow_DataType, ms: Tensorflow_DataType, mom: Tensorflow_DataType, lr: Tensorflow_DataType, rho: Tensorflow_DataType, momentum: Tensorflow_DataType, epsilon: Tensorflow_DataType, grad: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ApplyCenteredRMSProp",
        Input: [ `var`, mg, ms, mom, lr, rho, momentum, epsilon, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Update '*var' according to the Ftrl-proximal scheme.

accum_new = accum + grad * grad
linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new

*/




func applyFtrl(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, linear: Tensorflow_DataType, grad: Tensorflow_DataType, lr: Tensorflow_DataType, l1: Tensorflow_DataType, l2: Tensorflow_DataType, lrPower: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ApplyFtrl",
        Input: [ `var`, accum, linear, grad, lr, l1, l2, lrPower],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Update '*var' by subtracting 'alpha' * 'delta' from it.


*/




func applyGradientDescent(scope:Scope,`var`: Tensorflow_DataType, alpha: Tensorflow_DataType, delta: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ApplyGradientDescent",
        Input: [ `var`, alpha, delta],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Update '*var' according to the momentum scheme. Set use_nesterov = True if you

want to use Nesterov momentum.
accum = accum * momentum + grad
var -= lr * accum

*/




func applyMomentum(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, lr: Tensorflow_DataType, grad: Tensorflow_DataType, momentum: Tensorflow_DataType, t :type  , useLocking :bool  , useNesterov :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)
    attr.append("use_nesterov":useNesterov)

    let opspec = tf.OpSpec(
        OpType: "ApplyMomentum",
        Input: [ `var`, accum, lr, grad, momentum],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.

accum += grad * grad
prox_v = var - lr * grad * (1 / sqrt(accum))
var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}

*/




func applyProximalAdagrad(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, lr: Tensorflow_DataType, l1: Tensorflow_DataType, l2: Tensorflow_DataType, grad: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ApplyProximalAdagrad",
        Input: [ `var`, accum, lr, l1, l2, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Update '*var' as FOBOS algorithm with fixed learning rate.

prox_v = var - alpha * delta
var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

*/




func applyProximalGradientDescent(scope:Scope,`var`: Tensorflow_DataType, alpha: Tensorflow_DataType, l1: Tensorflow_DataType, l2: Tensorflow_DataType, delta: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ApplyProximalGradientDescent",
        Input: [ `var`, alpha, l1, l2, delta],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func applyRMSProp(scope:Scope,`var`: Tensorflow_DataType, ms: Tensorflow_DataType, mom: Tensorflow_DataType, lr: Tensorflow_DataType, rho: Tensorflow_DataType, momentum: Tensorflow_DataType, epsilon: Tensorflow_DataType, grad: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ApplyRMSProp",
        Input: [ `var`, ms, mom, lr, rho, momentum, epsilon, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the truth value of abs(x-y) < tolerance element-wise.


*/




func approximateEqual(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  , tolerance :float  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("tolerance":tolerance)

    let opspec = tf.OpSpec(
        OpType: "ApproximateEqual",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the index with the largest value across dimensions of a tensor.

Note that in case of ties the identity of the return value is not guaranteed.

*/




func argMax(scope:Scope,input: Tensorflow_DataType, dimension: Tensorflow_DataType, t :type  , tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "ArgMax",
        Input: [ input, dimension],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the index with the smallest value across dimensions of a tensor.

Note that in case of ties the identity of the return value is not guaranteed.

*/




func argMin(scope:Scope,input: Tensorflow_DataType, dimension: Tensorflow_DataType, t :type  , tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "ArgMin",
        Input: [ input, dimension],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Converts each entry in the given tensor to strings.  Supports many numeric

types and boolean.

*/




func asString(scope:Scope,input: Tensorflow_DataType, t :type  , precision :uInt8  , scientific :bool  , shortest :bool  , width :uInt8  , fill :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("precision":precision)
    attr.append("scientific":scientific)
    attr.append("shortest":shortest)
    attr.append("width":width)
    attr.append("fill":fill)

    let opspec = tf.OpSpec(
        OpType: "AsString",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes asin of x element-wise.


*/




func asin(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Asin",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Asserts that the given condition is true.

If `condition` evaluates to false, print the list of tensors in `data`.
`summarize` determines how many entries of the tensors to print.

*/




func assert(scope:Scope,condition: Tensorflow_DataType, data: Tensorflow_DataType, t :list(type)  , summarize :uInt8  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("summarize":summarize)

    let opspec = tf.OpSpec(
        OpType: "Assert",
        Input: [ condition, data],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Update 'ref' by assigning 'value' to it.

This operation outputs "ref" after the assignment is done.
This makes it easier to chain operations that need to use the reset value.

*/




func assign(scope:Scope,ref: Tensorflow_DataType, value: Tensorflow_DataType, t :type  , validateShape :bool  , useLocking :bool  )  ->(outputRef: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("validate_shape":validateShape)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "Assign",
        Input: [ ref, value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Update 'ref' by adding 'value' to it.

This operation outputs "ref" after the update is done.
This makes it easier to chain operations that need to use the reset value.

*/




func assignAdd(scope:Scope,ref: Tensorflow_DataType, value: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(outputRef: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "AssignAdd",
        Input: [ ref, value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Update 'ref' by subtracting 'value' from it.

This operation outputs "ref" after the update is done.
This makes it easier to chain operations that need to use the reset value.

*/




func assignSub(scope:Scope,ref: Tensorflow_DataType, value: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(outputRef: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "AssignSub",
        Input: [ ref, value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes atan of x element-wise.


*/




func atan(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Atan",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes arctangent of `y/x` element-wise, respecting signs of the arguments.

This is the angle \( \theta \in [-\pi, \pi] \) such that
\[ x = r \cos(\theta) \]
and
\[ y = r \sin(\theta) \]
where \(r = \sqrt(x^2 + y^2) \).

*/




func atan2(scope:Scope,y: Tensorflow_DataType, x: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Atan2",
        Input: [ y, x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func audioSpectrogram(scope:Scope,input: Tensorflow_DataType, windowSize :uInt8  , stride :uInt8  , magnitudeSquared :bool  )  ->(spectrogram: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("window_size":windowSize)
    attr.append("stride":stride)
    attr.append("magnitude_squared":magnitudeSquared)

    let opspec = tf.OpSpec(
        OpType: "AudioSpectrogram",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func audioSummary(scope:Scope,tag: Tensorflow_DataType, tensor: Tensorflow_DataType, sampleRate :float  , maxOutputs :uInt8  )  ->(summary: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("sample_rate":sampleRate)
    attr.append("max_outputs":maxOutputs)

    let opspec = tf.OpSpec(
        OpType: "AudioSummary",
        Input: [ tag, tensor],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func audioSummaryV2(scope:Scope,tag: Tensorflow_DataType, tensor: Tensorflow_DataType, sampleRate: Tensorflow_DataType, maxOutputs :uInt8  )  ->(summary: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("max_outputs":maxOutputs)

    let opspec = tf.OpSpec(
        OpType: "AudioSummaryV2",
        Input: [ tag, tensor, sampleRate],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Performs average pooling on the input.

Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.

*/




func avgPool(scope:Scope,value: Tensorflow_DataType, ksize :[Int]  , strides :[Int]  , padding :string  , dataFormat :string  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("ksize":ksize)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "AvgPool",
        Input: [ value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Performs 3D average pooling on the input.


*/




func avgPool3D(scope:Scope,input: Tensorflow_DataType, ksize :[Int]  , strides :[Int]  , padding :string  , dataFormat :string  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("ksize":ksize)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "AvgPool3D",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes gradients of average pooling function.


*/




func avgPool3DGrad(scope:Scope,origInputShape: Tensorflow_DataType, grad: Tensorflow_DataType, ksize :[Int]  , strides :[Int]  , padding :string  , dataFormat :string  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("ksize":ksize)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "AvgPool3DGrad",
        Input: [ origInputShape, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes gradients of the average pooling function.


*/




func avgPoolGrad(scope:Scope,origInputShape: Tensorflow_DataType, grad: Tensorflow_DataType, ksize :[Int]  , strides :[Int]  , padding :string  , dataFormat :string  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("ksize":ksize)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "AvgPoolGrad",
        Input: [ origInputShape, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func barrier(scope:Scope componentTypes :list(type)  , shapes :[Shape]  , capacity :uInt8  , container :string  , sharedName :string  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("component_types":componentTypes)
    attr.append("shapes":shapes)
    attr.append("capacity":capacity)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "Barrier",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Closes the given barrier.

This operation signals that no more new elements will be inserted in the
given barrier. Subsequent InsertMany that try to introduce a new key will fail.
Subsequent InsertMany operations that just add missing components to already
existing elements will continue to succeed. Subsequent TakeMany operations will
continue to succeed if sufficient completed elements remain in the barrier.
Subsequent TakeMany operations that would block will fail immediately.

*/




func barrierClose(scope:Scope,handle: Tensorflow_DataType, cancelPendingEnqueues :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("cancel_pending_enqueues":cancelPendingEnqueues)

    let opspec = tf.OpSpec(
        OpType: "BarrierClose",
        Input: [ handle],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Computes the number of incomplete elements in the given barrier.


*/




func barrierIncompleteSize(scope:Scope,handle: Tensorflow_DataType )  ->(size: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "BarrierIncompleteSize",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
For each key, assigns the respective value to the specified component.

If a key is not found in the barrier, this operation will create a new
incomplete element. If a key is found in the barrier, and the element
already has a value at component_index, this operation will fail with
INVALID_ARGUMENT, and leave the barrier in an undefined state.

*/




func barrierInsertMany(scope:Scope,handle: Tensorflow_DataType, keys: Tensorflow_DataType, values: Tensorflow_DataType, t :type  , componentIndex :uInt8  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("component_index":componentIndex)

    let opspec = tf.OpSpec(
        OpType: "BarrierInsertMany",
        Input: [ handle, keys, values],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Computes the number of complete elements in the given barrier.


*/




func barrierReadySize(scope:Scope,handle: Tensorflow_DataType )  ->(size: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "BarrierReadySize",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Takes the given number of completed elements from a barrier.

This operation concatenates completed-element component tensors along
the 0th dimension to make a single component tensor.
Elements come out of the barrier when they are complete, and in the order
in which they were placed into the barrier.  The indices output provides
information about the batch in which each element was originally inserted
into the barrier.

*/




func barrierTakeMany(scope:Scope,handle: Tensorflow_DataType, numElements: Tensorflow_DataType, componentTypes :list(type)  , allowSmallBatch :bool  , waitForIncomplete :bool  , timeoutMs :uInt8  )  ->(indices: tf.Output, keys: tf.Output, values: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("component_types":componentTypes)
    attr.append("allow_small_batch":allowSmallBatch)
    attr.append("wait_for_incomplete":waitForIncomplete)
    attr.append("timeout_ms":timeoutMs)

    let opspec = tf.OpSpec(
        OpType: "BarrierTakeMany",
        Input: [ handle, numElements],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*


*/




func batchCholesky(scope:Scope,input: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "BatchCholesky",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func batchCholeskyGrad(scope:Scope,l: Tensorflow_DataType, grad: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "BatchCholeskyGrad",
        Input: [ l, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a dataset that batches `batch_size` elements from `input_dataset`.


*/




func batchDataset(scope:Scope,inputDataset: Tensorflow_DataType, batchSize: Tensorflow_DataType, outputTypes :list(type)  , outputShapes :[Shape]  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("output_types":outputTypes)
    attr.append("output_shapes":outputShapes)

    let opspec = tf.OpSpec(
        OpType: "BatchDataset",
        Input: [ inputDataset, batchSize],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func batchFFT(scope:Scope,input: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "BatchFFT",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func batchFFT2D(scope:Scope,input: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "BatchFFT2D",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func batchFFT3D(scope:Scope,input: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "BatchFFT3D",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func batchIFFT(scope:Scope,input: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "BatchIFFT",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func batchIFFT2D(scope:Scope,input: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "BatchIFFT2D",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func batchIFFT3D(scope:Scope,input: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "BatchIFFT3D",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func batchMatMul(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  , adjX :bool  , adjY :bool  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("adj_x":adjX)
    attr.append("adj_y":adjY)

    let opspec = tf.OpSpec(
        OpType: "BatchMatMul",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func batchMatrixBandPart(scope:Scope,input: Tensorflow_DataType, numLower: Tensorflow_DataType, numUpper: Tensorflow_DataType, t :type  )  ->(band: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "BatchMatrixBandPart",
        Input: [ input, numLower, numUpper],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func batchMatrixDeterminant(scope:Scope,input: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "BatchMatrixDeterminant",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func batchMatrixDiag(scope:Scope,diagonal: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "BatchMatrixDiag",
        Input: [ diagonal],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func batchMatrixDiagPart(scope:Scope,input: Tensorflow_DataType, t :type  )  ->(diagonal: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "BatchMatrixDiagPart",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func batchMatrixInverse(scope:Scope,input: Tensorflow_DataType, adjoint :bool  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("adjoint":adjoint)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "BatchMatrixInverse",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func batchMatrixSetDiag(scope:Scope,input: Tensorflow_DataType, diagonal: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "BatchMatrixSetDiag",
        Input: [ input, diagonal],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func batchMatrixSolve(scope:Scope,matrix: Tensorflow_DataType, rhs: Tensorflow_DataType, adjoint :bool  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("adjoint":adjoint)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "BatchMatrixSolve",
        Input: [ matrix, rhs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func batchMatrixSolveLs(scope:Scope,matrix: Tensorflow_DataType, rhs: Tensorflow_DataType, l2Regularizer: Tensorflow_DataType, t :type  , fast :bool  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("fast":fast)

    let opspec = tf.OpSpec(
        OpType: "BatchMatrixSolveLs",
        Input: [ matrix, rhs, l2Regularizer],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func batchMatrixTriangularSolve(scope:Scope,matrix: Tensorflow_DataType, rhs: Tensorflow_DataType, lower :bool  , adjoint :bool  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("lower":lower)
    attr.append("adjoint":adjoint)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "BatchMatrixTriangularSolve",
        Input: [ matrix, rhs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Batch normalization.

This op is deprecated. Prefer `tf.nn.batch_normalization`.

*/




func batchNormWithGlobalNormalization(scope:Scope,t: Tensorflow_DataType, m: Tensorflow_DataType, v: Tensorflow_DataType, beta: Tensorflow_DataType, gamma: Tensorflow_DataType, t :type  , varianceEpsilon :float  , scaleAfterNormalization :bool  )  ->(result: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("variance_epsilon":varianceEpsilon)
    attr.append("scale_after_normalization":scaleAfterNormalization)

    let opspec = tf.OpSpec(
        OpType: "BatchNormWithGlobalNormalization",
        Input: [ t, m, v, beta, gamma],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Gradients for batch normalization.

This op is deprecated. See `tf.nn.batch_normalization`.

*/




func batchNormWithGlobalNormalizationGrad(scope:Scope,t: Tensorflow_DataType, m: Tensorflow_DataType, v: Tensorflow_DataType, gamma: Tensorflow_DataType, backprop: Tensorflow_DataType, t :type  , varianceEpsilon :float  , scaleAfterNormalization :bool  )  ->(dx: tf.Output, dm: tf.Output, dv: tf.Output, db: tf.Output, dg: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("variance_epsilon":varianceEpsilon)
    attr.append("scale_after_normalization":scaleAfterNormalization)

    let opspec = tf.OpSpec(
        OpType: "BatchNormWithGlobalNormalizationGrad",
        Input: [ t, m, v, gamma, backprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1),op.Output(4 - 1),op.Output(5 - 1))
}

/*


*/




func batchSelfAdjointEig(scope:Scope,input: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "BatchSelfAdjointEig",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func batchSelfAdjointEigV2(scope:Scope,input: Tensorflow_DataType, computeV :bool  , t :type  )  ->(e: tf.Output, v: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("compute_v":computeV)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "BatchSelfAdjointEigV2",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*


*/




func batchSvd(scope:Scope,input: Tensorflow_DataType, computeUv :bool  , fullMatrices :bool  , t :type  )  ->(s: tf.Output, u: tf.Output, v: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("compute_uv":computeUv)
    attr.append("full_matrices":fullMatrices)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "BatchSvd",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
BatchToSpace for 4-D tensors of type T.

This is a legacy version of the more general BatchToSpaceND.
Rearranges (permutes) data from batch into blocks of spatial data, followed by
cropping. This is the reverse transformation of SpaceToBatch. More specifically,
this op outputs a copy of the input tensor where values from the `batch`
dimension are moved in spatial blocks to the `height` and `width` dimensions,
followed by cropping along the `height` and `width` dimensions.

*/




func batchToSpace(scope:Scope,input: Tensorflow_DataType, crops: Tensorflow_DataType, t :type  , blockSize :uInt8  , tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("block_size":blockSize)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "BatchToSpace",
        Input: [ input, crops],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
BatchToSpace for N-D tensors of type T.

This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of shape
`block_shape + [batch]`, interleaves these blocks back into the grid defined by
the spatial dimensions `[1, ..., M]`, to obtain a result with the same rank as
the input.  The spatial dimensions of this intermediate result are then
optionally cropped according to `crops` to produce the output.  This is the
reverse of SpaceToBatch.  See below for a precise description.

*/




func batchToSpaceND(scope:Scope,input: Tensorflow_DataType, blockShape: Tensorflow_DataType, crops: Tensorflow_DataType, t :type  , tblockShape :type  , tcrops :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tblock_shape":tblockShape)
    attr.append("Tcrops":tcrops)

    let opspec = tf.OpSpec(
        OpType: "BatchToSpaceND",
        Input: [ input, blockShape, crops],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Compute the regularized incomplete beta integral \\(I_x(a, b)\\).

The regularized incomplete beta integral is defined as:
\\(I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}\\)
where
\\(B(x; a, b) = \int_0^x t^{a-1} (1 - t)^{b-1} dt\\)
is the incomplete beta function and \\(B(a, b)\\) is the *complete*
beta function.

*/




func betainc(scope:Scope,a: Tensorflow_DataType, b: Tensorflow_DataType, x: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Betainc",
        Input: [ a, b, x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Adds `bias` to `value`.

This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions.

*/




func biasAdd(scope:Scope,value: Tensorflow_DataType, bias: Tensorflow_DataType, t :type  , dataFormat :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("data_format":dataFormat)

    let opspec = tf.OpSpec(
        OpType: "BiasAdd",
        Input: [ value, bias],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
The backward operation for "BiasAdd" on the "bias" tensor.

It accumulates all the values from out_backprop into the feature dimension.
For NHWC data format, the feature dimension is the last. For NCHW data format,
the feature dimension is the third-to-last.

*/




func biasAddGrad(scope:Scope,outBackprop: Tensorflow_DataType, t :type  , dataFormat :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("data_format":dataFormat)

    let opspec = tf.OpSpec(
        OpType: "BiasAddGrad",
        Input: [ outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Adds `bias` to `value`.

This is a deprecated version of BiasAdd and will be soon removed.
This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions.

*/




func biasAddV1(scope:Scope,value: Tensorflow_DataType, bias: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "BiasAddV1",
        Input: [ value, bias],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Counts the number of occurrences of each value in an integer array.

Outputs a vector with length `size` and the same dtype as `weights`. If
`weights` are empty, then index `i` stores the number of times the value `i` is
counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
the value in `weights` at each index where the corresponding value in `arr` is
`i`.
Values in `arr` outside of the range [0, size) are ignored.

*/




func bincount(scope:Scope,arr: Tensorflow_DataType, size: Tensorflow_DataType, weights: Tensorflow_DataType, t :type  )  ->(bins: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Bincount",
        Input: [ arr, size, weights],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func bitcast(scope:Scope,input: Tensorflow_DataType, t :type  , type :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("type":type)

    let opspec = tf.OpSpec(
        OpType: "Bitcast",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Return the shape of s0 op s1 with broadcast.

Given `s0` and `s1`, tensors that represent shapes, compute `r0`, the
broadcasted shape. `s0`, `s1` and `r0` are all integer vectors.

*/




func broadcastArgs(scope:Scope,s0: Tensorflow_DataType, s1: Tensorflow_DataType, t :type  )  ->(r0: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "BroadcastArgs",
        Input: [ s0, s1],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Return the reduction indices for computing gradients of s0 op s1 with broadcast.

This is typically used by gradient computations for a broadcasting operation.

*/




func broadcastGradientArgs(scope:Scope,s0: Tensorflow_DataType, s1: Tensorflow_DataType, t :type  )  ->(r0: tf.Output, r1: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "BroadcastGradientArgs",
        Input: [ s0, s1],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
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

*/




func bucketize(scope:Scope,input: Tensorflow_DataType, t :type  , boundaries :list(float)  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("boundaries":boundaries)

    let opspec = tf.OpSpec(
        OpType: "Bucketize",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Performs beam search decoding on the logits given in input.

A note about the attribute merge_repeated: For the beam search decoder,
this means that if consecutive entries in a beam are the same, only
the first of these is emitted.  That is, when the top path is "A B B B B",
"A B" is returned if merge_repeated = True but "A B B B B" is
returned if merge_repeated = False.

*/




func ctcBeamSearchDecoder(scope:Scope,inputs: Tensorflow_DataType, sequenceLength: Tensorflow_DataType, beamWidth :uInt8  , topPaths :uInt8  , mergeRepeated :bool  )  ->(decodedIndices: tf.Output, decodedValues: tf.Output, decodedShape: tf.Output, logProbability: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("beam_width":beamWidth)
    attr.append("top_paths":topPaths)
    attr.append("merge_repeated":mergeRepeated)

    let opspec = tf.OpSpec(
        OpType: "CTCBeamSearchDecoder",
        Input: [ inputs, sequenceLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1),op.Output(4 - 1))
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

*/




func ctcGreedyDecoder(scope:Scope,inputs: Tensorflow_DataType, sequenceLength: Tensorflow_DataType, mergeRepeated :bool  )  ->(decodedIndices: tf.Output, decodedValues: tf.Output, decodedShape: tf.Output, logProbability: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("merge_repeated":mergeRepeated)

    let opspec = tf.OpSpec(
        OpType: "CTCGreedyDecoder",
        Input: [ inputs, sequenceLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1),op.Output(4 - 1))
}

/*
Calculates the CTC Loss (log probability) for each batch entry.  Also calculates

the gradient.  This class performs the softmax operation for you, so inputs
should be e.g. linear projections of outputs by an LSTM.

*/




func ctcLoss(scope:Scope,inputs: Tensorflow_DataType, labelsIndices: Tensorflow_DataType, labelsValues: Tensorflow_DataType, sequenceLength: Tensorflow_DataType, preprocessCollapseRepeated :bool  , ctcMergeRepeated :bool  , ignoreLongerOutputsThanInputs :bool  )  ->(loss: tf.Output, gradient: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("preprocess_collapse_repeated":preprocessCollapseRepeated)
    attr.append("ctc_merge_repeated":ctcMergeRepeated)
    attr.append("ignore_longer_outputs_than_inputs":ignoreLongerOutputsThanInputs)

    let opspec = tf.OpSpec(
        OpType: "CTCLoss",
        Input: [ inputs, labelsIndices, labelsValues, sequenceLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Cast x of type SrcT to y of DstT.


*/




func cast(scope:Scope,x: Tensorflow_DataType, srcT :type  , dstT :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("SrcT":srcT)
    attr.append("DstT":dstT)

    let opspec = tf.OpSpec(
        OpType: "Cast",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns element-wise smallest integer in not less than x.


*/




func ceil(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Ceil",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Checks a tensor for NaN and Inf values.

When run, reports an `InvalidArgument` error if `tensor` has any values
that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.

*/




func checkNumerics(scope:Scope,tensor: Tensorflow_DataType, t :type  , message :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("message":message)

    let opspec = tf.OpSpec(
        OpType: "CheckNumerics",
        Input: [ tensor],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the Cholesky decomposition of one or more square matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices, with the same constraints as the single matrix Cholesky
decomposition above. The output is a tensor of the same shape as the input
containing the Cholesky decompositions for all input submatrices `[..., :, :]`.

*/




func cholesky(scope:Scope,input: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Cholesky",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the reverse mode backpropagated gradient of the Cholesky algorithm.

For an explanation see "Differentiation of the Cholesky algorithm" by
Iain Murray http://arxiv.org/abs/1602.07527.

*/




func choleskyGrad(scope:Scope,l: Tensorflow_DataType, grad: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "CholeskyGrad",
        Input: [ l, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func complex(scope:Scope,real: Tensorflow_DataType, imag: Tensorflow_DataType, t :type  , tout :type  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tout":tout)

    let opspec = tf.OpSpec(
        OpType: "Complex",
        Input: [ real, imag],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the complex absolute value of a tensor.

Given a tensor `x` of complex numbers, this operation returns a tensor of type
`float` or `double` that is the absolute value of each element in `x`. All
elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
value is computed as \\( \sqrt{a^2 + b^2}\\).

*/




func complexAbs(scope:Scope,x: Tensorflow_DataType, t :type  , tout :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tout":tout)

    let opspec = tf.OpSpec(
        OpType: "ComplexAbs",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the ids of the positions in sampled_candidates that match true_labels.

When doing log-odds NCE, the result of this op should be passed through a
SparseToDense op, then added to the logits of the sampled candidates. This has
the effect of 'removing' the sampled labels that match the true labels by
making the classifier sure that they are sampled labels.

*/




func computeAccidentalHits(scope:Scope,trueClasses: Tensorflow_DataType, sampledCandidates: Tensorflow_DataType, numTrue :uInt8  , seed :uInt8  , seed2 :uInt8  )  ->(indices: tf.Output, ids: tf.Output, weights: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_true":numTrue)
    attr.append("seed":seed)
    attr.append("seed2":seed2)

    let opspec = tf.OpSpec(
        OpType: "ComputeAccidentalHits",
        Input: [ trueClasses, sampledCandidates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Concatenates tensors along one dimension.


*/




func concat(scope:Scope,concatDim: Tensorflow_DataType, values: Tensorflow_DataType, n :uInt8  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("N":n)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Concat",
        Input: [ concatDim, values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func concatOffset(scope:Scope,concatDim: Tensorflow_DataType, shape: Tensorflow_DataType, n :uInt8  )  ->(offset: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("N":n)

    let opspec = tf.OpSpec(
        OpType: "ConcatOffset",
        Input: [ concatDim, shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Concatenates tensors along one dimension.


*/




func concatV2(scope:Scope,values: Tensorflow_DataType, axis: Tensorflow_DataType, n :uInt8  , t :type  , tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("N":n)
    attr.append("T":t)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "ConcatV2",
        Input: [ values, axis],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A conditional accumulator for aggregating gradients.

The accumulator accepts gradients marked with local_step greater or
equal to the most recent global_step known to the accumulator. The
average can be extracted from the accumulator, provided sufficient
gradients have been accumulated. Extracting the average automatically
resets the aggregate to 0, and increments the global_step recorded by
the accumulator.

*/




func conditionalAccumulator(scope:Scope dtype :type  , shape :shape  , container :string  , sharedName :string  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("shape":shape)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "ConditionalAccumulator",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func conj(scope:Scope,input: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Conj",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns a constant tensor.


*/




func const(scope:Scope value :tensor  , dtype :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("value":value)
    attr.append("dtype":dtype)

    let opspec = tf.OpSpec(
        OpType: "Const",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Does nothing. Serves as a control trigger for scheduling.

Only useful as a placeholder for control edges.

*/




func controlTrigger(scope:Scope )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ControlTrigger",
        Input: [ ],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
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

*/




func conv2D(scope:Scope,input: Tensorflow_DataType, filter: Tensorflow_DataType, t :type  , strides :[Int]  , useCudnnOnGpu :bool  , padding :string  , dataFormat :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("strides":strides)
    attr.append("use_cudnn_on_gpu":useCudnnOnGpu)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)

    let opspec = tf.OpSpec(
        OpType: "Conv2D",
        Input: [ input, filter],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradients of convolution with respect to the filter.


*/




func conv2DBackpropFilter(scope:Scope,input: Tensorflow_DataType, filterSizes: Tensorflow_DataType, outBackprop: Tensorflow_DataType, t :type  , strides :[Int]  , useCudnnOnGpu :bool  , padding :string  , dataFormat :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("strides":strides)
    attr.append("use_cudnn_on_gpu":useCudnnOnGpu)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)

    let opspec = tf.OpSpec(
        OpType: "Conv2DBackpropFilter",
        Input: [ input, filterSizes, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradients of convolution with respect to the input.


*/




func conv2DBackpropInput(scope:Scope,inputSizes: Tensorflow_DataType, filter: Tensorflow_DataType, outBackprop: Tensorflow_DataType, t :type  , strides :[Int]  , useCudnnOnGpu :bool  , padding :string  , dataFormat :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("strides":strides)
    attr.append("use_cudnn_on_gpu":useCudnnOnGpu)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)

    let opspec = tf.OpSpec(
        OpType: "Conv2DBackpropInput",
        Input: [ inputSizes, filter, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes a 3-D convolution given 5-D `input` and `filter` tensors.

In signal processing, cross-correlation is a measure of similarity of
two waveforms as a function of a time-lag applied to one of them. This
is also known as a sliding dot product or sliding inner-product.
Our Conv3D implements a form of cross-correlation.

*/




func conv3D(scope:Scope,input: Tensorflow_DataType, filter: Tensorflow_DataType, t :type  , strides :[Int]  , padding :string  , dataFormat :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)

    let opspec = tf.OpSpec(
        OpType: "Conv3D",
        Input: [ input, filter],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradients of 3-D convolution with respect to the filter.


*/




func conv3DBackpropFilter(scope:Scope,input: Tensorflow_DataType, filter: Tensorflow_DataType, outBackprop: Tensorflow_DataType, t :type  , strides :[Int]  , padding :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("strides":strides)
    attr.append("padding":padding)

    let opspec = tf.OpSpec(
        OpType: "Conv3DBackpropFilter",
        Input: [ input, filter, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradients of 3-D convolution with respect to the filter.


*/




func conv3DBackpropFilterV2(scope:Scope,input: Tensorflow_DataType, filterSizes: Tensorflow_DataType, outBackprop: Tensorflow_DataType, t :type  , strides :[Int]  , padding :string  , dataFormat :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)

    let opspec = tf.OpSpec(
        OpType: "Conv3DBackpropFilterV2",
        Input: [ input, filterSizes, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradients of 3-D convolution with respect to the input.


*/




func conv3DBackpropInput(scope:Scope,input: Tensorflow_DataType, filter: Tensorflow_DataType, outBackprop: Tensorflow_DataType, t :type  , strides :[Int]  , padding :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("strides":strides)
    attr.append("padding":padding)

    let opspec = tf.OpSpec(
        OpType: "Conv3DBackpropInput",
        Input: [ input, filter, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradients of 3-D convolution with respect to the input.


*/




func conv3DBackpropInputV2(scope:Scope,inputSizes: Tensorflow_DataType, filter: Tensorflow_DataType, outBackprop: Tensorflow_DataType, t :type  , strides :[Int]  , padding :string  , dataFormat :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)

    let opspec = tf.OpSpec(
        OpType: "Conv3DBackpropInputV2",
        Input: [ inputSizes, filter, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func copy(scope:Scope,input: Tensorflow_DataType, t :type  , tensorName :string  , debugOpsSpec :list(string)  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("tensor_name":tensorName)
    attr.append("debug_ops_spec":debugOpsSpec)

    let opspec = tf.OpSpec(
        OpType: "Copy",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Copy Host Op.

Performs CPU-to-CPU deep-copying of tensor.
N.B.: If the all downstream attached debug ops are disabled given the current
gRPC gating status, the output will simply forward the input tensor without
deep-copying. See the documentation of Debug* ops for more details.
Unlike the Copy Op, this op has HostMemory constraint on its input or output.

*/




func copyHost(scope:Scope,input: Tensorflow_DataType, t :type  , tensorName :string  , debugOpsSpec :list(string)  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("tensor_name":tensorName)
    attr.append("debug_ops_spec":debugOpsSpec)

    let opspec = tf.OpSpec(
        OpType: "CopyHost",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes cos of x element-wise.


*/




func cos(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Cos",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Increments 'ref' until it reaches 'limit'.


*/




func countUpTo(scope:Scope,ref: Tensorflow_DataType, limit :uInt8  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("limit":limit)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "CountUpTo",
        Input: [ ref],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func cropAndResize(scope:Scope,image: Tensorflow_DataType, boxes: Tensorflow_DataType, boxInd: Tensorflow_DataType, cropSize: Tensorflow_DataType, t :type  , method :string  , extrapolationValue :float  )  ->(crops: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("method":method)
    attr.append("extrapolation_value":extrapolationValue)

    let opspec = tf.OpSpec(
        OpType: "CropAndResize",
        Input: [ image, boxes, boxInd, cropSize],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradient of the crop_and_resize op wrt the input boxes tensor.


*/




func cropAndResizeGradBoxes(scope:Scope,grads: Tensorflow_DataType, image: Tensorflow_DataType, boxes: Tensorflow_DataType, boxInd: Tensorflow_DataType, t :type  , method :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("method":method)

    let opspec = tf.OpSpec(
        OpType: "CropAndResizeGradBoxes",
        Input: [ grads, image, boxes, boxInd],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradient of the crop_and_resize op wrt the input image tensor.


*/




func cropAndResizeGradImage(scope:Scope,grads: Tensorflow_DataType, boxes: Tensorflow_DataType, boxInd: Tensorflow_DataType, imageSize: Tensorflow_DataType, t :type  , method :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("method":method)

    let opspec = tf.OpSpec(
        OpType: "CropAndResizeGradImage",
        Input: [ grads, boxes, boxInd, imageSize],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Compute the pairwise cross product.

`a` and `b` must be the same shape; they can either be simple 3-element vectors,
or any shape where the innermost dimension is 3. In the latter case, each pair
of corresponding 3-element vectors is cross-multiplied independently.

*/




func cross(scope:Scope,a: Tensorflow_DataType, b: Tensorflow_DataType, t :type  )  ->(product: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Cross",
        Input: [ a, b],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func cumprod(scope:Scope,x: Tensorflow_DataType, axis: Tensorflow_DataType, exclusive :bool  , reverse :bool  , t :type  , tidx :type  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("exclusive":exclusive)
    attr.append("reverse":reverse)
    attr.append("T":t)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "Cumprod",
        Input: [ x, axis],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func cumsum(scope:Scope,x: Tensorflow_DataType, axis: Tensorflow_DataType, exclusive :bool  , reverse :bool  , t :type  , tidx :type  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("exclusive":exclusive)
    attr.append("reverse":reverse)
    attr.append("T":t)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "Cumsum",
        Input: [ x, axis],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Debug Identity Op.

Provides an identity mapping of the non-Ref type input tensor for debugging.

*/




func debugIdentity(scope:Scope,input: Tensorflow_DataType, t :type  , deviceName :string  , tensorName :string  , debugUrls :list(string)  , gatedGrpc :bool  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("device_name":deviceName)
    attr.append("tensor_name":tensorName)
    attr.append("debug_urls":debugUrls)
    attr.append("gated_grpc":gatedGrpc)

    let opspec = tf.OpSpec(
        OpType: "DebugIdentity",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Debug NaN Value Counter Op

Counts number of NaNs in the input tensor, for debugging.

*/




func debugNanCount(scope:Scope,input: Tensorflow_DataType, t :type  , deviceName :string  , tensorName :string  , debugUrls :list(string)  , gatedGrpc :bool  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("device_name":deviceName)
    attr.append("tensor_name":tensorName)
    attr.append("debug_urls":debugUrls)
    attr.append("gated_grpc":gatedGrpc)

    let opspec = tf.OpSpec(
        OpType: "DebugNanCount",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Debug Numeric Summary Op.

Provide a basic summary of numeric value types, range and distribution.

*/




func debugNumericSummary(scope:Scope,input: Tensorflow_DataType, t :type  , deviceName :string  , tensorName :string  , debugUrls :list(string)  , lowerBound :float  , upperBound :float  , muteIfHealthy :bool  , gatedGrpc :bool  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("device_name":deviceName)
    attr.append("tensor_name":tensorName)
    attr.append("debug_urls":debugUrls)
    attr.append("lower_bound":lowerBound)
    attr.append("upper_bound":upperBound)
    attr.append("mute_if_healthy":muteIfHealthy)
    attr.append("gated_grpc":gatedGrpc)

    let opspec = tf.OpSpec(
        OpType: "DebugNumericSummary",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Decode web-safe base64-encoded strings.

Input may or may not have padding at the end. See EncodeBase64 for padding.
Web-safe means that input must use - and _ instead of + and /.

*/




func decodeBase64(scope:Scope,input: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "DecodeBase64",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Decode the first frame of a BMP-encoded image to a uint8 tensor.

The attr `channels` indicates the desired number of color channels for the
decoded image.
Accepted values are:
*   0: Use the number of channels in the BMP-encoded image.
*   3: output an RGB image.
*   4: output an RGBA image.

*/




func decodeBmp(scope:Scope,contents: Tensorflow_DataType, channels :uInt8  )  ->(image: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("channels":channels)

    let opspec = tf.OpSpec(
        OpType: "DecodeBmp",
        Input: [ contents],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Convert CSV records to tensors. Each column maps to one tensor.

RFC 4180 format is expected for the CSV records.
(https://tools.ietf.org/html/rfc4180)
Note that we allow leading and trailing spaces with int or float field.

*/




func decodeCSV(scope:Scope,records: Tensorflow_DataType, recordDefaults: Tensorflow_DataType, outType :list(type)  , fieldDelim :string  , useQuoteDelim :bool  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("OUT_TYPE":outType)
    attr.append("field_delim":fieldDelim)
    attr.append("use_quote_delim":useQuoteDelim)

    let opspec = tf.OpSpec(
        OpType: "DecodeCSV",
        Input: [ records, recordDefaults],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Decode the first frame of a GIF-encoded image to a uint8 tensor.

GIF with frame or transparency compression are not supported
convert animated GIF from compressed to uncompressed by:
    convert $src.gif -coalesce $dst.gif
This op also supports decoding JPEGs and PNGs, though it is cleaner to use
`tf.image.decode_image`.

*/




func decodeGif(scope:Scope,contents: Tensorflow_DataType )  ->(image: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "DecodeGif",
        Input: [ contents],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Convert JSON-encoded Example records to binary protocol buffer strings.

This op translates a tensor containing Example records, encoded using
the [standard JSON
mapping](https://developers.google.com/protocol-buffers/docs/proto3#json),
into a tensor containing the same records encoded as binary protocol
buffers. The resulting tensor can then be fed to any of the other
Example-parsing ops.

*/




func decodeJSONExample(scope:Scope,jsonExamples: Tensorflow_DataType )  ->(binaryExamples: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "DecodeJSONExample",
        Input: [ jsonExamples],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func decodeJpeg(scope:Scope,contents: Tensorflow_DataType, channels :uInt8  , ratio :uInt8  , fancyUpscaling :bool  , tryRecoverTruncated :bool  , acceptableFraction :float  , dctMethod :string  )  ->(image: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("channels":channels)
    attr.append("ratio":ratio)
    attr.append("fancy_upscaling":fancyUpscaling)
    attr.append("try_recover_truncated":tryRecoverTruncated)
    attr.append("acceptable_fraction":acceptableFraction)
    attr.append("dct_method":dctMethod)

    let opspec = tf.OpSpec(
        OpType: "DecodeJpeg",
        Input: [ contents],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func decodePng(scope:Scope,contents: Tensorflow_DataType, channels :uInt8  , dtype :type  )  ->(image: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("channels":channels)
    attr.append("dtype":dtype)

    let opspec = tf.OpSpec(
        OpType: "DecodePng",
        Input: [ contents],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Reinterpret the bytes of a string as a vector of numbers.


*/




func decodeRaw(scope:Scope,bytes: Tensorflow_DataType, outType :type  , littleEndian :bool  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("out_type":outType)
    attr.append("little_endian":littleEndian)

    let opspec = tf.OpSpec(
        OpType: "DecodeRaw",
        Input: [ bytes],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func decodeWav(scope:Scope,contents: Tensorflow_DataType, desiredChannels :uInt8  , desiredSamples :uInt8  )  ->(audio: tf.Output, sampleRate: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("desired_channels":desiredChannels)
    attr.append("desired_samples":desiredSamples)

    let opspec = tf.OpSpec(
        OpType: "DecodeWav",
        Input: [ contents],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Delete the tensor specified by its handle in the session.


*/




func deleteSessionTensor(scope:Scope,handle: Tensorflow_DataType )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "DeleteSessionTensor",
        Input: [ handle],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Applies set operation along last dimension of 2 `Tensor` inputs.

See SetOperationOp::SetOperationFromContext for values of `set_operation`.
Output `result` is a `SparseTensor` represented by `result_indices`,
`result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
dimension contains the result of `set_operation` applied to the corresponding
`[0...n-1]` dimension of `set`.

*/




func denseToDenseSetOperation(scope:Scope,set1: Tensorflow_DataType, set2: Tensorflow_DataType, setOperation :string  , validateIndices :bool  , t :type  )  ->(resultIndices: tf.Output, resultValues: tf.Output, resultShape: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("set_operation":setOperation)
    attr.append("validate_indices":validateIndices)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "DenseToDenseSetOperation",
        Input: [ set1, set2],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Creates a dataset that yields a SparseTensor for each element of the input.


*/




func denseToSparseBatchDataset(scope:Scope,inputDataset: Tensorflow_DataType, batchSize: Tensorflow_DataType, rowShape: Tensorflow_DataType, outputTypes :list(type)  , outputShapes :[Shape]  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("output_types":outputTypes)
    attr.append("output_shapes":outputShapes)

    let opspec = tf.OpSpec(
        OpType: "DenseToSparseBatchDataset",
        Input: [ inputDataset, batchSize, rowShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func denseToSparseSetOperation(scope:Scope,set1: Tensorflow_DataType, set2Indices: Tensorflow_DataType, set2Values: Tensorflow_DataType, set2Shape: Tensorflow_DataType, setOperation :string  , validateIndices :bool  , t :type  )  ->(resultIndices: tf.Output, resultValues: tf.Output, resultShape: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("set_operation":setOperation)
    attr.append("validate_indices":validateIndices)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "DenseToSparseSetOperation",
        Input: [ set1, set2Indices, set2Values, set2Shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
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

*/




func depthToSpace(scope:Scope,input: Tensorflow_DataType, t :type  , blockSize :uInt8  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("block_size":blockSize)

    let opspec = tf.OpSpec(
        OpType: "DepthToSpace",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func depthwiseConv2dNative(scope:Scope,input: Tensorflow_DataType, filter: Tensorflow_DataType, t :type  , strides :[Int]  , padding :string  , dataFormat :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)

    let opspec = tf.OpSpec(
        OpType: "DepthwiseConv2dNative",
        Input: [ input, filter],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradients of depthwise convolution with respect to the filter.


*/




func depthwiseConv2dNativeBackpropFilter(scope:Scope,input: Tensorflow_DataType, filterSizes: Tensorflow_DataType, outBackprop: Tensorflow_DataType, t :type  , strides :[Int]  , padding :string  , dataFormat :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)

    let opspec = tf.OpSpec(
        OpType: "DepthwiseConv2dNativeBackpropFilter",
        Input: [ input, filterSizes, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradients of depthwise convolution with respect to the input.


*/




func depthwiseConv2dNativeBackpropInput(scope:Scope,inputSizes: Tensorflow_DataType, filter: Tensorflow_DataType, outBackprop: Tensorflow_DataType, t :type  , strides :[Int]  , padding :string  , dataFormat :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)

    let opspec = tf.OpSpec(
        OpType: "DepthwiseConv2dNativeBackpropInput",
        Input: [ inputSizes, filter, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func dequantize(scope:Scope,input: Tensorflow_DataType, minRange: Tensorflow_DataType, maxRange: Tensorflow_DataType, t :type  , mode :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("mode":mode)

    let opspec = tf.OpSpec(
        OpType: "Dequantize",
        Input: [ input, minRange, maxRange],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func deserializeManySparse(scope:Scope,serializedSparse: Tensorflow_DataType, dtype :type  )  ->(sparseIndices: tf.Output, sparseValues: tf.Output, sparseShape: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)

    let opspec = tf.OpSpec(
        OpType: "DeserializeManySparse",
        Input: [ serializedSparse],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Destroys the temporary variable and returns its final value.

Sets output to the value of the Tensor pointed to by 'ref', then destroys
the temporary variable called 'var_name'.
All other uses of 'ref' *must* have executed before this op.
This is typically achieved by chaining the ref through each assign op, or by
using control dependencies.
Outputs the final value of the tensor pointed to by 'ref'.

*/




func destroyTemporaryVariable(scope:Scope,ref: Tensorflow_DataType, t :type  , varName :string  )  ->(value: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("var_name":varName)

    let opspec = tf.OpSpec(
        OpType: "DestroyTemporaryVariable",
        Input: [ ref],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func diag(scope:Scope,diagonal: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Diag",
        Input: [ diagonal],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func diagPart(scope:Scope,input: Tensorflow_DataType, t :type  )  ->(diagonal: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "DiagPart",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes Psi, the derivative of Lgamma (the log of the absolute value of

`Gamma(x)`), element-wise.

*/




func digamma(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Digamma",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func dilation2D(scope:Scope,input: Tensorflow_DataType, filter: Tensorflow_DataType, t :type  , strides :[Int]  , rates :[Int]  , padding :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("strides":strides)
    attr.append("rates":rates)
    attr.append("padding":padding)

    let opspec = tf.OpSpec(
        OpType: "Dilation2D",
        Input: [ input, filter],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradient of morphological 2-D dilation with respect to the filter.


*/




func dilation2DBackpropFilter(scope:Scope,input: Tensorflow_DataType, filter: Tensorflow_DataType, outBackprop: Tensorflow_DataType, t :type  , strides :[Int]  , rates :[Int]  , padding :string  )  ->(filterBackprop: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("strides":strides)
    attr.append("rates":rates)
    attr.append("padding":padding)

    let opspec = tf.OpSpec(
        OpType: "Dilation2DBackpropFilter",
        Input: [ input, filter, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradient of morphological 2-D dilation with respect to the input.


*/




func dilation2DBackpropInput(scope:Scope,input: Tensorflow_DataType, filter: Tensorflow_DataType, outBackprop: Tensorflow_DataType, t :type  , strides :[Int]  , rates :[Int]  , padding :string  )  ->(inBackprop: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("strides":strides)
    attr.append("rates":rates)
    attr.append("padding":padding)

    let opspec = tf.OpSpec(
        OpType: "Dilation2DBackpropInput",
        Input: [ input, filter, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns x / y element-wise.

*NOTE*: `Div` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func div(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Div",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func drawBoundingBoxes(scope:Scope,images: Tensorflow_DataType, boxes: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "DrawBoundingBoxes",
        Input: [ images, boxes],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func dynamicPartition(scope:Scope,data: Tensorflow_DataType, partitions: Tensorflow_DataType, numPartitions :uInt8  , t :type  )  ->(outputs: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_partitions":numPartitions)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "DynamicPartition",
        Input: [ data, partitions],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func dynamicStitch(scope:Scope,indices: Tensorflow_DataType, data: Tensorflow_DataType, n :uInt8  , t :type  )  ->(merged: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("N":n)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "DynamicStitch",
        Input: [ indices, data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the (possibly normalized) Levenshtein Edit Distance.

The inputs are variable-length sequences provided by SparseTensors
  (hypothesis_indices, hypothesis_values, hypothesis_shape)
and
  (truth_indices, truth_values, truth_shape).
The inputs are:

*/




func editDistance(scope:Scope,hypothesisIndices: Tensorflow_DataType, hypothesisValues: Tensorflow_DataType, hypothesisShape: Tensorflow_DataType, truthIndices: Tensorflow_DataType, truthValues: Tensorflow_DataType, truthShape: Tensorflow_DataType, normalize :bool  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("normalize":normalize)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "EditDistance",
        Input: [ hypothesisIndices, hypothesisValues, hypothesisShape, truthIndices, truthValues, truthShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.

See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
](http://arxiv.org/abs/1511.07289)

*/




func elu(scope:Scope,features: Tensorflow_DataType, t :type  )  ->(activations: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Elu",
        Input: [ features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes gradients for the exponential linear (Elu) operation.


*/




func eluGrad(scope:Scope,gradients: Tensorflow_DataType, outputs: Tensorflow_DataType, t :type  )  ->(backprops: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "EluGrad",
        Input: [ gradients, outputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Encode strings into web-safe base64 format.

Refer to the following article for more information on base64 format:
en.wikipedia.org/wiki/Base64. Base64 strings may have padding with '=' at the
end so that the encoded has length multiple of 4. See Padding section of the
link above.
Web-safe means that the encoder uses - and _ instead of + and /.

*/




func encodeBase64(scope:Scope,input: Tensorflow_DataType, pad :bool  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("pad":pad)

    let opspec = tf.OpSpec(
        OpType: "EncodeBase64",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func encodeJpeg(scope:Scope,image: Tensorflow_DataType, format :string  , quality :uInt8  , progressive :bool  , optimizeSize :bool  , chromaDownsampling :bool  , densityUnit :string  , xDensity :uInt8  , yDensity :uInt8  , xmpMetadata :string  )  ->(contents: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("format":format)
    attr.append("quality":quality)
    attr.append("progressive":progressive)
    attr.append("optimize_size":optimizeSize)
    attr.append("chroma_downsampling":chromaDownsampling)
    attr.append("density_unit":densityUnit)
    attr.append("x_density":xDensity)
    attr.append("y_density":yDensity)
    attr.append("xmp_metadata":xmpMetadata)

    let opspec = tf.OpSpec(
        OpType: "EncodeJpeg",
        Input: [ image],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func encodePng(scope:Scope,image: Tensorflow_DataType, compression :uInt8  , t :type  )  ->(contents: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("compression":compression)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "EncodePng",
        Input: [ image],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Encode audio data using the WAV file format.

This operation will generate a string suitable to be saved out to create a .wav
audio file. It will be encoded in the 16-bit PCM format. It takes in float
values in the range -1.0f to 1.0f, and any outside that value will be clamped to
that range.
`audio` is a 2-D float Tensor of shape `[length, channels]`.
`sample_rate` is a scalar Tensor holding the rate to use (e.g. 44100).

*/




func encodeWav(scope:Scope,audio: Tensorflow_DataType, sampleRate: Tensorflow_DataType )  ->(contents: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "EncodeWav",
        Input: [ audio, sampleRate],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates or finds a child frame, and makes `data` available to the child frame.

This op is used together with `Exit` to create loops in the graph.
The unique `frame_name` is used by the `Executor` to identify frames. If
`is_constant` is true, `output` is a constant in the child frame; otherwise
it may be changed in the child frame. At most `parallel_iterations` iterations
are run in parallel in the child frame.

*/




func enter(scope:Scope,data: Tensorflow_DataType, t :type  , frameName :string  , isConstant :bool  , parallelIterations :uInt8  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("frame_name":frameName)
    attr.append("is_constant":isConstant)
    attr.append("parallel_iterations":parallelIterations)

    let opspec = tf.OpSpec(
        OpType: "Enter",
        Input: [ data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the truth value of (x == y) element-wise.

*NOTE*: `Equal` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func equal(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Equal",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the Gauss error function of `x` element-wise.


*/




func erf(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Erf",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the complementary error function of `x` element-wise.


*/




func erfc(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Erfc",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Exits the current frame to its parent frame.

Exit makes its input `data` available to the parent frame.

*/




func exit(scope:Scope,data: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Exit",
        Input: [ data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes exponential of x element-wise.  \\(y = e^x\\).


*/




func exp(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Exp",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func expandDims(scope:Scope,input: Tensorflow_DataType, dim: Tensorflow_DataType, t :type  , tdim :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tdim":tdim)

    let opspec = tf.OpSpec(
        OpType: "ExpandDims",
        Input: [ input, dim],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes exponential of x - 1 element-wise.

I.e., \\(y = (\exp x) - 1\\).

*/




func expm1(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Expm1",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func extractGlimpse(scope:Scope,input: Tensorflow_DataType, size: Tensorflow_DataType, offsets: Tensorflow_DataType, centered :bool  , normalized :bool  , uniformNoise :bool  )  ->(glimpse: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("centered":centered)
    attr.append("normalized":normalized)
    attr.append("uniform_noise":uniformNoise)

    let opspec = tf.OpSpec(
        OpType: "ExtractGlimpse",
        Input: [ input, size, offsets],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Extract `patches` from `images` and put them in the "depth" output dimension.


*/




func extractImagePatches(scope:Scope,images: Tensorflow_DataType, ksizes :[Int]  , strides :[Int]  , rates :[Int]  , t :type  , padding :string  )  ->(patches: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("ksizes":ksizes)
    attr.append("strides":strides)
    attr.append("rates":rates)
    attr.append("T":t)
    attr.append("padding":padding)

    let opspec = tf.OpSpec(
        OpType: "ExtractImagePatches",
        Input: [ images],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Fast Fourier transform.

Computes the 1-dimensional discrete Fourier transform over the inner-most
dimension of `input`.

*/




func fft(scope:Scope,input: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "FFT",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
2D fast Fourier transform.

Computes the 2-dimensional discrete Fourier transform over the inner-most
2 dimensions of `input`.

*/




func fft2D(scope:Scope,input: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "FFT2D",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
3D fast Fourier transform.

Computes the 3-dimensional discrete Fourier transform over the inner-most 3
dimensions of `input`.

*/




func fft3D(scope:Scope,input: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "FFT3D",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A queue that produces elements in first-in first-out order.


*/




func fifoQueue(scope:Scope componentTypes :list(type)  , shapes :[Shape]  , capacity :uInt8  , container :string  , sharedName :string  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("component_types":componentTypes)
    attr.append("shapes":shapes)
    attr.append("capacity":capacity)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "FIFOQueue",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A queue that produces elements in first-in first-out order.


*/




func fifoQueueV2(scope:Scope componentTypes :list(type)  , shapes :[Shape]  , capacity :uInt8  , container :string  , sharedName :string  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("component_types":componentTypes)
    attr.append("shapes":shapes)
    attr.append("capacity":capacity)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "FIFOQueueV2",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Output a fact about factorials.


*/




func fact(scope:Scope )  ->(fact: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "Fact",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same type.

Attributes [min; max] define the clamping range for the 'inputs' data.  Op
divides this range into 255 steps (total of 256 values), then replaces each
'inputs' value with the closest of the quantized step values.
'num_bits' is the bitwidth of the quantization; between 2 and 8, inclusive.
Quantization is called fake since the output is still in floating point.

*/




func fakeQuantWithMinMaxArgs(scope:Scope,inputs: Tensorflow_DataType, min :float  , max :float  , numBits :uInt8  )  ->(outputs: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("min":min)
    attr.append("max":max)
    attr.append("num_bits":numBits)

    let opspec = tf.OpSpec(
        OpType: "FakeQuantWithMinMaxArgs",
        Input: [ inputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Compute gradients for a FakeQuantWithMinMaxArgs operation.


*/




func fakeQuantWithMinMaxArgsGradient(scope:Scope,gradients: Tensorflow_DataType, inputs: Tensorflow_DataType, min :float  , max :float  , numBits :uInt8  )  ->(backprops: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("min":min)
    attr.append("max":max)
    attr.append("num_bits":numBits)

    let opspec = tf.OpSpec(
        OpType: "FakeQuantWithMinMaxArgsGradient",
        Input: [ gradients, inputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Fake-quantize the 'inputs' tensor of type float via global float scalars `min`

and `max` to 'outputs' tensor of same shape as `inputs`.
[min; max] is the clamping range for the 'inputs' data.  Op divides this range
into 255 steps (total of 256 values), then replaces each 'inputs' value with the
closest of the quantized step values.
'num_bits' is the bitwidth of the quantization; between 2 and 8, inclusive.
This operation has a gradient and thus allows for training `min` and `max` values.

*/




func fakeQuantWithMinMaxVars(scope:Scope,inputs: Tensorflow_DataType, min: Tensorflow_DataType, max: Tensorflow_DataType, numBits :uInt8  )  ->(outputs: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_bits":numBits)

    let opspec = tf.OpSpec(
        OpType: "FakeQuantWithMinMaxVars",
        Input: [ inputs, min, max],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Compute gradients for a FakeQuantWithMinMaxVars operation.


*/




func fakeQuantWithMinMaxVarsGradient(scope:Scope,gradients: Tensorflow_DataType, inputs: Tensorflow_DataType, min: Tensorflow_DataType, max: Tensorflow_DataType, numBits :uInt8  )  ->(backpropsWrtInput: tf.Output, backpropWrtMin: tf.Output, backpropWrtMax: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_bits":numBits)

    let opspec = tf.OpSpec(
        OpType: "FakeQuantWithMinMaxVarsGradient",
        Input: [ gradients, inputs, min, max],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
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

*/




func fakeQuantWithMinMaxVarsPerChannel(scope:Scope,inputs: Tensorflow_DataType, min: Tensorflow_DataType, max: Tensorflow_DataType, numBits :uInt8  )  ->(outputs: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_bits":numBits)

    let opspec = tf.OpSpec(
        OpType: "FakeQuantWithMinMaxVarsPerChannel",
        Input: [ inputs, min, max],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation.


*/




func fakeQuantWithMinMaxVarsPerChannelGradient(scope:Scope,gradients: Tensorflow_DataType, inputs: Tensorflow_DataType, min: Tensorflow_DataType, max: Tensorflow_DataType, numBits :uInt8  )  ->(backpropsWrtInput: tf.Output, backpropWrtMin: tf.Output, backpropWrtMax: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_bits":numBits)

    let opspec = tf.OpSpec(
        OpType: "FakeQuantWithMinMaxVarsPerChannelGradient",
        Input: [ gradients, inputs, min, max],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Deprecated. Do not use.


*/




func fakeQueue(scope:Scope,resource: Tensorflow_DataType )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "FakeQueue",
        Input: [ resource],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func fill(scope:Scope,dims: Tensorflow_DataType, value: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Fill",
        Input: [ dims, value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a dataset containing elements of `input_dataset` matching `predicate`.

The `predicate` function must return a scalar boolean and accept the
following arguments:
* One tensor for each component of an element of `input_dataset`.
* One tensor for each value in `other_arguments`.

*/




func filterDataset(scope:Scope,inputDataset: Tensorflow_DataType, otherArguments: Tensorflow_DataType, predicate :func  , targuments :list(type)  , outputTypes :list(type)  , outputShapes :[Shape]  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("predicate":predicate)
    attr.append("Targuments":targuments)
    attr.append("output_types":outputTypes)
    attr.append("output_shapes":outputShapes)

    let opspec = tf.OpSpec(
        OpType: "FilterDataset",
        Input: [ inputDataset, otherArguments],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a dataset that emits the records from one or more binary files.


*/




func fixedLengthRecordDataset(scope:Scope,filenames: Tensorflow_DataType, headerBytes: Tensorflow_DataType, recordBytes: Tensorflow_DataType, footerBytes: Tensorflow_DataType )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "FixedLengthRecordDataset",
        Input: [ filenames, headerBytes, recordBytes, footerBytes],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A Reader that outputs fixed-length records from a file.


*/




func fixedLengthRecordReader(scope:Scope headerBytes :uInt8  , recordBytes :uInt8  , footerBytes :uInt8  , hopBytes :uInt8  , container :string  , sharedName :string  )  ->(readerHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("header_bytes":headerBytes)
    attr.append("record_bytes":recordBytes)
    attr.append("footer_bytes":footerBytes)
    attr.append("hop_bytes":hopBytes)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "FixedLengthRecordReader",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A Reader that outputs fixed-length records from a file.


*/




func fixedLengthRecordReaderV2(scope:Scope headerBytes :uInt8  , recordBytes :uInt8  , footerBytes :uInt8  , hopBytes :uInt8  , container :string  , sharedName :string  )  ->(readerHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("header_bytes":headerBytes)
    attr.append("record_bytes":recordBytes)
    attr.append("footer_bytes":footerBytes)
    attr.append("hop_bytes":hopBytes)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "FixedLengthRecordReaderV2",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func fixedUnigramCandidateSampler(scope:Scope,trueClasses: Tensorflow_DataType, numTrue :uInt8  , numSampled :uInt8  , unique :bool  , rangeMax :uInt8  , vocabFile :string  , distortion :float  , numReservedIds :uInt8  , numShards :uInt8  , shard :uInt8  , unigrams :list(float)  , seed :uInt8  , seed2 :uInt8  )  ->(sampledCandidates: tf.Output, trueExpectedCount: tf.Output, sampledExpectedCount: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_true":numTrue)
    attr.append("num_sampled":numSampled)
    attr.append("unique":unique)
    attr.append("range_max":rangeMax)
    attr.append("vocab_file":vocabFile)
    attr.append("distortion":distortion)
    attr.append("num_reserved_ids":numReservedIds)
    attr.append("num_shards":numShards)
    attr.append("shard":shard)
    attr.append("unigrams":unigrams)
    attr.append("seed":seed)
    attr.append("seed2":seed2)

    let opspec = tf.OpSpec(
        OpType: "FixedUnigramCandidateSampler",
        Input: [ trueClasses],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Creates a dataset that applies `f` to the outputs of `input_dataset`.

Unlike MapDataset, the `f` in FlatMapDataset is expected to return a
Dataset resource, and FlatMapDataset will flatten successive results
into a single Dataset.

*/




func flatMapDataset(scope:Scope,inputDataset: Tensorflow_DataType, otherArguments: Tensorflow_DataType, f :func  , targuments :list(type)  , outputTypes :list(type)  , outputShapes :[Shape]  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("f":f)
    attr.append("Targuments":targuments)
    attr.append("output_types":outputTypes)
    attr.append("output_shapes":outputShapes)

    let opspec = tf.OpSpec(
        OpType: "FlatMapDataset",
        Input: [ inputDataset, otherArguments],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns element-wise largest integer not greater than x.


*/




func floor(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Floor",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns x // y element-wise.

*NOTE*: `FloorDiv` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func floorDiv(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "FloorDiv",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns element-wise remainder of division. When `x < 0` xor `y < 0` is

true, this follows Python semantics in that the result here is consistent
with a flooring divide. E.g. `floor(x / y) * y + mod(x, y) = x`.
*NOTE*: `FloorMod` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func floorMod(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "FloorMod",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Performs fractional average pooling on the input.

Fractional average pooling is similar to Fractional max pooling in the pooling
region generation step. The only difference is that after pooling regions are
generated, a mean operation is performed instead of a max operation in each
pooling region.

*/




func fractionalAvgPool(scope:Scope,value: Tensorflow_DataType, poolingRatio :list(float)  , pseudoRandom :bool  , overlapping :bool  , deterministic :bool  , seed :uInt8  , seed2 :uInt8  , t :type  )  ->(output: tf.Output, rowPoolingSequence: tf.Output, colPoolingSequence: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("pooling_ratio":poolingRatio)
    attr.append("pseudo_random":pseudoRandom)
    attr.append("overlapping":overlapping)
    attr.append("deterministic":deterministic)
    attr.append("seed":seed)
    attr.append("seed2":seed2)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "FractionalAvgPool",
        Input: [ value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Computes gradient of the FractionalAvgPool function.

Unlike FractionalMaxPoolGrad, we don't need to find arg_max for
FractionalAvgPoolGrad, we just need to evenly back-propagate each element of
out_backprop to those indices that form the same pooling cell. Therefore, we
just need to know the shape of original input tensor, instead of the whole
tensor.

*/




func fractionalAvgPoolGrad(scope:Scope,origInputTensorShape: Tensorflow_DataType, outBackprop: Tensorflow_DataType, rowPoolingSequence: Tensorflow_DataType, colPoolingSequence: Tensorflow_DataType, overlapping :bool  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("overlapping":overlapping)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "FractionalAvgPoolGrad",
        Input: [ origInputTensorShape, outBackprop, rowPoolingSequence, colPoolingSequence],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func fractionalMaxPool(scope:Scope,value: Tensorflow_DataType, poolingRatio :list(float)  , pseudoRandom :bool  , overlapping :bool  , deterministic :bool  , seed :uInt8  , seed2 :uInt8  , t :type  )  ->(output: tf.Output, rowPoolingSequence: tf.Output, colPoolingSequence: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("pooling_ratio":poolingRatio)
    attr.append("pseudo_random":pseudoRandom)
    attr.append("overlapping":overlapping)
    attr.append("deterministic":deterministic)
    attr.append("seed":seed)
    attr.append("seed2":seed2)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "FractionalMaxPool",
        Input: [ value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Computes gradient of the FractionalMaxPool function.


*/




func fractionalMaxPoolGrad(scope:Scope,origInput: Tensorflow_DataType, origOutput: Tensorflow_DataType, outBackprop: Tensorflow_DataType, rowPoolingSequence: Tensorflow_DataType, colPoolingSequence: Tensorflow_DataType, overlapping :bool  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("overlapping":overlapping)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "FractionalMaxPoolGrad",
        Input: [ origInput, origOutput, outBackprop, rowPoolingSequence, colPoolingSequence],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Batch normalization.

Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.

*/




func fusedBatchNorm(scope:Scope,x: Tensorflow_DataType, scale: Tensorflow_DataType, offset: Tensorflow_DataType, mean: Tensorflow_DataType, variance: Tensorflow_DataType, t :type  , epsilon :float  , dataFormat :string  , isTraining :bool  )  ->(y: tf.Output, batchMean: tf.Output, batchVariance: tf.Output, reserveSpace1: tf.Output, reserveSpace2: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("epsilon":epsilon)
    attr.append("data_format":dataFormat)
    attr.append("is_training":isTraining)

    let opspec = tf.OpSpec(
        OpType: "FusedBatchNorm",
        Input: [ x, scale, offset, mean, variance],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1),op.Output(4 - 1),op.Output(5 - 1))
}

/*
Gradient for batch normalization.

Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.

*/




func fusedBatchNormGrad(scope:Scope,yBackprop: Tensorflow_DataType, x: Tensorflow_DataType, scale: Tensorflow_DataType, reserveSpace1: Tensorflow_DataType, reserveSpace2: Tensorflow_DataType, t :type  , epsilon :float  , dataFormat :string  , isTraining :bool  )  ->(xBackprop: tf.Output, scaleBackprop: tf.Output, offsetBackprop: tf.Output, reserveSpace3: tf.Output, reserveSpace4: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("epsilon":epsilon)
    attr.append("data_format":dataFormat)
    attr.append("is_training":isTraining)

    let opspec = tf.OpSpec(
        OpType: "FusedBatchNormGrad",
        Input: [ yBackprop, x, scale, reserveSpace1, reserveSpace2],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1),op.Output(4 - 1),op.Output(5 - 1))
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

*/




func fusedPadConv2D(scope:Scope,input: Tensorflow_DataType, paddings: Tensorflow_DataType, filter: Tensorflow_DataType, t :type  , mode :string  , strides :[Int]  , padding :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("mode":mode)
    attr.append("strides":strides)
    attr.append("padding":padding)

    let opspec = tf.OpSpec(
        OpType: "FusedPadConv2D",
        Input: [ input, paddings, filter],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func fusedResizeAndPadConv2D(scope:Scope,input: Tensorflow_DataType, size: Tensorflow_DataType, paddings: Tensorflow_DataType, filter: Tensorflow_DataType, t :type  , resizeAlignCorners :bool  , mode :string  , strides :[Int]  , padding :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("resize_align_corners":resizeAlignCorners)
    attr.append("mode":mode)
    attr.append("strides":strides)
    attr.append("padding":padding)

    let opspec = tf.OpSpec(
        OpType: "FusedResizeAndPadConv2D",
        Input: [ input, size, paddings, filter],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func gather(scope:Scope,params: Tensorflow_DataType, indices: Tensorflow_DataType, validateIndices :bool  , tparams :type  , tindices :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("validate_indices":validateIndices)
    attr.append("Tparams":tparams)
    attr.append("Tindices":tindices)

    let opspec = tf.OpSpec(
        OpType: "Gather",
        Input: [ params, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func gatherNd(scope:Scope,params: Tensorflow_DataType, indices: Tensorflow_DataType, tparams :type  , tindices :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tparams":tparams)
    attr.append("Tindices":tindices)

    let opspec = tf.OpSpec(
        OpType: "GatherNd",
        Input: [ params, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Store the input tensor in the state of the current session.


*/




func getSessionHandle(scope:Scope,value: Tensorflow_DataType, t :type  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "GetSessionHandle",
        Input: [ value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Store the input tensor in the state of the current session.


*/




func getSessionHandleV2(scope:Scope,value: Tensorflow_DataType, t :type  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "GetSessionHandleV2",
        Input: [ value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Get the value of the tensor specified by its handle.


*/




func getSessionTensor(scope:Scope,handle: Tensorflow_DataType, dtype :type  )  ->(value: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)

    let opspec = tf.OpSpec(
        OpType: "GetSessionTensor",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the truth value of (x > y) element-wise.

*NOTE*: `Greater` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func greater(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Greater",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the truth value of (x >= y) element-wise.

*NOTE*: `GreaterEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func greaterEqual(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "GreaterEqual",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a dataset that computes a windowed group-by on `input_dataset`.

// TODO(mrry): Support non-int64 keys.

*/




func groupByWindowDataset(scope:Scope,inputDataset: Tensorflow_DataType, keyFuncOtherArguments: Tensorflow_DataType, reduceFuncOtherArguments: Tensorflow_DataType, windowSize: Tensorflow_DataType, keyFunc :func  , reduceFunc :func  , tkeyFuncOtherArguments :list(type)  , treduceFuncOtherArguments :list(type)  , outputTypes :list(type)  , outputShapes :[Shape]  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("key_func":keyFunc)
    attr.append("reduce_func":reduceFunc)
    attr.append("Tkey_func_other_arguments":tkeyFuncOtherArguments)
    attr.append("Treduce_func_other_arguments":treduceFuncOtherArguments)
    attr.append("output_types":outputTypes)
    attr.append("output_shapes":outputShapes)

    let opspec = tf.OpSpec(
        OpType: "GroupByWindowDataset",
        Input: [ inputDataset, keyFuncOtherArguments, reduceFuncOtherArguments, windowSize],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Convert one or more images from HSV to RGB.

Outputs a tensor of the same shape as the `images` tensor, containing the RGB
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.
See `rgb_to_hsv` for a description of the HSV encoding.

*/




func hsvToRGB(scope:Scope,images: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "HSVToRGB",
        Input: [ images],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a non-initialized hash table.

This op creates a hash table, specifying the type of its keys and values.
Before using the table you will have to initialize it.  After initialization the
table will be immutable.

*/




func hashTable(scope:Scope container :string  , sharedName :string  , useNodeNameSharing :bool  , keyDtype :type  , valueDtype :type  )  ->(tableHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("container":container)
    attr.append("shared_name":sharedName)
    attr.append("use_node_name_sharing":useNodeNameSharing)
    attr.append("key_dtype":keyDtype)
    attr.append("value_dtype":valueDtype)

    let opspec = tf.OpSpec(
        OpType: "HashTable",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a non-initialized hash table.

This op creates a hash table, specifying the type of its keys and values.
Before using the table you will have to initialize it.  After initialization the
table will be immutable.

*/




func hashTableV2(scope:Scope container :string  , sharedName :string  , useNodeNameSharing :bool  , keyDtype :type  , valueDtype :type  )  ->(tableHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("container":container)
    attr.append("shared_name":sharedName)
    attr.append("use_node_name_sharing":useNodeNameSharing)
    attr.append("key_dtype":keyDtype)
    attr.append("value_dtype":valueDtype)

    let opspec = tf.OpSpec(
        OpType: "HashTableV2",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Outputs a `Summary` protocol buffer with a histogram.

The generated
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
has one summary value containing a histogram for `values`.
This op reports an `InvalidArgument` error if any value is not finite.

*/




func histogramSummary(scope:Scope,tag: Tensorflow_DataType, values: Tensorflow_DataType, t :type  )  ->(summary: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "HistogramSummary",
        Input: [ tag, values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Inverse fast Fourier transform.

Computes the inverse 1-dimensional discrete Fourier transform over the
inner-most dimension of `input`.

*/




func ifft(scope:Scope,input: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "IFFT",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Inverse 2D fast Fourier transform.

Computes the inverse 2-dimensional discrete Fourier transform over the
inner-most 2 dimensions of `input`.

*/




func ifft2D(scope:Scope,input: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "IFFT2D",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Inverse 3D fast Fourier transform.

Computes the inverse 3-dimensional discrete Fourier transform over the
inner-most 3 dimensions of `input`.

*/




func ifft3D(scope:Scope,input: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "IFFT3D",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func irfft(scope:Scope,input: Tensorflow_DataType, fftLength: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "IRFFT",
        Input: [ input, fftLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func irfft2D(scope:Scope,input: Tensorflow_DataType, fftLength: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "IRFFT2D",
        Input: [ input, fftLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func irfft3D(scope:Scope,input: Tensorflow_DataType, fftLength: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "IRFFT3D",
        Input: [ input, fftLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Return a tensor with the same shape and contents as the input tensor or value.


*/




func identity(scope:Scope,input: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Identity",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A Reader that outputs the queued work as both the key and value.

To use, enqueue strings in a Queue.  ReaderRead will take the front
work string and output (work, work).

*/




func identityReader(scope:Scope container :string  , sharedName :string  )  ->(readerHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "IdentityReader",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A Reader that outputs the queued work as both the key and value.

To use, enqueue strings in a Queue.  ReaderRead will take the front
work string and output (work, work).

*/




func identityReaderV2(scope:Scope container :string  , sharedName :string  )  ->(readerHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "IdentityReaderV2",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func igamma(scope:Scope,a: Tensorflow_DataType, x: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Igamma",
        Input: [ a, x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func igammac(scope:Scope,a: Tensorflow_DataType, x: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Igammac",
        Input: [ a, x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func imag(scope:Scope,input: Tensorflow_DataType, t :type  , tout :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tout":tout)

    let opspec = tf.OpSpec(
        OpType: "Imag",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func imageSummary(scope:Scope,tag: Tensorflow_DataType, tensor: Tensorflow_DataType, maxImages :uInt8  , t :type  , badColor :tensor  )  ->(summary: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("max_images":maxImages)
    attr.append("T":t)
    attr.append("bad_color":badColor)

    let opspec = tf.OpSpec(
        OpType: "ImageSummary",
        Input: [ tag, tensor],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns immutable tensor from memory region.

The current implementation memmaps the tensor from a file.

*/




func immutableConst(scope:Scope dtype :type  , shape :shape  , memoryRegionName :string  )  ->(tensor: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("shape":shape)
    attr.append("memory_region_name":memoryRegionName)

    let opspec = tf.OpSpec(
        OpType: "ImmutableConst",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func inTopK(scope:Scope,predictions: Tensorflow_DataType, targets: Tensorflow_DataType, k :uInt8  , t :type  )  ->(precision: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("k":k)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "InTopK",
        Input: [ predictions, targets],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Table initializer that takes two tensors for keys and values respectively.


*/




func initializeTable(scope:Scope,tableHandle: Tensorflow_DataType, keys: Tensorflow_DataType, values: Tensorflow_DataType, tkey :type  , tval :type  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tkey":tkey)
    attr.append("Tval":tval)

    let opspec = tf.OpSpec(
        OpType: "InitializeTable",
        Input: [ tableHandle, keys, values],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
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

*/




func initializeTableFromTextFile(scope:Scope,tableHandle: Tensorflow_DataType, filename: Tensorflow_DataType, keyIndex :uInt8  , valueIndex :uInt8  , vocabSize :uInt8  , delimiter :string  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("key_index":keyIndex)
    attr.append("value_index":valueIndex)
    attr.append("vocab_size":vocabSize)
    attr.append("delimiter":delimiter)

    let opspec = tf.OpSpec(
        OpType: "InitializeTableFromTextFile",
        Input: [ tableHandle, filename],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
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

*/




func initializeTableFromTextFileV2(scope:Scope,tableHandle: Tensorflow_DataType, filename: Tensorflow_DataType, keyIndex :uInt8  , valueIndex :uInt8  , vocabSize :uInt8  , delimiter :string  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("key_index":keyIndex)
    attr.append("value_index":valueIndex)
    attr.append("vocab_size":vocabSize)
    attr.append("delimiter":delimiter)

    let opspec = tf.OpSpec(
        OpType: "InitializeTableFromTextFileV2",
        Input: [ tableHandle, filename],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Table initializer that takes two tensors for keys and values respectively.


*/




func initializeTableV2(scope:Scope,tableHandle: Tensorflow_DataType, keys: Tensorflow_DataType, values: Tensorflow_DataType, tkey :type  , tval :type  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tkey":tkey)
    attr.append("Tval":tval)

    let opspec = tf.OpSpec(
        OpType: "InitializeTableV2",
        Input: [ tableHandle, keys, values],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Computes the reciprocal of x element-wise.

I.e., \\(y = 1 / x\\).

*/




func inv(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Inv",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradient for the inverse of `x` wrt its input.

Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
is the corresponding input gradient.

*/




func invGrad(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "InvGrad",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func invertPermutation(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "InvertPermutation",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns which elements of x are finite.

@compatibility(numpy)
Equivalent to np.isfinite
@end_compatibility

*/




func isFinite(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "IsFinite",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns which elements of x are Inf.

@compatibility(numpy)
Equivalent to np.isinf
@end_compatibility

*/




func isInf(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "IsInf",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns which elements of x are NaN.

@compatibility(numpy)
Equivalent to np.isnan
@end_compatibility

*/




func isNan(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "IsNan",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Checks whether a tensor has been initialized.

Outputs boolean scalar indicating whether the tensor has been initialized.

*/




func isVariableInitialized(scope:Scope,ref: Tensorflow_DataType, dtype :type  )  ->(isInitialized: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)

    let opspec = tf.OpSpec(
        OpType: "IsVariableInitialized",
        Input: [ ref],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A container for an iterator resource.


*/




func iterator(scope:Scope sharedName :string  , container :string  , outputTypes :list(type)  , outputShapes :[Shape]  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("shared_name":sharedName)
    attr.append("container":container)
    attr.append("output_types":outputTypes)
    attr.append("output_shapes":outputShapes)

    let opspec = tf.OpSpec(
        OpType: "Iterator",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Releases any resources used by the given iterator.


*/




func iteratorDispose(scope:Scope,iterator: Tensorflow_DataType )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "IteratorDispose",
        Input: [ iterator],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Gets the next output from the given iterator.


*/




func iteratorGetNext(scope:Scope,iterator: Tensorflow_DataType, outputTypes :list(type)  , outputShapes :[Shape]  )  ->(components: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("output_types":outputTypes)
    attr.append("output_shapes":outputShapes)

    let opspec = tf.OpSpec(
        OpType: "IteratorGetNext",
        Input: [ iterator],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
L2 Loss.

Computes half the L2 norm of a tensor without the `sqrt`:
    output = sum(t ** 2) / 2

*/




func l2Loss(scope:Scope,t: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "L2Loss",
        Input: [ t],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func lrn(scope:Scope,input: Tensorflow_DataType, depthRadius :uInt8  , bias :float  , alpha :float  , beta :float  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("depth_radius":depthRadius)
    attr.append("bias":bias)
    attr.append("alpha":alpha)
    attr.append("beta":beta)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "LRN",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Gradients for Local Response Normalization.


*/




func lrnGrad(scope:Scope,inputGrads: Tensorflow_DataType, inputImage: Tensorflow_DataType, outputImage: Tensorflow_DataType, depthRadius :uInt8  , bias :float  , alpha :float  , beta :float  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("depth_radius":depthRadius)
    attr.append("bias":bias)
    attr.append("alpha":alpha)
    attr.append("beta":beta)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "LRNGrad",
        Input: [ inputGrads, inputImage, outputImage],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func learnedUnigramCandidateSampler(scope:Scope,trueClasses: Tensorflow_DataType, numTrue :uInt8  , numSampled :uInt8  , unique :bool  , rangeMax :uInt8  , seed :uInt8  , seed2 :uInt8  )  ->(sampledCandidates: tf.Output, trueExpectedCount: tf.Output, sampledExpectedCount: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_true":numTrue)
    attr.append("num_sampled":numSampled)
    attr.append("unique":unique)
    attr.append("range_max":rangeMax)
    attr.append("seed":seed)
    attr.append("seed2":seed2)

    let opspec = tf.OpSpec(
        OpType: "LearnedUnigramCandidateSampler",
        Input: [ trueClasses],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Returns the truth value of (x < y) element-wise.

*NOTE*: `Less` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func less(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Less",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the truth value of (x <= y) element-wise.

*NOTE*: `LessEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func lessEqual(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "LessEqual",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the log of the absolute value of `Gamma(x)` element-wise.


*/




func lgamma(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Lgamma",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func linSpace(scope:Scope,start: Tensorflow_DataType, stop: Tensorflow_DataType, num: Tensorflow_DataType, t :type  , tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "LinSpace",
        Input: [ start, stop, num],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func listDiff(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  , outIdx :type  )  ->(out: tf.Output, idx: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("out_idx":outIdx)

    let opspec = tf.OpSpec(
        OpType: "ListDiff",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Computes natural logarithm of x element-wise.

I.e., \\(y = \log_e x\\).

*/




func log(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Log",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes natural logarithm of (1 + x) element-wise.

I.e., \\(y = \log_e (1 + x)\\).

*/




func log1p(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Log1p",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes log softmax activations.

For each batch `i` and class `j` we have
    logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))

*/




func logSoftmax(scope:Scope,logits: Tensorflow_DataType, t :type  )  ->(logsoftmax: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "LogSoftmax",
        Input: [ logits],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func logUniformCandidateSampler(scope:Scope,trueClasses: Tensorflow_DataType, numTrue :uInt8  , numSampled :uInt8  , unique :bool  , rangeMax :uInt8  , seed :uInt8  , seed2 :uInt8  )  ->(sampledCandidates: tf.Output, trueExpectedCount: tf.Output, sampledExpectedCount: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_true":numTrue)
    attr.append("num_sampled":numSampled)
    attr.append("unique":unique)
    attr.append("range_max":rangeMax)
    attr.append("seed":seed)
    attr.append("seed2":seed2)

    let opspec = tf.OpSpec(
        OpType: "LogUniformCandidateSampler",
        Input: [ trueClasses],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Returns the truth value of x AND y element-wise.

*NOTE*: `LogicalAnd` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func logicalAnd(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "LogicalAnd",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the truth value of NOT x element-wise.


*/




func logicalNot(scope:Scope,x: Tensorflow_DataType )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "LogicalNot",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the truth value of x OR y element-wise.

*NOTE*: `LogicalOr` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func logicalOr(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "LogicalOr",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Outputs all keys and values in the table.


*/




func lookupTableExport(scope:Scope,tableHandle: Tensorflow_DataType, tkeys :type  , tvalues :type  )  ->(keys: tf.Output, values: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tkeys":tkeys)
    attr.append("Tvalues":tvalues)

    let opspec = tf.OpSpec(
        OpType: "LookupTableExport",
        Input: [ tableHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Outputs all keys and values in the table.


*/




func lookupTableExportV2(scope:Scope,tableHandle: Tensorflow_DataType, tkeys :type  , tvalues :type  )  ->(keys: tf.Output, values: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tkeys":tkeys)
    attr.append("Tvalues":tvalues)

    let opspec = tf.OpSpec(
        OpType: "LookupTableExportV2",
        Input: [ tableHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Looks up keys in a table, outputs the corresponding values.

The tensor `keys` must of the same type as the keys of the table.
The output `values` is of the type of the table values.
The scalar `default_value` is the value output for keys not present in the
table. It must also be of the same type as the table values.

*/




func lookupTableFind(scope:Scope,tableHandle: Tensorflow_DataType, keys: Tensorflow_DataType, defaultValue: Tensorflow_DataType, tin :type  , tout :type  )  ->(values: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tin":tin)
    attr.append("Tout":tout)

    let opspec = tf.OpSpec(
        OpType: "LookupTableFind",
        Input: [ tableHandle, keys, defaultValue],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Looks up keys in a table, outputs the corresponding values.

The tensor `keys` must of the same type as the keys of the table.
The output `values` is of the type of the table values.
The scalar `default_value` is the value output for keys not present in the
table. It must also be of the same type as the table values.

*/




func lookupTableFindV2(scope:Scope,tableHandle: Tensorflow_DataType, keys: Tensorflow_DataType, defaultValue: Tensorflow_DataType, tin :type  , tout :type  )  ->(values: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tin":tin)
    attr.append("Tout":tout)

    let opspec = tf.OpSpec(
        OpType: "LookupTableFindV2",
        Input: [ tableHandle, keys, defaultValue],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Replaces the contents of the table with the specified keys and values.

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

*/




func lookupTableImport(scope:Scope,tableHandle: Tensorflow_DataType, keys: Tensorflow_DataType, values: Tensorflow_DataType, tin :type  , tout :type  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tin":tin)
    attr.append("Tout":tout)

    let opspec = tf.OpSpec(
        OpType: "LookupTableImport",
        Input: [ tableHandle, keys, values],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Replaces the contents of the table with the specified keys and values.

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

*/




func lookupTableImportV2(scope:Scope,tableHandle: Tensorflow_DataType, keys: Tensorflow_DataType, values: Tensorflow_DataType, tin :type  , tout :type  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tin":tin)
    attr.append("Tout":tout)

    let opspec = tf.OpSpec(
        OpType: "LookupTableImportV2",
        Input: [ tableHandle, keys, values],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Updates the table to associates keys with values.

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

*/




func lookupTableInsert(scope:Scope,tableHandle: Tensorflow_DataType, keys: Tensorflow_DataType, values: Tensorflow_DataType, tin :type  , tout :type  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tin":tin)
    attr.append("Tout":tout)

    let opspec = tf.OpSpec(
        OpType: "LookupTableInsert",
        Input: [ tableHandle, keys, values],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Updates the table to associates keys with values.

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

*/




func lookupTableInsertV2(scope:Scope,tableHandle: Tensorflow_DataType, keys: Tensorflow_DataType, values: Tensorflow_DataType, tin :type  , tout :type  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tin":tin)
    attr.append("Tout":tout)

    let opspec = tf.OpSpec(
        OpType: "LookupTableInsertV2",
        Input: [ tableHandle, keys, values],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Computes the number of elements in the given table.


*/




func lookupTableSize(scope:Scope,tableHandle: Tensorflow_DataType )  ->(size: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "LookupTableSize",
        Input: [ tableHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the number of elements in the given table.


*/




func lookupTableSizeV2(scope:Scope,tableHandle: Tensorflow_DataType )  ->(size: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "LookupTableSizeV2",
        Input: [ tableHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Forwards the input to the output.

This operator represents the loop termination condition used by the
"pivot" switches of a loop.

*/




func loopCond(scope:Scope,input: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "LoopCond",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Makes a new iterator from the given `dataset` and stores it in `iterator`.

This operation may be executed multiple times. Each execution will reset the
iterator in `iterator` to the first element of `dataset`.

*/




func makeIterator(scope:Scope,dataset: Tensorflow_DataType, iterator: Tensorflow_DataType )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "MakeIterator",
        Input: [ dataset, iterator],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Creates a dataset that applies `f` to the outputs of `input_dataset`.


*/




func mapDataset(scope:Scope,inputDataset: Tensorflow_DataType, otherArguments: Tensorflow_DataType, f :func  , targuments :list(type)  , outputTypes :list(type)  , outputShapes :[Shape]  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("f":f)
    attr.append("Targuments":targuments)
    attr.append("output_types":outputTypes)
    attr.append("output_shapes":outputShapes)

    let opspec = tf.OpSpec(
        OpType: "MapDataset",
        Input: [ inputDataset, otherArguments],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Multiply the matrix "a" by the matrix "b".

The inputs must be two-dimensional matrices and the inner dimension of
"a" (after being transposed if transpose_a is true) must match the
outer dimension of "b" (after being transposed if transposed_b is
true).
*Note*: The default kernel implementation for MatMul on GPUs uses
cublas.

*/




func matMul(scope:Scope,a: Tensorflow_DataType, b: Tensorflow_DataType, transposeA :bool  , transposeB :bool  , t :type  )  ->(product: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("transpose_a":transposeA)
    attr.append("transpose_b":transposeB)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "MatMul",
        Input: [ a, b],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the set of files matching one or more glob patterns.

Note that this routine only supports wildcard characters in the
basename portion of the pattern, not in the directory portion.

*/




func matchingFiles(scope:Scope,pattern: Tensorflow_DataType )  ->(filenames: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "MatchingFiles",
        Input: [ pattern],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func matrixBandPart(scope:Scope,input: Tensorflow_DataType, numLower: Tensorflow_DataType, numUpper: Tensorflow_DataType, t :type  )  ->(band: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "MatrixBandPart",
        Input: [ input, numLower, numUpper],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the determinant of one ore more square matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor containing the determinants
for all input submatrices `[..., :, :]`.

*/




func matrixDeterminant(scope:Scope,input: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "MatrixDeterminant",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func matrixDiag(scope:Scope,diagonal: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "MatrixDiag",
        Input: [ diagonal],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func matrixDiagPart(scope:Scope,input: Tensorflow_DataType, t :type  )  ->(diagonal: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "MatrixDiagPart",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func matrixInverse(scope:Scope,input: Tensorflow_DataType, adjoint :bool  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("adjoint":adjoint)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "MatrixInverse",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func matrixSetDiag(scope:Scope,input: Tensorflow_DataType, diagonal: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "MatrixSetDiag",
        Input: [ input, diagonal],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Solves systems of linear equations.

`Matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. `Rhs` is a tensor of shape `[..., M, K]`. The `output` is
a tensor shape `[..., M, K]`.  If `adjoint` is `False` then each output matrix
satisfies `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
If `adjoint` is `True` then each output matrix satisfies
`adjoint(matrix[..., :, :]) * output[..., :, :] = rhs[..., :, :]`.

*/




func matrixSolve(scope:Scope,matrix: Tensorflow_DataType, rhs: Tensorflow_DataType, adjoint :bool  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("adjoint":adjoint)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "MatrixSolve",
        Input: [ matrix, rhs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func matrixSolveLs(scope:Scope,matrix: Tensorflow_DataType, rhs: Tensorflow_DataType, l2Regularizer: Tensorflow_DataType, t :type  , fast :bool  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("fast":fast)

    let opspec = tf.OpSpec(
        OpType: "MatrixSolveLs",
        Input: [ matrix, rhs, l2Regularizer],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func matrixTriangularSolve(scope:Scope,matrix: Tensorflow_DataType, rhs: Tensorflow_DataType, lower :bool  , adjoint :bool  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("lower":lower)
    attr.append("adjoint":adjoint)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "MatrixTriangularSolve",
        Input: [ matrix, rhs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the maximum of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

*/




func max(scope:Scope,input: Tensorflow_DataType, reductionIndices: Tensorflow_DataType, keepDims :bool  , t :type  , tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("keep_dims":keepDims)
    attr.append("T":t)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "Max",
        Input: [ input, reductionIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Performs max pooling on the input.


*/




func maxPool(scope:Scope,input: Tensorflow_DataType, t :type  , ksize :[Int]  , strides :[Int]  , padding :string  , dataFormat :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("ksize":ksize)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)

    let opspec = tf.OpSpec(
        OpType: "MaxPool",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Performs 3D max pooling on the input.


*/




func maxPool3D(scope:Scope,input: Tensorflow_DataType, ksize :[Int]  , strides :[Int]  , padding :string  , dataFormat :string  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("ksize":ksize)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "MaxPool3D",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes gradients of max pooling function.


*/




func maxPool3DGrad(scope:Scope,origInput: Tensorflow_DataType, origOutput: Tensorflow_DataType, grad: Tensorflow_DataType, ksize :[Int]  , strides :[Int]  , padding :string  , dataFormat :string  , t :type  , tInput :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("ksize":ksize)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)
    attr.append("T":t)
    attr.append("TInput":tInput)

    let opspec = tf.OpSpec(
        OpType: "MaxPool3DGrad",
        Input: [ origInput, origOutput, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes second-order gradients of the maxpooling function.


*/




func maxPool3DGradGrad(scope:Scope,origInput: Tensorflow_DataType, origOutput: Tensorflow_DataType, grad: Tensorflow_DataType, ksize :[Int]  , strides :[Int]  , padding :string  , dataFormat :string  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("ksize":ksize)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "MaxPool3DGradGrad",
        Input: [ origInput, origOutput, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes gradients of the maxpooling function.


*/




func maxPoolGrad(scope:Scope,origInput: Tensorflow_DataType, origOutput: Tensorflow_DataType, grad: Tensorflow_DataType, ksize :[Int]  , strides :[Int]  , padding :string  , dataFormat :string  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("ksize":ksize)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "MaxPoolGrad",
        Input: [ origInput, origOutput, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes second-order gradients of the maxpooling function.


*/




func maxPoolGradGrad(scope:Scope,origInput: Tensorflow_DataType, origOutput: Tensorflow_DataType, grad: Tensorflow_DataType, ksize :[Int]  , strides :[Int]  , padding :string  , dataFormat :string  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("ksize":ksize)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("data_format":dataFormat)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "MaxPoolGradGrad",
        Input: [ origInput, origOutput, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes second-order gradients of the maxpooling function.


*/




func maxPoolGradGradWithArgmax(scope:Scope,input: Tensorflow_DataType, grad: Tensorflow_DataType, argmax: Tensorflow_DataType, ksize :[Int]  , strides :[Int]  , padding :string  , targmax :type  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("ksize":ksize)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("Targmax":targmax)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "MaxPoolGradGradWithArgmax",
        Input: [ input, grad, argmax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes gradients of the maxpooling function.


*/




func maxPoolGradWithArgmax(scope:Scope,input: Tensorflow_DataType, grad: Tensorflow_DataType, argmax: Tensorflow_DataType, ksize :[Int]  , strides :[Int]  , padding :string  , targmax :type  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("ksize":ksize)
    attr.append("strides":strides)
    attr.append("padding":padding)
    attr.append("Targmax":targmax)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "MaxPoolGradWithArgmax",
        Input: [ input, grad, argmax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Performs max pooling on the input and outputs both max values and indices.

The indices in `argmax` are flattened, so that a maximum value at position
`[b, y, x, c]` becomes flattened index
`((b * height + y) * width + x) * channels + c`.

*/




func maxPoolWithArgmax(scope:Scope,input: Tensorflow_DataType, ksize :[Int]  , strides :[Int]  , targmax :type  , padding :string  , t :type  )  ->(output: tf.Output, argmax: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("ksize":ksize)
    attr.append("strides":strides)
    attr.append("Targmax":targmax)
    attr.append("padding":padding)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "MaxPoolWithArgmax",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Returns the max of x and y (i.e. x > y ? x : y) element-wise.

*NOTE*: `Maximum` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func maximum(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Maximum",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the mean of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

*/




func mean(scope:Scope,input: Tensorflow_DataType, reductionIndices: Tensorflow_DataType, keepDims :bool  , t :type  , tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("keep_dims":keepDims)
    attr.append("T":t)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "Mean",
        Input: [ input, reductionIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Forwards the value of an available tensor from `inputs` to `output`.

`Merge` waits for at least one of the tensors in `inputs` to become available.
It is usually combined with `Switch` to implement branching.
`Merge` forwards the first tensor to become available to `output`, and sets
`value_index` to its index in `inputs`.

*/




func merge(scope:Scope,inputs: Tensorflow_DataType, t :type  , n :uInt8  )  ->(output: tf.Output, valueIndex: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("N":n)

    let opspec = tf.OpSpec(
        OpType: "Merge",
        Input: [ inputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Merges summaries.

This op creates a
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
protocol buffer that contains the union of all the values in the input
summaries.
When the Op is run, it reports an `InvalidArgument` error if multiple values
in the summaries to merge use the same tag.

*/




func mergeSummary(scope:Scope,inputs: Tensorflow_DataType, n :uInt8  )  ->(summary: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("N":n)

    let opspec = tf.OpSpec(
        OpType: "MergeSummary",
        Input: [ inputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
V2 format specific: merges the metadata files of sharded checkpoints.  The

result is one logical checkpoint, with one physical metadata file and renamed
data files.
Intended for "grouping" multiple checkpoints in a sharded checkpoint setup.
If delete_old_dirs is true, attempts to delete recursively the dirname of each
path in the input checkpoint_prefixes.  This is useful when those paths are non
user-facing temporary locations.

*/




func mergeV2Checkpoints(scope:Scope,checkpointPrefixes: Tensorflow_DataType, destinationPrefix: Tensorflow_DataType, deleteOldDirs :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("delete_old_dirs":deleteOldDirs)

    let opspec = tf.OpSpec(
        OpType: "MergeV2Checkpoints",
        Input: [ checkpointPrefixes, destinationPrefix],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Transforms a spectrogram into a form that's useful for speech recognition.

Mel Frequency Cepstral Coefficients are a way of representing audio data that's
been effective as an input feature for machine learning. They are created by
taking the spectrum of a spectrogram (a 'cepstrum'), and discarding some of the
higher frequencies that are less significant to the human ear. They have a long
history in the speech recognition world, and https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
is a good resource to learn more.

*/




func mfcc(scope:Scope,spectrogram: Tensorflow_DataType, sampleRate: Tensorflow_DataType, upperFrequencyLimit :float  , lowerFrequencyLimit :float  , filterbankChannelCount :uInt8  , dctCoefficientCount :uInt8  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("upper_frequency_limit":upperFrequencyLimit)
    attr.append("lower_frequency_limit":lowerFrequencyLimit)
    attr.append("filterbank_channel_count":filterbankChannelCount)
    attr.append("dct_coefficient_count":dctCoefficientCount)

    let opspec = tf.OpSpec(
        OpType: "Mfcc",
        Input: [ spectrogram, sampleRate],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the minimum of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

*/




func min(scope:Scope,input: Tensorflow_DataType, reductionIndices: Tensorflow_DataType, keepDims :bool  , t :type  , tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("keep_dims":keepDims)
    attr.append("T":t)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "Min",
        Input: [ input, reductionIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the min of x and y (i.e. x < y ? x : y) element-wise.

*NOTE*: `Minimum` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func minimum(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Minimum",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func mirrorPad(scope:Scope,input: Tensorflow_DataType, paddings: Tensorflow_DataType, t :type  , tpaddings :type  , mode :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tpaddings":tpaddings)
    attr.append("mode":mode)

    let opspec = tf.OpSpec(
        OpType: "MirrorPad",
        Input: [ input, paddings],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func mirrorPadGrad(scope:Scope,input: Tensorflow_DataType, paddings: Tensorflow_DataType, t :type  , tpaddings :type  , mode :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tpaddings":tpaddings)
    attr.append("mode":mode)

    let opspec = tf.OpSpec(
        OpType: "MirrorPadGrad",
        Input: [ input, paddings],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns element-wise remainder of division. This emulates C semantics in that

the result here is consistent with a truncating divide. E.g. `truncate(x / y) *
y + truncate_mod(x, y) = x`.
*NOTE*: `Mod` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func mod(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Mod",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns x * y element-wise.

*NOTE*: `Mul` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func mul(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Mul",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Draws samples from a multinomial distribution.


*/




func multinomial(scope:Scope,logits: Tensorflow_DataType, numSamples: Tensorflow_DataType, seed :uInt8  , seed2 :uInt8  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("seed":seed)
    attr.append("seed2":seed2)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Multinomial",
        Input: [ logits, numSamples],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates an empty hash table that uses tensors as the backing store.

It uses "open addressing" with quadratic reprobing to resolve
collisions.
This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

*/




func mutableDenseHashTable(scope:Scope,emptyKey: Tensorflow_DataType, container :string  , sharedName :string  , useNodeNameSharing :bool  , keyDtype :type  , valueDtype :type  , valueShape :shape  , initialNumBuckets :uInt8  , maxLoadFactor :float  )  ->(tableHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("container":container)
    attr.append("shared_name":sharedName)
    attr.append("use_node_name_sharing":useNodeNameSharing)
    attr.append("key_dtype":keyDtype)
    attr.append("value_dtype":valueDtype)
    attr.append("value_shape":valueShape)
    attr.append("initial_num_buckets":initialNumBuckets)
    attr.append("max_load_factor":maxLoadFactor)

    let opspec = tf.OpSpec(
        OpType: "MutableDenseHashTable",
        Input: [ emptyKey],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates an empty hash table that uses tensors as the backing store.

It uses "open addressing" with quadratic reprobing to resolve
collisions.
This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

*/




func mutableDenseHashTableV2(scope:Scope,emptyKey: Tensorflow_DataType, container :string  , sharedName :string  , useNodeNameSharing :bool  , keyDtype :type  , valueDtype :type  , valueShape :shape  , initialNumBuckets :uInt8  , maxLoadFactor :float  )  ->(tableHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("container":container)
    attr.append("shared_name":sharedName)
    attr.append("use_node_name_sharing":useNodeNameSharing)
    attr.append("key_dtype":keyDtype)
    attr.append("value_dtype":valueDtype)
    attr.append("value_shape":valueShape)
    attr.append("initial_num_buckets":initialNumBuckets)
    attr.append("max_load_factor":maxLoadFactor)

    let opspec = tf.OpSpec(
        OpType: "MutableDenseHashTableV2",
        Input: [ emptyKey],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

*/




func mutableHashTable(scope:Scope container :string  , sharedName :string  , useNodeNameSharing :bool  , keyDtype :type  , valueDtype :type  )  ->(tableHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("container":container)
    attr.append("shared_name":sharedName)
    attr.append("use_node_name_sharing":useNodeNameSharing)
    attr.append("key_dtype":keyDtype)
    attr.append("value_dtype":valueDtype)

    let opspec = tf.OpSpec(
        OpType: "MutableHashTable",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a vector. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

*/




func mutableHashTableOfTensors(scope:Scope container :string  , sharedName :string  , useNodeNameSharing :bool  , keyDtype :type  , valueDtype :type  , valueShape :shape  )  ->(tableHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("container":container)
    attr.append("shared_name":sharedName)
    attr.append("use_node_name_sharing":useNodeNameSharing)
    attr.append("key_dtype":keyDtype)
    attr.append("value_dtype":valueDtype)
    attr.append("value_shape":valueShape)

    let opspec = tf.OpSpec(
        OpType: "MutableHashTableOfTensors",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a vector. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

*/




func mutableHashTableOfTensorsV2(scope:Scope container :string  , sharedName :string  , useNodeNameSharing :bool  , keyDtype :type  , valueDtype :type  , valueShape :shape  )  ->(tableHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("container":container)
    attr.append("shared_name":sharedName)
    attr.append("use_node_name_sharing":useNodeNameSharing)
    attr.append("key_dtype":keyDtype)
    attr.append("value_dtype":valueDtype)
    attr.append("value_shape":valueShape)

    let opspec = tf.OpSpec(
        OpType: "MutableHashTableOfTensorsV2",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

*/




func mutableHashTableV2(scope:Scope container :string  , sharedName :string  , useNodeNameSharing :bool  , keyDtype :type  , valueDtype :type  )  ->(tableHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("container":container)
    attr.append("shared_name":sharedName)
    attr.append("use_node_name_sharing":useNodeNameSharing)
    attr.append("key_dtype":keyDtype)
    attr.append("value_dtype":valueDtype)

    let opspec = tf.OpSpec(
        OpType: "MutableHashTableV2",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes numerical negative value element-wise.

I.e., \\(y = -x\\).

*/




func neg(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Neg",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Training via negative sampling.


*/




func negTrain(scope:Scope,wIn: Tensorflow_DataType, wOut: Tensorflow_DataType, examples: Tensorflow_DataType, labels: Tensorflow_DataType, lr: Tensorflow_DataType, vocabCount :[Int]  , numNegativeSamples :uInt8  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("vocab_count":vocabCount)
    attr.append("num_negative_samples":numNegativeSamples)

    let opspec = tf.OpSpec(
        OpType: "NegTrain",
        Input: [ wIn, wOut, examples, labels, lr],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Makes its input available to the next iteration.


*/




func nextIteration(scope:Scope,data: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "NextIteration",
        Input: [ data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Does nothing. Only useful as a placeholder for control edges.


*/




func noOp(scope:Scope )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "NoOp",
        Input: [ ],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
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

*/




func nonMaxSuppression(scope:Scope,boxes: Tensorflow_DataType, scores: Tensorflow_DataType, maxOutputSize: Tensorflow_DataType, iouThreshold :float  )  ->(selectedIndices: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("iou_threshold":iouThreshold)

    let opspec = tf.OpSpec(
        OpType: "NonMaxSuppression",
        Input: [ boxes, scores, maxOutputSize],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func nonMaxSuppressionV2(scope:Scope,boxes: Tensorflow_DataType, scores: Tensorflow_DataType, maxOutputSize: Tensorflow_DataType, iouThreshold: Tensorflow_DataType )  ->(selectedIndices: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "NonMaxSuppressionV2",
        Input: [ boxes, scores, maxOutputSize, iouThreshold],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the truth value of (x != y) element-wise.

*NOTE*: `NotEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func notEqual(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "NotEqual",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func oneHot(scope:Scope,indices: Tensorflow_DataType, depth: Tensorflow_DataType, onValue: Tensorflow_DataType, offValue: Tensorflow_DataType, axis :uInt8  , t :type  , ti :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("axis":axis)
    attr.append("T":t)
    attr.append("TI":ti)

    let opspec = tf.OpSpec(
        OpType: "OneHot",
        Input: [ indices, depth, onValue, offValue],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func oneShotIterator(scope:Scope datasetFactory :func  , outputTypes :list(type)  , outputShapes :[Shape]  , container :string  , sharedName :string  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dataset_factory":datasetFactory)
    attr.append("output_types":outputTypes)
    attr.append("output_shapes":outputShapes)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "OneShotIterator",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns a tensor of ones with the same shape and type as x.


*/




func onesLike(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "OnesLike",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func pack(scope:Scope,values: Tensorflow_DataType, n :uInt8  , t :type  , axis :uInt8  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("N":n)
    attr.append("T":t)
    attr.append("axis":axis)

    let opspec = tf.OpSpec(
        OpType: "Pack",
        Input: [ values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func pad(scope:Scope,input: Tensorflow_DataType, paddings: Tensorflow_DataType, t :type  , tpaddings :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tpaddings":tpaddings)

    let opspec = tf.OpSpec(
        OpType: "Pad",
        Input: [ input, paddings],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a dataset that batches and pads `batch_size` elements from the input.


*/




func paddedBatchDataset(scope:Scope,inputDataset: Tensorflow_DataType, batchSize: Tensorflow_DataType, paddedShapes: Tensorflow_DataType, paddingValues: Tensorflow_DataType, toutputTypes :list(type)  , outputShapes :[Shape]  , n :uInt8  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Toutput_types":toutputTypes)
    attr.append("output_shapes":outputShapes)
    attr.append("N":n)

    let opspec = tf.OpSpec(
        OpType: "PaddedBatchDataset",
        Input: [ inputDataset, batchSize, paddedShapes, paddingValues],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A queue that produces elements in first-in first-out order.

Variable-size shapes are allowed by setting the corresponding shape dimensions
to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
size of any given element in the minibatch.  See below for details.

*/




func paddingFIFOQueue(scope:Scope componentTypes :list(type)  , shapes :[Shape]  , capacity :uInt8  , container :string  , sharedName :string  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("component_types":componentTypes)
    attr.append("shapes":shapes)
    attr.append("capacity":capacity)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "PaddingFIFOQueue",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A queue that produces elements in first-in first-out order.

Variable-size shapes are allowed by setting the corresponding shape dimensions
to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
size of any given element in the minibatch.  See below for details.

*/




func paddingFIFOQueueV2(scope:Scope componentTypes :list(type)  , shapes :[Shape]  , capacity :uInt8  , container :string  , sharedName :string  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("component_types":componentTypes)
    attr.append("shapes":shapes)
    attr.append("capacity":capacity)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "PaddingFIFOQueueV2",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func parallelConcat(scope:Scope,values: Tensorflow_DataType, n :uInt8  , t :type  , shape :shape  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("N":n)
    attr.append("T":t)
    attr.append("shape":shape)

    let opspec = tf.OpSpec(
        OpType: "ParallelConcat",
        Input: [ values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a dataset that applies `f` to the outputs of `input_dataset`.

Unlike a "MapDataset", which applies `f` sequentially, this dataset uses
up to `num_threads` threads to process elements from `input_dataset`
in parallel.

*/




func parallelMapDataset(scope:Scope,inputDataset: Tensorflow_DataType, otherArguments: Tensorflow_DataType, numThreads: Tensorflow_DataType, outputBufferSize: Tensorflow_DataType, f :func  , targuments :list(type)  , outputTypes :list(type)  , outputShapes :[Shape]  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("f":f)
    attr.append("Targuments":targuments)
    attr.append("output_types":outputTypes)
    attr.append("output_shapes":outputShapes)

    let opspec = tf.OpSpec(
        OpType: "ParallelMapDataset",
        Input: [ inputDataset, otherArguments, numThreads, outputBufferSize],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Outputs random values from a normal distribution. The parameters may each be a

scalar which applies to the entire output, or a vector of length shape[0] which
stores the parameters for each batch.

*/




func parameterizedTruncatedNormal(scope:Scope,shape: Tensorflow_DataType, means: Tensorflow_DataType, stdevs: Tensorflow_DataType, minvals: Tensorflow_DataType, maxvals: Tensorflow_DataType, seed :uInt8  , seed2 :uInt8  , dtype :type  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("seed":seed)
    attr.append("seed2":seed2)
    attr.append("dtype":dtype)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "ParameterizedTruncatedNormal",
        Input: [ shape, means, stdevs, minvals, maxvals],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Transforms a vector of brain.Example protos (as strings) into typed tensors.


*/




func parseExample(scope:Scope,serialized: Tensorflow_DataType, names: Tensorflow_DataType, sparseKeys: Tensorflow_DataType, denseKeys: Tensorflow_DataType, denseDefaults: Tensorflow_DataType, nsparse :uInt8  , ndense :uInt8  , sparseTypes :list(type)  , tdense :list(type)  , denseShapes :[Shape]  )  ->(sparseIndices: tf.Output, sparseValues: tf.Output, sparseShapes: tf.Output, denseValues: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Nsparse":nsparse)
    attr.append("Ndense":ndense)
    attr.append("sparse_types":sparseTypes)
    attr.append("Tdense":tdense)
    attr.append("dense_shapes":denseShapes)

    let opspec = tf.OpSpec(
        OpType: "ParseExample",
        Input: [ serialized, names, sparseKeys, denseKeys, denseDefaults],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1),op.Output(4 - 1))
}

/*
Transforms a scalar brain.SequenceExample proto (as strings) into typed tensors.


*/




func parseSingleSequenceExample(scope:Scope,serialized: Tensorflow_DataType, featureListDenseMissingAssumedEmpty: Tensorflow_DataType, contextSparseKeys: Tensorflow_DataType, contextDenseKeys: Tensorflow_DataType, featureListSparseKeys: Tensorflow_DataType, featureListDenseKeys: Tensorflow_DataType, contextDenseDefaults: Tensorflow_DataType, debugName: Tensorflow_DataType, ncontextSparse :uInt8  , ncontextDense :uInt8  , nfeatureListSparse :uInt8  , nfeatureListDense :uInt8  , contextSparseTypes :list(type)  , tcontextDense :list(type)  , featureListDenseTypes :list(type)  , contextDenseShapes :[Shape]  , featureListSparseTypes :list(type)  , featureListDenseShapes :[Shape]  )  ->(contextSparseIndices: tf.Output, contextSparseValues: tf.Output, contextSparseShapes: tf.Output, contextDenseValues: tf.Output, featureListSparseIndices: tf.Output, featureListSparseValues: tf.Output, featureListSparseShapes: tf.Output, featureListDenseValues: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Ncontext_sparse":ncontextSparse)
    attr.append("Ncontext_dense":ncontextDense)
    attr.append("Nfeature_list_sparse":nfeatureListSparse)
    attr.append("Nfeature_list_dense":nfeatureListDense)
    attr.append("context_sparse_types":contextSparseTypes)
    attr.append("Tcontext_dense":tcontextDense)
    attr.append("feature_list_dense_types":featureListDenseTypes)
    attr.append("context_dense_shapes":contextDenseShapes)
    attr.append("feature_list_sparse_types":featureListSparseTypes)
    attr.append("feature_list_dense_shapes":featureListDenseShapes)

    let opspec = tf.OpSpec(
        OpType: "ParseSingleSequenceExample",
        Input: [ serialized, featureListDenseMissingAssumedEmpty, contextSparseKeys, contextDenseKeys, featureListSparseKeys, featureListDenseKeys, contextDenseDefaults, debugName],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1),op.Output(4 - 1),op.Output(5 - 1),op.Output(6 - 1),op.Output(7 - 1),op.Output(8 - 1))
}

/*
Transforms a serialized tensorflow.TensorProto proto into a Tensor.


*/




func parseTensor(scope:Scope,serialized: Tensorflow_DataType, outType :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("out_type":outType)

    let opspec = tf.OpSpec(
        OpType: "ParseTensor",
        Input: [ serialized],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A placeholder op for a value that will be fed into the computation.

N.B. This operation will fail with an error if it is executed. It is
intended as a way to represent a value that will always be fed, and to
provide attrs that enable the fed value to be checked at runtime.

*/




func placeholder(scope:Scope dtype :type  , shape :shape  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("shape":shape)

    let opspec = tf.OpSpec(
        OpType: "Placeholder",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A placeholder op for a value that will be fed into the computation.

N.B. This operation will fail with an error if it is executed. It is
intended as a way to represent a value that will always be fed, and to
provide attrs that enable the fed value to be checked at runtime.

*/




func placeholderV2(scope:Scope dtype :type  , shape :shape  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("shape":shape)

    let opspec = tf.OpSpec(
        OpType: "PlaceholderV2",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A placeholder op that passes through `input` when its output is not fed.


*/




func placeholderWithDefault(scope:Scope,input: Tensorflow_DataType, dtype :type  , shape :shape  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("shape":shape)

    let opspec = tf.OpSpec(
        OpType: "PlaceholderWithDefault",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Compute the polygamma function \\(\psi^{(n)}(x)\\).

The polygamma function is defined as:
\\(\psi^{(n)}(x) = \frac{d^n}{dx^n} \psi(x)\\)
where \\(\psi(x)\\) is the digamma function.

*/




func polygamma(scope:Scope,a: Tensorflow_DataType, x: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Polygamma",
        Input: [ a, x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func pow(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Pow",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
An identity op that triggers an error if a gradient is requested.

When executed in a graph, this op outputs its input tensor as-is.
When building ops to compute gradients, the TensorFlow gradient system
will return an error when trying to lookup the gradient of this op,
because no gradient must ever be registered for this function.  This
op exists to prevent subtle bugs from silently returning unimplemented
gradients in some corner cases.

*/




func preventGradient(scope:Scope,input: Tensorflow_DataType, t :type  , message :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("message":message)

    let opspec = tf.OpSpec(
        OpType: "PreventGradient",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Prints a list of tensors.

Passes `input` through to `output` and prints `data` when evaluating.

*/




func print(scope:Scope,input: Tensorflow_DataType, data: Tensorflow_DataType, t :type  , u :list(type)  , message :string  , firstN :uInt8  , summarize :uInt8  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("U":u)
    attr.append("message":message)
    attr.append("first_n":firstN)
    attr.append("summarize":summarize)

    let opspec = tf.OpSpec(
        OpType: "Print",
        Input: [ input, data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A queue that produces elements sorted by the first component value.

Note that the PriorityQueue requires the first component of any element
to be a scalar int64, in addition to the other elements declared by
component_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
entry in their input (resp. output) lists.

*/




func priorityQueue(scope:Scope componentTypes :list(type)  , shapes :[Shape]  , capacity :uInt8  , container :string  , sharedName :string  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("component_types":componentTypes)
    attr.append("shapes":shapes)
    attr.append("capacity":capacity)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "PriorityQueue",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A queue that produces elements sorted by the first component value.

Note that the PriorityQueue requires the first component of any element
to be a scalar int64, in addition to the other elements declared by
component_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
entry in their input (resp. output) lists.

*/




func priorityQueueV2(scope:Scope componentTypes :list(type)  , shapes :[Shape]  , capacity :uInt8  , container :string  , sharedName :string  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("component_types":componentTypes)
    attr.append("shapes":shapes)
    attr.append("capacity":capacity)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "PriorityQueueV2",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the product of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

*/




func prod(scope:Scope,input: Tensorflow_DataType, reductionIndices: Tensorflow_DataType, keepDims :bool  , t :type  , tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("keep_dims":keepDims)
    attr.append("T":t)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "Prod",
        Input: [ input, reductionIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Invokes a python function to compute func(input)->output.

This operation is considered stateful. For a stateless version, see
PyFuncStateless.

*/




func pyFunc(scope:Scope,input: Tensorflow_DataType, token :string  , tin :list(type)  , tout :list(type)  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("token":token)
    attr.append("Tin":tin)
    attr.append("Tout":tout)

    let opspec = tf.OpSpec(
        OpType: "PyFunc",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A stateless version of PyFunc.


*/




func pyFuncStateless(scope:Scope,input: Tensorflow_DataType, token :string  , tin :list(type)  , tout :list(type)  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("token":token)
    attr.append("Tin":tin)
    attr.append("Tout":tout)

    let opspec = tf.OpSpec(
        OpType: "PyFuncStateless",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func qr(scope:Scope,input: Tensorflow_DataType, fullMatrices :bool  , t :type  )  ->(q: tf.Output, r: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("full_matrices":fullMatrices)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Qr",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Use QuantizeAndDequantizeV2 instead.


*/




func quantizeAndDequantize(scope:Scope,input: Tensorflow_DataType, signedInput :bool  , numBits :uInt8  , rangeGiven :bool  , inputMin :float  , inputMax :float  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("signed_input":signedInput)
    attr.append("num_bits":numBits)
    attr.append("range_given":rangeGiven)
    attr.append("input_min":inputMin)
    attr.append("input_max":inputMax)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "QuantizeAndDequantize",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func quantizeAndDequantizeV2(scope:Scope,input: Tensorflow_DataType, inputMin: Tensorflow_DataType, inputMax: Tensorflow_DataType, signedInput :bool  , numBits :uInt8  , rangeGiven :bool  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("signed_input":signedInput)
    attr.append("num_bits":numBits)
    attr.append("range_given":rangeGiven)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "QuantizeAndDequantizeV2",
        Input: [ input, inputMin, inputMax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func quantizeDownAndShrinkRange(scope:Scope,input: Tensorflow_DataType, inputMin: Tensorflow_DataType, inputMax: Tensorflow_DataType, tinput :type  , outType :type  )  ->(output: tf.Output, outputMin: tf.Output, outputMax: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tinput":tinput)
    attr.append("out_type":outType)

    let opspec = tf.OpSpec(
        OpType: "QuantizeDownAndShrinkRange",
        Input: [ input, inputMin, inputMax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
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

*/




func quantizeV2(scope:Scope,input: Tensorflow_DataType, minRange: Tensorflow_DataType, maxRange: Tensorflow_DataType, t :type  , mode :string  )  ->(output: tf.Output, outputMin: tf.Output, outputMax: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("mode":mode)

    let opspec = tf.OpSpec(
        OpType: "QuantizeV2",
        Input: [ input, minRange, maxRange],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Returns x + y element-wise, working on quantized buffers.


*/




func quantizedAdd(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, minX: Tensorflow_DataType, maxX: Tensorflow_DataType, minY: Tensorflow_DataType, maxY: Tensorflow_DataType, t1 :type  , t2 :type  , toutput :type  )  ->(z: tf.Output, minZ: tf.Output, maxZ: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T1":t1)
    attr.append("T2":t2)
    attr.append("Toutput":toutput)

    let opspec = tf.OpSpec(
        OpType: "QuantizedAdd",
        Input: [ x, y, minX, maxX, minY, maxY],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Produces the average pool of the input tensor for quantized types.


*/




func quantizedAvgPool(scope:Scope,input: Tensorflow_DataType, minInput: Tensorflow_DataType, maxInput: Tensorflow_DataType, t :type  , ksize :[Int]  , strides :[Int]  , padding :string  )  ->(output: tf.Output, minOutput: tf.Output, maxOutput: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("ksize":ksize)
    attr.append("strides":strides)
    attr.append("padding":padding)

    let opspec = tf.OpSpec(
        OpType: "QuantizedAvgPool",
        Input: [ input, minInput, maxInput],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Quantized Batch normalization.

This op is deprecated and will be removed in the future. Prefer
`tf.nn.batch_normalization`.

*/




func quantizedBatchNormWithGlobalNormalization(scope:Scope,t: Tensorflow_DataType, tMin: Tensorflow_DataType, tMax: Tensorflow_DataType, m: Tensorflow_DataType, mMin: Tensorflow_DataType, mMax: Tensorflow_DataType, v: Tensorflow_DataType, vMin: Tensorflow_DataType, vMax: Tensorflow_DataType, beta: Tensorflow_DataType, betaMin: Tensorflow_DataType, betaMax: Tensorflow_DataType, gamma: Tensorflow_DataType, gammaMin: Tensorflow_DataType, gammaMax: Tensorflow_DataType, tinput :type  , outType :type  , varianceEpsilon :float  , scaleAfterNormalization :bool  )  ->(result: tf.Output, resultMin: tf.Output, resultMax: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tinput":tinput)
    attr.append("out_type":outType)
    attr.append("variance_epsilon":varianceEpsilon)
    attr.append("scale_after_normalization":scaleAfterNormalization)

    let opspec = tf.OpSpec(
        OpType: "QuantizedBatchNormWithGlobalNormalization",
        Input: [ t, tMin, tMax, m, mMin, mMax, v, vMin, vMax, beta, betaMin, betaMax, gamma, gammaMin, gammaMax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Adds Tensor 'bias' to Tensor 'input' for Quantized types.

Broadcasts the values of bias on dimensions 0..N-2 of 'input'.

*/




func quantizedBiasAdd(scope:Scope,input: Tensorflow_DataType, bias: Tensorflow_DataType, minInput: Tensorflow_DataType, maxInput: Tensorflow_DataType, minBias: Tensorflow_DataType, maxBias: Tensorflow_DataType, t1 :type  , t2 :type  , outType :type  )  ->(output: tf.Output, minOut: tf.Output, maxOut: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T1":t1)
    attr.append("T2":t2)
    attr.append("out_type":outType)

    let opspec = tf.OpSpec(
        OpType: "QuantizedBiasAdd",
        Input: [ input, bias, minInput, maxInput, minBias, maxBias],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Concatenates quantized tensors along one dimension.


*/




func quantizedConcat(scope:Scope,concatDim: Tensorflow_DataType, values: Tensorflow_DataType, inputMins: Tensorflow_DataType, inputMaxes: Tensorflow_DataType, n :uInt8  , t :type  )  ->(output: tf.Output, outputMin: tf.Output, outputMax: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("N":n)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "QuantizedConcat",
        Input: [ concatDim, values, inputMins, inputMaxes],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Computes a 2D convolution given quantized 4D input and filter tensors.

The inputs are quantized tensors where the lowest value represents the real
number of the associated minimum, and the highest represents the maximum.
This means that you can only interpret the quantized output in the same way, by
taking the returned minimum and maximum values into account.

*/




func quantizedConv2D(scope:Scope,input: Tensorflow_DataType, filter: Tensorflow_DataType, minInput: Tensorflow_DataType, maxInput: Tensorflow_DataType, minFilter: Tensorflow_DataType, maxFilter: Tensorflow_DataType, tinput :type  , tfilter :type  , outType :type  , strides :[Int]  , padding :string  )  ->(output: tf.Output, minOutput: tf.Output, maxOutput: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tinput":tinput)
    attr.append("Tfilter":tfilter)
    attr.append("out_type":outType)
    attr.append("strides":strides)
    attr.append("padding":padding)

    let opspec = tf.OpSpec(
        OpType: "QuantizedConv2D",
        Input: [ input, filter, minInput, maxInput, minFilter, maxFilter],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Quantized Instance normalization.


*/




func quantizedInstanceNorm(scope:Scope,x: Tensorflow_DataType, xMin: Tensorflow_DataType, xMax: Tensorflow_DataType, t :type  , outputRangeGiven :bool  , givenYMin :float  , givenYMax :float  , varianceEpsilon :float  , minSeparation :float  )  ->(y: tf.Output, yMin: tf.Output, yMax: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("output_range_given":outputRangeGiven)
    attr.append("given_y_min":givenYMin)
    attr.append("given_y_max":givenYMax)
    attr.append("variance_epsilon":varianceEpsilon)
    attr.append("min_separation":minSeparation)

    let opspec = tf.OpSpec(
        OpType: "QuantizedInstanceNorm",
        Input: [ x, xMin, xMax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Perform a quantized matrix multiplication of  `a` by the matrix `b`.

The inputs must be two-dimensional matrices and the inner dimension of
`a` (after being transposed if `transpose_a` is non-zero) must match the
outer dimension of `b` (after being transposed if `transposed_b` is
non-zero).

*/




func quantizedMatMul(scope:Scope,a: Tensorflow_DataType, b: Tensorflow_DataType, minA: Tensorflow_DataType, maxA: Tensorflow_DataType, minB: Tensorflow_DataType, maxB: Tensorflow_DataType, t1 :type  , t2 :type  , toutput :type  , transposeA :bool  , transposeB :bool  , tactivation :type  )  ->(out: tf.Output, minOut: tf.Output, maxOut: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T1":t1)
    attr.append("T2":t2)
    attr.append("Toutput":toutput)
    attr.append("transpose_a":transposeA)
    attr.append("transpose_b":transposeB)
    attr.append("Tactivation":tactivation)

    let opspec = tf.OpSpec(
        OpType: "QuantizedMatMul",
        Input: [ a, b, minA, maxA, minB, maxB],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Produces the max pool of the input tensor for quantized types.


*/




func quantizedMaxPool(scope:Scope,input: Tensorflow_DataType, minInput: Tensorflow_DataType, maxInput: Tensorflow_DataType, t :type  , ksize :[Int]  , strides :[Int]  , padding :string  )  ->(output: tf.Output, minOutput: tf.Output, maxOutput: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("ksize":ksize)
    attr.append("strides":strides)
    attr.append("padding":padding)

    let opspec = tf.OpSpec(
        OpType: "QuantizedMaxPool",
        Input: [ input, minInput, maxInput],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Returns x * y element-wise, working on quantized buffers.


*/




func quantizedMul(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, minX: Tensorflow_DataType, maxX: Tensorflow_DataType, minY: Tensorflow_DataType, maxY: Tensorflow_DataType, t1 :type  , t2 :type  , toutput :type  )  ->(z: tf.Output, minZ: tf.Output, maxZ: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T1":t1)
    attr.append("T2":t2)
    attr.append("Toutput":toutput)

    let opspec = tf.OpSpec(
        OpType: "QuantizedMul",
        Input: [ x, y, minX, maxX, minY, maxY],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Computes Quantized Rectified Linear: `max(features, 0)`


*/




func quantizedRelu(scope:Scope,features: Tensorflow_DataType, minFeatures: Tensorflow_DataType, maxFeatures: Tensorflow_DataType, tinput :type  , outType :type  )  ->(activations: tf.Output, minActivations: tf.Output, maxActivations: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tinput":tinput)
    attr.append("out_type":outType)

    let opspec = tf.OpSpec(
        OpType: "QuantizedRelu",
        Input: [ features, minFeatures, maxFeatures],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Computes Quantized Rectified Linear 6: `min(max(features, 0), 6)`


*/




func quantizedRelu6(scope:Scope,features: Tensorflow_DataType, minFeatures: Tensorflow_DataType, maxFeatures: Tensorflow_DataType, tinput :type  , outType :type  )  ->(activations: tf.Output, minActivations: tf.Output, maxActivations: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tinput":tinput)
    attr.append("out_type":outType)

    let opspec = tf.OpSpec(
        OpType: "QuantizedRelu6",
        Input: [ features, minFeatures, maxFeatures],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`


*/




func quantizedReluX(scope:Scope,features: Tensorflow_DataType, maxValue: Tensorflow_DataType, minFeatures: Tensorflow_DataType, maxFeatures: Tensorflow_DataType, tinput :type  , outType :type  )  ->(activations: tf.Output, minActivations: tf.Output, maxActivations: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tinput":tinput)
    attr.append("out_type":outType)

    let opspec = tf.OpSpec(
        OpType: "QuantizedReluX",
        Input: [ features, maxValue, minFeatures, maxFeatures],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Reshapes a quantized tensor as per the Reshape op.

```

*/




func quantizedReshape(scope:Scope,tensor: Tensorflow_DataType, shape: Tensorflow_DataType, inputMin: Tensorflow_DataType, inputMax: Tensorflow_DataType, t :type  , tshape :type  )  ->(output: tf.Output, outputMin: tf.Output, outputMax: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tshape":tshape)

    let opspec = tf.OpSpec(
        OpType: "QuantizedReshape",
        Input: [ tensor, shape, inputMin, inputMax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Closes the given queue.

This operation signals that no more elements will be enqueued in the
given queue. Subsequent Enqueue(Many) operations will fail.
Subsequent Dequeue(Many) operations will continue to succeed if
sufficient elements remain in the queue. Subsequent Dequeue(Many)
operations that would block will fail immediately.

*/




func queueClose(scope:Scope,handle: Tensorflow_DataType, cancelPendingEnqueues :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("cancel_pending_enqueues":cancelPendingEnqueues)

    let opspec = tf.OpSpec(
        OpType: "QueueClose",
        Input: [ handle],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Closes the given queue.

This operation signals that no more elements will be enqueued in the
given queue. Subsequent Enqueue(Many) operations will fail.
Subsequent Dequeue(Many) operations will continue to succeed if
sufficient elements remain in the queue. Subsequent Dequeue(Many)
operations that would block will fail immediately.

*/




func queueCloseV2(scope:Scope,handle: Tensorflow_DataType, cancelPendingEnqueues :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("cancel_pending_enqueues":cancelPendingEnqueues)

    let opspec = tf.OpSpec(
        OpType: "QueueCloseV2",
        Input: [ handle],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Dequeues a tuple of one or more tensors from the given queue.

This operation has k outputs, where k is the number of components
in the tuples stored in the given queue, and output i is the ith
component of the dequeued tuple.
N.B. If the queue is empty, this operation will block until an element
has been dequeued (or 'timeout_ms' elapses, if specified).

*/




func queueDequeue(scope:Scope,handle: Tensorflow_DataType, componentTypes :list(type)  , timeoutMs :uInt8  )  ->(components: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("component_types":componentTypes)
    attr.append("timeout_ms":timeoutMs)

    let opspec = tf.OpSpec(
        OpType: "QueueDequeue",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func queueDequeueMany(scope:Scope,handle: Tensorflow_DataType, n: Tensorflow_DataType, componentTypes :list(type)  , timeoutMs :uInt8  )  ->(components: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("component_types":componentTypes)
    attr.append("timeout_ms":timeoutMs)

    let opspec = tf.OpSpec(
        OpType: "QueueDequeueMany",
        Input: [ handle, n],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func queueDequeueManyV2(scope:Scope,handle: Tensorflow_DataType, n: Tensorflow_DataType, componentTypes :list(type)  , timeoutMs :uInt8  )  ->(components: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("component_types":componentTypes)
    attr.append("timeout_ms":timeoutMs)

    let opspec = tf.OpSpec(
        OpType: "QueueDequeueManyV2",
        Input: [ handle, n],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func queueDequeueUpTo(scope:Scope,handle: Tensorflow_DataType, n: Tensorflow_DataType, componentTypes :list(type)  , timeoutMs :uInt8  )  ->(components: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("component_types":componentTypes)
    attr.append("timeout_ms":timeoutMs)

    let opspec = tf.OpSpec(
        OpType: "QueueDequeueUpTo",
        Input: [ handle, n],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func queueDequeueUpToV2(scope:Scope,handle: Tensorflow_DataType, n: Tensorflow_DataType, componentTypes :list(type)  , timeoutMs :uInt8  )  ->(components: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("component_types":componentTypes)
    attr.append("timeout_ms":timeoutMs)

    let opspec = tf.OpSpec(
        OpType: "QueueDequeueUpToV2",
        Input: [ handle, n],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Dequeues a tuple of one or more tensors from the given queue.

This operation has k outputs, where k is the number of components
in the tuples stored in the given queue, and output i is the ith
component of the dequeued tuple.
N.B. If the queue is empty, this operation will block until an element
has been dequeued (or 'timeout_ms' elapses, if specified).

*/




func queueDequeueV2(scope:Scope,handle: Tensorflow_DataType, componentTypes :list(type)  , timeoutMs :uInt8  )  ->(components: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("component_types":componentTypes)
    attr.append("timeout_ms":timeoutMs)

    let opspec = tf.OpSpec(
        OpType: "QueueDequeueV2",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Enqueues a tuple of one or more tensors in the given queue.

The components input has k elements, which correspond to the components of
tuples stored in the given queue.
N.B. If the queue is full, this operation will block until the given
element has been enqueued (or 'timeout_ms' elapses, if specified).

*/




func queueEnqueue(scope:Scope,handle: Tensorflow_DataType, components: Tensorflow_DataType, tcomponents :list(type)  , timeoutMs :uInt8  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tcomponents":tcomponents)
    attr.append("timeout_ms":timeoutMs)

    let opspec = tf.OpSpec(
        OpType: "QueueEnqueue",
        Input: [ handle, components],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
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

*/




func queueEnqueueMany(scope:Scope,handle: Tensorflow_DataType, components: Tensorflow_DataType, tcomponents :list(type)  , timeoutMs :uInt8  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tcomponents":tcomponents)
    attr.append("timeout_ms":timeoutMs)

    let opspec = tf.OpSpec(
        OpType: "QueueEnqueueMany",
        Input: [ handle, components],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
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

*/




func queueEnqueueManyV2(scope:Scope,handle: Tensorflow_DataType, components: Tensorflow_DataType, tcomponents :list(type)  , timeoutMs :uInt8  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tcomponents":tcomponents)
    attr.append("timeout_ms":timeoutMs)

    let opspec = tf.OpSpec(
        OpType: "QueueEnqueueManyV2",
        Input: [ handle, components],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Enqueues a tuple of one or more tensors in the given queue.

The components input has k elements, which correspond to the components of
tuples stored in the given queue.
N.B. If the queue is full, this operation will block until the given
element has been enqueued (or 'timeout_ms' elapses, if specified).

*/




func queueEnqueueV2(scope:Scope,handle: Tensorflow_DataType, components: Tensorflow_DataType, tcomponents :list(type)  , timeoutMs :uInt8  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tcomponents":tcomponents)
    attr.append("timeout_ms":timeoutMs)

    let opspec = tf.OpSpec(
        OpType: "QueueEnqueueV2",
        Input: [ handle, components],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Computes the number of elements in the given queue.


*/




func queueSize(scope:Scope,handle: Tensorflow_DataType )  ->(size: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "QueueSize",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the number of elements in the given queue.


*/




func queueSizeV2(scope:Scope,handle: Tensorflow_DataType )  ->(size: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "QueueSizeV2",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Real-valued fast Fourier transform.

Computes the 1-dimensional discrete Fourier transform of a real-valued signal
over the inner-most dimension of `input`.
Since the DFT of a real signal is Hermitian-symmetric, `RFFT` only returns the
`fft_length / 2 + 1` unique components of the FFT: the zero-frequency term,
followed by the `fft_length / 2` positive-frequency terms.

*/




func rfft(scope:Scope,input: Tensorflow_DataType, fftLength: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "RFFT",
        Input: [ input, fftLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
2D real-valued fast Fourier transform.

Computes the 2-dimensional discrete Fourier transform of a real-valued signal
over the inner-most 2 dimensions of `input`.
Since the DFT of a real signal is Hermitian-symmetric, `RFFT2D` only returns the
`fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
of `output`: the zero-frequency term, followed by the `fft_length / 2`
positive-frequency terms.

*/




func rfft2D(scope:Scope,input: Tensorflow_DataType, fftLength: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "RFFT2D",
        Input: [ input, fftLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
3D real-valued fast Fourier transform.

Computes the 3-dimensional discrete Fourier transform of a real-valued signal
over the inner-most 3 dimensions of `input`.
Since the DFT of a real signal is Hermitian-symmetric, `RFFT3D` only returns the
`fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
of `output`: the zero-frequency term, followed by the `fft_length / 2`
positive-frequency terms.

*/




func rfft3D(scope:Scope,input: Tensorflow_DataType, fftLength: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "RFFT3D",
        Input: [ input, fftLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Converts one or more images from RGB to HSV.

Outputs a tensor of the same shape as the `images` tensor, containing the HSV
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.
`output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
`output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.

*/




func rgbToHSV(scope:Scope,images: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "RGBToHSV",
        Input: [ images],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Randomly crop `image`.

`size` is a 1-D int64 tensor with 2 elements representing the crop height and
width.  The values must be non negative.
This Op picks a random location in `image` and crops a `height` by `width`
rectangle from that location.  The random location is picked so the cropped
area will fit inside the original image.

*/




func randomCrop(scope:Scope,image: Tensorflow_DataType, size: Tensorflow_DataType, t :type  , seed :uInt8  , seed2 :uInt8  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("seed":seed)
    attr.append("seed2":seed2)

    let opspec = tf.OpSpec(
        OpType: "RandomCrop",
        Input: [ image, size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Outputs random values from the Gamma distribution(s) described by alpha.

This op uses the algorithm by Marsaglia et al. to acquire samples via
transformation-rejection from pairs of uniform and normal random variables.
See http://dl.acm.org/citation.cfm?id=358414

*/




func randomGamma(scope:Scope,shape: Tensorflow_DataType, alpha: Tensorflow_DataType, seed :uInt8  , seed2 :uInt8  , s :type  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("seed":seed)
    attr.append("seed2":seed2)
    attr.append("S":s)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "RandomGamma",
        Input: [ shape, alpha],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func randomPoisson(scope:Scope,shape: Tensorflow_DataType, rate: Tensorflow_DataType, seed :uInt8  , seed2 :uInt8  , s :type  , dtype :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("seed":seed)
    attr.append("seed2":seed2)
    attr.append("S":s)
    attr.append("dtype":dtype)

    let opspec = tf.OpSpec(
        OpType: "RandomPoisson",
        Input: [ shape, rate],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func randomShuffle(scope:Scope,value: Tensorflow_DataType, seed :uInt8  , seed2 :uInt8  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("seed":seed)
    attr.append("seed2":seed2)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "RandomShuffle",
        Input: [ value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A queue that randomizes the order of elements.


*/




func randomShuffleQueue(scope:Scope componentTypes :list(type)  , shapes :[Shape]  , capacity :uInt8  , minAfterDequeue :uInt8  , seed :uInt8  , seed2 :uInt8  , container :string  , sharedName :string  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("component_types":componentTypes)
    attr.append("shapes":shapes)
    attr.append("capacity":capacity)
    attr.append("min_after_dequeue":minAfterDequeue)
    attr.append("seed":seed)
    attr.append("seed2":seed2)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "RandomShuffleQueue",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A queue that randomizes the order of elements.


*/




func randomShuffleQueueV2(scope:Scope componentTypes :list(type)  , shapes :[Shape]  , capacity :uInt8  , minAfterDequeue :uInt8  , seed :uInt8  , seed2 :uInt8  , container :string  , sharedName :string  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("component_types":componentTypes)
    attr.append("shapes":shapes)
    attr.append("capacity":capacity)
    attr.append("min_after_dequeue":minAfterDequeue)
    attr.append("seed":seed)
    attr.append("seed2":seed2)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "RandomShuffleQueueV2",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Outputs random values from a normal distribution.

The generated values will have mean 0 and standard deviation 1.

*/




func randomStandardNormal(scope:Scope,shape: Tensorflow_DataType, seed :uInt8  , seed2 :uInt8  , dtype :type  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("seed":seed)
    attr.append("seed2":seed2)
    attr.append("dtype":dtype)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "RandomStandardNormal",
        Input: [ shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Outputs random values from a uniform distribution.

The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.

*/




func randomUniform(scope:Scope,shape: Tensorflow_DataType, seed :uInt8  , seed2 :uInt8  , dtype :type  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("seed":seed)
    attr.append("seed2":seed2)
    attr.append("dtype":dtype)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "RandomUniform",
        Input: [ shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Outputs random integers from a uniform distribution.

The generated values are uniform integers in the range `[minval, maxval)`.
The lower bound `minval` is included in the range, while the upper bound
`maxval` is excluded.
The random integers are slightly biased unless `maxval - minval` is an exact
power of two.  The bias is small for values of `maxval - minval` significantly
smaller than the range of the output (either `2^32` or `2^64`).

*/




func randomUniformInt(scope:Scope,shape: Tensorflow_DataType, minval: Tensorflow_DataType, maxval: Tensorflow_DataType, seed :uInt8  , seed2 :uInt8  , tout :type  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("seed":seed)
    attr.append("seed2":seed2)
    attr.append("Tout":tout)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "RandomUniformInt",
        Input: [ shape, minval, maxval],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func range(scope:Scope,start: Tensorflow_DataType, limit: Tensorflow_DataType, delta: Tensorflow_DataType, tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "Range",
        Input: [ start, limit, delta],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a dataset with a range of values. Corresponds to python's xrange.


*/




func rangeDataset(scope:Scope,start: Tensorflow_DataType, stop: Tensorflow_DataType, step: Tensorflow_DataType, outputTypes :list(type)  , outputShapes :[Shape]  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("output_types":outputTypes)
    attr.append("output_shapes":outputShapes)

    let opspec = tf.OpSpec(
        OpType: "RangeDataset",
        Input: [ start, stop, step],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func rank(scope:Scope,input: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Rank",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Reads and outputs the entire contents of the input filename.


*/




func readFile(scope:Scope,filename: Tensorflow_DataType )  ->(contents: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ReadFile",
        Input: [ filename],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the number of records this Reader has produced.

This is the same as the number of ReaderRead executions that have
succeeded.

*/




func readerNumRecordsProduced(scope:Scope,readerHandle: Tensorflow_DataType )  ->(recordsProduced: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ReaderNumRecordsProduced",
        Input: [ readerHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the number of records this Reader has produced.

This is the same as the number of ReaderRead executions that have
succeeded.

*/




func readerNumRecordsProducedV2(scope:Scope,readerHandle: Tensorflow_DataType )  ->(recordsProduced: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ReaderNumRecordsProducedV2",
        Input: [ readerHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the number of work units this Reader has finished processing.


*/




func readerNumWorkUnitsCompleted(scope:Scope,readerHandle: Tensorflow_DataType )  ->(unitsCompleted: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ReaderNumWorkUnitsCompleted",
        Input: [ readerHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the number of work units this Reader has finished processing.


*/




func readerNumWorkUnitsCompletedV2(scope:Scope,readerHandle: Tensorflow_DataType )  ->(unitsCompleted: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ReaderNumWorkUnitsCompletedV2",
        Input: [ readerHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the next record (key, value pair) produced by a Reader.

Will dequeue from the input queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has finished
with the previous file).

*/




func readerRead(scope:Scope,readerHandle: Tensorflow_DataType, queueHandle: Tensorflow_DataType )  ->(key: tf.Output, value: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ReaderRead",
        Input: [ readerHandle, queueHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Returns up to `num_records` (key, value) pairs produced by a Reader.

Will dequeue from the input queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has finished
with the previous file).
It may return less than `num_records` even before the last batch.

*/




func readerReadUpTo(scope:Scope,readerHandle: Tensorflow_DataType, queueHandle: Tensorflow_DataType, numRecords: Tensorflow_DataType )  ->(keys: tf.Output, values: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ReaderReadUpTo",
        Input: [ readerHandle, queueHandle, numRecords],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Returns up to `num_records` (key, value) pairs produced by a Reader.

Will dequeue from the input queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has finished
with the previous file).
It may return less than `num_records` even before the last batch.

*/




func readerReadUpToV2(scope:Scope,readerHandle: Tensorflow_DataType, queueHandle: Tensorflow_DataType, numRecords: Tensorflow_DataType )  ->(keys: tf.Output, values: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ReaderReadUpToV2",
        Input: [ readerHandle, queueHandle, numRecords],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Returns the next record (key, value pair) produced by a Reader.

Will dequeue from the input queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has finished
with the previous file).

*/




func readerReadV2(scope:Scope,readerHandle: Tensorflow_DataType, queueHandle: Tensorflow_DataType )  ->(key: tf.Output, value: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ReaderReadV2",
        Input: [ readerHandle, queueHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Restore a Reader to its initial clean state.


*/




func readerReset(scope:Scope,readerHandle: Tensorflow_DataType )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ReaderReset",
        Input: [ readerHandle],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Restore a Reader to its initial clean state.


*/




func readerResetV2(scope:Scope,readerHandle: Tensorflow_DataType )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ReaderResetV2",
        Input: [ readerHandle],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Restore a reader to a previously saved state.

Not all Readers support being restored, so this can produce an
Unimplemented error.

*/




func readerRestoreState(scope:Scope,readerHandle: Tensorflow_DataType, state: Tensorflow_DataType )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ReaderRestoreState",
        Input: [ readerHandle, state],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Restore a reader to a previously saved state.

Not all Readers support being restored, so this can produce an
Unimplemented error.

*/




func readerRestoreStateV2(scope:Scope,readerHandle: Tensorflow_DataType, state: Tensorflow_DataType )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ReaderRestoreStateV2",
        Input: [ readerHandle, state],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Produce a string tensor that encodes the state of a Reader.

Not all Readers support being serialized, so this can produce an
Unimplemented error.

*/




func readerSerializeState(scope:Scope,readerHandle: Tensorflow_DataType )  ->(state: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ReaderSerializeState",
        Input: [ readerHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Produce a string tensor that encodes the state of a Reader.

Not all Readers support being serialized, so this can produce an
Unimplemented error.

*/




func readerSerializeStateV2(scope:Scope,readerHandle: Tensorflow_DataType )  ->(state: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ReaderSerializeStateV2",
        Input: [ readerHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func real(scope:Scope,input: Tensorflow_DataType, t :type  , tout :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tout":tout)

    let opspec = tf.OpSpec(
        OpType: "Real",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns x / y element-wise for real types.

If `x` and `y` are reals, this will return the floating-point division.
*NOTE*: `Div` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func realDiv(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "RealDiv",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the reciprocal of x element-wise.

I.e., \\(y = 1 / x\\).

*/




func reciprocal(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Reciprocal",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradient for the inverse of `x` wrt its input.

Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
is the corresponding input gradient.

*/




func reciprocalGrad(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "ReciprocalGrad",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Emits randomized records.


*/




func recordInput(scope:Scope filePattern :string  , fileRandomSeed :uInt8  , fileShuffleShiftRatio :float  , fileBufferSize :uInt8  , fileParallelism :uInt8  , batchSize :uInt8  )  ->(records: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("file_pattern":filePattern)
    attr.append("file_random_seed":fileRandomSeed)
    attr.append("file_shuffle_shift_ratio":fileShuffleShiftRatio)
    attr.append("file_buffer_size":fileBufferSize)
    attr.append("file_parallelism":fileParallelism)
    attr.append("batch_size":batchSize)

    let opspec = tf.OpSpec(
        OpType: "RecordInput",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func reduceJoin(scope:Scope,inputs: Tensorflow_DataType, reductionIndices: Tensorflow_DataType, keepDims :bool  , separator :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("keep_dims":keepDims)
    attr.append("separator":separator)

    let opspec = tf.OpSpec(
        OpType: "ReduceJoin",
        Input: [ inputs, reductionIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates or finds a child frame, and makes `data` available to the child frame.

The unique `frame_name` is used by the `Executor` to identify frames. If
`is_constant` is true, `output` is a constant in the child frame; otherwise
it may be changed in the child frame. At most `parallel_iterations` iterations
are run in parallel in the child frame.

*/




func refEnter(scope:Scope,data: Tensorflow_DataType, t :type  , frameName :string  , isConstant :bool  , parallelIterations :uInt8  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("frame_name":frameName)
    attr.append("is_constant":isConstant)
    attr.append("parallel_iterations":parallelIterations)

    let opspec = tf.OpSpec(
        OpType: "RefEnter",
        Input: [ data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Exits the current frame to its parent frame.

Exit makes its input `data` available to the parent frame.

*/




func refExit(scope:Scope,data: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "RefExit",
        Input: [ data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Return the same ref tensor as the input ref tensor.


*/




func refIdentity(scope:Scope,input: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "RefIdentity",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Forwards the value of an available tensor from `inputs` to `output`.

`Merge` waits for at least one of the tensors in `inputs` to become available.
It is usually combined with `Switch` to implement branching.
`Merge` forwards the first tensor for become available to `output`, and sets
`value_index` to its index in `inputs`.

*/




func refMerge(scope:Scope,inputs: Tensorflow_DataType, t :type  , n :uInt8  )  ->(output: tf.Output, valueIndex: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("N":n)

    let opspec = tf.OpSpec(
        OpType: "RefMerge",
        Input: [ inputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Makes its input available to the next iteration.


*/




func refNextIteration(scope:Scope,data: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "RefNextIteration",
        Input: [ data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Forwards the `index`th element of `inputs` to `output`.


*/




func refSelect(scope:Scope,index: Tensorflow_DataType, inputs: Tensorflow_DataType, t :type  , n :uInt8  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("N":n)

    let opspec = tf.OpSpec(
        OpType: "RefSelect",
        Input: [ index, inputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Forwards the ref tensor `data` to the output port determined by `pred`.

If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
the data goes to `output_false`.
See also `Switch` and `Merge`.

*/




func refSwitch(scope:Scope,data: Tensorflow_DataType, pred: Tensorflow_DataType, t :type  )  ->(outputFalse: tf.Output, outputTrue: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "RefSwitch",
        Input: [ data, pred],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Computes rectified linear: `max(features, 0)`.


*/




func relu(scope:Scope,features: Tensorflow_DataType, t :type  )  ->(activations: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Relu",
        Input: [ features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes rectified linear 6: `min(max(features, 0), 6)`.


*/




func relu6(scope:Scope,features: Tensorflow_DataType, t :type  )  ->(activations: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Relu6",
        Input: [ features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes rectified linear 6 gradients for a Relu6 operation.


*/




func relu6Grad(scope:Scope,gradients: Tensorflow_DataType, features: Tensorflow_DataType, t :type  )  ->(backprops: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Relu6Grad",
        Input: [ gradients, features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes rectified linear gradients for a Relu operation.


*/




func reluGrad(scope:Scope,gradients: Tensorflow_DataType, features: Tensorflow_DataType, t :type  )  ->(backprops: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "ReluGrad",
        Input: [ gradients, features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a dataset that emits the outputs of `input_dataset` `count` times.


*/




func repeatDataset(scope:Scope,inputDataset: Tensorflow_DataType, count: Tensorflow_DataType, outputTypes :list(type)  , outputShapes :[Shape]  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("output_types":outputTypes)
    attr.append("output_shapes":outputShapes)

    let opspec = tf.OpSpec(
        OpType: "RepeatDataset",
        Input: [ inputDataset, count],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Given a quantized tensor described by (input, input_min, input_max), outputs a

range that covers the actual values present in that tensor.  This op is
typically used to produce the requested_output_min and requested_output_max for
Requantize.

*/




func requantizationRange(scope:Scope,input: Tensorflow_DataType, inputMin: Tensorflow_DataType, inputMax: Tensorflow_DataType, tinput :type  )  ->(outputMin: tf.Output, outputMax: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tinput":tinput)

    let opspec = tf.OpSpec(
        OpType: "RequantizationRange",
        Input: [ input, inputMin, inputMax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Convert the quantized 'input' tensor into a lower-precision 'output', using the

output range specified with 'requested_output_min' and 'requested_output_max'.
[input_min, input_max] are scalar floats that specify the range for the float
interpretation of the 'input' data. For example, if input_min is -1.0f and
input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.

*/




func requantize(scope:Scope,input: Tensorflow_DataType, inputMin: Tensorflow_DataType, inputMax: Tensorflow_DataType, requestedOutputMin: Tensorflow_DataType, requestedOutputMax: Tensorflow_DataType, tinput :type  , outType :type  )  ->(output: tf.Output, outputMin: tf.Output, outputMax: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tinput":tinput)
    attr.append("out_type":outType)

    let opspec = tf.OpSpec(
        OpType: "Requantize",
        Input: [ input, inputMin, inputMax, requestedOutputMin, requestedOutputMax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
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

*/




func reshape(scope:Scope,tensor: Tensorflow_DataType, shape: Tensorflow_DataType, t :type  , tshape :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tshape":tshape)

    let opspec = tf.OpSpec(
        OpType: "Reshape",
        Input: [ tensor, shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Resize `images` to `size` using area interpolation.

Input images can be of different types but output images are always float.

*/




func resizeArea(scope:Scope,images: Tensorflow_DataType, size: Tensorflow_DataType, t :type  , alignCorners :bool  )  ->(resizedImages: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("align_corners":alignCorners)

    let opspec = tf.OpSpec(
        OpType: "ResizeArea",
        Input: [ images, size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Resize `images` to `size` using bicubic interpolation.

Input images can be of different types but output images are always float.

*/




func resizeBicubic(scope:Scope,images: Tensorflow_DataType, size: Tensorflow_DataType, t :type  , alignCorners :bool  )  ->(resizedImages: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("align_corners":alignCorners)

    let opspec = tf.OpSpec(
        OpType: "ResizeBicubic",
        Input: [ images, size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Resize `images` to `size` using bilinear interpolation.

Input images can be of different types but output images are always float.

*/




func resizeBilinear(scope:Scope,images: Tensorflow_DataType, size: Tensorflow_DataType, t :type  , alignCorners :bool  )  ->(resizedImages: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("align_corners":alignCorners)

    let opspec = tf.OpSpec(
        OpType: "ResizeBilinear",
        Input: [ images, size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradient of bilinear interpolation.


*/




func resizeBilinearGrad(scope:Scope,grads: Tensorflow_DataType, originalImage: Tensorflow_DataType, t :type  , alignCorners :bool  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("align_corners":alignCorners)

    let opspec = tf.OpSpec(
        OpType: "ResizeBilinearGrad",
        Input: [ grads, originalImage],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Resize `images` to `size` using nearest neighbor interpolation.


*/




func resizeNearestNeighbor(scope:Scope,images: Tensorflow_DataType, size: Tensorflow_DataType, t :type  , alignCorners :bool  )  ->(resizedImages: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("align_corners":alignCorners)

    let opspec = tf.OpSpec(
        OpType: "ResizeNearestNeighbor",
        Input: [ images, size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradient of nearest neighbor interpolation.


*/




func resizeNearestNeighborGrad(scope:Scope,grads: Tensorflow_DataType, size: Tensorflow_DataType, t :type  , alignCorners :bool  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("align_corners":alignCorners)

    let opspec = tf.OpSpec(
        OpType: "ResizeNearestNeighborGrad",
        Input: [ grads, size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Update '*var' according to the adadelta scheme.

accum = rho() * accum + (1 - rho()) * grad.square();
update = (update_accum + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad;
update_accum = rho() * update_accum + (1 - rho()) * update.square();
var -= update;

*/




func resourceApplyAdadelta(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, accumUpdate: Tensorflow_DataType, lr: Tensorflow_DataType, rho: Tensorflow_DataType, epsilon: Tensorflow_DataType, grad: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ResourceApplyAdadelta",
        Input: [ `var`, accum, accumUpdate, lr, rho, epsilon, grad],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Update '*var' according to the adagrad scheme.

accum += grad * grad
var -= lr * grad * (1 / sqrt(accum))

*/




func resourceApplyAdagrad(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, lr: Tensorflow_DataType, grad: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ResourceApplyAdagrad",
        Input: [ `var`, accum, lr, grad],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Update '*var' according to the proximal adagrad scheme.


*/




func resourceApplyAdagradDA(scope:Scope,`var`: Tensorflow_DataType, gradientAccumulator: Tensorflow_DataType, gradientSquaredAccumulator: Tensorflow_DataType, grad: Tensorflow_DataType, lr: Tensorflow_DataType, l1: Tensorflow_DataType, l2: Tensorflow_DataType, globalStep: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ResourceApplyAdagradDA",
        Input: [ `var`, gradientAccumulator, gradientSquaredAccumulator, grad, lr, l1, l2, globalStep],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Update '*var' according to the Adam algorithm.

lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)

*/




func resourceApplyAdam(scope:Scope,`var`: Tensorflow_DataType, m: Tensorflow_DataType, v: Tensorflow_DataType, beta1Power: Tensorflow_DataType, beta2Power: Tensorflow_DataType, lr: Tensorflow_DataType, beta1: Tensorflow_DataType, beta2: Tensorflow_DataType, epsilon: Tensorflow_DataType, grad: Tensorflow_DataType, t :type  , useLocking :bool  , useNesterov :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)
    attr.append("use_nesterov":useNesterov)

    let opspec = tf.OpSpec(
        OpType: "ResourceApplyAdam",
        Input: [ `var`, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
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

*/




func resourceApplyCenteredRMSProp(scope:Scope,`var`: Tensorflow_DataType, mg: Tensorflow_DataType, ms: Tensorflow_DataType, mom: Tensorflow_DataType, lr: Tensorflow_DataType, rho: Tensorflow_DataType, momentum: Tensorflow_DataType, epsilon: Tensorflow_DataType, grad: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ResourceApplyCenteredRMSProp",
        Input: [ `var`, mg, ms, mom, lr, rho, momentum, epsilon, grad],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Update '*var' according to the Ftrl-proximal scheme.

accum_new = accum + grad * grad
linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new

*/




func resourceApplyFtrl(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, linear: Tensorflow_DataType, grad: Tensorflow_DataType, lr: Tensorflow_DataType, l1: Tensorflow_DataType, l2: Tensorflow_DataType, lrPower: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ResourceApplyFtrl",
        Input: [ `var`, accum, linear, grad, lr, l1, l2, lrPower],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Update '*var' by subtracting 'alpha' * 'delta' from it.


*/




func resourceApplyGradientDescent(scope:Scope,`var`: Tensorflow_DataType, alpha: Tensorflow_DataType, delta: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ResourceApplyGradientDescent",
        Input: [ `var`, alpha, delta],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Update '*var' according to the momentum scheme. Set use_nesterov = True if you

want to use Nesterov momentum.
accum = accum * momentum + grad
var -= lr * accum

*/




func resourceApplyMomentum(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, lr: Tensorflow_DataType, grad: Tensorflow_DataType, momentum: Tensorflow_DataType, t :type  , useLocking :bool  , useNesterov :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)
    attr.append("use_nesterov":useNesterov)

    let opspec = tf.OpSpec(
        OpType: "ResourceApplyMomentum",
        Input: [ `var`, accum, lr, grad, momentum],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.

accum += grad * grad
prox_v = var - lr * grad * (1 / sqrt(accum))
var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}

*/




func resourceApplyProximalAdagrad(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, lr: Tensorflow_DataType, l1: Tensorflow_DataType, l2: Tensorflow_DataType, grad: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ResourceApplyProximalAdagrad",
        Input: [ `var`, accum, lr, l1, l2, grad],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Update '*var' as FOBOS algorithm with fixed learning rate.

prox_v = var - alpha * delta
var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

*/




func resourceApplyProximalGradientDescent(scope:Scope,`var`: Tensorflow_DataType, alpha: Tensorflow_DataType, l1: Tensorflow_DataType, l2: Tensorflow_DataType, delta: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ResourceApplyProximalGradientDescent",
        Input: [ `var`, alpha, l1, l2, delta],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
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

*/




func resourceApplyRMSProp(scope:Scope,`var`: Tensorflow_DataType, ms: Tensorflow_DataType, mom: Tensorflow_DataType, lr: Tensorflow_DataType, rho: Tensorflow_DataType, momentum: Tensorflow_DataType, epsilon: Tensorflow_DataType, grad: Tensorflow_DataType, t :type  , useLocking :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ResourceApplyRMSProp",
        Input: [ `var`, ms, mom, lr, rho, momentum, epsilon, grad],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
var: Should be from a Variable().


*/




func resourceSparseApplyAdadelta(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, accumUpdate: Tensorflow_DataType, lr: Tensorflow_DataType, rho: Tensorflow_DataType, epsilon: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ResourceSparseApplyAdadelta",
        Input: [ `var`, accum, accumUpdate, lr, rho, epsilon, grad, indices],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Update relevant entries in '*var' and '*accum' according to the adagrad scheme.

That is for rows we have grad for, we update var and accum as follows:
accum += grad * grad
var -= lr * grad * (1 / sqrt(accum))

*/




func resourceSparseApplyAdagrad(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, lr: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ResourceSparseApplyAdagrad",
        Input: [ `var`, accum, lr, grad, indices],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Update entries in '*var' and '*accum' according to the proximal adagrad scheme.


*/




func resourceSparseApplyAdagradDA(scope:Scope,`var`: Tensorflow_DataType, gradientAccumulator: Tensorflow_DataType, gradientSquaredAccumulator: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, lr: Tensorflow_DataType, l1: Tensorflow_DataType, l2: Tensorflow_DataType, globalStep: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ResourceSparseApplyAdagradDA",
        Input: [ `var`, gradientAccumulator, gradientSquaredAccumulator, grad, indices, lr, l1, l2, globalStep],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
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

*/




func resourceSparseApplyCenteredRMSProp(scope:Scope,`var`: Tensorflow_DataType, mg: Tensorflow_DataType, ms: Tensorflow_DataType, mom: Tensorflow_DataType, lr: Tensorflow_DataType, rho: Tensorflow_DataType, momentum: Tensorflow_DataType, epsilon: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ResourceSparseApplyCenteredRMSProp",
        Input: [ `var`, mg, ms, mom, lr, rho, momentum, epsilon, grad, indices],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Update relevant entries in '*var' according to the Ftrl-proximal scheme.

That is for rows we have grad for, we update var, accum and linear as follows:
accum_new = accum + grad * grad
linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new

*/




func resourceSparseApplyFtrl(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, linear: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, lr: Tensorflow_DataType, l1: Tensorflow_DataType, l2: Tensorflow_DataType, lrPower: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ResourceSparseApplyFtrl",
        Input: [ `var`, accum, linear, grad, indices, lr, l1, l2, lrPower],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Update relevant entries in '*var' and '*accum' according to the momentum scheme.

Set use_nesterov = True if you want to use Nesterov momentum.
That is for rows we have grad for, we update var and accum as follows:
accum = accum * momentum + grad
var -= lr * accum

*/




func resourceSparseApplyMomentum(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, lr: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, momentum: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  , useNesterov :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)
    attr.append("use_nesterov":useNesterov)

    let opspec = tf.OpSpec(
        OpType: "ResourceSparseApplyMomentum",
        Input: [ `var`, accum, lr, grad, indices, momentum],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Sparse update entries in '*var' and '*accum' according to FOBOS algorithm.

That is for rows we have grad for, we update var and accum as follows:
accum += grad * grad
prox_v = var
prox_v -= lr * grad * (1 / sqrt(accum))
var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}

*/




func resourceSparseApplyProximalAdagrad(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, lr: Tensorflow_DataType, l1: Tensorflow_DataType, l2: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ResourceSparseApplyProximalAdagrad",
        Input: [ `var`, accum, lr, l1, l2, grad, indices],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Sparse update '*var' as FOBOS algorithm with fixed learning rate.

That is for rows we have grad for, we update var as follows:
prox_v = var - alpha * grad
var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

*/




func resourceSparseApplyProximalGradientDescent(scope:Scope,`var`: Tensorflow_DataType, alpha: Tensorflow_DataType, l1: Tensorflow_DataType, l2: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ResourceSparseApplyProximalGradientDescent",
        Input: [ `var`, alpha, l1, l2, grad, indices],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
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

*/




func resourceSparseApplyRMSProp(scope:Scope,`var`: Tensorflow_DataType, ms: Tensorflow_DataType, mom: Tensorflow_DataType, lr: Tensorflow_DataType, rho: Tensorflow_DataType, momentum: Tensorflow_DataType, epsilon: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ResourceSparseApplyRMSProp",
        Input: [ `var`, ms, mom, lr, rho, momentum, epsilon, grad, indices],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Assign `value` to the sliced l-value reference of `ref`.

The values of `value` are assigned to the positions in the variable
`ref` that are selected by the slice parameters. The slice parameters
`begin, `end`, `strides`, etc. work exactly as in `StridedSlice`.
NOTE this op currently does not support broadcasting and so `value`'s
shape must be exactly the shape produced by the slice of `ref`.

*/




func resourceStridedSliceAssign(scope:Scope,ref: Tensorflow_DataType, begin: Tensorflow_DataType, end: Tensorflow_DataType, strides: Tensorflow_DataType, value: Tensorflow_DataType, t :type  , index :type  , beginMask :uInt8  , endMask :uInt8  , ellipsisMask :uInt8  , newAxisMask :uInt8  , shrinkAxisMask :uInt8  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Index":index)
    attr.append("begin_mask":beginMask)
    attr.append("end_mask":endMask)
    attr.append("ellipsis_mask":ellipsisMask)
    attr.append("new_axis_mask":newAxisMask)
    attr.append("shrink_axis_mask":shrinkAxisMask)

    let opspec = tf.OpSpec(
        OpType: "ResourceStridedSliceAssign",
        Input: [ ref, begin, end, strides, value],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
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

*/




func restore(scope:Scope,filePattern: Tensorflow_DataType, tensorName: Tensorflow_DataType, dt :type  , preferredShard :uInt8  )  ->(tensor: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dt":dt)
    attr.append("preferred_shard":preferredShard)

    let opspec = tf.OpSpec(
        OpType: "Restore",
        Input: [ filePattern, tensorName],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Restores a tensor from checkpoint files.

This is like `Restore` except that restored tensor can be listed as filling
only a slice of a larger tensor.  `shape_and_slice` specifies the shape of the
larger tensor and the slice that the restored tensor covers.
The `shape_and_slice` input has the same format as the
elements of the `shapes_and_slices` input of the `SaveSlices` op.

*/




func restoreSlice(scope:Scope,filePattern: Tensorflow_DataType, tensorName: Tensorflow_DataType, shapeAndSlice: Tensorflow_DataType, dt :type  , preferredShard :uInt8  )  ->(tensor: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dt":dt)
    attr.append("preferred_shard":preferredShard)

    let opspec = tf.OpSpec(
        OpType: "RestoreSlice",
        Input: [ filePattern, tensorName, shapeAndSlice],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func restoreV2(scope:Scope,`prefix`: Tensorflow_DataType, tensorNames: Tensorflow_DataType, shapeAndSlices: Tensorflow_DataType, dtypes :list(type)  )  ->(tensors: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtypes":dtypes)

    let opspec = tf.OpSpec(
        OpType: "RestoreV2",
        Input: [ `prefix`, tensorNames, shapeAndSlices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func reverse(scope:Scope,tensor: Tensorflow_DataType, dims: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Reverse",
        Input: [ tensor, dims],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func reverseSequence(scope:Scope,input: Tensorflow_DataType, seqLengths: Tensorflow_DataType, seqDim :uInt8  , batchDim :uInt8  , t :type  , tlen :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("seq_dim":seqDim)
    attr.append("batch_dim":batchDim)
    attr.append("T":t)
    attr.append("Tlen":tlen)

    let opspec = tf.OpSpec(
        OpType: "ReverseSequence",
        Input: [ input, seqLengths],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func reverseV2(scope:Scope,tensor: Tensorflow_DataType, axis: Tensorflow_DataType, tidx :type  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tidx":tidx)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "ReverseV2",
        Input: [ tensor, axis],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func rint(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Rint",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Rounds the values of a tensor to the nearest integer, element-wise.

Rounds half to even.  Also known as bankers rounding. If you want to round
according to the current system rounding mode use std::cint.

*/




func round(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Round",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes reciprocal of square root of x element-wise.

I.e., \\(y = 1 / \sqrt{x}\\).

*/




func rsqrt(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Rsqrt",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradient for the rsqrt of `x` wrt its input.

Specifically, `grad = dy * -0.5 * y^3`, where `y = rsqrt(x)`, and `dy`
is the corresponding input gradient.

*/




func rsqrtGrad(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "RsqrtGrad",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func sampleDistortedBoundingBox(scope:Scope,imageSize: Tensorflow_DataType, boundingBoxes: Tensorflow_DataType, t :type  , seed :uInt8  , seed2 :uInt8  , minObjectCovered :float  , aspectRatioRange :list(float)  , areaRange :list(float)  , maxAttempts :uInt8  , useImageIfNoBoundingBoxes :bool  )  ->(begin: tf.Output, size: tf.Output, bboxes: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("seed":seed)
    attr.append("seed2":seed2)
    attr.append("min_object_covered":minObjectCovered)
    attr.append("aspect_ratio_range":aspectRatioRange)
    attr.append("area_range":areaRange)
    attr.append("max_attempts":maxAttempts)
    attr.append("use_image_if_no_bounding_boxes":useImageIfNoBoundingBoxes)

    let opspec = tf.OpSpec(
        OpType: "SampleDistortedBoundingBox",
        Input: [ imageSize, boundingBoxes],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Saves the input tensors to disk.

The size of `tensor_names` must match the number of tensors in `data`. `data[i]`
is written to `filename` with name `tensor_names[i]`.
See also `SaveSlices`.

*/




func save(scope:Scope,filename: Tensorflow_DataType, tensorNames: Tensorflow_DataType, data: Tensorflow_DataType, t :list(type)  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Save",
        Input: [ filename, tensorNames, data],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
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

*/




func saveSlices(scope:Scope,filename: Tensorflow_DataType, tensorNames: Tensorflow_DataType, shapesAndSlices: Tensorflow_DataType, data: Tensorflow_DataType, t :list(type)  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SaveSlices",
        Input: [ filename, tensorNames, shapesAndSlices, data],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Saves tensors in V2 checkpoint format.

By default, saves the named tensors in full.  If the caller wishes to save
specific slices of full tensors, "shape_and_slices" should be non-empty strings
and correspondingly well-formed.

*/




func saveV2(scope:Scope,`prefix`: Tensorflow_DataType, tensorNames: Tensorflow_DataType, shapeAndSlices: Tensorflow_DataType, tensors: Tensorflow_DataType, dtypes :list(type)  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtypes":dtypes)

    let opspec = tf.OpSpec(
        OpType: "SaveV2",
        Input: [ `prefix`, tensorNames, shapeAndSlices, tensors],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Outputs a `Summary` protocol buffer with scalar values.

The input `tags` and `values` must have the same shape.  The generated summary
has a summary value for each tag-value pair in `tags` and `values`.

*/




func scalarSummary(scope:Scope,tags: Tensorflow_DataType, values: Tensorflow_DataType, t :type  )  ->(summary: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "ScalarSummary",
        Input: [ tags, values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func scatterAdd(scope:Scope,ref: Tensorflow_DataType, indices: Tensorflow_DataType, updates: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(outputRef: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ScatterAdd",
        Input: [ ref, indices, updates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func scatterDiv(scope:Scope,ref: Tensorflow_DataType, indices: Tensorflow_DataType, updates: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(outputRef: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ScatterDiv",
        Input: [ ref, indices, updates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func scatterMul(scope:Scope,ref: Tensorflow_DataType, indices: Tensorflow_DataType, updates: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(outputRef: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ScatterMul",
        Input: [ ref, indices, updates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func scatterNd(scope:Scope,indices: Tensorflow_DataType, updates: Tensorflow_DataType, shape: Tensorflow_DataType, t :type  , tindices :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)

    let opspec = tf.OpSpec(
        OpType: "ScatterNd",
        Input: [ indices, updates, shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func scatterNdAdd(scope:Scope,ref: Tensorflow_DataType, indices: Tensorflow_DataType, updates: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(outputRef: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ScatterNdAdd",
        Input: [ ref, indices, updates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func scatterNdSub(scope:Scope,ref: Tensorflow_DataType, indices: Tensorflow_DataType, updates: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(outputRef: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ScatterNdSub",
        Input: [ ref, indices, updates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func scatterNdUpdate(scope:Scope,ref: Tensorflow_DataType, indices: Tensorflow_DataType, updates: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(outputRef: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ScatterNdUpdate",
        Input: [ ref, indices, updates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func scatterSub(scope:Scope,ref: Tensorflow_DataType, indices: Tensorflow_DataType, updates: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(outputRef: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ScatterSub",
        Input: [ ref, indices, updates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func scatterUpdate(scope:Scope,ref: Tensorflow_DataType, indices: Tensorflow_DataType, updates: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(outputRef: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "ScatterUpdate",
        Input: [ ref, indices, updates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes fingerprints of the input strings.


*/




func sdcaFprint(scope:Scope,input: Tensorflow_DataType )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "SdcaFprint",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func sdcaOptimizer(scope:Scope,sparseExampleIndices: Tensorflow_DataType, sparseFeatureIndices: Tensorflow_DataType, sparseFeatureValues: Tensorflow_DataType, denseFeatures: Tensorflow_DataType, exampleWeights: Tensorflow_DataType, exampleLabels: Tensorflow_DataType, sparseIndices: Tensorflow_DataType, sparseWeights: Tensorflow_DataType, denseWeights: Tensorflow_DataType, exampleStateData: Tensorflow_DataType, lossType :string  , adaptative :bool  , numSparseFeatures :uInt8  , numSparseFeaturesWithValues :uInt8  , numDenseFeatures :uInt8  , l1 :float  , l2 :float  , numLossPartitions :uInt8  , numInnerIterations :uInt8  )  ->(outExampleStateData: tf.Output, outDeltaSparseWeights: tf.Output, outDeltaDenseWeights: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("loss_type":lossType)
    attr.append("adaptative":adaptative)
    attr.append("num_sparse_features":numSparseFeatures)
    attr.append("num_sparse_features_with_values":numSparseFeaturesWithValues)
    attr.append("num_dense_features":numDenseFeatures)
    attr.append("l1":l1)
    attr.append("l2":l2)
    attr.append("num_loss_partitions":numLossPartitions)
    attr.append("num_inner_iterations":numInnerIterations)

    let opspec = tf.OpSpec(
        OpType: "SdcaOptimizer",
        Input: [ sparseExampleIndices, sparseFeatureIndices, sparseFeatureValues, denseFeatures, exampleWeights, exampleLabels, sparseIndices, sparseWeights, denseWeights, exampleStateData],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Applies L1 regularization shrink step on the parameters.


*/




func sdcaShrinkL1(scope:Scope,weights: Tensorflow_DataType, numFeatures :uInt8  , l1 :float  , l2 :float  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_features":numFeatures)
    attr.append("l1":l1)
    attr.append("l2":l2)

    let opspec = tf.OpSpec(
        OpType: "SdcaShrinkL1",
        Input: [ weights],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
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

*/




func segmentMax(scope:Scope,data: Tensorflow_DataType, segmentIds: Tensorflow_DataType, t :type  , tindices :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)

    let opspec = tf.OpSpec(
        OpType: "SegmentMax",
        Input: [ data, segmentIds],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func segmentMean(scope:Scope,data: Tensorflow_DataType, segmentIds: Tensorflow_DataType, t :type  , tindices :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)

    let opspec = tf.OpSpec(
        OpType: "SegmentMean",
        Input: [ data, segmentIds],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func segmentMin(scope:Scope,data: Tensorflow_DataType, segmentIds: Tensorflow_DataType, t :type  , tindices :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)

    let opspec = tf.OpSpec(
        OpType: "SegmentMin",
        Input: [ data, segmentIds],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func segmentProd(scope:Scope,data: Tensorflow_DataType, segmentIds: Tensorflow_DataType, t :type  , tindices :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)

    let opspec = tf.OpSpec(
        OpType: "SegmentProd",
        Input: [ data, segmentIds],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func segmentSum(scope:Scope,data: Tensorflow_DataType, segmentIds: Tensorflow_DataType, t :type  , tindices :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)

    let opspec = tf.OpSpec(
        OpType: "SegmentSum",
        Input: [ data, segmentIds],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func select(scope:Scope,condition: Tensorflow_DataType, t: Tensorflow_DataType, e: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Select",
        Input: [ condition, t, e],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the Eigen Decomposition of a batch of square self-adjoint matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices, with the same constraints as the single matrix
SelfAdjointEig.
The result is a [..., M+1, M] matrix with [..., 0,:] containing the
eigenvalues, and subsequent [...,1:, :] containing the eigenvectors.

*/




func selfAdjointEig(scope:Scope,input: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SelfAdjointEig",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func selfAdjointEigV2(scope:Scope,input: Tensorflow_DataType, computeV :bool  , t :type  )  ->(e: tf.Output, v: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("compute_v":computeV)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SelfAdjointEigV2",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` string `Tensor`.

The `SparseTensor` must have rank `R` greater than 1, and the first dimension
is treated as the minibatch dimension.  Elements of the `SparseTensor`
must be sorted in increasing order of this first dimension.  The serialized
`SparseTensor` objects going into each row of `serialized_sparse` will have
rank `R-1`.
The minibatch size `N` is extracted from `sparse_shape[0]`.

*/




func serializeManySparse(scope:Scope,sparseIndices: Tensorflow_DataType, sparseValues: Tensorflow_DataType, sparseShape: Tensorflow_DataType, t :type  )  ->(serializedSparse: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SerializeManySparse",
        Input: [ sparseIndices, sparseValues, sparseShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Serialize a `SparseTensor` into a string 3-vector (1-D `Tensor`) object.


*/




func serializeSparse(scope:Scope,sparseIndices: Tensorflow_DataType, sparseValues: Tensorflow_DataType, sparseShape: Tensorflow_DataType, t :type  )  ->(serializedSparse: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SerializeSparse",
        Input: [ sparseIndices, sparseValues, sparseShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Number of unique elements along last dimension of input `set`.

Input `set` is a `SparseTensor` represented by `set_indices`, `set_values`,
and `set_shape`. The last dimension contains values in a set, duplicates are
allowed but ignored.
If `validate_indices` is `True`, this op validates the order and range of `set`
indices.

*/




func setSize(scope:Scope,setIndices: Tensorflow_DataType, setValues: Tensorflow_DataType, setShape: Tensorflow_DataType, validateIndices :bool  , t :type  )  ->(size: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("validate_indices":validateIndices)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SetSize",
        Input: [ setIndices, setValues, setShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the shape of a tensor.

This operation returns a 1-D integer tensor representing the shape of `input`.
For example:
```
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
shape(t) ==> [2, 2, 3]
```

*/




func shape(scope:Scope,input: Tensorflow_DataType, t :type  , outType :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("out_type":outType)

    let opspec = tf.OpSpec(
        OpType: "Shape",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns shape of tensors.

This operation returns N 1-D integer tensors representing shape of `input[i]s`.

*/




func shapeN(scope:Scope,input: Tensorflow_DataType, n :uInt8  , t :type  , outType :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("N":n)
    attr.append("T":t)
    attr.append("out_type":outType)

    let opspec = tf.OpSpec(
        OpType: "ShapeN",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Generate a sharded filename. The filename is printf formatted as

   %s-%05d-of-%05d, basename, shard, num_shards.

*/




func shardedFilename(scope:Scope,basename: Tensorflow_DataType, shard: Tensorflow_DataType, numShards: Tensorflow_DataType )  ->(filename: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ShardedFilename",
        Input: [ basename, shard, numShards],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Generate a glob pattern matching all sharded file names.


*/




func shardedFilespec(scope:Scope,basename: Tensorflow_DataType, numShards: Tensorflow_DataType )  ->(filename: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "ShardedFilespec",
        Input: [ basename, numShards],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a dataset that shuffles elements from `input_dataset` pseudorandomly.


*/




func shuffleDataset(scope:Scope,inputDataset: Tensorflow_DataType, bufferSize: Tensorflow_DataType, seed: Tensorflow_DataType, seed2: Tensorflow_DataType, outputTypes :list(type)  , outputShapes :[Shape]  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("output_types":outputTypes)
    attr.append("output_shapes":outputShapes)

    let opspec = tf.OpSpec(
        OpType: "ShuffleDataset",
        Input: [ inputDataset, bufferSize, seed, seed2],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes sigmoid of `x` element-wise.

Specifically, `y = 1 / (1 + exp(-x))`.

*/




func sigmoid(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Sigmoid",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradient of the sigmoid of `x` wrt its input.

Specifically, `grad = dy * y * (1 - y)`, where `y = sigmoid(x)`, and
`dy` is the corresponding input gradient.

*/




func sigmoidGrad(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SigmoidGrad",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns an element-wise indication of the sign of a number.

`y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.
For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.

*/




func sign(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Sign",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes sin of x element-wise.


*/




func sin(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Sin",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func size(scope:Scope,input: Tensorflow_DataType, t :type  , outType :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("out_type":outType)

    let opspec = tf.OpSpec(
        OpType: "Size",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a dataset that skips `count` elements from the `input_dataset`.


*/




func skipDataset(scope:Scope,inputDataset: Tensorflow_DataType, count: Tensorflow_DataType, outputTypes :list(type)  , outputShapes :[Shape]  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("output_types":outputTypes)
    attr.append("output_shapes":outputShapes)

    let opspec = tf.OpSpec(
        OpType: "SkipDataset",
        Input: [ inputDataset, count],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Parses a text file and creates a batch of examples.


*/




func skipgram(scope:Scope filename :string  , batchSize :uInt8  , windowSize :uInt8  , minCount :uInt8  , subsample :float  )  ->(vocabWord: tf.Output, vocabFreq: tf.Output, wordsPerEpoch: tf.Output, currentEpoch: tf.Output, totalWordsProcessed: tf.Output, examples: tf.Output, labels: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("filename":filename)
    attr.append("batch_size":batchSize)
    attr.append("window_size":windowSize)
    attr.append("min_count":minCount)
    attr.append("subsample":subsample)

    let opspec = tf.OpSpec(
        OpType: "Skipgram",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1),op.Output(4 - 1),op.Output(5 - 1),op.Output(6 - 1),op.Output(7 - 1))
}

/*
Return a slice from 'input'.

The output tensor is a tensor with dimensions described by 'size'
whose values are extracted from 'input' starting at the offsets in
'begin'.
*Requirements*:
  0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)

*/




func slice(scope:Scope,input: Tensorflow_DataType, begin: Tensorflow_DataType, size: Tensorflow_DataType, t :type  , index :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Index":index)

    let opspec = tf.OpSpec(
        OpType: "Slice",
        Input: [ input, begin, size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes softmax activations.

For each batch `i` and class `j` we have
    softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))

*/




func softmax(scope:Scope,logits: Tensorflow_DataType, t :type  )  ->(softmax: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Softmax",
        Input: [ logits],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes softmax cross entropy cost and gradients to backpropagate.

Inputs are the logits, not probabilities.

*/




func softmaxCrossEntropyWithLogits(scope:Scope,features: Tensorflow_DataType, labels: Tensorflow_DataType, t :type  )  ->(loss: tf.Output, backprop: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SoftmaxCrossEntropyWithLogits",
        Input: [ features, labels],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Computes softplus: `log(exp(features) + 1)`.


*/




func softplus(scope:Scope,features: Tensorflow_DataType, t :type  )  ->(activations: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Softplus",
        Input: [ features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes softplus gradients for a softplus operation.


*/




func softplusGrad(scope:Scope,gradients: Tensorflow_DataType, features: Tensorflow_DataType, t :type  )  ->(backprops: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SoftplusGrad",
        Input: [ gradients, features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes softsign: `features / (abs(features) + 1)`.


*/




func softsign(scope:Scope,features: Tensorflow_DataType, t :type  )  ->(activations: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Softsign",
        Input: [ features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes softsign gradients for a softsign operation.


*/




func softsignGrad(scope:Scope,gradients: Tensorflow_DataType, features: Tensorflow_DataType, t :type  )  ->(backprops: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SoftsignGrad",
        Input: [ gradients, features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
SpaceToBatch for 4-D tensors of type T.

This is a legacy version of the more general SpaceToBatchND.
Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
More specifically, this op outputs a copy of the input tensor where values from
the `height` and `width` dimensions are moved to the `batch` dimension. After
the zero-padding, both `height` and `width` of the input must be divisible by the
block size.

*/




func spaceToBatch(scope:Scope,input: Tensorflow_DataType, paddings: Tensorflow_DataType, t :type  , tpaddings :type  , blockSize :uInt8  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tpaddings":tpaddings)
    attr.append("block_size":blockSize)

    let opspec = tf.OpSpec(
        OpType: "SpaceToBatch",
        Input: [ input, paddings],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func spaceToBatchND(scope:Scope,input: Tensorflow_DataType, blockShape: Tensorflow_DataType, paddings: Tensorflow_DataType, t :type  , tblockShape :type  , tpaddings :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tblock_shape":tblockShape)
    attr.append("Tpaddings":tpaddings)

    let opspec = tf.OpSpec(
        OpType: "SpaceToBatchND",
        Input: [ input, blockShape, paddings],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func spaceToDepth(scope:Scope,input: Tensorflow_DataType, t :type  , blockSize :uInt8  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("block_size":blockSize)

    let opspec = tf.OpSpec(
        OpType: "SpaceToDepth",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Applies a sparse gradient to a given accumulator.

Does not add if local_step is smaller than the accumulator's
global_step.

*/




func sparseAccumulatorApplyGradient(scope:Scope,handle: Tensorflow_DataType, localStep: Tensorflow_DataType, gradientIndices: Tensorflow_DataType, gradientValues: Tensorflow_DataType, gradientShape: Tensorflow_DataType, dtype :type  , hasKnownShape :bool  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("has_known_shape":hasKnownShape)

    let opspec = tf.OpSpec(
        OpType: "SparseAccumulatorApplyGradient",
        Input: [ handle, localStep, gradientIndices, gradientValues, gradientShape],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Extracts the average sparse gradient in a SparseConditionalAccumulator.

The op will blocks until sufficient (i.e., more than num_required)
gradients have been accumulated. If the accumulator has already
aggregated more than num_required gradients, it will return its
average of the accumulated gradients.  Also automatically increments
the recorded global_step in the accumulator by 1, and resets the
aggregate to 0.

*/




func sparseAccumulatorTakeGradient(scope:Scope,handle: Tensorflow_DataType, numRequired: Tensorflow_DataType, dtype :type  )  ->(indices: tf.Output, values: tf.Output, shape: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)

    let opspec = tf.OpSpec(
        OpType: "SparseAccumulatorTakeGradient",
        Input: [ handle, numRequired],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
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

*/




func sparseAdd(scope:Scope,aIndices: Tensorflow_DataType, aValues: Tensorflow_DataType, aShape: Tensorflow_DataType, bIndices: Tensorflow_DataType, bValues: Tensorflow_DataType, bShape: Tensorflow_DataType, thresh: Tensorflow_DataType, t :type  , treal :type  )  ->(sumIndices: tf.Output, sumValues: tf.Output, sumShape: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Treal":treal)

    let opspec = tf.OpSpec(
        OpType: "SparseAdd",
        Input: [ aIndices, aValues, aShape, bIndices, bValues, bShape, thresh],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
The gradient operator for the SparseAdd op.

The SparseAdd op calculates A + B, where A, B, and the sum are all represented
as `SparseTensor` objects.  This op takes in the upstream gradient w.r.t.
non-empty values of the sum, and outputs the gradients w.r.t. the non-empty
values of A and B.

*/




func sparseAddGrad(scope:Scope,backpropValGrad: Tensorflow_DataType, aIndices: Tensorflow_DataType, bIndices: Tensorflow_DataType, sumIndices: Tensorflow_DataType, t :type  )  ->(aValGrad: tf.Output, bValGrad: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SparseAddGrad",
        Input: [ backpropValGrad, aIndices, bIndices, sumIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
var: Should be from a Variable().


*/




func sparseApplyAdadelta(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, accumUpdate: Tensorflow_DataType, lr: Tensorflow_DataType, rho: Tensorflow_DataType, epsilon: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "SparseApplyAdadelta",
        Input: [ `var`, accum, accumUpdate, lr, rho, epsilon, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Update relevant entries in '*var' and '*accum' according to the adagrad scheme.

That is for rows we have grad for, we update var and accum as follows:
accum += grad * grad
var -= lr * grad * (1 / sqrt(accum))

*/




func sparseApplyAdagrad(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, lr: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "SparseApplyAdagrad",
        Input: [ `var`, accum, lr, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Update entries in '*var' and '*accum' according to the proximal adagrad scheme.


*/




func sparseApplyAdagradDA(scope:Scope,`var`: Tensorflow_DataType, gradientAccumulator: Tensorflow_DataType, gradientSquaredAccumulator: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, lr: Tensorflow_DataType, l1: Tensorflow_DataType, l2: Tensorflow_DataType, globalStep: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "SparseApplyAdagradDA",
        Input: [ `var`, gradientAccumulator, gradientSquaredAccumulator, grad, indices, lr, l1, l2, globalStep],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func sparseApplyCenteredRMSProp(scope:Scope,`var`: Tensorflow_DataType, mg: Tensorflow_DataType, ms: Tensorflow_DataType, mom: Tensorflow_DataType, lr: Tensorflow_DataType, rho: Tensorflow_DataType, momentum: Tensorflow_DataType, epsilon: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "SparseApplyCenteredRMSProp",
        Input: [ `var`, mg, ms, mom, lr, rho, momentum, epsilon, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Update relevant entries in '*var' according to the Ftrl-proximal scheme.

That is for rows we have grad for, we update var, accum and linear as follows:
accum_new = accum + grad * grad
linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new

*/




func sparseApplyFtrl(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, linear: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, lr: Tensorflow_DataType, l1: Tensorflow_DataType, l2: Tensorflow_DataType, lrPower: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "SparseApplyFtrl",
        Input: [ `var`, accum, linear, grad, indices, lr, l1, l2, lrPower],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Update relevant entries in '*var' and '*accum' according to the momentum scheme.

Set use_nesterov = True if you want to use Nesterov momentum.
That is for rows we have grad for, we update var and accum as follows:
accum = accum * momentum + grad
var -= lr * accum

*/




func sparseApplyMomentum(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, lr: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, momentum: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  , useNesterov :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)
    attr.append("use_nesterov":useNesterov)

    let opspec = tf.OpSpec(
        OpType: "SparseApplyMomentum",
        Input: [ `var`, accum, lr, grad, indices, momentum],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Sparse update entries in '*var' and '*accum' according to FOBOS algorithm.

That is for rows we have grad for, we update var and accum as follows:
accum += grad * grad
prox_v = var
prox_v -= lr * grad * (1 / sqrt(accum))
var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}

*/




func sparseApplyProximalAdagrad(scope:Scope,`var`: Tensorflow_DataType, accum: Tensorflow_DataType, lr: Tensorflow_DataType, l1: Tensorflow_DataType, l2: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "SparseApplyProximalAdagrad",
        Input: [ `var`, accum, lr, l1, l2, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Sparse update '*var' as FOBOS algorithm with fixed learning rate.

That is for rows we have grad for, we update var as follows:
prox_v = var - alpha * grad
var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

*/




func sparseApplyProximalGradientDescent(scope:Scope,`var`: Tensorflow_DataType, alpha: Tensorflow_DataType, l1: Tensorflow_DataType, l2: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "SparseApplyProximalGradientDescent",
        Input: [ `var`, alpha, l1, l2, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func sparseApplyRMSProp(scope:Scope,`var`: Tensorflow_DataType, ms: Tensorflow_DataType, mom: Tensorflow_DataType, lr: Tensorflow_DataType, rho: Tensorflow_DataType, momentum: Tensorflow_DataType, epsilon: Tensorflow_DataType, grad: Tensorflow_DataType, indices: Tensorflow_DataType, t :type  , tindices :type  , useLocking :bool  )  ->(out: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("use_locking":useLocking)

    let opspec = tf.OpSpec(
        OpType: "SparseApplyRMSProp",
        Input: [ `var`, ms, mom, lr, rho, momentum, epsilon, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func sparseConcat(scope:Scope,indices: Tensorflow_DataType, values: Tensorflow_DataType, shapes: Tensorflow_DataType, concatDim :uInt8  , n :uInt8  , t :type  )  ->(outputIndices: tf.Output, outputValues: tf.Output, outputShape: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("concat_dim":concatDim)
    attr.append("N":n)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SparseConcat",
        Input: [ indices, values, shapes],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
A conditional accumulator for aggregating sparse gradients.

The accumulator accepts gradients marked with local_step greater or
equal to the most recent global_step known to the accumulator. The
average can be extracted from the accumulator, provided sufficient
gradients have been accumulated. Extracting the average automatically
resets the aggregate to 0, and increments the global_step recorded by
the accumulator.

*/




func sparseConditionalAccumulator(scope:Scope dtype :type  , shape :shape  , container :string  , sharedName :string  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("shape":shape)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "SparseConditionalAccumulator",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func sparseCross(scope:Scope,indices: Tensorflow_DataType, values: Tensorflow_DataType, shapes: Tensorflow_DataType, denseInputs: Tensorflow_DataType, n :uInt8  , hashedOutput :bool  , numBuckets :uInt8  , hashKey :uInt8  , sparseTypes :list(type)  , denseTypes :list(type)  , outType :type  , internalType :type  )  ->(outputIndices: tf.Output, outputValues: tf.Output, outputShape: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("N":n)
    attr.append("hashed_output":hashedOutput)
    attr.append("num_buckets":numBuckets)
    attr.append("hash_key":hashKey)
    attr.append("sparse_types":sparseTypes)
    attr.append("dense_types":denseTypes)
    attr.append("out_type":outType)
    attr.append("internal_type":internalType)

    let opspec = tf.OpSpec(
        OpType: "SparseCross",
        Input: [ indices, values, shapes, denseInputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
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

*/




func sparseDenseCwiseAdd(scope:Scope,spIndices: Tensorflow_DataType, spValues: Tensorflow_DataType, spShape: Tensorflow_DataType, dense: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SparseDenseCwiseAdd",
        Input: [ spIndices, spValues, spShape, dense],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Component-wise divides a SparseTensor by a dense Tensor.

*Limitation*: this Op only broadcasts the dense side to the sparse side, but not
the other direction.

*/




func sparseDenseCwiseDiv(scope:Scope,spIndices: Tensorflow_DataType, spValues: Tensorflow_DataType, spShape: Tensorflow_DataType, dense: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SparseDenseCwiseDiv",
        Input: [ spIndices, spValues, spShape, dense],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Component-wise multiplies a SparseTensor by a dense Tensor.

The output locations corresponding to the implicitly zero elements in the sparse
tensor will be zero (i.e., will not take up storage space), regardless of the
contents of the dense tensor (even if it's +/-INF and that INF*0 == NaN).
*Limitation*: this Op only broadcasts the dense side to the sparse side, but not
the other direction.

*/




func sparseDenseCwiseMul(scope:Scope,spIndices: Tensorflow_DataType, spValues: Tensorflow_DataType, spShape: Tensorflow_DataType, dense: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SparseDenseCwiseMul",
        Input: [ spIndices, spValues, spShape, dense],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func sparseFillEmptyRows(scope:Scope,indices: Tensorflow_DataType, values: Tensorflow_DataType, denseShape: Tensorflow_DataType, defaultValue: Tensorflow_DataType, t :type  )  ->(outputIndices: tf.Output, outputValues: tf.Output, emptyRowIndicator: tf.Output, reverseIndexMap: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SparseFillEmptyRows",
        Input: [ indices, values, denseShape, defaultValue],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1),op.Output(4 - 1))
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

*/




func sparseFillEmptyRowsGrad(scope:Scope,reverseIndexMap: Tensorflow_DataType, gradValues: Tensorflow_DataType, t :type  )  ->(dValues: tf.Output, dDefaultValue: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SparseFillEmptyRowsGrad",
        Input: [ reverseIndexMap, gradValues],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Multiply matrix "a" by matrix "b".

The inputs must be two-dimensional matrices and the inner dimension of "a" must
match the outer dimension of "b". This op is optimized for the case where at
least one of "a" or "b" is sparse. The breakeven for using this versus a dense
matrix multiply on one platform was 30% zero values in the sparse matrix.

*/




func sparseMatMul(scope:Scope,a: Tensorflow_DataType, b: Tensorflow_DataType, transposeA :bool  , transposeB :bool  , aIsSparse :bool  , bIsSparse :bool  , ta :type  , tb :type  )  ->(product: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("transpose_a":transposeA)
    attr.append("transpose_b":transposeB)
    attr.append("a_is_sparse":aIsSparse)
    attr.append("b_is_sparse":bIsSparse)
    attr.append("Ta":ta)
    attr.append("Tb":tb)

    let opspec = tf.OpSpec(
        OpType: "SparseMatMul",
        Input: [ a, b],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func sparseReduceSum(scope:Scope,inputIndices: Tensorflow_DataType, inputValues: Tensorflow_DataType, inputShape: Tensorflow_DataType, reductionAxes: Tensorflow_DataType, keepDims :bool  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("keep_dims":keepDims)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SparseReduceSum",
        Input: [ inputIndices, inputValues, inputShape, reductionAxes],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func sparseReduceSumSparse(scope:Scope,inputIndices: Tensorflow_DataType, inputValues: Tensorflow_DataType, inputShape: Tensorflow_DataType, reductionAxes: Tensorflow_DataType, keepDims :bool  , t :type  )  ->(outputIndices: tf.Output, outputValues: tf.Output, outputShape: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("keep_dims":keepDims)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SparseReduceSumSparse",
        Input: [ inputIndices, inputValues, inputShape, reductionAxes],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Reorders a SparseTensor into the canonical, row-major ordering.

Note that by convention, all sparse ops preserve the canonical ordering along
increasing dimension number. The only time ordering can be violated is during
manual manipulation of the indices and values vectors to add entries.
Reordering does not affect the shape of the SparseTensor.
If the tensor has rank `R` and `N` non-empty values, `input_indices` has
shape `[N, R]`, input_values has length `N`, and input_shape has length `R`.

*/




func sparseReorder(scope:Scope,inputIndices: Tensorflow_DataType, inputValues: Tensorflow_DataType, inputShape: Tensorflow_DataType, t :type  )  ->(outputIndices: tf.Output, outputValues: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SparseReorder",
        Input: [ inputIndices, inputValues, inputShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
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

*/




func sparseReshape(scope:Scope,inputIndices: Tensorflow_DataType, inputShape: Tensorflow_DataType, newShape: Tensorflow_DataType )  ->(outputIndices: tf.Output, outputShape: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "SparseReshape",
        Input: [ inputIndices, inputShape, newShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Computes the mean along sparse segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
segments.
Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
dimension, selecting a subset of dimension 0, specified by `indices`.

*/




func sparseSegmentMean(scope:Scope,data: Tensorflow_DataType, indices: Tensorflow_DataType, segmentIds: Tensorflow_DataType, t :type  , tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "SparseSegmentMean",
        Input: [ data, indices, segmentIds],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes gradients for SparseSegmentMean.

Returns tensor "output" with same shape as grad, except for dimension 0 whose
value is output_dim0.

*/




func sparseSegmentMeanGrad(scope:Scope,grad: Tensorflow_DataType, indices: Tensorflow_DataType, segmentIds: Tensorflow_DataType, outputDim0: Tensorflow_DataType, t :type  , tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "SparseSegmentMeanGrad",
        Input: [ grad, indices, segmentIds, outputDim0],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the sum along sparse segments of a tensor divided by the sqrt of N.

N is the size of the segment being reduced.
Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
segments.

*/




func sparseSegmentSqrtN(scope:Scope,data: Tensorflow_DataType, indices: Tensorflow_DataType, segmentIds: Tensorflow_DataType, t :type  , tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "SparseSegmentSqrtN",
        Input: [ data, indices, segmentIds],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes gradients for SparseSegmentSqrtN.

Returns tensor "output" with same shape as grad, except for dimension 0 whose
value is output_dim0.

*/




func sparseSegmentSqrtNGrad(scope:Scope,grad: Tensorflow_DataType, indices: Tensorflow_DataType, segmentIds: Tensorflow_DataType, outputDim0: Tensorflow_DataType, t :type  , tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "SparseSegmentSqrtNGrad",
        Input: [ grad, indices, segmentIds, outputDim0],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func sparseSegmentSum(scope:Scope,data: Tensorflow_DataType, indices: Tensorflow_DataType, segmentIds: Tensorflow_DataType, t :type  , tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "SparseSegmentSum",
        Input: [ data, indices, segmentIds],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func sparseSoftmax(scope:Scope,spIndices: Tensorflow_DataType, spValues: Tensorflow_DataType, spShape: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SparseSoftmax",
        Input: [ spIndices, spValues, spShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes softmax cross entropy cost and gradients to backpropagate.

Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
a matrix of label probabilities, but rather a single label per row
of features.  This label is considered to have probability 1.0 for the
given row.
Inputs are the logits, not probabilities.

*/




func sparseSoftmaxCrossEntropyWithLogits(scope:Scope,features: Tensorflow_DataType, labels: Tensorflow_DataType, t :type  , tlabels :type  )  ->(loss: tf.Output, backprop: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tlabels":tlabels)

    let opspec = tf.OpSpec(
        OpType: "SparseSoftmaxCrossEntropyWithLogits",
        Input: [ features, labels],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Returns the element-wise max of two SparseTensors.

Assumes the two SparseTensors have the same shape, i.e., no broadcasting.

*/




func sparseSparseMaximum(scope:Scope,aIndices: Tensorflow_DataType, aValues: Tensorflow_DataType, aShape: Tensorflow_DataType, bIndices: Tensorflow_DataType, bValues: Tensorflow_DataType, bShape: Tensorflow_DataType, t :type  )  ->(outputIndices: tf.Output, outputValues: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SparseSparseMaximum",
        Input: [ aIndices, aValues, aShape, bIndices, bValues, bShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Returns the element-wise min of two SparseTensors.

Assumes the two SparseTensors have the same shape, i.e., no broadcasting.

*/




func sparseSparseMinimum(scope:Scope,aIndices: Tensorflow_DataType, aValues: Tensorflow_DataType, aShape: Tensorflow_DataType, bIndices: Tensorflow_DataType, bValues: Tensorflow_DataType, bShape: Tensorflow_DataType, t :type  )  ->(outputIndices: tf.Output, outputValues: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SparseSparseMinimum",
        Input: [ aIndices, aValues, aShape, bIndices, bValues, bShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
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

*/




func sparseSplit(scope:Scope,splitDim: Tensorflow_DataType, indices: Tensorflow_DataType, values: Tensorflow_DataType, shape: Tensorflow_DataType, numSplit :uInt8  , t :type  )  ->(outputIndices: tf.Output, outputValues: tf.Output, outputShape: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_split":numSplit)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SparseSplit",
        Input: [ splitDim, indices, values, shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Adds up a `SparseTensor` and a dense `Tensor`, producing a dense `Tensor`.

This Op does not require `a_indices` be sorted in standard lexicographic order.

*/




func sparseTensorDenseAdd(scope:Scope,aIndices: Tensorflow_DataType, aValues: Tensorflow_DataType, aShape: Tensorflow_DataType, b: Tensorflow_DataType, t :type  , tindices :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)

    let opspec = tf.OpSpec(
        OpType: "SparseTensorDenseAdd",
        Input: [ aIndices, aValues, aShape, b],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func sparseTensorDenseMatMul(scope:Scope,aIndices: Tensorflow_DataType, aValues: Tensorflow_DataType, aShape: Tensorflow_DataType, b: Tensorflow_DataType, t :type  , tindices :type  , adjointA :bool  , adjointB :bool  )  ->(product: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)
    attr.append("adjoint_a":adjointA)
    attr.append("adjoint_b":adjointB)

    let opspec = tf.OpSpec(
        OpType: "SparseTensorDenseMatMul",
        Input: [ aIndices, aValues, aShape, b],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a dataset that splits a SparseTensor into elements row-wise.


*/




func sparseTensorSliceDataset(scope:Scope,indices: Tensorflow_DataType, values: Tensorflow_DataType, denseShape: Tensorflow_DataType, tvalues :type  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tvalues":tvalues)

    let opspec = tf.OpSpec(
        OpType: "SparseTensorSliceDataset",
        Input: [ indices, values, denseShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func sparseToDense(scope:Scope,sparseIndices: Tensorflow_DataType, outputShape: Tensorflow_DataType, sparseValues: Tensorflow_DataType, defaultValue: Tensorflow_DataType, validateIndices :bool  , t :type  , tindices :type  )  ->(dense: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("validate_indices":validateIndices)
    attr.append("T":t)
    attr.append("Tindices":tindices)

    let opspec = tf.OpSpec(
        OpType: "SparseToDense",
        Input: [ sparseIndices, outputShape, sparseValues, defaultValue],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func sparseToSparseSetOperation(scope:Scope,set1Indices: Tensorflow_DataType, set1Values: Tensorflow_DataType, set1Shape: Tensorflow_DataType, set2Indices: Tensorflow_DataType, set2Values: Tensorflow_DataType, set2Shape: Tensorflow_DataType, setOperation :string  , validateIndices :bool  , t :type  )  ->(resultIndices: tf.Output, resultValues: tf.Output, resultShape: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("set_operation":setOperation)
    attr.append("validate_indices":validateIndices)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SparseToSparseSetOperation",
        Input: [ set1Indices, set1Values, set1Shape, set2Indices, set2Values, set2Shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Splits a tensor into `num_split` tensors along one dimension.


*/




func split(scope:Scope,splitDim: Tensorflow_DataType, value: Tensorflow_DataType, numSplit :uInt8  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_split":numSplit)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Split",
        Input: [ splitDim, value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Splits a tensor into `num_split` tensors along one dimension.


*/




func splitV(scope:Scope,value: Tensorflow_DataType, sizeSplits: Tensorflow_DataType, splitDim: Tensorflow_DataType, numSplit :uInt8  , t :type  , tlen :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_split":numSplit)
    attr.append("T":t)
    attr.append("Tlen":tlen)

    let opspec = tf.OpSpec(
        OpType: "SplitV",
        Input: [ value, sizeSplits, splitDim],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes square root of x element-wise.

I.e., \\(y = \sqrt{x} = x^{1/2}\\).

*/




func sqrt(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Sqrt",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradient for the sqrt of `x` wrt its input.

Specifically, `grad = dy * 0.5 / y`, where `y = sqrt(x)`, and `dy`
is the corresponding input gradient.

*/




func sqrtGrad(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SqrtGrad",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes square of x element-wise.

I.e., \\(y = x * x = x^2\\).

*/




func square(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Square",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns (x - y)(x - y) element-wise.

*NOTE*: `SquaredDifference` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func squaredDifference(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "SquaredDifference",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func squeeze(scope:Scope,input: Tensorflow_DataType, t :type  , squeezeDims :[Int]  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("squeeze_dims":squeezeDims)

    let opspec = tf.OpSpec(
        OpType: "Squeeze",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A stack that produces elements in first-in last-out order.


*/




func stack(scope:Scope elemType :type  , stackName :string  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("elem_type":elemType)
    attr.append("stack_name":stackName)

    let opspec = tf.OpSpec(
        OpType: "Stack",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Delete the stack from its resource container.


*/




func stackClose(scope:Scope,handle: Tensorflow_DataType )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "StackClose",
        Input: [ handle],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Pop the element at the top of the stack.


*/




func stackPop(scope:Scope,handle: Tensorflow_DataType, elemType :type  )  ->(elem: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("elem_type":elemType)

    let opspec = tf.OpSpec(
        OpType: "StackPop",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Push an element onto the stack.


*/




func stackPush(scope:Scope,handle: Tensorflow_DataType, elem: Tensorflow_DataType, t :type  , swapMemory :bool  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("swap_memory":swapMemory)

    let opspec = tf.OpSpec(
        OpType: "StackPush",
        Input: [ handle, elem],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Stage values similar to a lightweight Enqueue.

The basic functionality of this Op is similar to a queue with many
fewer capabilities and options.  This Op is optimized for performance.

*/




func stage(scope:Scope,values: Tensorflow_DataType, dtypes :list(type)  , container :string  , sharedName :string  )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtypes":dtypes)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "Stage",
        Input: [ values],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Outputs deterministic pseudorandom values from a normal distribution.

The generated values will have mean 0 and standard deviation 1.
The outputs are a deterministic function of `shape` and `seed`.

*/




func statelessRandomNormal(scope:Scope,shape: Tensorflow_DataType, seed: Tensorflow_DataType, dtype :type  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "StatelessRandomNormal",
        Input: [ shape, seed],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Outputs deterministic pseudorandom random values from a uniform distribution.

The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.
The outputs are a deterministic function of `shape` and `seed`.

*/




func statelessRandomUniform(scope:Scope,shape: Tensorflow_DataType, seed: Tensorflow_DataType, dtype :type  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "StatelessRandomUniform",
        Input: [ shape, seed],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Outputs deterministic pseudorandom values from a truncated normal distribution.

The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked.
The outputs are a deterministic function of `shape` and `seed`.

*/




func statelessTruncatedNormal(scope:Scope,shape: Tensorflow_DataType, seed: Tensorflow_DataType, dtype :type  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "StatelessTruncatedNormal",
        Input: [ shape, seed],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func stopGradient(scope:Scope,input: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "StopGradient",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func stridedSlice(scope:Scope,input: Tensorflow_DataType, begin: Tensorflow_DataType, end: Tensorflow_DataType, strides: Tensorflow_DataType, t :type  , index :type  , beginMask :uInt8  , endMask :uInt8  , ellipsisMask :uInt8  , newAxisMask :uInt8  , shrinkAxisMask :uInt8  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Index":index)
    attr.append("begin_mask":beginMask)
    attr.append("end_mask":endMask)
    attr.append("ellipsis_mask":ellipsisMask)
    attr.append("new_axis_mask":newAxisMask)
    attr.append("shrink_axis_mask":shrinkAxisMask)

    let opspec = tf.OpSpec(
        OpType: "StridedSlice",
        Input: [ input, begin, end, strides],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Assign `value` to the sliced l-value reference of `ref`.

The values of `value` are assigned to the positions in the variable
`ref` that are selected by the slice parameters. The slice parameters
`begin, `end`, `strides`, etc. work exactly as in `StridedSlice`.
NOTE this op currently does not support broadcasting and so `value`'s
shape must be exactly the shape produced by the slice of `ref`.

*/




func stridedSliceAssign(scope:Scope,ref: Tensorflow_DataType, begin: Tensorflow_DataType, end: Tensorflow_DataType, strides: Tensorflow_DataType, value: Tensorflow_DataType, t :type  , index :type  , beginMask :uInt8  , endMask :uInt8  , ellipsisMask :uInt8  , newAxisMask :uInt8  , shrinkAxisMask :uInt8  )  ->(outputRef: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Index":index)
    attr.append("begin_mask":beginMask)
    attr.append("end_mask":endMask)
    attr.append("ellipsis_mask":ellipsisMask)
    attr.append("new_axis_mask":newAxisMask)
    attr.append("shrink_axis_mask":shrinkAxisMask)

    let opspec = tf.OpSpec(
        OpType: "StridedSliceAssign",
        Input: [ ref, begin, end, strides, value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func stridedSliceGrad(scope:Scope,shape: Tensorflow_DataType, begin: Tensorflow_DataType, end: Tensorflow_DataType, strides: Tensorflow_DataType, dy: Tensorflow_DataType, t :type  , index :type  , beginMask :uInt8  , endMask :uInt8  , ellipsisMask :uInt8  , newAxisMask :uInt8  , shrinkAxisMask :uInt8  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Index":index)
    attr.append("begin_mask":beginMask)
    attr.append("end_mask":endMask)
    attr.append("ellipsis_mask":ellipsisMask)
    attr.append("new_axis_mask":newAxisMask)
    attr.append("shrink_axis_mask":shrinkAxisMask)

    let opspec = tf.OpSpec(
        OpType: "StridedSliceGrad",
        Input: [ shape, begin, end, strides, dy],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Joins the strings in the given list of string tensors into one tensor;

with the given separator (default is an empty separator).

*/




func stringJoin(scope:Scope,inputs: Tensorflow_DataType, n :uInt8  , separator :string  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("N":n)
    attr.append("separator":separator)

    let opspec = tf.OpSpec(
        OpType: "StringJoin",
        Input: [ inputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func stringSplit(scope:Scope,input: Tensorflow_DataType, delimiter: Tensorflow_DataType )  ->(indices: tf.Output, values: tf.Output, shape: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "StringSplit",
        Input: [ input, delimiter],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Converts each string in the input Tensor to its hash mod by a number of buckets.

The hash function is deterministic on the content of the string within the
process.
Note that the hash function may change from time to time.
This functionality will be deprecated and it's recommended to use
`tf.string_to_hash_bucket_fast()` or `tf.string_to_hash_bucket_strong()`.

*/




func stringToHashBucket(scope:Scope,stringTensor: Tensorflow_DataType, numBuckets :uInt8  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_buckets":numBuckets)

    let opspec = tf.OpSpec(
        OpType: "StringToHashBucket",
        Input: [ stringTensor],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Converts each string in the input Tensor to its hash mod by a number of buckets.

The hash function is deterministic on the content of the string within the
process and will never change. However, it is not suitable for cryptography.
This function may be used when CPU time is scarce and inputs are trusted or
unimportant. There is a risk of adversaries constructing inputs that all hash
to the same bucket. To prevent this problem, use a strong hash function with
`tf.string_to_hash_bucket_strong`.

*/




func stringToHashBucketFast(scope:Scope,input: Tensorflow_DataType, numBuckets :uInt8  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_buckets":numBuckets)

    let opspec = tf.OpSpec(
        OpType: "StringToHashBucketFast",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func stringToHashBucketStrong(scope:Scope,input: Tensorflow_DataType, numBuckets :uInt8  , key :[Int]  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_buckets":numBuckets)
    attr.append("key":key)

    let opspec = tf.OpSpec(
        OpType: "StringToHashBucketStrong",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Converts each string in the input Tensor to the specified numeric type.

(Note that int32 overflow results in an error while float overflow
results in a rounded value.)

*/




func stringToNumber(scope:Scope,stringTensor: Tensorflow_DataType, outType :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("out_type":outType)

    let opspec = tf.OpSpec(
        OpType: "StringToNumber",
        Input: [ stringTensor],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns x - y element-wise.

*NOTE*: `Sub` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func sub(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Sub",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func substr(scope:Scope,input: Tensorflow_DataType, pos: Tensorflow_DataType, len: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Substr",
        Input: [ input, pos, len],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the sum of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

*/




func sum(scope:Scope,input: Tensorflow_DataType, reductionIndices: Tensorflow_DataType, keepDims :bool  , t :type  , tidx :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("keep_dims":keepDims)
    attr.append("T":t)
    attr.append("Tidx":tidx)

    let opspec = tf.OpSpec(
        OpType: "Sum",
        Input: [ input, reductionIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func svd(scope:Scope,input: Tensorflow_DataType, computeUv :bool  , fullMatrices :bool  , t :type  )  ->(s: tf.Output, u: tf.Output, v: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("compute_uv":computeUv)
    attr.append("full_matrices":fullMatrices)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Svd",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Forwards `data` to the output port determined by `pred`.

If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
the data goes to `output_false`.
See also `RefSwitch` and `Merge`.

*/




func switch(scope:Scope,data: Tensorflow_DataType, pred: Tensorflow_DataType, t :type  )  ->(outputFalse: tf.Output, outputTrue: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Switch",
        Input: [ data, pred],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Computes the gradient function for function f via backpropagation.


*/




func symbolicGradient(scope:Scope,input: Tensorflow_DataType, tin :list(type)  , tout :list(type)  , f :func  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Tin":tin)
    attr.append("Tout":tout)
    attr.append("f":f)

    let opspec = tf.OpSpec(
        OpType: "SymbolicGradient",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a dataset that emits the records from one or more TFRecord files.


*/




func tfRecordDataset(scope:Scope,filenames: Tensorflow_DataType, compressionType: Tensorflow_DataType )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "TFRecordDataset",
        Input: [ filenames, compressionType],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A Reader that outputs the records from a TensorFlow Records file.


*/




func tfRecordReader(scope:Scope container :string  , sharedName :string  , compressionType :string  )  ->(readerHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("container":container)
    attr.append("shared_name":sharedName)
    attr.append("compression_type":compressionType)

    let opspec = tf.OpSpec(
        OpType: "TFRecordReader",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A Reader that outputs the records from a TensorFlow Records file.


*/




func tfRecordReaderV2(scope:Scope container :string  , sharedName :string  , compressionType :string  )  ->(readerHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("container":container)
    attr.append("shared_name":sharedName)
    attr.append("compression_type":compressionType)

    let opspec = tf.OpSpec(
        OpType: "TFRecordReaderV2",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a dataset that contains `count` elements from the `input_dataset`.


*/




func takeDataset(scope:Scope,inputDataset: Tensorflow_DataType, count: Tensorflow_DataType, outputTypes :list(type)  , outputShapes :[Shape]  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("output_types":outputTypes)
    attr.append("output_shapes":outputShapes)

    let opspec = tf.OpSpec(
        OpType: "TakeDataset",
        Input: [ inputDataset, count],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func takeManySparseFromTensorsMap(scope:Scope,sparseHandles: Tensorflow_DataType, dtype :type  , container :string  , sharedName :string  )  ->(sparseIndices: tf.Output, sparseValues: tf.Output, sparseShape: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "TakeManySparseFromTensorsMap",
        Input: [ sparseHandles],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Computes tan of x element-wise.


*/




func tan(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Tan",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes hyperbolic tangent of `x` element-wise.


*/




func tanh(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Tanh",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Computes the gradient for the tanh of `x` wrt its input.

Specifically, `grad = dy * (1 - y*y)`, where `y = tanh(x)`, and `dy`
is the corresponding input gradient.

*/




func tanhGrad(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "TanhGrad",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func temporaryVariable(scope:Scope shape :shape  , dtype :type  , varName :string  )  ->(ref: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("shape":shape)
    attr.append("dtype":dtype)
    attr.append("var_name":varName)

    let opspec = tf.OpSpec(
        OpType: "TemporaryVariable",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func tensorArray(scope:Scope,size: Tensorflow_DataType, dtype :type  , dynamicSize :bool  , clearAfterRead :bool  , tensorArrayName :string  , elementShape :shape  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("dynamic_size":dynamicSize)
    attr.append("clear_after_read":clearAfterRead)
    attr.append("tensor_array_name":tensorArrayName)
    attr.append("element_shape":elementShape)

    let opspec = tf.OpSpec(
        OpType: "TensorArray",
        Input: [ size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func tensorArrayClose(scope:Scope,handle: Tensorflow_DataType )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "TensorArrayClose",
        Input: [ handle],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Deprecated. Use TensorArrayCloseV3


*/




func tensorArrayCloseV2(scope:Scope,handle: Tensorflow_DataType )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "TensorArrayCloseV2",
        Input: [ handle],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Delete the TensorArray from its resource container.

This enables the user to close and release the resource in the middle
of a step/run.

*/




func tensorArrayCloseV3(scope:Scope,handle: Tensorflow_DataType )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "TensorArrayCloseV3",
        Input: [ handle],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*


*/




func tensorArrayConcat(scope:Scope,handle: Tensorflow_DataType, flowIn: Tensorflow_DataType, dtype :type  , elementShapeExcept0 :shape  )  ->(value: tf.Output, lengths: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("element_shape_except0":elementShapeExcept0)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayConcat",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Deprecated. Use TensorArrayConcatV3


*/




func tensorArrayConcatV2(scope:Scope,handle: Tensorflow_DataType, flowIn: Tensorflow_DataType, dtype :type  , elementShapeExcept0 :shape  )  ->(value: tf.Output, lengths: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("element_shape_except0":elementShapeExcept0)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayConcatV2",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
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

*/




func tensorArrayConcatV3(scope:Scope,handle: Tensorflow_DataType, flowIn: Tensorflow_DataType, dtype :type  , elementShapeExcept0 :shape  )  ->(value: tf.Output, lengths: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("element_shape_except0":elementShapeExcept0)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayConcatV3",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*


*/




func tensorArrayGather(scope:Scope,handle: Tensorflow_DataType, indices: Tensorflow_DataType, flowIn: Tensorflow_DataType, dtype :type  , elementShape :shape  )  ->(value: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("element_shape":elementShape)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayGather",
        Input: [ handle, indices, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Deprecated. Use TensorArrayGatherV3


*/




func tensorArrayGatherV2(scope:Scope,handle: Tensorflow_DataType, indices: Tensorflow_DataType, flowIn: Tensorflow_DataType, dtype :type  , elementShape :shape  )  ->(value: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("element_shape":elementShape)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayGatherV2",
        Input: [ handle, indices, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Gather specific elements from the TensorArray into output `value`.

All elements selected by `indices` must have the same shape.

*/




func tensorArrayGatherV3(scope:Scope,handle: Tensorflow_DataType, indices: Tensorflow_DataType, flowIn: Tensorflow_DataType, dtype :type  , elementShape :shape  )  ->(value: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("element_shape":elementShape)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayGatherV3",
        Input: [ handle, indices, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func tensorArrayGrad(scope:Scope,handle: Tensorflow_DataType, flowIn: Tensorflow_DataType, source :string  )  ->(gradHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("source":source)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayGrad",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Deprecated. Use TensorArrayGradV3


*/




func tensorArrayGradV2(scope:Scope,handle: Tensorflow_DataType, flowIn: Tensorflow_DataType, source :string  )  ->(gradHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("source":source)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayGradV2",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func tensorArrayGradV3(scope:Scope,handle: Tensorflow_DataType, flowIn: Tensorflow_DataType, source :string  )  ->(gradHandle: tf.Output, flowOut: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("source":source)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayGradV3",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*


*/




func tensorArrayPack(scope:Scope,handle: Tensorflow_DataType, flowIn: Tensorflow_DataType, dtype :type  , elementShape :shape  )  ->(value: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("element_shape":elementShape)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayPack",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func tensorArrayRead(scope:Scope,handle: Tensorflow_DataType, index: Tensorflow_DataType, flowIn: Tensorflow_DataType, dtype :type  )  ->(value: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayRead",
        Input: [ handle, index, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Deprecated. Use TensorArrayReadV3


*/




func tensorArrayReadV2(scope:Scope,handle: Tensorflow_DataType, index: Tensorflow_DataType, flowIn: Tensorflow_DataType, dtype :type  )  ->(value: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayReadV2",
        Input: [ handle, index, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Read an element from the TensorArray into output `value`.


*/




func tensorArrayReadV3(scope:Scope,handle: Tensorflow_DataType, index: Tensorflow_DataType, flowIn: Tensorflow_DataType, dtype :type  )  ->(value: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayReadV3",
        Input: [ handle, index, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func tensorArrayScatter(scope:Scope,handle: Tensorflow_DataType, indices: Tensorflow_DataType, value: Tensorflow_DataType, flowIn: Tensorflow_DataType, t :type  )  ->(flowOut: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayScatter",
        Input: [ handle, indices, value, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Deprecated. Use TensorArrayScatterV3


*/




func tensorArrayScatterV2(scope:Scope,handle: Tensorflow_DataType, indices: Tensorflow_DataType, value: Tensorflow_DataType, flowIn: Tensorflow_DataType, t :type  )  ->(flowOut: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayScatterV2",
        Input: [ handle, indices, value, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Scatter the data from the input value into specific TensorArray elements.

`indices` must be a vector, its length must match the first dim of `value`.

*/




func tensorArrayScatterV3(scope:Scope,handle: Tensorflow_DataType, indices: Tensorflow_DataType, value: Tensorflow_DataType, flowIn: Tensorflow_DataType, t :type  )  ->(flowOut: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayScatterV3",
        Input: [ handle, indices, value, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func tensorArraySize(scope:Scope,handle: Tensorflow_DataType, flowIn: Tensorflow_DataType )  ->(size: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "TensorArraySize",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Deprecated. Use TensorArraySizeV3


*/




func tensorArraySizeV2(scope:Scope,handle: Tensorflow_DataType, flowIn: Tensorflow_DataType )  ->(size: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "TensorArraySizeV2",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Get the current size of the TensorArray.


*/




func tensorArraySizeV3(scope:Scope,handle: Tensorflow_DataType, flowIn: Tensorflow_DataType )  ->(size: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "TensorArraySizeV3",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func tensorArraySplit(scope:Scope,handle: Tensorflow_DataType, value: Tensorflow_DataType, lengths: Tensorflow_DataType, flowIn: Tensorflow_DataType, t :type  )  ->(flowOut: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "TensorArraySplit",
        Input: [ handle, value, lengths, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Deprecated. Use TensorArraySplitV3


*/




func tensorArraySplitV2(scope:Scope,handle: Tensorflow_DataType, value: Tensorflow_DataType, lengths: Tensorflow_DataType, flowIn: Tensorflow_DataType, t :type  )  ->(flowOut: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "TensorArraySplitV2",
        Input: [ handle, value, lengths, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func tensorArraySplitV3(scope:Scope,handle: Tensorflow_DataType, value: Tensorflow_DataType, lengths: Tensorflow_DataType, flowIn: Tensorflow_DataType, t :type  )  ->(flowOut: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "TensorArraySplitV3",
        Input: [ handle, value, lengths, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*


*/




func tensorArrayUnpack(scope:Scope,handle: Tensorflow_DataType, value: Tensorflow_DataType, flowIn: Tensorflow_DataType, t :type  )  ->(flowOut: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayUnpack",
        Input: [ handle, value, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Deprecated. Use TensorArrayV3


*/




func tensorArrayV2(scope:Scope,size: Tensorflow_DataType, dtype :type  , elementShape :shape  , dynamicSize :bool  , clearAfterRead :bool  , tensorArrayName :string  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("element_shape":elementShape)
    attr.append("dynamic_size":dynamicSize)
    attr.append("clear_after_read":clearAfterRead)
    attr.append("tensor_array_name":tensorArrayName)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayV2",
        Input: [ size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
An array of Tensors of given size.

Write data via Write and read via Read or Pack.

*/




func tensorArrayV3(scope:Scope,size: Tensorflow_DataType, dtype :type  , elementShape :shape  , dynamicSize :bool  , clearAfterRead :bool  , tensorArrayName :string  )  ->(handle: tf.Output, flow: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtype":dtype)
    attr.append("element_shape":elementShape)
    attr.append("dynamic_size":dynamicSize)
    attr.append("clear_after_read":clearAfterRead)
    attr.append("tensor_array_name":tensorArrayName)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayV3",
        Input: [ size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*


*/




func tensorArrayWrite(scope:Scope,handle: Tensorflow_DataType, index: Tensorflow_DataType, value: Tensorflow_DataType, flowIn: Tensorflow_DataType, t :type  )  ->(flowOut: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayWrite",
        Input: [ handle, index, value, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Deprecated. Use TensorArrayGradV3


*/




func tensorArrayWriteV2(scope:Scope,handle: Tensorflow_DataType, index: Tensorflow_DataType, value: Tensorflow_DataType, flowIn: Tensorflow_DataType, t :type  )  ->(flowOut: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayWriteV2",
        Input: [ handle, index, value, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Push an element onto the tensor_array.


*/




func tensorArrayWriteV3(scope:Scope,handle: Tensorflow_DataType, index: Tensorflow_DataType, value: Tensorflow_DataType, flowIn: Tensorflow_DataType, t :type  )  ->(flowOut: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "TensorArrayWriteV3",
        Input: [ handle, index, value, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a dataset that emits `components` as a tuple of tensors once.


*/




func tensorDataset(scope:Scope,components: Tensorflow_DataType, toutputTypes :list(type)  , outputShapes :[Shape]  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Toutput_types":toutputTypes)
    attr.append("output_shapes":outputShapes)

    let opspec = tf.OpSpec(
        OpType: "TensorDataset",
        Input: [ components],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a dataset that emits each dim-0 slice of `components` once.


*/




func tensorSliceDataset(scope:Scope,components: Tensorflow_DataType, toutputTypes :list(type)  , outputShapes :[Shape]  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("Toutput_types":toutputTypes)
    attr.append("output_shapes":outputShapes)

    let opspec = tf.OpSpec(
        OpType: "TensorSliceDataset",
        Input: [ components],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Outputs a `Summary` protocol buffer with a tensor.


*/




func tensorSummary(scope:Scope,tensor: Tensorflow_DataType, t :type  , description :string  , labels :list(string)  , displayName :string  )  ->(summary: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("description":description)
    attr.append("labels":labels)
    attr.append("display_name":displayName)

    let opspec = tf.OpSpec(
        OpType: "TensorSummary",
        Input: [ tensor],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a dataset that emits the lines of one or more text files.


*/




func textLineDataset(scope:Scope,filenames: Tensorflow_DataType )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "TextLineDataset",
        Input: [ filenames],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A Reader that outputs the lines of a file delimited by '\n'.


*/




func textLineReader(scope:Scope skipHeaderLines :uInt8  , container :string  , sharedName :string  )  ->(readerHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("skip_header_lines":skipHeaderLines)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "TextLineReader",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A Reader that outputs the lines of a file delimited by '\n'.


*/




func textLineReaderV2(scope:Scope skipHeaderLines :uInt8  , container :string  , sharedName :string  )  ->(readerHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("skip_header_lines":skipHeaderLines)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "TextLineReaderV2",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func threadUnsafeUnigramCandidateSampler(scope:Scope,trueClasses: Tensorflow_DataType, numTrue :uInt8  , numSampled :uInt8  , unique :bool  , rangeMax :uInt8  , seed :uInt8  , seed2 :uInt8  )  ->(sampledCandidates: tf.Output, trueExpectedCount: tf.Output, sampledExpectedCount: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_true":numTrue)
    attr.append("num_sampled":numSampled)
    attr.append("unique":unique)
    attr.append("range_max":rangeMax)
    attr.append("seed":seed)
    attr.append("seed2":seed2)

    let opspec = tf.OpSpec(
        OpType: "ThreadUnsafeUnigramCandidateSampler",
        Input: [ trueClasses],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
}

/*
Constructs a tensor by tiling a given tensor.

This operation creates a new tensor by replicating `input` `multiples` times.
The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
and the values of `input` are replicated `multiples[i]` times along the 'i'th
dimension. For example, tiling `[a b c d]` by `[2]` produces
`[a b c d a b c d]`.

*/




func tile(scope:Scope,input: Tensorflow_DataType, multiples: Tensorflow_DataType, t :type  , tmultiples :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tmultiples":tmultiples)

    let opspec = tf.OpSpec(
        OpType: "Tile",
        Input: [ input, multiples],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns the gradient of `Tile`.

Since `Tile` takes an input and repeats the input `multiples` times
along each dimension, `TileGrad` takes in `multiples` and aggregates
each repeated tile of `input` into `output`.

*/




func tileGrad(scope:Scope,input: Tensorflow_DataType, multiples: Tensorflow_DataType, t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "TileGrad",
        Input: [ input, multiples],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func topK(scope:Scope,input: Tensorflow_DataType, k :uInt8  , sorted :bool  , t :type  )  ->(values: tf.Output, indices: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("k":k)
    attr.append("sorted":sorted)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "TopK",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
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

*/




func topKV2(scope:Scope,input: Tensorflow_DataType, k: Tensorflow_DataType, sorted :bool  , t :type  )  ->(values: tf.Output, indices: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("sorted":sorted)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "TopKV2",
        Input: [ input, k],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
}

/*
Shuffle dimensions of x according to a permutation.

The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
  `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`

*/




func transpose(scope:Scope,x: Tensorflow_DataType, perm: Tensorflow_DataType, t :type  , tperm :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tperm":tperm)

    let opspec = tf.OpSpec(
        OpType: "Transpose",
        Input: [ x, perm],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns x / y element-wise for integer types.

Truncation designates that negative numbers will round fractional quantities
toward zero. I.e. -7 / 5 = 1. This matches C semantics but it is different
than Python semantics. See `FloorDiv` for a division function that matches
Python Semantics.
*NOTE*: `TruncateDiv` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func truncateDiv(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "TruncateDiv",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Returns element-wise remainder of division. This emulates C semantics in that

the result here is consistent with a truncating divide. E.g. `truncate(x / y) *
y + truncate_mod(x, y) = x`.
*NOTE*: `TruncateMod` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/




func truncateMod(scope:Scope,x: Tensorflow_DataType, y: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "TruncateMod",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Outputs random values from a truncated normal distribution.

The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked.

*/




func truncatedNormal(scope:Scope,shape: Tensorflow_DataType, seed :uInt8  , seed2 :uInt8  , dtype :type  , t :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("seed":seed)
    attr.append("seed2":seed2)
    attr.append("dtype":dtype)
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "TruncatedNormal",
        Input: [ shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func uniformCandidateSampler(scope:Scope,trueClasses: Tensorflow_DataType, numTrue :uInt8  , numSampled :uInt8  , unique :bool  , rangeMax :uInt8  , seed :uInt8  , seed2 :uInt8  )  ->(sampledCandidates: tf.Output, trueExpectedCount: tf.Output, sampledExpectedCount: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num_true":numTrue)
    attr.append("num_sampled":numSampled)
    attr.append("unique":unique)
    attr.append("range_max":rangeMax)
    attr.append("seed":seed)
    attr.append("seed2":seed2)

    let opspec = tf.OpSpec(
        OpType: "UniformCandidateSampler",
        Input: [ trueClasses],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
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

*/




func unique(scope:Scope,x: Tensorflow_DataType, t :type  , outIdx :type  )  ->(y: tf.Output, idx: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("out_idx":outIdx)

    let opspec = tf.OpSpec(
        OpType: "Unique",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1))
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

*/




func uniqueWithCounts(scope:Scope,x: Tensorflow_DataType, t :type  , outIdx :type  )  ->(y: tf.Output, idx: tf.Output, count: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("out_idx":outIdx)

    let opspec = tf.OpSpec(
        OpType: "UniqueWithCounts",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1),op.Output(2 - 1),op.Output(3 - 1))
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

*/




func unpack(scope:Scope,value: Tensorflow_DataType, num :uInt8  , t :type  , axis :uInt8  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("num":num)
    attr.append("T":t)
    attr.append("axis":axis)

    let opspec = tf.OpSpec(
        OpType: "Unpack",
        Input: [ value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func unsortedSegmentMax(scope:Scope,data: Tensorflow_DataType, segmentIds: Tensorflow_DataType, numSegments: Tensorflow_DataType, t :type  , tindices :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)

    let opspec = tf.OpSpec(
        OpType: "UnsortedSegmentMax",
        Input: [ data, segmentIds, numSegments],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func unsortedSegmentSum(scope:Scope,data: Tensorflow_DataType, segmentIds: Tensorflow_DataType, numSegments: Tensorflow_DataType, t :type  , tindices :type  )  ->(output: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)
    attr.append("Tindices":tindices)

    let opspec = tf.OpSpec(
        OpType: "UnsortedSegmentSum",
        Input: [ data, segmentIds, numSegments],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Op is similar to a lightweight Dequeue.

The basic funtionality is similar to dequeue with many fewer
capabilities and options.  This Op is optimized for performance.

*/




func unstage(scope:Scope dtypes :list(type)  , container :string  , sharedName :string  )  ->(values: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("dtypes":dtypes)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "Unstage",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Use VariableV2 instead.


*/




func variable(scope:Scope shape :shape  , dtype :type  , container :string  , sharedName :string  )  ->(ref: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("shape":shape)
    attr.append("dtype":dtype)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "Variable",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Holds state in the form of a tensor that persists across steps.

Outputs a ref to the tensor state so it may be read or modified.
TODO(zhifengc/mrry): Adds a pointer to a more detail document
about sharing states in tensorflow.

*/




func variableV2(scope:Scope shape :shape  , dtype :type  , container :string  , sharedName :string  )  ->(ref: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("shape":shape)
    attr.append("dtype":dtype)
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "VariableV2",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
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

*/




func where(scope:Scope,input: Tensorflow_DataType )  ->(index: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "Where",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A Reader that outputs the entire contents of a file as a value.

To use, enqueue filenames in a Queue.  The output of ReaderRead will
be a filename (key) and the contents of that file (value).

*/




func wholeFileReader(scope:Scope container :string  , sharedName :string  )  ->(readerHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "WholeFileReader",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
A Reader that outputs the entire contents of a file as a value.

To use, enqueue filenames in a Queue.  The output of ReaderRead will
be a filename (key) and the contents of that file (value).

*/




func wholeFileReaderV2(scope:Scope container :string  , sharedName :string  )  ->(readerHandle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("container":container)
    attr.append("shared_name":sharedName)

    let opspec = tf.OpSpec(
        OpType: "WholeFileReaderV2",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Writes contents to the file at input filename. Creates file if not existing.


*/




func writeFile(scope:Scope,filename: Tensorflow_DataType, contents: Tensorflow_DataType )  ->(o:tf.Operation){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]

    let opspec = tf.OpSpec(
        OpType: "WriteFile",
        Input: [ filename, contents],
        Attrs: attrs
    )
    return scope.AddOperation(opspec)
}

/*
Returns a tensor of zeros with the same shape and type as x.


*/




func zerosLike(scope:Scope,x: Tensorflow_DataType, t :type  )  ->(y: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "ZerosLike",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Compute the Hurwitz zeta function \\(\zeta(x, q)\\).

The Hurwitz zeta function is defined as:
\\(\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}\\)

*/




func zeta(scope:Scope,x: Tensorflow_DataType, q: Tensorflow_DataType, t :type  )  ->(z: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("T":t)

    let opspec = tf.OpSpec(
        OpType: "Zeta",
        Input: [ x, q],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

/*
Creates a dataset that zips together `input_datasets`.


*/




func zipDataset(scope:Scope,inputDatasets: Tensorflow_DataType, outputTypes :list(type)  , outputShapes :[Shape]  , n :uInt8  )  ->(handle: tf.Output){
    if scope.error() != nil {
        return
    }

    var attrs:Dictionary<String,Tensorflow_AttrValue> = [:]
    attr.append("output_types":outputTypes)
    attr.append("output_shapes":outputShapes)
    attr.append("N":n)

    let opspec = tf.OpSpec(
        OpType: "ZipDataset",
        Input: [ inputDatasets],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op.Output(1 - 1))
}

