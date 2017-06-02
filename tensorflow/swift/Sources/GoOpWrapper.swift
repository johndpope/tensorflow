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
Raise a exception to abort the process when called.

If exit_without_error is true, the process will exit normally,
// otherwise it will exit with a SIGABORT signal.
// 
// Returns nothing but an exception.

*/








func abort( scope:Scope, errorMsg :String  , exitWithoutError :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["error_msg"] = errorMsg
    attrs["exit_without_error"] = exitWithoutError

    let opspec = OpSpec(
        OpType: "Abort",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Computes the absolute value of a tensor.

Given a tensor `x`, this operation returns a tensor containing the absolute
// value of each element in `x`. For example, if x is an input element and y is
// an output element, this operation computes \\(y = |x|\\).

*/








func abs( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Abs",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Applies a gradient to a given accumulator.

Does not add if local_step is lesser than the accumulator's global_step.

*/








func accumulatorApplyGradient( scope:Scope,handle: Output, localStep: Output, gradient: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "AccumulatorApplyGradient",
        Name: "Type",
        Input: [ handle, localStep, gradient],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Returns the number of gradients aggregated in the given accumulators.


*/








func accumulatorNumAccumulated( scope:Scope,handle: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "AccumulatorNumAccumulated",
        Name: "Type",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Updates the accumulator with a new value for global_step.

Logs warning if the accumulator's value is already higher than
// new_global_step.

*/








func accumulatorSetGlobalStep( scope:Scope,handle: Output, newGlobalStep: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "AccumulatorSetGlobalStep",
        Name: "Type",
        Input: [ handle, newGlobalStep],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Extracts the average gradient in the given ConditionalAccumulator.

The op blocks until sufficient (i.e., more than num_required)
// gradients have been accumulated.  If the accumulator has already
// aggregated more than num_required gradients, it returns the average of
// the accumulated gradients.  Also automatically increments the recorded
// global_step in the accumulator by 1, and resets the aggregate to 0.

*/








func accumulatorTakeGradient( scope:Scope,handle: Output, numRequired: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "AccumulatorTakeGradient",
        Name: "Type",
        Input: [ handle, numRequired],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes acos of x element-wise.


*/








func acos( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Acos",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns x + y element-wise.

 * NOTE * : `Add` supports broadcasting. `AddN` does not. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func add( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

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
Add an `N`-minibatch `SparseTensor` to a `SparseTensorsMap`, return `N` handles.

A `SparseTensor` of rank `R` is represented by three tensors: `sparse_indices`,
// `sparse_values`, and `sparse_shape`, where
// 
// ```sparse_indices.shape[1] == sparse_shape.shape[0] == R```
// 
// An `N`-minibatch of `SparseTensor` objects is represented as a `SparseTensor`
// having a first `sparse_indices` column taking values between `[0, N)`, where
// the minibatch size `N == sparse_shape[0]`.
// 
// The input `SparseTensor` must have rank `R` greater than 1, and the first
// dimension is treated as the minibatch dimension.  Elements of the `SparseTensor`
// must be sorted in increasing order of this first dimension.  The stored
// `SparseTensor` objects pointed to by each row of the output `sparse_handles`
// will have rank `R-1`.
// 
// The `SparseTensor` values can then be read out as part of a minibatch by passing
// the given keys as vector elements to `TakeManySparseFromTensorsMap`.  To ensure
// the correct `SparseTensorsMap` is accessed, ensure that the same
// `container` and `shared_name` are passed to that Op.  If no `shared_name`
// is provided here, instead use the  * name *  of the Operation created by calling
// `AddManySparseToTensorsMap` as the `shared_name` passed to
// `TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.

*/








func addManySparseToTensorsMap( scope:Scope,sparseIndices: Output, sparseValues: Output, sparseShape: Output, container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "AddManySparseToTensorsMap",
        Name: "Type",
        Input: [ sparseIndices, sparseValues, sparseShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Add all input tensors element wise.


*/








func addN( scope:Scope,inputs: Output, n :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["N"] = n

    let opspec = OpSpec(
        OpType: "AddN",
        Name: "Type",
        Input: [ inputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Add a `SparseTensor` to a `SparseTensorsMap` return its handle.

A `SparseTensor` is represented by three tensors: `sparse_indices`,
// `sparse_values`, and `sparse_shape`.
// 
// This operator takes the given `SparseTensor` and adds it to a container
// object (a `SparseTensorsMap`).  A unique key within this container is generated
// in the form of an `int64`, and this is the value that is returned.
// 
// The `SparseTensor` can then be read out as part of a minibatch by passing
// the key as a vector element to `TakeManySparseFromTensorsMap`.  To ensure
// the correct `SparseTensorsMap` is accessed, ensure that the same
// `container` and `shared_name` are passed to that Op.  If no `shared_name`
// is provided here, instead use the  * name *  of the Operation created by calling
// `AddSparseToTensorsMap` as the `shared_name` passed to
// `TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.

*/








func addSparseToTensorsMap( scope:Scope,sparseIndices: Output, sparseValues: Output, sparseShape: Output, container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "AddSparseToTensorsMap",
        Name: "Type",
        Input: [ sparseIndices, sparseValues, sparseShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Deprecated. Disallowed in GraphDef version >= 2.


*/








func adjustContrast( scope:Scope,images: Output, contrastFactor: Output, minValue: Output, maxValue: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "AdjustContrast",
        Name: "Type",
        Input: [ images, contrastFactor, minValue, maxValue],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Adjust the contrast of one or more images.

`images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
// interpreted as `[height, width, channels]`.  The other dimensions only
// represent a collection of images, such as `[batch, height, width, channels].`
// 
// Contrast is adjusted independently for each channel of each image.
// 
// For each channel, the Op first computes the mean of the image pixels in the
// channel and then adjusts each component of each pixel to
// `(x - mean)  *  contrast_factor + mean`.

*/








func adjustContrastv2( scope:Scope,images: Output, contrastFactor: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "AdjustContrastv2",
        Name: "Type",
        Input: [ images, contrastFactor],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Adjust the hue of one or more images.

`images` is a tensor of at least 3 dimensions.  The last dimension is
// interpretted as channels, and must be three.
// 
// The input image is considered in the RGB colorspace. Conceptually, the RGB
// colors are first mapped into HSV. A delta is then applied all the hue values,
// and then remapped back to RGB colorspace.

*/








func adjustHue( scope:Scope,images: Output, delta: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "AdjustHue",
        Name: "Type",
        Input: [ images, delta],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Adjust the saturation of one or more images.

`images` is a tensor of at least 3 dimensions.  The last dimension is
// interpretted as channels, and must be three.
// 
// The input image is considered in the RGB colorspace. Conceptually, the RGB
// colors are first mapped into HSV. A scale is then applied all the saturation
// values, and then remapped back to RGB colorspace.

*/








func adjustSaturation( scope:Scope,images: Output, scale: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "AdjustSaturation",
        Name: "Type",
        Input: [ images, scale],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the "logical and" of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.

*/








func all( scope:Scope,input: Output, reductionIndices: Output, keepDims :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["keep_dims"] = keepDims

    let opspec = OpSpec(
        OpType: "All",
        Name: "Type",
        Input: [ input, reductionIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Generates labels for candidate sampling with a learned unigram distribution.

See explanations of candidate sampling and the data formats at
// go/candidate-sampling.
// 
// For each batch, this op picks a single set of sampled candidate labels.
// 
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.

*/








func allCandidateSampler( scope:Scope,trueClasses: Output, numTrue :UInt8  , numSampled :UInt8  , unique :Bool  , seed :UInt8  , seed2 :UInt8  )  ->(sampledCandidates: Output?, trueExpectedCount: Output?, sampledExpectedCount: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_true"] = numTrue
    attrs["num_sampled"] = numSampled
    attrs["unique"] = unique
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "AllCandidateSampler",
        Name: "Type",
        Input: [ trueClasses],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Computes the "logical or" of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.

*/








func any( scope:Scope,input: Output, reductionIndices: Output, keepDims :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["keep_dims"] = keepDims

    let opspec = OpSpec(
        OpType: "Any",
        Name: "Type",
        Input: [ input, reductionIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update '*var' according to the adadelta scheme.

accum = rho()  *  accum + (1 - rho())  *  grad.square();
// update = (update_accum + epsilon).sqrt()  *  (accum + epsilon()).rsqrt()  *  grad;
// update_accum = rho()  *  update_accum + (1 - rho())  *  update.square();
// var -= update;

*/








func applyAdadelta( scope:Scope,`var`: Output, accum: Output, accumUpdate: Output, lr: Output, rho: Output, epsilon: Output, grad: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ApplyAdadelta",
        Name: "Type",
        Input: [ `var`, accum, accumUpdate, lr, rho, epsilon, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update '*var' according to the adagrad scheme.

accum += grad  *  grad
// var -= lr  *  grad  *  (1 / sqrt(accum))

*/








func applyAdagrad( scope:Scope,`var`: Output, accum: Output, lr: Output, grad: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ApplyAdagrad",
        Name: "Type",
        Input: [ `var`, accum, lr, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update '*var' according to the proximal adagrad scheme.


*/








func applyAdagradDA( scope:Scope,`var`: Output, gradientAccumulator: Output, gradientSquaredAccumulator: Output, grad: Output, lr: Output, l1: Output, l2: Output, globalStep: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ApplyAdagradDA",
        Name: "Type",
        Input: [ `var`, gradientAccumulator, gradientSquaredAccumulator, grad, lr, l1, l2, globalStep],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update '*var' according to the Adam algorithm.

lr_t <- learning_rate  *  sqrt(1 - beta2// ^t) / (1 - beta1// ^t)
// m_t <- beta1  *  m_{t-1} + (1 - beta1)  *  g_t
// v_t <- beta2  *  v_{t-1} + (1 - beta2)  *  g_t  *  g_t
// variable <- variable - lr_t  *  m_t / (sqrt(v_t) + epsilon)

*/








func applyAdam( scope:Scope,`var`: Output, m: Output, v: Output, beta1Power: Output, beta2Power: Output, lr: Output, beta1: Output, beta2: Output, epsilon: Output, grad: Output, useLocking :Bool  , useNesterov :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking
    attrs["use_nesterov"] = useNesterov

    let opspec = OpSpec(
        OpType: "ApplyAdam",
        Name: "Type",
        Input: [ `var`, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update '*var' according to the centered RMSProp algorithm.

The centered RMSProp algorithm uses an estimate of the centered second moment
// (i.e., the variance) for normalization, as opposed to regular RMSProp, which
// uses the (uncentered) second moment. This often helps with training, but is
// slightly more expensive in terms of computation and memory.
// 
// Note that in dense implementation of this algorithm, mg, ms, and mom will
// update even if the grad is zero, but in this sparse implementation, mg, ms,
// and mom will not update in iterations during which the grad is zero.
// 
// mean_square = decay  *  mean_square + (1-decay)  *  gradient  *  *  2
// mean_grad = decay  *  mean_grad + (1-decay)  *  gradient
// 
// Delta = learning_rate  *  gradient / sqrt(mean_square + epsilon - mean_grad  *  *  2)
// 
// mg <- rho  *  mg_{t-1} + (1-rho)  *  grad
// ms <- rho  *  ms_{t-1} + (1-rho)  *  grad  *  grad
// mom <- momentum  *  mom_{t-1} + lr  *  grad / sqrt(ms - mg  *  mg + epsilon)
// var <- var - mom

*/








func applyCenteredRMSProp( scope:Scope,`var`: Output, mg: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ApplyCenteredRMSProp",
        Name: "Type",
        Input: [ `var`, mg, ms, mom, lr, rho, momentum, epsilon, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update '*var' according to the Ftrl-proximal scheme.

accum_new = accum + grad  *  grad
// linear += grad + (accum_new// ^(-lr_power) - accum// ^(-lr_power)) / lr  *  var
// quadratic = 1.0 / (accum_new// ^(lr_power)  *  lr) + 2  *  l2
// var = (sign(linear)  *  l1 - linear) / quadratic if |linear| > l1 else 0.0
// accum = accum_new

*/








func applyFtrl( scope:Scope,`var`: Output, accum: Output, linear: Output, grad: Output, lr: Output, l1: Output, l2: Output, lrPower: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ApplyFtrl",
        Name: "Type",
        Input: [ `var`, accum, linear, grad, lr, l1, l2, lrPower],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update '*var' by subtracting 'alpha' * 'delta' from it.


*/








func applyGradientDescent( scope:Scope,`var`: Output, alpha: Output, delta: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ApplyGradientDescent",
        Name: "Type",
        Input: [ `var`, alpha, delta],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update '*var' according to the momentum scheme. Set use_nesterov = True if you

want to use Nesterov momentum.
// 
// accum = accum  *  momentum + grad
// var -= lr  *  accum

*/








func applyMomentum( scope:Scope,`var`: Output, accum: Output, lr: Output, grad: Output, momentum: Output, useLocking :Bool  , useNesterov :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking
    attrs["use_nesterov"] = useNesterov

    let opspec = OpSpec(
        OpType: "ApplyMomentum",
        Name: "Type",
        Input: [ `var`, accum, lr, grad, momentum],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.

accum += grad  *  grad
// prox_v = var - lr  *  grad  *  (1 / sqrt(accum))
// var = sign(prox_v)/(1+lr * l2)  *  max{|prox_v|-lr * l1,0}

*/








func applyProximalAdagrad( scope:Scope,`var`: Output, accum: Output, lr: Output, l1: Output, l2: Output, grad: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ApplyProximalAdagrad",
        Name: "Type",
        Input: [ `var`, accum, lr, l1, l2, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update '*var' as FOBOS algorithm with fixed learning rate.

prox_v = var - alpha  *  delta
// var = sign(prox_v)/(1+alpha * l2)  *  max{|prox_v|-alpha * l1,0}

*/








func applyProximalGradientDescent( scope:Scope,`var`: Output, alpha: Output, l1: Output, l2: Output, delta: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ApplyProximalGradientDescent",
        Name: "Type",
        Input: [ `var`, alpha, l1, l2, delta],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update '*var' according to the RMSProp algorithm.

Note that in dense implementation of this algorithm, ms and mom will
// update even if the grad is zero, but in this sparse implementation, ms
// and mom will not update in iterations during which the grad is zero.
// 
// mean_square = decay  *  mean_square + (1-decay)  *  gradient  *  *  2
// Delta = learning_rate  *  gradient / sqrt(mean_square + epsilon)
// 
// ms <- rho  *  ms_{t-1} + (1-rho)  *  grad  *  grad
// mom <- momentum  *  mom_{t-1} + lr  *  grad / sqrt(ms + epsilon)
// var <- var - mom

*/








func applyRMSProp( scope:Scope,`var`: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ApplyRMSProp",
        Name: "Type",
        Input: [ `var`, ms, mom, lr, rho, momentum, epsilon, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the truth value of abs(x-y) < tolerance element-wise.


*/








func approximateEqual( scope:Scope,x: Output, y: Output, tolerance :Float  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["tolerance"] = tolerance

    let opspec = OpSpec(
        OpType: "ApproximateEqual",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the index with the largest value across dimensions of a tensor.

Note that in case of ties the identity of the return value is not guaranteed.

*/








func argMax( scope:Scope,input: Output, dimension: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ArgMax",
        Name: "Type",
        Input: [ input, dimension],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the index with the smallest value across dimensions of a tensor.

Note that in case of ties the identity of the return value is not guaranteed.

*/








func argMin( scope:Scope,input: Output, dimension: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ArgMin",
        Name: "Type",
        Input: [ input, dimension],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Converts each entry in the given tensor to strings.  Supports many numeric

types and boolean.

*/








func asString( scope:Scope,input: Output, precision :UInt8  , scientific :Bool  , shortest :Bool  , width :UInt8  , fill :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["precision"] = precision
    attrs["scientific"] = scientific
    attrs["shortest"] = shortest
    attrs["width"] = width
    attrs["fill"] = fill

    let opspec = OpSpec(
        OpType: "AsString",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes asin of x element-wise.


*/








func asin( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Asin",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Asserts that the given condition is true.

If `condition` evaluates to false, print the list of tensors in `data`.
// `summarize` determines how many entries of the tensors to print.

*/








func assert( scope:Scope,condition: Output, data: Output, summarize :UInt8  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["summarize"] = summarize

    let opspec = OpSpec(
        OpType: "Assert",
        Name: "Type",
        Input: [ condition, data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Update 'ref' by assigning 'value' to it.

This operation outputs "ref" after the assignment is done.
// This makes it easier to chain operations that need to use the reset value.

*/








func assign( scope:Scope,ref: Output, value: Output, validateShape :Bool  , useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["validate_shape"] = validateShape
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "Assign",
        Name: "Type",
        Input: [ ref, value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update 'ref' by adding 'value' to it.

This operation outputs "ref" after the update is done.
// This makes it easier to chain operations that need to use the reset value.

*/








func assignAdd( scope:Scope,ref: Output, value: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "AssignAdd",
        Name: "Type",
        Input: [ ref, value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update 'ref' by subtracting 'value' from it.

This operation outputs "ref" after the update is done.
// This makes it easier to chain operations that need to use the reset value.

*/








func assignSub( scope:Scope,ref: Output, value: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "AssignSub",
        Name: "Type",
        Input: [ ref, value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes atan of x element-wise.


*/








func atan( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Atan",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes arctangent of `y/x` element-wise, respecting signs of the arguments.

This is the angle \( \theta \in [-\pi, \pi] \) such that
// \[ x = r \cos(\theta) \]
// and
// \[ y = r \sin(\theta) \]
// where \(r = \sqrt(x// ^2 + y// ^2) \).

*/








func atan2( scope:Scope,y: Output, x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Atan2",
        Name: "Type",
        Input: [ y, x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Produces a visualization of audio data over time.

Spectrograms are a standard way of representing audio information as a series of
// slices of frequency information, one slice for each window of time. By joining
// these together into a sequence, they form a distinctive fingerprint of the sound
// over time.
// 
// This op expects to receive audio data as an input, stored as floats in the range
// -1 to 1, together with a window width in samples, and a stride specifying how
// far to move the window between slices. From this it generates a three
// dimensional output. The lowest dimension has an amplitude value for each
// frequency during that time slice. The next dimension is time, with successive
// frequency slices. The final dimension is for the channels in the input, so a
// stereo audio input would have two here for example.
// 
// This means the layout when converted and saved as an image is rotated 90 degrees
// clockwise from a typical spectrogram. Time is descending down the Y axis, and
// the frequency decreases from left to right.
// 
// Each value in the result represents the square root of the sum of the real and
// imaginary parts of an FFT on the current window of samples. In this way, the
// lowest dimension represents the power of each frequency in the current window,
// and adjacent windows are concatenated in the next dimension.
// 
// To get a more intuitive and visual look at what this operation does, you can run
// tensorflow/examples/wav_to_spectrogram to read in an audio file and save out the
// resulting spectrogram as a PNG image.

*/








func audioSpectrogram( scope:Scope,input: Output, windowSize :UInt8  , stride :UInt8  , magnitudeSquared :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["window_size"] = windowSize
    attrs["stride"] = stride
    attrs["magnitude_squared"] = magnitudeSquared

    let opspec = OpSpec(
        OpType: "AudioSpectrogram",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Outputs a `Summary` protocol buffer with audio.

The summary has up to `max_outputs` summary values containing audio. The
// audio is built from `tensor` which must be 3-D with shape `[batch_size,
// frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
// assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.
// 
// The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
// build the `tag` of the summary values:
// 
//  *   If `max_outputs` is 1, the summary value tag is ' * tag * /audio'.
//  *   If `max_outputs` is greater than 1, the summary value tags are
//    generated sequentially as ' * tag * /audio/0', ' * tag * /audio/1', etc.

*/








func audioSummary( scope:Scope,tag: Output, tensor: Output, sampleRate :Float  , maxOutputs :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["sample_rate"] = sampleRate
    attrs["max_outputs"] = maxOutputs

    let opspec = OpSpec(
        OpType: "AudioSummary",
        Name: "Type",
        Input: [ tag, tensor],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Outputs a `Summary` protocol buffer with audio.

The summary has up to `max_outputs` summary values containing audio. The
// audio is built from `tensor` which must be 3-D with shape `[batch_size,
// frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
// assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.
// 
// The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
// build the `tag` of the summary values:
// 
//  *   If `max_outputs` is 1, the summary value tag is ' * tag * /audio'.
//  *   If `max_outputs` is greater than 1, the summary value tags are
//    generated sequentially as ' * tag * /audio/0', ' * tag * /audio/1', etc.

*/








func audioSummaryV2( scope:Scope,tag: Output, tensor: Output, sampleRate: Output, maxOutputs :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["max_outputs"] = maxOutputs

    let opspec = OpSpec(
        OpType: "AudioSummaryV2",
        Name: "Type",
        Input: [ tag, tensor, sampleRate],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Performs average pooling on the input.

Each entry in `output` is the mean of the corresponding size `ksize`
// window in `value`.

*/








func avgPool( scope:Scope,value: Output, ksize :[Int64]  , strides :[Int64]  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["ksize"] = ksize
    attrs["strides"] = strides
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "AvgPool",
        Name: "Type",
        Input: [ value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Performs 3D average pooling on the input.


*/








func avgPool3D( scope:Scope,input: Output, ksize :[Int64]  , strides :[Int64]  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["ksize"] = ksize
    attrs["strides"] = strides
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "AvgPool3D",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes gradients of average pooling function.


*/








func avgPool3DGrad( scope:Scope,origInputShape: Output, grad: Output, ksize :[Int64]  , strides :[Int64]  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["ksize"] = ksize
    attrs["strides"] = strides
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "AvgPool3DGrad",
        Name: "Type",
        Input: [ origInputShape, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes gradients of the average pooling function.


*/








func avgPoolGrad( scope:Scope,origInputShape: Output, grad: Output, ksize :[Int64]  , strides :[Int64]  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["ksize"] = ksize
    attrs["strides"] = strides
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "AvgPoolGrad",
        Name: "Type",
        Input: [ origInputShape, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Defines a barrier that persists across different graph executions.

A barrier represents a key-value map, where each key is a string, and
// each value is a tuple of tensors.
// 
// At runtime, the barrier contains 'complete' and 'incomplete'
// elements. A complete element has defined tensors for all components of
// its value tuple, and may be accessed using BarrierTakeMany. An
// incomplete element has some undefined components in its value tuple,
// and may be updated using BarrierInsertMany.

*/








func barrier( scope:Scope, shapes :[Shape]  , capacity :UInt8  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shapes"] = shapes
    attrs["capacity"] = capacity
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "Barrier",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Closes the given barrier.

This operation signals that no more new elements will be inserted in the
// given barrier. Subsequent InsertMany that try to introduce a new key will fail.
// Subsequent InsertMany operations that just add missing components to already
// existing elements will continue to succeed. Subsequent TakeMany operations will
// continue to succeed if sufficient completed elements remain in the barrier.
// Subsequent TakeMany operations that would block will fail immediately.

*/








func barrierClose( scope:Scope,handle: Output, cancelPendingEnqueues :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["cancel_pending_enqueues"] = cancelPendingEnqueues

    let opspec = OpSpec(
        OpType: "BarrierClose",
        Name: "Type",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Computes the number of incomplete elements in the given barrier.


*/








func barrierIncompleteSize( scope:Scope,handle: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BarrierIncompleteSize",
        Name: "Type",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
For each key, assigns the respective value to the specified component.

If a key is not found in the barrier, this operation will create a new
// incomplete element. If a key is found in the barrier, and the element
// already has a value at component_index, this operation will fail with
// INVALID_ARGUMENT, and leave the barrier in an undefined state.

*/








func barrierInsertMany( scope:Scope,handle: Output, keys: Output, values: Output, componentIndex :UInt8  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["component_index"] = componentIndex

    let opspec = OpSpec(
        OpType: "BarrierInsertMany",
        Name: "Type",
        Input: [ handle, keys, values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Computes the number of complete elements in the given barrier.


*/








func barrierReadySize( scope:Scope,handle: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BarrierReadySize",
        Name: "Type",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Takes the given number of completed elements from a barrier.

This operation concatenates completed-element component tensors along
// the 0th dimension to make a single component tensor.
// 
// Elements come out of the barrier when they are complete, and in the order
// in which they were placed into the barrier.  The indices output provides
// information about the batch in which each element was originally inserted
// into the barrier.

*/








func barrierTakeMany( scope:Scope,handle: Output, numElements: Output, allowSmallBatch :Bool  , waitForIncomplete :Bool  , timeoutMs :UInt8  )  ->(indices: Output?, keys: Output?, values: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["allow_small_batch"] = allowSmallBatch
    attrs["wait_for_incomplete"] = waitForIncomplete
    attrs["timeout_ms"] = timeoutMs

    let opspec = OpSpec(
        OpType: "BarrierTakeMany",
        Name: "Type",
        Input: [ handle, numElements],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*


*/








func batchCholesky( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BatchCholesky",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func batchCholeskyGrad( scope:Scope,l: Output, grad: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BatchCholeskyGrad",
        Name: "Type",
        Input: [ l, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a dataset that batches `batch_size` elements from `input_dataset`.


*/








func batchDataset( scope:Scope,inputDataset: Output, batchSize: Output, outputShapes :[Shape]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["output_shapes"] = outputShapes

    let opspec = OpSpec(
        OpType: "BatchDataset",
        Name: "Type",
        Input: [ inputDataset, batchSize],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func batchFFT( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BatchFFT",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func batchFFT2D( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BatchFFT2D",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func batchFFT3D( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BatchFFT3D",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func batchIFFT( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BatchIFFT",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func batchIFFT2D( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BatchIFFT2D",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func batchIFFT3D( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BatchIFFT3D",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Multiplies slices of two tensors in batches.

Multiplies all slices of `Tensor` `x` and `y` (each slice can be
// viewed as an element of a batch), and arranges the individual results
// in a single output tensor of the same batch size. Each of the
// individual slices can optionally be adjointed (to adjoint a matrix
// means to transpose and conjugate it) before multiplication by setting
// the `adj_x` or `adj_y` flag to `True`, which are by default `False`.
// 
// The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
// and `[..., r_y, c_y]`.
// 
// The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:
// 
//     r_o = c_x if adj_x else r_x
//     c_o = r_y if adj_y else c_y
// 
// It is computed as:
// 
//     output[..., :, :] = matrix(x[..., :, :])  *  matrix(y[..., :, :])

*/








func batchMatMul( scope:Scope,x: Output, y: Output, adjX :Bool  , adjY :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["adj_x"] = adjX
    attrs["adj_y"] = adjY

    let opspec = OpSpec(
        OpType: "BatchMatMul",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func batchMatrixBandPart( scope:Scope,input: Output, numLower: Output, numUpper: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BatchMatrixBandPart",
        Name: "Type",
        Input: [ input, numLower, numUpper],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func batchMatrixDeterminant( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BatchMatrixDeterminant",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func batchMatrixDiag( scope:Scope,diagonal: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BatchMatrixDiag",
        Name: "Type",
        Input: [ diagonal],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func batchMatrixDiagPart( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BatchMatrixDiagPart",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func batchMatrixInverse( scope:Scope,input: Output, adjoint :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["adjoint"] = adjoint

    let opspec = OpSpec(
        OpType: "BatchMatrixInverse",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func batchMatrixSetDiag( scope:Scope,input: Output, diagonal: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BatchMatrixSetDiag",
        Name: "Type",
        Input: [ input, diagonal],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func batchMatrixSolve( scope:Scope,matrix: Output, rhs: Output, adjoint :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["adjoint"] = adjoint

    let opspec = OpSpec(
        OpType: "BatchMatrixSolve",
        Name: "Type",
        Input: [ matrix, rhs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func batchMatrixSolveLs( scope:Scope,matrix: Output, rhs: Output, l2Regularizer: Output, fast :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["fast"] = fast

    let opspec = OpSpec(
        OpType: "BatchMatrixSolveLs",
        Name: "Type",
        Input: [ matrix, rhs, l2Regularizer],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func batchMatrixTriangularSolve( scope:Scope,matrix: Output, rhs: Output, lower :Bool  , adjoint :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["lower"] = lower
    attrs["adjoint"] = adjoint

    let opspec = OpSpec(
        OpType: "BatchMatrixTriangularSolve",
        Name: "Type",
        Input: [ matrix, rhs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Batch normalization.

This op is deprecated. Prefer `tf.nn.batch_normalization`.

*/








func batchNormWithGlobalNormalization( scope:Scope,t: Output, m: Output, v: Output, beta: Output, gamma: Output, varianceEpsilon :Float  , scaleAfterNormalization :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["variance_epsilon"] = varianceEpsilon
    attrs["scale_after_normalization"] = scaleAfterNormalization

    let opspec = OpSpec(
        OpType: "BatchNormWithGlobalNormalization",
        Name: "Type",
        Input: [ t, m, v, beta, gamma],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Gradients for batch normalization.

This op is deprecated. See `tf.nn.batch_normalization`.

*/








func batchNormWithGlobalNormalizationGrad( scope:Scope,t: Output, m: Output, v: Output, gamma: Output, backprop: Output, varianceEpsilon :Float  , scaleAfterNormalization :Bool  )  ->(dx: Output?, dm: Output?, dv: Output?, db: Output?, dg: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["variance_epsilon"] = varianceEpsilon
    attrs["scale_after_normalization"] = scaleAfterNormalization

    let opspec = OpSpec(
        OpType: "BatchNormWithGlobalNormalizationGrad",
        Name: "Type",
        Input: [ t, m, v, gamma, backprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1),op?.output(4 - 1),op?.output(5 - 1))
}

/*


*/








func batchSelfAdjointEig( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BatchSelfAdjointEig",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func batchSelfAdjointEigV2( scope:Scope,input: Output, computeV :Bool  )  ->(e: Output?, v: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["compute_v"] = computeV

    let opspec = OpSpec(
        OpType: "BatchSelfAdjointEigV2",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*


*/








func batchSvd( scope:Scope,input: Output, computeUv :Bool  , fullMatrices :Bool  )  ->(s: Output?, u: Output?, v: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["compute_uv"] = computeUv
    attrs["full_matrices"] = fullMatrices

    let opspec = OpSpec(
        OpType: "BatchSvd",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
BatchToSpace for 4-D tensors of type T.

This is a legacy version of the more general BatchToSpaceND.
// 
// Rearranges (permutes) data from batch into blocks of spatial data, followed by
// cropping. This is the reverse transformation of SpaceToBatch. More specifically,
// this op outputs a copy of the input tensor where values from the `batch`
// dimension are moved in spatial blocks to the `height` and `width` dimensions,
// followed by cropping along the `height` and `width` dimensions.

*/








func batchToSpace( scope:Scope,input: Output, crops: Output, blockSize :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["block_size"] = blockSize

    let opspec = OpSpec(
        OpType: "BatchToSpace",
        Name: "Type",
        Input: [ input, crops],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
BatchToSpace for N-D tensors of type T.

This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of shape
// `block_shape + [batch]`, interleaves these blocks back into the grid defined by
// the spatial dimensions `[1, ..., M]`, to obtain a result with the same rank as
// the input.  The spatial dimensions of this intermediate result are then
// optionally cropped according to `crops` to produce the output.  This is the
// reverse of SpaceToBatch.  See below for a precise description.

*/








func batchToSpaceND( scope:Scope,input: Output, blockShape: Output, crops: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BatchToSpaceND",
        Name: "Type",
        Input: [ input, blockShape, crops],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Compute the regularized incomplete beta integral \\(I_x(a, b)\\).

The regularized incomplete beta integral is defined as:
// 
// 
// \\(I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}\\)
// 
// where
// 
// 
// \\(B(x; a, b) = \int_0// ^x t// ^{a-1} (1 - t)// ^{b-1} dt\\)
// 
// 
// is the incomplete beta function and \\(B(a, b)\\) is the  * complete * 
// beta function.

*/








func betainc( scope:Scope,a: Output, b: Output, x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Betainc",
        Name: "Type",
        Input: [ a, b, x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Adds `bias` to `value`.

This is a special case of `tf.add` where `bias` is restricted to be 1-D.
// Broadcasting is supported, so `value` may have any number of dimensions.

*/








func biasAdd( scope:Scope,value: Output, bias: Output, dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "BiasAdd",
        Name: "Type",
        Input: [ value, bias],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
The backward operation for "BiasAdd" on the "bias" tensor.

It accumulates all the values from out_backprop into the feature dimension.
// For NHWC data format, the feature dimension is the last. For NCHW data format,
// the feature dimension is the third-to-last.

*/








func biasAddGrad( scope:Scope,outBackprop: Output, dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "BiasAddGrad",
        Name: "Type",
        Input: [ outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Adds `bias` to `value`.

This is a deprecated version of BiasAdd and will be soon removed.
// 
// This is a special case of `tf.add` where `bias` is restricted to be 1-D.
// Broadcasting is supported, so `value` may have any number of dimensions.

*/








func biasAddV1( scope:Scope,value: Output, bias: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BiasAddV1",
        Name: "Type",
        Input: [ value, bias],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Counts the number of occurrences of each value in an integer array.

Outputs a vector with length `size` and the same dtype as `weights`. If
// `weights` are empty, then index `i` stores the number of times the value `i` is
// counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
// the value in `weights` at each index where the corresponding value in `arr` is
// `i`.
// 
// Values in `arr` outside of the range [0, size) are ignored.

*/








func bincount( scope:Scope,arr: Output, size: Output, weights: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Bincount",
        Name: "Type",
        Input: [ arr, size, weights],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Bitcasts a tensor from one type to another without copying data.

Given a tensor `input`, this operation returns a tensor that has the same buffer
// data as `input` with datatype `type`.
// 
// If the input datatype `T` is larger than the output datatype `type` then the
// shape changes from [...] to [..., sizeof(`T`)/sizeof(`type`)].
// 
// If `T` is smaller than `type`, the operator requires that the rightmost
// dimension be equal to sizeof(`type`)/sizeof(`T`). The shape then goes from
// [..., sizeof(`type`)/sizeof(`T`)] to [...].
// 
//  * NOTE * : Bitcast is implemented as a low-level cast, so machines with different
// endian orderings will give different results.

*/








func bitcast( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Bitcast",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Return the shape of s0 op s1 with broadcast.

Given `s0` and `s1`, tensors that represent shapes, compute `r0`, the
// broadcasted shape. `s0`, `s1` and `r0` are all integer vectors.

*/








func broadcastArgs( scope:Scope,s0: Output, s1: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BroadcastArgs",
        Name: "Type",
        Input: [ s0, s1],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Return the reduction indices for computing gradients of s0 op s1 with broadcast.

This is typically used by gradient computations for a broadcasting operation.

*/








func broadcastGradientArgs( scope:Scope,s0: Output, s1: Output )  ->(r0: Output?, r1: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "BroadcastGradientArgs",
        Name: "Type",
        Input: [ s0, s1],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Bucketizes 'input' based on 'boundaries'.

For example, if the inputs are
//     boundaries = [0, 10, 100]
//     input = [[-5, 10000]
//              [150,   10]
//              [5,    100]]
// 
// then the output will be
//     output = [[0, 3]
//               [3, 2]
//               [1, 3]]

*/








func bucketize( scope:Scope,input: Output, boundaries :[Float]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["boundaries"] = boundaries

    let opspec = OpSpec(
        OpType: "Bucketize",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Performs beam search decoding on the logits given in input.

A note about the attribute merge_repeated: For the beam search decoder,
// this means that if consecutive entries in a beam are the same, only
// the first of these is emitted.  That is, when the top path is "A B B B B",
// "A B" is returned if merge_repeated = True but "A B B B B" is
// returned if merge_repeated = False.

*/








func ctcBeamSearchDecoder( scope:Scope,inputs: Output, sequenceLength: Output, beamWidth :UInt8  , topPaths :UInt8  , mergeRepeated :Bool  )  ->(decodedIndices: Output?, decodedValues: Output?, decodedShape: Output?, logProbability: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["beam_width"] = beamWidth
    attrs["top_paths"] = topPaths
    attrs["merge_repeated"] = mergeRepeated

    let opspec = OpSpec(
        OpType: "CTCBeamSearchDecoder",
        Name: "Type",
        Input: [ inputs, sequenceLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1),op?.output(4 - 1))
}

/*
Performs greedy decoding on the logits given in inputs.

A note about the attribute merge_repeated: if enabled, when
// consecutive logits' maximum indices are the same, only the first of
// these is emitted.  Labeling the blank ' * ', the sequence "A B B  *  B B"
// becomes "A B B" if merge_repeated = True and "A B B B B" if
// merge_repeated = False.
// 
// Regardless of the value of merge_repeated, if the maximum index of a given
// time and batch corresponds to the blank, index `(num_classes - 1)`, no new
// element is emitted.

*/








func ctcGreedyDecoder( scope:Scope,inputs: Output, sequenceLength: Output, mergeRepeated :Bool  )  ->(decodedIndices: Output?, decodedValues: Output?, decodedShape: Output?, logProbability: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["merge_repeated"] = mergeRepeated

    let opspec = OpSpec(
        OpType: "CTCGreedyDecoder",
        Name: "Type",
        Input: [ inputs, sequenceLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1),op?.output(4 - 1))
}

/*
Calculates the CTC Loss (log probability) for each batch entry.  Also calculates

the gradient.  This class performs the softmax operation for you, so inputs
// should be e.g. linear projections of outputs by an LSTM.

*/








func ctcLoss( scope:Scope,inputs: Output, labelsIndices: Output, labelsValues: Output, sequenceLength: Output, preprocessCollapseRepeated :Bool  , ctcMergeRepeated :Bool  , ignoreLongerOutputsThanInputs :Bool  )  ->(loss: Output?, gradient: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["preprocess_collapse_repeated"] = preprocessCollapseRepeated
    attrs["ctc_merge_repeated"] = ctcMergeRepeated
    attrs["ignore_longer_outputs_than_inputs"] = ignoreLongerOutputsThanInputs

    let opspec = OpSpec(
        OpType: "CTCLoss",
        Name: "Type",
        Input: [ inputs, labelsIndices, labelsValues, sequenceLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Cast x of type SrcT to y of DstT.


*/








func cast( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Cast",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns element-wise smallest integer in not less than x.


*/








func ceil( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Ceil",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Checks a tensor for NaN and Inf values.

When run, reports an `InvalidArgument` error if `tensor` has any values
// that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.

*/








func checkNumerics( scope:Scope,tensor: Output, message :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["message"] = message

    let opspec = OpSpec(
        OpType: "CheckNumerics",
        Name: "Type",
        Input: [ tensor],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the Cholesky decomposition of one or more square matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
// form square matrices, with the same constraints as the single matrix Cholesky
// decomposition above. The output is a tensor of the same shape as the input
// containing the Cholesky decompositions for all input submatrices `[..., :, :]`.

*/








func cholesky( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Cholesky",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the reverse mode backpropagated gradient of the Cholesky algorithm.

For an explanation see "Differentiation of the Cholesky algorithm" by
// Iain Murray http://arxiv.org/abs/1602.07527.

*/








func choleskyGrad( scope:Scope,l: Output, grad: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "CholeskyGrad",
        Name: "Type",
        Input: [ l, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Converts two real numbers to a complex number.

Given a tensor `real` representing the real part of a complex number, and a
// tensor `imag` representing the imaginary part of a complex number, this
// operation returns complex numbers elementwise of the form \\(a + bj\\), where
//  * a *  represents the `real` part and  * b *  represents the `imag` part.
// 
// The input tensors `real` and `imag` must have the same shape.
// 
// For example:
// 
// ```
// # tensor 'real' is [2.25, 3.25]
// # tensor `imag` is [4.75, 5.75]
// tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
// ```

*/








func complex( scope:Scope,real: Output, imag: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Complex",
        Name: "Type",
        Input: [ real, imag],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the complex absolute value of a tensor.

Given a tensor `x` of complex numbers, this operation returns a tensor of type
// `float` or `double` that is the absolute value of each element in `x`. All
// elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
// value is computed as \\( \sqrt{a// ^2 + b// ^2}\\).

*/








func complexAbs( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ComplexAbs",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the ids of the positions in sampled_candidates that match true_labels.

When doing log-odds NCE, the result of this op should be passed through a
// SparseToDense op, then added to the logits of the sampled candidates. This has
// the effect of 'removing' the sampled labels that match the true labels by
// making the classifier sure that they are sampled labels.

*/








func computeAccidentalHits( scope:Scope,trueClasses: Output, sampledCandidates: Output, numTrue :UInt8  , seed :UInt8  , seed2 :UInt8  )  ->(indices: Output?, ids: Output?, weights: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_true"] = numTrue
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "ComputeAccidentalHits",
        Name: "Type",
        Input: [ trueClasses, sampledCandidates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Concatenates tensors along one dimension.


*/








func concat( scope:Scope,concatDim: Output, values: Output, n :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["N"] = n

    let opspec = OpSpec(
        OpType: "Concat",
        Name: "Type",
        Input: [ concatDim, values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes offsets of concat inputs within its output.

For example:
// 
// ```
// # 'x' is [2, 2, 7]
// # 'y' is [2, 3, 7]
// # 'z' is [2, 5, 7]
// concat_offset(2, [x, y, z]) => [0, 0, 0], [0, 2, 0], [0, 5, 0]
// ```
// 
// This is typically used by gradient computations for a concat operation.

*/








func concatOffset( scope:Scope,concatDim: Output, shape: Output, n :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["N"] = n

    let opspec = OpSpec(
        OpType: "ConcatOffset",
        Name: "Type",
        Input: [ concatDim, shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Concatenates tensors along one dimension.


*/








func concatV2( scope:Scope,values: Output, axis: Output, n :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["N"] = n

    let opspec = OpSpec(
        OpType: "ConcatV2",
        Name: "Type",
        Input: [ values, axis],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A conditional accumulator for aggregating gradients.

The accumulator accepts gradients marked with local_step greater or
// equal to the most recent global_step known to the accumulator. The
// average can be extracted from the accumulator, provided sufficient
// gradients have been accumulated. Extracting the average automatically
// resets the aggregate to 0, and increments the global_step recorded by
// the accumulator.

*/








func conditionalAccumulator( scope:Scope, shape :Shape  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shape"] = shape
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "ConditionalAccumulator",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the complex conjugate of a complex number.

Given a tensor `input` of complex numbers, this operation returns a tensor of
// complex numbers that are the complex conjugate of each element in `input`. The
// complex numbers in `input` must be of the form \\(a + bj\\), where  * a *  is the
// real part and  * b *  is the imaginary part.
// 
// The complex conjugate returned by this operation is of the form \\(a - bj\\).
// 
// For example:
// 
// ```
// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
// tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
// ```

*/








func conj( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Conj",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns a constant tensor.


*/








func const( scope:Scope, value :Tensor  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["value"] = value

    let opspec = OpSpec(
        OpType: "Const",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Does nothing. Serves as a control trigger for scheduling.

Only useful as a placeholder for control edges.

*/








func controlTrigger( scope:Scope )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ControlTrigger",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Computes a 2-D convolution given 4-D `input` and `filter` tensors.

Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
// and a filter / kernel tensor of shape
// `[filter_height, filter_width, in_channels, out_channels]`, this op
// performs the following:
// 
// 1. Flattens the filter to a 2-D matrix with shape
//    `[filter_height  *  filter_width  *  in_channels, output_channels]`.
// 2. Extracts image patches from the input tensor to form a  * virtual * 
//    tensor of shape `[batch, out_height, out_width,
//    filter_height  *  filter_width  *  in_channels]`.
// 3. For each patch, right-multiplies the filter matrix and the image patch
//    vector.
// 
// In detail, with the default NHWC format,
// 
//     output[b, i, j, k] =
//         sum_{di, dj, q} input[b, strides[1]  *  i + di, strides[2]  *  j + dj, q]  * 
//                         filter[di, dj, q, k]
// 
// Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
// horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

*/








func conv2D( scope:Scope,input: Output, filter: Output, strides :[Int64]  , useCudnnOnGpu :Bool  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["strides"] = strides
    attrs["use_cudnn_on_gpu"] = useCudnnOnGpu
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "Conv2D",
        Name: "Type",
        Input: [ input, filter],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradients of convolution with respect to the filter.


*/








func conv2DBackpropFilter( scope:Scope,input: Output, filterSizes: Output, outBackprop: Output, strides :[Int64]  , useCudnnOnGpu :Bool  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["strides"] = strides
    attrs["use_cudnn_on_gpu"] = useCudnnOnGpu
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "Conv2DBackpropFilter",
        Name: "Type",
        Input: [ input, filterSizes, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradients of convolution with respect to the input.


*/








func conv2DBackpropInput( scope:Scope,inputSizes: Output, filter: Output, outBackprop: Output, strides :[Int64]  , useCudnnOnGpu :Bool  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["strides"] = strides
    attrs["use_cudnn_on_gpu"] = useCudnnOnGpu
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "Conv2DBackpropInput",
        Name: "Type",
        Input: [ inputSizes, filter, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes a 3-D convolution given 5-D `input` and `filter` tensors.

In signal processing, cross-correlation is a measure of similarity of
// two waveforms as a function of a time-lag applied to one of them. This
// is also known as a sliding dot product or sliding inner-product.
// 
// Our Conv3D implements a form of cross-correlation.

*/








func conv3D( scope:Scope,input: Output, filter: Output, strides :[Int64]  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["strides"] = strides
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "Conv3D",
        Name: "Type",
        Input: [ input, filter],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradients of 3-D convolution with respect to the filter.


*/








func conv3DBackpropFilter( scope:Scope,input: Output, filter: Output, outBackprop: Output, strides :[Int64]  , padding :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["strides"] = strides
    attrs["padding"] = padding

    let opspec = OpSpec(
        OpType: "Conv3DBackpropFilter",
        Name: "Type",
        Input: [ input, filter, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradients of 3-D convolution with respect to the filter.


*/








func conv3DBackpropFilterV2( scope:Scope,input: Output, filterSizes: Output, outBackprop: Output, strides :[Int64]  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["strides"] = strides
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "Conv3DBackpropFilterV2",
        Name: "Type",
        Input: [ input, filterSizes, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradients of 3-D convolution with respect to the input.


*/








func conv3DBackpropInput( scope:Scope,input: Output, filter: Output, outBackprop: Output, strides :[Int64]  , padding :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["strides"] = strides
    attrs["padding"] = padding

    let opspec = OpSpec(
        OpType: "Conv3DBackpropInput",
        Name: "Type",
        Input: [ input, filter, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradients of 3-D convolution with respect to the input.


*/








func conv3DBackpropInputV2( scope:Scope,inputSizes: Output, filter: Output, outBackprop: Output, strides :[Int64]  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["strides"] = strides
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "Conv3DBackpropInputV2",
        Name: "Type",
        Input: [ inputSizes, filter, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Copy Op.

Performs CPU-to-CPU or GPU-to-GPU deep-copying of tensor, depending on the
// device on which the tensor is allocated.
// N.B.: If the all downstream attached debug ops are disabled given the current
// gRPC gating status, the output will simply forward the input tensor without
// deep-copying. See the documentation of Debug *  ops for more details.
// 
// Unlike the CopyHost Op, this op does not have HostMemory constraint on its
// input or output.

*/








func copy( scope:Scope,input: Output, tensorName :String  , debugOpsSpec :[Data]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["tensor_name"] = tensorName
    attrs["debug_ops_spec"] = debugOpsSpec

    let opspec = OpSpec(
        OpType: "Copy",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Copy Host Op.

Performs CPU-to-CPU deep-copying of tensor.
// N.B.: If the all downstream attached debug ops are disabled given the current
// gRPC gating status, the output will simply forward the input tensor without
// deep-copying. See the documentation of Debug *  ops for more details.
// 
// Unlike the Copy Op, this op has HostMemory constraint on its input or output.

*/








func copyHost( scope:Scope,input: Output, tensorName :String  , debugOpsSpec :[Data]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["tensor_name"] = tensorName
    attrs["debug_ops_spec"] = debugOpsSpec

    let opspec = OpSpec(
        OpType: "CopyHost",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes cos of x element-wise.


*/








func cos( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Cos",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Increments 'ref' until it reaches 'limit'.


*/








func countUpTo( scope:Scope,ref: Output, limit :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["limit"] = limit

    let opspec = OpSpec(
        OpType: "CountUpTo",
        Name: "Type",
        Input: [ ref],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Extracts crops from the input image tensor and bilinearly resizes them (possibly

with aspect ratio change) to a common output size specified by `crop_size`. This
// is more general than the `crop_to_bounding_box` op which extracts a fixed size
// slice from the input image and does not allow resizing or aspect ratio change.
// 
// Returns a tensor with `crops` from the input `image` at positions defined at the
// bounding box locations in `boxes`. The cropped boxes are all resized (with
// bilinear interpolation) to a fixed `size = [crop_height, crop_width]`. The
// result is a 4-D tensor `[num_boxes, crop_height, crop_width, depth]`.

*/








func cropAndResize( scope:Scope,image: Output, boxes: Output, boxInd: Output, cropSize: Output, method :String  , extrapolationValue :Float  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["method"] = method
    attrs["extrapolation_value"] = extrapolationValue

    let opspec = OpSpec(
        OpType: "CropAndResize",
        Name: "Type",
        Input: [ image, boxes, boxInd, cropSize],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradient of the crop_and_resize op wrt the input boxes tensor.


*/








func cropAndResizeGradBoxes( scope:Scope,grads: Output, image: Output, boxes: Output, boxInd: Output, method :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["method"] = method

    let opspec = OpSpec(
        OpType: "CropAndResizeGradBoxes",
        Name: "Type",
        Input: [ grads, image, boxes, boxInd],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradient of the crop_and_resize op wrt the input image tensor.


*/








func cropAndResizeGradImage( scope:Scope,grads: Output, boxes: Output, boxInd: Output, imageSize: Output, method :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["method"] = method

    let opspec = OpSpec(
        OpType: "CropAndResizeGradImage",
        Name: "Type",
        Input: [ grads, boxes, boxInd, imageSize],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Compute the pairwise cross product.

`a` and `b` must be the same shape; they can either be simple 3-element vectors,
// or any shape where the innermost dimension is 3. In the latter case, each pair
// of corresponding 3-element vectors is cross-multiplied independently.

*/








func cross( scope:Scope,a: Output, b: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Cross",
        Name: "Type",
        Input: [ a, b],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Compute the cumulative product of the tensor `x` along `axis`.

By default, this op performs an inclusive cumprod, which means that the first
// element of the input is identical to the first element of the output:
// 
// ```python
// tf.cumprod([a, b, c])  # => [a, a  *  b, a  *  b  *  c]
// ```
// 
// By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
// performed instead:
// 
// ```python
// tf.cumprod([a, b, c], exclusive=True)  # => [1, a, a  *  b]
// ```
// 
// By setting the `reverse` kwarg to `True`, the cumprod is performed in the
// opposite direction:
// 
// ```python
// tf.cumprod([a, b, c], reverse=True)  # => [a  *  b  *  c, b  *  c, c]
// ```
// 
// This is more efficient than using separate `tf.reverse` ops.
// 
// The `reverse` and `exclusive` kwargs can also be combined:
// 
// ```python
// tf.cumprod([a, b, c], exclusive=True, reverse=True)  # => [b  *  c, c, 1]
// ```

*/








func cumprod( scope:Scope,x: Output, axis: Output, exclusive :Bool  , reverse :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["exclusive"] = exclusive
    attrs["reverse"] = reverse

    let opspec = OpSpec(
        OpType: "Cumprod",
        Name: "Type",
        Input: [ x, axis],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Compute the cumulative sum of the tensor `x` along `axis`.

By default, this op performs an inclusive cumsum, which means that the first
// element of the input is identical to the first element of the output:
// 
// ```python
// tf.cumsum([a, b, c])  # => [a, a + b, a + b + c]
// ```
// 
// By setting the `exclusive` kwarg to `True`, an exclusive cumsum is
// performed instead:
// 
// ```python
// tf.cumsum([a, b, c], exclusive=True)  # => [0, a, a + b]
// ```
// 
// By setting the `reverse` kwarg to `True`, the cumsum is performed in the
// opposite direction:
// 
// ```python
// tf.cumsum([a, b, c], reverse=True)  # => [a + b + c, b + c, c]
// ```
// 
// This is more efficient than using separate `tf.reverse` ops.
// 
// The `reverse` and `exclusive` kwargs can also be combined:
// 
// ```python
// tf.cumsum([a, b, c], exclusive=True, reverse=True)  # => [b + c, c, 0]
// ```

*/








func cumsum( scope:Scope,x: Output, axis: Output, exclusive :Bool  , reverse :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["exclusive"] = exclusive
    attrs["reverse"] = reverse

    let opspec = OpSpec(
        OpType: "Cumsum",
        Name: "Type",
        Input: [ x, axis],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Debug Identity Op.

Provides an identity mapping of the non-Ref type input tensor for debugging.

*/








func debugIdentity( scope:Scope,input: Output, deviceName :String  , tensorName :String  , debugUrls :[Data]  , gatedGrpc :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["device_name"] = deviceName
    attrs["tensor_name"] = tensorName
    attrs["debug_urls"] = debugUrls
    attrs["gated_grpc"] = gatedGrpc

    let opspec = OpSpec(
        OpType: "DebugIdentity",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Debug NaN Value Counter Op

Counts number of NaNs in the input tensor, for debugging.

*/








func debugNanCount( scope:Scope,input: Output, deviceName :String  , tensorName :String  , debugUrls :[Data]  , gatedGrpc :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["device_name"] = deviceName
    attrs["tensor_name"] = tensorName
    attrs["debug_urls"] = debugUrls
    attrs["gated_grpc"] = gatedGrpc

    let opspec = OpSpec(
        OpType: "DebugNanCount",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Debug Numeric Summary Op.

Provide a basic summary of numeric value types, range and distribution.

*/








func debugNumericSummary( scope:Scope,input: Output, deviceName :String  , tensorName :String  , debugUrls :[Data]  , lowerBound :Float  , upperBound :Float  , muteIfHealthy :Bool  , gatedGrpc :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["device_name"] = deviceName
    attrs["tensor_name"] = tensorName
    attrs["debug_urls"] = debugUrls
    attrs["lower_bound"] = lowerBound
    attrs["upper_bound"] = upperBound
    attrs["mute_if_healthy"] = muteIfHealthy
    attrs["gated_grpc"] = gatedGrpc

    let opspec = OpSpec(
        OpType: "DebugNumericSummary",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Decode web-safe base64-encoded strings.

Input may or may not have padding at the end. See EncodeBase64 for padding.
// Web-safe means that input must use - and _ instead of + and /.

*/








func decodeBase64( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "DecodeBase64",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Decode the first frame of a BMP-encoded image to a uint8 tensor.

The attr `channels` indicates the desired number of color channels for the
// decoded image.
// 
// Accepted values are:
// 
//  *    0: Use the number of channels in the BMP-encoded image.
//  *    3: output an RGB image.
//  *    4: output an RGBA image.

*/








func decodeBmp( scope:Scope,contents: Output, channels :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["channels"] = channels

    let opspec = OpSpec(
        OpType: "DecodeBmp",
        Name: "Type",
        Input: [ contents],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Convert CSV records to tensors. Each column maps to one tensor.

RFC 4180 format is expected for the CSV records.
// (https://tools.ietf.org/html/rfc4180)
// Note that we allow leading and trailing spaces with int or float field.

*/








func decodeCSV( scope:Scope,records: Output, recordDefaults: Output, fieldDelim :String  , useQuoteDelim :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["field_delim"] = fieldDelim
    attrs["use_quote_delim"] = useQuoteDelim

    let opspec = OpSpec(
        OpType: "DecodeCSV",
        Name: "Type",
        Input: [ records, recordDefaults],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Decode the first frame of a GIF-encoded image to a uint8 tensor.

GIF with frame or transparency compression are not supported
// convert animated GIF from compressed to uncompressed by:
// 
//     convert $src.gif -coalesce $dst.gif
// 
// This op also supports decoding JPEGs and PNGs, though it is cleaner to use
// `tf.image.decode_image`.

*/








func decodeGif( scope:Scope,contents: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "DecodeGif",
        Name: "Type",
        Input: [ contents],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Convert JSON-encoded Example records to binary protocol buffer strings.

This op translates a tensor containing Example records, encoded using
// the [standard JSON
// mapping](https://developers.google.com/protocol-buffers/docs/proto3#json),
// into a tensor containing the same records encoded as binary protocol
// buffers. The resulting tensor can then be fed to any of the other
// Example-parsing ops.

*/








func decodeJSONExample( scope:Scope,jsonExamples: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "DecodeJSONExample",
        Name: "Type",
        Input: [ jsonExamples],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Decode a JPEG-encoded image to a uint8 tensor.

The attr `channels` indicates the desired number of color channels for the
// decoded image.
// 
// Accepted values are:
// 
//  *    0: Use the number of channels in the JPEG-encoded image.
//  *    1: output a grayscale image.
//  *    3: output an RGB image.
// 
// If needed, the JPEG-encoded image is transformed to match the requested number
// of color channels.
// 
// The attr `ratio` allows downscaling the image by an integer factor during
// decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
// downscaling the image later.
// 
// This op also supports decoding PNGs and non-animated GIFs since the interface is
// the same, though it is cleaner to use `tf.image.decode_image`.

*/








func decodeJpeg( scope:Scope,contents: Output, channels :UInt8  , ratio :UInt8  , fancyUpscaling :Bool  , tryRecoverTruncated :Bool  , acceptableFraction :Float  , dctMethod :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["channels"] = channels
    attrs["ratio"] = ratio
    attrs["fancy_upscaling"] = fancyUpscaling
    attrs["try_recover_truncated"] = tryRecoverTruncated
    attrs["acceptable_fraction"] = acceptableFraction
    attrs["dct_method"] = dctMethod

    let opspec = OpSpec(
        OpType: "DecodeJpeg",
        Name: "Type",
        Input: [ contents],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Decode a PNG-encoded image to a uint8 or uint16 tensor.

The attr `channels` indicates the desired number of color channels for the
// decoded image.
// 
// Accepted values are:
// 
//  *    0: Use the number of channels in the PNG-encoded image.
//  *    1: output a grayscale image.
//  *    3: output an RGB image.
//  *    4: output an RGBA image.
// 
// If needed, the PNG-encoded image is transformed to match the requested number
// of color channels.
// 
// This op also supports decoding JPEGs and non-animated GIFs since the interface
// is the same, though it is cleaner to use `tf.image.decode_image`.

*/








func decodePng( scope:Scope,contents: Output, channels :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["channels"] = channels

    let opspec = OpSpec(
        OpType: "DecodePng",
        Name: "Type",
        Input: [ contents],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Reinterpret the bytes of a string as a vector of numbers.


*/








func decodeRaw( scope:Scope,bytes: Output, littleEndian :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["little_endian"] = littleEndian

    let opspec = OpSpec(
        OpType: "DecodeRaw",
        Name: "Type",
        Input: [ bytes],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Decode a 16-bit PCM WAV file to a float tensor.

The -32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float.
// 
// When desired_channels is set, if the input contains fewer channels than this
// then the last channel will be duplicated to give the requested number, else if
// the input has more channels than requested then the additional channels will be
// ignored.
// 
// If desired_samples is set, then the audio will be cropped or padded with zeroes
// to the requested length.
// 
// The first output contains a Tensor with the content of the audio samples. The
// lowest dimension will be the number of channels, and the second will be the
// number of samples. For example, a ten-sample-long stereo WAV file should give an
// output shape of [10, 2].

*/








func decodeWav( scope:Scope,contents: Output, desiredChannels :UInt8  , desiredSamples :UInt8  )  ->(audio: Output?, sampleRate: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["desired_channels"] = desiredChannels
    attrs["desired_samples"] = desiredSamples

    let opspec = OpSpec(
        OpType: "DecodeWav",
        Name: "Type",
        Input: [ contents],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Delete the tensor specified by its handle in the session.


*/








func deleteSessionTensor( scope:Scope,handle: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "DeleteSessionTensor",
        Name: "Type",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Applies set operation along last dimension of 2 `Tensor` inputs.

See SetOperationOp::SetOperationFromContext for values of `set_operation`.
// 
// Output `result` is a `SparseTensor` represented by `result_indices`,
// `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
// has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
// dimension contains the result of `set_operation` applied to the corresponding
// `[0...n-1]` dimension of `set`.

*/








func denseToDenseSetOperation( scope:Scope,set1: Output, set2: Output, setOperation :String  , validateIndices :Bool  )  ->(resultIndices: Output?, resultValues: Output?, resultShape: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["set_operation"] = setOperation
    attrs["validate_indices"] = validateIndices

    let opspec = OpSpec(
        OpType: "DenseToDenseSetOperation",
        Name: "Type",
        Input: [ set1, set2],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Creates a dataset that yields a SparseTensor for each element of the input.


*/








func denseToSparseBatchDataset( scope:Scope,inputDataset: Output, batchSize: Output, rowShape: Output, outputShapes :[Shape]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["output_shapes"] = outputShapes

    let opspec = OpSpec(
        OpType: "DenseToSparseBatchDataset",
        Name: "Type",
        Input: [ inputDataset, batchSize, rowShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Applies set operation along last dimension of `Tensor` and `SparseTensor`.

See SetOperationOp::SetOperationFromContext for values of `set_operation`.
// 
// Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
// and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
// as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
// ignored.
// 
// If `validate_indices` is `True`, this op validates the order and range of `set2`
// indices.
// 
// Output `result` is a `SparseTensor` represented by `result_indices`,
// `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
// has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
// dimension contains the result of `set_operation` applied to the corresponding
// `[0...n-1]` dimension of `set`.

*/








func denseToSparseSetOperation( scope:Scope,set1: Output, set2Indices: Output, set2Values: Output, set2Shape: Output, setOperation :String  , validateIndices :Bool  )  ->(resultIndices: Output?, resultValues: Output?, resultShape: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["set_operation"] = setOperation
    attrs["validate_indices"] = validateIndices

    let opspec = OpSpec(
        OpType: "DenseToSparseSetOperation",
        Name: "Type",
        Input: [ set1, set2Indices, set2Values, set2Shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
DepthToSpace for tensors of type T.

Rearranges data from depth into blocks of spatial data.
// This is the reverse transformation of SpaceToDepth. More specifically,
// this op outputs a copy of the input tensor where values from the `depth`
// dimension are moved in spatial blocks to the `height` and `width` dimensions.
// The attr `block_size` indicates the input block size and how the data is moved.
// 
//    *  Chunks of data of size `block_size  *  block_size` from depth are rearranged
//     into non-overlapping blocks of size `block_size x block_size`
//    *  The width the output tensor is `input_depth  *  block_size`, whereas the
//     height is `input_height  *  block_size`.
//    *  The depth of the input tensor must be divisible by
//     `block_size  *  block_size`.
// 
// That is, assuming the input is in the shape:
// `[batch, height, width, depth]`,
// the shape of the output will be:
// `[batch, height * block_size, width * block_size, depth/(block_size * block_size)]`
// 
// This operation requires that the input tensor be of rank 4, and that
// `block_size` be >=1 and that `block_size  *  block_size` be a divisor of the
// input depth.
// 
// This operation is useful for resizing the activations between convolutions
// (but keeping all data), e.g. instead of pooling. It is also useful for training
// purely convolutional models.
// 
// For example, given this input of shape `[1, 1, 1, 4]`, and a block size of 2:
// 
// ```
// x = [[[[1, 2, 3, 4]]]]
// 
// ```
// 
// This operation will output a tensor of shape `[1, 2, 2, 1]`:
// 
// ```
//    [[[[1], [2]],
//      [[3], [4]]]]
// ```
// 
// Here, the input has a batch of 1 and each batch element has shape `[1, 1, 4]`,
// the corresponding output will have 2x2 elements and will have a depth of
// 1 channel (1 = `4 / (block_size  *  block_size)`).
// The output element shape is `[2, 2, 1]`.
// 
// For an input tensor with larger depth, here of shape `[1, 1, 1, 12]`, e.g.
// 
// ```
// x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
// ```
// 
// This operation, for block size of 2, will return the following tensor of shape
// `[1, 2, 2, 3]`
// 
// ```
//    [[[[1, 2, 3], [4, 5, 6]],
//      [[7, 8, 9], [10, 11, 12]]]]
// 
// ```
// 
// Similarly, for the following input of shape `[1 2 2 4]`, and a block size of 2:
// 
// ```
// x =  [[[[1, 2, 3, 4],
//        [5, 6, 7, 8]],
//       [[9, 10, 11, 12],
//        [13, 14, 15, 16]]]]
// ```
// 
// the operator will return the following tensor of shape `[1 4 4 1]`:
// 
// ```
// x = [[ [1],   [2],  [5],  [6]],
//      [ [3],   [4],  [7],  [8]],
//      [ [9],  [10], [13],  [14]],
//      [ [11], [12], [15],  [16]]]
// 
// ```

*/








func depthToSpace( scope:Scope,input: Output, blockSize :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["block_size"] = blockSize

    let opspec = OpSpec(
        OpType: "DepthToSpace",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.

Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
// and a filter / kernel tensor of shape
// `[filter_height, filter_width, in_channels, channel_multiplier]`, containing
// `in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
// a different filter to each input channel (expanding from 1 channel to
// `channel_multiplier` channels for each), then concatenates the results
// together. Thus, the output has `in_channels  *  channel_multiplier` channels.
// 
// for k in 0..in_channels-1
//   for q in 0..channel_multiplier-1
//     output[b, i, j, k  *  channel_multiplier + q] =
//       sum_{di, dj} input[b, strides[1]  *  i + di, strides[2]  *  j + dj, k]  * 
//                         filter[di, dj, k, q]
// 
// Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
// horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

*/








func depthwiseConv2dNative( scope:Scope,input: Output, filter: Output, strides :[Int64]  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["strides"] = strides
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "DepthwiseConv2dNative",
        Name: "Type",
        Input: [ input, filter],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradients of depthwise convolution with respect to the filter.


*/








func depthwiseConv2dNativeBackpropFilter( scope:Scope,input: Output, filterSizes: Output, outBackprop: Output, strides :[Int64]  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["strides"] = strides
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "DepthwiseConv2dNativeBackpropFilter",
        Name: "Type",
        Input: [ input, filterSizes, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradients of depthwise convolution with respect to the input.


*/








func depthwiseConv2dNativeBackpropInput( scope:Scope,inputSizes: Output, filter: Output, outBackprop: Output, strides :[Int64]  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["strides"] = strides
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "DepthwiseConv2dNativeBackpropInput",
        Name: "Type",
        Input: [ inputSizes, filter, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Dequantize the 'input' tensor into a float Tensor.

[min_range, max_range] are scalar floats that specify the range for
// the 'input' data. The 'mode' attribute controls exactly which calculations are
// used to convert the float values to their quantized equivalents.
// 
// In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:
// 
// ```
// if T == qint8, in[i] += (range(T) + 1)/ 2.0
// out[i] = min_range + (in[i] *  (max_range - min_range) / range(T))
// ```
// here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`
// 
//  * MIN_COMBINED Mode Example * 
// 
// If the input comes from a QuantizedRelu6, the output type is
// quint8 (range of 0-255) but the possible range of QuantizedRelu6 is
// 0-6.  The min_range and max_range values are therefore 0.0 and 6.0.
// Dequantize on quint8 will take each value, cast to float, and multiply
// by 6 / 255.
// Note that if quantizedtype is qint8, the operation will additionally add
// each value by 128 prior to casting.
// 
// If the mode is 'MIN_FIRST', then this approach is used:
// 
// ```c++
// number_of_steps = 1 << (# of bits in T)
// range_adjust = number_of_steps / (number_of_steps - 1)
// range = (range_max - range_min)  *  range_adjust
// range_scale = range / number_of_steps
// const double offset_input = static_cast<double>(input) - lowest_quantized;
// result = range_min + ((input - numeric_limits<T>::min())  *  range_scale)
// ```

*/








func dequantize( scope:Scope,input: Output, minRange: Output, maxRange: Output, mode :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["mode"] = mode

    let opspec = OpSpec(
        OpType: "Dequantize",
        Name: "Type",
        Input: [ input, minRange, maxRange],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Deserialize and concatenate `SparseTensors` from a serialized minibatch.

The input `serialized_sparse` must be a string matrix of shape `[N x 3]` where
// `N` is the minibatch size and the rows correspond to packed outputs of
// `SerializeSparse`.  The ranks of the original `SparseTensor` objects
// must all match.  When the final `SparseTensor` is created, it has rank one
// higher than the ranks of the incoming `SparseTensor` objects
// (they have been concatenated along a new row dimension).
// 
// The output `SparseTensor` object's shape values for all dimensions but the
// first are the max across the input `SparseTensor` objects' shape values
// for the corresponding dimensions.  Its first shape value is `N`, the minibatch
// size.
// 
// The input `SparseTensor` objects' indices are assumed ordered in
// standard lexicographic order.  If this is not the case, after this
// step run `SparseReorder` to restore index ordering.
// 
// For example, if the serialized input is a `[2 x 3]` matrix representing two
// original `SparseTensor` objects:
// 
//     index = [ 0]
//             [10]
//             [20]
//     values = [1, 2, 3]
//     shape = [50]
// 
// and
// 
//     index = [ 2]
//             [10]
//     values = [4, 5]
//     shape = [30]
// 
// then the final deserialized `SparseTensor` will be:
// 
//     index = [0  0]
//             [0 10]
//             [0 20]
//             [1  2]
//             [1 10]
//     values = [1, 2, 3, 4, 5]
//     shape = [2 50]

*/








func deserializeManySparse( scope:Scope,serializedSparse: Output )  ->(sparseIndices: Output?, sparseValues: Output?, sparseShape: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "DeserializeManySparse",
        Name: "Type",
        Input: [ serializedSparse],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Destroys the temporary variable and returns its final value.

Sets output to the value of the Tensor pointed to by 'ref', then destroys
// the temporary variable called 'var_name'.
// All other uses of 'ref'  * must *  have executed before this op.
// This is typically achieved by chaining the ref through each assign op, or by
// using control dependencies.
// 
// Outputs the final value of the tensor pointed to by 'ref'.

*/








func destroyTemporaryVariable( scope:Scope,ref: Output, varName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["var_name"] = varName

    let opspec = OpSpec(
        OpType: "DestroyTemporaryVariable",
        Name: "Type",
        Input: [ ref],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns a diagonal tensor with a given diagonal values.

Given a `diagonal`, this operation returns a tensor with the `diagonal` and
// everything else padded with zeros. The diagonal is computed as follows:
// 
// Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
// rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:
// 
// `output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.
// 
// For example:
// 
// ```
// # 'diagonal' is [1, 2, 3, 4]
// tf.diag(diagonal) ==> [[1, 0, 0, 0]
//                        [0, 2, 0, 0]
//                        [0, 0, 3, 0]
//                        [0, 0, 0, 4]]
// ```

*/








func diag( scope:Scope,diagonal: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Diag",
        Name: "Type",
        Input: [ diagonal],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the diagonal part of the tensor.

This operation returns a tensor with the `diagonal` part
// of the `input`. The `diagonal` part is computed as follows:
// 
// Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
// tensor of rank `k` with dimensions `[D1,..., Dk]` where:
// 
// `diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.
// 
// For example:
// 
// ```
// # 'input' is [[1, 0, 0, 0]
//               [0, 2, 0, 0]
//               [0, 0, 3, 0]
//               [0, 0, 0, 4]]
// 
// tf.diag_part(input) ==> [1, 2, 3, 4]
// ```

*/








func diagPart( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "DiagPart",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes Psi, the derivative of Lgamma (the log of the absolute value of

`Gamma(x)`), element-wise.

*/








func digamma( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Digamma",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors.

The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
// `filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
// input channel is processed independently of the others with its own structuring
// function. The `output` tensor has shape
// `[batch, out_height, out_width, depth]`. The spatial dimensions of the output
// tensor depend on the `padding` algorithm. We currently only support the default
// "NHWC" `data_format`.
// 
// In detail, the grayscale morphological 2-D dilation is the max-sum correlation
// (for consistency with `conv2d`, we use unmirrored filters):
// 
//     output[b, y, x, c] =
//        max_{dy, dx} input[b,
//                           strides[1]  *  y + rates[1]  *  dy,
//                           strides[2]  *  x + rates[2]  *  dx,
//                           c] +
//                     filter[dy, dx, c]
// 
// Max-pooling is a special case when the filter has size equal to the pooling
// kernel size and contains all zeros.
// 
// Note on duality: The dilation of `input` by the `filter` is equal to the
// negation of the erosion of `-input` by the reflected `filter`.

*/








func dilation2D( scope:Scope,input: Output, filter: Output, strides :[Int64]  , rates :[Int64]  , padding :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["strides"] = strides
    attrs["rates"] = rates
    attrs["padding"] = padding

    let opspec = OpSpec(
        OpType: "Dilation2D",
        Name: "Type",
        Input: [ input, filter],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradient of morphological 2-D dilation with respect to the filter.


*/








func dilation2DBackpropFilter( scope:Scope,input: Output, filter: Output, outBackprop: Output, strides :[Int64]  , rates :[Int64]  , padding :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["strides"] = strides
    attrs["rates"] = rates
    attrs["padding"] = padding

    let opspec = OpSpec(
        OpType: "Dilation2DBackpropFilter",
        Name: "Type",
        Input: [ input, filter, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradient of morphological 2-D dilation with respect to the input.


*/








func dilation2DBackpropInput( scope:Scope,input: Output, filter: Output, outBackprop: Output, strides :[Int64]  , rates :[Int64]  , padding :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["strides"] = strides
    attrs["rates"] = rates
    attrs["padding"] = padding

    let opspec = OpSpec(
        OpType: "Dilation2DBackpropInput",
        Name: "Type",
        Input: [ input, filter, outBackprop],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns x / y element-wise.

 * NOTE * : `Div` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func div( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Div",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Draw bounding boxes on a batch of images.

Outputs a copy of `images` but draws on top of the pixels zero or more bounding
// boxes specified by the locations in `boxes`. The coordinates of the each
// bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`. The
// bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
// height of the underlying image.
// 
// For example, if an image is 100 x 200 pixels and the bounding box is
// `[0.1, 0.2, 0.5, 0.9]`, the bottom-left and upper-right coordinates of the
// bounding box will be `(10, 40)` to `(50, 180)`.
// 
// Parts of the bounding box may fall outside the image.

*/








func drawBoundingBoxes( scope:Scope,images: Output, boxes: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "DrawBoundingBoxes",
        Name: "Type",
        Input: [ images, boxes],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Partitions `data` into `num_partitions` tensors using indices from `partitions`.

For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
// becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
// are placed in `outputs[i]` in lexicographic order of `js`, and the first
// dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
// In detail,
// 
// ```python
//     outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]
// 
//     outputs[i] = pack([data[js, ...] for js if partitions[js] == i])
// ```
// 
// `data.shape` must start with `partitions.shape`.
// 
// For example:
// 
// ```python
//     # Scalar partitions.
//     partitions = 1
//     num_partitions = 2
//     data = [10, 20]
//     outputs[0] = []  # Empty with shape [0, 2]
//     outputs[1] = [[10, 20]]
// 
//     # Vector partitions.
//     partitions = [0, 0, 1, 1, 0]
//     num_partitions = 2
//     data = [10, 20, 30, 40, 50]
//     outputs[0] = [10, 20, 50]
//     outputs[1] = [30, 40]
// ```
// 
// See `dynamic_stitch` for an example on how to merge partitions back.
// 
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/DynamicPartition.png" alt>
// </div>

*/








func dynamicPartition( scope:Scope,data: Output, partitions: Output, numPartitions :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_partitions"] = numPartitions

    let opspec = OpSpec(
        OpType: "DynamicPartition",
        Name: "Type",
        Input: [ data, partitions],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Interleave the values from the `data` tensors into a single tensor.

Builds a merged tensor such that
// 
// ```python
//     merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
// ```
// 
// For example, if each `indices[m]` is scalar or vector, we have
// 
// ```python
//     # Scalar indices:
//     merged[indices[m], ...] = data[m][...]
// 
//     # Vector indices:
//     merged[indices[m][i], ...] = data[m][i, ...]
// ```
// 
// Each `data[i].shape` must start with the corresponding `indices[i].shape`,
// and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
// must have `data[i].shape = indices[i].shape + constant`.  In terms of this
// `constant`, the output shape is
// 
//     merged.shape = [max(indices)] + constant
// 
// Values are merged in order, so if an index appears in both `indices[m][i]` and
// `indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
// merged result.
// 
// For example:
// 
// ```python
//     indices[0] = 6
//     indices[1] = [4, 1]
//     indices[2] = [[5, 2], [0, 3]]
//     data[0] = [61, 62]
//     data[1] = [[41, 42], [11, 12]]
//     data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
//     merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
//               [51, 52], [61, 62]]
// ```
// 
// This method can be used to merge partitions created by `dynamic_partition`
// as illustrated on the following example:
// 
// ```python
//     # Apply function (increments x_i) on elements for which a certain condition
//     # apply (x_i != -1 in this example).
//     x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
//     condition_mask=tf.not_equal(x,tf.constant(-1.))
//     partitioned_data = tf.dynamic_partition(
//         x, tf.cast(condition_mask, tf.int32) , 2)
//     partitioned_data[1] = partitioned_data[1] + 1.0
//     condition_indices = tf.dynamic_partition(
//         tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
//     x = tf.dynamic_stitch(condition_indices, partitioned_data)
//     # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
//     # unchanged.
// ```
// 
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/DynamicStitch.png" alt>
// </div>

*/








func dynamicStitch( scope:Scope,indices: Output, data: Output, n :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["N"] = n

    let opspec = OpSpec(
        OpType: "DynamicStitch",
        Name: "Type",
        Input: [ indices, data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the (possibly normalized) Levenshtein Edit Distance.

The inputs are variable-length sequences provided by SparseTensors
//   (hypothesis_indices, hypothesis_values, hypothesis_shape)
// and
//   (truth_indices, truth_values, truth_shape).
// 
// The inputs are:

*/








func editDistance( scope:Scope,hypothesisIndices: Output, hypothesisValues: Output, hypothesisShape: Output, truthIndices: Output, truthValues: Output, truthShape: Output, normalize :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["normalize"] = normalize

    let opspec = OpSpec(
        OpType: "EditDistance",
        Name: "Type",
        Input: [ hypothesisIndices, hypothesisValues, hypothesisShape, truthIndices, truthValues, truthShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.

See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
// ](http://arxiv.org/abs/1511.07289)

*/








func elu( scope:Scope,features: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Elu",
        Name: "Type",
        Input: [ features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes gradients for the exponential linear (Elu) operation.


*/








func eluGrad( scope:Scope,gradients: Output, outputs: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "EluGrad",
        Name: "Type",
        Input: [ gradients, outputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Encode strings into web-safe base64 format.

Refer to the following article for more information on base64 format:
// en.wikipedia.org/wiki/Base64. Base64 strings may have padding with '=' at the
// end so that the encoded has length multiple of 4. See Padding section of the
// link above.
// 
// Web-safe means that the encoder uses - and _ instead of + and /.

*/








func encodeBase64( scope:Scope,input: Output, pad :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["pad"] = pad

    let opspec = OpSpec(
        OpType: "EncodeBase64",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
JPEG-encode an image.

`image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.
// 
// The attr `format` can be used to override the color format of the encoded
// output.  Values can be:
// 
//  *    `''`: Use a default format based on the number of channels in the image.
//  *    `grayscale`: Output a grayscale JPEG image.  The `channels` dimension
//     of `image` must be 1.
//  *    `rgb`: Output an RGB JPEG image. The `channels` dimension
//     of `image` must be 3.
// 
// If `format` is not specified or is the empty string, a default format is picked
// in function of the number of channels in `image`:
// 
//  *    1: Output a grayscale image.
//  *    3: Output an RGB image.

*/








func encodeJpeg( scope:Scope,image: Output, format :String  , quality :UInt8  , progressive :Bool  , optimizeSize :Bool  , chromaDownsampling :Bool  , densityUnit :String  , xDensity :UInt8  , yDensity :UInt8  , xmpMetadata :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["format"] = format
    attrs["quality"] = quality
    attrs["progressive"] = progressive
    attrs["optimize_size"] = optimizeSize
    attrs["chroma_downsampling"] = chromaDownsampling
    attrs["density_unit"] = densityUnit
    attrs["x_density"] = xDensity
    attrs["y_density"] = yDensity
    attrs["xmp_metadata"] = xmpMetadata

    let opspec = OpSpec(
        OpType: "EncodeJpeg",
        Name: "Type",
        Input: [ image],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
PNG-encode an image.

`image` is a 3-D uint8 or uint16 Tensor of shape `[height, width, channels]`
// where `channels` is:
// 
//  *    1: for grayscale.
//  *    2: for grayscale + alpha.
//  *    3: for RGB.
//  *    4: for RGBA.
// 
// The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
// default or a value from 0 to 9.  9 is the highest compression level, generating
// the smallest output, but is slower.

*/








func encodePng( scope:Scope,image: Output, compression :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["compression"] = compression

    let opspec = OpSpec(
        OpType: "EncodePng",
        Name: "Type",
        Input: [ image],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Encode audio data using the WAV file format.

This operation will generate a string suitable to be saved out to create a .wav
// audio file. It will be encoded in the 16-bit PCM format. It takes in float
// values in the range -1.0f to 1.0f, and any outside that value will be clamped to
// that range.
// 
// `audio` is a 2-D float Tensor of shape `[length, channels]`.
// `sample_rate` is a scalar Tensor holding the rate to use (e.g. 44100).

*/








func encodeWav( scope:Scope,audio: Output, sampleRate: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "EncodeWav",
        Name: "Type",
        Input: [ audio, sampleRate],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates or finds a child frame, and makes `data` available to the child frame.

This op is used together with `Exit` to create loops in the graph.
// The unique `frame_name` is used by the `Executor` to identify frames. If
// `is_constant` is true, `output` is a constant in the child frame; otherwise
// it may be changed in the child frame. At most `parallel_iterations` iterations
// are run in parallel in the child frame.

*/








func enter( scope:Scope,data: Output, frameName :String  , isConstant :Bool  , parallelIterations :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["frame_name"] = frameName
    attrs["is_constant"] = isConstant
    attrs["parallel_iterations"] = parallelIterations

    let opspec = OpSpec(
        OpType: "Enter",
        Name: "Type",
        Input: [ data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the truth value of (x == y) element-wise.

 * NOTE * : `Equal` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func equal( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Equal",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the Gauss error function of `x` element-wise.


*/








func erf( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Erf",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the complementary error function of `x` element-wise.


*/








func erfc( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Erfc",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Exits the current frame to its parent frame.

Exit makes its input `data` available to the parent frame.

*/








func exit( scope:Scope,data: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Exit",
        Name: "Type",
        Input: [ data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes exponential of x element-wise.  \\(y = e^x\\).


*/








func exp( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Exp",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Inserts a dimension of 1 into a tensor's shape.

Given a tensor `input`, this operation inserts a dimension of 1 at the
// dimension index `dim` of `input`'s shape. The dimension index `dim` starts at
// zero; if you specify a negative number for `dim` it is counted backward from
// the end.
// 
// This operation is useful if you want to add a batch dimension to a single
// element. For example, if you have a single image of shape `[height, width,
// channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
// which will make the shape `[1, height, width, channels]`.
// 
// Other examples:
// 
// ```
// # 't' is a tensor of shape [2]
// shape(expand_dims(t, 0)) ==> [1, 2]
// shape(expand_dims(t, 1)) ==> [2, 1]
// shape(expand_dims(t, -1)) ==> [2, 1]
// 
// # 't2' is a tensor of shape [2, 3, 5]
// shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
// shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
// shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
// ```
// 
// This operation requires that:
// 
// `-1-input.dims() <= dim <= input.dims()`
// 
// This operation is related to `squeeze()`, which removes dimensions of
// size 1.

*/








func expandDims( scope:Scope,input: Output, dim: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ExpandDims",
        Name: "Type",
        Input: [ input, dim],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes exponential of x - 1 element-wise.

I.e., \\(y = (\exp x) - 1\\).

*/








func expm1( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Expm1",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Extracts a glimpse from the input tensor.

Returns a set of windows called glimpses extracted at location
// `offsets` from the input tensor. If the windows only partially
// overlaps the inputs, the non overlapping areas will be filled with
// random noise.
// 
// The result is a 4-D tensor of shape `[batch_size, glimpse_height,
// glimpse_width, channels]`. The channels and batch dimensions are the
// same as that of the input tensor. The height and width of the output
// windows are specified in the `size` parameter.
// 
// The argument `normalized` and `centered` controls how the windows are built:
// 
//  *  If the coordinates are normalized but not centered, 0.0 and 1.0
//   correspond to the minimum and maximum of each height and width
//   dimension.
//  *  If the coordinates are both normalized and centered, they range from
//   -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper
//   left corner, the lower right corner is located at (1.0, 1.0) and the
//   center is at (0, 0).
//  *  If the coordinates are not normalized they are interpreted as
//   numbers of pixels.

*/








func extractGlimpse( scope:Scope,input: Output, size: Output, offsets: Output, centered :Bool  , normalized :Bool  , uniformNoise :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["centered"] = centered
    attrs["normalized"] = normalized
    attrs["uniform_noise"] = uniformNoise

    let opspec = OpSpec(
        OpType: "ExtractGlimpse",
        Name: "Type",
        Input: [ input, size, offsets],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Extract `patches` from `images` and put them in the "depth" output dimension.


*/








func extractImagePatches( scope:Scope,images: Output, ksizes :[Int64]  , strides :[Int64]  , rates :[Int64]  , padding :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["ksizes"] = ksizes
    attrs["strides"] = strides
    attrs["rates"] = rates
    attrs["padding"] = padding

    let opspec = OpSpec(
        OpType: "ExtractImagePatches",
        Name: "Type",
        Input: [ images],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Fast Fourier transform.

Computes the 1-dimensional discrete Fourier transform over the inner-most
// dimension of `input`.

*/








func fft( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "FFT",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
2D fast Fourier transform.

Computes the 2-dimensional discrete Fourier transform over the inner-most
// 2 dimensions of `input`.

*/








func fft2D( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "FFT2D",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
3D fast Fourier transform.

Computes the 3-dimensional discrete Fourier transform over the inner-most 3
// dimensions of `input`.

*/








func fft3D( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "FFT3D",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A queue that produces elements in first-in first-out order.


*/








func fifoQueue( scope:Scope, shapes :[Shape]  , capacity :UInt8  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shapes"] = shapes
    attrs["capacity"] = capacity
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "FIFOQueue",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A queue that produces elements in first-in first-out order.


*/








func fifoQueueV2( scope:Scope, shapes :[Shape]  , capacity :UInt8  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shapes"] = shapes
    attrs["capacity"] = capacity
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "FIFOQueueV2",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Output a fact about factorials.


*/








func fact( scope:Scope )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Fact",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same type.

Attributes [min; max] define the clamping range for the 'inputs' data.  Op
// divides this range into 255 steps (total of 256 values), then replaces each
// 'inputs' value with the closest of the quantized step values.
// 'num_bits' is the bitwidth of the quantization; between 2 and 8, inclusive.
// 
// Quantization is called fake since the output is still in floating point.

*/








func fakeQuantWithMinMaxArgs( scope:Scope,inputs: Output, min :Float  , max :Float  , numBits :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["min"] = min
    attrs["max"] = max
    attrs["num_bits"] = numBits

    let opspec = OpSpec(
        OpType: "FakeQuantWithMinMaxArgs",
        Name: "Type",
        Input: [ inputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Compute gradients for a FakeQuantWithMinMaxArgs operation.


*/








func fakeQuantWithMinMaxArgsGradient( scope:Scope,gradients: Output, inputs: Output, min :Float  , max :Float  , numBits :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["min"] = min
    attrs["max"] = max
    attrs["num_bits"] = numBits

    let opspec = OpSpec(
        OpType: "FakeQuantWithMinMaxArgsGradient",
        Name: "Type",
        Input: [ gradients, inputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Fake-quantize the 'inputs' tensor of type float via global float scalars `min`

and `max` to 'outputs' tensor of same shape as `inputs`.
// 
// [min; max] is the clamping range for the 'inputs' data.  Op divides this range
// into 255 steps (total of 256 values), then replaces each 'inputs' value with the
// closest of the quantized step values.
// 'num_bits' is the bitwidth of the quantization; between 2 and 8, inclusive.
// 
// This operation has a gradient and thus allows for training `min` and `max` values.

*/








func fakeQuantWithMinMaxVars( scope:Scope,inputs: Output, min: Output, max: Output, numBits :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_bits"] = numBits

    let opspec = OpSpec(
        OpType: "FakeQuantWithMinMaxVars",
        Name: "Type",
        Input: [ inputs, min, max],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Compute gradients for a FakeQuantWithMinMaxVars operation.


*/








func fakeQuantWithMinMaxVarsGradient( scope:Scope,gradients: Output, inputs: Output, min: Output, max: Output, numBits :UInt8  )  ->(backpropsWrtInput: Output?, backpropWrtMin: Output?, backpropWrtMax: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_bits"] = numBits

    let opspec = OpSpec(
        OpType: "FakeQuantWithMinMaxVarsGradient",
        Name: "Type",
        Input: [ gradients, inputs, min, max],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Fake-quantize the 'inputs' tensor of type float and one of the shapes: `[d]`,

`[b, d]` `[b, h, w, d]` via per-channel floats `min` and `max` of shape `[d]`
// to 'outputs' tensor of same shape as `inputs`.
// 
// [min; max] is the clamping range for the 'inputs' data in the corresponding
// depth channel.  Op divides this range into 255 steps (total of 256 values), then
// replaces each 'inputs' value with the closest of the quantized step values.
// 'num_bits' is the bitwidth of the quantization; between 2 and 8, inclusive.
// 
// This operation has a gradient and thus allows for training `min` and `max` values.

*/








func fakeQuantWithMinMaxVarsPerChannel( scope:Scope,inputs: Output, min: Output, max: Output, numBits :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_bits"] = numBits

    let opspec = OpSpec(
        OpType: "FakeQuantWithMinMaxVarsPerChannel",
        Name: "Type",
        Input: [ inputs, min, max],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation.


*/








func fakeQuantWithMinMaxVarsPerChannelGradient( scope:Scope,gradients: Output, inputs: Output, min: Output, max: Output, numBits :UInt8  )  ->(backpropsWrtInput: Output?, backpropWrtMin: Output?, backpropWrtMax: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_bits"] = numBits

    let opspec = OpSpec(
        OpType: "FakeQuantWithMinMaxVarsPerChannelGradient",
        Name: "Type",
        Input: [ gradients, inputs, min, max],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Deprecated. Do not use.


*/








func fakeQueue( scope:Scope,resource: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "FakeQueue",
        Name: "Type",
        Input: [ resource],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a tensor filled with a scalar value.

This operation creates a tensor of shape `dims` and fills it with `value`.
// 
// For example:
// 
// ```
// # Output tensor has shape [2, 3].
// fill([2, 3], 9) ==> [[9, 9, 9]
//                      [9, 9, 9]]
// ```

*/








func fill( scope:Scope,dims: Output, value: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Fill",
        Name: "Type",
        Input: [ dims, value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a dataset containing elements of `input_dataset` matching `predicate`.

The `predicate` function must return a scalar boolean and accept the
// following arguments:
// 
//  *  One tensor for each component of an element of `input_dataset`.
//  *  One tensor for each value in `other_arguments`.

*/








func filterDataset( scope:Scope,inputDataset: Output, otherArguments: Output, predicate :TensorflowNameAttrList  , outputShapes :[Shape]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["predicate"] = predicate
    attrs["output_shapes"] = outputShapes

    let opspec = OpSpec(
        OpType: "FilterDataset",
        Name: "Type",
        Input: [ inputDataset, otherArguments],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a dataset that emits the records from one or more binary files.


*/








func fixedLengthRecordDataset( scope:Scope,filenames: Output, headerBytes: Output, recordBytes: Output, footerBytes: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "FixedLengthRecordDataset",
        Name: "Type",
        Input: [ filenames, headerBytes, recordBytes, footerBytes],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A Reader that outputs fixed-length records from a file.


*/








func fixedLengthRecordReader( scope:Scope, headerBytes :UInt8  , recordBytes :UInt8  , footerBytes :UInt8  , hopBytes :UInt8  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["header_bytes"] = headerBytes
    attrs["record_bytes"] = recordBytes
    attrs["footer_bytes"] = footerBytes
    attrs["hop_bytes"] = hopBytes
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "FixedLengthRecordReader",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A Reader that outputs fixed-length records from a file.


*/








func fixedLengthRecordReaderV2( scope:Scope, headerBytes :UInt8  , recordBytes :UInt8  , footerBytes :UInt8  , hopBytes :UInt8  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["header_bytes"] = headerBytes
    attrs["record_bytes"] = recordBytes
    attrs["footer_bytes"] = footerBytes
    attrs["hop_bytes"] = hopBytes
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "FixedLengthRecordReaderV2",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Generates labels for candidate sampling with a learned unigram distribution.

A unigram sampler could use a fixed unigram distribution read from a
// file or passed in as an in-memory array instead of building up the distribution
// from data on the fly. There is also an option to skew the distribution by
// applying a distortion power to the weights.
// 
// The vocabulary file should be in CSV-like format, with the last field
// being the weight associated with the word.
// 
// For each batch, this op picks a single set of sampled candidate labels.
// 
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.

*/








func fixedUnigramCandidateSampler( scope:Scope,trueClasses: Output, numTrue :UInt8  , numSampled :UInt8  , unique :Bool  , rangeMax :UInt8  , vocabFile :String  , distortion :Float  , numReservedIds :UInt8  , numShards :UInt8  , shard :UInt8  , unigrams :[Float]  , seed :UInt8  , seed2 :UInt8  )  ->(sampledCandidates: Output?, trueExpectedCount: Output?, sampledExpectedCount: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_true"] = numTrue
    attrs["num_sampled"] = numSampled
    attrs["unique"] = unique
    attrs["range_max"] = rangeMax
    attrs["vocab_file"] = vocabFile
    attrs["distortion"] = distortion
    attrs["num_reserved_ids"] = numReservedIds
    attrs["num_shards"] = numShards
    attrs["shard"] = shard
    attrs["unigrams"] = unigrams
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "FixedUnigramCandidateSampler",
        Name: "Type",
        Input: [ trueClasses],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Creates a dataset that applies `f` to the outputs of `input_dataset`.

Unlike MapDataset, the `f` in FlatMapDataset is expected to return a
// Dataset resource, and FlatMapDataset will flatten successive results
// into a single Dataset.

*/








func flatMapDataset( scope:Scope,inputDataset: Output, otherArguments: Output, f :TensorflowNameAttrList  , outputShapes :[Shape]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["f"] = f
    attrs["output_shapes"] = outputShapes

    let opspec = OpSpec(
        OpType: "FlatMapDataset",
        Name: "Type",
        Input: [ inputDataset, otherArguments],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns element-wise largest integer not greater than x.


*/








func floor( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Floor",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns x // y element-wise.

 * NOTE * : `FloorDiv` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func floorDiv( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "FloorDiv",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns element-wise remainder of division. When `x < 0` xor `y < 0` is

true, this follows Python semantics in that the result here is consistent
// with a flooring divide. E.g. `floor(x / y)  *  y + mod(x, y) = x`.
// 
//  * NOTE * : `FloorMod` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func floorMod( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "FloorMod",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Performs fractional average pooling on the input.

Fractional average pooling is similar to Fractional max pooling in the pooling
// region generation step. The only difference is that after pooling regions are
// generated, a mean operation is performed instead of a max operation in each
// pooling region.

*/








func fractionalAvgPool( scope:Scope,value: Output, poolingRatio :[Float]  , pseudoRandom :Bool  , overlapping :Bool  , deterministic :Bool  , seed :UInt8  , seed2 :UInt8  )  ->(output: Output?, rowPoolingSequence: Output?, colPoolingSequence: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["pooling_ratio"] = poolingRatio
    attrs["pseudo_random"] = pseudoRandom
    attrs["overlapping"] = overlapping
    attrs["deterministic"] = deterministic
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "FractionalAvgPool",
        Name: "Type",
        Input: [ value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Computes gradient of the FractionalAvgPool function.

Unlike FractionalMaxPoolGrad, we don't need to find arg_max for
// FractionalAvgPoolGrad, we just need to evenly back-propagate each element of
// out_backprop to those indices that form the same pooling cell. Therefore, we
// just need to know the shape of original input tensor, instead of the whole
// tensor.

*/








func fractionalAvgPoolGrad( scope:Scope,origInputTensorShape: Output, outBackprop: Output, rowPoolingSequence: Output, colPoolingSequence: Output, overlapping :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["overlapping"] = overlapping

    let opspec = OpSpec(
        OpType: "FractionalAvgPoolGrad",
        Name: "Type",
        Input: [ origInputTensorShape, outBackprop, rowPoolingSequence, colPoolingSequence],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Performs fractional max pooling on the input.

Fractional max pooling is slightly different than regular max pooling.  In
// regular max pooling, you downsize an input set by taking the maximum value of
// smaller N x N subsections of the set (often 2x2), and try to reduce the set by
// a factor of N, where N is an integer.  Fractional max pooling, as you might
// expect from the word "fractional", means that the overall reduction ratio N
// does not have to be an integer.
// 
// The sizes of the pooling regions are generated randomly but are fairly uniform.
// For example, let's look at the height dimension, and the constraints on the
// list of rows that will be pool boundaries.
// 
// First we define the following:
// 
// 1.  input_row_length : the number of rows from the input set
// 2.  output_row_length : which will be smaller than the input
// 3.  alpha = input_row_length / output_row_length : our reduction ratio
// 4.  K = floor(alpha)
// 5.  row_pooling_sequence : this is the result list of pool boundary rows
// 
// Then, row_pooling_sequence should satisfy:
// 
// 1.  a[0] = 0 : the first value of the sequence is 0
// 2.  a[end] = input_row_length : the last value of the sequence is the size
// 3.  K <= (a[i+1] - a[i]) <= K+1 : all intervals are K or K+1 size
// 4.  length(row_pooling_sequence) = output_row_length+1
// 
// For more details on fractional max pooling, see this paper:
// [Benjamin Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071)

*/








func fractionalMaxPool( scope:Scope,value: Output, poolingRatio :[Float]  , pseudoRandom :Bool  , overlapping :Bool  , deterministic :Bool  , seed :UInt8  , seed2 :UInt8  )  ->(output: Output?, rowPoolingSequence: Output?, colPoolingSequence: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["pooling_ratio"] = poolingRatio
    attrs["pseudo_random"] = pseudoRandom
    attrs["overlapping"] = overlapping
    attrs["deterministic"] = deterministic
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "FractionalMaxPool",
        Name: "Type",
        Input: [ value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Computes gradient of the FractionalMaxPool function.


*/








func fractionalMaxPoolGrad( scope:Scope,origInput: Output, origOutput: Output, outBackprop: Output, rowPoolingSequence: Output, colPoolingSequence: Output, overlapping :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["overlapping"] = overlapping

    let opspec = OpSpec(
        OpType: "FractionalMaxPoolGrad",
        Name: "Type",
        Input: [ origInput, origOutput, outBackprop, rowPoolingSequence, colPoolingSequence],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Batch normalization.

Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
// The size of 1D Tensors matches the dimension C of the 4D Tensors.

*/








func fusedBatchNorm( scope:Scope,x: Output, scale: Output, offset: Output, mean: Output, variance: Output, epsilon :Float  , dataFormat :String  , isTraining :Bool  )  ->(y: Output?, batchMean: Output?, batchVariance: Output?, reserveSpace1: Output?, reserveSpace2: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["epsilon"] = epsilon
    attrs["data_format"] = dataFormat
    attrs["is_training"] = isTraining

    let opspec = OpSpec(
        OpType: "FusedBatchNorm",
        Name: "Type",
        Input: [ x, scale, offset, mean, variance],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1),op?.output(4 - 1),op?.output(5 - 1))
}

/*
Gradient for batch normalization.

Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
// The size of 1D Tensors matches the dimension C of the 4D Tensors.

*/








func fusedBatchNormGrad( scope:Scope,yBackprop: Output, x: Output, scale: Output, reserveSpace1: Output, reserveSpace2: Output, epsilon :Float  , dataFormat :String  , isTraining :Bool  )  ->(xBackprop: Output?, scaleBackprop: Output?, offsetBackprop: Output?, reserveSpace3: Output?, reserveSpace4: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["epsilon"] = epsilon
    attrs["data_format"] = dataFormat
    attrs["is_training"] = isTraining

    let opspec = OpSpec(
        OpType: "FusedBatchNormGrad",
        Name: "Type",
        Input: [ yBackprop, x, scale, reserveSpace1, reserveSpace2],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1),op?.output(4 - 1),op?.output(5 - 1))
}

/*
Performs a padding as a preprocess during a convolution.

Similar to FusedResizeAndPadConv2d, this op allows for an optimized
// implementation where the spatial padding transformation stage is fused with the
// im2col lookup, but in this case without the bilinear filtering required for
// resizing. Fusing the padding prevents the need to write out the intermediate
// results as whole tensors, reducing memory pressure, and we can get some latency
// gains by merging the transformation calculations.
// The data_format attribute for Conv2D isn't supported by this op, and 'NHWC'
// order is used instead.
// Internally this op uses a single per-graph scratch buffer, which means that it
// will block if multiple versions are being run in parallel. This is because this
// operator is primarily an optimization to minimize memory usage.

*/








func fusedPadConv2D( scope:Scope,input: Output, paddings: Output, filter: Output, mode :String  , strides :[Int64]  , padding :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["mode"] = mode
    attrs["strides"] = strides
    attrs["padding"] = padding

    let opspec = OpSpec(
        OpType: "FusedPadConv2D",
        Name: "Type",
        Input: [ input, paddings, filter],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Performs a resize and padding as a preprocess during a convolution.

It's often possible to do spatial transformations more efficiently as part of
// the packing stage of a convolution, so this op allows for an optimized
// implementation where these stages are fused together. This prevents the need to
// write out the intermediate results as whole tensors, reducing memory pressure,
// and we can get some latency gains by merging the transformation calculations.
// The data_format attribute for Conv2D isn't supported by this op, and defaults to
// 'NHWC' order.
// Internally this op uses a single per-graph scratch buffer, which means that it
// will block if multiple versions are being run in parallel. This is because this
// operator is primarily an optimization to minimize memory usage.

*/








func fusedResizeAndPadConv2D( scope:Scope,input: Output, size: Output, paddings: Output, filter: Output, resizeAlignCorners :Bool  , mode :String  , strides :[Int64]  , padding :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["resize_align_corners"] = resizeAlignCorners
    attrs["mode"] = mode
    attrs["strides"] = strides
    attrs["padding"] = padding

    let opspec = OpSpec(
        OpType: "FusedResizeAndPadConv2D",
        Name: "Type",
        Input: [ input, size, paddings, filter],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Gather slices from `params` according to `indices`.

`indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
// Produces an output tensor with shape `indices.shape + params.shape[1:]` where:
// 
// ```python
//     # Scalar indices
//     output[:, ..., :] = params[indices, :, ... :]
// 
//     # Vector indices
//     output[i, :, ..., :] = params[indices[i], :, ... :]
// 
//     # Higher rank indices
//     output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
// ```
// 
// If `indices` is a permutation and `len(indices) == params.shape[0]` then
// this operation will permute `params` accordingly.
// 
// `validate_indices`: DEPRECATED. If this operation is assigned to CPU, values in
// `indices` are always validated to be within range. If assigned to GPU,
// out-of-bound indices result in safe but unspecified behavior, which may include
// raising an error.
// 
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
// </div>

*/








func gather( scope:Scope,params: Output, indices: Output, validateIndices :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["validate_indices"] = validateIndices

    let opspec = OpSpec(
        OpType: "Gather",
        Name: "Type",
        Input: [ params, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Gather values or slices from `params` according to `indices`.

`indices` is an integer tensor containing indices into `params`.  The last
// dimension of `indices` can be at most the rank of `params`:
// 
//     indices.shape[-1] <= params.rank
// 
// The last dimension of `indices` corresponds to elements
// (if `indices.shape[-1] = params.rank`) or slices
// (if `indices.shape[-1] < params.rank`) along dimension `indices.shape[-1]`
// of `params`.  The output tensor has shape
// 
//     indices.shape[:-1] + params.shape[indices.shape[-1]:]
// 
// Some examples below.
// 
// Simple indexing into a matrix:
// 
// ```python
//     indices = [[0, 0], [1, 1]]
//     params = [['a', 'b'], ['c', 'd']]
//     output = ['a', 'd']
// ```
// 
// Slice indexing into a matrix:
// 
// ```python
//     indices = [[1], [0]]
//     params = [['a', 'b'], ['c', 'd']]
//     output = [['c', 'd'], ['a', 'b']]
// ```
// 
// Indexing into a 3-tensor:
// 
// ```python
//     indices = [[1]]
//     params = [[['a0', 'b0'], ['c0', 'd0']],
//               [['a1', 'b1'], ['c1', 'd1']]]
//     output = [[['a1', 'b1'], ['c1', 'd1']]]
// 
// 
//     indices = [[0, 1], [1, 0]]
//     params = [[['a0', 'b0'], ['c0', 'd0']],
//               [['a1', 'b1'], ['c1', 'd1']]]
//     output = [['c0', 'd0'], ['a1', 'b1']]
// 
// 
//     indices = [[0, 0, 1], [1, 0, 1]]
//     params = [[['a0', 'b0'], ['c0', 'd0']],
//               [['a1', 'b1'], ['c1', 'd1']]]
//     output = ['b0', 'b1']
// ```
// 
// Batched indexing into a matrix:
// 
// ```python
//     indices = [[[0, 0]], [[0, 1]]]
//     params = [['a', 'b'], ['c', 'd']]
//     output = [['a'], ['b']]
// ```
// 
// Batched slice indexing into a matrix:
// 
// ```python
//     indices = [[[1]], [[0]]]
//     params = [['a', 'b'], ['c', 'd']]
//     output = [[['c', 'd']], [['a', 'b']]]
// ```
// 
// Batched indexing into a 3-tensor:
// 
// ```python
//     indices = [[[1]], [[0]]]
//     params = [[['a0', 'b0'], ['c0', 'd0']],
//               [['a1', 'b1'], ['c1', 'd1']]]
//     output = [[[['a1', 'b1'], ['c1', 'd1']]],
//               [[['a0', 'b0'], ['c0', 'd0']]]]
// 
//     indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
//     params = [[['a0', 'b0'], ['c0', 'd0']],
//               [['a1', 'b1'], ['c1', 'd1']]]
//     output = [[['c0', 'd0'], ['a1', 'b1']],
//               [['a0', 'b0'], ['c1', 'd1']]]
// 
// 
//     indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
//     params = [[['a0', 'b0'], ['c0', 'd0']],
//               [['a1', 'b1'], ['c1', 'd1']]]
//     output = [['b0', 'b1'], ['d0', 'c1']]
// ```

*/








func gatherNd( scope:Scope,params: Output, indices: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "GatherNd",
        Name: "Type",
        Input: [ params, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Store the input tensor in the state of the current session.


*/








func getSessionHandle( scope:Scope,value: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "GetSessionHandle",
        Name: "Type",
        Input: [ value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Store the input tensor in the state of the current session.


*/








func getSessionHandleV2( scope:Scope,value: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "GetSessionHandleV2",
        Name: "Type",
        Input: [ value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Get the value of the tensor specified by its handle.


*/








func getSessionTensor( scope:Scope,handle: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "GetSessionTensor",
        Name: "Type",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the truth value of (x > y) element-wise.

 * NOTE * : `Greater` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func greater( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Greater",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the truth value of (x >= y) element-wise.

 * NOTE * : `GreaterEqual` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func greaterEqual( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "GreaterEqual",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a dataset that computes a windowed group-by on `input_dataset`.

// TODO(mrry): Support non-int64 keys.

*/








func groupByWindowDataset( scope:Scope,inputDataset: Output, keyFuncOtherArguments: Output, reduceFuncOtherArguments: Output, windowSize: Output, keyFunc :TensorflowNameAttrList  , reduceFunc :TensorflowNameAttrList  , outputShapes :[Shape]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["key_func"] = keyFunc
    attrs["reduce_func"] = reduceFunc
    attrs["output_shapes"] = outputShapes

    let opspec = OpSpec(
        OpType: "GroupByWindowDataset",
        Name: "Type",
        Input: [ inputDataset, keyFuncOtherArguments, reduceFuncOtherArguments, windowSize],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Convert one or more images from HSV to RGB.

Outputs a tensor of the same shape as the `images` tensor, containing the RGB
// value of the pixels. The output is only well defined if the value in `images`
// are in `[0,1]`.
// 
// See `rgb_to_hsv` for a description of the HSV encoding.

*/








func hsvToRGB( scope:Scope,images: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "HSVToRGB",
        Name: "Type",
        Input: [ images],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a non-initialized hash table.

This op creates a hash table, specifying the type of its keys and values.
// Before using the table you will have to initialize it.  After initialization the
// table will be immutable.

*/








func hashTable( scope:Scope, container :String  , sharedName :String  , useNodeNameSharing :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName
    attrs["use_node_name_sharing"] = useNodeNameSharing

    let opspec = OpSpec(
        OpType: "HashTable",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a non-initialized hash table.

This op creates a hash table, specifying the type of its keys and values.
// Before using the table you will have to initialize it.  After initialization the
// table will be immutable.

*/








func hashTableV2( scope:Scope, container :String  , sharedName :String  , useNodeNameSharing :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName
    attrs["use_node_name_sharing"] = useNodeNameSharing

    let opspec = OpSpec(
        OpType: "HashTableV2",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Outputs a `Summary` protocol buffer with a histogram.

The generated
// [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
// has one summary value containing a histogram for `values`.
// 
// This op reports an `InvalidArgument` error if any value is not finite.

*/








func histogramSummary( scope:Scope,tag: Output, values: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "HistogramSummary",
        Name: "Type",
        Input: [ tag, values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Inverse fast Fourier transform.

Computes the inverse 1-dimensional discrete Fourier transform over the
// inner-most dimension of `input`.

*/








func ifft( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "IFFT",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Inverse 2D fast Fourier transform.

Computes the inverse 2-dimensional discrete Fourier transform over the
// inner-most 2 dimensions of `input`.

*/








func ifft2D( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "IFFT2D",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Inverse 3D fast Fourier transform.

Computes the inverse 3-dimensional discrete Fourier transform over the
// inner-most 3 dimensions of `input`.

*/








func ifft3D( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "IFFT3D",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Inverse real-valued fast Fourier transform.

Computes the inverse 1-dimensional discrete Fourier transform of a real-valued
// signal over the inner-most dimension of `input`.
// 
// The inner-most dimension of `input` is assumed to be the result of `RFFT`: the
// `fft_length / 2 + 1` unique components of the DFT of a real-valued signal. If
// `fft_length` is not provided, it is computed from the size of the inner-most
// dimension of `input` (`fft_length = 2  *  (inner - 1)`). If the FFT length used to
// compute `input` is odd, it should be provided since it cannot be inferred
// properly.

*/








func irfft( scope:Scope,input: Output, fftLength: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "IRFFT",
        Name: "Type",
        Input: [ input, fftLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Inverse 2D real-valued fast Fourier transform.

Computes the inverse 2-dimensional discrete Fourier transform of a real-valued
// signal over the inner-most 2 dimensions of `input`.
// 
// The inner-most 2 dimensions of `input` are assumed to be the result of `RFFT2D`:
// The inner-most dimension contains the `fft_length / 2 + 1` unique components of
// the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
// from the size of the inner-most 2 dimensions of `input`. If the FFT length used
// to compute `input` is odd, it should be provided since it cannot be inferred
// properly.

*/








func irfft2D( scope:Scope,input: Output, fftLength: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "IRFFT2D",
        Name: "Type",
        Input: [ input, fftLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Inverse 3D real-valued fast Fourier transform.

Computes the inverse 3-dimensional discrete Fourier transform of a real-valued
// signal over the inner-most 3 dimensions of `input`.
// 
// The inner-most 3 dimensions of `input` are assumed to be the result of `RFFT3D`:
// The inner-most dimension contains the `fft_length / 2 + 1` unique components of
// the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
// from the size of the inner-most 3 dimensions of `input`. If the FFT length used
// to compute `input` is odd, it should be provided since it cannot be inferred
// properly.

*/








func irfft3D( scope:Scope,input: Output, fftLength: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "IRFFT3D",
        Name: "Type",
        Input: [ input, fftLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Return a tensor with the same shape and contents as the input tensor or value.


*/








func identity( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Identity",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A Reader that outputs the queued work as both the key and value.

To use, enqueue strings in a Queue.  ReaderRead will take the front
// work string and output (work, work).

*/








func identityReader( scope:Scope, container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "IdentityReader",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A Reader that outputs the queued work as both the key and value.

To use, enqueue strings in a Queue.  ReaderRead will take the front
// work string and output (work, work).

*/








func identityReaderV2( scope:Scope, container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "IdentityReaderV2",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Compute the lower regularized incomplete Gamma function `Q(a, x)`.

The lower regularized incomplete Gamma function is defined as:
// 
// 
// \\(P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)\\)
// 
// where
// 
// \\(gamma(a, x) = int_{0}// ^{x} t// ^{a-1} exp(-t) dt\\)
// 
// is the lower incomplete Gamma function.
// 
// Note, above `Q(a, x)` (`Igammac`) is the upper regularized complete
// Gamma function.

*/








func igamma( scope:Scope,a: Output, x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Igamma",
        Name: "Type",
        Input: [ a, x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Compute the upper regularized incomplete Gamma function `Q(a, x)`.

The upper regularized incomplete Gamma function is defined as:
// 
// \\(Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)\\)
// 
// where
// 
// \\(Gamma(a, x) = int_{x}// ^{\infty} t// ^{a-1} exp(-t) dt\\)
// 
// is the upper incomplete Gama function.
// 
// Note, above `P(a, x)` (`Igamma`) is the lower regularized complete
// Gamma function.

*/








func igammac( scope:Scope,a: Output, x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Igammac",
        Name: "Type",
        Input: [ a, x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the imaginary part of a complex number.

Given a tensor `input` of complex numbers, this operation returns a tensor of
// type `float` that is the imaginary part of each element in `input`. All
// elements in `input` must be complex numbers of the form \\(a + bj\\), where  * a * 
// is the real part and  * b *  is the imaginary part returned by this operation.
// 
// For example:
// 
// ```
// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
// tf.imag(input) ==> [4.75, 5.75]
// ```

*/








func imag( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Imag",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Outputs a `Summary` protocol buffer with images.

The summary has up to `max_images` summary values containing images. The
// images are built from `tensor` which must be 4-D with shape `[batch_size,
// height, width, channels]` and where `channels` can be:
// 
//  *   1: `tensor` is interpreted as Grayscale.
//  *   3: `tensor` is interpreted as RGB.
//  *   4: `tensor` is interpreted as RGBA.
// 
// The images have the same number of channels as the input tensor. For float
// input, the values are normalized one image at a time to fit in the range
// `[0, 255]`.  `uint8` values are unchanged.  The op uses two different
// normalization algorithms:
// 
//  *   If the input values are all positive, they are rescaled so the largest one
//    is 255.
// 
//  *   If any input value is negative, the values are shifted so input value 0.0
//    is at 127.  They are then rescaled so that either the smallest value is 0,
//    or the largest one is 255.
// 
// The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
// build the `tag` of the summary values:
// 
//  *   If `max_images` is 1, the summary value tag is ' * tag * /image'.
//  *   If `max_images` is greater than 1, the summary value tags are
//    generated sequentially as ' * tag * /image/0', ' * tag * /image/1', etc.
// 
// The `bad_color` argument is the color to use in the generated images for
// non-finite input values.  It is a `unit8` 1-D tensor of length `channels`.
// Each element must be in the range `[0, 255]` (It represents the value of a
// pixel in the output image).  Non-finite values in the input tensor are
// replaced by this tensor in the output image.  The default value is the color
// red.

*/








func imageSummary( scope:Scope,tag: Output, tensor: Output, maxImages :UInt8  , badColor :Tensor  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["max_images"] = maxImages
    attrs["bad_color"] = badColor

    let opspec = OpSpec(
        OpType: "ImageSummary",
        Name: "Type",
        Input: [ tag, tensor],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns immutable tensor from memory region.

The current implementation memmaps the tensor from a file.

*/








func immutableConst( scope:Scope, shape :Shape  , memoryRegionName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shape"] = shape
    attrs["memory_region_name"] = memoryRegionName

    let opspec = OpSpec(
        OpType: "ImmutableConst",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Says whether the targets are in the top `K` predictions.

This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
// prediction for the target class is among the top `k` predictions among
// all predictions for example `i`. Note that the behavior of `InTopK` differs
// from the `TopK` op in its handling of ties; if multiple classes have the
// same prediction value and straddle the top-`k` boundary, all of those
// classes are considered to be in the top `k`.
// 
// More formally, let
// 
//   \\(predictions_i\\) be the predictions for all classes for example `i`,
//   \\(targets_i\\) be the target class for example `i`,
//   \\(out_i\\) be the output for example `i`,
// 
// $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$

*/








func inTopK( scope:Scope,predictions: Output, targets: Output, k :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["k"] = k

    let opspec = OpSpec(
        OpType: "InTopK",
        Name: "Type",
        Input: [ predictions, targets],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Table initializer that takes two tensors for keys and values respectively.


*/








func initializeTable( scope:Scope,tableHandle: Output, keys: Output, values: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "InitializeTable",
        Name: "Type",
        Input: [ tableHandle, keys, values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Initializes a table from a text file.

It inserts one key-value pair into the table for each line of the file.
// The key and value is extracted from the whole line content, elements from the
// split line based on `delimiter` or the line number (starting from zero).
// Where to extract the key and value from a line is specified by `key_index` and
// `value_index`.
// 
// - A value of -1 means use the line number(starting from zero), expects `int64`.
// - A value of -2 means use the whole line content, expects `string`.
// - A value >= 0 means use the index (starting at zero) of the split line based
//   on `delimiter`.

*/








func initializeTableFromTextFile( scope:Scope,tableHandle: Output, filename: Output, keyIndex :UInt8  , valueIndex :UInt8  , vocabSize :UInt8  , delimiter :String  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["key_index"] = keyIndex
    attrs["value_index"] = valueIndex
    attrs["vocab_size"] = vocabSize
    attrs["delimiter"] = delimiter

    let opspec = OpSpec(
        OpType: "InitializeTableFromTextFile",
        Name: "Type",
        Input: [ tableHandle, filename],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Initializes a table from a text file.

It inserts one key-value pair into the table for each line of the file.
// The key and value is extracted from the whole line content, elements from the
// split line based on `delimiter` or the line number (starting from zero).
// Where to extract the key and value from a line is specified by `key_index` and
// `value_index`.
// 
// - A value of -1 means use the line number(starting from zero), expects `int64`.
// - A value of -2 means use the whole line content, expects `string`.
// - A value >= 0 means use the index (starting at zero) of the split line based
//   on `delimiter`.

*/








func initializeTableFromTextFileV2( scope:Scope,tableHandle: Output, filename: Output, keyIndex :UInt8  , valueIndex :UInt8  , vocabSize :UInt8  , delimiter :String  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["key_index"] = keyIndex
    attrs["value_index"] = valueIndex
    attrs["vocab_size"] = vocabSize
    attrs["delimiter"] = delimiter

    let opspec = OpSpec(
        OpType: "InitializeTableFromTextFileV2",
        Name: "Type",
        Input: [ tableHandle, filename],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Table initializer that takes two tensors for keys and values respectively.


*/








func initializeTableV2( scope:Scope,tableHandle: Output, keys: Output, values: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "InitializeTableV2",
        Name: "Type",
        Input: [ tableHandle, keys, values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Computes the reciprocal of x element-wise.

I.e., \\(y = 1 / x\\).

*/








func inv( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Inv",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradient for the inverse of `x` wrt its input.

Specifically, `grad = -dy  *  y * y`, where `y = 1/x`, and `dy`
// is the corresponding input gradient.

*/








func invGrad( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "InvGrad",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the inverse permutation of a tensor.

This operation computes the inverse of an index permutation. It takes a 1-D
// integer tensor `x`, which represents the indices of a zero-based array, and
// swaps each value with its index position. In other words, for an output tensor
// `y` and an input tensor `x`, this operation computes the following:
// 
// `y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`
// 
// The values must include 0. There can be no duplicate values or negative values.
// 
// For example:
// 
// ```
// # tensor `x` is [3, 4, 0, 2, 1]
// invert_permutation(x) ==> [2, 4, 3, 0, 1]
// ```

*/








func invertPermutation( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "InvertPermutation",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns which elements of x are finite.

@compatibility(numpy)
// Equivalent to np.isfinite
// @end_compatibility

*/








func isFinite( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "IsFinite",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns which elements of x are Inf.

@compatibility(numpy)
// Equivalent to np.isinf
// @end_compatibility

*/








func isInf( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "IsInf",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns which elements of x are NaN.

@compatibility(numpy)
// Equivalent to np.isnan
// @end_compatibility

*/








func isNan( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "IsNan",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Checks whether a tensor has been initialized.

Outputs boolean scalar indicating whether the tensor has been initialized.

*/








func isVariableInitialized( scope:Scope,ref: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "IsVariableInitialized",
        Name: "Type",
        Input: [ ref],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A container for an iterator resource.


*/








func iterator( scope:Scope, sharedName :String  , container :String  , outputShapes :[Shape]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shared_name"] = sharedName
    attrs["container"] = container
    attrs["output_shapes"] = outputShapes

    let opspec = OpSpec(
        OpType: "Iterator",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Releases any resources used by the given iterator.


*/








func iteratorDispose( scope:Scope,iterator: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "IteratorDispose",
        Name: "Type",
        Input: [ iterator],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Gets the next output from the given iterator.


*/








func iteratorGetNext( scope:Scope,iterator: Output, outputShapes :[Shape]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["output_shapes"] = outputShapes

    let opspec = OpSpec(
        OpType: "IteratorGetNext",
        Name: "Type",
        Input: [ iterator],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
L2 Loss.

Computes half the L2 norm of a tensor without the `sqrt`:
// 
//     output = sum(t  *  *  2) / 2

*/








func l2Loss( scope:Scope,t: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "L2Loss",
        Name: "Type",
        Input: [ t],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Local Response Normalization.

The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
// dimension), and each vector is normalized independently.  Within a given vector,
// each component is divided by the weighted, squared sum of inputs within
// `depth_radius`.  In detail,
// 
//     sqr_sum[a, b, c, d] =
//         sum(input[a, b, c, d - depth_radius : d + depth_radius + 1]  *  *  2)
//     output = input / (bias + alpha  *  sqr_sum)  *  *  beta
// 
// For details, see [Krizhevsky et al., ImageNet classification with deep
// convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

*/








func lrn( scope:Scope,input: Output, depthRadius :UInt8  , bias :Float  , alpha :Float  , beta :Float  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["depth_radius"] = depthRadius
    attrs["bias"] = bias
    attrs["alpha"] = alpha
    attrs["beta"] = beta

    let opspec = OpSpec(
        OpType: "LRN",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Gradients for Local Response Normalization.


*/








func lrnGrad( scope:Scope,inputGrads: Output, inputImage: Output, outputImage: Output, depthRadius :UInt8  , bias :Float  , alpha :Float  , beta :Float  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["depth_radius"] = depthRadius
    attrs["bias"] = bias
    attrs["alpha"] = alpha
    attrs["beta"] = beta

    let opspec = OpSpec(
        OpType: "LRNGrad",
        Name: "Type",
        Input: [ inputGrads, inputImage, outputImage],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Generates labels for candidate sampling with a learned unigram distribution.

See explanations of candidate sampling and the data formats at
// go/candidate-sampling.
// 
// For each batch, this op picks a single set of sampled candidate labels.
// 
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.

*/








func learnedUnigramCandidateSampler( scope:Scope,trueClasses: Output, numTrue :UInt8  , numSampled :UInt8  , unique :Bool  , rangeMax :UInt8  , seed :UInt8  , seed2 :UInt8  )  ->(sampledCandidates: Output?, trueExpectedCount: Output?, sampledExpectedCount: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_true"] = numTrue
    attrs["num_sampled"] = numSampled
    attrs["unique"] = unique
    attrs["range_max"] = rangeMax
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "LearnedUnigramCandidateSampler",
        Name: "Type",
        Input: [ trueClasses],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Returns the truth value of (x < y) element-wise.

 * NOTE * : `Less` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func less( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Less",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the truth value of (x <= y) element-wise.

 * NOTE * : `LessEqual` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func lessEqual( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "LessEqual",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the log of the absolute value of `Gamma(x)` element-wise.


*/








func lgamma( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Lgamma",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Generates values in an interval.

A sequence of `num` evenly-spaced values are generated beginning at `start`.
// If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
// so that the last one is exactly `stop`.
// 
// For example:
// 
// ```
// tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
// ```

*/








func linSpace( scope:Scope,start: Output, stop: Output, num: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "LinSpace",
        Name: "Type",
        Input: [ start, stop, num],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the difference between two lists of numbers or strings.

Given a list `x` and a list `y`, this operation returns a list `out` that
// represents all values that are in `x` but not in `y`. The returned list `out`
// is sorted in the same order that the numbers appear in `x` (duplicates are
// preserved). This operation also returns a list `idx` that represents the
// position of each `out` element in `x`. In other words:
// 
// `out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`
// 
// For example, given this input:
// 
// ```
// x = [1, 2, 3, 4, 5, 6]
// y = [1, 3, 5]
// ```
// 
// This operation would return:
// 
// ```
// out ==> [2, 4, 6]
// idx ==> [1, 3, 5]
// ```

*/








func listDiff( scope:Scope,x: Output, y: Output )  ->(out: Output?, idx: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ListDiff",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Computes natural logarithm of x element-wise.

I.e., \\(y = \log_e x\\).

*/








func log( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Log",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes natural logarithm of (1 + x) element-wise.

I.e., \\(y = \log_e (1 + x)\\).

*/








func log1p( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Log1p",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes log softmax activations.

For each batch `i` and class `j` we have
// 
//     logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))

*/








func logSoftmax( scope:Scope,logits: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "LogSoftmax",
        Name: "Type",
        Input: [ logits],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Generates labels for candidate sampling with a log-uniform distribution.

See explanations of candidate sampling and the data formats at
// go/candidate-sampling.
// 
// For each batch, this op picks a single set of sampled candidate labels.
// 
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.

*/








func logUniformCandidateSampler( scope:Scope,trueClasses: Output, numTrue :UInt8  , numSampled :UInt8  , unique :Bool  , rangeMax :UInt8  , seed :UInt8  , seed2 :UInt8  )  ->(sampledCandidates: Output?, trueExpectedCount: Output?, sampledExpectedCount: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_true"] = numTrue
    attrs["num_sampled"] = numSampled
    attrs["unique"] = unique
    attrs["range_max"] = rangeMax
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "LogUniformCandidateSampler",
        Name: "Type",
        Input: [ trueClasses],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Returns the truth value of x AND y element-wise.

 * NOTE * : `LogicalAnd` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func logicalAnd( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "LogicalAnd",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the truth value of NOT x element-wise.


*/








func logicalNot( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "LogicalNot",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the truth value of x OR y element-wise.

 * NOTE * : `LogicalOr` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func logicalOr( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "LogicalOr",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Outputs all keys and values in the table.


*/








func lookupTableExport( scope:Scope,tableHandle: Output )  ->(keys: Output?, values: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "LookupTableExport",
        Name: "Type",
        Input: [ tableHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Outputs all keys and values in the table.


*/








func lookupTableExportV2( scope:Scope,tableHandle: Output )  ->(keys: Output?, values: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "LookupTableExportV2",
        Name: "Type",
        Input: [ tableHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Looks up keys in a table, outputs the corresponding values.

The tensor `keys` must of the same type as the keys of the table.
// The output `values` is of the type of the table values.
// 
// The scalar `default_value` is the value output for keys not present in the
// table. It must also be of the same type as the table values.

*/








func lookupTableFind( scope:Scope,tableHandle: Output, keys: Output, defaultValue: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "LookupTableFind",
        Name: "Type",
        Input: [ tableHandle, keys, defaultValue],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Looks up keys in a table, outputs the corresponding values.

The tensor `keys` must of the same type as the keys of the table.
// The output `values` is of the type of the table values.
// 
// The scalar `default_value` is the value output for keys not present in the
// table. It must also be of the same type as the table values.

*/








func lookupTableFindV2( scope:Scope,tableHandle: Output, keys: Output, defaultValue: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "LookupTableFindV2",
        Name: "Type",
        Input: [ tableHandle, keys, defaultValue],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Replaces the contents of the table with the specified keys and values.

The tensor `keys` must be of the same type as the keys of the table.
// The tensor `values` must be of the type of the table values.

*/








func lookupTableImport( scope:Scope,tableHandle: Output, keys: Output, values: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "LookupTableImport",
        Name: "Type",
        Input: [ tableHandle, keys, values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Replaces the contents of the table with the specified keys and values.

The tensor `keys` must be of the same type as the keys of the table.
// The tensor `values` must be of the type of the table values.

*/








func lookupTableImportV2( scope:Scope,tableHandle: Output, keys: Output, values: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "LookupTableImportV2",
        Name: "Type",
        Input: [ tableHandle, keys, values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Updates the table to associates keys with values.

The tensor `keys` must be of the same type as the keys of the table.
// The tensor `values` must be of the type of the table values.

*/








func lookupTableInsert( scope:Scope,tableHandle: Output, keys: Output, values: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "LookupTableInsert",
        Name: "Type",
        Input: [ tableHandle, keys, values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Updates the table to associates keys with values.

The tensor `keys` must be of the same type as the keys of the table.
// The tensor `values` must be of the type of the table values.

*/








func lookupTableInsertV2( scope:Scope,tableHandle: Output, keys: Output, values: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "LookupTableInsertV2",
        Name: "Type",
        Input: [ tableHandle, keys, values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Computes the number of elements in the given table.


*/








func lookupTableSize( scope:Scope,tableHandle: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "LookupTableSize",
        Name: "Type",
        Input: [ tableHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the number of elements in the given table.


*/








func lookupTableSizeV2( scope:Scope,tableHandle: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "LookupTableSizeV2",
        Name: "Type",
        Input: [ tableHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Forwards the input to the output.

This operator represents the loop termination condition used by the
// "pivot" switches of a loop.

*/








func loopCond( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "LoopCond",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Makes a new iterator from the given `dataset` and stores it in `iterator`.

This operation may be executed multiple times. Each execution will reset the
// iterator in `iterator` to the first element of `dataset`.

*/








func makeIterator( scope:Scope,dataset: Output, iterator: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "MakeIterator",
        Name: "Type",
        Input: [ dataset, iterator],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Creates a dataset that applies `f` to the outputs of `input_dataset`.


*/








func mapDataset( scope:Scope,inputDataset: Output, otherArguments: Output, f :TensorflowNameAttrList  , outputShapes :[Shape]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["f"] = f
    attrs["output_shapes"] = outputShapes

    let opspec = OpSpec(
        OpType: "MapDataset",
        Name: "Type",
        Input: [ inputDataset, otherArguments],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Multiply the matrix "a" by the matrix "b".

The inputs must be two-dimensional matrices and the inner dimension of
// "a" (after being transposed if transpose_a is true) must match the
// outer dimension of "b" (after being transposed if transposed_b is
// true).
// 
//  * Note * : The default kernel implementation for MatMul on GPUs uses
// cublas.

*/








func matMul( scope:Scope,a: Output, b: Output, transposeA :Bool  , transposeB :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["transpose_a"] = transposeA
    attrs["transpose_b"] = transposeB

    let opspec = OpSpec(
        OpType: "MatMul",
        Name: "Type",
        Input: [ a, b],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the set of files matching one or more glob patterns.

Note that this routine only supports wildcard characters in the
// basename portion of the pattern, not in the directory portion.

*/








func matchingFiles( scope:Scope,pattern: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "MatchingFiles",
        Name: "Type",
        Input: [ pattern],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Copy a tensor setting everything outside a central band in each innermost matrix

to zero.
// 
// The `band` part is computed as follows:
// Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
// tensor with the same shape where
// 
// `band[i, j, k, ..., m, n] = in_band(m, n)  *  input[i, j, k, ..., m, n]`.
// 
// The indicator function
// 
// `in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) &&
//                  (num_upper < 0 || (n-m) <= num_upper)`.
// 
// For example:
// 
// ```
// # if 'input' is [[ 0,  1,  2, 3]
//                  [-1,  0,  1, 2]
//                  [-2, -1,  0, 1]
//                  [-3, -2, -1, 0]],
// 
// tf.matrix_band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
//                                        [-1,  0,  1, 2]
//                                        [ 0, -1,  0, 1]
//                                        [ 0,  0, -1, 0]],
// 
// tf.matrix_band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
//                                       [-1,  0,  1, 0]
//                                       [-2, -1,  0, 1]
//                                       [ 0, -2, -1, 0]]
// ```
// 
// Useful special cases:
// 
// ```
//  tf.matrix_band_part(input, 0, -1) ==> Upper triangular part.
//  tf.matrix_band_part(input, -1, 0) ==> Lower triangular part.
//  tf.matrix_band_part(input, 0, 0) ==> Diagonal.
// ```

*/








func matrixBandPart( scope:Scope,input: Output, numLower: Output, numUpper: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "MatrixBandPart",
        Name: "Type",
        Input: [ input, numLower, numUpper],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the determinant of one ore more square matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
// form square matrices. The output is a tensor containing the determinants
// for all input submatrices `[..., :, :]`.

*/








func matrixDeterminant( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "MatrixDeterminant",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns a batched diagonal tensor with a given batched diagonal values.

Given a `diagonal`, this operation returns a tensor with the `diagonal` and
// everything else padded with zeros. The diagonal is computed as follows:
// 
// Assume `diagonal` has `k` dimensions `[I, J, K, ..., N]`, then the output is a
// tensor of rank `k+1` with dimensions [I, J, K, ..., N, N]` where:
// 
// `output[i, j, k, ..., m, n] = 1{m=n}  *  diagonal[i, j, k, ..., n]`.
// 
// For example:
// 
// ```
// # 'diagonal' is [[1, 2, 3, 4], [5, 6, 7, 8]]
// 
// and diagonal.shape = (2, 4)
// 
// tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0]
//                                      [0, 2, 0, 0]
//                                      [0, 0, 3, 0]
//                                      [0, 0, 0, 4]],
//                                     [[5, 0, 0, 0]
//                                      [0, 6, 0, 0]
//                                      [0, 0, 7, 0]
//                                      [0, 0, 0, 8]]]
// 
// which has shape (2, 4, 4)
// ```

*/








func matrixDiag( scope:Scope,diagonal: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "MatrixDiag",
        Name: "Type",
        Input: [ diagonal],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the batched diagonal part of a batched tensor.

This operation returns a tensor with the `diagonal` part
// of the batched `input`. The `diagonal` part is computed as follows:
// 
// Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
// tensor of rank `k - 1` with dimensions `[I, J, K, ..., min(M, N)]` where:
// 
// `diagonal[i, j, k, ..., n] = input[i, j, k, ..., n, n]`.
// 
// The input must be at least a matrix.
// 
// For example:
// 
// ```
// # 'input' is [[[1, 0, 0, 0]
//                [0, 2, 0, 0]
//                [0, 0, 3, 0]
//                [0, 0, 0, 4]],
//               [[5, 0, 0, 0]
//                [0, 6, 0, 0]
//                [0, 0, 7, 0]
//                [0, 0, 0, 8]]]
// 
// and input.shape = (2, 4, 4)
// 
// tf.matrix_diag_part(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]
// 
// which has shape (2, 4)
// ```

*/








func matrixDiagPart( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "MatrixDiagPart",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the inverse of one or more square invertible matrices or their

adjoints (conjugate transposes).
// 
// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
// form square matrices. The output is a tensor of the same shape as the input
// containing the inverse for all input submatrices `[..., :, :]`.
// 
// The op uses LU decomposition with partial pivoting to compute the inverses.
// 
// If a matrix is not invertible there is no guarantee what the op does. It
// may detect the condition and raise an exception or it may simply return a
// garbage result.

*/








func matrixInverse( scope:Scope,input: Output, adjoint :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["adjoint"] = adjoint

    let opspec = OpSpec(
        OpType: "MatrixInverse",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns a batched matrix tensor with new batched diagonal values.

Given `input` and `diagonal`, this operation returns a tensor with the
// same shape and values as `input`, except for the main diagonal of the
// innermost matrices.  These will be overwritten by the values in `diagonal`.
// 
// The output is computed as follows:
// 
// Assume `input` has `k+1` dimensions `[I, J, K, ..., M, N]` and `diagonal` has
// `k` dimensions `[I, J, K, ..., min(M, N)]`.  Then the output is a
// tensor of rank `k+1` with dimensions `[I, J, K, ..., M, N]` where:
// 
//    *  `output[i, j, k, ..., m, n] = diagonal[i, j, k, ..., n]` for `m == n`.
//    *  `output[i, j, k, ..., m, n] = input[i, j, k, ..., m, n]` for `m != n`.

*/








func matrixSetDiag( scope:Scope,input: Output, diagonal: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "MatrixSetDiag",
        Name: "Type",
        Input: [ input, diagonal],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Solves systems of linear equations.

`Matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
// form square matrices. `Rhs` is a tensor of shape `[..., M, K]`. The `output` is
// a tensor shape `[..., M, K]`.  If `adjoint` is `False` then each output matrix
// satisfies `matrix[..., :, :]  *  output[..., :, :] = rhs[..., :, :]`.
// If `adjoint` is `True` then each output matrix satisfies
// `adjoint(matrix[..., :, :])  *  output[..., :, :] = rhs[..., :, :]`.

*/








func matrixSolve( scope:Scope,matrix: Output, rhs: Output, adjoint :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["adjoint"] = adjoint

    let opspec = OpSpec(
        OpType: "MatrixSolve",
        Name: "Type",
        Input: [ matrix, rhs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Solves one or more linear least-squares problems.

`matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
// form matrices of size `[M, N]`. Rhs is a tensor of shape `[..., M, K]`.
// The output is a tensor shape `[..., N, K]` where each output matrix solves
// each of the equations matrix[..., :, :]  *  output[..., :, :] = rhs[..., :, :]
// in the least squares sense.
// 
// matrix and right-hand sides in the batch:
// 
// `matrix`=\\(A \in \Re// ^{m \times n}\\),
// `rhs`=\\(B  \in \Re// ^{m \times k}\\),
// `output`=\\(X  \in \Re// ^{n \times k}\\),
// `l2_regularizer`=\\(\lambda\\).
// 
// If `fast` is `True`, then the solution is computed by solving the normal
// equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
// \\(X = (A// ^T A + \lambda I)// ^{-1} A// ^T B\\), which solves the least-squares
// problem \\(X = \mathrm{argmin}_{Z \in \Re// ^{n \times k} } ||A Z - B||_F// ^2 +
// \lambda ||Z||_F// ^2\\). If \\(m \lt n\\) then `output` is computed as
// \\(X = A// ^T (A A// ^T + \lambda I)// ^{-1} B\\), which (for \\(\lambda = 0\\)) is the
// minimum-norm solution to the under-determined linear system, i.e.
// \\(X = \mathrm{argmin}_{Z \in \Re// ^{n \times k} } ||Z||_F// ^2 \\), subject to
// \\(A Z = B\\). Notice that the fast path is only numerically stable when
// \\(A\\) is numerically full rank and has a condition number
// \\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach} } }\\) or\\(\lambda\\) is
// sufficiently large.
// 
// If `fast` is `False` an algorithm based on the numerically robust complete
// orthogonal decomposition is used. This computes the minimum-norm
// least-squares solution, even when \\(A\\) is rank deficient. This path is
// typically 6-7 times slower than the fast path. If `fast` is `False` then
// `l2_regularizer` is ignored.

*/








func matrixSolveLs( scope:Scope,matrix: Output, rhs: Output, l2Regularizer: Output, fast :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["fast"] = fast

    let opspec = OpSpec(
        OpType: "MatrixSolveLs",
        Name: "Type",
        Input: [ matrix, rhs, l2Regularizer],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Solves systems of linear equations with upper or lower triangular matrices by

backsubstitution.
// 
// `matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
// square matrices. If `lower` is `True` then the strictly upper triangular part
// of each inner-most matrix is assumed to be zero and not accessed.
// If `lower` is False then the strictly lower triangular part of each inner-most
// matrix is assumed to be zero and not accessed.
// `rhs` is a tensor of shape `[..., M, K]`.
// 
// The output is a tensor of shape `[..., M, K]`. If `adjoint` is
// `True` then the innermost matrices in output` satisfy matrix equations
// `matrix[..., :, :]  *  output[..., :, :] = rhs[..., :, :]`.
// If `adjoint` is `False` then the strictly then the  innermost matrices in
// `output` satisfy matrix equations
// `adjoint(matrix[..., i, k])  *  output[..., k, j] = rhs[..., i, j]`.

*/








func matrixTriangularSolve( scope:Scope,matrix: Output, rhs: Output, lower :Bool  , adjoint :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["lower"] = lower
    attrs["adjoint"] = adjoint

    let opspec = OpSpec(
        OpType: "MatrixTriangularSolve",
        Name: "Type",
        Input: [ matrix, rhs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the maximum of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.

*/








func max( scope:Scope,input: Output, reductionIndices: Output, keepDims :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["keep_dims"] = keepDims

    let opspec = OpSpec(
        OpType: "Max",
        Name: "Type",
        Input: [ input, reductionIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Performs max pooling on the input.


*/








func maxPool( scope:Scope,input: Output, ksize :[Int64]  , strides :[Int64]  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["ksize"] = ksize
    attrs["strides"] = strides
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "MaxPool",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Performs 3D max pooling on the input.


*/








func maxPool3D( scope:Scope,input: Output, ksize :[Int64]  , strides :[Int64]  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["ksize"] = ksize
    attrs["strides"] = strides
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "MaxPool3D",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes gradients of max pooling function.


*/








func maxPool3DGrad( scope:Scope,origInput: Output, origOutput: Output, grad: Output, ksize :[Int64]  , strides :[Int64]  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["ksize"] = ksize
    attrs["strides"] = strides
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "MaxPool3DGrad",
        Name: "Type",
        Input: [ origInput, origOutput, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes second-order gradients of the maxpooling function.


*/








func maxPool3DGradGrad( scope:Scope,origInput: Output, origOutput: Output, grad: Output, ksize :[Int64]  , strides :[Int64]  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["ksize"] = ksize
    attrs["strides"] = strides
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "MaxPool3DGradGrad",
        Name: "Type",
        Input: [ origInput, origOutput, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes gradients of the maxpooling function.


*/








func maxPoolGrad( scope:Scope,origInput: Output, origOutput: Output, grad: Output, ksize :[Int64]  , strides :[Int64]  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["ksize"] = ksize
    attrs["strides"] = strides
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "MaxPoolGrad",
        Name: "Type",
        Input: [ origInput, origOutput, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes second-order gradients of the maxpooling function.


*/








func maxPoolGradGrad( scope:Scope,origInput: Output, origOutput: Output, grad: Output, ksize :[Int64]  , strides :[Int64]  , padding :String  , dataFormat :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["ksize"] = ksize
    attrs["strides"] = strides
    attrs["padding"] = padding
    attrs["data_format"] = dataFormat

    let opspec = OpSpec(
        OpType: "MaxPoolGradGrad",
        Name: "Type",
        Input: [ origInput, origOutput, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes second-order gradients of the maxpooling function.


*/








func maxPoolGradGradWithArgmax( scope:Scope,input: Output, grad: Output, argmax: Output, ksize :[Int64]  , strides :[Int64]  , padding :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["ksize"] = ksize
    attrs["strides"] = strides
    attrs["padding"] = padding

    let opspec = OpSpec(
        OpType: "MaxPoolGradGradWithArgmax",
        Name: "Type",
        Input: [ input, grad, argmax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes gradients of the maxpooling function.


*/








func maxPoolGradWithArgmax( scope:Scope,input: Output, grad: Output, argmax: Output, ksize :[Int64]  , strides :[Int64]  , padding :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["ksize"] = ksize
    attrs["strides"] = strides
    attrs["padding"] = padding

    let opspec = OpSpec(
        OpType: "MaxPoolGradWithArgmax",
        Name: "Type",
        Input: [ input, grad, argmax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Performs max pooling on the input and outputs both max values and indices.

The indices in `argmax` are flattened, so that a maximum value at position
// `[b, y, x, c]` becomes flattened index
// `((b  *  height + y)  *  width + x)  *  channels + c`.

*/








func maxPoolWithArgmax( scope:Scope,input: Output, ksize :[Int64]  , strides :[Int64]  , padding :String  )  ->(output: Output?, argmax: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["ksize"] = ksize
    attrs["strides"] = strides
    attrs["padding"] = padding

    let opspec = OpSpec(
        OpType: "MaxPoolWithArgmax",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Returns the max of x and y (i.e. x > y ? x : y) element-wise.

 * NOTE * : `Maximum` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func maximum( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Maximum",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the mean of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.

*/








func mean( scope:Scope,input: Output, reductionIndices: Output, keepDims :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["keep_dims"] = keepDims

    let opspec = OpSpec(
        OpType: "Mean",
        Name: "Type",
        Input: [ input, reductionIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Forwards the value of an available tensor from `inputs` to `output`.

`Merge` waits for at least one of the tensors in `inputs` to become available.
// It is usually combined with `Switch` to implement branching.
// 
// `Merge` forwards the first tensor to become available to `output`, and sets
// `value_index` to its index in `inputs`.

*/








func merge( scope:Scope,inputs: Output, n :UInt8  )  ->(output: Output?, valueIndex: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["N"] = n

    let opspec = OpSpec(
        OpType: "Merge",
        Name: "Type",
        Input: [ inputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Merges summaries.

This op creates a
// [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
// protocol buffer that contains the union of all the values in the input
// summaries.
// 
// When the Op is run, it reports an `InvalidArgument` error if multiple values
// in the summaries to merge use the same tag.

*/








func mergeSummary( scope:Scope,inputs: Output, n :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["N"] = n

    let opspec = OpSpec(
        OpType: "MergeSummary",
        Name: "Type",
        Input: [ inputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
V2 format specific: merges the metadata files of sharded checkpoints.  The

result is one logical checkpoint, with one physical metadata file and renamed
// data files.
// 
// Intended for "grouping" multiple checkpoints in a sharded checkpoint setup.
// 
// If delete_old_dirs is true, attempts to delete recursively the dirname of each
// path in the input checkpoint_prefixes.  This is useful when those paths are non
// user-facing temporary locations.

*/








func mergeV2Checkpoints( scope:Scope,checkpointPrefixes: Output, destinationPrefix: Output, deleteOldDirs :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["delete_old_dirs"] = deleteOldDirs

    let opspec = OpSpec(
        OpType: "MergeV2Checkpoints",
        Name: "Type",
        Input: [ checkpointPrefixes, destinationPrefix],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Transforms a spectrogram into a form that's useful for speech recognition.

Mel Frequency Cepstral Coefficients are a way of representing audio data that's
// been effective as an input feature for machine learning. They are created by
// taking the spectrum of a spectrogram (a 'cepstrum'), and discarding some of the
// higher frequencies that are less significant to the human ear. They have a long
// history in the speech recognition world, and https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
// is a good resource to learn more.

*/








func mfcc( scope:Scope,spectrogram: Output, sampleRate: Output, upperFrequencyLimit :Float  , lowerFrequencyLimit :Float  , filterbankChannelCount :UInt8  , dctCoefficientCount :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["upper_frequency_limit"] = upperFrequencyLimit
    attrs["lower_frequency_limit"] = lowerFrequencyLimit
    attrs["filterbank_channel_count"] = filterbankChannelCount
    attrs["dct_coefficient_count"] = dctCoefficientCount

    let opspec = OpSpec(
        OpType: "Mfcc",
        Name: "Type",
        Input: [ spectrogram, sampleRate],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the minimum of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.

*/








func min( scope:Scope,input: Output, reductionIndices: Output, keepDims :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["keep_dims"] = keepDims

    let opspec = OpSpec(
        OpType: "Min",
        Name: "Type",
        Input: [ input, reductionIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the min of x and y (i.e. x < y ? x : y) element-wise.

 * NOTE * : `Minimum` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func minimum( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Minimum",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Pads a tensor with mirrored values.

This operation pads a `input` with mirrored values according to the `paddings`
// you specify. `paddings` is an integer tensor with shape `[n, 2]`, where n is
// the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
// how many values to add before the contents of `input` in that dimension, and
// `paddings[D, 1]` indicates how many values to add after the contents of `input`
// in that dimension. Both `paddings[D, 0]` and `paddings[D, 1]` must be no greater
// than `input.dim_size(D)` (or `input.dim_size(D) - 1`) if `copy_border` is true
// (if false, respectively).
// 
// The padded size of each dimension D of the output is:
// 
// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
// 
// For example:
// 
// ```
// # 't' is [[1, 2, 3], [4, 5, 6]].
// # 'paddings' is [[1, 1]], [2, 2]].
// # 'mode' is SYMMETRIC.
// # rank of 't' is 2.
// pad(t, paddings) ==> [[2, 1, 1, 2, 3, 3, 2]
//                       [2, 1, 1, 2, 3, 3, 2]
//                       [5, 4, 4, 5, 6, 6, 5]
//                       [5, 4, 4, 5, 6, 6, 5]]
// ```

*/








func mirrorPad( scope:Scope,input: Output, paddings: Output, mode :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["mode"] = mode

    let opspec = OpSpec(
        OpType: "MirrorPad",
        Name: "Type",
        Input: [ input, paddings],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Gradient op for `MirrorPad` op. This op folds a mirror-padded tensor.

This operation folds the padded areas of `input` by `MirrorPad` according to the
// `paddings` you specify. `paddings` must be the same as `paddings` argument
// given to the corresponding `MirrorPad` op.
// 
// The folded size of each dimension D of the output is:
// 
// `input.dim_size(D) - paddings(D, 0) - paddings(D, 1)`
// 
// For example:
// 
// ```
// # 't' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
// # 'paddings' is [[0, 1]], [0, 1]].
// # 'mode' is SYMMETRIC.
// # rank of 't' is 2.
// pad(t, paddings) ==> [[ 1,  5]
//                       [11, 28]]
// ```

*/








func mirrorPadGrad( scope:Scope,input: Output, paddings: Output, mode :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["mode"] = mode

    let opspec = OpSpec(
        OpType: "MirrorPadGrad",
        Name: "Type",
        Input: [ input, paddings],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns element-wise remainder of division. This emulates C semantics in that

the result here is consistent with a truncating divide. E.g. `truncate(x / y)  * 
// y + truncate_mod(x, y) = x`.
// 
//  * NOTE * : `Mod` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func mod( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Mod",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns x * y element-wise.

 * NOTE * : `Mul` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func mul( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Mul",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Draws samples from a multinomial distribution.


*/








func multinomial( scope:Scope,logits: Output, numSamples: Output, seed :UInt8  , seed2 :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "Multinomial",
        Name: "Type",
        Input: [ logits, numSamples],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates an empty hash table that uses tensors as the backing store.

It uses "open addressing" with quadratic reprobing to resolve
// collisions.
// 
// This op creates a mutable hash table, specifying the type of its keys and
// values. Each value must be a scalar. Data can be inserted into the table using
// the insert operations. It does not support the initialization operation.

*/








func mutableDenseHashTable( scope:Scope,emptyKey: Output, container :String  , sharedName :String  , useNodeNameSharing :Bool  , valueShape :Shape  , initialNumBuckets :UInt8  , maxLoadFactor :Float  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName
    attrs["use_node_name_sharing"] = useNodeNameSharing
    attrs["value_shape"] = valueShape
    attrs["initial_num_buckets"] = initialNumBuckets
    attrs["max_load_factor"] = maxLoadFactor

    let opspec = OpSpec(
        OpType: "MutableDenseHashTable",
        Name: "Type",
        Input: [ emptyKey],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates an empty hash table that uses tensors as the backing store.

It uses "open addressing" with quadratic reprobing to resolve
// collisions.
// 
// This op creates a mutable hash table, specifying the type of its keys and
// values. Each value must be a scalar. Data can be inserted into the table using
// the insert operations. It does not support the initialization operation.

*/








func mutableDenseHashTableV2( scope:Scope,emptyKey: Output, container :String  , sharedName :String  , useNodeNameSharing :Bool  , valueShape :Shape  , initialNumBuckets :UInt8  , maxLoadFactor :Float  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName
    attrs["use_node_name_sharing"] = useNodeNameSharing
    attrs["value_shape"] = valueShape
    attrs["initial_num_buckets"] = initialNumBuckets
    attrs["max_load_factor"] = maxLoadFactor

    let opspec = OpSpec(
        OpType: "MutableDenseHashTableV2",
        Name: "Type",
        Input: [ emptyKey],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
// values. Each value must be a scalar. Data can be inserted into the table using
// the insert operations. It does not support the initialization operation.

*/








func mutableHashTable( scope:Scope, container :String  , sharedName :String  , useNodeNameSharing :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName
    attrs["use_node_name_sharing"] = useNodeNameSharing

    let opspec = OpSpec(
        OpType: "MutableHashTable",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
// values. Each value must be a vector. Data can be inserted into the table using
// the insert operations. It does not support the initialization operation.

*/








func mutableHashTableOfTensors( scope:Scope, container :String  , sharedName :String  , useNodeNameSharing :Bool  , valueShape :Shape  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName
    attrs["use_node_name_sharing"] = useNodeNameSharing
    attrs["value_shape"] = valueShape

    let opspec = OpSpec(
        OpType: "MutableHashTableOfTensors",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
// values. Each value must be a vector. Data can be inserted into the table using
// the insert operations. It does not support the initialization operation.

*/








func mutableHashTableOfTensorsV2( scope:Scope, container :String  , sharedName :String  , useNodeNameSharing :Bool  , valueShape :Shape  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName
    attrs["use_node_name_sharing"] = useNodeNameSharing
    attrs["value_shape"] = valueShape

    let opspec = OpSpec(
        OpType: "MutableHashTableOfTensorsV2",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
// values. Each value must be a scalar. Data can be inserted into the table using
// the insert operations. It does not support the initialization operation.

*/








func mutableHashTableV2( scope:Scope, container :String  , sharedName :String  , useNodeNameSharing :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName
    attrs["use_node_name_sharing"] = useNodeNameSharing

    let opspec = OpSpec(
        OpType: "MutableHashTableV2",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes numerical negative value element-wise.

I.e., \\(y = -x\\).

*/








func neg( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Neg",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Training via negative sampling.


*/








func negTrain( scope:Scope,wIn: Output, wOut: Output, examples: Output, labels: Output, lr: Output, vocabCount :[Int64]  , numNegativeSamples :UInt8  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["vocab_count"] = vocabCount
    attrs["num_negative_samples"] = numNegativeSamples

    let opspec = OpSpec(
        OpType: "NegTrain",
        Name: "Type",
        Input: [ wIn, wOut, examples, labels, lr],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Makes its input available to the next iteration.


*/








func nextIteration( scope:Scope,data: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "NextIteration",
        Name: "Type",
        Input: [ data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Does nothing. Only useful as a placeholder for control edges.


*/








func noOp( scope:Scope )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "NoOp",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Greedily selects a subset of bounding boxes in descending order of score,

pruning away boxes that have high intersection-over-union (IOU) overlap
// with previously selected boxes.  Bounding boxes are supplied as
// [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
// diagonal pair of box corners and the coordinates can be provided as normalized
// (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
// is agnostic to where the origin is in the coordinate system.  Note that this
// algorithm is invariant to orthogonal transformations and translations
// of the coordinate system; thus translating or reflections of the coordinate
// system result in the same boxes being selected by the algorithm.
// The output of this operation is a set of integers indexing into the input
// collection of bounding boxes representing the selected boxes.  The bounding
// box coordinates corresponding to the selected indices can then be obtained
// using the `tf.gather operation`.  For example:
//   selected_indices = tf.image.non_max_suppression(
//       boxes, scores, max_output_size, iou_threshold)
//   selected_boxes = tf.gather(boxes, selected_indices)

*/








func nonMaxSuppression( scope:Scope,boxes: Output, scores: Output, maxOutputSize: Output, iouThreshold :Float  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["iou_threshold"] = iouThreshold

    let opspec = OpSpec(
        OpType: "NonMaxSuppression",
        Name: "Type",
        Input: [ boxes, scores, maxOutputSize],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Greedily selects a subset of bounding boxes in descending order of score,

pruning away boxes that have high intersection-over-union (IOU) overlap
// with previously selected boxes.  Bounding boxes are supplied as
// [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
// diagonal pair of box corners and the coordinates can be provided as normalized
// (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
// is agnostic to where the origin is in the coordinate system.  Note that this
// algorithm is invariant to orthogonal transformations and translations
// of the coordinate system; thus translating or reflections of the coordinate
// system result in the same boxes being selected by the algorithm.
// 
// The output of this operation is a set of integers indexing into the input
// collection of bounding boxes representing the selected boxes.  The bounding
// box coordinates corresponding to the selected indices can then be obtained
// using the `tf.gather operation`.  For example:
// 
//   selected_indices = tf.image.non_max_suppression_v2(
//       boxes, scores, max_output_size, iou_threshold)
//   selected_boxes = tf.gather(boxes, selected_indices)

*/








func nonMaxSuppressionV2( scope:Scope,boxes: Output, scores: Output, maxOutputSize: Output, iouThreshold: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "NonMaxSuppressionV2",
        Name: "Type",
        Input: [ boxes, scores, maxOutputSize, iouThreshold],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the truth value of (x != y) element-wise.

 * NOTE * : `NotEqual` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func notEqual( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "NotEqual",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns a one-hot tensor.

The locations represented by indices in `indices` take value `on_value`,
// while all other locations take value `off_value`.
// 
// If the input `indices` is rank `N`, the output will have rank `N+1`,
// The new axis is created at dimension `axis` (default: the new axis is
// appended at the end).
// 
// If `indices` is a scalar the output shape will be a vector of length `depth`.
// 
// If `indices` is a vector of length `features`, the output shape will be:
// ```
//   features x depth if axis == -1
//   depth x features if axis == 0
// ```
// 
// If `indices` is a matrix (batch) with shape `[batch, features]`,
// the output shape will be:
// ```
//   batch x features x depth if axis == -1
//   batch x depth x features if axis == 1
//   depth x batch x features if axis == 0
// ```
// 
// 
// Examples
// =========
// 
// Suppose that
// 
// ```
//   indices = [0, 2, -1, 1]
//   depth = 3
//   on_value = 5.0
//   off_value = 0.0
//   axis = -1
// ```
// 
// Then output is `[4 x 3]`:
// 
//     ```output =
//       [5.0 0.0 0.0]  // one_hot(0)
//       [0.0 0.0 5.0]  // one_hot(2)
//       [0.0 0.0 0.0]  // one_hot(-1)
//       [0.0 5.0 0.0]  // one_hot(1)
//     ```
// 
// Suppose that
// 
// ```
//   indices = [0, 2, -1, 1]
//   depth = 3
//   on_value = 0.0
//   off_value = 3.0
//   axis = 0
// ```
// 
// Then output is `[3 x 4]`:
// 
//     ```output =
//       [0.0 3.0 3.0 3.0]
//       [3.0 3.0 3.0 0.0]
//       [3.0 3.0 3.0 3.0]
//       [3.0 0.0 3.0 3.0]
//     //  // ^                one_hot(0)
//     //      // ^            one_hot(2)
//     //          // ^        one_hot(-1)
//     //              // ^    one_hot(1)
//     ```
// Suppose that
// 
// ```
//   indices = [[0, 2], [1, -1]]
//   depth = 3
//   on_value = 1.0
//   off_value = 0.0
//   axis = -1
// ```
// 
// Then output is `[2 x 2 x 3]`:
// 
//     ```output =
//       [
//         [1.0, 0.0, 0.0]  // one_hot(0)
//         [0.0, 0.0, 1.0]  // one_hot(2)
//       ][
//         [0.0, 1.0, 0.0]  // one_hot(1)
//         [0.0, 0.0, 0.0]  // one_hot(-1)
//       ]```

*/








func oneHot( scope:Scope,indices: Output, depth: Output, onValue: Output, offValue: Output, axis :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["axis"] = axis

    let opspec = OpSpec(
        OpType: "OneHot",
        Name: "Type",
        Input: [ indices, depth, onValue, offValue],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Makes a "one-shot" iterator that can be iterated only once.

A one-shot iterator bundles the logic for defining the dataset and
// the state of the iterator in a single op, which allows simple input
// pipelines to be defined without an additional initialization
// ("MakeIterator") step.
// 
// One-shot iterators have the following limitations:
// 
//  *  They do not support parameterization: all logic for creating the underlying
//   dataset must be bundled in the `dataset_factory` function.
//  *  They are not resettable. Once a one-shot iterator reaches the end of its
//   underlying dataset, subsequent "IteratorGetNext" operations on that
//   iterator will always produce an `OutOfRange` error.
// 
// For greater flexibility, use "Iterator" and "MakeIterator" to define
// an iterator using an arbitrary subgraph, which may capture tensors
// (including fed values) as parameters, and which may be reset multiple
// times by rerunning "MakeIterator".

*/








func oneShotIterator( scope:Scope, datasetFactory :TensorflowNameAttrList  , outputShapes :[Shape]  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["dataset_factory"] = datasetFactory
    attrs["output_shapes"] = outputShapes
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "OneShotIterator",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns a tensor of ones with the same shape and type as x.


*/








func onesLike( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "OnesLike",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor.

Packs the `N` tensors in `values` into a tensor with rank one higher than each
// tensor in `values`, by packing them along the `axis` dimension.
// Given a list of tensors of shape `(A, B, C)`;
// 
// if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
// if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
// Etc.
// 
// For example:
// 
// ```
// # 'x' is [1, 4]
// # 'y' is [2, 5]
// # 'z' is [3, 6]
// pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
// pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
// ```
// 
// This is the opposite of `unpack`.

*/








func pack( scope:Scope,values: Output, n :UInt8  , axis :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["N"] = n
    attrs["axis"] = axis

    let opspec = OpSpec(
        OpType: "Pack",
        Name: "Type",
        Input: [ values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Pads a tensor with zeros.

This operation pads a `input` with zeros according to the `paddings` you
// specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
// rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
// how many zeros to add before the contents of `input` in that dimension, and
// `paddings[D, 1]` indicates how many zeros to add after the contents of `input`
// in that dimension.
// 
// The padded size of each dimension D of the output is:
// 
// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
// 
// For example:
// 
// ```
// # 't' is [[1, 1], [2, 2]]
// # 'paddings' is [[1, 1], [2, 2]]
// # rank of 't' is 2
// pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
//                       [0, 0, 1, 1, 0, 0]
//                       [0, 0, 2, 2, 0, 0]
//                       [0, 0, 0, 0, 0, 0]]
// ```

*/








func pad( scope:Scope,input: Output, paddings: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Pad",
        Name: "Type",
        Input: [ input, paddings],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a dataset that batches and pads `batch_size` elements from the input.


*/








func paddedBatchDataset( scope:Scope,inputDataset: Output, batchSize: Output, paddedShapes: Output, paddingValues: Output, outputShapes :[Shape]  , n :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["output_shapes"] = outputShapes
    attrs["N"] = n

    let opspec = OpSpec(
        OpType: "PaddedBatchDataset",
        Name: "Type",
        Input: [ inputDataset, batchSize, paddedShapes, paddingValues],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A queue that produces elements in first-in first-out order.

Variable-size shapes are allowed by setting the corresponding shape dimensions
// to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
// size of any given element in the minibatch.  See below for details.

*/








func paddingFIFOQueue( scope:Scope, shapes :[Shape]  , capacity :UInt8  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shapes"] = shapes
    attrs["capacity"] = capacity
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "PaddingFIFOQueue",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A queue that produces elements in first-in first-out order.

Variable-size shapes are allowed by setting the corresponding shape dimensions
// to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
// size of any given element in the minibatch.  See below for details.

*/








func paddingFIFOQueueV2( scope:Scope, shapes :[Shape]  , capacity :UInt8  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shapes"] = shapes
    attrs["capacity"] = capacity
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "PaddingFIFOQueueV2",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Concatenates a list of `N` tensors along the first dimension.

The input tensors are all required to have size 1 in the first dimension.
// 
// For example:
// 
// ```
// # 'x' is [[1, 4]]
// # 'y' is [[2, 5]]
// # 'z' is [[3, 6]]
// parallel_concat([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
// ```
// 
// The difference between concat and parallel_concat is that concat requires all
// of the inputs be computed before the operation will begin but doesn't require
// that the input shapes be known during graph construction.  Parallel concat
// will copy pieces of the input into the output as they become available, in
// some situations this can provide a performance benefit.

*/








func parallelConcat( scope:Scope,values: Output, n :UInt8  , shape :Shape  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["N"] = n
    attrs["shape"] = shape

    let opspec = OpSpec(
        OpType: "ParallelConcat",
        Name: "Type",
        Input: [ values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a dataset that applies `f` to the outputs of `input_dataset`.

Unlike a "MapDataset", which applies `f` sequentially, this dataset uses
// up to `num_threads` threads to process elements from `input_dataset`
// in parallel.

*/








func parallelMapDataset( scope:Scope,inputDataset: Output, otherArguments: Output, numThreads: Output, outputBufferSize: Output, f :TensorflowNameAttrList  , outputShapes :[Shape]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["f"] = f
    attrs["output_shapes"] = outputShapes

    let opspec = OpSpec(
        OpType: "ParallelMapDataset",
        Name: "Type",
        Input: [ inputDataset, otherArguments, numThreads, outputBufferSize],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Outputs random values from a normal distribution. The parameters may each be a

scalar which applies to the entire output, or a vector of length shape[0] which
// stores the parameters for each batch.

*/








func parameterizedTruncatedNormal( scope:Scope,shape: Output, means: Output, stdevs: Output, minvals: Output, maxvals: Output, seed :UInt8  , seed2 :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "ParameterizedTruncatedNormal",
        Name: "Type",
        Input: [ shape, means, stdevs, minvals, maxvals],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Transforms a vector of brain.Example protos (as strings) into typed tensors.


*/








func parseExample( scope:Scope,serialized: Output, names: Output, sparseKeys: Output, denseKeys: Output, denseDefaults: Output, nsparse :UInt8  , ndense :UInt8  , denseShapes :[Shape]  )  ->(sparseIndices: Output?, sparseValues: Output?, sparseShapes: Output?, denseValues: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["Nsparse"] = nsparse
    attrs["Ndense"] = ndense
    attrs["dense_shapes"] = denseShapes

    let opspec = OpSpec(
        OpType: "ParseExample",
        Name: "Type",
        Input: [ serialized, names, sparseKeys, denseKeys, denseDefaults],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1),op?.output(4 - 1))
}

/*
Transforms a scalar brain.SequenceExample proto (as strings) into typed tensors.


*/








func parseSingleSequenceExample( scope:Scope,serialized: Output, featureListDenseMissingAssumedEmpty: Output, contextSparseKeys: Output, contextDenseKeys: Output, featureListSparseKeys: Output, featureListDenseKeys: Output, contextDenseDefaults: Output, debugName: Output, ncontextSparse :UInt8  , ncontextDense :UInt8  , nfeatureListSparse :UInt8  , nfeatureListDense :UInt8  , contextDenseShapes :[Shape]  , featureListDenseShapes :[Shape]  )  ->(contextSparseIndices: Output?, contextSparseValues: Output?, contextSparseShapes: Output?, contextDenseValues: Output?, featureListSparseIndices: Output?, featureListSparseValues: Output?, featureListSparseShapes: Output?, featureListDenseValues: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil, nil, nil, nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["Ncontext_sparse"] = ncontextSparse
    attrs["Ncontext_dense"] = ncontextDense
    attrs["Nfeature_list_sparse"] = nfeatureListSparse
    attrs["Nfeature_list_dense"] = nfeatureListDense
    attrs["context_dense_shapes"] = contextDenseShapes
    attrs["feature_list_dense_shapes"] = featureListDenseShapes

    let opspec = OpSpec(
        OpType: "ParseSingleSequenceExample",
        Name: "Type",
        Input: [ serialized, featureListDenseMissingAssumedEmpty, contextSparseKeys, contextDenseKeys, featureListSparseKeys, featureListDenseKeys, contextDenseDefaults, debugName],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1),op?.output(4 - 1),op?.output(5 - 1),op?.output(6 - 1),op?.output(7 - 1),op?.output(8 - 1))
}

/*
Transforms a serialized tensorflow.TensorProto proto into a Tensor.


*/








func parseTensor( scope:Scope,serialized: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ParseTensor",
        Name: "Type",
        Input: [ serialized],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A placeholder op for a value that will be fed into the computation.

N.B. This operation will fail with an error if it is executed. It is
// intended as a way to represent a value that will always be fed, and to
// provide attrs that enable the fed value to be checked at runtime.

*/








func placeholder( scope:Scope, shape :Shape  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shape"] = shape

    let opspec = OpSpec(
        OpType: "Placeholder",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A placeholder op for a value that will be fed into the computation.

N.B. This operation will fail with an error if it is executed. It is
// intended as a way to represent a value that will always be fed, and to
// provide attrs that enable the fed value to be checked at runtime.

*/








func placeholderV2( scope:Scope, shape :Shape  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shape"] = shape

    let opspec = OpSpec(
        OpType: "PlaceholderV2",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A placeholder op that passes through `input` when its output is not fed.


*/








func placeholderWithDefault( scope:Scope,input: Output, shape :Shape  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shape"] = shape

    let opspec = OpSpec(
        OpType: "PlaceholderWithDefault",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Compute the polygamma function \\(\psi^{(n)}(x)\\).

The polygamma function is defined as:
// 
// 
// \\(\psi// ^{(n)}(x) = \frac{d// ^n}{dx// ^n} \psi(x)\\)
// 
// where \\(\psi(x)\\) is the digamma function.

*/








func polygamma( scope:Scope,a: Output, x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Polygamma",
        Name: "Type",
        Input: [ a, x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the power of one value to another.

Given a tensor `x` and a tensor `y`, this operation computes \\(x// ^y\\) for
// corresponding elements in `x` and `y`. For example:
// 
// ```
// # tensor 'x' is [[2, 2]], [3, 3]]
// # tensor 'y' is [[8, 16], [2, 3]]
// tf.pow(x, y) ==> [[256, 65536], [9, 27]]
// ```

*/








func pow( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Pow",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
An identity op that triggers an error if a gradient is requested.

When executed in a graph, this op outputs its input tensor as-is.
// 
// When building ops to compute gradients, the TensorFlow gradient system
// will return an error when trying to lookup the gradient of this op,
// because no gradient must ever be registered for this function.  This
// op exists to prevent subtle bugs from silently returning unimplemented
// gradients in some corner cases.

*/








func preventGradient( scope:Scope,input: Output, message :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["message"] = message

    let opspec = OpSpec(
        OpType: "PreventGradient",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Prints a list of tensors.

Passes `input` through to `output` and prints `data` when evaluating.

*/








func print( scope:Scope,input: Output, data: Output, message :String  , firstN :UInt8  , summarize :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["message"] = message
    attrs["first_n"] = firstN
    attrs["summarize"] = summarize

    let opspec = OpSpec(
        OpType: "Print",
        Name: "Type",
        Input: [ input, data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A queue that produces elements sorted by the first component value.

Note that the PriorityQueue requires the first component of any element
// to be a scalar int64, in addition to the other elements declared by
// component_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
// and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
// entry in their input (resp. output) lists.

*/








func priorityQueue( scope:Scope, shapes :[Shape]  , capacity :UInt8  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shapes"] = shapes
    attrs["capacity"] = capacity
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "PriorityQueue",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A queue that produces elements sorted by the first component value.

Note that the PriorityQueue requires the first component of any element
// to be a scalar int64, in addition to the other elements declared by
// component_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
// and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
// entry in their input (resp. output) lists.

*/








func priorityQueueV2( scope:Scope, shapes :[Shape]  , capacity :UInt8  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shapes"] = shapes
    attrs["capacity"] = capacity
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "PriorityQueueV2",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the product of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.

*/








func prod( scope:Scope,input: Output, reductionIndices: Output, keepDims :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["keep_dims"] = keepDims

    let opspec = OpSpec(
        OpType: "Prod",
        Name: "Type",
        Input: [ input, reductionIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Invokes a python function to compute func(input)->output.

This operation is considered stateful. For a stateless version, see
// PyFuncStateless.

*/








func pyFunc( scope:Scope,input: Output, token :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["token"] = token

    let opspec = OpSpec(
        OpType: "PyFunc",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A stateless version of PyFunc.


*/








func pyFuncStateless( scope:Scope,input: Output, token :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["token"] = token

    let opspec = OpSpec(
        OpType: "PyFuncStateless",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the QR decompositions of one or more matrices.

Computes the QR decomposition of each inner matrix in `tensor` such that
// `tensor[..., :, :] = q[..., :, :]  *  r[..., :,:])`
// 
// ```python
// # a is a tensor.
// # q is a tensor of orthonormal matrices.
// # r is a tensor of upper triangular matrices.
// q, r = qr(a)
// q_full, r_full = qr(a, full_matrices=True)
// ```

*/








func qr( scope:Scope,input: Output, fullMatrices :Bool  )  ->(q: Output?, r: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["full_matrices"] = fullMatrices

    let opspec = OpSpec(
        OpType: "Qr",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Use QuantizeAndDequantizeV2 instead.


*/








func quantizeAndDequantize( scope:Scope,input: Output, signedInput :Bool  , numBits :UInt8  , rangeGiven :Bool  , inputMin :Float  , inputMax :Float  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["signed_input"] = signedInput
    attrs["num_bits"] = numBits
    attrs["range_given"] = rangeGiven
    attrs["input_min"] = inputMin
    attrs["input_max"] = inputMax

    let opspec = OpSpec(
        OpType: "QuantizeAndDequantize",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Quantizes then dequantizes a tensor.

This op simulates the precision loss from the quantized forward pass by:
// 1. Quantizing the tensor to fixed point numbers, which should match the target
//    quantization method when it is used in inference.
// 2. Dequantizing it back to floating point numbers for the following ops, most
//    likely matmul.
// 
// There are different ways to quantize. This version does not use the full range
// of the output type, choosing to elide the lowest possible value for symmetry
// (e.g., output range is -127 to 127, not -128 to 127 for signed 8 bit
// quantization), so that 0.0 maps to 0.
// 
// To perform this op, we first find the range of values in our tensor. The range
// we use is always centered on 0, so we find m such that
// 
// 1. m = max(abs(input_min), abs(input_max)) if range_given is true,
// 2. m = max(abs(min_elem(input)), abs(max_elem(input))) otherwise.
// 
// Our input tensor range is then [-m, m].
// 
// Next, we choose our fixed-point quantization buckets, [min_fixed, max_fixed].
// If signed_input is true, this is
// 
//   [min_fixed, max_fixed ] =
//       [-(1 << (num_bits - 1) - 1), (1 << (num_bits - 1)) - 1].
// 
// Otherwise, if signed_input is false, the fixed-point range is
// 
//   [min_fixed, max_fixed] = [0, (1 << num_bits) - 1].
// 
// From this we compute our scaling factor, s:
// 
//   s = (max_fixed - min_fixed) / (2  *  m).
// 
// Now we can quantize and dequantize the elements of our tensor.  An element e
// is transformed into e':
// 
//   e' = (e  *  s).round_to_nearest() / s.
// 
// Note that we have a different number of buckets in the signed vs. unsigned
// cases.  For example, if num_bits == 8, we get 254 buckets in the signed case
// vs. 255 in the unsigned case.
// 
// For example, suppose num_bits = 8 and m = 1.  Then
// 
//   [min_fixed, max_fixed] = [-127, 127], and
//   s = (127 + 127) / 2 = 127.
// 
// Given the vector {-1, -0.5, 0, 0.3}, this is quantized to
// {-127, -63, 0, 38}, and dequantized to {-1, -63.0/127, 0, 38.0/127}.

*/








func quantizeAndDequantizeV2( scope:Scope,input: Output, inputMin: Output, inputMax: Output, signedInput :Bool  , numBits :UInt8  , rangeGiven :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["signed_input"] = signedInput
    attrs["num_bits"] = numBits
    attrs["range_given"] = rangeGiven

    let opspec = OpSpec(
        OpType: "QuantizeAndDequantizeV2",
        Name: "Type",
        Input: [ input, inputMin, inputMax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Convert the quantized 'input' tensor into a lower-precision 'output', using the

actual distribution of the values to maximize the usage of the lower bit depth
// and adjusting the output min and max ranges accordingly.
// 
// [input_min, input_max] are scalar floats that specify the range for the float
// interpretation of the 'input' data. For example, if input_min is -1.0f and
// input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
// value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.
// 
// This operator tries to squeeze as much precision as possible into an output with
// a lower bit depth by calculating the actual min and max values found in the
// data. For example, maybe that quint16 input has no values lower than 16,384 and
// none higher than 49,152. That means only half the range is actually needed, all
// the float interpretations are between -0.5f and 0.5f, so if we want to compress
// the data into a quint8 output, we can use that range rather than the theoretical
// -1.0f to 1.0f that is suggested by the input min and max.
// 
// In practice, this is most useful for taking output from operations like
// QuantizedMatMul that can produce higher bit-depth outputs than their inputs and
// may have large potential output ranges, but in practice have a distribution of
// input values that only uses a small fraction of the possible range. By feeding
// that output into this operator, we can reduce it from 32 bits down to 8 with
// minimal loss of accuracy.

*/








func quantizeDownAndShrinkRange( scope:Scope,input: Output, inputMin: Output, inputMax: Output )  ->(output: Output?, outputMin: Output?, outputMax: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "QuantizeDownAndShrinkRange",
        Name: "Type",
        Input: [ input, inputMin, inputMax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Quantize the 'input' tensor of type float to 'output' tensor of type 'T'.

[min_range, max_range] are scalar floats that specify the range for
// the 'input' data. The 'mode' attribute controls exactly which calculations are
// used to convert the float values to their quantized equivalents.
// 
// In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:
// 
// ```
// out[i] = (in[i] - min_range)  *  range(T) / (max_range - min_range)
// if T == qint8, out[i] -= (range(T) + 1) / 2.0
// ```
// here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`
// 
//  * MIN_COMBINED Mode Example * 
// 
// Assume the input is type float and has a possible range of [0.0, 6.0] and the
// output type is quint8 ([0, 255]). The min_range and max_range values should be
// specified as 0.0 and 6.0. Quantizing from float to quint8 will multiply each
// value of the input by 255/6 and cast to quint8.
// 
// If the output type was qint8 ([-128, 127]), the operation will additionally
// subtract each value by 128 prior to casting, so that the range of values aligns
// with the range of qint8.
// 
// If the mode is 'MIN_FIRST', then this approach is used:
// 
// ```
// number_of_steps = 1 << (# of bits in T)
// range_adjust = number_of_steps / (number_of_steps - 1)
// range = (range_max - range_min)  *  range_adjust
// range_scale = number_of_steps / range
// quantized = round(input  *  range_scale) - round(range_min  *  range_scale) +
//   numeric_limits<T>::min()
// quantized = max(quantized, numeric_limits<T>::min())
// quantized = min(quantized, numeric_limits<T>::max())
// ```
// 
// The biggest difference between this and MIN_COMBINED is that the minimum range
// is rounded first, before it's subtracted from the rounded value. With
// MIN_COMBINED, a small bias is introduced where repeated iterations of quantizing
// and dequantizing will introduce a larger and larger error.
// 
// One thing to watch out for is that the operator may choose to adjust the
// requested minimum and maximum values slightly during the quantization process,
// so you should always use the output ports as the range for further calculations.
// For example, if the requested minimum and maximum values are close to equal,
// they will be separated by a small epsilon value to prevent ill-formed quantized
// buffers from being created. Otherwise, you can end up with buffers where all the
// quantized values map to the same float value, which causes problems for
// operations that have to perform further calculations on them.

*/








func quantizeV2( scope:Scope,input: Output, minRange: Output, maxRange: Output, mode :String  )  ->(output: Output?, outputMin: Output?, outputMax: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["mode"] = mode

    let opspec = OpSpec(
        OpType: "QuantizeV2",
        Name: "Type",
        Input: [ input, minRange, maxRange],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Returns x + y element-wise, working on quantized buffers.


*/








func quantizedAdd( scope:Scope,x: Output, y: Output, minX: Output, maxX: Output, minY: Output, maxY: Output )  ->(z: Output?, minZ: Output?, maxZ: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "QuantizedAdd",
        Name: "Type",
        Input: [ x, y, minX, maxX, minY, maxY],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Produces the average pool of the input tensor for quantized types.


*/








func quantizedAvgPool( scope:Scope,input: Output, minInput: Output, maxInput: Output, ksize :[Int64]  , strides :[Int64]  , padding :String  )  ->(output: Output?, minOutput: Output?, maxOutput: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["ksize"] = ksize
    attrs["strides"] = strides
    attrs["padding"] = padding

    let opspec = OpSpec(
        OpType: "QuantizedAvgPool",
        Name: "Type",
        Input: [ input, minInput, maxInput],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Quantized Batch normalization.

This op is deprecated and will be removed in the future. Prefer
// `tf.nn.batch_normalization`.

*/








func quantizedBatchNormWithGlobalNormalization( scope:Scope,t: Output, tMin: Output, tMax: Output, m: Output, mMin: Output, mMax: Output, v: Output, vMin: Output, vMax: Output, beta: Output, betaMin: Output, betaMax: Output, gamma: Output, gammaMin: Output, gammaMax: Output, varianceEpsilon :Float  , scaleAfterNormalization :Bool  )  ->(result: Output?, resultMin: Output?, resultMax: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["variance_epsilon"] = varianceEpsilon
    attrs["scale_after_normalization"] = scaleAfterNormalization

    let opspec = OpSpec(
        OpType: "QuantizedBatchNormWithGlobalNormalization",
        Name: "Type",
        Input: [ t, tMin, tMax, m, mMin, mMax, v, vMin, vMax, beta, betaMin, betaMax, gamma, gammaMin, gammaMax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Adds Tensor 'bias' to Tensor 'input' for Quantized types.

Broadcasts the values of bias on dimensions 0..N-2 of 'input'.

*/








func quantizedBiasAdd( scope:Scope,input: Output, bias: Output, minInput: Output, maxInput: Output, minBias: Output, maxBias: Output )  ->(output: Output?, minOut: Output?, maxOut: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "QuantizedBiasAdd",
        Name: "Type",
        Input: [ input, bias, minInput, maxInput, minBias, maxBias],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Concatenates quantized tensors along one dimension.


*/








func quantizedConcat( scope:Scope,concatDim: Output, values: Output, inputMins: Output, inputMaxes: Output, n :UInt8  )  ->(output: Output?, outputMin: Output?, outputMax: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["N"] = n

    let opspec = OpSpec(
        OpType: "QuantizedConcat",
        Name: "Type",
        Input: [ concatDim, values, inputMins, inputMaxes],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Computes a 2D convolution given quantized 4D input and filter tensors.

The inputs are quantized tensors where the lowest value represents the real
// number of the associated minimum, and the highest represents the maximum.
// This means that you can only interpret the quantized output in the same way, by
// taking the returned minimum and maximum values into account.

*/








func quantizedConv2D( scope:Scope,input: Output, filter: Output, minInput: Output, maxInput: Output, minFilter: Output, maxFilter: Output, strides :[Int64]  , padding :String  )  ->(output: Output?, minOutput: Output?, maxOutput: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["strides"] = strides
    attrs["padding"] = padding

    let opspec = OpSpec(
        OpType: "QuantizedConv2D",
        Name: "Type",
        Input: [ input, filter, minInput, maxInput, minFilter, maxFilter],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Quantized Instance normalization.


*/








func quantizedInstanceNorm( scope:Scope,x: Output, xMin: Output, xMax: Output, outputRangeGiven :Bool  , givenYMin :Float  , givenYMax :Float  , varianceEpsilon :Float  , minSeparation :Float  )  ->(y: Output?, yMin: Output?, yMax: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["output_range_given"] = outputRangeGiven
    attrs["given_y_min"] = givenYMin
    attrs["given_y_max"] = givenYMax
    attrs["variance_epsilon"] = varianceEpsilon
    attrs["min_separation"] = minSeparation

    let opspec = OpSpec(
        OpType: "QuantizedInstanceNorm",
        Name: "Type",
        Input: [ x, xMin, xMax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Perform a quantized matrix multiplication of  `a` by the matrix `b`.

The inputs must be two-dimensional matrices and the inner dimension of
// `a` (after being transposed if `transpose_a` is non-zero) must match the
// outer dimension of `b` (after being transposed if `transposed_b` is
// non-zero).

*/








func quantizedMatMul( scope:Scope,a: Output, b: Output, minA: Output, maxA: Output, minB: Output, maxB: Output, transposeA :Bool  , transposeB :Bool  )  ->(out: Output?, minOut: Output?, maxOut: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["transpose_a"] = transposeA
    attrs["transpose_b"] = transposeB

    let opspec = OpSpec(
        OpType: "QuantizedMatMul",
        Name: "Type",
        Input: [ a, b, minA, maxA, minB, maxB],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Produces the max pool of the input tensor for quantized types.


*/








func quantizedMaxPool( scope:Scope,input: Output, minInput: Output, maxInput: Output, ksize :[Int64]  , strides :[Int64]  , padding :String  )  ->(output: Output?, minOutput: Output?, maxOutput: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["ksize"] = ksize
    attrs["strides"] = strides
    attrs["padding"] = padding

    let opspec = OpSpec(
        OpType: "QuantizedMaxPool",
        Name: "Type",
        Input: [ input, minInput, maxInput],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Returns x * y element-wise, working on quantized buffers.


*/








func quantizedMul( scope:Scope,x: Output, y: Output, minX: Output, maxX: Output, minY: Output, maxY: Output )  ->(z: Output?, minZ: Output?, maxZ: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "QuantizedMul",
        Name: "Type",
        Input: [ x, y, minX, maxX, minY, maxY],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Computes Quantized Rectified Linear: `max(features, 0)`


*/








func quantizedRelu( scope:Scope,features: Output, minFeatures: Output, maxFeatures: Output )  ->(activations: Output?, minActivations: Output?, maxActivations: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "QuantizedRelu",
        Name: "Type",
        Input: [ features, minFeatures, maxFeatures],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Computes Quantized Rectified Linear 6: `min(max(features, 0), 6)`


*/








func quantizedRelu6( scope:Scope,features: Output, minFeatures: Output, maxFeatures: Output )  ->(activations: Output?, minActivations: Output?, maxActivations: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "QuantizedRelu6",
        Name: "Type",
        Input: [ features, minFeatures, maxFeatures],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`


*/








func quantizedReluX( scope:Scope,features: Output, maxValue: Output, minFeatures: Output, maxFeatures: Output )  ->(activations: Output?, minActivations: Output?, maxActivations: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "QuantizedReluX",
        Name: "Type",
        Input: [ features, maxValue, minFeatures, maxFeatures],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Reshapes a quantized tensor as per the Reshape op.

```

*/








func quantizedReshape( scope:Scope,tensor: Output, shape: Output, inputMin: Output, inputMax: Output )  ->(output: Output?, outputMin: Output?, outputMax: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "QuantizedReshape",
        Name: "Type",
        Input: [ tensor, shape, inputMin, inputMax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Closes the given queue.

This operation signals that no more elements will be enqueued in the
// given queue. Subsequent Enqueue(Many) operations will fail.
// Subsequent Dequeue(Many) operations will continue to succeed if
// sufficient elements remain in the queue. Subsequent Dequeue(Many)
// operations that would block will fail immediately.

*/








func queueClose( scope:Scope,handle: Output, cancelPendingEnqueues :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["cancel_pending_enqueues"] = cancelPendingEnqueues

    let opspec = OpSpec(
        OpType: "QueueClose",
        Name: "Type",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Closes the given queue.

This operation signals that no more elements will be enqueued in the
// given queue. Subsequent Enqueue(Many) operations will fail.
// Subsequent Dequeue(Many) operations will continue to succeed if
// sufficient elements remain in the queue. Subsequent Dequeue(Many)
// operations that would block will fail immediately.

*/








func queueCloseV2( scope:Scope,handle: Output, cancelPendingEnqueues :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["cancel_pending_enqueues"] = cancelPendingEnqueues

    let opspec = OpSpec(
        OpType: "QueueCloseV2",
        Name: "Type",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Dequeues a tuple of one or more tensors from the given queue.

This operation has k outputs, where k is the number of components
// in the tuples stored in the given queue, and output i is the ith
// component of the dequeued tuple.
// 
// N.B. If the queue is empty, this operation will block until an element
// has been dequeued (or 'timeout_ms' elapses, if specified).

*/








func queueDequeue( scope:Scope,handle: Output, timeoutMs :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["timeout_ms"] = timeoutMs

    let opspec = OpSpec(
        OpType: "QueueDequeue",
        Name: "Type",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Dequeues `n` tuples of one or more tensors from the given queue.

If the queue is closed and there are fewer than `n` elements, then an
// OutOfRange error is returned.
// 
// This operation concatenates queue-element component tensors along the
// 0th dimension to make a single component tensor.  All of the components
// in the dequeued tuple will have size `n` in the 0th dimension.
// 
// This operation has `k` outputs, where `k` is the number of components in
// the tuples stored in the given queue, and output `i` is the ith
// component of the dequeued tuple.
// 
// N.B. If the queue is empty, this operation will block until `n` elements
// have been dequeued (or 'timeout_ms' elapses, if specified).

*/








func queueDequeueMany( scope:Scope,handle: Output, n: Output, timeoutMs :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["timeout_ms"] = timeoutMs

    let opspec = OpSpec(
        OpType: "QueueDequeueMany",
        Name: "Type",
        Input: [ handle, n],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Dequeues `n` tuples of one or more tensors from the given queue.

If the queue is closed and there are fewer than `n` elements, then an
// OutOfRange error is returned.
// 
// This operation concatenates queue-element component tensors along the
// 0th dimension to make a single component tensor.  All of the components
// in the dequeued tuple will have size `n` in the 0th dimension.
// 
// This operation has `k` outputs, where `k` is the number of components in
// the tuples stored in the given queue, and output `i` is the ith
// component of the dequeued tuple.
// 
// N.B. If the queue is empty, this operation will block until `n` elements
// have been dequeued (or 'timeout_ms' elapses, if specified).

*/








func queueDequeueManyV2( scope:Scope,handle: Output, n: Output, timeoutMs :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["timeout_ms"] = timeoutMs

    let opspec = OpSpec(
        OpType: "QueueDequeueManyV2",
        Name: "Type",
        Input: [ handle, n],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Dequeues `n` tuples of one or more tensors from the given queue.

This operation is not supported by all queues.  If a queue does not support
// DequeueUpTo, then an Unimplemented error is returned.
// 
// If the queue is closed and there are more than 0 but less than `n`
// elements remaining, then instead of returning an OutOfRange error like
// QueueDequeueMany, less than `n` elements are returned immediately.  If
// the queue is closed and there are 0 elements left in the queue, then
// an OutOfRange error is returned just like in QueueDequeueMany.
// Otherwise the behavior is identical to QueueDequeueMany:
// 
// This operation concatenates queue-element component tensors along the
// 0th dimension to make a single component tensor.  All of the components
// in the dequeued tuple will have size `n` in the 0th dimension.
// 
// This operation has k outputs, where `k` is the number of components in
// the tuples stored in the given queue, and output `i` is the ith
// component of the dequeued tuple.

*/








func queueDequeueUpTo( scope:Scope,handle: Output, n: Output, timeoutMs :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["timeout_ms"] = timeoutMs

    let opspec = OpSpec(
        OpType: "QueueDequeueUpTo",
        Name: "Type",
        Input: [ handle, n],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Dequeues `n` tuples of one or more tensors from the given queue.

This operation is not supported by all queues.  If a queue does not support
// DequeueUpTo, then an Unimplemented error is returned.
// 
// If the queue is closed and there are more than 0 but less than `n`
// elements remaining, then instead of returning an OutOfRange error like
// QueueDequeueMany, less than `n` elements are returned immediately.  If
// the queue is closed and there are 0 elements left in the queue, then
// an OutOfRange error is returned just like in QueueDequeueMany.
// Otherwise the behavior is identical to QueueDequeueMany:
// 
// This operation concatenates queue-element component tensors along the
// 0th dimension to make a single component tensor.  All of the components
// in the dequeued tuple will have size n in the 0th dimension.
// 
// This operation has `k` outputs, where `k` is the number of components in
// the tuples stored in the given queue, and output `i` is the ith
// component of the dequeued tuple.

*/








func queueDequeueUpToV2( scope:Scope,handle: Output, n: Output, timeoutMs :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["timeout_ms"] = timeoutMs

    let opspec = OpSpec(
        OpType: "QueueDequeueUpToV2",
        Name: "Type",
        Input: [ handle, n],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Dequeues a tuple of one or more tensors from the given queue.

This operation has k outputs, where k is the number of components
// in the tuples stored in the given queue, and output i is the ith
// component of the dequeued tuple.
// 
// N.B. If the queue is empty, this operation will block until an element
// has been dequeued (or 'timeout_ms' elapses, if specified).

*/








func queueDequeueV2( scope:Scope,handle: Output, timeoutMs :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["timeout_ms"] = timeoutMs

    let opspec = OpSpec(
        OpType: "QueueDequeueV2",
        Name: "Type",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Enqueues a tuple of one or more tensors in the given queue.

The components input has k elements, which correspond to the components of
// tuples stored in the given queue.
// 
// N.B. If the queue is full, this operation will block until the given
// element has been enqueued (or 'timeout_ms' elapses, if specified).

*/








func queueEnqueue( scope:Scope,handle: Output, components: Output, timeoutMs :UInt8  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["timeout_ms"] = timeoutMs

    let opspec = OpSpec(
        OpType: "QueueEnqueue",
        Name: "Type",
        Input: [ handle, components],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Enqueues zero or more tuples of one or more tensors in the given queue.

This operation slices each component tensor along the 0th dimension to
// make multiple queue elements. All of the tuple components must have the
// same size in the 0th dimension.
// 
// The components input has k elements, which correspond to the components of
// tuples stored in the given queue.
// 
// N.B. If the queue is full, this operation will block until the given
// elements have been enqueued (or 'timeout_ms' elapses, if specified).

*/








func queueEnqueueMany( scope:Scope,handle: Output, components: Output, timeoutMs :UInt8  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["timeout_ms"] = timeoutMs

    let opspec = OpSpec(
        OpType: "QueueEnqueueMany",
        Name: "Type",
        Input: [ handle, components],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Enqueues zero or more tuples of one or more tensors in the given queue.

This operation slices each component tensor along the 0th dimension to
// make multiple queue elements. All of the tuple components must have the
// same size in the 0th dimension.
// 
// The components input has k elements, which correspond to the components of
// tuples stored in the given queue.
// 
// N.B. If the queue is full, this operation will block until the given
// elements have been enqueued (or 'timeout_ms' elapses, if specified).

*/








func queueEnqueueManyV2( scope:Scope,handle: Output, components: Output, timeoutMs :UInt8  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["timeout_ms"] = timeoutMs

    let opspec = OpSpec(
        OpType: "QueueEnqueueManyV2",
        Name: "Type",
        Input: [ handle, components],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Enqueues a tuple of one or more tensors in the given queue.

The components input has k elements, which correspond to the components of
// tuples stored in the given queue.
// 
// N.B. If the queue is full, this operation will block until the given
// element has been enqueued (or 'timeout_ms' elapses, if specified).

*/








func queueEnqueueV2( scope:Scope,handle: Output, components: Output, timeoutMs :UInt8  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["timeout_ms"] = timeoutMs

    let opspec = OpSpec(
        OpType: "QueueEnqueueV2",
        Name: "Type",
        Input: [ handle, components],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Computes the number of elements in the given queue.


*/








func queueSize( scope:Scope,handle: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "QueueSize",
        Name: "Type",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the number of elements in the given queue.


*/








func queueSizeV2( scope:Scope,handle: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "QueueSizeV2",
        Name: "Type",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Real-valued fast Fourier transform.

Computes the 1-dimensional discrete Fourier transform of a real-valued signal
// over the inner-most dimension of `input`.
// 
// Since the DFT of a real signal is Hermitian-symmetric, `RFFT` only returns the
// `fft_length / 2 + 1` unique components of the FFT: the zero-frequency term,
// followed by the `fft_length / 2` positive-frequency terms.

*/








func rfft( scope:Scope,input: Output, fftLength: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "RFFT",
        Name: "Type",
        Input: [ input, fftLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
2D real-valued fast Fourier transform.

Computes the 2-dimensional discrete Fourier transform of a real-valued signal
// over the inner-most 2 dimensions of `input`.
// 
// Since the DFT of a real signal is Hermitian-symmetric, `RFFT2D` only returns the
// `fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
// of `output`: the zero-frequency term, followed by the `fft_length / 2`
// positive-frequency terms.

*/








func rfft2D( scope:Scope,input: Output, fftLength: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "RFFT2D",
        Name: "Type",
        Input: [ input, fftLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
3D real-valued fast Fourier transform.

Computes the 3-dimensional discrete Fourier transform of a real-valued signal
// over the inner-most 3 dimensions of `input`.
// 
// Since the DFT of a real signal is Hermitian-symmetric, `RFFT3D` only returns the
// `fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
// of `output`: the zero-frequency term, followed by the `fft_length / 2`
// positive-frequency terms.

*/








func rfft3D( scope:Scope,input: Output, fftLength: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "RFFT3D",
        Name: "Type",
        Input: [ input, fftLength],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Converts one or more images from RGB to HSV.

Outputs a tensor of the same shape as the `images` tensor, containing the HSV
// value of the pixels. The output is only well defined if the value in `images`
// are in `[0,1]`.
// 
// `output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
// `output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
// corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.

*/








func rgbToHSV( scope:Scope,images: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "RGBToHSV",
        Name: "Type",
        Input: [ images],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Randomly crop `image`.

`size` is a 1-D int64 tensor with 2 elements representing the crop height and
// width.  The values must be non negative.
// 
// This Op picks a random location in `image` and crops a `height` by `width`
// rectangle from that location.  The random location is picked so the cropped
// area will fit inside the original image.

*/








func randomCrop( scope:Scope,image: Output, size: Output, seed :UInt8  , seed2 :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "RandomCrop",
        Name: "Type",
        Input: [ image, size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Outputs random values from the Gamma distribution(s) described by alpha.

This op uses the algorithm by Marsaglia et al. to acquire samples via
// transformation-rejection from pairs of uniform and normal random variables.
// See http://dl.acm.org/citation.cfm?id=358414

*/








func randomGamma( scope:Scope,shape: Output, alpha: Output, seed :UInt8  , seed2 :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "RandomGamma",
        Name: "Type",
        Input: [ shape, alpha],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Outputs random values from the Poisson distribution(s) described by rate.

This op uses two algorithms, depending on rate. If rate >= 10, then
// the algorithm by Hormann is used to acquire samples via
// transformation-rejection.
// See http://www.sciencedirect.com/science/article/pii/0167668793909974.
// 
// Otherwise, Knuth's algorithm is used to acquire samples via multiplying uniform
// random variables.
// See Donald E. Knuth (1969). Seminumerical Algorithms. The Art of Computer
// Programming, Volume 2. Addison Wesley

*/








func randomPoisson( scope:Scope,shape: Output, rate: Output, seed :UInt8  , seed2 :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "RandomPoisson",
        Name: "Type",
        Input: [ shape, rate],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Randomly shuffles a tensor along its first dimension.

  The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
//   to one and only one `output[i]`. For example, a mapping that might occur for a
//   3x2 tensor is:
// 
// ```
// [[1, 2],       [[5, 6],
//  [3, 4],  ==>   [1, 2],
//  [5, 6]]        [3, 4]]
// ```

*/








func randomShuffle( scope:Scope,value: Output, seed :UInt8  , seed2 :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "RandomShuffle",
        Name: "Type",
        Input: [ value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A queue that randomizes the order of elements.


*/








func randomShuffleQueue( scope:Scope, shapes :[Shape]  , capacity :UInt8  , minAfterDequeue :UInt8  , seed :UInt8  , seed2 :UInt8  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shapes"] = shapes
    attrs["capacity"] = capacity
    attrs["min_after_dequeue"] = minAfterDequeue
    attrs["seed"] = seed
    attrs["seed2"] = seed2
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "RandomShuffleQueue",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A queue that randomizes the order of elements.


*/








func randomShuffleQueueV2( scope:Scope, shapes :[Shape]  , capacity :UInt8  , minAfterDequeue :UInt8  , seed :UInt8  , seed2 :UInt8  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shapes"] = shapes
    attrs["capacity"] = capacity
    attrs["min_after_dequeue"] = minAfterDequeue
    attrs["seed"] = seed
    attrs["seed2"] = seed2
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "RandomShuffleQueueV2",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Outputs random values from a normal distribution.

The generated values will have mean 0 and standard deviation 1.

*/








func randomStandardNormal( scope:Scope,shape: Output, seed :UInt8  , seed2 :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "RandomStandardNormal",
        Name: "Type",
        Input: [ shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Outputs random values from a uniform distribution.

The generated values follow a uniform distribution in the range `[0, 1)`. The
// lower bound 0 is included in the range, while the upper bound 1 is excluded.

*/








func randomUniform( scope:Scope,shape: Output, seed :UInt8  , seed2 :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "RandomUniform",
        Name: "Type",
        Input: [ shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Outputs random integers from a uniform distribution.

The generated values are uniform integers in the range `[minval, maxval)`.
// The lower bound `minval` is included in the range, while the upper bound
// `maxval` is excluded.
// 
// The random integers are slightly biased unless `maxval - minval` is an exact
// power of two.  The bias is small for values of `maxval - minval` significantly
// smaller than the range of the output (either `2// ^32` or `2// ^64`).

*/








func randomUniformInt( scope:Scope,shape: Output, minval: Output, maxval: Output, seed :UInt8  , seed2 :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "RandomUniformInt",
        Name: "Type",
        Input: [ shape, minval, maxval],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a sequence of numbers.

This operation creates a sequence of numbers that begins at `start` and
// extends by increments of `delta` up to but not including `limit`.
// 
// For example:
// 
// ```
// # 'start' is 3
// # 'limit' is 18
// # 'delta' is 3
// tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
// ```

*/








func range( scope:Scope,start: Output, limit: Output, delta: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Range",
        Name: "Type",
        Input: [ start, limit, delta],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a dataset with a range of values. Corresponds to python's xrange.


*/








func rangeDataset( scope:Scope,start: Output, stop: Output, step: Output, outputShapes :[Shape]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["output_shapes"] = outputShapes

    let opspec = OpSpec(
        OpType: "RangeDataset",
        Name: "Type",
        Input: [ start, stop, step],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the rank of a tensor.

This operation returns an integer representing the rank of `input`.
// 
// For example:
// 
// ```
// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
// # shape of tensor 't' is [2, 2, 3]
// rank(t) ==> 3
// ```
// 
//  *  * Note *  * : The rank of a tensor is not the same as the rank of a matrix. The rank
// of a tensor is the number of indices required to uniquely select each element
// of the tensor. Rank is also known as "order", "degree", or "ndims."

*/








func rank( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Rank",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Reads and outputs the entire contents of the input filename.


*/








func readFile( scope:Scope,filename: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReadFile",
        Name: "Type",
        Input: [ filename],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the number of records this Reader has produced.

This is the same as the number of ReaderRead executions that have
// succeeded.

*/








func readerNumRecordsProduced( scope:Scope,readerHandle: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReaderNumRecordsProduced",
        Name: "Type",
        Input: [ readerHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the number of records this Reader has produced.

This is the same as the number of ReaderRead executions that have
// succeeded.

*/








func readerNumRecordsProducedV2( scope:Scope,readerHandle: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReaderNumRecordsProducedV2",
        Name: "Type",
        Input: [ readerHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the number of work units this Reader has finished processing.


*/








func readerNumWorkUnitsCompleted( scope:Scope,readerHandle: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReaderNumWorkUnitsCompleted",
        Name: "Type",
        Input: [ readerHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the number of work units this Reader has finished processing.


*/








func readerNumWorkUnitsCompletedV2( scope:Scope,readerHandle: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReaderNumWorkUnitsCompletedV2",
        Name: "Type",
        Input: [ readerHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the next record (key, value pair) produced by a Reader.

Will dequeue from the input queue if necessary (e.g. when the
// Reader needs to start reading from a new file since it has finished
// with the previous file).

*/








func readerRead( scope:Scope,readerHandle: Output, queueHandle: Output )  ->(key: Output?, value: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReaderRead",
        Name: "Type",
        Input: [ readerHandle, queueHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Returns up to `num_records` (key, value) pairs produced by a Reader.

Will dequeue from the input queue if necessary (e.g. when the
// Reader needs to start reading from a new file since it has finished
// with the previous file).
// It may return less than `num_records` even before the last batch.

*/








func readerReadUpTo( scope:Scope,readerHandle: Output, queueHandle: Output, numRecords: Output )  ->(keys: Output?, values: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReaderReadUpTo",
        Name: "Type",
        Input: [ readerHandle, queueHandle, numRecords],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Returns up to `num_records` (key, value) pairs produced by a Reader.

Will dequeue from the input queue if necessary (e.g. when the
// Reader needs to start reading from a new file since it has finished
// with the previous file).
// It may return less than `num_records` even before the last batch.

*/








func readerReadUpToV2( scope:Scope,readerHandle: Output, queueHandle: Output, numRecords: Output )  ->(keys: Output?, values: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReaderReadUpToV2",
        Name: "Type",
        Input: [ readerHandle, queueHandle, numRecords],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Returns the next record (key, value pair) produced by a Reader.

Will dequeue from the input queue if necessary (e.g. when the
// Reader needs to start reading from a new file since it has finished
// with the previous file).

*/








func readerReadV2( scope:Scope,readerHandle: Output, queueHandle: Output )  ->(key: Output?, value: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReaderReadV2",
        Name: "Type",
        Input: [ readerHandle, queueHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Restore a Reader to its initial clean state.


*/








func readerReset( scope:Scope,readerHandle: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReaderReset",
        Name: "Type",
        Input: [ readerHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Restore a Reader to its initial clean state.


*/








func readerResetV2( scope:Scope,readerHandle: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReaderResetV2",
        Name: "Type",
        Input: [ readerHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Restore a reader to a previously saved state.

Not all Readers support being restored, so this can produce an
// Unimplemented error.

*/








func readerRestoreState( scope:Scope,readerHandle: Output, state: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReaderRestoreState",
        Name: "Type",
        Input: [ readerHandle, state],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Restore a reader to a previously saved state.

Not all Readers support being restored, so this can produce an
// Unimplemented error.

*/








func readerRestoreStateV2( scope:Scope,readerHandle: Output, state: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReaderRestoreStateV2",
        Name: "Type",
        Input: [ readerHandle, state],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Produce a string tensor that encodes the state of a Reader.

Not all Readers support being serialized, so this can produce an
// Unimplemented error.

*/








func readerSerializeState( scope:Scope,readerHandle: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReaderSerializeState",
        Name: "Type",
        Input: [ readerHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Produce a string tensor that encodes the state of a Reader.

Not all Readers support being serialized, so this can produce an
// Unimplemented error.

*/








func readerSerializeStateV2( scope:Scope,readerHandle: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReaderSerializeStateV2",
        Name: "Type",
        Input: [ readerHandle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the real part of a complex number.

Given a tensor `input` of complex numbers, this operation returns a tensor of
// type `float` that is the real part of each element in `input`. All elements in
// `input` must be complex numbers of the form \\(a + bj\\), where  * a *  is the real
//  part returned by this operation and  * b *  is the imaginary part.
// 
// For example:
// 
// ```
// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
// tf.real(input) ==> [-2.25, 3.25]
// ```

*/








func real( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Real",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns x / y element-wise for real types.

If `x` and `y` are reals, this will return the floating-point division.
// 
//  * NOTE * : `Div` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func realDiv( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "RealDiv",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the reciprocal of x element-wise.

I.e., \\(y = 1 / x\\).

*/








func reciprocal( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Reciprocal",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradient for the inverse of `x` wrt its input.

Specifically, `grad = -dy  *  y * y`, where `y = 1/x`, and `dy`
// is the corresponding input gradient.

*/








func reciprocalGrad( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReciprocalGrad",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Emits randomized records.


*/








func recordInput( scope:Scope, filePattern :String  , fileRandomSeed :UInt8  , fileShuffleShiftRatio :Float  , fileBufferSize :UInt8  , fileParallelism :UInt8  , batchSize :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["file_pattern"] = filePattern
    attrs["file_random_seed"] = fileRandomSeed
    attrs["file_shuffle_shift_ratio"] = fileShuffleShiftRatio
    attrs["file_buffer_size"] = fileBufferSize
    attrs["file_parallelism"] = fileParallelism
    attrs["batch_size"] = batchSize

    let opspec = OpSpec(
        OpType: "RecordInput",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Joins a string Tensor across the given dimensions.

Computes the string join across dimensions in the given string Tensor of shape
// `[d_0, d_1, ..., d_n-1]`.  Returns a new Tensor created by joining the input
// strings with the given separator (default: empty string).  Negative indices are
// counted backwards from the end, with `-1` being equivalent to `n - 1`.
// 
// For example:
// 
// ```python
// # tensor `a` is [["a", "b"], ["c", "d"]]
// tf.reduce_join(a, 0) ==> ["ac", "bd"]
// tf.reduce_join(a, 1) ==> ["ab", "cd"]
// tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
// tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
// tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
// tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
// tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
// tf.reduce_join(a, [0, 1]) ==> ["acbd"]
// tf.reduce_join(a, [1, 0]) ==> ["abcd"]
// tf.reduce_join(a, []) ==> ["abcd"]
// ```

*/








func reduceJoin( scope:Scope,inputs: Output, reductionIndices: Output, keepDims :Bool  , separator :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["keep_dims"] = keepDims
    attrs["separator"] = separator

    let opspec = OpSpec(
        OpType: "ReduceJoin",
        Name: "Type",
        Input: [ inputs, reductionIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates or finds a child frame, and makes `data` available to the child frame.

The unique `frame_name` is used by the `Executor` to identify frames. If
// `is_constant` is true, `output` is a constant in the child frame; otherwise
// it may be changed in the child frame. At most `parallel_iterations` iterations
// are run in parallel in the child frame.

*/








func refEnter( scope:Scope,data: Output, frameName :String  , isConstant :Bool  , parallelIterations :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["frame_name"] = frameName
    attrs["is_constant"] = isConstant
    attrs["parallel_iterations"] = parallelIterations

    let opspec = OpSpec(
        OpType: "RefEnter",
        Name: "Type",
        Input: [ data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Exits the current frame to its parent frame.

Exit makes its input `data` available to the parent frame.

*/








func refExit( scope:Scope,data: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "RefExit",
        Name: "Type",
        Input: [ data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Return the same ref tensor as the input ref tensor.


*/








func refIdentity( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "RefIdentity",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Forwards the value of an available tensor from `inputs` to `output`.

`Merge` waits for at least one of the tensors in `inputs` to become available.
// It is usually combined with `Switch` to implement branching.
// 
// `Merge` forwards the first tensor for become available to `output`, and sets
// `value_index` to its index in `inputs`.

*/








func refMerge( scope:Scope,inputs: Output, n :UInt8  )  ->(output: Output?, valueIndex: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["N"] = n

    let opspec = OpSpec(
        OpType: "RefMerge",
        Name: "Type",
        Input: [ inputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Makes its input available to the next iteration.


*/








func refNextIteration( scope:Scope,data: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "RefNextIteration",
        Name: "Type",
        Input: [ data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Forwards the `index`th element of `inputs` to `output`.


*/








func refSelect( scope:Scope,index: Output, inputs: Output, n :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["N"] = n

    let opspec = OpSpec(
        OpType: "RefSelect",
        Name: "Type",
        Input: [ index, inputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Forwards the ref tensor `data` to the output port determined by `pred`.

If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
// the data goes to `output_false`.
// 
// See also `Switch` and `Merge`.

*/








func refSwitch( scope:Scope,data: Output, pred: Output )  ->(outputFalse: Output?, outputTrue: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "RefSwitch",
        Name: "Type",
        Input: [ data, pred],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Computes rectified linear: `max(features, 0)`.


*/








func relu( scope:Scope,features: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Relu",
        Name: "Type",
        Input: [ features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes rectified linear 6: `min(max(features, 0), 6)`.


*/








func relu6( scope:Scope,features: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Relu6",
        Name: "Type",
        Input: [ features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes rectified linear 6 gradients for a Relu6 operation.


*/








func relu6Grad( scope:Scope,gradients: Output, features: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Relu6Grad",
        Name: "Type",
        Input: [ gradients, features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes rectified linear gradients for a Relu operation.


*/








func reluGrad( scope:Scope,gradients: Output, features: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReluGrad",
        Name: "Type",
        Input: [ gradients, features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a dataset that emits the outputs of `input_dataset` `count` times.


*/








func repeatDataset( scope:Scope,inputDataset: Output, count: Output, outputShapes :[Shape]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["output_shapes"] = outputShapes

    let opspec = OpSpec(
        OpType: "RepeatDataset",
        Name: "Type",
        Input: [ inputDataset, count],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Given a quantized tensor described by (input, input_min, input_max), outputs a

range that covers the actual values present in that tensor.  This op is
// typically used to produce the requested_output_min and requested_output_max for
// Requantize.

*/








func requantizationRange( scope:Scope,input: Output, inputMin: Output, inputMax: Output )  ->(outputMin: Output?, outputMax: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "RequantizationRange",
        Name: "Type",
        Input: [ input, inputMin, inputMax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Convert the quantized 'input' tensor into a lower-precision 'output', using the

output range specified with 'requested_output_min' and 'requested_output_max'.
// 
// [input_min, input_max] are scalar floats that specify the range for the float
// interpretation of the 'input' data. For example, if input_min is -1.0f and
// input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
// value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.

*/








func requantize( scope:Scope,input: Output, inputMin: Output, inputMax: Output, requestedOutputMin: Output, requestedOutputMax: Output )  ->(output: Output?, outputMin: Output?, outputMax: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Requantize",
        Name: "Type",
        Input: [ input, inputMin, inputMax, requestedOutputMin, requestedOutputMax],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Reshapes a tensor.

Given `tensor`, this operation returns a tensor that has the same values
// as `tensor` with shape `shape`.
// 
// If one component of `shape` is the special value -1, the size of that dimension
// is computed so that the total size remains constant.  In particular, a `shape`
// of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.
// 
// If `shape` is 1-D or higher, then the operation returns a tensor with shape
// `shape` filled with the values of `tensor`. In this case, the number of elements
// implied by `shape` must be the same as the number of elements in `tensor`.
// 
// For example:
// 
// ```
// # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
// # tensor 't' has shape [9]
// reshape(t, [3, 3]) ==> [[1, 2, 3],
//                         [4, 5, 6],
//                         [7, 8, 9]]
// 
// # tensor 't' is [[[1, 1], [2, 2]],
// #                [[3, 3], [4, 4]]]
// # tensor 't' has shape [2, 2, 2]
// reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
//                         [3, 3, 4, 4]]
// 
// # tensor 't' is [[[1, 1, 1],
// #                 [2, 2, 2]],
// #                [[3, 3, 3],
// #                 [4, 4, 4]],
// #                [[5, 5, 5],
// #                 [6, 6, 6]]]
// # tensor 't' has shape [3, 2, 3]
// # pass '[-1]' to flatten 't'
// reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
// 
// # -1 can also be used to infer the shape
// 
// # -1 is inferred to be 9:
// reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
//                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
// # -1 is inferred to be 2:
// reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
//                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
// # -1 is inferred to be 3:
// reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
//                               [2, 2, 2],
//                               [3, 3, 3]],
//                              [[4, 4, 4],
//                               [5, 5, 5],
//                               [6, 6, 6]]]
// 
// # tensor 't' is [7]
// # shape `[]` reshapes to a scalar
// reshape(t, []) ==> 7
// ```

*/








func reshape( scope:Scope,tensor: Output, shape: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Reshape",
        Name: "Type",
        Input: [ tensor, shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Resize `images` to `size` using area interpolation.

Input images can be of different types but output images are always float.

*/








func resizeArea( scope:Scope,images: Output, size: Output, alignCorners :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["align_corners"] = alignCorners

    let opspec = OpSpec(
        OpType: "ResizeArea",
        Name: "Type",
        Input: [ images, size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Resize `images` to `size` using bicubic interpolation.

Input images can be of different types but output images are always float.

*/








func resizeBicubic( scope:Scope,images: Output, size: Output, alignCorners :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["align_corners"] = alignCorners

    let opspec = OpSpec(
        OpType: "ResizeBicubic",
        Name: "Type",
        Input: [ images, size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Resize `images` to `size` using bilinear interpolation.

Input images can be of different types but output images are always float.

*/








func resizeBilinear( scope:Scope,images: Output, size: Output, alignCorners :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["align_corners"] = alignCorners

    let opspec = OpSpec(
        OpType: "ResizeBilinear",
        Name: "Type",
        Input: [ images, size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradient of bilinear interpolation.


*/








func resizeBilinearGrad( scope:Scope,grads: Output, originalImage: Output, alignCorners :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["align_corners"] = alignCorners

    let opspec = OpSpec(
        OpType: "ResizeBilinearGrad",
        Name: "Type",
        Input: [ grads, originalImage],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Resize `images` to `size` using nearest neighbor interpolation.


*/








func resizeNearestNeighbor( scope:Scope,images: Output, size: Output, alignCorners :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["align_corners"] = alignCorners

    let opspec = OpSpec(
        OpType: "ResizeNearestNeighbor",
        Name: "Type",
        Input: [ images, size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradient of nearest neighbor interpolation.


*/








func resizeNearestNeighborGrad( scope:Scope,grads: Output, size: Output, alignCorners :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["align_corners"] = alignCorners

    let opspec = OpSpec(
        OpType: "ResizeNearestNeighborGrad",
        Name: "Type",
        Input: [ grads, size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update '*var' according to the adadelta scheme.

accum = rho()  *  accum + (1 - rho())  *  grad.square();
// update = (update_accum + epsilon).sqrt()  *  (accum + epsilon()).rsqrt()  *  grad;
// update_accum = rho()  *  update_accum + (1 - rho())  *  update.square();
// var -= update;

*/








func resourceApplyAdadelta( scope:Scope,`var`: Output, accum: Output, accumUpdate: Output, lr: Output, rho: Output, epsilon: Output, grad: Output, useLocking :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ResourceApplyAdadelta",
        Name: "Type",
        Input: [ `var`, accum, accumUpdate, lr, rho, epsilon, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Update '*var' according to the adagrad scheme.

accum += grad  *  grad
// var -= lr  *  grad  *  (1 / sqrt(accum))

*/








func resourceApplyAdagrad( scope:Scope,`var`: Output, accum: Output, lr: Output, grad: Output, useLocking :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ResourceApplyAdagrad",
        Name: "Type",
        Input: [ `var`, accum, lr, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Update '*var' according to the proximal adagrad scheme.


*/








func resourceApplyAdagradDA( scope:Scope,`var`: Output, gradientAccumulator: Output, gradientSquaredAccumulator: Output, grad: Output, lr: Output, l1: Output, l2: Output, globalStep: Output, useLocking :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ResourceApplyAdagradDA",
        Name: "Type",
        Input: [ `var`, gradientAccumulator, gradientSquaredAccumulator, grad, lr, l1, l2, globalStep],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Update '*var' according to the Adam algorithm.

lr_t <- learning_rate  *  sqrt(1 - beta2// ^t) / (1 - beta1// ^t)
// m_t <- beta1  *  m_{t-1} + (1 - beta1)  *  g_t
// v_t <- beta2  *  v_{t-1} + (1 - beta2)  *  g_t  *  g_t
// variable <- variable - lr_t  *  m_t / (sqrt(v_t) + epsilon)

*/








func resourceApplyAdam( scope:Scope,`var`: Output, m: Output, v: Output, beta1Power: Output, beta2Power: Output, lr: Output, beta1: Output, beta2: Output, epsilon: Output, grad: Output, useLocking :Bool  , useNesterov :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking
    attrs["use_nesterov"] = useNesterov

    let opspec = OpSpec(
        OpType: "ResourceApplyAdam",
        Name: "Type",
        Input: [ `var`, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Update '*var' according to the centered RMSProp algorithm.

The centered RMSProp algorithm uses an estimate of the centered second moment
// (i.e., the variance) for normalization, as opposed to regular RMSProp, which
// uses the (uncentered) second moment. This often helps with training, but is
// slightly more expensive in terms of computation and memory.
// 
// Note that in dense implementation of this algorithm, mg, ms, and mom will
// update even if the grad is zero, but in this sparse implementation, mg, ms,
// and mom will not update in iterations during which the grad is zero.
// 
// mean_square = decay  *  mean_square + (1-decay)  *  gradient  *  *  2
// mean_grad = decay  *  mean_grad + (1-decay)  *  gradient
// 
// Delta = learning_rate  *  gradient / sqrt(mean_square + epsilon - mean_grad  *  *  2)
// 
// mg <- rho  *  mg_{t-1} + (1-rho)  *  grad
// ms <- rho  *  ms_{t-1} + (1-rho)  *  grad  *  grad
// mom <- momentum  *  mom_{t-1} + lr  *  grad / sqrt(ms - mg  *  mg + epsilon)
// var <- var - mom

*/








func resourceApplyCenteredRMSProp( scope:Scope,`var`: Output, mg: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, useLocking :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ResourceApplyCenteredRMSProp",
        Name: "Type",
        Input: [ `var`, mg, ms, mom, lr, rho, momentum, epsilon, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Update '*var' according to the Ftrl-proximal scheme.

accum_new = accum + grad  *  grad
// linear += grad + (accum_new// ^(-lr_power) - accum// ^(-lr_power)) / lr  *  var
// quadratic = 1.0 / (accum_new// ^(lr_power)  *  lr) + 2  *  l2
// var = (sign(linear)  *  l1 - linear) / quadratic if |linear| > l1 else 0.0
// accum = accum_new

*/








func resourceApplyFtrl( scope:Scope,`var`: Output, accum: Output, linear: Output, grad: Output, lr: Output, l1: Output, l2: Output, lrPower: Output, useLocking :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ResourceApplyFtrl",
        Name: "Type",
        Input: [ `var`, accum, linear, grad, lr, l1, l2, lrPower],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Update '*var' by subtracting 'alpha' * 'delta' from it.


*/








func resourceApplyGradientDescent( scope:Scope,`var`: Output, alpha: Output, delta: Output, useLocking :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ResourceApplyGradientDescent",
        Name: "Type",
        Input: [ `var`, alpha, delta],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Update '*var' according to the momentum scheme. Set use_nesterov = True if you

want to use Nesterov momentum.
// 
// accum = accum  *  momentum + grad
// var -= lr  *  accum

*/








func resourceApplyMomentum( scope:Scope,`var`: Output, accum: Output, lr: Output, grad: Output, momentum: Output, useLocking :Bool  , useNesterov :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking
    attrs["use_nesterov"] = useNesterov

    let opspec = OpSpec(
        OpType: "ResourceApplyMomentum",
        Name: "Type",
        Input: [ `var`, accum, lr, grad, momentum],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.

accum += grad  *  grad
// prox_v = var - lr  *  grad  *  (1 / sqrt(accum))
// var = sign(prox_v)/(1+lr * l2)  *  max{|prox_v|-lr * l1,0}

*/








func resourceApplyProximalAdagrad( scope:Scope,`var`: Output, accum: Output, lr: Output, l1: Output, l2: Output, grad: Output, useLocking :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ResourceApplyProximalAdagrad",
        Name: "Type",
        Input: [ `var`, accum, lr, l1, l2, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Update '*var' as FOBOS algorithm with fixed learning rate.

prox_v = var - alpha  *  delta
// var = sign(prox_v)/(1+alpha * l2)  *  max{|prox_v|-alpha * l1,0}

*/








func resourceApplyProximalGradientDescent( scope:Scope,`var`: Output, alpha: Output, l1: Output, l2: Output, delta: Output, useLocking :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ResourceApplyProximalGradientDescent",
        Name: "Type",
        Input: [ `var`, alpha, l1, l2, delta],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Update '*var' according to the RMSProp algorithm.

Note that in dense implementation of this algorithm, ms and mom will
// update even if the grad is zero, but in this sparse implementation, ms
// and mom will not update in iterations during which the grad is zero.
// 
// mean_square = decay  *  mean_square + (1-decay)  *  gradient  *  *  2
// Delta = learning_rate  *  gradient / sqrt(mean_square + epsilon)
// 
// ms <- rho  *  ms_{t-1} + (1-rho)  *  grad  *  grad
// mom <- momentum  *  mom_{t-1} + lr  *  grad / sqrt(ms + epsilon)
// var <- var - mom

*/








func resourceApplyRMSProp( scope:Scope,`var`: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, useLocking :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ResourceApplyRMSProp",
        Name: "Type",
        Input: [ `var`, ms, mom, lr, rho, momentum, epsilon, grad],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
var: Should be from a Variable().


*/








func resourceSparseApplyAdadelta( scope:Scope,`var`: Output, accum: Output, accumUpdate: Output, lr: Output, rho: Output, epsilon: Output, grad: Output, indices: Output, useLocking :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ResourceSparseApplyAdadelta",
        Name: "Type",
        Input: [ `var`, accum, accumUpdate, lr, rho, epsilon, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Update relevant entries in '*var' and '*accum' according to the adagrad scheme.

That is for rows we have grad for, we update var and accum as follows:
// accum += grad  *  grad
// var -= lr  *  grad  *  (1 / sqrt(accum))

*/








func resourceSparseApplyAdagrad( scope:Scope,`var`: Output, accum: Output, lr: Output, grad: Output, indices: Output, useLocking :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ResourceSparseApplyAdagrad",
        Name: "Type",
        Input: [ `var`, accum, lr, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Update entries in '*var' and '*accum' according to the proximal adagrad scheme.


*/








func resourceSparseApplyAdagradDA( scope:Scope,`var`: Output, gradientAccumulator: Output, gradientSquaredAccumulator: Output, grad: Output, indices: Output, lr: Output, l1: Output, l2: Output, globalStep: Output, useLocking :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ResourceSparseApplyAdagradDA",
        Name: "Type",
        Input: [ `var`, gradientAccumulator, gradientSquaredAccumulator, grad, indices, lr, l1, l2, globalStep],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Update '*var' according to the centered RMSProp algorithm.

The centered RMSProp algorithm uses an estimate of the centered second moment
// (i.e., the variance) for normalization, as opposed to regular RMSProp, which
// uses the (uncentered) second moment. This often helps with training, but is
// slightly more expensive in terms of computation and memory.
// 
// Note that in dense implementation of this algorithm, mg, ms, and mom will
// update even if the grad is zero, but in this sparse implementation, mg, ms,
// and mom will not update in iterations during which the grad is zero.
// 
// mean_square = decay  *  mean_square + (1-decay)  *  gradient  *  *  2
// mean_grad = decay  *  mean_grad + (1-decay)  *  gradient
// Delta = learning_rate  *  gradient / sqrt(mean_square + epsilon - mean_grad  *  *  2)
// 
// ms <- rho  *  ms_{t-1} + (1-rho)  *  grad  *  grad
// mom <- momentum  *  mom_{t-1} + lr  *  grad / sqrt(ms + epsilon)
// var <- var - mom

*/








func resourceSparseApplyCenteredRMSProp( scope:Scope,`var`: Output, mg: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, indices: Output, useLocking :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ResourceSparseApplyCenteredRMSProp",
        Name: "Type",
        Input: [ `var`, mg, ms, mom, lr, rho, momentum, epsilon, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Update relevant entries in '*var' according to the Ftrl-proximal scheme.

That is for rows we have grad for, we update var, accum and linear as follows:
// accum_new = accum + grad  *  grad
// linear += grad + (accum_new// ^(-lr_power) - accum// ^(-lr_power)) / lr  *  var
// quadratic = 1.0 / (accum_new// ^(lr_power)  *  lr) + 2  *  l2
// var = (sign(linear)  *  l1 - linear) / quadratic if |linear| > l1 else 0.0
// accum = accum_new

*/








func resourceSparseApplyFtrl( scope:Scope,`var`: Output, accum: Output, linear: Output, grad: Output, indices: Output, lr: Output, l1: Output, l2: Output, lrPower: Output, useLocking :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ResourceSparseApplyFtrl",
        Name: "Type",
        Input: [ `var`, accum, linear, grad, indices, lr, l1, l2, lrPower],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Update relevant entries in '*var' and '*accum' according to the momentum scheme.

Set use_nesterov = True if you want to use Nesterov momentum.
// 
// That is for rows we have grad for, we update var and accum as follows:
// 
// accum = accum  *  momentum + grad
// var -= lr  *  accum

*/








func resourceSparseApplyMomentum( scope:Scope,`var`: Output, accum: Output, lr: Output, grad: Output, indices: Output, momentum: Output, useLocking :Bool  , useNesterov :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking
    attrs["use_nesterov"] = useNesterov

    let opspec = OpSpec(
        OpType: "ResourceSparseApplyMomentum",
        Name: "Type",
        Input: [ `var`, accum, lr, grad, indices, momentum],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Sparse update entries in '*var' and '*accum' according to FOBOS algorithm.

That is for rows we have grad for, we update var and accum as follows:
// accum += grad  *  grad
// prox_v = var
// prox_v -= lr  *  grad  *  (1 / sqrt(accum))
// var = sign(prox_v)/(1+lr * l2)  *  max{|prox_v|-lr * l1,0}

*/








func resourceSparseApplyProximalAdagrad( scope:Scope,`var`: Output, accum: Output, lr: Output, l1: Output, l2: Output, grad: Output, indices: Output, useLocking :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ResourceSparseApplyProximalAdagrad",
        Name: "Type",
        Input: [ `var`, accum, lr, l1, l2, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Sparse update '*var' as FOBOS algorithm with fixed learning rate.

That is for rows we have grad for, we update var as follows:
// prox_v = var - alpha  *  grad
// var = sign(prox_v)/(1+alpha * l2)  *  max{|prox_v|-alpha * l1,0}

*/








func resourceSparseApplyProximalGradientDescent( scope:Scope,`var`: Output, alpha: Output, l1: Output, l2: Output, grad: Output, indices: Output, useLocking :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ResourceSparseApplyProximalGradientDescent",
        Name: "Type",
        Input: [ `var`, alpha, l1, l2, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Update '*var' according to the RMSProp algorithm.

Note that in dense implementation of this algorithm, ms and mom will
// update even if the grad is zero, but in this sparse implementation, ms
// and mom will not update in iterations during which the grad is zero.
// 
// mean_square = decay  *  mean_square + (1-decay)  *  gradient  *  *  2
// Delta = learning_rate  *  gradient / sqrt(mean_square + epsilon)
// 
// ms <- rho  *  ms_{t-1} + (1-rho)  *  grad  *  grad
// mom <- momentum  *  mom_{t-1} + lr  *  grad / sqrt(ms + epsilon)
// var <- var - mom

*/








func resourceSparseApplyRMSProp( scope:Scope,`var`: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, indices: Output, useLocking :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ResourceSparseApplyRMSProp",
        Name: "Type",
        Input: [ `var`, ms, mom, lr, rho, momentum, epsilon, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Assign `value` to the sliced l-value reference of `ref`.

The values of `value` are assigned to the positions in the variable
// `ref` that are selected by the slice parameters. The slice parameters
// `begin, `end`, `strides`, etc. work exactly as in `StridedSlice`.
// 
// NOTE this op currently does not support broadcasting and so `value`'s
// shape must be exactly the shape produced by the slice of `ref`.

*/








func resourceStridedSliceAssign( scope:Scope,ref: Output, begin: Output, end: Output, strides: Output, value: Output, beginMask :UInt8  , endMask :UInt8  , ellipsisMask :UInt8  , newAxisMask :UInt8  , shrinkAxisMask :UInt8  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["begin_mask"] = beginMask
    attrs["end_mask"] = endMask
    attrs["ellipsis_mask"] = ellipsisMask
    attrs["new_axis_mask"] = newAxisMask
    attrs["shrink_axis_mask"] = shrinkAxisMask

    let opspec = OpSpec(
        OpType: "ResourceStridedSliceAssign",
        Name: "Type",
        Input: [ ref, begin, end, strides, value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Restores a tensor from checkpoint files.

Reads a tensor stored in one or several files. If there are several files (for
// instance because a tensor was saved as slices), `file_pattern` may contain
// wildcard symbols (` * ` and `?`) in the filename portion only, not in the
// directory portion.
// 
// If a `file_pattern` matches several files, `preferred_shard` can be used to hint
// in which file the requested tensor is likely to be found. This op will first
// open the file at index `preferred_shard` in the list of matching files and try
// to restore tensors from that file.  Only if some tensors or tensor slices are
// not found in that first file, then the Op opens all the files. Setting
// `preferred_shard` to match the value passed as the `shard` input
// of a matching `Save` Op may speed up Restore.  This attribute only affects
// performance, not correctness.  The default value -1 means files are processed in
// order.
// 
// See also `RestoreSlice`.

*/








func restore( scope:Scope,filePattern: Output, tensorName: Output, preferredShard :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["preferred_shard"] = preferredShard

    let opspec = OpSpec(
        OpType: "Restore",
        Name: "Type",
        Input: [ filePattern, tensorName],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Restores a tensor from checkpoint files.

This is like `Restore` except that restored tensor can be listed as filling
// only a slice of a larger tensor.  `shape_and_slice` specifies the shape of the
// larger tensor and the slice that the restored tensor covers.
// 
// The `shape_and_slice` input has the same format as the
// elements of the `shapes_and_slices` input of the `SaveSlices` op.

*/








func restoreSlice( scope:Scope,filePattern: Output, tensorName: Output, shapeAndSlice: Output, preferredShard :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["preferred_shard"] = preferredShard

    let opspec = OpSpec(
        OpType: "RestoreSlice",
        Name: "Type",
        Input: [ filePattern, tensorName, shapeAndSlice],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Restores tensors from a V2 checkpoint.

For backward compatibility with the V1 format, this Op currently allows
// restoring from a V1 checkpoint as well:
//   - This Op first attempts to find the V2 index file pointed to by "prefix", and
//     if found proceed to read it as a V2 checkpoint;
//   - Otherwise the V1 read path is invoked.
// Relying on this behavior is not recommended, as the ability to fall back to read
// V1 might be deprecated and eventually removed.
// 
// By default, restores the named tensors in full.  If the caller wishes to restore
// specific slices of stored tensors, "shape_and_slices" should be non-empty
// strings and correspondingly well-formed.
// 
// Callers must ensure all the named tensors are indeed stored in the checkpoint.

*/








func restoreV2( scope:Scope,`prefix`: Output, tensorNames: Output, shapeAndSlices: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "RestoreV2",
        Name: "Type",
        Input: [ `prefix`, tensorNames, shapeAndSlices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Reverses specific dimensions of a tensor.

Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
// of `tensor`, this operation reverses each dimension i of `tensor` where
// `dims[i]` is `True`.
// 
// `tensor` can have up to 8 dimensions. The number of dimensions
// of `tensor` must equal the number of elements in `dims`. In other words:
// 
// `rank(tensor) = size(dims)`
// 
// For example:
// 
// ```
// # tensor 't' is [[[[ 0,  1,  2,  3],
// #                  [ 4,  5,  6,  7],
// #                  [ 8,  9, 10, 11]],
// #                 [[12, 13, 14, 15],
// #                  [16, 17, 18, 19],
// #                  [20, 21, 22, 23]]]]
// # tensor 't' shape is [1, 2, 3, 4]
// 
// # 'dims' is [False, False, False, True]
// reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
//                         [ 7,  6,  5,  4],
//                         [ 11, 10, 9, 8]],
//                        [[15, 14, 13, 12],
//                         [19, 18, 17, 16],
//                         [23, 22, 21, 20]]]]
// 
// # 'dims' is [False, True, False, False]
// reverse(t, dims) ==> [[[[12, 13, 14, 15],
//                         [16, 17, 18, 19],
//                         [20, 21, 22, 23]
//                        [[ 0,  1,  2,  3],
//                         [ 4,  5,  6,  7],
//                         [ 8,  9, 10, 11]]]]
// 
// # 'dims' is [False, False, True, False]
// reverse(t, dims) ==> [[[[8, 9, 10, 11],
//                         [4, 5, 6, 7],
//                         [0, 1, 2, 3]]
//                        [[20, 21, 22, 23],
//                         [16, 17, 18, 19],
//                         [12, 13, 14, 15]]]]
// ```

*/








func reverse( scope:Scope,tensor: Output, dims: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Reverse",
        Name: "Type",
        Input: [ tensor, dims],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Reverses variable length slices.

This op first slices `input` along the dimension `batch_dim`, and for each
// slice `i`, reverses the first `seq_lengths[i]` elements along
// the dimension `seq_dim`.
// 
// The elements of `seq_lengths` must obey `seq_lengths[i] <= input.dims[seq_dim]`,
// and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.
// 
// The output slice `i` along dimension `batch_dim` is then given by input
// slice `i`, with the first `seq_lengths[i]` slices along dimension
// `seq_dim` reversed.
// 
// For example:
// 
// ```
// # Given this:
// batch_dim = 0
// seq_dim = 1
// input.dims = (4, 8, ...)
// seq_lengths = [7, 2, 3, 5]
// 
// # then slices of input are reversed on seq_dim, but only up to seq_lengths:
// output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
// output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
// output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
// output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]
// 
// # while entries past seq_lens are copied through:
// output[0, 7:, :, ...] = input[0, 7:, :, ...]
// output[1, 2:, :, ...] = input[1, 2:, :, ...]
// output[2, 3:, :, ...] = input[2, 3:, :, ...]
// output[3, 2:, :, ...] = input[3, 2:, :, ...]
// ```
// 
// In contrast, if:
// 
// ```
// # Given this:
// batch_dim = 2
// seq_dim = 0
// input.dims = (8, ?, 4, ...)
// seq_lengths = [7, 2, 3, 5]
// 
// # then slices of input are reversed on seq_dim, but only up to seq_lengths:
// output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
// output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
// output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
// output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]
// 
// # while entries past seq_lens are copied through:
// output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
// output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
// output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
// output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
// ```

*/








func reverseSequence( scope:Scope,input: Output, seqLengths: Output, seqDim :UInt8  , batchDim :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["seq_dim"] = seqDim
    attrs["batch_dim"] = batchDim

    let opspec = OpSpec(
        OpType: "ReverseSequence",
        Name: "Type",
        Input: [ input, seqLengths],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Reverses specific dimensions of a tensor.

NOTE `tf.reverse` has now changed behavior in preparation for 1.0.
// `tf.reverse_v2` is currently an alias that will be deprecated before TF 1.0.
// 
// Given a `tensor`, and a `int32` tensor `axis` representing the set of
// dimensions of `tensor` to reverse. This operation reverses each dimension
// `i` for which there exists `j` s.t. `axis[j] == i`.
// 
// `tensor` can have up to 8 dimensions. The number of dimensions specified
// in `axis` may be 0 or more entries. If an index is specified more than
// once, a InvalidArgument error is raised.
// 
// For example:
// 
// ```
// # tensor 't' is [[[[ 0,  1,  2,  3],
// #                  [ 4,  5,  6,  7],
// #                  [ 8,  9, 10, 11]],
// #                 [[12, 13, 14, 15],
// #                  [16, 17, 18, 19],
// #                  [20, 21, 22, 23]]]]
// # tensor 't' shape is [1, 2, 3, 4]
// 
// # 'dims' is [3] or 'dims' is -1
// reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
//                         [ 7,  6,  5,  4],
//                         [ 11, 10, 9, 8]],
//                        [[15, 14, 13, 12],
//                         [19, 18, 17, 16],
//                         [23, 22, 21, 20]]]]
// 
// # 'dims' is '[1]' (or 'dims' is '[-3]')
// reverse(t, dims) ==> [[[[12, 13, 14, 15],
//                         [16, 17, 18, 19],
//                         [20, 21, 22, 23]
//                        [[ 0,  1,  2,  3],
//                         [ 4,  5,  6,  7],
//                         [ 8,  9, 10, 11]]]]
// 
// # 'dims' is '[2]' (or 'dims' is '[-2]')
// reverse(t, dims) ==> [[[[8, 9, 10, 11],
//                         [4, 5, 6, 7],
//                         [0, 1, 2, 3]]
//                        [[20, 21, 22, 23],
//                         [16, 17, 18, 19],
//                         [12, 13, 14, 15]]]]
// ```

*/








func reverseV2( scope:Scope,tensor: Output, axis: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ReverseV2",
        Name: "Type",
        Input: [ tensor, axis],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns element-wise integer closest to x.

If the result is midway between two representable values,
// the even representable is chosen.
// For example:
// 
// ```
// rint(-1.5) ==> -2.0
// rint(0.5000001) ==> 1.0
// rint([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
// ```

*/








func rint( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Rint",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Rounds the values of a tensor to the nearest integer, element-wise.

Rounds half to even.  Also known as bankers rounding. If you want to round
// according to the current system rounding mode use std::cint.

*/








func round( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Round",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes reciprocal of square root of x element-wise.

I.e., \\(y = 1 / \sqrt{x}\\).

*/








func rsqrt( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Rsqrt",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradient for the rsqrt of `x` wrt its input.

Specifically, `grad = dy  *  -0.5  *  y// ^3`, where `y = rsqrt(x)`, and `dy`
// is the corresponding input gradient.

*/








func rsqrtGrad( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "RsqrtGrad",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Generate a single randomly distorted bounding box for an image.

Bounding box annotations are often supplied in addition to ground-truth labels
// in image recognition or object localization tasks. A common technique for
// training such a system is to randomly distort an image while preserving
// its content, i.e.  * data augmentation * . This Op outputs a randomly distorted
// localization of an object, i.e. bounding box, given an `image_size`,
// `bounding_boxes` and a series of constraints.
// 
// The output of this Op is a single bounding box that may be used to crop the
// original image. The output is returned as 3 tensors: `begin`, `size` and
// `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
// image. The latter may be supplied to `tf.image.draw_bounding_boxes` to visualize
// what the bounding box looks like.
// 
// Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
// bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
// height of the underlying image.
// 
// For example,
// 
// ```python
//     # Generate a single distorted bounding box.
//     begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
//         tf.shape(image),
//         bounding_boxes=bounding_boxes)
// 
//     # Draw the bounding box in an image summary.
//     image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
//                                                   bbox_for_draw)
//     tf.image_summary('images_with_box', image_with_box)
// 
//     # Employ the bounding box to distort the image.
//     distorted_image = tf.slice(image, begin, size)
// ```
// 
// Note that if no bounding box information is available, setting
// `use_image_if_no_bounding_boxes = true` will assume there is a single implicit
// bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
// false and no bounding boxes are supplied, an error is raised.

*/








func sampleDistortedBoundingBox( scope:Scope,imageSize: Output, boundingBoxes: Output, seed :UInt8  , seed2 :UInt8  , minObjectCovered :Float  , aspectRatioRange :[Float]  , areaRange :[Float]  , maxAttempts :UInt8  , useImageIfNoBoundingBoxes :Bool  )  ->(begin: Output?, size: Output?, bboxes: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["seed"] = seed
    attrs["seed2"] = seed2
    attrs["min_object_covered"] = minObjectCovered
    attrs["aspect_ratio_range"] = aspectRatioRange
    attrs["area_range"] = areaRange
    attrs["max_attempts"] = maxAttempts
    attrs["use_image_if_no_bounding_boxes"] = useImageIfNoBoundingBoxes

    let opspec = OpSpec(
        OpType: "SampleDistortedBoundingBox",
        Name: "Type",
        Input: [ imageSize, boundingBoxes],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Saves the input tensors to disk.

The size of `tensor_names` must match the number of tensors in `data`. `data[i]`
// is written to `filename` with name `tensor_names[i]`.
// 
// See also `SaveSlices`.

*/








func save( scope:Scope,filename: Output, tensorNames: Output, data: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Save",
        Name: "Type",
        Input: [ filename, tensorNames, data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Saves input tensors slices to disk.

This is like `Save` except that tensors can be listed in the saved file as being
// a slice of a larger tensor.  `shapes_and_slices` specifies the shape of the
// larger tensor and the slice that this tensor covers. `shapes_and_slices` must
// have as many elements as `tensor_names`.
// 
// Elements of the `shapes_and_slices` input must either be:
// 
//  *   The empty string, in which case the corresponding tensor is
//    saved normally.
//  *   A string of the form `dim0 dim1 ... dimN-1 slice-spec` where the
//    `dimI` are the dimensions of the larger tensor and `slice-spec`
//    specifies what part is covered by the tensor to save.
// 
// `slice-spec` itself is a `:`-separated list: `slice0:slice1:...:sliceN-1`
// where each `sliceI` is either:
// 
//  *   The string `-` meaning that the slice covers all indices of this dimension
//  *   `start,length` where `start` and `length` are integers.  In that
//    case the slice covers `length` indices starting at `start`.
// 
// See also `Save`.

*/








func saveSlices( scope:Scope,filename: Output, tensorNames: Output, shapesAndSlices: Output, data: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SaveSlices",
        Name: "Type",
        Input: [ filename, tensorNames, shapesAndSlices, data],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Saves tensors in V2 checkpoint format.

By default, saves the named tensors in full.  If the caller wishes to save
// specific slices of full tensors, "shape_and_slices" should be non-empty strings
// and correspondingly well-formed.

*/








func saveV2( scope:Scope,`prefix`: Output, tensorNames: Output, shapeAndSlices: Output, tensors: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SaveV2",
        Name: "Type",
        Input: [ `prefix`, tensorNames, shapeAndSlices, tensors],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Outputs a `Summary` protocol buffer with scalar values.

The input `tags` and `values` must have the same shape.  The generated summary
// has a summary value for each tag-value pair in `tags` and `values`.

*/








func scalarSummary( scope:Scope,tags: Output, values: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ScalarSummary",
        Name: "Type",
        Input: [ tags, values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Adds sparse updates to a variable reference.

This operation computes
// 
//     # Scalar indices
//     ref[indices, ...] += updates[...]
// 
//     # Vector indices (for each i)
//     ref[indices[i], ...] += updates[i, ...]
// 
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]
// 
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
// 
// Duplicate entries are handled correctly: if multiple `indices` reference
// the same location, their contributions add.
// 
// Requires `updates.shape = indices.shape + ref.shape[1:]`.
// 
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterAdd.png" alt>
// </div>

*/








func scatterAdd( scope:Scope,ref: Output, indices: Output, updates: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ScatterAdd",
        Name: "Type",
        Input: [ ref, indices, updates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Divides a variable reference by sparse updates.

This operation computes
// 
// ```python
//     # Scalar indices
//     ref[indices, ...] /= updates[...]
// 
//     # Vector indices (for each i)
//     ref[indices[i], ...] /= updates[i, ...]
// 
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...] /= updates[i, ..., j, ...]
// ```
// 
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
// 
// Duplicate entries are handled correctly: if multiple `indices` reference
// the same location, their contributions divide.
// 
// Requires `updates.shape = indices.shape + ref.shape[1:]`.

*/








func scatterDiv( scope:Scope,ref: Output, indices: Output, updates: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ScatterDiv",
        Name: "Type",
        Input: [ ref, indices, updates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Multiplies sparse updates into a variable reference.

This operation computes
// 
// ```python
//     # Scalar indices
//     ref[indices, ...]  * = updates[...]
// 
//     # Vector indices (for each i)
//     ref[indices[i], ...]  * = updates[i, ...]
// 
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...]  * = updates[i, ..., j, ...]
// ```
// 
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
// 
// Duplicate entries are handled correctly: if multiple `indices` reference
// the same location, their contributions multiply.
// 
// Requires `updates.shape = indices.shape + ref.shape[1:]`.

*/








func scatterMul( scope:Scope,ref: Output, indices: Output, updates: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ScatterMul",
        Name: "Type",
        Input: [ ref, indices, updates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Scatter `updates` into a new (initially zero) tensor according to `indices`.

Creates a new tensor by applying sparse `updates` to individual
// values or slices within a zero tensor of the given `shape` according to
// indices.  This operator is the inverse of the [tf.gather_nd](#gather_nd)
// operator which extracts values or slices from a given tensor.
// 
//  *  * WARNING *  * : The order in which updates are applied is nondeterministic, so the
// output will be nondeterministic if `indices` contains duplicates.
// 
// `indices` is an integer tensor containing indices into a new tensor of shape
// `shape`.  The last dimension of `indices` can be at most the rank of `shape`:
// 
//     indices.shape[-1] <= shape.rank
// 
// The last dimension of `indices` corresponds to indices into elements
// (if `indices.shape[-1] = shape.rank`) or slices
// (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
// `shape`.  `updates` is a tensor with shape
// 
//     indices.shape[:-1] + shape[indices.shape[-1]:]
// 
// The simplest form of scatter is to insert individual elements in a tensor by
// index. For example, say we want to insert 4 scattered elements in a rank-1
// tensor with 8 elements.
// 
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd1.png" alt>
// </div>
// 
// In Python, this scatter operation would look like this:
// 
// ```python
//     indices = tf.constant([[4], [3], [1], [7]])
//     updates = tf.constant([9, 10, 11, 12])
//     shape = tf.constant([8])
//     scatter = tf.scatter_nd(indices, updates, shape)
//     with tf.Session() as sess:
//       print(sess.run(scatter))
// ```
// 
// The resulting tensor would look like this:
// 
//     [0, 11, 0, 10, 9, 0, 0, 12]
// 
// We can also, insert entire slices of a higher rank tensor all at once. For
// example, if we wanted to insert two slices in the first dimension of a
// rank-3 tensor with two matrices of new values.
// 
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd2.png" alt>
// </div>
// 
// In Python, this scatter operation would look like this:
// 
// ```python
//     indices = tf.constant([[0], [2]])
//     updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
//                             [7, 7, 7, 7], [8, 8, 8, 8]],
//                            [[5, 5, 5, 5], [6, 6, 6, 6],
//                             [7, 7, 7, 7], [8, 8, 8, 8]]])
//     shape = tf.constant([4, 4, 4])
//     scatter = tf.scatter_nd(indices, updates, shape)
//     with tf.Session() as sess:
//       print(sess.run(scatter))
// ```
// 
// The resulting tensor would look like this:
// 
//     [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
//      [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
//      [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
//      [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

*/








func scatterNd( scope:Scope,indices: Output, updates: Output, shape: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ScatterNd",
        Name: "Type",
        Input: [ indices, updates, shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Applies sparse addition between `updates` and individual values or slices

within a given variable according to `indices`.
// 
// `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
// 
// `indices` must be integer tensor, containing indices into `ref`.
// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
// 
// The innermost dimension of `indices` (with length `K`) corresponds to
// indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
// dimension of `ref`.
// 
// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
// 
// ```
// [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
// ```
// 
// For example, say we want to add 4 scattered elements to a rank-1 tensor to 8
// elements. In Python, that addition would look like this:
// 
//     ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
//     indices = tf.constant([[4], [3], [1], [7]])
//     updates = tf.constant([9, 10, 11, 12])
//     add = tf.scatter_nd_add(ref, indices, updates)
//     with tf.Session() as sess:
//       print sess.run(add)
// 
// The resulting update to ref would look like this:
// 
//     [1, 13, 3, 14, 14, 6, 7, 20]
// 
// See [tf.scatter_nd](#scatter_nd) for more details about how to make updates to
// slices.

*/








func scatterNdAdd( scope:Scope,ref: Output, indices: Output, updates: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ScatterNdAdd",
        Name: "Type",
        Input: [ ref, indices, updates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Applies sparse subtraction between `updates` and individual values or slices

within a given variable according to `indices`.
// 
// `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
// 
// `indices` must be integer tensor, containing indices into `ref`.
// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
// 
// The innermost dimension of `indices` (with length `K`) corresponds to
// indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
// dimension of `ref`.
// 
// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
// 
// ```
// [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
// ```
// 
// For example, say we want to subtract 4 scattered elements from a rank-1 tensor
// with 8 elements. In Python, that subtraction would look like this:
// 
//     ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
//     indices = tf.constant([[4], [3], [1], [7]])
//     updates = tf.constant([9, 10, 11, 12])
//     sub = tf.scatter_nd_sub(ref, indices, updates)
//     with tf.Session() as sess:
//       print sess.run(sub)
// 
// The resulting update to ref would look like this:
// 
//     [1, -9, 3, -6, -4, 6, 7, -4]
// 
// See [tf.scatter_nd](#scatter_nd) for more details about how to make updates to
// slices.

*/








func scatterNdSub( scope:Scope,ref: Output, indices: Output, updates: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ScatterNdSub",
        Name: "Type",
        Input: [ ref, indices, updates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Applies sparse `updates` to individual values or slices within a given

variable according to `indices`.
// 
// `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
// 
// `indices` must be integer tensor, containing indices into `ref`.
// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
// 
// The innermost dimension of `indices` (with length `K`) corresponds to
// indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
// dimension of `ref`.
// 
// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
// 
// ```
// [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
// ```
// 
// For example, say we want to update 4 scattered elements to a rank-1 tensor to
// 8 elements. In Python, that update would look like this:
// 
// ```python
//     ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
//     indices = tf.constant([[4], [3], [1] ,[7]])
//     updates = tf.constant([9, 10, 11, 12])
//     update = tf.scatter_nd_update(ref, indices, updates)
//     with tf.Session() as sess:
//       print sess.run(update)
// ```
// 
// The resulting update to ref would look like this:
// 
//     [1, 11, 3, 10, 9, 6, 7, 12]
// 
// See [tf.scatter_nd](#scatter_nd) for more details about how to make updates to
// slices.

*/








func scatterNdUpdate( scope:Scope,ref: Output, indices: Output, updates: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ScatterNdUpdate",
        Name: "Type",
        Input: [ ref, indices, updates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Subtracts sparse updates to a variable reference.

```python
//     # Scalar indices
//     ref[indices, ...] -= updates[...]
// 
//     # Vector indices (for each i)
//     ref[indices[i], ...] -= updates[i, ...]
// 
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]
// ```
// 
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
// 
// Duplicate entries are handled correctly: if multiple `indices` reference
// the same location, their (negated) contributions add.
// 
// Requires `updates.shape = indices.shape + ref.shape[1:]`.
// 
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterSub.png" alt>
// </div>

*/








func scatterSub( scope:Scope,ref: Output, indices: Output, updates: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ScatterSub",
        Name: "Type",
        Input: [ ref, indices, updates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Applies sparse updates to a variable reference.

This operation computes
// 
// ```python
//     # Scalar indices
//     ref[indices, ...] = updates[...]
// 
//     # Vector indices (for each i)
//     ref[indices[i], ...] = updates[i, ...]
// 
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]
// ```
// 
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
// 
// If values in `ref` is to be updated more than once, because there are
// duplicate entries in `indices`, the order at which the updates happen
// for each value is undefined.
// 
// Requires `updates.shape = indices.shape + ref.shape[1:]`.
// 
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterUpdate.png" alt>
// </div>

*/








func scatterUpdate( scope:Scope,ref: Output, indices: Output, updates: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "ScatterUpdate",
        Name: "Type",
        Input: [ ref, indices, updates],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes fingerprints of the input strings.


*/








func sdcaFprint( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SdcaFprint",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Distributed version of Stochastic Dual Coordinate Ascent (SDCA) optimizer for

linear models with L1 + L2 regularization. As global optimization objective is
// strongly-convex, the optimizer optimizes the dual objective at each step. The
// optimizer applies each update one example at a time. Examples are sampled
// uniformly, and the optimizer is learning rate free and enjoys linear convergence
// rate.
// 
// [Proximal Stochastic Dual Coordinate Ascent](http://arxiv.org/pdf/1211.2717v1.pdf).<br>
// Shai Shalev-Shwartz, Tong Zhang. 2012
// 
// $$Loss Objective = \sum f_{i} (wx_{i}) + (l2 / 2)  *  |w|// ^2 + l1  *  |w|$$
// 
// [Adding vs. Averaging in Distributed Primal-Dual Optimization](http://arxiv.org/abs/1502.03508).<br>
// Chenxin Ma, Virginia Smith, Martin Jaggi, Michael I. Jordan,
// Peter Richtarik, Martin Takac. 2015
// 
// [Stochastic Dual Coordinate Ascent with Adaptive Probabilities](https://arxiv.org/abs/1502.08053).<br>
// Dominik Csiba, Zheng Qu, Peter Richtarik. 2015

*/








func sdcaOptimizer( scope:Scope,sparseExampleIndices: Output, sparseFeatureIndices: Output, sparseFeatureValues: Output, denseFeatures: Output, exampleWeights: Output, exampleLabels: Output, sparseIndices: Output, sparseWeights: Output, denseWeights: Output, exampleStateData: Output, lossType :String  , adaptative :Bool  , numSparseFeatures :UInt8  , numSparseFeaturesWithValues :UInt8  , numDenseFeatures :UInt8  , l1 :Float  , l2 :Float  , numLossPartitions :UInt8  , numInnerIterations :UInt8  )  ->(outExampleStateData: Output?, outDeltaSparseWeights: Output?, outDeltaDenseWeights: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["loss_type"] = lossType
    attrs["adaptative"] = adaptative
    attrs["num_sparse_features"] = numSparseFeatures
    attrs["num_sparse_features_with_values"] = numSparseFeaturesWithValues
    attrs["num_dense_features"] = numDenseFeatures
    attrs["l1"] = l1
    attrs["l2"] = l2
    attrs["num_loss_partitions"] = numLossPartitions
    attrs["num_inner_iterations"] = numInnerIterations

    let opspec = OpSpec(
        OpType: "SdcaOptimizer",
        Name: "Type",
        Input: [ sparseExampleIndices, sparseFeatureIndices, sparseFeatureValues, denseFeatures, exampleWeights, exampleLabels, sparseIndices, sparseWeights, denseWeights, exampleStateData],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Applies L1 regularization shrink step on the parameters.


*/








func sdcaShrinkL1( scope:Scope,weights: Output, numFeatures :UInt8  , l1 :Float  , l2 :Float  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_features"] = numFeatures
    attrs["l1"] = l1
    attrs["l2"] = l2

    let opspec = OpSpec(
        OpType: "SdcaShrinkL1",
        Name: "Type",
        Input: [ weights],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Computes the maximum along segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
// segments.
// 
// Computes a tensor such that
// \\(output_i = \max_j(data_j)\\) where `max` is over `j` such
// that `segment_ids[j] == i`.
// 
// If the max is empty for a given segment ID `i`, `output[i] = 0`.
// 
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMax.png" alt>
// </div>

*/








func segmentMax( scope:Scope,data: Output, segmentIds: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SegmentMax",
        Name: "Type",
        Input: [ data, segmentIds],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the mean along segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
// segments.
// 
// Computes a tensor such that
// \\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
// over `j` such that `segment_ids[j] == i` and `N` is the total number of
// values summed.
// 
// If the mean is empty for a given segment ID `i`, `output[i] = 0`.
// 
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMean.png" alt>
// </div>

*/








func segmentMean( scope:Scope,data: Output, segmentIds: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SegmentMean",
        Name: "Type",
        Input: [ data, segmentIds],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the minimum along segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
// segments.
// 
// Computes a tensor such that
// \\(output_i = \min_j(data_j)\\) where `min` is over `j` such
// that `segment_ids[j] == i`.
// 
// If the min is empty for a given segment ID `i`, `output[i] = 0`.
// 
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMin.png" alt>
// </div>

*/








func segmentMin( scope:Scope,data: Output, segmentIds: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SegmentMin",
        Name: "Type",
        Input: [ data, segmentIds],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the product along segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
// segments.
// 
// Computes a tensor such that
// \\(output_i = \prod_j data_j\\) where the product is over `j` such
// that `segment_ids[j] == i`.
// 
// If the product is empty for a given segment ID `i`, `output[i] = 1`.
// 
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentProd.png" alt>
// </div>

*/








func segmentProd( scope:Scope,data: Output, segmentIds: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SegmentProd",
        Name: "Type",
        Input: [ data, segmentIds],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the sum along segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
// segments.
// 
// Computes a tensor such that
// \\(output_i = \sum_j data_j\\) where sum is over `j` such
// that `segment_ids[j] == i`.
// 
// If the sum is empty for a given segment ID `i`, `output[i] = 0`.
// 
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentSum.png" alt>
// </div>

*/








func segmentSum( scope:Scope,data: Output, segmentIds: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SegmentSum",
        Name: "Type",
        Input: [ data, segmentIds],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Selects elements from `t` or `e`, depending on `condition`.

The `t`, and `e` tensors must all have the same shape, and the
// output will also have that shape.
// 
// The `condition` tensor must be a scalar if `t` and `e` are scalars.
// If `t` and `e` are vectors or higher rank, then `condition` must be either a
// scalar, a vector with size matching the first dimension of `t`, or must have
// the same shape as `t`.
// 
// The `condition` tensor acts as a mask that chooses, based on the value at each
// element, whether the corresponding element / row in the output should be
// taken from `t` (if true) or `e` (if false).
// 
// If `condition` is a vector and `t` and `e` are higher rank matrices, then
// it chooses which row (outer dimension) to copy from `t` and `e`.
// If `condition` has the same shape as `t` and `e`, then it chooses which
// element to copy from `t` and `e`.
// 
// For example:
// 
// ```python
// # 'condition' tensor is [[True,  False]
// #                        [False, True]]
// # 't' is [[1, 2],
// #         [3, 4]]
// # 'e' is [[5, 6],
// #         [7, 8]]
// select(condition, t, e)  # => [[1, 6], [7, 4]]
// 
// 
// # 'condition' tensor is [True, False]
// # 't' is [[1, 2],
// #         [3, 4]]
// # 'e' is [[5, 6],
// #         [7, 8]]
// select(condition, t, e) ==> [[1, 2],
//                              [7, 8]]
// 
// ```

*/








func select( scope:Scope,condition: Output, t: Output, e: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Select",
        Name: "Type",
        Input: [ condition, t, e],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the Eigen Decomposition of a batch of square self-adjoint matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
// form square matrices, with the same constraints as the single matrix
// SelfAdjointEig.
// 
// The result is a [..., M+1, M] matrix with [..., 0,:] containing the
// eigenvalues, and subsequent [...,1:, :] containing the eigenvectors.

*/








func selfAdjointEig( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SelfAdjointEig",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the eigen decomposition of one or more square self-adjoint matrices.

Computes the eigenvalues and (optionally) eigenvectors of each inner matrix in
// `input` such that `input[..., :, :] = v[..., :, :]  *  diag(e[..., :])`.
// 
// ```python
// # a is a tensor.
// # e is a tensor of eigenvalues.
// # v is a tensor of eigenvectors.
// e, v = self_adjoint_eig(a)
// e = self_adjoint_eig(a, compute_v=False)
// ```

*/








func selfAdjointEigV2( scope:Scope,input: Output, computeV :Bool  )  ->(e: Output?, v: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["compute_v"] = computeV

    let opspec = OpSpec(
        OpType: "SelfAdjointEigV2",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` string `Tensor`.

The `SparseTensor` must have rank `R` greater than 1, and the first dimension
// is treated as the minibatch dimension.  Elements of the `SparseTensor`
// must be sorted in increasing order of this first dimension.  The serialized
// `SparseTensor` objects going into each row of `serialized_sparse` will have
// rank `R-1`.
// 
// The minibatch size `N` is extracted from `sparse_shape[0]`.

*/








func serializeManySparse( scope:Scope,sparseIndices: Output, sparseValues: Output, sparseShape: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SerializeManySparse",
        Name: "Type",
        Input: [ sparseIndices, sparseValues, sparseShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Serialize a `SparseTensor` into a string 3-vector (1-D `Tensor`) object.


*/








func serializeSparse( scope:Scope,sparseIndices: Output, sparseValues: Output, sparseShape: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SerializeSparse",
        Name: "Type",
        Input: [ sparseIndices, sparseValues, sparseShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Number of unique elements along last dimension of input `set`.

Input `set` is a `SparseTensor` represented by `set_indices`, `set_values`,
// and `set_shape`. The last dimension contains values in a set, duplicates are
// allowed but ignored.
// 
// If `validate_indices` is `True`, this op validates the order and range of `set`
// indices.

*/








func setSize( scope:Scope,setIndices: Output, setValues: Output, setShape: Output, validateIndices :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["validate_indices"] = validateIndices

    let opspec = OpSpec(
        OpType: "SetSize",
        Name: "Type",
        Input: [ setIndices, setValues, setShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the shape of a tensor.

This operation returns a 1-D integer tensor representing the shape of `input`.
// 
// For example:
// 
// ```
// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
// shape(t) ==> [2, 2, 3]
// ```

*/








func shape( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Shape",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns shape of tensors.

This operation returns N 1-D integer tensors representing shape of `input[i]s`.

*/








func shapeN( scope:Scope,input: Output, n :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["N"] = n

    let opspec = OpSpec(
        OpType: "ShapeN",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Generate a sharded filename. The filename is printf formatted as

   %s-%05d-of-%05d, basename, shard, num_shards.

*/








func shardedFilename( scope:Scope,basename: Output, shard: Output, numShards: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ShardedFilename",
        Name: "Type",
        Input: [ basename, shard, numShards],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Generate a glob pattern matching all sharded file names.


*/








func shardedFilespec( scope:Scope,basename: Output, numShards: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ShardedFilespec",
        Name: "Type",
        Input: [ basename, numShards],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a dataset that shuffles elements from `input_dataset` pseudorandomly.


*/








func shuffleDataset( scope:Scope,inputDataset: Output, bufferSize: Output, seed: Output, seed2: Output, outputShapes :[Shape]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["output_shapes"] = outputShapes

    let opspec = OpSpec(
        OpType: "ShuffleDataset",
        Name: "Type",
        Input: [ inputDataset, bufferSize, seed, seed2],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes sigmoid of `x` element-wise.

Specifically, `y = 1 / (1 + exp(-x))`.

*/








func sigmoid( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Sigmoid",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradient of the sigmoid of `x` wrt its input.

Specifically, `grad = dy  *  y  *  (1 - y)`, where `y = sigmoid(x)`, and
// `dy` is the corresponding input gradient.

*/








func sigmoidGrad( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SigmoidGrad",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns an element-wise indication of the sign of a number.

`y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.
// 
// For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.

*/








func sign( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Sign",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes sin of x element-wise.


*/








func sin( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Sin",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the size of a tensor.

This operation returns an integer representing the number of elements in
// `input`.
// 
// For example:
// 
// ```
// # 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
// size(t) ==> 12
// ```

*/








func size( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Size",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a dataset that skips `count` elements from the `input_dataset`.


*/








func skipDataset( scope:Scope,inputDataset: Output, count: Output, outputShapes :[Shape]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["output_shapes"] = outputShapes

    let opspec = OpSpec(
        OpType: "SkipDataset",
        Name: "Type",
        Input: [ inputDataset, count],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Parses a text file and creates a batch of examples.


*/








func skipgram( scope:Scope, filename :String  , batchSize :UInt8  , windowSize :UInt8  , minCount :UInt8  , subsample :Float  )  ->(vocabWord: Output?, vocabFreq: Output?, wordsPerEpoch: Output?, currentEpoch: Output?, totalWordsProcessed: Output?, examples: Output?, labels: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil, nil, nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["filename"] = filename
    attrs["batch_size"] = batchSize
    attrs["window_size"] = windowSize
    attrs["min_count"] = minCount
    attrs["subsample"] = subsample

    let opspec = OpSpec(
        OpType: "Skipgram",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1),op?.output(4 - 1),op?.output(5 - 1),op?.output(6 - 1),op?.output(7 - 1))
}

/*
Return a slice from 'input'.

The output tensor is a tensor with dimensions described by 'size'
// whose values are extracted from 'input' starting at the offsets in
// 'begin'.
// 
//  * Requirements * :
//   0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)

*/








func slice( scope:Scope,input: Output, begin: Output, size: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Slice",
        Name: "Type",
        Input: [ input, begin, size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes softmax activations.

For each batch `i` and class `j` we have
// 
//     softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))

*/








func softmax( scope:Scope,logits: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Softmax",
        Name: "Type",
        Input: [ logits],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes softmax cross entropy cost and gradients to backpropagate.

Inputs are the logits, not probabilities.

*/








func softmaxCrossEntropyWithLogits( scope:Scope,features: Output, labels: Output )  ->(loss: Output?, backprop: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SoftmaxCrossEntropyWithLogits",
        Name: "Type",
        Input: [ features, labels],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Computes softplus: `log(exp(features) + 1)`.


*/








func softplus( scope:Scope,features: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Softplus",
        Name: "Type",
        Input: [ features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes softplus gradients for a softplus operation.


*/








func softplusGrad( scope:Scope,gradients: Output, features: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SoftplusGrad",
        Name: "Type",
        Input: [ gradients, features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes softsign: `features / (abs(features) + 1)`.


*/








func softsign( scope:Scope,features: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Softsign",
        Name: "Type",
        Input: [ features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes softsign gradients for a softsign operation.


*/








func softsignGrad( scope:Scope,gradients: Output, features: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SoftsignGrad",
        Name: "Type",
        Input: [ gradients, features],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
SpaceToBatch for 4-D tensors of type T.

This is a legacy version of the more general SpaceToBatchND.
// 
// Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
// More specifically, this op outputs a copy of the input tensor where values from
// the `height` and `width` dimensions are moved to the `batch` dimension. After
// the zero-padding, both `height` and `width` of the input must be divisible by the
// block size.

*/








func spaceToBatch( scope:Scope,input: Output, paddings: Output, blockSize :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["block_size"] = blockSize

    let opspec = OpSpec(
        OpType: "SpaceToBatch",
        Name: "Type",
        Input: [ input, paddings],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
SpaceToBatch for N-D tensors of type T.

This operation divides "spatial" dimensions `[1, ..., M]` of the input into a
// grid of blocks of shape `block_shape`, and interleaves these blocks with the
// "batch" dimension (0) such that in the output, the spatial dimensions
// `[1, ..., M]` correspond to the position within the grid, and the batch
// dimension combines both the position within a spatial block and the original
// batch position.  Prior to division into blocks, the spatial dimensions of the
// input are optionally zero padded according to `paddings`.  See below for a
// precise description.

*/








func spaceToBatchND( scope:Scope,input: Output, blockShape: Output, paddings: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SpaceToBatchND",
        Name: "Type",
        Input: [ input, blockShape, paddings],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
SpaceToDepth for tensors of type T.

Rearranges blocks of spatial data, into depth. More specifically,
// this op outputs a copy of the input tensor where values from the `height`
// and `width` dimensions are moved to the `depth` dimension.
// The attr `block_size` indicates the input block size and how the data is moved.
// 
//    *  Non-overlapping blocks of size `block_size x block size` are rearranged
//     into depth at each location.
//    *  The depth of the output tensor is `input_depth  *  block_size  *  block_size`.
//    *  The input tensor's height and width must be divisible by block_size.
// 
// That is, assuming the input is in the shape:
// `[batch, height, width, depth]`,
// the shape of the output will be:
// `[batch, height/block_size, width/block_size, depth * block_size * block_size]`
// 
// This operation requires that the input tensor be of rank 4, and that
// `block_size` be >=1 and a divisor of both the input `height` and `width`.
// 
// This operation is useful for resizing the activations between convolutions
// (but keeping all data), e.g. instead of pooling. It is also useful for training
// purely convolutional models.
// 
// For example, given this input of shape `[1, 2, 2, 1]`, and block_size of 2:
// 
// ```
// x = [[[[1], [2]],
//       [[3], [4]]]]
// ```
// 
// This operation will output a tensor of shape `[1, 1, 1, 4]`:
// 
// ```
// [[[[1, 2, 3, 4]]]]
// ```
// 
// Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
// the corresponding output will have a single element (i.e. width and height are
// both 1) and will have a depth of 4 channels (1  *  block_size  *  block_size).
// The output element shape is `[1, 1, 4]`.
// 
// For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.
// 
// ```
// x = [[[[1, 2, 3], [4, 5, 6]],
//       [[7, 8, 9], [10, 11, 12]]]]
// ```
// 
// This operation, for block_size of 2, will return the following tensor of shape
// `[1, 1, 1, 12]`
// 
// ```
// [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
// ```
// 
// Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:
// 
// ```
// x = [[[[1],   [2],  [5],  [6]],
//       [[3],   [4],  [7],  [8]],
//       [[9],  [10], [13],  [14]],
//       [[11], [12], [15],  [16]]]]
// ```
// 
// the operator will return the following tensor of shape `[1 2 2 4]`:
// 
// ```
// x = [[[[1, 2, 3, 4],
//        [5, 6, 7, 8]],
//       [[9, 10, 11, 12],
//        [13, 14, 15, 16]]]]
// ```

*/








func spaceToDepth( scope:Scope,input: Output, blockSize :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["block_size"] = blockSize

    let opspec = OpSpec(
        OpType: "SpaceToDepth",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Applies a sparse gradient to a given accumulator.

Does not add if local_step is smaller than the accumulator's
// global_step.

*/








func sparseAccumulatorApplyGradient( scope:Scope,handle: Output, localStep: Output, gradientIndices: Output, gradientValues: Output, gradientShape: Output, hasKnownShape :Bool  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["has_known_shape"] = hasKnownShape

    let opspec = OpSpec(
        OpType: "SparseAccumulatorApplyGradient",
        Name: "Type",
        Input: [ handle, localStep, gradientIndices, gradientValues, gradientShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Extracts the average sparse gradient in a SparseConditionalAccumulator.

The op will blocks until sufficient (i.e., more than num_required)
// gradients have been accumulated. If the accumulator has already
// aggregated more than num_required gradients, it will return its
// average of the accumulated gradients.  Also automatically increments
// the recorded global_step in the accumulator by 1, and resets the
// aggregate to 0.

*/








func sparseAccumulatorTakeGradient( scope:Scope,handle: Output, numRequired: Output )  ->(indices: Output?, values: Output?, shape: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseAccumulatorTakeGradient",
        Name: "Type",
        Input: [ handle, numRequired],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Adds two `SparseTensor` objects to produce another `SparseTensor`.

The input `SparseTensor` objects' indices are assumed ordered in standard
// lexicographic order.  If this is not the case, before this step run
// `SparseReorder` to restore index ordering.
// 
// By default, if two values sum to zero at some index, the output `SparseTensor`
// would still include that particular location in its index, storing a zero in the
// corresponding value slot.  To override this, callers can specify `thresh`,
// indicating that if the sum has a magnitude strictly smaller than `thresh`, its
// corresponding value and index would then not be included.  In particular,
// `thresh == 0` (default) means everything is kept and actual thresholding happens
// only for a positive value.
// 
// In the following shapes, `nnz` is the count after taking `thresh` into account.

*/








func sparseAdd( scope:Scope,aIndices: Output, aValues: Output, aShape: Output, bIndices: Output, bValues: Output, bShape: Output, thresh: Output )  ->(sumIndices: Output?, sumValues: Output?, sumShape: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseAdd",
        Name: "Type",
        Input: [ aIndices, aValues, aShape, bIndices, bValues, bShape, thresh],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
The gradient operator for the SparseAdd op.

The SparseAdd op calculates A + B, where A, B, and the sum are all represented
// as `SparseTensor` objects.  This op takes in the upstream gradient w.r.t.
// non-empty values of the sum, and outputs the gradients w.r.t. the non-empty
// values of A and B.

*/








func sparseAddGrad( scope:Scope,backpropValGrad: Output, aIndices: Output, bIndices: Output, sumIndices: Output )  ->(aValGrad: Output?, bValGrad: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseAddGrad",
        Name: "Type",
        Input: [ backpropValGrad, aIndices, bIndices, sumIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
var: Should be from a Variable().


*/








func sparseApplyAdadelta( scope:Scope,`var`: Output, accum: Output, accumUpdate: Output, lr: Output, rho: Output, epsilon: Output, grad: Output, indices: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "SparseApplyAdadelta",
        Name: "Type",
        Input: [ `var`, accum, accumUpdate, lr, rho, epsilon, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update relevant entries in '*var' and '*accum' according to the adagrad scheme.

That is for rows we have grad for, we update var and accum as follows:
// accum += grad  *  grad
// var -= lr  *  grad  *  (1 / sqrt(accum))

*/








func sparseApplyAdagrad( scope:Scope,`var`: Output, accum: Output, lr: Output, grad: Output, indices: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "SparseApplyAdagrad",
        Name: "Type",
        Input: [ `var`, accum, lr, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update entries in '*var' and '*accum' according to the proximal adagrad scheme.


*/








func sparseApplyAdagradDA( scope:Scope,`var`: Output, gradientAccumulator: Output, gradientSquaredAccumulator: Output, grad: Output, indices: Output, lr: Output, l1: Output, l2: Output, globalStep: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "SparseApplyAdagradDA",
        Name: "Type",
        Input: [ `var`, gradientAccumulator, gradientSquaredAccumulator, grad, indices, lr, l1, l2, globalStep],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update '*var' according to the centered RMSProp algorithm.

The centered RMSProp algorithm uses an estimate of the centered second moment
// (i.e., the variance) for normalization, as opposed to regular RMSProp, which
// uses the (uncentered) second moment. This often helps with training, but is
// slightly more expensive in terms of computation and memory.
// 
// Note that in dense implementation of this algorithm, mg, ms, and mom will
// update even if the grad is zero, but in this sparse implementation, mg, ms,
// and mom will not update in iterations during which the grad is zero.
// 
// mean_square = decay  *  mean_square + (1-decay)  *  gradient  *  *  2
// mean_grad = decay  *  mean_grad + (1-decay)  *  gradient
// Delta = learning_rate  *  gradient / sqrt(mean_square + epsilon - mean_grad  *  *  2)
// 
// ms <- rho  *  ms_{t-1} + (1-rho)  *  grad  *  grad
// mom <- momentum  *  mom_{t-1} + lr  *  grad / sqrt(ms + epsilon)
// var <- var - mom

*/








func sparseApplyCenteredRMSProp( scope:Scope,`var`: Output, mg: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, indices: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "SparseApplyCenteredRMSProp",
        Name: "Type",
        Input: [ `var`, mg, ms, mom, lr, rho, momentum, epsilon, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update relevant entries in '*var' according to the Ftrl-proximal scheme.

That is for rows we have grad for, we update var, accum and linear as follows:
// accum_new = accum + grad  *  grad
// linear += grad + (accum_new// ^(-lr_power) - accum// ^(-lr_power)) / lr  *  var
// quadratic = 1.0 / (accum_new// ^(lr_power)  *  lr) + 2  *  l2
// var = (sign(linear)  *  l1 - linear) / quadratic if |linear| > l1 else 0.0
// accum = accum_new

*/








func sparseApplyFtrl( scope:Scope,`var`: Output, accum: Output, linear: Output, grad: Output, indices: Output, lr: Output, l1: Output, l2: Output, lrPower: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "SparseApplyFtrl",
        Name: "Type",
        Input: [ `var`, accum, linear, grad, indices, lr, l1, l2, lrPower],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update relevant entries in '*var' and '*accum' according to the momentum scheme.

Set use_nesterov = True if you want to use Nesterov momentum.
// 
// That is for rows we have grad for, we update var and accum as follows:
// 
// accum = accum  *  momentum + grad
// var -= lr  *  accum

*/








func sparseApplyMomentum( scope:Scope,`var`: Output, accum: Output, lr: Output, grad: Output, indices: Output, momentum: Output, useLocking :Bool  , useNesterov :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking
    attrs["use_nesterov"] = useNesterov

    let opspec = OpSpec(
        OpType: "SparseApplyMomentum",
        Name: "Type",
        Input: [ `var`, accum, lr, grad, indices, momentum],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Sparse update entries in '*var' and '*accum' according to FOBOS algorithm.

That is for rows we have grad for, we update var and accum as follows:
// accum += grad  *  grad
// prox_v = var
// prox_v -= lr  *  grad  *  (1 / sqrt(accum))
// var = sign(prox_v)/(1+lr * l2)  *  max{|prox_v|-lr * l1,0}

*/








func sparseApplyProximalAdagrad( scope:Scope,`var`: Output, accum: Output, lr: Output, l1: Output, l2: Output, grad: Output, indices: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "SparseApplyProximalAdagrad",
        Name: "Type",
        Input: [ `var`, accum, lr, l1, l2, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Sparse update '*var' as FOBOS algorithm with fixed learning rate.

That is for rows we have grad for, we update var as follows:
// prox_v = var - alpha  *  grad
// var = sign(prox_v)/(1+alpha * l2)  *  max{|prox_v|-alpha * l1,0}

*/








func sparseApplyProximalGradientDescent( scope:Scope,`var`: Output, alpha: Output, l1: Output, l2: Output, grad: Output, indices: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "SparseApplyProximalGradientDescent",
        Name: "Type",
        Input: [ `var`, alpha, l1, l2, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Update '*var' according to the RMSProp algorithm.

Note that in dense implementation of this algorithm, ms and mom will
// update even if the grad is zero, but in this sparse implementation, ms
// and mom will not update in iterations during which the grad is zero.
// 
// mean_square = decay  *  mean_square + (1-decay)  *  gradient  *  *  2
// Delta = learning_rate  *  gradient / sqrt(mean_square + epsilon)
// 
// ms <- rho  *  ms_{t-1} + (1-rho)  *  grad  *  grad
// mom <- momentum  *  mom_{t-1} + lr  *  grad / sqrt(ms + epsilon)
// var <- var - mom

*/








func sparseApplyRMSProp( scope:Scope,`var`: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, indices: Output, useLocking :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["use_locking"] = useLocking

    let opspec = OpSpec(
        OpType: "SparseApplyRMSProp",
        Name: "Type",
        Input: [ `var`, ms, mom, lr, rho, momentum, epsilon, grad, indices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Concatenates a list of `SparseTensor` along the specified dimension.

Concatenation is with respect to the dense versions of these sparse tensors.
// It is assumed that each input is a `SparseTensor` whose elements are ordered
// along increasing dimension number.
// 
// All inputs' shapes must match, except for the concat dimension.  The
// `indices`, `values`, and `shapes` lists must have the same length.
// 
// The output shape is identical to the inputs', except along the concat
// dimension, where it is the sum of the inputs' sizes along that dimension.
// 
// The output elements will be resorted to preserve the sort order along
// increasing dimension number.
// 
// This op runs in `O(M log M)` time, where `M` is the total number of non-empty
// values across all inputs. This is due to the need for an internal sort in
// order to concatenate efficiently across an arbitrary dimension.
// 
// For example, if `concat_dim = 1` and the inputs are
// 
//     sp_inputs[0]: shape = [2, 3]
//     [0, 2]: "a"
//     [1, 0]: "b"
//     [1, 1]: "c"
// 
//     sp_inputs[1]: shape = [2, 4]
//     [0, 1]: "d"
//     [0, 2]: "e"
// 
// then the output will be
// 
//     shape = [2, 7]
//     [0, 2]: "a"
//     [0, 4]: "d"
//     [0, 5]: "e"
//     [1, 0]: "b"
//     [1, 1]: "c"
// 
// Graphically this is equivalent to doing
// 
//     [    a] concat [  d e  ] = [    a   d e  ]
//     [b c  ]        [       ]   [b c          ]

*/








func sparseConcat( scope:Scope,indices: Output, values: Output, shapes: Output, concatDim :UInt8  , n :UInt8  )  ->(outputIndices: Output?, outputValues: Output?, outputShape: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["concat_dim"] = concatDim
    attrs["N"] = n

    let opspec = OpSpec(
        OpType: "SparseConcat",
        Name: "Type",
        Input: [ indices, values, shapes],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
A conditional accumulator for aggregating sparse gradients.

The accumulator accepts gradients marked with local_step greater or
// equal to the most recent global_step known to the accumulator. The
// average can be extracted from the accumulator, provided sufficient
// gradients have been accumulated. Extracting the average automatically
// resets the aggregate to 0, and increments the global_step recorded by
// the accumulator.

*/








func sparseConditionalAccumulator( scope:Scope, shape :Shape  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shape"] = shape
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "SparseConditionalAccumulator",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Generates sparse cross from a list of sparse and dense tensors.

The op takes two lists, one of 2D `SparseTensor` and one of 2D `Tensor`, each
// representing features of one feature column. It outputs a 2D `SparseTensor` with
// the batchwise crosses of these features.
// 
// For example, if the inputs are
// 
//     inputs[0]: SparseTensor with shape = [2, 2]
//     [0, 0]: "a"
//     [1, 0]: "b"
//     [1, 1]: "c"
// 
//     inputs[1]: SparseTensor with shape = [2, 1]
//     [0, 0]: "d"
//     [1, 0]: "e"
// 
//     inputs[2]: Tensor [["f"], ["g"]]
// 
// then the output will be
// 
//     shape = [2, 2]
//     [0, 0]: "a_X_d_X_f"
//     [1, 0]: "b_X_e_X_g"
//     [1, 1]: "c_X_e_X_g"
// 
// if hashed_output=true then the output will be
// 
//     shape = [2, 2]
//     [0, 0]: FingerprintCat64(
//                 Fingerprint64("f"), FingerprintCat64(
//                     Fingerprint64("d"), Fingerprint64("a")))
//     [1, 0]: FingerprintCat64(
//                 Fingerprint64("g"), FingerprintCat64(
//                     Fingerprint64("e"), Fingerprint64("b")))
//     [1, 1]: FingerprintCat64(
//                 Fingerprint64("g"), FingerprintCat64(
//                     Fingerprint64("e"), Fingerprint64("c")))

*/








func sparseCross( scope:Scope,indices: Output, values: Output, shapes: Output, denseInputs: Output, n :UInt8  , hashedOutput :Bool  , numBuckets :UInt8  , hashKey :UInt8  )  ->(outputIndices: Output?, outputValues: Output?, outputShape: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["N"] = n
    attrs["hashed_output"] = hashedOutput
    attrs["num_buckets"] = numBuckets
    attrs["hash_key"] = hashKey

    let opspec = OpSpec(
        OpType: "SparseCross",
        Name: "Type",
        Input: [ indices, values, shapes, denseInputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Adds up a SparseTensor and a dense Tensor, using these special rules:

(1) Broadcasts the dense side to have the same shape as the sparse side, if
//     eligible;
// (2) Then, only the dense values pointed to by the indices of the SparseTensor
//     participate in the cwise addition.
// 
// By these rules, the result is a logical SparseTensor with exactly the same
// indices and shape, but possibly with different non-zero values.  The output of
// this Op is the resultant non-zero values.

*/








func sparseDenseCwiseAdd( scope:Scope,spIndices: Output, spValues: Output, spShape: Output, dense: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseDenseCwiseAdd",
        Name: "Type",
        Input: [ spIndices, spValues, spShape, dense],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Component-wise divides a SparseTensor by a dense Tensor.

 * Limitation * : this Op only broadcasts the dense side to the sparse side, but not
// the other direction.

*/








func sparseDenseCwiseDiv( scope:Scope,spIndices: Output, spValues: Output, spShape: Output, dense: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseDenseCwiseDiv",
        Name: "Type",
        Input: [ spIndices, spValues, spShape, dense],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Component-wise multiplies a SparseTensor by a dense Tensor.

The output locations corresponding to the implicitly zero elements in the sparse
// tensor will be zero (i.e., will not take up storage space), regardless of the
// contents of the dense tensor (even if it's +/-INF and that INF * 0 == NaN).
// 
//  * Limitation * : this Op only broadcasts the dense side to the sparse side, but not
// the other direction.

*/








func sparseDenseCwiseMul( scope:Scope,spIndices: Output, spValues: Output, spShape: Output, dense: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseDenseCwiseMul",
        Name: "Type",
        Input: [ spIndices, spValues, spShape, dense],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Fills empty rows in the input 2-D `SparseTensor` with a default value.

The input `SparseTensor` is represented via the tuple of inputs
// (`indices`, `values`, `dense_shape`).  The output `SparseTensor` has the
// same `dense_shape` but with indices `output_indices` and values
// `output_values`.
// 
// This op inserts a single entry for every row that doesn't have any values.
// The index is created as `[row, 0, ..., 0]` and the inserted value
// is `default_value`.
// 
// For example, suppose `sp_input` has shape `[5, 6]` and non-empty values:
// 
//     [0, 1]: a
//     [0, 3]: b
//     [2, 0]: c
//     [3, 1]: d
// 
// Rows 1 and 4 are empty, so the output will be of shape `[5, 6]` with values:
// 
//     [0, 1]: a
//     [0, 3]: b
//     [1, 0]: default_value
//     [2, 0]: c
//     [3, 1]: d
//     [4, 0]: default_value
// 
// The output `SparseTensor` will be in row-major order and will have the
// same shape as the input.
// 
// This op also returns an indicator vector shaped `[dense_shape[0]]` such that
// 
//     empty_row_indicator[i] = True iff row i was an empty row.
// 
// And a reverse index map vector shaped `[indices.shape[0]]` that is used during
// backpropagation,
// 
//     reverse_index_map[j] = out_j s.t. indices[j, :] == output_indices[out_j, :]

*/








func sparseFillEmptyRows( scope:Scope,indices: Output, values: Output, denseShape: Output, defaultValue: Output )  ->(outputIndices: Output?, outputValues: Output?, emptyRowIndicator: Output?, reverseIndexMap: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseFillEmptyRows",
        Name: "Type",
        Input: [ indices, values, denseShape, defaultValue],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1),op?.output(4 - 1))
}

/*
The gradient of SparseFillEmptyRows.

Takes vectors reverse_index_map, shaped `[N]`, and grad_values,
// shaped `[N_full]`, where `N_full >= N` and copies data into either
// `d_values` or `d_default_value`.  Here `d_values` is shaped `[N]` and
// `d_default_value` is a scalar.
// 
//   d_values[j] = grad_values[reverse_index_map[j]]
//   d_default_value = sum_{k : 0 .. N_full - 1} (
//      grad_values[k]  *  1{k not in reverse_index_map})

*/








func sparseFillEmptyRowsGrad( scope:Scope,reverseIndexMap: Output, gradValues: Output )  ->(dValues: Output?, dDefaultValue: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseFillEmptyRowsGrad",
        Name: "Type",
        Input: [ reverseIndexMap, gradValues],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Multiply matrix "a" by matrix "b".

The inputs must be two-dimensional matrices and the inner dimension of "a" must
// match the outer dimension of "b". This op is optimized for the case where at
// least one of "a" or "b" is sparse. The breakeven for using this versus a dense
// matrix multiply on one platform was 30% zero values in the sparse matrix.

*/








func sparseMatMul( scope:Scope,a: Output, b: Output, transposeA :Bool  , transposeB :Bool  , aIsSparse :Bool  , bIsSparse :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["transpose_a"] = transposeA
    attrs["transpose_b"] = transposeB
    attrs["a_is_sparse"] = aIsSparse
    attrs["b_is_sparse"] = bIsSparse

    let opspec = OpSpec(
        OpType: "SparseMatMul",
        Name: "Type",
        Input: [ a, b],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the sum of elements across dimensions of a SparseTensor.

This Op takes a SparseTensor and is the sparse counterpart to
// `tf.reduce_sum()`.  In particular, this Op also returns a dense `Tensor`
// instead of a sparse one.
// 
// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
// with length 1.
// 
// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
// with a single element is returned.  Additionally, the axes can be negative,
// which are interpreted according to the indexing rules in Python.

*/








func sparseReduceSum( scope:Scope,inputIndices: Output, inputValues: Output, inputShape: Output, reductionAxes: Output, keepDims :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["keep_dims"] = keepDims

    let opspec = OpSpec(
        OpType: "SparseReduceSum",
        Name: "Type",
        Input: [ inputIndices, inputValues, inputShape, reductionAxes],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the sum of elements across dimensions of a SparseTensor.

This Op takes a SparseTensor and is the sparse counterpart to
// `tf.reduce_sum()`.  In contrast to SparseReduceSum, this Op returns a
// SparseTensor.
// 
// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
// with length 1.
// 
// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
// with a single element is returned.  Additionally, the axes can be negative,
// which are interpreted according to the indexing rules in Python.

*/








func sparseReduceSumSparse( scope:Scope,inputIndices: Output, inputValues: Output, inputShape: Output, reductionAxes: Output, keepDims :Bool  )  ->(outputIndices: Output?, outputValues: Output?, outputShape: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["keep_dims"] = keepDims

    let opspec = OpSpec(
        OpType: "SparseReduceSumSparse",
        Name: "Type",
        Input: [ inputIndices, inputValues, inputShape, reductionAxes],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Reorders a SparseTensor into the canonical, row-major ordering.

Note that by convention, all sparse ops preserve the canonical ordering along
// increasing dimension number. The only time ordering can be violated is during
// manual manipulation of the indices and values vectors to add entries.
// 
// Reordering does not affect the shape of the SparseTensor.
// 
// If the tensor has rank `R` and `N` non-empty values, `input_indices` has
// shape `[N, R]`, input_values has length `N`, and input_shape has length `R`.

*/








func sparseReorder( scope:Scope,inputIndices: Output, inputValues: Output, inputShape: Output )  ->(outputIndices: Output?, outputValues: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseReorder",
        Name: "Type",
        Input: [ inputIndices, inputValues, inputShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Reshapes a SparseTensor to represent values in a new dense shape.

This operation has the same semantics as reshape on the represented dense
// tensor.  The `input_indices` are recomputed based on the requested `new_shape`.
// 
// If one component of `new_shape` is the special value -1, the size of that
// dimension is computed so that the total dense size remains constant.  At
// most one component of `new_shape` can be -1.  The number of dense elements
// implied by `new_shape` must be the same as the number of dense elements
// originally implied by `input_shape`.
// 
// Reshaping does not affect the order of values in the SparseTensor.
// 
// If the input tensor has rank `R_in` and `N` non-empty values, and `new_shape`
// has length `R_out`, then `input_indices` has shape `[N, R_in]`,
// `input_shape` has length `R_in`, `output_indices` has shape `[N, R_out]`, and
// `output_shape` has length `R_out`.

*/








func sparseReshape( scope:Scope,inputIndices: Output, inputShape: Output, newShape: Output )  ->(outputIndices: Output?, outputShape: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseReshape",
        Name: "Type",
        Input: [ inputIndices, inputShape, newShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Computes the mean along sparse segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
// segments.
// 
// Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
// dimension, selecting a subset of dimension 0, specified by `indices`.

*/








func sparseSegmentMean( scope:Scope,data: Output, indices: Output, segmentIds: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseSegmentMean",
        Name: "Type",
        Input: [ data, indices, segmentIds],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes gradients for SparseSegmentMean.

Returns tensor "output" with same shape as grad, except for dimension 0 whose
// value is output_dim0.

*/








func sparseSegmentMeanGrad( scope:Scope,grad: Output, indices: Output, segmentIds: Output, outputDim0: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseSegmentMeanGrad",
        Name: "Type",
        Input: [ grad, indices, segmentIds, outputDim0],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the sum along sparse segments of a tensor divided by the sqrt of N.

N is the size of the segment being reduced.
// 
// Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
// segments.

*/








func sparseSegmentSqrtN( scope:Scope,data: Output, indices: Output, segmentIds: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseSegmentSqrtN",
        Name: "Type",
        Input: [ data, indices, segmentIds],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes gradients for SparseSegmentSqrtN.

Returns tensor "output" with same shape as grad, except for dimension 0 whose
// value is output_dim0.

*/








func sparseSegmentSqrtNGrad( scope:Scope,grad: Output, indices: Output, segmentIds: Output, outputDim0: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseSegmentSqrtNGrad",
        Name: "Type",
        Input: [ grad, indices, segmentIds, outputDim0],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the sum along sparse segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
// segments.
// 
// Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
// dimension, selecting a subset of dimension 0, specified by `indices`.
// 
// For example:
// 
// ```python
// c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
// 
// # Select two rows, one segment.
// tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
// # => [[0 0 0 0]]
// 
// # Select two rows, two segment.
// tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
// # => [[ 1  2  3  4]
// #     [-1 -2 -3 -4]]
// 
// # Select all rows, two segments.
// tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
// # => [[0 0 0 0]
// #     [5 6 7 8]]
// 
// # Which is equivalent to:
// tf.segment_sum(c, tf.constant([0, 0, 1]))
// ```

*/








func sparseSegmentSum( scope:Scope,data: Output, indices: Output, segmentIds: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseSegmentSum",
        Name: "Type",
        Input: [ data, indices, segmentIds],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Applies softmax to a batched N-D `SparseTensor`.

The inputs represent an N-D SparseTensor  with logical shape `[..., B, C]`
// (where `N >= 2`), and with indices sorted in the canonical lexicographic order.
// 
// This op is equivalent to applying the normal `tf.nn.softmax()` to each innermost
// logical submatrix with shape `[B, C]`, but with the catch that  * the implicitly
// zero elements do not participate * .  Specifically, the algorithm is equivalent
// to the following:
// 
//   (1) Applies `tf.nn.softmax()` to a densified view of each innermost submatrix
//       with shape `[B, C]`, along the size-C dimension;
//   (2) Masks out the original implicitly-zero locations;
//   (3) Renormalizes the remaining elements.
// 
// Hence, the `SparseTensor` result has exactly the same non-zero indices and
// shape.

*/








func sparseSoftmax( scope:Scope,spIndices: Output, spValues: Output, spShape: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseSoftmax",
        Name: "Type",
        Input: [ spIndices, spValues, spShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes softmax cross entropy cost and gradients to backpropagate.

Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
// a matrix of label probabilities, but rather a single label per row
// of features.  This label is considered to have probability 1.0 for the
// given row.
// 
// Inputs are the logits, not probabilities.

*/








func sparseSoftmaxCrossEntropyWithLogits( scope:Scope,features: Output, labels: Output )  ->(loss: Output?, backprop: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseSoftmaxCrossEntropyWithLogits",
        Name: "Type",
        Input: [ features, labels],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Returns the element-wise max of two SparseTensors.

Assumes the two SparseTensors have the same shape, i.e., no broadcasting.

*/








func sparseSparseMaximum( scope:Scope,aIndices: Output, aValues: Output, aShape: Output, bIndices: Output, bValues: Output, bShape: Output )  ->(outputIndices: Output?, outputValues: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseSparseMaximum",
        Name: "Type",
        Input: [ aIndices, aValues, aShape, bIndices, bValues, bShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Returns the element-wise min of two SparseTensors.

Assumes the two SparseTensors have the same shape, i.e., no broadcasting.

*/








func sparseSparseMinimum( scope:Scope,aIndices: Output, aValues: Output, aShape: Output, bIndices: Output, bValues: Output, bShape: Output )  ->(outputIndices: Output?, outputValues: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseSparseMinimum",
        Name: "Type",
        Input: [ aIndices, aValues, aShape, bIndices, bValues, bShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Split a `SparseTensor` into `num_split` tensors along one dimension.

If the `shape[split_dim]` is not an integer multiple of `num_split`. Slices
// `[0 : shape[split_dim] % num_split]` gets one extra dimension.
// For example, if `split_dim = 1` and `num_split = 2` and the input is
// 
//     input_tensor = shape = [2, 7]
//     [    a   d e  ]
//     [b c          ]
// 
// Graphically the output tensors are:
// 
//     output_tensor[0] = shape = [2, 4]
//     [    a  ]
//     [b c    ]
// 
//     output_tensor[1] = shape = [2, 3]
//     [ d e  ]
//     [      ]

*/








func sparseSplit( scope:Scope,splitDim: Output, indices: Output, values: Output, shape: Output, numSplit :UInt8  )  ->(outputIndices: Output?, outputValues: Output?, outputShape: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_split"] = numSplit

    let opspec = OpSpec(
        OpType: "SparseSplit",
        Name: "Type",
        Input: [ splitDim, indices, values, shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Adds up a `SparseTensor` and a dense `Tensor`, producing a dense `Tensor`.

This Op does not require `a_indices` be sorted in standard lexicographic order.

*/








func sparseTensorDenseAdd( scope:Scope,aIndices: Output, aValues: Output, aShape: Output, b: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseTensorDenseAdd",
        Name: "Type",
        Input: [ aIndices, aValues, aShape, b],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Multiply SparseTensor (of rank 2) "A" by dense matrix "B".

No validity checking is performed on the indices of A.  However, the following
// input format is recommended for optimal behavior:
// 
// if adjoint_a == false:
//   A should be sorted in lexicographically increasing order.  Use SparseReorder
//   if you're not sure.
// if adjoint_a == true:
//   A should be sorted in order of increasing dimension 1 (i.e., "column major"
//   order instead of "row major" order).

*/








func sparseTensorDenseMatMul( scope:Scope,aIndices: Output, aValues: Output, aShape: Output, b: Output, adjointA :Bool  , adjointB :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["adjoint_a"] = adjointA
    attrs["adjoint_b"] = adjointB

    let opspec = OpSpec(
        OpType: "SparseTensorDenseMatMul",
        Name: "Type",
        Input: [ aIndices, aValues, aShape, b],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a dataset that splits a SparseTensor into elements row-wise.


*/








func sparseTensorSliceDataset( scope:Scope,indices: Output, values: Output, denseShape: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SparseTensorSliceDataset",
        Name: "Type",
        Input: [ indices, values, denseShape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Converts a sparse representation into a dense tensor.

Builds an array `dense` with shape `output_shape` such that
// 
// ```
// # If sparse_indices is scalar
// dense[i] = (i == sparse_indices ? sparse_values : default_value)
// 
// # If sparse_indices is a vector, then for each i
// dense[sparse_indices[i]] = sparse_values[i]
// 
// # If sparse_indices is an n by d matrix, then for each i in [0, n)
// dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
// ```
// 
// All other values in `dense` are set to `default_value`.  If `sparse_values` is a
// scalar, all sparse indices are set to this single value.
// 
// Indices should be sorted in lexicographic order, and indices must not
// contain any repeats. If `validate_indices` is true, these properties
// are checked during execution.

*/








func sparseToDense( scope:Scope,sparseIndices: Output, outputShape: Output, sparseValues: Output, defaultValue: Output, validateIndices :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["validate_indices"] = validateIndices

    let opspec = OpSpec(
        OpType: "SparseToDense",
        Name: "Type",
        Input: [ sparseIndices, outputShape, sparseValues, defaultValue],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Applies set operation along last dimension of 2 `SparseTensor` inputs.

See SetOperationOp::SetOperationFromContext for values of `set_operation`.
// 
// If `validate_indices` is `True`, `SparseToSparseSetOperation` validates the
// order and range of `set1` and `set2` indices.
// 
// Input `set1` is a `SparseTensor` represented by `set1_indices`, `set1_values`,
// and `set1_shape`. For `set1` ranked `n`, 1st `n-1` dimensions must be the same
// as `set2`. Dimension `n` contains values in a set, duplicates are allowed but
// ignored.
// 
// Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
// and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
// as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
// ignored.
// 
// If `validate_indices` is `True`, this op validates the order and range of `set1`
// and `set2` indices.
// 
// Output `result` is a `SparseTensor` represented by `result_indices`,
// `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
// has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
// dimension contains the result of `set_operation` applied to the corresponding
// `[0...n-1]` dimension of `set`.

*/








func sparseToSparseSetOperation( scope:Scope,set1Indices: Output, set1Values: Output, set1Shape: Output, set2Indices: Output, set2Values: Output, set2Shape: Output, setOperation :String  , validateIndices :Bool  )  ->(resultIndices: Output?, resultValues: Output?, resultShape: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["set_operation"] = setOperation
    attrs["validate_indices"] = validateIndices

    let opspec = OpSpec(
        OpType: "SparseToSparseSetOperation",
        Name: "Type",
        Input: [ set1Indices, set1Values, set1Shape, set2Indices, set2Values, set2Shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Splits a tensor into `num_split` tensors along one dimension.


*/








func split( scope:Scope,splitDim: Output, value: Output, numSplit :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_split"] = numSplit

    let opspec = OpSpec(
        OpType: "Split",
        Name: "Type",
        Input: [ splitDim, value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Splits a tensor into `num_split` tensors along one dimension.


*/








func splitV( scope:Scope,value: Output, sizeSplits: Output, splitDim: Output, numSplit :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_split"] = numSplit

    let opspec = OpSpec(
        OpType: "SplitV",
        Name: "Type",
        Input: [ value, sizeSplits, splitDim],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes square root of x element-wise.

I.e., \\(y = \sqrt{x} = x// ^{1/2}\\).

*/








func sqrt( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Sqrt",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradient for the sqrt of `x` wrt its input.

Specifically, `grad = dy  *  0.5 / y`, where `y = sqrt(x)`, and `dy`
// is the corresponding input gradient.

*/








func sqrtGrad( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SqrtGrad",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes square of x element-wise.

I.e., \\(y = x  *  x = x// ^2\\).

*/








func square( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Square",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns (x - y)(x - y) element-wise.

 * NOTE * : `SquaredDifference` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func squaredDifference( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "SquaredDifference",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Removes dimensions of size 1 from the shape of a tensor.

Given a tensor `input`, this operation returns a tensor of the same type with
// all dimensions of size 1 removed. If you don't want to remove all size 1
// dimensions, you can remove specific size 1 dimensions by specifying
// `squeeze_dims`.
// 
// For example:
// 
// ```
// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
// shape(squeeze(t)) ==> [2, 3]
// ```
// 
// Or, to remove specific size 1 dimensions:
// 
// ```
// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
// shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
// ```

*/








func squeeze( scope:Scope,input: Output, squeezeDims :[Int64]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["squeeze_dims"] = squeezeDims

    let opspec = OpSpec(
        OpType: "Squeeze",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A stack that produces elements in first-in last-out order.


*/








func stack( scope:Scope, stackName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["stack_name"] = stackName

    let opspec = OpSpec(
        OpType: "Stack",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Delete the stack from its resource container.


*/








func stackClose( scope:Scope,handle: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "StackClose",
        Name: "Type",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Pop the element at the top of the stack.


*/








func stackPop( scope:Scope,handle: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "StackPop",
        Name: "Type",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Push an element onto the stack.


*/








func stackPush( scope:Scope,handle: Output, elem: Output, swapMemory :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["swap_memory"] = swapMemory

    let opspec = OpSpec(
        OpType: "StackPush",
        Name: "Type",
        Input: [ handle, elem],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Stage values similar to a lightweight Enqueue.

The basic functionality of this Op is similar to a queue with many
// fewer capabilities and options.  This Op is optimized for performance.

*/








func stage( scope:Scope,values: Output, container :String  , sharedName :String  )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "Stage",
        Name: "Type",
        Input: [ values],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Outputs deterministic pseudorandom values from a normal distribution.

The generated values will have mean 0 and standard deviation 1.
// 
// The outputs are a deterministic function of `shape` and `seed`.

*/








func statelessRandomNormal( scope:Scope,shape: Output, seed: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "StatelessRandomNormal",
        Name: "Type",
        Input: [ shape, seed],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Outputs deterministic pseudorandom random values from a uniform distribution.

The generated values follow a uniform distribution in the range `[0, 1)`. The
// lower bound 0 is included in the range, while the upper bound 1 is excluded.
// 
// The outputs are a deterministic function of `shape` and `seed`.

*/








func statelessRandomUniform( scope:Scope,shape: Output, seed: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "StatelessRandomUniform",
        Name: "Type",
        Input: [ shape, seed],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Outputs deterministic pseudorandom values from a truncated normal distribution.

The generated values follow a normal distribution with mean 0 and standard
// deviation 1, except that values whose magnitude is more than 2 standard
// deviations from the mean are dropped and re-picked.
// 
// The outputs are a deterministic function of `shape` and `seed`.

*/








func statelessTruncatedNormal( scope:Scope,shape: Output, seed: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "StatelessTruncatedNormal",
        Name: "Type",
        Input: [ shape, seed],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Stops gradient computation.

When executed in a graph, this op outputs its input tensor as-is.
// 
// When building ops to compute gradients, this op prevents the contribution of
// its inputs to be taken into account.  Normally, the gradient generator adds ops
// to a graph to compute the derivatives of a specified 'loss' by recursively
// finding out inputs that contributed to its computation.  If you insert this op
// in the graph it inputs are masked from the gradient generator.  They are not
// taken into account for computing gradients.
// 
// This is useful any time you want to compute a value with TensorFlow but need
// to pretend that the value was a constant. Some examples include:
// 
//  *   The  * EM *  algorithm where the  * M-step *  should not involve backpropagation
//    through the output of the  * E-step * .
//  *   Contrastive divergence training of Boltzmann machines where, when
//    differentiating the energy function, the training must not backpropagate
//    through the graph that generated the samples from the model.
//  *   Adversarial training, where no backprop should happen through the adversarial
//    example generation process.

*/








func stopGradient( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "StopGradient",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Return a strided slice from `input`.

Note, most python users will want to use the Python `Tensor.__getitem__`
// or `Variable.__getitem__` rather than this op directly.
// 
// The goal of this op is to produce a new tensor with a subset of
// the elements from the `n` dimensional `input` tensor. The subset is chosen using
// a sequence of `m` sparse range specifications encoded into the arguments
// of this function. Note, in some cases
// `m` could be equal to `n`, but this need not be the case. Each
// range specification entry can be one of the following:
// 
// - An ellipsis (...). Ellipses are used to imply zero or more
//   dimensions of full-dimension selection and are produced using
//   `ellipsis_mask`. For example, `foo[...]` is the identity slice.
// 
// - A new axis. This is used to insert a new shape=1 dimension and is
//   produced using `new_axis_mask`. For example, `foo[:, ...]` where
//   `foo` is shape `(3, 4)` produces a `(1, 3, 4)` tensor.
// 
// 
// - A range `begin:end:stride`. This is used to specify how much to choose from
//   a given dimension. `stride` can be any integer but 0.  `begin` is an integer
//   which represents the index of the first value to select while `end` represents
//   the index of the last value to select. The number of values selected in each
//   dimension is `end - begin` if `stride > 0` and `begin - end` if `stride < 0`.
//   `begin` and `end` can be negative where `-1` is the last element, `-2` is
//   the second to last. `begin_mask` controls whether to replace the explicitly
//   given `begin` with an implicit effective value of `0` if `stride > 0` and
//   `-1` if `stride < 0`. `end_mask` is analogous but produces the number
//   required to create the largest open interval. For example, given a shape
//   `(3,)` tensor `foo[:]`, the effective `begin` and `end` are `0` and `3`. Do
//   not assume this is equivalent to `foo[0:-1]` which has an effective `begin`
//   and `end` of `0` and `2`. Another example is `foo[-2::-1]` which reverses the
//   first dimension of a tensor while dropping the last two (in the original
//   order elements). For example `foo = [1,2,3,4]; foo[-2::-1]` is `[4,3]`.
// 
// - A single index. This is used to keep only elements that have a given
//   index. For example (`foo[2, :]` on a shape `(5,6)` tensor produces a
//   shape `(6,)` tensor. This is encoded in `begin` and `end` and
//   `shrink_axis_mask`.
// 
// Each conceptual range specification is encoded in the op's argument. This
// encoding is best understand by considering a non-trivial example. In
// particular,
// `foo[1, 2:4, None, ..., :-3:-1, :]` will be encoded as
// 
// ```
// begin = [1, 2, x, x, 0, x] # x denotes don't care (usually 0)
// end = [2, 4, x, x, -3, x]
// strides = [1, 1, x, x, -1, 1]
// begin_mask = 1<<4 | 1 << 5 = 48
// end_mask = 1<<5 = 32
// ellipsis_mask = 1<<3 = 8
// new_axis_mask = 1<<2 4
// shrink_axis_mask = 1<<0
// ```
// 
// In this case if `foo.shape` is (5, 5, 5, 5, 5, 5) the final shape of
// the slice becomes (2, 1, 5, 5, 2, 5).
// Let us walk step by step through each argument specification.
// 
// 1.  The first argument in the example slice is turned into `begin = 1` and
// `end = begin + 1 = 2`. To disambiguate from the original spec `2:4` we
// also set the appropriate bit in `shrink_axis_mask`.
// 
// 2. `2:4` is contributes 2, 4, 1 to begin, end, and stride. All masks have
// zero bits contributed.
// 
// 3. None is a synonym for `tf.newaxis`. This means insert a dimension of size 1
// dimension in the final shape. Dummy values are contributed to begin,
// end and stride, while the new_axis_mask bit is set.
// 
// 4. `...` grab the full ranges from as many dimensions as needed to
// fully specify a slice for every dimension of the input shape.
// 
// 5. `:-3:-1` shows the use of negative indices. A negative index `i` associated
// with a dimension that has shape `s` is converted to a positive index
// `s + i`. So `-1` becomes `s-1` (i.e. the last element). This conversion
// is done internally so begin, end and strides receive x, -3, and -1.
// The appropriate begin_mask bit is set to indicate the start range is the
// full range (ignoring the x).
// 
// 6. `:` indicates that the entire contents of the corresponding dimension
// is selected. This is equivalent to `::` or `0::1`. begin, end, and strides
// receive 0, 0, and 1, respectively. The appropriate bits in `begin_mask` and
// `end_mask` are also set.
// 
//  * Requirements * :
//   `0 != strides[i] for i in [0, m)`
//   `ellipsis_mask must be a power of two (only one ellipsis)`

*/








func stridedSlice( scope:Scope,input: Output, begin: Output, end: Output, strides: Output, beginMask :UInt8  , endMask :UInt8  , ellipsisMask :UInt8  , newAxisMask :UInt8  , shrinkAxisMask :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["begin_mask"] = beginMask
    attrs["end_mask"] = endMask
    attrs["ellipsis_mask"] = ellipsisMask
    attrs["new_axis_mask"] = newAxisMask
    attrs["shrink_axis_mask"] = shrinkAxisMask

    let opspec = OpSpec(
        OpType: "StridedSlice",
        Name: "Type",
        Input: [ input, begin, end, strides],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Assign `value` to the sliced l-value reference of `ref`.

The values of `value` are assigned to the positions in the variable
// `ref` that are selected by the slice parameters. The slice parameters
// `begin, `end`, `strides`, etc. work exactly as in `StridedSlice`.
// 
// NOTE this op currently does not support broadcasting and so `value`'s
// shape must be exactly the shape produced by the slice of `ref`.

*/








func stridedSliceAssign( scope:Scope,ref: Output, begin: Output, end: Output, strides: Output, value: Output, beginMask :UInt8  , endMask :UInt8  , ellipsisMask :UInt8  , newAxisMask :UInt8  , shrinkAxisMask :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["begin_mask"] = beginMask
    attrs["end_mask"] = endMask
    attrs["ellipsis_mask"] = ellipsisMask
    attrs["new_axis_mask"] = newAxisMask
    attrs["shrink_axis_mask"] = shrinkAxisMask

    let opspec = OpSpec(
        OpType: "StridedSliceAssign",
        Name: "Type",
        Input: [ ref, begin, end, strides, value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the gradient of `StridedSlice`.

Since `StridedSlice` cuts out pieces of its `input` which is size
// `shape`, its gradient will have the same shape (which is passed here
// as `shape`). The gradient will be zero in any element that the slice
// does not select.
// 
// Arguments are the same as StridedSliceGrad with the exception that
// `dy` is the input gradient to be propagated and `shape` is the
// shape of `StridedSlice`'s `input`.

*/








func stridedSliceGrad( scope:Scope,shape: Output, begin: Output, end: Output, strides: Output, dy: Output, beginMask :UInt8  , endMask :UInt8  , ellipsisMask :UInt8  , newAxisMask :UInt8  , shrinkAxisMask :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["begin_mask"] = beginMask
    attrs["end_mask"] = endMask
    attrs["ellipsis_mask"] = ellipsisMask
    attrs["new_axis_mask"] = newAxisMask
    attrs["shrink_axis_mask"] = shrinkAxisMask

    let opspec = OpSpec(
        OpType: "StridedSliceGrad",
        Name: "Type",
        Input: [ shape, begin, end, strides, dy],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Joins the strings in the given list of string tensors into one tensor;

with the given separator (default is an empty separator).

*/








func stringJoin( scope:Scope,inputs: Output, n :UInt8  , separator :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["N"] = n
    attrs["separator"] = separator

    let opspec = OpSpec(
        OpType: "StringJoin",
        Name: "Type",
        Input: [ inputs],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Split elements of `input` based on `delimiter` into a `SparseTensor`.

Let N be the size of source (typically N will be the batch size). Split each
// element of `input` based on `delimiter` and return a `SparseTensor`
// containing the splitted tokens. Empty tokens are ignored.
// 
// `delimiter` can be empty, or a string of split characters. If `delimiter` is an
//  empty string, each element of `input` is split into individual single-byte
//  character strings, including splitting of UTF-8 multibyte sequences. Otherwise
//  every character of `delimiter` is a potential split point.
// 
// For example:
//   N = 2, input[0] is 'hello world' and input[1] is 'a b c', then the output
//   will be
// 
//   indices = [0, 0;
//              0, 1;
//              1, 0;
//              1, 1;
//              1, 2]
//   shape = [2, 3]
//   values = ['hello', 'world', 'a', 'b', 'c']

*/








func stringSplit( scope:Scope,input: Output, delimiter: Output )  ->(indices: Output?, values: Output?, shape: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "StringSplit",
        Name: "Type",
        Input: [ input, delimiter],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Converts each string in the input Tensor to its hash mod by a number of buckets.

The hash function is deterministic on the content of the string within the
// process.
// 
// Note that the hash function may change from time to time.
// This functionality will be deprecated and it's recommended to use
// `tf.string_to_hash_bucket_fast()` or `tf.string_to_hash_bucket_strong()`.

*/








func stringToHashBucket( scope:Scope,stringTensor: Output, numBuckets :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_buckets"] = numBuckets

    let opspec = OpSpec(
        OpType: "StringToHashBucket",
        Name: "Type",
        Input: [ stringTensor],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Converts each string in the input Tensor to its hash mod by a number of buckets.

The hash function is deterministic on the content of the string within the
// process and will never change. However, it is not suitable for cryptography.
// This function may be used when CPU time is scarce and inputs are trusted or
// unimportant. There is a risk of adversaries constructing inputs that all hash
// to the same bucket. To prevent this problem, use a strong hash function with
// `tf.string_to_hash_bucket_strong`.

*/








func stringToHashBucketFast( scope:Scope,input: Output, numBuckets :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_buckets"] = numBuckets

    let opspec = OpSpec(
        OpType: "StringToHashBucketFast",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Converts each string in the input Tensor to its hash mod by a number of buckets.

The hash function is deterministic on the content of the string within the
// process. The hash function is a keyed hash function, where attribute `key`
// defines the key of the hash function. `key` is an array of 2 elements.
// 
// A strong hash is important when inputs may be malicious, e.g. URLs with
// additional components. Adversaries could try to make their inputs hash to the
// same bucket for a denial-of-service attack or to skew the results. A strong
// hash prevents this by making it difficult, if not infeasible, to compute inputs
// that hash to the same bucket. This comes at a cost of roughly 4x higher compute
// time than `tf.string_to_hash_bucket_fast`.

*/








func stringToHashBucketStrong( scope:Scope,input: Output, numBuckets :UInt8  , key :[Int64]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_buckets"] = numBuckets
    attrs["key"] = key

    let opspec = OpSpec(
        OpType: "StringToHashBucketStrong",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Converts each string in the input Tensor to the specified numeric type.

(Note that int32 overflow results in an error while float overflow
// results in a rounded value.)

*/








func stringToNumber( scope:Scope,stringTensor: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "StringToNumber",
        Name: "Type",
        Input: [ stringTensor],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns x - y element-wise.

 * NOTE * : `Sub` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func sub( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Sub",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Return substrings from `Tensor` of strings.

For each string in the input `Tensor`, creates a substring starting at index
// `pos` with a total length of `len`.
// 
// If `len` defines a substring that would extend beyond the length of the input
// string, then as many characters as possible are used.
// 
// If `pos` is negative or specifies a character index larger than any of the input
// strings, then an `InvalidArgumentError` is thrown.
// 
// `pos` and `len` must have the same shape, otherwise a `ValueError` is thrown on
// Op creation.
// 
//  * NOTE * : `Substr` supports broadcasting up to two dimensions. More about
// broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
// 
// ---
// 
// Examples
// 
// Using scalar `pos` and `len`:
// 
// ```python
// input = [b'Hello', b'World']
// position = 1
// length = 3
// 
// output = [b'ell', b'orl']
// ```
// 
// Using `pos` and `len` with same shape as `input`:
// 
// ```python
// input = [[b'ten', b'eleven', b'twelve'],
//          [b'thirteen', b'fourteen', b'fifteen'],
//          [b'sixteen', b'seventeen', b'eighteen']]
// position = [[1, 2, 3],
//             [1, 2, 3],
//             [1, 2, 3]]
// length =   [[2, 3, 4],
//             [4, 3, 2],
//             [5, 5, 5]]
// 
// output = [[b'en', b'eve', b'lve'],
//           [b'hirt', b'urt', b'te'],
//           [b'ixtee', b'vente', b'hteen']]
// ```
// 
// Broadcasting `pos` and `len` onto `input`:
// 
// ```
// input = [[b'ten', b'eleven', b'twelve'],
//          [b'thirteen', b'fourteen', b'fifteen'],
//          [b'sixteen', b'seventeen', b'eighteen'],
//          [b'nineteen', b'twenty', b'twentyone']]
// position = [1, 2, 3]
// length =   [1, 2, 3]
// 
// output = [[b'e', b'ev', b'lve'],
//           [b'h', b'ur', b'tee'],
//           [b'i', b've', b'hte'],
//           [b'i', b'en', b'nty']]
// ```
// 
// Broadcasting `input` onto `pos` and `len`:
// 
// ```
// input = b'thirteen'
// position = [1, 5, 7]
// length =   [3, 2, 1]
// 
// output = [b'hir', b'ee', b'n"]
// ```

*/








func substr( scope:Scope,input: Output, pos: Output, len: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Substr",
        Name: "Type",
        Input: [ input, pos, len],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the sum of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.

*/








func sum( scope:Scope,input: Output, reductionIndices: Output, keepDims :Bool  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["keep_dims"] = keepDims

    let opspec = OpSpec(
        OpType: "Sum",
        Name: "Type",
        Input: [ input, reductionIndices],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the singular value decompositions of one or more matrices.

Computes the SVD of each inner matrix in `input` such that
// `input[..., :, :] = u[..., :, :]  *  diag(s[..., :, :])  *  transpose(v[..., :, :])`
// 
// ```python
// # a is a tensor containing a batch of matrices.
// # s is a tensor of singular values for each matrix.
// # u is the tensor containing of left singular vectors for each matrix.
// # v is the tensor containing of right singular vectors for each matrix.
// s, u, v = svd(a)
// s, _, _ = svd(a, compute_uv=False)
// ```

*/








func svd( scope:Scope,input: Output, computeUv :Bool  , fullMatrices :Bool  )  ->(s: Output?, u: Output?, v: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["compute_uv"] = computeUv
    attrs["full_matrices"] = fullMatrices

    let opspec = OpSpec(
        OpType: "Svd",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Forwards `data` to the output port determined by `pred`.

If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
// the data goes to `output_false`.
// 
// See also `RefSwitch` and `Merge`.

*/








func switch_p( scope:Scope,data: Output, pred: Output )  ->(outputFalse: Output?, outputTrue: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "switch_p",
        Name: "Type",
        Input: [ data, pred],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Computes the gradient function for function f via backpropagation.


*/








func symbolicGradient( scope:Scope,input: Output, f :TensorflowNameAttrList  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["f"] = f

    let opspec = OpSpec(
        OpType: "SymbolicGradient",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a dataset that emits the records from one or more TFRecord files.


*/








func tfRecordDataset( scope:Scope,filenames: Output, compressionType: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TFRecordDataset",
        Name: "Type",
        Input: [ filenames, compressionType],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A Reader that outputs the records from a TensorFlow Records file.


*/








func tfRecordReader( scope:Scope, container :String  , sharedName :String  , compressionType :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName
    attrs["compression_type"] = compressionType

    let opspec = OpSpec(
        OpType: "TFRecordReader",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A Reader that outputs the records from a TensorFlow Records file.


*/








func tfRecordReaderV2( scope:Scope, container :String  , sharedName :String  , compressionType :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName
    attrs["compression_type"] = compressionType

    let opspec = OpSpec(
        OpType: "TFRecordReaderV2",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a dataset that contains `count` elements from the `input_dataset`.


*/








func takeDataset( scope:Scope,inputDataset: Output, count: Output, outputShapes :[Shape]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["output_shapes"] = outputShapes

    let opspec = OpSpec(
        OpType: "TakeDataset",
        Name: "Type",
        Input: [ inputDataset, count],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Read `SparseTensors` from a `SparseTensorsMap` and concatenate them.

The input `sparse_handles` must be an `int64` matrix of shape `[N, 1]` where
// `N` is the minibatch size and the rows correspond to the output handles of
// `AddSparseToTensorsMap` or `AddManySparseToTensorsMap`.  The ranks of the
// original `SparseTensor` objects that went into the given input ops must all
// match.  When the final `SparseTensor` is created, it has rank one
// higher than the ranks of the incoming `SparseTensor` objects
// (they have been concatenated along a new row dimension on the left).
// 
// The output `SparseTensor` object's shape values for all dimensions but the
// first are the max across the input `SparseTensor` objects' shape values
// for the corresponding dimensions.  Its first shape value is `N`, the minibatch
// size.
// 
// The input `SparseTensor` objects' indices are assumed ordered in
// standard lexicographic order.  If this is not the case, after this
// step run `SparseReorder` to restore index ordering.
// 
// For example, if the handles represent an input, which is a `[2, 3]` matrix
// representing two original `SparseTensor` objects:
// 
// ```
//     index = [ 0]
//             [10]
//             [20]
//     values = [1, 2, 3]
//     shape = [50]
// ```
// 
// and
// 
// ```
//     index = [ 2]
//             [10]
//     values = [4, 5]
//     shape = [30]
// ```
// 
// then the final `SparseTensor` will be:
// 
// ```
//     index = [0  0]
//             [0 10]
//             [0 20]
//             [1  2]
//             [1 10]
//     values = [1, 2, 3, 4, 5]
//     shape = [2 50]
// ```

*/








func takeManySparseFromTensorsMap( scope:Scope,sparseHandles: Output, container :String  , sharedName :String  )  ->(sparseIndices: Output?, sparseValues: Output?, sparseShape: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "TakeManySparseFromTensorsMap",
        Name: "Type",
        Input: [ sparseHandles],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Computes tan of x element-wise.


*/








func tan( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Tan",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes hyperbolic tangent of `x` element-wise.


*/








func tanh( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Tanh",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the gradient for the tanh of `x` wrt its input.

Specifically, `grad = dy  *  (1 - y * y)`, where `y = tanh(x)`, and `dy`
// is the corresponding input gradient.

*/








func tanhGrad( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TanhGrad",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns a tensor that may be mutated, but only persists within a single step.

This is an experimental op for internal use only and it is possible to use this
// op in unsafe ways.  DO NOT USE unless you fully understand the risks.
// 
// It is the caller's responsibility to ensure that 'ref' is eventually passed to a
// matching 'DestroyTemporaryVariable' op after all other uses have completed.
// 
// Outputs a ref to the tensor state so it may be read or modified.
// 
//   E.g.
//       var = state_ops._temporary_variable([1, 2], types.float_)
//       var_name = var.op.name
//       var = state_ops.assign(var, [[4.0, 5.0]])
//       var = state_ops.assign_add(var, [[6.0, 7.0]])
//       final = state_ops._destroy_temporary_variable(var, var_name=var_name)

*/








func temporaryVariable( scope:Scope, shape :Shape  , varName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shape"] = shape
    attrs["var_name"] = varName

    let opspec = OpSpec(
        OpType: "TemporaryVariable",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func tensorArray( scope:Scope,size: Output, dynamicSize :Bool  , clearAfterRead :Bool  , tensorArrayName :String  , elementShape :Shape  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["dynamic_size"] = dynamicSize
    attrs["clear_after_read"] = clearAfterRead
    attrs["tensor_array_name"] = tensorArrayName
    attrs["element_shape"] = elementShape

    let opspec = OpSpec(
        OpType: "TensorArray",
        Name: "Type",
        Input: [ size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func tensorArrayClose( scope:Scope,handle: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArrayClose",
        Name: "Type",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Deprecated. Use TensorArrayCloseV3


*/








func tensorArrayCloseV2( scope:Scope,handle: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArrayCloseV2",
        Name: "Type",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Delete the TensorArray from its resource container.

This enables the user to close and release the resource in the middle
// of a step/run.

*/








func tensorArrayCloseV3( scope:Scope,handle: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArrayCloseV3",
        Name: "Type",
        Input: [ handle],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*


*/








func tensorArrayConcat( scope:Scope,handle: Output, flowIn: Output, elementShapeExcept0 :Shape  )  ->(value: Output?, lengths: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["element_shape_except0"] = elementShapeExcept0

    let opspec = OpSpec(
        OpType: "TensorArrayConcat",
        Name: "Type",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Deprecated. Use TensorArrayConcatV3


*/








func tensorArrayConcatV2( scope:Scope,handle: Output, flowIn: Output, elementShapeExcept0 :Shape  )  ->(value: Output?, lengths: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["element_shape_except0"] = elementShapeExcept0

    let opspec = OpSpec(
        OpType: "TensorArrayConcatV2",
        Name: "Type",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Concat the elements from the TensorArray into value `value`.

Takes `T` elements of shapes
// 
//   ```
//   (n0 x d0 x d1 x ...), (n1 x d0 x d1 x ...), ..., (n(T-1) x d0 x d1 x ...)
//   ```
// 
// and concatenates them into a Tensor of shape:
// 
//   ```(n0 + n1 + ... + n(T-1) x d0 x d1 x ...)```
// 
// All elements must have the same shape (excepting the first dimension).

*/








func tensorArrayConcatV3( scope:Scope,handle: Output, flowIn: Output, elementShapeExcept0 :Shape  )  ->(value: Output?, lengths: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["element_shape_except0"] = elementShapeExcept0

    let opspec = OpSpec(
        OpType: "TensorArrayConcatV3",
        Name: "Type",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*


*/








func tensorArrayGather( scope:Scope,handle: Output, indices: Output, flowIn: Output, elementShape :Shape  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["element_shape"] = elementShape

    let opspec = OpSpec(
        OpType: "TensorArrayGather",
        Name: "Type",
        Input: [ handle, indices, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Deprecated. Use TensorArrayGatherV3


*/








func tensorArrayGatherV2( scope:Scope,handle: Output, indices: Output, flowIn: Output, elementShape :Shape  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["element_shape"] = elementShape

    let opspec = OpSpec(
        OpType: "TensorArrayGatherV2",
        Name: "Type",
        Input: [ handle, indices, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Gather specific elements from the TensorArray into output `value`.

All elements selected by `indices` must have the same shape.

*/








func tensorArrayGatherV3( scope:Scope,handle: Output, indices: Output, flowIn: Output, elementShape :Shape  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["element_shape"] = elementShape

    let opspec = OpSpec(
        OpType: "TensorArrayGatherV3",
        Name: "Type",
        Input: [ handle, indices, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func tensorArrayGrad( scope:Scope,handle: Output, flowIn: Output, source :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["source"] = source

    let opspec = OpSpec(
        OpType: "TensorArrayGrad",
        Name: "Type",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Deprecated. Use TensorArrayGradV3


*/








func tensorArrayGradV2( scope:Scope,handle: Output, flowIn: Output, source :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["source"] = source

    let opspec = OpSpec(
        OpType: "TensorArrayGradV2",
        Name: "Type",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a TensorArray for storing the gradients of values in the given handle.

If the given TensorArray gradient already exists, returns a reference to it.
// 
// Locks the size of the original TensorArray by disabling its dynamic size flag.
// 
//  *  * A note about the input flow_in: *  * 
// 
// The handle flow_in forces the execution of the gradient lookup to occur
// only after certain other operations have occurred.  For example, when
// the forward TensorArray is dynamically sized, writes to this TensorArray
// may resize the object.  The gradient TensorArray is statically sized based
// on the size of the forward TensorArray when this operation executes.
// Furthermore, the size of the forward TensorArray is frozen by this call.
// As a result, the flow is used to ensure that the call to generate the gradient
// TensorArray only happens after all writes are executed.
// 
// In the case of dynamically sized TensorArrays, gradient computation should
// only be performed on read operations that have themselves been chained via
// flow to occur only after all writes have executed. That way the final size
// of the forward TensorArray is known when this operation is called.
// 
//  *  * A note about the source attribute: *  * 
// 
// TensorArray gradient calls use an accumulator TensorArray object.  If
// multiple gradients are calculated and run in the same session, the multiple
// gradient nodes may accidentally flow throuth the same accumulator TensorArray.
// This double counts and generally breaks the TensorArray gradient flow.
// 
// The solution is to identify which gradient call this particular
// TensorArray gradient is being called in.  This is performed by identifying
// a unique string (e.g. "gradients", "gradients_1", ...) from the input
// gradient Tensor's name.  This string is used as a suffix when creating
// the TensorArray gradient object here (the attribute `source`).
// 
// The attribute `source` is added as a suffix to the forward TensorArray's
// name when performing the creation / lookup, so that each separate gradient
// calculation gets its own TensorArray accumulator.

*/








func tensorArrayGradV3( scope:Scope,handle: Output, flowIn: Output, source :String  )  ->(gradHandle: Output?, flowOut: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["source"] = source

    let opspec = OpSpec(
        OpType: "TensorArrayGradV3",
        Name: "Type",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*


*/








func tensorArrayPack( scope:Scope,handle: Output, flowIn: Output, elementShape :Shape  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["element_shape"] = elementShape

    let opspec = OpSpec(
        OpType: "TensorArrayPack",
        Name: "Type",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func tensorArrayRead( scope:Scope,handle: Output, index: Output, flowIn: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArrayRead",
        Name: "Type",
        Input: [ handle, index, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Deprecated. Use TensorArrayReadV3


*/








func tensorArrayReadV2( scope:Scope,handle: Output, index: Output, flowIn: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArrayReadV2",
        Name: "Type",
        Input: [ handle, index, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Read an element from the TensorArray into output `value`.


*/








func tensorArrayReadV3( scope:Scope,handle: Output, index: Output, flowIn: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArrayReadV3",
        Name: "Type",
        Input: [ handle, index, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func tensorArrayScatter( scope:Scope,handle: Output, indices: Output, value: Output, flowIn: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArrayScatter",
        Name: "Type",
        Input: [ handle, indices, value, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Deprecated. Use TensorArrayScatterV3


*/








func tensorArrayScatterV2( scope:Scope,handle: Output, indices: Output, value: Output, flowIn: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArrayScatterV2",
        Name: "Type",
        Input: [ handle, indices, value, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Scatter the data from the input value into specific TensorArray elements.

`indices` must be a vector, its length must match the first dim of `value`.

*/








func tensorArrayScatterV3( scope:Scope,handle: Output, indices: Output, value: Output, flowIn: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArrayScatterV3",
        Name: "Type",
        Input: [ handle, indices, value, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func tensorArraySize( scope:Scope,handle: Output, flowIn: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArraySize",
        Name: "Type",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Deprecated. Use TensorArraySizeV3


*/








func tensorArraySizeV2( scope:Scope,handle: Output, flowIn: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArraySizeV2",
        Name: "Type",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Get the current size of the TensorArray.


*/








func tensorArraySizeV3( scope:Scope,handle: Output, flowIn: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArraySizeV3",
        Name: "Type",
        Input: [ handle, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func tensorArraySplit( scope:Scope,handle: Output, value: Output, lengths: Output, flowIn: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArraySplit",
        Name: "Type",
        Input: [ handle, value, lengths, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Deprecated. Use TensorArraySplitV3


*/








func tensorArraySplitV2( scope:Scope,handle: Output, value: Output, lengths: Output, flowIn: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArraySplitV2",
        Name: "Type",
        Input: [ handle, value, lengths, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Split the data from the input value into TensorArray elements.

Assuming that `lengths` takes on values
// 
//   ```(n0, n1, ..., n(T-1))```
// 
// and that `value` has shape
// 
//   ```(n0 + n1 + ... + n(T-1) x d0 x d1 x ...)```,
// 
// this splits values into a TensorArray with T tensors.
// 
// TensorArray index t will be the subtensor of values with starting position
// 
//   ```(n0 + n1 + ... + n(t-1), 0, 0, ...)```
// 
// and having size
// 
//   ```nt x d0 x d1 x ...```

*/








func tensorArraySplitV3( scope:Scope,handle: Output, value: Output, lengths: Output, flowIn: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArraySplitV3",
        Name: "Type",
        Input: [ handle, value, lengths, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*


*/








func tensorArrayUnpack( scope:Scope,handle: Output, value: Output, flowIn: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArrayUnpack",
        Name: "Type",
        Input: [ handle, value, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Deprecated. Use TensorArrayV3


*/








func tensorArrayV2( scope:Scope,size: Output, elementShape :Shape  , dynamicSize :Bool  , clearAfterRead :Bool  , tensorArrayName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["element_shape"] = elementShape
    attrs["dynamic_size"] = dynamicSize
    attrs["clear_after_read"] = clearAfterRead
    attrs["tensor_array_name"] = tensorArrayName

    let opspec = OpSpec(
        OpType: "TensorArrayV2",
        Name: "Type",
        Input: [ size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
An array of Tensors of given size.

Write data via Write and read via Read or Pack.

*/








func tensorArrayV3( scope:Scope,size: Output, elementShape :Shape  , dynamicSize :Bool  , clearAfterRead :Bool  , tensorArrayName :String  )  ->(handle: Output?, flow: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["element_shape"] = elementShape
    attrs["dynamic_size"] = dynamicSize
    attrs["clear_after_read"] = clearAfterRead
    attrs["tensor_array_name"] = tensorArrayName

    let opspec = OpSpec(
        OpType: "TensorArrayV3",
        Name: "Type",
        Input: [ size],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*


*/








func tensorArrayWrite( scope:Scope,handle: Output, index: Output, value: Output, flowIn: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArrayWrite",
        Name: "Type",
        Input: [ handle, index, value, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Deprecated. Use TensorArrayGradV3


*/








func tensorArrayWriteV2( scope:Scope,handle: Output, index: Output, value: Output, flowIn: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArrayWriteV2",
        Name: "Type",
        Input: [ handle, index, value, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Push an element onto the tensor_array.


*/








func tensorArrayWriteV3( scope:Scope,handle: Output, index: Output, value: Output, flowIn: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TensorArrayWriteV3",
        Name: "Type",
        Input: [ handle, index, value, flowIn],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a dataset that emits `components` as a tuple of tensors once.


*/








func tensorDataset( scope:Scope,components: Output, outputShapes :[Shape]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["output_shapes"] = outputShapes

    let opspec = OpSpec(
        OpType: "TensorDataset",
        Name: "Type",
        Input: [ components],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a dataset that emits each dim-0 slice of `components` once.


*/








func tensorSliceDataset( scope:Scope,components: Output, outputShapes :[Shape]  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["output_shapes"] = outputShapes

    let opspec = OpSpec(
        OpType: "TensorSliceDataset",
        Name: "Type",
        Input: [ components],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Outputs a `Summary` protocol buffer with a tensor.


*/








func tensorSummary( scope:Scope,tensor: Output, description :String  , labels :[Data]  , displayName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["description"] = description
    attrs["labels"] = labels
    attrs["display_name"] = displayName

    let opspec = OpSpec(
        OpType: "TensorSummary",
        Name: "Type",
        Input: [ tensor],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a dataset that emits the lines of one or more text files.


*/








func textLineDataset( scope:Scope,filenames: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TextLineDataset",
        Name: "Type",
        Input: [ filenames],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A Reader that outputs the lines of a file delimited by '\n'.


*/








func textLineReader( scope:Scope, skipHeaderLines :UInt8  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["skip_header_lines"] = skipHeaderLines
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "TextLineReader",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A Reader that outputs the lines of a file delimited by '\n'.


*/








func textLineReaderV2( scope:Scope, skipHeaderLines :UInt8  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["skip_header_lines"] = skipHeaderLines
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "TextLineReaderV2",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Generates labels for candidate sampling with a learned unigram distribution.

See explanations of candidate sampling and the data formats at
// go/candidate-sampling.
// 
// For each batch, this op picks a single set of sampled candidate labels.
// 
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.

*/








func threadUnsafeUnigramCandidateSampler( scope:Scope,trueClasses: Output, numTrue :UInt8  , numSampled :UInt8  , unique :Bool  , rangeMax :UInt8  , seed :UInt8  , seed2 :UInt8  )  ->(sampledCandidates: Output?, trueExpectedCount: Output?, sampledExpectedCount: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_true"] = numTrue
    attrs["num_sampled"] = numSampled
    attrs["unique"] = unique
    attrs["range_max"] = rangeMax
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "ThreadUnsafeUnigramCandidateSampler",
        Name: "Type",
        Input: [ trueClasses],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Constructs a tensor by tiling a given tensor.

This operation creates a new tensor by replicating `input` `multiples` times.
// The output tensor's i'th dimension has `input.dims(i)  *  multiples[i]` elements,
// and the values of `input` are replicated `multiples[i]` times along the 'i'th
// dimension. For example, tiling `[a b c d]` by `[2]` produces
// `[a b c d a b c d]`.

*/








func tile( scope:Scope,input: Output, multiples: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Tile",
        Name: "Type",
        Input: [ input, multiples],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns the gradient of `Tile`.

Since `Tile` takes an input and repeats the input `multiples` times
// along each dimension, `TileGrad` takes in `multiples` and aggregates
// each repeated tile of `input` into `output`.

*/








func tileGrad( scope:Scope,input: Output, multiples: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TileGrad",
        Name: "Type",
        Input: [ input, multiples],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Finds values and indices of the `k` largest elements for the last dimension.

If the input is a vector (rank-1), finds the `k` largest entries in the vector
// and outputs their values and indices as vectors.  Thus `values[j]` is the
// `j`-th largest entry in `input`, and its index is `indices[j]`.
// 
// For matrices (resp. higher rank input), computes the top `k` entries in each
// row (resp. vector along the last dimension).  Thus,
// 
//     values.shape = indices.shape = input.shape[:-1] + [k]
// 
// If two elements are equal, the lower-index element appears first.
// 
// If `k` varies dynamically, use `TopKV2` below.

*/








func topK( scope:Scope,input: Output, k :UInt8  , sorted :Bool  )  ->(values: Output?, indices: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["k"] = k
    attrs["sorted"] = sorted

    let opspec = OpSpec(
        OpType: "TopK",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Finds values and indices of the `k` largest elements for the last dimension.

If the input is a vector (rank-1), finds the `k` largest entries in the vector
// and outputs their values and indices as vectors.  Thus `values[j]` is the
// `j`-th largest entry in `input`, and its index is `indices[j]`.
// 
// For matrices (resp. higher rank input), computes the top `k` entries in each
// row (resp. vector along the last dimension).  Thus,
// 
//     values.shape = indices.shape = input.shape[:-1] + [k]
// 
// If two elements are equal, the lower-index element appears first.

*/








func topKV2( scope:Scope,input: Output, k: Output, sorted :Bool  )  ->(values: Output?, indices: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["sorted"] = sorted

    let opspec = OpSpec(
        OpType: "TopKV2",
        Name: "Type",
        Input: [ input, k],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Shuffle dimensions of x according to a permutation.

The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
//   `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`

*/








func transpose( scope:Scope,x: Output, perm: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Transpose",
        Name: "Type",
        Input: [ x, perm],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns x / y element-wise for integer types.

Truncation designates that negative numbers will round fractional quantities
// toward zero. I.e. -7 / 5 = 1. This matches C semantics but it is different
// than Python semantics. See `FloorDiv` for a division function that matches
// Python Semantics.
// 
//  * NOTE * : `TruncateDiv` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func truncateDiv( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TruncateDiv",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns element-wise remainder of division. This emulates C semantics in that

the result here is consistent with a truncating divide. E.g. `truncate(x / y)  * 
// y + truncate_mod(x, y) = x`.
// 
//  * NOTE * : `TruncateMod` supports broadcasting. More about broadcasting
// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

*/








func truncateMod( scope:Scope,x: Output, y: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "TruncateMod",
        Name: "Type",
        Input: [ x, y],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Outputs random values from a truncated normal distribution.

The generated values follow a normal distribution with mean 0 and standard
// deviation 1, except that values whose magnitude is more than 2 standard
// deviations from the mean are dropped and re-picked.

*/








func truncatedNormal( scope:Scope,shape: Output, seed :UInt8  , seed2 :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "TruncatedNormal",
        Name: "Type",
        Input: [ shape],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Generates labels for candidate sampling with a uniform distribution.

See explanations of candidate sampling and the data formats at
// go/candidate-sampling.
// 
// For each batch, this op picks a single set of sampled candidate labels.
// 
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.

*/








func uniformCandidateSampler( scope:Scope,trueClasses: Output, numTrue :UInt8  , numSampled :UInt8  , unique :Bool  , rangeMax :UInt8  , seed :UInt8  , seed2 :UInt8  )  ->(sampledCandidates: Output?, trueExpectedCount: Output?, sampledExpectedCount: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num_true"] = numTrue
    attrs["num_sampled"] = numSampled
    attrs["unique"] = unique
    attrs["range_max"] = rangeMax
    attrs["seed"] = seed
    attrs["seed2"] = seed2

    let opspec = OpSpec(
        OpType: "UniformCandidateSampler",
        Name: "Type",
        Input: [ trueClasses],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Finds unique elements in a 1-D tensor.

This operation returns a tensor `y` containing all of the unique elements of `x`
// sorted in the same order that they occur in `x`. This operation also returns a
// tensor `idx` the same size as `x` that contains the index of each value of `x`
// in the unique output `y`. In other words:
// 
// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
// 
// For example:
// 
// ```
// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
// y, idx = unique(x)
// y ==> [1, 2, 4, 7, 8]
// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
// ```

*/








func unique( scope:Scope,x: Output )  ->(y: Output?, idx: Output?){
    if scope.error.error != nil {
        return (nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Unique",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1))
}

/*
Finds unique elements in a 1-D tensor.

This operation returns a tensor `y` containing all of the unique elements of `x`
// sorted in the same order that they occur in `x`. This operation also returns a
// tensor `idx` the same size as `x` that contains the index of each value of `x`
// in the unique output `y`. Finally, it returns a third tensor `count` that
// contains the count of each element of `y` in `x`. In other words:
// 
// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
// 
// For example:
// 
// ```
// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
// y, idx, count = unique_with_counts(x)
// y ==> [1, 2, 4, 7, 8]
// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
// count ==> [2, 1, 3, 1, 2]
// ```

*/








func uniqueWithCounts( scope:Scope,x: Output )  ->(y: Output?, idx: Output?, count: Output?){
    if scope.error.error != nil {
        return (nil, nil, nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "UniqueWithCounts",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1),op?.output(2 - 1),op?.output(3 - 1))
}

/*
Unpacks a given dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors.

Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
// For example, given a tensor of shape `(A, B, C, D)`;
// 
// If `axis == 0` then the i'th tensor in `output` is the slice `value[i, :, :, :]`
//   and each tensor in `output` will have shape `(B, C, D)`. (Note that the
//   dimension unpacked along is gone, unlike `split`).
// 
// If `axis == 1` then the i'th tensor in `output` is the slice `value[:, i, :, :]`
//   and each tensor in `output` will have shape `(A, C, D)`.
// Etc.
// 
// This is the opposite of `pack`.

*/








func unpack( scope:Scope,value: Output, num :UInt8  , axis :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["num"] = num
    attrs["axis"] = axis

    let opspec = OpSpec(
        OpType: "Unpack",
        Name: "Type",
        Input: [ value],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the Max along segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
// segments.
// 
// This operator is similar to the [unsorted segment sum operator](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
// Instead of computing the sum over segments, it computes the maximum
// such that:
// 
// \\(output_i = \max_j data_j\\) where max is over `j` such
// that `segment_ids[j] == i`.
// 
// If the maximum is empty for a given segment ID `i`, it outputs the smallest possible value for specific numeric type,
//  `output[i] = numeric_limits<T>::min()`.
// 
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentSum.png" alt>
// </div>

*/








func unsortedSegmentMax( scope:Scope,data: Output, segmentIds: Output, numSegments: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "UnsortedSegmentMax",
        Name: "Type",
        Input: [ data, segmentIds, numSegments],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Computes the sum along segments of a tensor.

Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
// segments.
// 
// Computes a tensor such that
// `(output[i] = sum_{j...} data[j...]` where the sum is over tuples `j...` such
// that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
// need not be sorted and need not cover all values in the full
// range of valid values.
// 
// If the sum is empty for a given segment ID `i`, `output[i] = 0`.
// 
// `num_segments` should equal the number of distinct segment IDs.
// 
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentSum.png" alt>
// </div>

*/








func unsortedSegmentSum( scope:Scope,data: Output, segmentIds: Output, numSegments: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "UnsortedSegmentSum",
        Name: "Type",
        Input: [ data, segmentIds, numSegments],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Op is similar to a lightweight Dequeue.

The basic funtionality is similar to dequeue with many fewer
// capabilities and options.  This Op is optimized for performance.

*/








func unstage( scope:Scope, container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "Unstage",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Use VariableV2 instead.


*/








func variable( scope:Scope, shape :Shape  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shape"] = shape
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "Variable",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Holds state in the form of a tensor that persists across steps.

Outputs a ref to the tensor state so it may be read or modified.
// TODO(zhifengc/mrry): Adds a pointer to a more detail document
// about sharing states in tensorflow.

*/








func variableV2( scope:Scope, shape :Shape  , container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["shape"] = shape
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "VariableV2",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Returns locations of true values in a boolean tensor.

This operation returns the coordinates of true elements in `input`. The
// coordinates are returned in a 2-D tensor where the first dimension (rows)
// represents the number of true elements, and the second dimension (columns)
// represents the coordinates of the true elements. Keep in mind, the shape of
// the output tensor can vary depending on how many true values there are in
// `input`. Indices are output in row-major order.
// 
// For example:
// 
// ```
// # 'input' tensor is [[True, False]
// #                    [True, False]]
// # 'input' has two true values, so output has two coordinates.
// # 'input' has rank of 2, so coordinates have two indices.
// where(input) ==> [[0, 0],
//                   [1, 0]]
// 
// # `input` tensor is [[[True, False]
// #                     [True, False]]
// #                    [[False, True]
// #                     [False, True]]
// #                    [[False, False]
// #                     [False, True]]]
// # 'input' has 5 true values, so output has 5 coordinates.
// # 'input' has rank of 3, so coordinates have three indices.
// where(input) ==> [[0, 0, 0],
//                   [0, 1, 0],
//                   [1, 0, 1],
//                   [1, 1, 1],
//                   [2, 1, 1]]
// ```

*/








func where_p( scope:Scope,input: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "where_p",
        Name: "Type",
        Input: [ input],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A Reader that outputs the entire contents of a file as a value.

To use, enqueue filenames in a Queue.  The output of ReaderRead will
// be a filename (key) and the contents of that file (value).

*/








func wholeFileReader( scope:Scope, container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "WholeFileReader",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
A Reader that outputs the entire contents of a file as a value.

To use, enqueue filenames in a Queue.  The output of ReaderRead will
// be a filename (key) and the contents of that file (value).

*/








func wholeFileReaderV2( scope:Scope, container :String  , sharedName :String  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["container"] = container
    attrs["shared_name"] = sharedName

    let opspec = OpSpec(
        OpType: "WholeFileReaderV2",
        Name: "Type",
        Input: [ ],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Writes contents to the file at input filename. Creates file if not existing.


*/








func writeFile( scope:Scope,filename: Output, contents: Output )  ->(GoOperation?){
    if scope.error.error != nil {
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "WriteFile",
        Name: "Type",
        Input: [ filename, contents],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return op
}

/*
Returns a tensor of zeros with the same shape and type as x.


*/








func zerosLike( scope:Scope,x: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "ZerosLike",
        Name: "Type",
        Input: [ x],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Compute the Hurwitz zeta function \\(\zeta(x, q)\\).

The Hurwitz zeta function is defined as:
// 
// 
// \\(\zeta(x, q) = \sum_{n=0}// ^{\infty} (q + n)// ^{-x}\\)

*/








func zeta( scope:Scope,x: Output, q: Output )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]

    let opspec = OpSpec(
        OpType: "Zeta",
        Name: "Type",
        Input: [ x, q],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

/*
Creates a dataset that zips together `input_datasets`.


*/








func zipDataset( scope:Scope,inputDatasets: Output, outputShapes :[Shape]  , n :UInt8  )  ->(Output?){
    if scope.error.error != nil {
        return (nil)
    }

    var attrs:Dictionary<String,Any> = [:]
    attrs["output_shapes"] = outputShapes
    attrs["N"] = n

    let opspec = OpSpec(
        OpType: "ZipDataset",
        Name: "Type",
        Input: [ inputDatasets],
        Attrs: attrs
    )
    let op = scope.AddOperation(opspec)
    return (op?.output(1 - 1))
}

