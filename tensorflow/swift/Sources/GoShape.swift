/// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
///
/// Licensed under the Apache License, Version 2.0 (the "License");
/// you may not use this file except in compliance with the License.
/// You may obtain a copy of the License at
///
/// http://www.apache.org/licenses/LICENSE-2.0
///
/// Unless required by applicable law or agreed to in writing, software
/// distributed under the License is distributed on an "AS IS" BASIS,
/// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
/// See the License for the specific language governing permissions and
/// limitations under the License.
//  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/shape.go
/// ==============================================================================
import Foundation


// Shape represents the (possibly partially known) shape of a tensor that will
// be produced by an operation.
//
// The zero-value of a Shape represents a shape with an unknown number of
// dimensions.

// TODO - use Tensorflow_TensorShapeProto see tensor_shape.pb.swift
struct Shape  {
    var  dims: [Int64]
}

// NumDimensions returns the number of dimensions represented by s, or -1 if
// unknown.
extension Shape{
    func NumDimensions() -> CInt{
        if(self.dims.count == 0){
            return -1
        }else{
            return CInt(dims.count)
        }
    }
}


/*
// ScalarShape returns a Shape representing a scalar.
func ScalarShape()->Shape {
    return Shape{dims: make([]int64, 0)}
}


// MakeShape returns a Shape with the provided size of each dimension.
//
// A value of -1 implies that the size of the corresponding dimension is not
// known.
func MakeShape(shape ...int64)->Shape {
    cpy = make([]int64, len(shape))
    copy(cpy, shape)
    return Shape{dims: cpy}
}

*/


// Size returns the size of the dim-th dimension of the shape, or -1 if it
// is unknown.
//
// REQUIRES: 0 <= dim < s.NumDimensions()
extension Shape{
    func Size(dim:Int) -> Int64 {
        if dim < 0 || dim > self.dims.count{
            return -1
        }
        return self.dims[dim]
    }
}

// IsFullySpecified returns true iff the size of all the dimensions of s are
// known.
extension Shape{
    func IsFullySpecified()-> Bool {
        if (self.dims.count == 0) {
            return false
        }
        for size in self.dims {
            if (size <= 1) {
                return false
            }
        }
        return true
    }
}

// ToSlice returns the (possibly partially known) shape represented by s as a
// slice, or an error if the number of dimensions is not known.
extension Shape{
    func ToSlice()-> ([Int64]?, NSError?) {
        if (self.dims.count == 0) {
            return (nil, NSError.newIoError("cannot create a slice for a Shape with an unknown number of dimensions",code:000))
        }
        let copy:[Int64] = self.dims
        return (copy, nil)
    }
}

// TODO - is this just to debug shape? maybe just use debugDescription instead.
extension Shape{
    func String()-> String {
        if (self.dims.count == 0) {
            return "?"
        }
        let ret = "\(self.dims)"
        for size in self.dims {
            if (size < 0 ){
               // ret = strings.Replace(ret, fmt.Sprint(size), "?", 1)
            }
        }
       // return strings.Replace(ret, " ", ", ", -1)
        return self.dims.debugDescription
    }
}

