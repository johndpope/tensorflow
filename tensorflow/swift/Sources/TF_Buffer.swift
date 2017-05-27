
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


extension tf{
    
    // Makes a copy of the input and sets an appropriate deallocator.  Useful for
    // passing in read-only, input protobufs.
    public class func NewBufferFromString(_ proto: UnsafeRawPointer!, _ proto_len: Int) -> UnsafeMutablePointer<TF_Buffer>!{
        return TF_NewBufferFromString(proto,proto_len)
    }
    
    // Useful for passing *out* a protobuf.
    public class func NewBuffer() -> UnsafeMutablePointer<TF_Buffer>!{
        return TF_NewBuffer()
    }
    
    public class func DeleteBuffer(_ unsafePointer: UnsafeMutablePointer<TF_Buffer>!){
        TF_DeleteBuffer(unsafePointer)
    }
    
    public class func GetBuffer(_ buffer: UnsafeMutablePointer<TF_Buffer>!) -> TF_Buffer{
        return TF_GetBuffer(buffer)
    }
}

