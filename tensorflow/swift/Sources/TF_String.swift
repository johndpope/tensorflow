
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
    // --------------------------------------------------------------------------
    // Encode the string `src` (`src_len` bytes long) into `dst` in the format
    // required by TF_STRING tensors. Does not write to memory more than `dst_len`
    // bytes beyond `*dst`. `dst_len` should be at least
    // TF_StringEncodedSize(src_len).
    //
    // On success returns the size in bytes of the encoded string.
    // Returns an error into `status` otherwise.
    public class func StringEncode(_ src: UnsafePointer<Int8>!, _ src_len: Int, _ dst: UnsafeMutablePointer<Int8>!, _ dst_len: Int, _ status: TF_Status!) -> Int{
        return TF_StringEncode(src,src_len,dst,dst_len,status)
    }
    
    // Decode a string encoded using TF_StringEncode.
    //
    // On success, sets `*dst` to the start of the decoded string and `*dst_len` to
    // its length. Returns the number of bytes starting at `src` consumed while
    // decoding. `*dst` points to memory within the encoded buffer.  On failure,
    // `*dst` and `*dst_len` are undefined and an error is set in `status`.
    //
    // Does not read memory more than `src_len` bytes beyond `src`.
    public class func StringDecode(_ src: UnsafePointer<Int8>!, _ src_len: Int, _ dst: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ dst_len: UnsafeMutablePointer<Int>!, _ status: TF_Status!) -> Int{
        return TF_StringDecode(src,src_len,dst,dst_len,status)
    }
    
    // Return the size in bytes required to encode a string `len` bytes long into a
    // TF_STRING tensor.
    public class func StringEncodedSize(_ len: Int) -> Int{
        return TF_StringEncodedSize(len)
    }
}

