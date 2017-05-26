
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
// TF_Status holds error information.  It either has an OK code, or
// else an error code with an associated error message.

// Return a new status object.
public func tfNewStatus() -> TF_Status!{
    return TF_NewStatus()
}

// Delete a previously created status object.
public func tfDeleteStatus(_ pointer:TF_Status!){
    TF_DeleteStatus(pointer)
}

// Record <code, msg> in *s.  Any previous information is lost.
// A common use is to clear a status: TF_SetStatus(s, TF_OK, "");
public func tfSetStatus(_ s: TF_Status!, _ code: TF_Code, _ msg: UnsafePointer<Int8>!){
    TF_SetStatus(s,code,msg)
}

// Return the code record in *s.
public func tfGetCode(_ s: TF_Status!) -> TF_Code{
    return TF_GetCode(s)
}

// Return a pointer to the (null-terminated) error message in *s.  The
// return value points to memory that is only usable until the next
// mutation to *s.  Always returns an empty string if TF_GetCode(s) is
// TF_OK.
public func tfMessage(_ s: TF_Status!) -> String{
    let str:UnsafePointer<Int8> = TF_Message(s)
    return  String(cString:str)
}



