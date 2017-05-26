
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
// Load plugins containing custom ops and kernels

// TF_Library holds information about dynamically loaded TensorFlow plugins.

// Load the library specified by library_filename and register the ops and
// kernels present in that library.
//
// Pass "library_filename" to a platform-specific mechanism for dynamically
// loading a library. The rules for determining the exact location of the
// library are platform-specific and are not documented here.
//
// On success, place OK in status and return the newly created library handle.
// The caller owns the library handle.
//
// On failure, place an error status in status and return NULL.
public func tfLoadLibrary(_ library_filename: UnsafePointer<Int8>!, _ status: TF_Status!) -> TF_Library!{
    return TF_LoadLibrary(library_filename,status)
}

// Get the OpList of OpDefs defined in the library pointed by lib_handle.
//
// Returns a TF_Buffer. The memory pointed to by the result is owned by
// lib_handle. The data in the buffer will be the serialized OpList proto for
// ops defined in the library.
public func tfGetOpList(_ lib_handle: OpaquePointer!) -> TF_Buffer{
    let buffer:TF_Buffer = TF_GetOpList(lib_handle)
    return buffer
}

// Frees the memory associated with the library handle.
// Does NOT unload the library.
public func tfDeleteLibraryHandle(_ lib_handle: TF_Library!){
    TF_DeleteLibraryHandle(lib_handle)
}

// Get the OpList of all OpDefs defined in this address space.
// Returns a TF_Buffer, ownership of which is transferred to the caller
// (and can be freed using TF_DeleteBuffer).
//
// The data in the buffer will be the serialized OpList proto for ops registered
// in this address space.
public func tfGetAllOpList() -> UnsafeMutablePointer<TF_Buffer>!{
    return TF_GetAllOpList()
}



