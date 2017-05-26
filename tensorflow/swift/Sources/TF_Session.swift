
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
// API for driving Graph execution.

// Return a new execution session with the associated graph, or NULL on error.
//
// *graph must be a valid graph (not deleted or nullptr).  This function will
// prevent the graph from being deleted until TF_DeleteSession() is called.
// Does not take ownership of opts.
public func tfNewSession(_ graph: TF_Graph!, _ opts: TF_SessionOptions!, _ status: TF_Status!) -> TF_Session!{
      let status:TF_Session = TF_NewSession(graph, opts,status)
      return status
}

// This function creates a new TF_Session (which is created on success) using
// `session_options`, and then initializes state (restoring tensors and other
// assets) using `run_options`.
//
// Any NULL and non-NULL value combinations for (`run_options, `meta_graph_def`)
// are valid.
//
// - `export_dir` must be set to the path of the exported SavedModel.
// - `tags` must include the set of tags used to identify one MetaGraphDef in
//    the SavedModel.
// - `graph` must be a graph newly allocated with TF_NewGraph().
//
// If successful, populates `graph` with the contents of the Graph and
// `meta_graph_def` with the MetaGraphDef of the loaded model.
public func tfLoadSessionFromSavedModel(_ session_options: TF_SessionOptions!, _ run_options: UnsafePointer<TF_Buffer>!, _ export_dir: UnsafePointer<Int8>!, _ tags: UnsafePointer<UnsafePointer<Int8>?>!, _ tags_len: Int32, _ graph: TF_Graph!, _ meta_graph_def: UnsafeMutablePointer<TF_Buffer>!, _ status: TF_Status!) -> TF_Session!{
    return TF_LoadSessionFromSavedModel(session_options,run_options,export_dir,tags,tags_len,graph,meta_graph_def,status)
}

// Close a session.
//
// Contacts any other processes associated with the session, if applicable.
// May not be called after TF_DeleteSession().
public func tfCloseSession(_ pointer:TF_Session!, _ status: TF_Status!){
    TF_CloseSession(pointer,status)
}

// Destroy a session object.
//
// Even if error information is recorded in *status, this call discards all
// local resources associated with the session.  The session may not be used
// during or after this call (and the session drops its reference to the
// corresponding graph).
public func tfDeleteSession(_ pointer:TF_Session!, _ status: TF_Status!){
    TF_DeleteSession(pointer,status)
}

// Run the graph associated with the session starting with the supplied inputs
// (inputs[0,ninputs-1] with corresponding values in input_values[0,ninputs-1]).
//
// Any NULL and non-NULL value combinations for (`run_options`,
// `run_metadata`) are valid.
//
//    - `run_options` may be NULL, in which case it will be ignored; or
//      non-NULL, in which case it must point to a `TF_Buffer` containing the
//      serialized representation of a `RunOptions` protocol buffer.
//    - `run_metadata` may be NULL, in which case it will be ignored; or
//      non-NULL, in which case it must point to an empty, freshly allocated
//      `TF_Buffer` that may be updated to contain the serialized representation
//      of a `RunMetadata` protocol buffer.
//
// The caller retains ownership of `input_values` (which can be deleted using
// TF_DeleteTensor). The caller also retains ownership of `run_options` and/or
// `run_metadata` (when not NULL) and should manually call TF_DeleteBuffer on
// them.
//
// On success, the tensors corresponding to outputs[0,noutputs-1] are placed in
// output_values[]. Ownership of the elements of output_values[] is transferred
// to the caller, which must eventually call TF_DeleteTensor on them.
//
// On failure, output_values[] contains NULLs.
public func tfSessionRun(_ session: TF_Session!, _ run_options: UnsafePointer<TF_Buffer>!, _ inputs: UnsafePointer<TF_Output>!, _ input_values: UnsafePointer<OpaquePointer?>!, _ ninputs: Int32, _ outputs: UnsafePointer<TF_Output>!, _ output_values: UnsafeMutablePointer<OpaquePointer?>!, _ noutputs: Int32, _ target_opers: UnsafePointer<OpaquePointer?>!, _ ntargets: Int32, _ run_metadata: UnsafeMutablePointer<TF_Buffer>!, pointer:OpaquePointer!){
     print("TO IMPLEMENT")
//TF_SessionPRunSetup(oPointer,inputs,ninputs,outputs,noutputs,target_opers,ntargets,handle,pointer)
}

// RunOptions

// Input tensors

// Output tensors

// Target operations

// RunMetadata

// Output status

// Set up the graph with the intended feeds (inputs) and fetches (outputs) for a
// sequence of partial run calls.
//
// On success, returns a handle that is used for subsequent PRun calls. The
// handle should be deleted with TF_DeletePRunHandle when it is no longer
// needed.
//
// On failure, out_status contains a tensorflow::Status with an error
// message.
// NOTE: This is EXPERIMENTAL and subject to change.
public func tfSessionPRunSetup(oPointer:TF_Session!, _ inputs: UnsafePointer<TF_Output>!, _ ninputs: Int32, _ outputs: UnsafePointer<TF_Output>!, _ noutputs: Int32, _ target_opers: UnsafePointer<OpaquePointer?>!, _ ntargets: Int32, _ handle: UnsafeMutablePointer<UnsafePointer<Int8>?>!, pointer:OpaquePointer!){
    TF_SessionPRunSetup(oPointer,inputs,ninputs,outputs,noutputs,target_opers,ntargets,handle,pointer)
}

// Input names

// Output names

// Target operations

// Output handle

// Output status

// Continue to run the graph with additional feeds and fetches. The
// execution state is uniquely identified by the handle.
// NOTE: This is EXPERIMENTAL and subject to change.
public func tfSessionPRun(oPointer:TF_Session!, _ handle: UnsafePointer<Int8>!, _ inputs: UnsafePointer<TF_Output>!, _ input_values: UnsafePointer<OpaquePointer?>!, _ ninputs: Int32, _ outputs: UnsafePointer<TF_Output>!, _ output_values: UnsafeMutablePointer<OpaquePointer?>!, _ noutputs: Int32, _ target_opers: UnsafePointer<OpaquePointer?>!, _ ntargets: Int32, pointer:OpaquePointer!){
    TF_SessionPRun(oPointer,handle,inputs,input_values,ninputs,outputs,output_values,noutputs,target_opers,ntargets,pointer)
}

// Input tensors

// Output tensors

// Target operations

// Output status

// Deletes a handle allocated by TF_SessionPRunSetup.
// Once called, no more calls to TF_SessionPRun should be made.
public func tfDeletePRunHandle(_ handle: UnsafePointer<Int8>!){
    return TF_DeletePRunHandle(handle)
}

// --------------------------------------------------------------------------
// The deprecated session API.  Please switch to the above instead of
// TF_ExtendGraph(). This deprecated API can be removed at any time without
// notice.

public func tfNewDeprecatedSession(_ pointer:TF_SessionOptions!, _ status: TF_Status!) -> TF_DeprecatedSession!{
    return TF_NewDeprecatedSession(pointer,status)
}
public func tfCloseDeprecatedSession(_ pointer:TF_DeprecatedSession!, _ status: TF_Status!){
 print("TO IMPLEMENT")
}
public func tfDeleteDeprecatedSession(_ pointer:TF_DeprecatedSession!, _ status: TF_Status!){
 print("TO IMPLEMENT")
}
public func tfReset(_ opt: OpaquePointer!, _ containers: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ ncontainers: Int32, _ status: TF_Status!){
 print("TO IMPLEMENT")
}
// Treat the bytes proto[0,proto_len-1] as a serialized GraphDef and
// add the nodes in that GraphDef to the graph for the session.
//
// Prefer use of TF_Session and TF_GraphImportGraphDef over this.
public func tfExtendGraph(oPointer:OpaquePointer!, _ proto: UnsafeRawPointer!, _ proto_len: Int, pointer:OpaquePointer!){
 print("TO IMPLEMENT")
}

// See TF_SessionRun() above.
public func tfRun(oPointer:OpaquePointer!, _ run_options: UnsafePointer<TF_Buffer>!, _ input_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ inputs: UnsafeMutablePointer<OpaquePointer?>!, _ ninputs: Int32, _ output_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ outputs: UnsafeMutablePointer<OpaquePointer?>!, _ noutputs: Int32, _ target_oper_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ ntargets: Int32, _ run_metadata: UnsafeMutablePointer<TF_Buffer>!, pointer:OpaquePointer!){
 print("TO IMPLEMENT")
}

// See TF_SessionPRunSetup() above.
public func tfPRunSetup(oPointer:OpaquePointer!, _ input_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ ninputs: Int32, _ output_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ noutputs: Int32, _ target_oper_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ ntargets: Int32, _ handle: UnsafeMutablePointer<UnsafePointer<Int8>?>!, pointer:OpaquePointer!){
 print("TO IMPLEMENT")
}

// See TF_SessionPRun above.
public func tfPRun(oPointer:OpaquePointer!, _ handle: UnsafePointer<Int8>!, _ input_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ inputs: UnsafeMutablePointer<OpaquePointer?>!, _ ninputs: Int32, _ output_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ outputs: UnsafeMutablePointer<OpaquePointer?>!, _ noutputs: Int32, _ target_oper_names: UnsafeMutablePointer<UnsafePointer<Int8>?>!, _ ntargets: Int32, pointer:OpaquePointer!){
    TF_PRun(oPointer,handle,input_names,inputs,ninputs,output_names,outputs,noutputs,target_oper_names,ntargets,pointer)
}


// --------------------------------------------------------------------------
// TF_SessionOptions holds options that can be passed during session creation.

// Return a new options object.
public func tfNewSessionOptions() -> TF_SessionOptions!{
    return TF_NewSessionOptions()
}

// Set the target in TF_SessionOptions.options.
// target can be empty, a single entry, or a comma separated list of entries.
// Each entry is in one of the following formats :
// "local"
// ip:port
// host:port
public func tfSetTarget(_ options: TF_SessionOptions!, _ target: UnsafePointer<Int8>!){
    TF_SetTarget(options,target)
}

// Set the config in TF_SessionOptions.options.
// config should be a serialized tensorflow.ConfigProto proto.
// If config was not parsed successfully as a ConfigProto, record the
// error information in *status.
public func tfSetConfig(_ options: TF_SessionOptions!, _ proto: UnsafeRawPointer!, _ proto_len: Int, _ status: TF_Status!){
    TF_SetConfig(options,proto,proto_len,status)
}

// Destroy an options object.
public func tfDeleteSessionOptions(_ pointer:TF_SessionOptions!){
    TF_DeleteSessionOptions(pointer)
}

