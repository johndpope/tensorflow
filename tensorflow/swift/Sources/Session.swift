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
/// ==============================================================================

import CTensorFlow


// Session drives a TensorFlow graph computation.
//
// When a Session is created with a given target, a new Session object is bound
// to the universe of resources specified by that target. Those resources are
// available to this session to perform computation described in the GraphDef.
// After creating the session with a graph, the caller uses the Run() API to
// perform the computation and potentially fetch outputs as Tensors.
// A Session allows concurrent calls to Run().
struct tfSession  {
    //var c:TF_Session
    var c:OpaquePointer
    
    // For ensuring that:
    // - Close() blocks on all Run() calls to complete.
    // - Close() can be called multiple times.
    //var wg:WaitGroup
    //var mu:Mutex
}


// SessionOptions contains configuration information for a session.
struct tfSessionOptions  {
    // Target indicates the TensorFlow runtime to connect to.
    //
    // If 'target' is empty or unspecified, the local TensorFlow runtime
    // implementation will be used.  Otherwise, the TensorFlow engine
    // defined by 'target' will be used to perform all computations.
    //
    // "target" can be either a single entry or a comma separated list
    // of entries. Each entry is a resolvable address of one of the
    // following formats:
    //   local
    //   ip:port
    //   host:port
    //   ... other system-specific formats to identify tasks and jobs ...
    //
    // NOTE: at the moment 'local' maps to an in-process service-based
    // runtime.
    //
    // Upon creation, a single session affines itself to one of the
    // remote processes, with possible load balancing choices when the
    // "target" resolves to a list of possible processes.
    //
    // If the session disconnects from the remote process during its
    // lifetime, session calls may fail immediately.
    var Target:String
    
    // Config is a binary-serialized representation of the
    // tensorflow.ConfigProto protocol message
    // (https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto).
   // var Config:Tensorflow_ConfigProto //- TODO - decide whether to import monolithic grpc + tensorflow grpc swift
    var Config:OpaquePointer
    
}




// NewSession creates a new execution session with the associated graph.
// options may be nil to use the default options.
func newSession(graph:tfGraph, options:tfSessionOptions)-> (session:tfSession?, error:Tensorflow_Error_Code?) {
    
    
    let status = newStatus()
   // var cOpt, doneOpt, err = options.c() // how to do this in swift??
   // defer doneOpt()
   // if err != nil {
   //     return nil, err
   // }
    let cOpt = TF_NewSessionOptions()
    if let cSess = TF_NewSession(graph.c, cOpt, status.c){
        
        let s =  tfSession(c: cSess)
        //    runtime.SetFinalizer(s, func(s *Session) { s.Close() }) // how to do this in swift??
        return (s, nil)
    }else{
    
        let code = tfGetCode(status.c)
        let intRaw:Int = Int(code.rawValue)
        return (nil,Tensorflow_Error_Code(rawValue: intRaw))
    }
}
/*
 

// Run the graph with the associated session starting with the supplied feeds
// to compute the value of the requested fetches. Runs, but does not return
// Tensors for operations specified in targets.
//
// On success, returns the fetched Tensors in the same order as supplied in
// the fetches argument. If fetches is set to nil, the returned Tensor fetches
// is empty.
func (s *Session) Run(feeds map[Output]*Tensor, fetches []Output, targets []*Operation) ([]*Tensor, error) {
    s.mu.Lock()
    if s.c == nil {
        s.mu.Unlock()
        return nil, errors.New("session is closed")
    }
    s.wg.Add(1)
    s.mu.Unlock()
    defer s.wg.Done()
    
    c = newCRunArgs(feeds, fetches, targets)
    status = newStatus()
    TF_SessionRun(s.c, nil,
    ptrOutput(c.feeds), ptrTensor(c.feedTensors), C.int(len(feeds)),
    ptrOutput(c.fetches), ptrTensor(c.fetchTensors), C.int(len(fetches)),
    ptrOperation(c.targets), C.int(len(targets)),
    nil, status.c)
    if err = status.Err(); err != nil {
        return nil, err
    }
    return c.toGo(), nil
}

// PartialRun enables incremental evaluation of graphs.
//
// PartialRun allows the caller to pause the evaluation of a graph, run
// arbitrary code that depends on the intermediate computation of the graph,
// and then resume graph execution. The results of the arbitrary code can be
// fed into the graph when resuming execution.  In contrast, Session.Run
// executes the graph to compute the requested fetches using the provided feeds
// and discards all intermediate state (e.g., value of intermediate tensors)
// when it returns.
//
// For example, consider a graph for unsupervised training of a neural network
// model. PartialRun can be used to pause execution after the forward pass of
// the network, let the caller actuate the output (e.g., play a game, actuate a
// robot etc.), determine the error/loss and then feed this calculated loss
// when resuming the backward pass of the graph.
type PartialRun struct {
    session *Session
    handle  *C.char
}

// Run resumes execution of the graph to compute the requested fetches and
// targets with the provided feeds.
func (pr *PartialRun) Run(feeds map[Output]*Tensor, fetches []Output, targets []*Operation) ([]*Tensor, error) {
    var (
    c      = newCRunArgs(feeds, fetches, targets)
    status = newStatus()
    s      = pr.session
    )
    s.mu.Lock()
    if s.c == nil {
        s.mu.Unlock()
        return nil, errors.New("session is closed")
    }
    s.wg.Add(1)
    s.mu.Unlock()
    defer s.wg.Done()
    
    TF_SessionPRun(s.c, pr.handle,
    ptrOutput(c.feeds), ptrTensor(c.feedTensors), C.int(len(feeds)),
    ptrOutput(c.fetches), ptrTensor(c.fetchTensors), C.int(len(fetches)),
    ptrOperation(c.targets), C.int(len(targets)),
    status.c)
    if err = status.Err(); err != nil {
        return nil, err
    }
    return c.toGo(), nil
}

// NewPartialRun sets up the graph for incremental evaluation.
//
// All values of feeds, fetches and targets that may be provided to Run calls
// on the returned PartialRun need to be provided to NewPartialRun.
//
// See documentation for the PartialRun type.
func (s *Session) NewPartialRun(feeds, fetches []Output, targets []*Operation) (*PartialRun, error) {
    var (
    cfeeds   = make([]TF_Output, len(feeds))
    cfetches = make([]TF_Output, len(fetches))
    ctargets = make([]*TF_Operation, len(targets))
    
    pcfeeds   *TF_Output
    pcfetches *TF_Output
    pctargets **TF_Operation
    
    status = newStatus()
    )
    if len(feeds) > 0 {
        pcfeeds = &cfeeds[0]
        for i, o = range feeds {
            cfeeds[i] = o.c()
        }
    }
    if len(fetches) > 0 {
        pcfetches = &cfetches[0]
        for i, o = range fetches {
            cfetches[i] = o.c()
        }
    }
    if len(targets) > 0 {
        pctargets = &ctargets[0]
        for i, o = range targets {
            ctargets[i] = o.c
        }
    }
    
    s.mu.Lock()
    if s.c == nil {
        s.mu.Unlock()
        return nil, errors.New("session is closed")
    }
    s.wg.Add(1)
    s.mu.Unlock()
    defer s.wg.Done()
    
    pr = &PartialRun{session: s}
    TF_SessionPRunSetup(s.c,
                          pcfeeds, C.int(len(feeds)),
                          pcfetches, C.int(len(fetches)),
                          pctargets, C.int(len(targets)),
                          &pr.handle, status.c)
    if err = status.Err(); err != nil {
        return nil, err
    }
    runtime.SetFinalizer(pr, func(pr *PartialRun) {
        deletePRunHandle(pr.handle)
    })
    return pr, nil
}

// Close a session. This contacts any other processes associated with this
// session, if applicable. Blocks until all previous calls to Run have returned.
func (s *Session) Close() error {
    s.mu.Lock()
    defer s.mu.Unlock()
    s.wg.Wait()
    if s.c == nil {
        return nil
    }
    status = newStatus()
    TF_CloseSession(s.c, status.c)
    if err = status.Err(); err != nil {
        return err
    }
    TF_DeleteSession(s.c, status.c)
    s.c = nil
    return status.Err()
}

// SessionOptions contains configuration information for a session.
type SessionOptions struct {
    // Target indicates the TensorFlow runtime to connect to.
    //
    // If 'target' is empty or unspecified, the local TensorFlow runtime
    // implementation will be used.  Otherwise, the TensorFlow engine
    // defined by 'target' will be used to perform all computations.
    //
    // "target" can be either a single entry or a comma separated list
    // of entries. Each entry is a resolvable address of one of the
    // following formats:
    //   local
    //   ip:port
    //   host:port
    //   ... other system-specific formats to identify tasks and jobs ...
    //
    // NOTE: at the moment 'local' maps to an in-process service-based
    // runtime.
    //
    // Upon creation, a single session affines itself to one of the
    // remote processes, with possible load balancing choices when the
    // "target" resolves to a list of possible processes.
    //
    // If the session disconnects from the remote process during its
    // lifetime, session calls may fail immediately.
    Target string
    
    // Config is a binary-serialized representation of the
    // tensorflow.ConfigProto protocol message
    // (https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto).
    Config []byte
}

// c converts the SessionOptions to the C API's TF_SessionOptions. Callers must
// deallocate by calling the returned done() closure.
func (o *SessionOptions) c() (ret *TF_SessionOptions, done func(), err error) {
    opt = TF_NewSessionOptions()
    if o == nil {
        return opt, func() { TF_DeleteSessionOptions(opt) }, nil
    }
    t = C.CString(o.Target)
    TF_SetTarget(opt, t)
    C.free(unsafe.Pointer(t))
    
    var cConfig unsafe.Pointer
    if sz = len(o.Config); sz > 0 {
        status = newStatus()
        // Copying into C-memory is the simplest thing to do in terms
        // of memory safety and cgo rules ("C code may not keep a copy
        // of a Go pointer after the call returns" from
        // https://golang.org/cmd/cgo/#hdr-Passing_pointers).
        cConfig = C.CBytes(o.Config)
        TF_SetConfig(opt, cConfig, C.size_t(sz), status.c)
        if err = status.Err(); err != nil {
            TF_DeleteSessionOptions(opt)
            return nil, func() {}, fmt.Errorf("invalid SessionOptions.Config: %v", err)
        }
    }
    return opt, func() {
        TF_DeleteSessionOptions(opt)
        C.free(cConfig)
    }, nil
}
*/


// cRunArgs translates the arguments to Session.Run and PartialRun.Run into
// values suitable for C library calls.
/*struct cRunArgs  {
    var feeds:TF_Output
    var feedTensors:TF_Tensor
    var fetches:TF_Output
    var fetchTensors:TF_Tensor
    var targets:TF_Operation
}*/
/*
func newCRunArgs(feeds map[Output]*Tensor, fetches []Output, targets []*Operation) *cRunArgs {
    c = &cRunArgs{
        fetches:      make([]TF_Output, len(fetches)),
        fetchTensors: make([]*TF_Tensor, len(fetches)),
        targets:      make([]*TF_Operation, len(targets)),
    }
    for o, t = range feeds {
        c.feeds = append(c.feeds, o.c())
        c.feedTensors = append(c.feedTensors, t.c)
    }
    for i, o = range fetches {
        c.fetches[i] = o.c()
    }
    for i, t = range targets {
        c.targets[i] = t.c
    }
    return c
}

func (c *cRunArgs) toGo() []*Tensor {
    ret = make([]*Tensor, len(c.fetchTensors))
    for i, ct = range c.fetchTensors {
        ret[i] = newTensorFromC(ct)
    }
    return ret
}

func ptrOutput(l []TF_Output) *TF_Output {
    if len(l) == 0 {
        return nil
    }
    return &l[0]
}

func ptrTensor(l []*TF_Tensor) **TF_Tensor {
    if len(l) == 0 {
        return nil
    }
    return &l[0]
}

func ptrOperation(l []*TF_Operation) **TF_Operation {
    if len(l) == 0 {
        return nil
    }
    return &l[0]
}
 */

