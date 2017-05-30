/*
 Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 
 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/op/scope.go
 
 */
import Foundation
/*
 */
// Scope encapsulates common operation properties when building a Graph.
//
// A Scope object (and its derivates, e.g., obtained from Scope.SubScope)
// act as a builder for graphs. They allow common properties (such as
// a name prefix) to be specified for multiple operations being added
// to the graph.
//
// A Scope object and all its derivates (e.g., obtained from Scope.SubScope)
// are not safe for concurrent use by multiple goroutines.
struct Scope  {
    var graph:Graph
    var namemap:Dictionary<String,Int>
    var namespace:String = ""
    var error:scopeError
    
}

// scopeErr is used to share errors between all derivatives of a root scope.
struct scopeError  {
    var error:NSError?
}

// NewScope creates a Scope initialized with an empty Graph.
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/op/scope.go#L47
func NewScope()->Scope {
    return  Scope(graph: newGraph(), namemap:   [String: Int]() , namespace: "", error: scopeError())
}

// NewScopeWithGraph creates a Scope initialized with the Graph thats passed in
func NewScopeWithGraph(g:Graph)-> Scope {
    return  Scope(graph: g, namemap:   [String: Int]() , namespace: "", error: scopeError())
}

// Finalize returns the Graph on which this scope operates on and renders s
// unusable. If there was an error during graph construction, that error is
// returned instead.

extension Scope{
    func Finalize()-> (Graph?, NSError?) {
        if let error = self.error.error {
            return (nil, error)
        }
        print("Scope has been finalized and is no longer usable")
        return (self.graph, nil)
    }
}



// AddOperation adds the operation to the Graph managed by s.
//
// If there is a name prefix associated with s (such as if s was created
// by a call to SubScope), then this prefix will be applied to the name
// of the operation being added. See also Graph.AddOperation.
extension Scope{
    
    mutating func  AddOperation(args:OpSpec) -> GoOperation? {
        
        var args = args
        
        if self.error.error != nil {
            return nil
        }
        
        if (args.Name == "") {
            args.Name = args.OpType
        }
        
        if (self.namespace != "") {
            args.Name = self.namespace + "/" + args.Name
        }
        
        let op:GoOperation?
        let error:NSError?
        
        
        (op, error) = self.graph.AddOperation(args:args)
        if let error = error {
            self.UpdateError(op: args.OpType, error: error)
        }
        return op
    }
}


// SubScope returns a new Scope which will cause all operations added to the
// graph to be namespaced with 'namespace'.  If namespace collides with an
// existing namespace within the scope, then a suffix will be added.
/*func (s *Scope) SubScope(namespace string) *Scope {
	namespace = s.uniqueName(namespace)
	if s.namespace != "" {
 namespace = s.namespace + "/" + namespace
	}
	return &Scope{
 graph:     s.graph,
 namemap:   make(map[string]int),
 namespace: namespace,
 err:       s.err,
	}
 }*/

// Err returns the error, if any, encountered during the construction
// of the Graph managed by s.
//
// Once Err returns a non-nil error, all future calls will do the same,
// indicating that the scope should be discarded as the graph could not
// be constructed.
extension Scope{
    func  Error() -> NSError? {
        return self.error.error
    }
}


// UpdateErr is used to notify Scope of any graph construction errors
// while creating the operation op.
extension Scope{
    mutating func  UpdateError(op:String, error: NSError) {
        if self.error.error == nil {
            let msg = "failed to add operation \(op.description): \(error) (Stacktrace: ))" //\(debug.Stack()
            self.error.error = NSError.newIoError("failed to add operation %q: %v (Stacktrace: %s)", code: 111)
            
        }
    }
}

//https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/op/scope.go#L124
extension Scope{
    func uniqueName(name:String)-> String {
        if let count = self.namemap[name], count > 0{
            return"\(name)_\(count)"
        }
        return name
        
    }
    
    func opName(type:String) -> String {
        if self.namespace == "" {
            return type
        }
        return self.namespace + "/" + type
    }
}

