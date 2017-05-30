/*
Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 
 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/saved_model.go
*/

import CTensorFlow
import Foundation


// SavedModel represents the contents of loaded SavedModel.
// TODO(jhseu): Add and document metagraphdef when we pre-generate protobufs.
struct SavedModel  {
    var Session:Session
    var Graph:Graph
    
    public init(_ Session:Session,_ Graph:Graph){
        self.Session = Session
        self.Graph = Graph
    }
}

// LoadSavedModel creates a new SavedModel from a model previously
// exported to a directory on disk.
//
// Exported models contain a set of graphs and, optionally, variable values.
// Tags in the model identify a single graph. LoadSavedModel initializes a
// session with the identified graph and with variables initialized to from the
// checkpoints on disk.
//
// The tensorflow package currently does not have the ability to export a model
// to a directory from Go. This function thus currently targets loading models
// exported in other languages, such as using tf.saved_model.builder in Python.
// See:
 //https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#tags

func LoadSavedModel(exportDir:String, tags:[String?], options:SessionOptions) -> (SavedModel?, NSError?) {
	let status = newStatus()
    var cOpt:TF_SessionOptions?, doneOpt:SessionDoneClosure, error:NSError?;
    (cOpt, doneOpt, error) = options.c()
 
    defer {
        doneOpt()
    }
    
	if error != nil {
		return (nil, error)
	}
    
    if let cExportDir:[CChar] = exportDir.cString(using: .utf8){

        let graph = newGraph()
        let cTags = tags.map { $0.flatMap { UnsafePointer<Int8>(strdup($0)) } }

        
        if let cSession = tf.LoadSessionFromSavedModel(cOpt, nil, cExportDir, cTags, Int32(cTags.count), graph.c, nil, status.c){
            let s = Session(c:cSession)
            
            //runtime.SetFinalizer(s, func(s *Session) { s.Close() })
            let savedModel = SavedModel( s,  graph)
            return (savedModel, nil)
        }
        

        if let error = status.error() {
            return (nil, error)
        }
        
    }
    
}
