import CTensorFlow


let myGraph:tfGraph = tfGraph()
let myConfig:OpaquePointer! = nil
let opts:tfSessionOptions = tfSessionOptions(Target:"",Config:myConfig)



var (mySession,error) = newSession(graph:myGraph,options:opts)
