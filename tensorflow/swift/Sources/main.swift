import CTensorFlow


let graph:OpaquePointer! = nil
let options = TF_NewSessionOptions()
let status:OpaquePointer! = nil

let session = tfNewSession(graph, options, status)
