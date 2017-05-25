import CTensorFlow


let myGraph:tfGraph = tfGraph()
let myConfig:Tensorflow_ConfigProto  = Tensorflow_ConfigProto()


let opts:tfSessionOptions = tfSessionOptions(Target:"",Config:myConfig)



var (mySession,error) = newSession(graph:myGraph,options:opts)

print("Hello from TensorFlow C library version ", tfVersion())

