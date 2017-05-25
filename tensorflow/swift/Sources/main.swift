import CTensorFlow
import protoTensorFlow



let myGraph:Graph = Graph()
let opts:SessionOptions = SessionOptions()

var (mySession,error) = newSession(graph:myGraph,options:opts)

print("Hello from TensorFlow C library version ", tfVersion())

