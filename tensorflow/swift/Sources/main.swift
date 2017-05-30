import CTensorFlow
import protoTensorFlow
import Foundation
import CommandLineKit
import IOSwift
import ByteTools
import protoTensorFlow

// An example for using the TensorFlow Go API for image recognition
// using a pre-trained inception model (http://arxiv.org/abs/1512.00567).
//
// Sample usage: <program> -dir=/tmp/modeldir -image=/path/to/some/jpeg
//
// The pre-trained model takes input in the form of a 4-dimensional
// tensor with shape [ BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3 ],
// where:
// - BATCH_SIZE allows for inference of multiple images in one pass through the graph
// - IMAGE_HEIGHT is the height of the images on which the model was trained
// - IMAGE_WIDTH is the width of the images on which the model was trained
// - 3 is the (R, G, B) values of the pixel colors represented as a float.
//
// And produces as output a vector with shape [ NUM_LABELS ].
// output[i] is the probability that the input image was recognized as
// having the i-th label.
//
// A separate file contains a list of string labels corresponding to the
// integer indices of the output.
//
// This example:
// - Loads the serialized representation of the pre-trained model into a Graph
// - Creates a Session to execute operations on the Graph
// - Converts an image file to a Tensor to provide as input to a Session run
// - Executes the Session and prints out the label with the highest probability
//
// To convert an image file to a Tensor suitable for input to the Inception model,
// this example:
// - Constructs another TensorFlow graph to normalize the image into a
//   form suitable for the model (for example, resizing the image)
// - Creates an executes a Session to obtain a Tensor in this normalized form.*/

public typealias ShapeProto = Tensorflow_TensorShapeProto



print("Hello from TensorFlow C library version ",  tf.Version())

let cmdLine = CommandLineKit.CommandLine()
let dirFlag = StringOption(shortFlag: "d",
                          longFlag: "dir",
                          required: true,
                          helpMessage: "Directory containing the trained model files. ")
let imageFlag = StringOption(shortFlag: "i",
                           longFlag: "image",
                           required: true,
                           helpMessage: "Path of a JPEG-image to extract labels for")

cmdLine.addOptions(dirFlag,imageFlag)


do {
    try cmdLine.parse()
    
    print("dirFlag:",dirFlag.value!)

    // The two files are extracted from a zip archive as so:
    /*
		   curl -L https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip -o misc/inception5h.zip
		   unzip misc/inception5h.zip -d /misc
     */
    let modelFile  = "tensorflow_inception_graph.pb"
    let imagefile = imageFlag.value! // change in the edit scheme arguments passed on launch
    
    // Load the serialized GraphDef from a file.
    let model = try Data(contentsOf:URL(fileURLWithPath: "\(dirFlag.value!)/\(modelFile)"))
    
    
    // Construct an in-memory graph from the serialized form.
    let graph = newGraph()
    if let error = graph.Import(def:model.cBytes(),prefix: "OK"){
        print(error)
    }
    
    // Create a session for inference over graph.
    var opts:SessionOptions = SessionOptions()
    var config = Tensorflow_ConfigProto()
    config.operationTimeoutInMs = 2
    opts.ConfigProto = config
    
    // Tensor
    var tensor = Tensorflow_TensorProto()
    // TODO make this setter infer dType automagically by introspecting setter.
    tensor.dtype = .dtInt64
    tensor.int64Val = [Int64(0.0),Int64(1.0),Int64(1.0),Int64(1.0)]
  
    // Shape
    var shape = ShapeProto()
    shape.unknownRank = false
    
    /// The order of entries in "dim" matters: It indicates the layout of the
    /// values in the tensor in-memory representation.
    var dimensions :[ShapeProto.Dim] = []
    
    // Rows
    var H = ShapeProto.Dim()
    H.size = 224
    H.name  = "height"
    dimensions.append(H)
    
    // Columns
    var W = ShapeProto.Dim()
    W.size = 224
    W.name  = "width"
    dimensions.append(W)

    // Colors
    var colors = ShapeProto.Dim()
    colors.size = 3
    colors.name  = "colors"
    dimensions.append(colors)
    
    shape.dim = dimensions
    tensor.tensorShape = shape
    
    
    
    
    
    var (session, error) = newSession( graph,opts)
    
    if error != nil {
        print(error.debugDescription)
    }
   
    
//     Run inference on *imageFile.
//     For multiple images, session.Run() can be called in a loop (and
//     concurrently). Alternatively, images can be batched since the model
//     accepts batches of image data as input.
/*    var tensor:Tensor
    (tensor, error) = makeTensorFromImage(imagefile)
    
    if err != nil {
        log.Fatal(err)
    }
    output, err = session.Run(
    map[tf.Output]*tf.Tensor{
    graph.Operation("input").Output(0): tensor,
    },
    []tf.Output{
    graph.Operation("output").Output(0),
    },
    nil)
    if err != nil {
        log.Fatal(err)
    }
     output[0].Value() is a vector containing probabilities of
     labels for each image in the "batch". The batch size was 1.
     Find the most probably label index.
    probabilities = output[0].Value().([][]float32)[0]
    printBestLabel(probabilities, labelsfile)

    
}catch {

    
    print("error: \(error)")
    exit(1)
}
*/


// Convert the image in filename to a Tensor suitable as input to the Inception model.
//https://github.com/ctava/tensorflow-go-imagerecognition/blob/ffce1d23cb7f4194a38023eeaf25632553ca483c/main.go#L133
func makeTensorFromImage(filename :String)-> (TF_Tensor?, NSError?) {
    
    let photo = try! Data(contentsOf:URL(fileURLWithPath: "\(dirFlag.value!)/\(imageFlag.value!)"))
    
    // DecodeJpeg uses a scalar String-valued tensor as input.
//    let tensor = newTensor(photo.cBytes())
//    if err != nil {
//        return nil, err
//    }
//    // Construct a graph to normalize the image
//    graph, input, output, err = constructGraphToNormalizeImage()
//    if err != nil {
//        return nil, err
//    }
//    // Execute that graph to normalize this one image
//    session, err = tf.NewSession(graph, nil)
//    if err != nil {
//        return nil, err
//    }
//    defer session.Close()
//    normalized, err = session.Run(
//    map[tf.Output]*tf.Tensor{input: tensor},
//    []tf.Output{output},
//    nil)
//    if err != nil {
//        return nil, err
//    }
//    return normalized[0], nil
     return (nil, nil)
}

// The inception model takes as input the image described by a Tensor in a very
// specific normalized format (a particular image size, shape of the input tensor,
// normalized pixel values etc.).
//
// This function constructs a graph of TensorFlow operations which takes as
// input a JPEG-encoded string and returns a tensor suitable as input to the
// inception model.
//func constructGraphToNormalizeImage() -> (graph :TF_Graph, input:TF_Input, output:TF_Output, err:NSError?) {
    // Some constants specific to the pre-trained model at:
    // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
    //
    // - The model was trained after with images scaled to 224x224 pixels.
    // - The colors, represented as R, G, B in 1-byte each were converted to
    //   float using (value - Mean)/Scale.
    /*let const (
        H, W  = 224, 224
        Mean  = float32(117)
        Scale = float32(1)
    )*/
    // - input is a String-Tensor, where the string the JPEG-encoded image.
    // - The inception model takes a 4D tensor of shape
    //   [BatchSize, Height, Width, Colors=3], where each pixel is
    //   represented as a triplet of floats
    // - Apply normalization on each pixel and use ExpandDims to make
    //   this single image be a "batch" of size 1 for ResizeBilinear.
   /* let s = op.NewScope()
    let input = op.Placeholder(s, tf.String)
    output = op.Div(s,
    op.Sub(s,
    op.ResizeBilinear(s,
				op.ExpandDims(s,
    op.Cast(s,
    op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)), tf.Float),
    op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{H, W})),
    op.Const(s.SubScope("mean"), Mean)),
    op.Const(s.SubScope("scale"), Scale))
    graph, err = s.Finalize()
    return graph, input, output, err*/

//}

}


/*
 
 func printBestLabel(probabilities []float32, labelsFile string) {
 bestIdx = 0
 for i, p = range probabilities {
 if p > probabilities[bestIdx] {
 bestIdx = i
 }
 }
 // Found the best match. Read the string from labelsFile, which
 // contains one line per label.
 file, err = os.Open(labelsFile)
 if err != nil {
 log.Fatal(err)
 }
 defer file.Close()
 scanner = bufio.NewScanner(file)
 var labels []string
 for scanner.Scan() {
 labels = append(labels, scanner.Text())
 }
 if err = scanner.Err(); err != nil {
 log.Printf("ERROR: failed to read %s: %v", labelsFile, err)
 }
 fmt.Printf("BEST MATCH: (%2.0f%% likely) %s\n", probabilities[bestIdx]*100.0, labels[bestIdx])
 }*/
