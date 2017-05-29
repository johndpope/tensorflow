🚀 Swift Tensorflow port notes OSX
---------------------------

- The intention of this unofficial/ unsupported tensorflow swift port is to mirror the supported / golang tensorflow project. By not diverging from their implementation - it should be simpler to leverage the patches / fixes that google implement. This means - the golang codebase - if it has limitations in its design - github issues should be raised against them in tensorflow github issues. They would then be addressed - and fixed - then we can port to swift. 


    
There are some puzzles in porting golang code. It would be good to coordinate questions / issues / fixes.
If you have all the answers and want to take on fixing them - be my guest.  **LET'S COORDINATE EFFORTS HERE ->** 
https://docs.google.com/spreadsheets/d/1-B61huuIoKqyjS7dUb6GGZm5bt1gVcA58uxAKnojdDU/edit#gid=0 .  


for each method - perhaps where appropriate hot link to respective golang class / line on github.


//https://github.com/johndpope/tensorflow-1/blob/master/tensorflow.go#L33

              extension SessionOptions{
               func setConfig(config:Tensorflow_ConfigProto){
                  let status = newStatus()

                defer{
                    tf.DeleteStatus(status.c)
                }

                if let data = try? config.serializedData(){
                    //https://stackoverflow.com/questions/39671789/in-swift-3-how-do-i-get-unsaferawpointer-from-data
                    data.withUnsafeBytes {(uint8Ptr: UnsafePointer<UInt8>) in
                        let rawPtr = UnsafeRawPointer(uint8Ptr)
                        tf.SetConfig(self.c, rawPtr, data.count, status.c)
                    }
                }

                if let error =  status.error(){
                    print("error:",error.localizedDescription)
                }
            }
        }
   

N.B. rather than calling TF_SessionRun -> Use the swift wrapper - > tf.SessionRun()
Complete wrapper here ->  TODO - make relative links
https://github.com/johndpope/tensorflow/blob/swift/tensorflow/swift/Sources/TensorFlow.swift
https://github.com/johndpope/tensorflow/blob/swift/tensorflow/swift/Sources/TF_Buffer.swift
https://github.com/johndpope/tensorflow/blob/swift/tensorflow/swift/Sources/TF_Graph.swift
https://github.com/johndpope/tensorflow/blob/swift/tensorflow/swift/Sources/TF_Session.swift
https://github.com/johndpope/tensorflow/blob/swift/tensorflow/swift/Sources/TF_Status.swift
https://github.com/johndpope/tensorflow/blob/swift/tensorflow/swift/Sources/TF_String.swift
https://github.com/johndpope/tensorflow/blob/swift/tensorflow/swift/Sources/TF_Tensor.swift



C API
--------------------------------------------------
I've included a misc folder - it has a script to download and build a helloWorldTensor flow app.
This is a sanity check. helloWorld should work. 

The c api references several proto files. 
Most notably error_codes.proto / types.proto and ConfigProto.
These currently live here -> 
https://github.com/johndpope/swift-grpc-tensorflow/tree/0.0.1
 



**IMPORTANT NOTES** 

The Makefile build swift has a bug in linking the library. 
The xcode project file that gets spat out - does successfully link.

- there's a CTensorflow swift umbrella package currently imported. 
It's hardcoded to use the osx path for tensorflow.
If you want to provide Linux access - fork / fix and change the tag in package.
https://github.com/johndpope/CTensorFlow/blob/master/CTensorFlow.h



- the Tensorflow.swift file was automatically generated using xcode in a copy and paste fashion. .   
Amazingly - you can create a new objective-c file in xcode and paste in all this c code - then click generate interface (it's hidden / google it if you need to). .    This has some caveats in that the c structs / and c objects are masked as opaquepointers. .   In the header of file - I've added several type alias to make the code less obsucre. .   
Eg.
 .   
Before  .   
    -public func TF_AllocateTensor(dt: TF_DataType, _ dims: UnsafePointer<Int64>!, _ num_dims: Int32, _ len: Int) -> OpaquePointer!{

After

class tf{

    +public class func AllocateTensor(dt: TF_DataType, _ dims: UnsafePointer<Int64>!, _ num_dims: Int32, _ len: Int) -> TF_Tensor!{
        return TF_AllocateTensor(dt,dims,num_dims,len)
    }
}
the methods were renamed from TF_MethodName -> to tf.MethodName .   
this avoids the c calls recursing into themselves and also makes things more swifty. .   
 .   
 .   
 .   
 .   
 .   
