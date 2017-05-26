🚀 Swift Tensorflow port notes OSX
---------------------------

- The intention of this unofficial/ unsupported tensorflow swift port is to mirror the supported / golang tensorflow project. By not diverging from their implementation - it should be simpler to leverage the patches / fixes that google implement. This means - the golang codebase - if it has limitations in its design - github issues should be raised against them in tensorflow github issues. They would then be addressed - and fixed - then we can port to swift. 


    
There are some puzzles in porting golang code. It would be good to coordinate questions / issues / fixes.
If you have all the answers and want to take on fixing them - be my guest.  **LET'S COORDINATE EFFORTS HERE ->** 
https://docs.google.com/spreadsheets/d/1-B61huuIoKqyjS7dUb6GGZm5bt1gVcA58uxAKnojdDU/edit#gid=0 .  


- while dead code (commented out code) is frowned upon / while this code  is still in it's infancy - and not officially supported - I would urge anyone wanting to help to actually keep the golang code in code base along side the swift code. This would help people reviewing code down the track / as well as fix update code when the golang code is patched / reworked. 

eg.

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


-> 
write out the corresponding swift code .   

N.B. rather than calling TF_SessionRun -> Use the swift wrapper - > tfSessionRun()
Complete wrapper here -> 
https://github.com/johndpope/tensorflow/blob/swift/tensorflow/swift/Sources/TensorFlow.swift




C API
--------------------------------------------------
I've included a misc folder - it has a script to download and build a helloWorldTensor flow app.
This is a sanity check 

The c api references several proto files. Most notably error_codes.proto / types.proto and ConfigProto.
These currently live here -> 
https://github.com/johndpope/swift-grpc-tensorflow/tree/0.0.1
 



**IMPORTANT NOTES** 

The Makefile build swift has a bug in linking the library. 
The xcode project file that gets spat out - does successfully link.

- there's a CTensorflow swift package currently imported. 
It's hardcoded to use the osx path for tensorflow.
If you want to provide Linux access - happy to accept a PR.
https://github.com/johndpope/CTensorFlow/blob/master/CTensorFlow.h



- the Tensorflow.swift file was automatically generated using xcode in a copy and paste fashion. .   
Amazingly - you can create a new objective-c file in xcode and paste in all this c code - then click generate interface (it's hidden / google it if you need to). .    This has some caveats in that the c structs / and c objects are masked as opaquepointers. .   In the header of file - I've added several type alias to make the code less obsucre. .   
Eg.
 .   
Before  .   
    -public func tfAllocateTensor(dt: TF_DataType, _ dims: UnsafePointer<Int64>!, _ num_dims: Int32, _ len: Int) -> OpaquePointer!{

After
    +public func tfAllocateTensor(dt: TF_DataType, _ dims: UnsafePointer<Int64>!, _ num_dims: Int32, _ len: Int) -> TF_Tensor!{
        return TF_AllocateTensor(dt,dims,num_dims,len)
    }

the methods were renamed from TF_MethodName -> to tfMethodName .   
this avoids the c calls recursing into themselves and also makes things more swifty. .   
 .   
 .   
 .   
 .   
Golang code makes use of cgo which wraps things. .   
The Tensorflow.swift class is the swift equivalent wrapper.  .   
I'd urge anyone wanting to work on this library to use this umbrella wrapper when making calls out to c interface. .   
 .   
