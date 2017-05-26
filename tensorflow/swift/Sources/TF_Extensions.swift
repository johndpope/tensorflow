import Foundation
import Darwin.C.stddef
import Darwin.C.stdint
import CTensorFlow
import protoTensorFlow

// Helper to encourage use of the proto Tensorflow_DataType type instead of c primitive
public func tfTensorType(_ pointer:TF_Tensor!) -> Tensorflow_DataType{
    
    let ptr = TF_TensorType(pointer)
    let int8Ptr = unsafeBitCast(ptr, to: Int.self)
    if let dt = Tensorflow_DataType(rawValue: int8Ptr){
        return dt
    }
    return  Tensorflow_DataType(rawValue: 2)! //unknown

}

public func tfNewTensor(dt: Tensorflow_DataType, _ dims: UnsafePointer<Int64>!, _ num_dims: Int32, _ data: UnsafeMutableRawPointer!, _ len: Int, _ deallocator: (@convention(c) (UnsafeMutableRawPointer?, Int, UnsafeMutableRawPointer?) -> Swift.Void)!, _ deallocator_arg: UnsafeMutableRawPointer!) -> TF_Tensor!{
    let cDt = dt.rawValue
    let nativeDt = unsafeBitCast(cDt, to: TF_DataType.self)
    
    return TF_NewTensor(nativeDt,dims, num_dims, data, len, deallocator, deallocator_arg)
    
}
