import Foundation

extension String {
    
    var pointer: UnsafePointer<Int8> {
        return withCString { (ptr) -> UnsafePointer<Int8> in
            return ptr
        }
}
