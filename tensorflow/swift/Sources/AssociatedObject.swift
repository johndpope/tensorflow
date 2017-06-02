import Foundation

public func associatedObject<T>(_ base: AnyObject, key: UnsafePointer<UInt8>, defaultValue: T) -> T {
    if let associated = objc_getAssociatedObject(base, key) as? T {
        return associated
    } else {
        associateObject(base, key: key, value: defaultValue)
        return defaultValue
    }
}

public func associatedObject<T>(_ base: AnyObject, key: UnsafePointer<UInt8>, initializer: () -> T) -> T {
    if let associated = objc_getAssociatedObject(base, key) as? T {
        return associated
    } else {
        let defaultValue = initializer()
        associateObject(base, key: key, value: defaultValue)
        return defaultValue
    }
}

public func associateObject<T>(_ base: AnyObject, key: UnsafePointer<UInt8>, value: T) {
    objc_setAssociatedObject(base, key, value, .OBJC_ASSOCIATION_RETAIN)
}
