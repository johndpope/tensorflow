/*
 * DO NOT EDIT.
 *
 * Generated by the protocol buffer compiler.
 * Source: tensorflow/core/protobuf/named_tensor.proto
 *
 */

import Foundation
import SwiftProtobuf

// If the compiler emits an error on this type, it is because this file
// was generated by a version of the `protoc` Swift plug-in that is
// incompatible with the version of SwiftProtobuf to which you are linking.
// Please ensure that your are building against the same version of the API
// that was used to generate this file.
fileprivate struct _GeneratedWithProtocGenSwiftVersion: SwiftProtobuf.ProtobufAPIVersionCheck {
  struct _1: SwiftProtobuf.ProtobufAPIVersion_1 {}
  typealias Version = _1
}

/// A pair of tensor name and tensor values.
public struct Tensorflow_NamedTensorProto: SwiftProtobuf.Message {
  public static let protoMessageName: String = _protobuf_package + ".NamedTensorProto"

  /// Name of the tensor.
  public var name: String {
    get {return _storage._name}
    set {_uniqueStorage()._name = newValue}
  }

  /// The client can populate a TensorProto using a tensorflow::Tensor`, or
  /// directly using the protobuf field accessors.
  ///
  /// The client specifies whether the returned tensor values should be
  /// filled tensor fields (float_val, int_val, etc.) or encoded in a
  /// compact form in tensor.tensor_content.
  public var tensor: Tensorflow_TensorProto {
    get {return _storage._tensor ?? Tensorflow_TensorProto()}
    set {_uniqueStorage()._tensor = newValue}
  }
  public var hasTensor: Bool {
    return _storage._tensor != nil
  }
  public mutating func clearTensor() {
    _storage._tensor = nil
  }

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    _ = _uniqueStorage()
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      while let fieldNumber = try decoder.nextFieldNumber() {
        switch fieldNumber {
        case 1: try decoder.decodeSingularStringField(value: &_storage._name)
        case 2: try decoder.decodeSingularMessageField(value: &_storage._tensor)
        default: break
        }
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      if !_storage._name.isEmpty {
        try visitor.visitSingularStringField(value: _storage._name, fieldNumber: 1)
      }
      if let v = _storage._tensor {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 2)
      }
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  fileprivate var _storage = _StorageClass()
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "tensorflow"

extension Tensorflow_NamedTensorProto: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "name"),
    2: .same(proto: "tensor"),
  ]

  fileprivate class _StorageClass {
    var _name: String = String()
    var _tensor: Tensorflow_TensorProto? = nil

    init() {}

    init(copying source: _StorageClass) {
      _name = source._name
      _tensor = source._tensor
    }
  }

  fileprivate mutating func _uniqueStorage() -> _StorageClass {
    if !isKnownUniquelyReferenced(&_storage) {
      _storage = _StorageClass(copying: _storage)
    }
    return _storage
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_NamedTensorProto) -> Bool {
    if _storage !== other._storage {
      let storagesAreEqual: Bool = withExtendedLifetime((_storage, other._storage)) { (_storage, other_storage) in
        if _storage._name != other_storage._name {return false}
        if _storage._tensor != other_storage._tensor {return false}
        return true
      }
      if !storagesAreEqual {return false}
    }
    if unknownFields != other.unknownFields {return false}
    return true
  }
}
