/*
 * DO NOT EDIT.
 *
 * Generated by the protocol buffer compiler.
 * Source: tensorflow/contrib/cloud/kernels/bigquery_table_partition.proto
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

/// This proto specifies a table partition in BigQuery.
public struct Tensorflow_BigQueryTablePartition: SwiftProtobuf.Message {
  public static let protoMessageName: String = _protobuf_package + ".BigQueryTablePartition"

  /// [start_index, end_index] specify the boundaries of a partition.
  /// If end_index is -1, every row starting from start_index is part of the
  /// partition.
  public var startIndex: Int64 = 0

  public var endIndex: Int64 = 0

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularInt64Field(value: &self.startIndex)
      case 2: try decoder.decodeSingularInt64Field(value: &self.endIndex)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.startIndex != 0 {
      try visitor.visitSingularInt64Field(value: self.startIndex, fieldNumber: 1)
    }
    if self.endIndex != 0 {
      try visitor.visitSingularInt64Field(value: self.endIndex, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "tensorflow"

extension Tensorflow_BigQueryTablePartition: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "start_index"),
    2: .standard(proto: "end_index"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_BigQueryTablePartition) -> Bool {
    if self.startIndex != other.startIndex {return false}
    if self.endIndex != other.endIndex {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}
