import PackageDescription

let package = Package(
    name: "TensorFlow",
    targets: [
        Target(name: "TensorFlow")
    ],
    dependencies: [
         .Package(url: "https://github.com/johndpope/CTensorFlow.git", Version(0,0,2)),
         .Package(url: "https://github.com/apple/swift-protobuf.git", Version(0,9,902))
//         .Package(url: "https://github.com/grpc/grpc-swift.git", Version(0,1,10)),
  //       .Package(url: "https://github.com/johndpope/swift-grpc-tensorflow.git", majorVersion: 1)

         //  .Package(url: "https://github.com/eminarcissus/Safe.git", Version(1,2,1))
        
    ]
)
