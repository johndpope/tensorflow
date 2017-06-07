import PackageDescription

let package = Package(
    name: "TensorFlow",
    targets: [
        Target(name: "TensorFlow")
    ],
    dependencies: [
         .Package(url: "https://github.com/johndpope/CTensorFlow.git", Version(0,0,2)),
         .Package(url: "https://github.com/apple/swift-protobuf.git", Version(0,9,903)),
         .Package(url: "https://github.com/johndpope/IO-swift.git", Version(0,0,3)),
         .Package(url: "https://github.com/harlanhaskins/CommandLine.git", majorVersion: 3),
         .Package(url: "https://github.com/johndpope/ByteTools-swift", Version(0,0,4)),
         .Package(url: "https://github.com/johndpope/StencilSwiftKit.git", Version(1,0,1)),
         .Package(url: "https://github.com/JohnSundell/Files.git", Version(1,8,0)),
         .Package(url: "https://github.com/johndpope/swift-grpc-tensorflow.git", Version(2,0,1))
    ]
)
