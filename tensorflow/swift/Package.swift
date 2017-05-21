import PackageDescription

let package = Package(
    name: "TensorFlow",
    targets: [
        Target(name: "TensorFlow")
    ],
    dependencies: [
         .Package(url: "https://github.com/rxwei/CTensorFlow", majorVersion: 0),
         .Package(url: "https://github.com/apple/swift-protobuf.git", Version(0,9,902))

        
    ]
)
