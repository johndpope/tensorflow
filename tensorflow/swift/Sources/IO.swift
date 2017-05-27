//
//  Swift.IO
//
//  Created by Alec Thomas on 25/02/2015.
//  Copyright (c) 2015 SwapOff. All rights reserved.
//
//
// This file provides a consistent, simple interface to stream-based data sources.
// It is based on Go's I/O library.
//
// All implementations should be concurrent safe unless stated otherwise.
//
// Currently supported are: file descriptors, NSData, NSInputStream and NSOutputStream.

import Foundation

private let CEOF = EOF

// EOF is returned for an end of stream.
public let EOF = NSError(domain: "IO", code: 0, userInfo: ["localizedDescription": "EOF"])


public protocol Reader {
    // Read *at most* size bytes.
    func read(size: Int) -> (NSData, NSError?)
}

public protocol ByteReader {
    func readByte() -> (UInt8, NSError?)
}

public protocol ByteScanner: ByteReader {
    func unreadByte(b: UInt8) -> NSError?
}

public protocol Writer {
    // Write data. Returns the number of bytes written and any error,
    // including the reason for a short write, if any.
    func write(data: NSData) -> (Int, NSError?)
}

public protocol ReadWriter: Reader, Writer {}

public protocol Closer {
    func close() -> NSError?
}

public protocol ReadCloser: Reader, Closer {}
public protocol WriteCloser: Writer, Closer {}
public protocol ReadWriteCloser: ReadWriter, Reader, Writer, Closer {}


public enum SeekWhence: Int {
    case Start = 0
    case Current = 1
    case End = 2
}

// Implementations of this protocol provide a seek function.
public protocol Seeker {
    func seek(offset: Int, whence: SeekWhence) -> (Int, NSError?)
}

public protocol WriteSeeker: Writer, Seeker {}
public protocol ReadSeeker: Reader, Seeker {}
public protocol ReadWriteSeeker: ReadWriter, Seeker {}


// Converts a Reader into a ReadCloser with a no-op close() method.
public class NopCloser: ReadCloser {
    public var reader: Reader

    public init(_ reader: Reader) {
        self.reader = reader
    }

    public func read(size: Int) -> (NSData, NSError?) {
        return reader.read(size: size)
    }

    public func close() -> NSError? {
        return nil
    }
}


//public class BufferedReader: Reader, ByteReader {
//    private var r: Reader
//    private var byte: Byte?
//
//    public init(reader: Reader) {
//        self.r = reader
//    }
//
//    public func read(size: Int) -> (NSData, NSError?) {
//        if let b = byte {
//            var data = NSMutableData(capacity: size)!
//            data.bytes[0] = byte!
//        }
//    }
//
//    public func readByte() -> (Byte, NSError?) {
//        return (0, nil)
//    }
//}


public let DEFAULT_FILE_MODE: mode_t = S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP


// A ReadWriteCloser implementation for C file descriptors.
public class File: ReadWriteCloser, Seeker {
    private var lock = NSLock()
    // The underlying FD.
    private(set) public var fd: Int32
    // Filename (if known)
    private(set) public var name: String?

    // Create a File attached to an existing FD.
    public init(fd: CInt, name: String? = nil) {
        self.fd = fd
        self.name = name
    }

    // Open a file at the given path. See open(2).
    public class func open(path: String, oflag: CInt, mode: mode_t = DEFAULT_FILE_MODE) -> (File?, NSError?) {
        let fd = Darwin.open(path, oflag, mode)
        if fd == -1 {
            return (nil, getError())
        }
        return (File(fd: fd, name: path), nil)
    }

    // Create a new file at the given path. See creat(2).
    public class func create(path: String, mode: mode_t = DEFAULT_FILE_MODE) -> (File?, NSError?) {
        let fd = Darwin.creat(path, mode)
        if fd == -1 {
            return (nil, getError())
        }
        return (File(fd: fd), nil)
    }

    
   
    
    // Create a new temporary file.
    public class func temporary(mode: mode_t = DEFAULT_FILE_MODE) -> (File?, NSError?) {
        // The template string:
        let template = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("file.XXXXXX") as NSURL
        
        // Fill buffer with a C string representing the local file system path.
        var buffer = [Int8](repeating: 0, count: Int(PATH_MAX))
        template.getFileSystemRepresentation(&buffer, maxLength: buffer.count)
        
        // Create unique file name (and open file):
        let fd = mkstemp(&buffer)
        if fd != -1 {
            
            // Create URL from file system string:
            let url = NSURL(fileURLWithFileSystemRepresentation: buffer, isDirectory: false, relativeTo: nil)
            print(url.path!)
          
            if let name =   url.path{
                return (File(fd: fd, name: name), nil)
            }

            
        }
         print("Error: " + String(cString: strerror(errno)))
         return (nil, File.getError())
       
    }

    public func read(size: Int) -> (NSData, NSError?) {
        lock.lock()
        let buf = NSMutableData(length: size)!
        var err: NSError?
        var n = Darwin.read(fd, buf.mutableBytes, buf.length)
        if n == -1 {
            err = File.getError() 
            n = 0
        } else if n == 0 {
            err = EOF
        }
        buf.length = n
        lock.unlock()
        return (buf, err)
    }

    public func write(data: NSData) -> (Int, NSError?) {
        lock.lock()
        var err: NSError?
        var n = Darwin.write(fd, data.bytes, data.length)
        if n == -1 {
            err = File.getError() 
            n = 0
        }
        lock.unlock()
        return (Int(n), err)
    }

    // Close the File and set its fd to -1.
    public func close() -> NSError? {
        var err: NSError?
        lock.lock()
        if fd == -1 {
            err = NSError.newError("already closed",code:101)
        } else if Darwin.close(fd) == -1 {
            err = File.getError()
        }
        fd = -1
        lock.unlock()
        return err
    }

    public func seek(offset: Int, whence: SeekWhence) -> (Int, NSError?) {
        lock.lock()
        let offset = Int(Darwin.lseek(fd, off_t(offset), Int32(whence.rawValue)))
        lock.unlock()
        if offset == -1 {
            return (offset, File.getError())
        }
        return (offset, nil)
    }

    // Get last error from errno as an NSError.
    private class func getError() -> NSError {
        var message = String(cString:strerror(errno))
        return NSError.newError(message, code: Int(errno))
    }

    deinit {
        close()
    }

    private struct Static {
        static var stdin = File(fd: 0, name: "/dev/stdin")
        static var stdout = File(fd: 1, name: "/dev/stdout")
        static var stderr = File(fd: 2, name: "/dev/stderr")
    }

    public class var stdin: File { return Static.stdin }
    public class var stdout: File { return Static.stdout }
    public class var stderr: File { return Static.stderr }
}



// A Reader over NSData.
public class BufferReader: Reader, Seeker {
    let data: NSData
    // Cursor into NSData object.
    private(set) public var cursor = 0

    public init( _ data: NSData) {
        self.data = data
    }

    public func read(size: Int) -> (NSData, NSError?) {
        let count = size > data.length - cursor ? data.length - cursor : size
        let bytes = NSData(bytes: data.bytes + cursor, length: count)
        cursor += count
        return (bytes, nil)
    }

    public func seek(offset: Int, whence: SeekWhence) -> (Int, NSError?) {
        switch whence {
        case .Current:
            self.cursor += Int(offset)
        case .End:
            self.cursor = data.length + Int(offset)
        case .Start:
            self.cursor = Int(offset)
        }
        if self.cursor < 0 || self.cursor > self.data.length {
            return (-1, NSError.newError("invalid seek offset",code:100))
        }
        return (self.cursor, nil)
    }
}

// A Buffer provides read and write functions over raw bytes.
// Writes will grow the NSMutableData buffer.
public class Buffer: ReadWriter {
    private var lock = NSLock()
    private(set) public var data: NSMutableData
    private(set) public var cursor: Int = 0

    public init(data: NSMutableData) {
        self.data = data
    }

    public convenience init(capacity: Int) {
        self.init(data: NSMutableData(capacity: capacity)!)
    }

    public convenience init() {
        self.init(data: NSMutableData())
    }

    public func read(size: Int) -> (NSData, NSError?) {
        lock.lock()
        let count = size > data.length - cursor ? data.length - cursor : size
        let bytes = NSData(bytes: data.bytes + cursor, length: count)
        cursor += count
        lock.unlock()
        return (bytes, nil)
    }

    public func write(data: NSData) -> (Int, NSError?) {
        lock.lock()
        self.data.append(data as Data)
        let length = data.length
        lock.unlock()
        return (length, nil)
    }
}


// Dedicated IO thread for Swift.IO.
class IOThreadRunLoop: NSObject {
    private var ready =  DispatchSemaphore(value: 0)
    private var thread: Thread?
    internal var runloop: RunLoop = RunLoop()

    override init() {
        super.init()
        Thread.detachNewThreadSelector(#selector(IOThreadRunLoop.run), toTarget: self, with: nil)
        ready.wait(timeout: DispatchTime.distantFuture)
    }

    @objc private func run() {
        Thread.current.name = "Swift.IO"
        self.runloop = RunLoop.current
        ready.signal()
        while (true) {
            self.runloop.run(mode: RunLoopMode.defaultRunLoopMode, before: NSDate(timeIntervalSinceNow: 0.1) as Date)
        }
    }
}


// Global dedicated IO thread for Swift.IO.
internal var io = IOThreadRunLoop()


// Resumes a queue whenever events are sent on a stream.
public class StreamEventNotifier: NSObject, StreamDelegate {
    var ready = DispatchSemaphore(value: 0)

    public init(_ q: DispatchSemaphore) {
        self.ready = q
        super.init()
    }

    public func stream(theStream: Stream, handleEvent streamEvent: Stream.Event) {
        ready.signal()
    }
}


public class NSStreamNotifierMixin {
    private var ready = DispatchSemaphore(value: 0)
    private var notifier: StreamEventNotifier

    init() {
        notifier = StreamEventNotifier(ready)
    }
}


// IO.Reader adapter for an NSInputStream. The stream should NOT already be open.
public class NSInputStreamReader: ReadCloser {
   

   
    private var istream: InputStream

    public init(_ stream: InputStream) {
        self.istream = stream
        self.istream.open()
    }


    public func read(size: Int) -> (NSData, NSError?) {
        var bytes = UnsafeMutablePointer<UInt8>.allocate(capacity: size)
        let count = istream.read(bytes, maxLength: size)
        var data = NSData(bytes: bytes, length: count)
        if (count == 0){
            return (data,EOF)
        }else{
            return (data,istream.streamError as! NSError )
        }

    }
    deinit {
        close()
    }

    public func close() -> NSError? {
        istream.close()
        return istream.streamError as! NSError
    }
}


public class NSOutputStreamWriter: WriteCloser {
    private var ostream: OutputStream

    public init(_ stream: OutputStream) {
        self.ostream = stream
        self.ostream.open()
    }

    public func write(data: NSData) -> (Int, NSError?) {
        var p: UnsafePointer = data.bytes.assumingMemoryBound(to: UInt8.self)
        let count = ostream.write(p, maxLength: data.length)
        return (count, ostream.streamError as! NSError)
    }

    deinit {
        close()
    }

    public func close() -> NSError? {
        ostream.close()
        return ostream.streamError as! NSError
    }
}


public class ReadWriterFromReaderAndWriter: ReadWriter {
    var reader: Reader
    var writer: Writer

    public init(reader: Reader, writer: Writer) {
        self.reader = reader
        self.writer = writer
    }

    public func read(size: Int) -> (NSData, NSError?) {
        return reader.read(size: size)
    }

    public func write(data: NSData) -> (Int, NSError?) {
        return writer.write(data: data)
    }
}


// A ReadWriteCloser around an NSInputStream and NSOutputStream pair.
public class NSStreamReadWriteCloser: ReadWriterFromReaderAndWriter, ReadWriteCloser {
    var input: NSInputStreamReader
    var output: NSOutputStreamWriter

    public init(input: InputStream, output: OutputStream) {
        self.input = NSInputStreamReader(input)
        self.output = NSOutputStreamWriter(output)
        super.init(reader: self.input, writer: self.output)
    }

    deinit {
        close()
    }

    public func close() -> NSError? {
        let rerr = input.close()
        let werr = output.close()
        return rerr ?? werr
    }
}



// Copies from src to dst until either EOF is reached on src or an error occurs. 
// If EOF is reached it does not return an error.
public func Copy(dst: Writer, src: Reader, size: Int = Int.max) -> (Int, NSError?) {
    var written = 0
    while written < size {
        let chunk = min(size - written, 1024)
        let (data, err) = src.read(size: chunk)
        written += data.length
        if data.length != 0 {
            let (wn, werr) = dst.write(data: data)
            if werr != nil || wn < data.length {
                return (written + wn, werr)
            }
        } else if data.length == 0 || err != nil {
            if err == EOF {
                return (written, nil)
            }
            return (written, err)
        }
    }
    return (written, nil)
}


// Convenience construct for dealing with (Closer?, NSError?) return values.
//
// Use like so:
//
//   with (File.create("/tmp/foo.txt")) {file in
//     // Do something with "file"
//   }
//
// And to handle errors:
//
//   with (File.create("/tmp/foo.txt")) {file in
//     // Do something with "file"
//   }.error {err in
//     // .create() failed with err
//   }
public class with {
    private var err: NSError?

    public init<T: Closer>(_ tuple: (value: T?, err: NSError?), _ success: (T) -> Void) {
        self.err = tuple.err
        if let value = tuple.value {
            success(value)
            value.close()
        }
    }


    public func error<R>(f: (NSError) -> R) -> Self {
        if let err = err {
            f(err)
        }
        return self
    }
}

// public class PipeReader: ReadCloser {
//     private var ch: Channel<NSData>
// 
//     public init(ch: Channel<NSData>) {
//         self.ch = ch
//     }
// 
//     public func read(size: Int) -> (NSData, NSError?) {
//         if let data = <-ch {
//             return (data, nil)
//         }
//         return (NSData(), EOF)
//     }
// 
//     public func close() -> NSError? {
//         return nil
//     }
// }
// 
// 
// public class PipeWriter: WriteCloser {
//     private var ch: Channel<NSData>
// 
//     public init(ch: Channel<NSData>) {
//         self.ch = ch
//     }
// 
//     public func write(data: NSData) -> (Int, NSError?) {
//         ch <- data
//         return (data.length, nil)
//     }
// 
//     public func close() -> NSError? {
//         return nil
//     }
// }
// 
// 
// public func pipe() -> (PipeReader, PipeWriter) {
//     var ch = Channel<NSData>()
//     return (PipeReader(ch: ch), PipeWriter(ch: ch))
// }
