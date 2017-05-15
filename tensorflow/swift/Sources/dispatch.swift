/*
* Dispatch (dispatch.swift) - Please be Safe
*
* Copyright (C) 2015 ONcast, LLC. All Rights Reserved.
* Created by Josh Baker (joshbaker77@gmail.com)
*
* This software may be modified and distributed under the terms
* of the MIT license.  See the LICENSE file for details.
*
* Portions of the documentation of this code are reproduced from
* work created and shared by Google and used according to terms
* described in the Creative Commons 3.0 Attribution License.
*
* http://golang.org/ref/spec
*/

#if os(Linux)
import Glibc
#endif

import Foundation

private class StrandClosure {
    let closure: () -> Void
    
    init(closure: @escaping () -> Void) {
        self.closure = closure
    }
}

#if os(Linux)
    private func runner(arg: UnsafeMutablePointer<Void>?) -> UnsafeMutablePointer<Void>? {
        guard let arg = arg else { return nil }
        let unmanaged = Unmanaged<StrandClosure>.fromOpaque(arg)
        unmanaged.takeUnretainedValue().closure()
        unmanaged.release()
        return nil
    }
#else
    private func runner(arg: UnsafeMutableRawPointer) -> UnsafeMutableRawPointer? {
        let unmanaged = Unmanaged<StrandClosure>.fromOpaque(arg)
        unmanaged.takeUnretainedValue().closure()
        unmanaged.release()
        return nil
    }
#endif

/// A `dispatch` statement starts the execution of an action as an independent concurrent thread of control within the same address space.
public func dispatch(_ action: @escaping () -> Void){
    let holder = Unmanaged.passRetained(StrandClosure(closure: action))
    let pointer = UnsafeMutableRawPointer(holder.toOpaque())
    #if os(Linux)
        var t : pthread_t = 0
        pthread_create(&t, nil, runner, pointer)
        pthread_detach(t)
    #else
        var pt : pthread_t?
        pthread_create(&pt, nil, runner, pointer)
        pthread_detach(pt!)
    #endif

}



