/*
Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/status.go
*/

import CTensorFlow
import protoTensorFlow


// status holds error information returned by TensorFlow. We convert all
// TF statuses to Go errors.
struct tfStatus  {
    var c:OpaquePointer!
    func errorMessage()->String?{
        return tfMessage(self.c)
    }
}

func newStatus() -> tfStatus {
    //	s = &status{TF_NewStatus()}
    let s = tfNewStatus()

//	runtime.SetFinalizer(s, (*status).finalizer) // TODO how to port to swift?
//https://github.com/reactive-swift/RunLoop/issues/11
//     let main = DispatchQueue.main
//    main.async() {
//        dispatch_set_finalizer_f(s,finalizer)
//    }
    
    var tfstatus = tfStatus()
    tfstatus.c = s
	return tfstatus
}




func  finalizer(s:tfStatus) {
	tfDeleteStatus(s.c)
}

func  code(s:tfStatus)-> TF_Code {
	return tfGetCode(s.c)
}

func string(s:tfStatus)-> String {
    return  tfMessage(s.c)
}


// Err converts the status to a Go error and returns nil if the status is OK.
/*func (s *status) Err() error {
	if s == nil || s.Code() == TF_OK {
		return nil
	}
	return (*statusError)(s)
}*/

/*

// statusError is distinct from status because it fulfills the error interface.
// status itself may have a TF_OK code and is not always considered an error.
//
// TODO(jhseu): Make public, rename to Error, and provide a way for users to
// check status codes.
type statusError status
*/

func error(s:tfStatus) -> Tensorflow_Error_Code {
    
    let code:TF_Code = tfGetCode(s.c)
    if let code = Tensorflow_Error_Code(rawValue: Int(code.rawValue)){
        return code
    }
    return Tensorflow_Error_Code(rawValue: 2)! //unknown
}
