import CTensorFlow
import protoTensorFlow
import Foundation
import IOSwift
import ByteTools
import protoTensorFlow
import StencilSwiftKit
import Stencil


extension Tensorflow_OpDef{
    func hasOutputArgs()->Bool{
        if(self.outputArg.count > 0) {return true}
        return false
    }
    
}
class OperationsStencil{
    
    

    class func generateClasses(){
        
        do {

        
        let projectDir = "/Users/jp/Documents/tensorflowWorkspace/tensorflow/tensorflow/swift/misc"
        let opsFile = "ops.pb"
        let stencilFile = "OperationDefinitions.stencil"
        let generatedFile = "Operations.swift"
 
        // TODO - build out the tensorflow operations in swift
        // This will correspond /align to go https://github.com/tensorflow/tensorflow/tree/master/tensorflow/go/genop
        // TODO - move this out of this library.
        //https://github.com/nubbel/swift-tensorflow/blob/master/RUNME.sh#L153
        let opDefListData = try Data(contentsOf:URL(fileURLWithPath: "\(projectDir)/\(opsFile)"))
        let opList = try? Tensorflow_OpList.init(serializedData:opDefListData)
        
        let stencilData = try Data(contentsOf:URL(fileURLWithPath: "\(projectDir)/\(stencilFile)"))
        if let stencilString = String(data: stencilData, encoding: String.Encoding.utf8){
            let template = StencilSwiftTemplate(templateString:stencilString,environment:stencilSwiftEnvironment())
            if let operations = opList?.op{
                let generated = try template.render(["operations": operations])
                let newURL = URL(fileURLWithPath: projectDir + "/" + generatedFile)
                try generated.data(using: .utf8)?.write(to: newURL)
                for op in operations{
                    let name = try StringFilters.snakeToCamelCase(op.name)
                    print("name:",name)
                }
               
            }
        }
        }catch {
        
        
        print("error: \(error)")
        exit(1)
        }
    }
}
