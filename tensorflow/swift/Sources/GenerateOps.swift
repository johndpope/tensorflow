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
    
    static var ops: [Tensorflow_OpDef] = []

    class func generateClasses(){
        
        let projectDir = "/Users/jpope/Documents/tensorflowWorkspace/tensorflow/tensorflow/swift/misc"
        let opsFile = "ops.pb"
        let stencilFile = "OperationDefinitions.stencil"
        let generatedFile = "Operations.swift"
        
        
        do {

            // This will correspond /align to go https://github.com/tensorflow/tensorflow/tree/master/tensorflow/go/genop
            // TODO - move this out of this library.
            //https://github.com/nubbel/swift-tensorflow/blob/master/RUNME.sh#L153
            let opDefListData = try Data(contentsOf:URL(fileURLWithPath: "\(projectDir)/\(opsFile)"))
            let opList = try? Tensorflow_OpList.init(serializedData:opDefListData)
            
          //  print("opList:",opList)
            
            let stencilData = try Data(contentsOf:URL(fileURLWithPath: "\(projectDir)/\(stencilFile)"))
            if let stencilString = String(data: stencilData, encoding: String.Encoding.utf8){
                let template = StencilSwiftTemplate(templateString:stencilString,environment:stencilSwiftEnvironment())
                
                if let operations = opList?.op{
                    OperationsStencil.ops = operations
                    updateOps()
                    let generated = try template.render(["operations":  OperationsStencil.ops ])
                    let newURL = URL(fileURLWithPath: projectDir + "/" + generatedFile)
                    try generated.data(using: .utf8)?.write(to: newURL)
                }
            }
        }catch {
            
            print("Project :",projectDir)
            print("error: \(error)")
            exit(1)
        }
    }
    
    class  func updateOps(){
        for (index,op) in  OperationsStencil.ops.enumerated(){
            for (indexA,att) in op.attr.enumerated(){
                print(">",att.type)
                if (att.type == "int"){
                    OperationsStencil.ops[index].attr[indexA].type = "UInt8"
                }
                if (att.type == "bool"){
                    OperationsStencil.ops[index].attr[indexA].type = "Bool"
                }
                if (att.type == "string"){
                    OperationsStencil.ops[index].attr[indexA].type = "String"
                }
                if (att.type == "list(int)"){
                    OperationsStencil.ops[index].attr[indexA].type = "[Int]"
                }
                
                if (att.type == "list(shape)"){
                    OperationsStencil.ops[index].attr[indexA].type = "[Shape]"
                }
                
                
            }

        }
    }
    
   
}
