import CTensorFlow
import protoTensorFlow
import Foundation
import IOSwift
import ByteTools
import protoTensorFlow
import StencilSwiftKit
import Stencil
import Files

extension Tensorflow_OpDef{
    func hasOutputArgs()->Bool{
        if(self.outputArg.count > 0) {return true}
        return false
    }
    
}
class OperationsStencil{
    
    static var ops: [Tensorflow_OpDef] = []

    class func generateClasses(){
        
        let projectDir = "\(Folder.home.path)/Documents/tensorflowWorkspace/tensorflow/tensorflow/swift"
        let generatedFile = "Sources/GoOpWrapper.swift"
        
        let opsFile = "misc/ops.pb"
        let stencilFile = "misc/OperationDefinitions.stencil"
        
        
        
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
        var bShouldBreak = false
        for (index,op) in  OperationsStencil.ops.enumerated(){
            OperationsStencil.ops[index].description_p  = op.description_p.replacingOccurrences(of: "*/", with: "* /")
            OperationsStencil.ops[index].description_p  = op.description_p.replacingOccurrences(of: "\\*", with: "\\ *")
 
            
            if (bShouldBreak){
                bShouldBreak = false
                continue;
            }
            print("op:",op)
            for (indexA,att) in op.attr.enumerated(){
                print(">",att.type)
               
                if (att.name == "T"){
                    if (att.type == "type"){
                        OperationsStencil.ops[index].attr[indexA].type = "Tensorflow_DataType"
                    }
                    OperationsStencil.ops[index].attr.remove(at: indexA)
                    bShouldBreak = true
                    break;
                }
                
                if (att.name == "dtype"){
                    if(att.allowedValues.list.type.count > 0){
                      OperationsStencil.ops[index].attr[indexA].type = "[Any]"
                    }
                    
                }
               

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
                    OperationsStencil.ops[index].attr[indexA].type = "[Int64]"
                }
               
                if (att.type == "list(shape)"){
                    OperationsStencil.ops[index].attr[indexA].type = "[Shape]"
                }
                
                
            }
            print("op:",OperationsStencil.ops[index].attr)

        }
    }
    
   
}
