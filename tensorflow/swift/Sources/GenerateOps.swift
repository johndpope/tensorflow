import CTensorFlow
import gRPCTensorFlow
import Foundation
import IOSwift
import ByteTools
import gRPCTensorFlow
import StencilSwiftKit
import Stencil
import Files



// WHY THESE ADDITIONAL Structs? 
// Needed to store properties for convenience with stencil kit template
struct MutableAttrDef{

    var name: String
    var type: String
    var defaultValue: Tensorflow_AttrValue
    var hasDefaultValue: Bool
    var description_p: String
    var hasMinimum_p: Bool
    var minimum: Int64
    var allowedValues: Tensorflow_AttrValue

    init(att:Tensorflow_OpDef.AttrDef) {
        self.name = att.name
        self.type = att.type
        self.defaultValue = att.defaultValue
        self.hasDefaultValue = att.hasDefaultValue
        self.description_p = att.description_p
        self.hasMinimum_p = att.hasMinimum_p
        self.minimum = att.minimum
        self.allowedValues = att.allowedValues
        
    }
}
struct MutableArgDef{

    var name: String = String()
    var description_p: String = String()
    var type: Tensorflow_DataType = Tensorflow_DataType.dtInvalid
    var typeAttr: String = String()
    var numberAttr: String = String()
    var typeListAttr: String = String()
    var isRef: Bool = false

    init(arg:Tensorflow_OpDef.ArgDef) {
        self.name = arg.name
        self.description_p = arg.description_p
        self.type = arg.type
        self.typeAttr = arg.typeAttr
        self.numberAttr = arg.numberAttr
        self.typeListAttr = arg.typeListAttr
        self.isRef = arg.isRef
    }
}
struct MutableTensorflow_OpDef{
    var jsonString:String? = String()
    var name: String = String()
    var inputArg: [MutableArgDef] = []
    var outputArg: [MutableArgDef] = []
    var attr: [MutableAttrDef] = []

    var summary: String = String()
    var description_p: String = String()
    var isCommutative: Bool = false
    var isAggregate: Bool = false
    var isStateful: Bool = false
    var allowsUninitializedInput: Bool = false
    
    var hasOutputArgs:Bool = false
    var hasOneOutputArg:Bool = false
    var hasNoOutputArg:Bool = false
    
    var hasAttributeOrInputArgs:Bool = false
    
    
    init(op:Tensorflow_OpDef) {
        self.jsonString = try?  op.jsonString()
        
        self.name = op.name
        var inputArrayArgs = Array<MutableArgDef>()
        
        for arg in op.inputArg{
            let mArg = MutableArgDef.init(arg:arg )
            inputArrayArgs.append(mArg)
        }
        
        self.inputArg = inputArrayArgs
        
        var outputArrayArgs = Array<MutableArgDef>()
        for arg in op.outputArg{
            let mArg = MutableArgDef.init(arg:arg )
            outputArrayArgs.append(mArg)
        }
        
        self.outputArg = outputArrayArgs
        
        var attArray = Array<MutableAttrDef>()
        for att in op.attr{
            let mAttr = MutableAttrDef.init(att:att )
            attArray.append(mAttr)
        }
        self.attr = attArray
        
        if(self.attr.count > 0){
            hasAttributeOrInputArgs = true
        }
        if(self.inputArg.count > 0){
            hasAttributeOrInputArgs = true
        }
        
        
        self.summary = op.summary
        self.description_p = op.description_p
        self.isCommutative = op.isCommutative
        self.isAggregate = op.isAggregate
        self.isStateful = op.isStateful
        self.allowsUninitializedInput =  op.allowsUninitializedInput
        
        
        if(self.outputArg.count > 0){
            if(self.outputArg.count == 1){
                self.hasOneOutputArg = true
            }else{
                self.hasOutputArgs = true
            }
        }else{
            self.hasNoOutputArg = true
        }
        
    }
    
}


class OperationsStencil{
    
    static var ops: [Tensorflow_OpDef] = []
    static var mops: [MutableTensorflow_OpDef] = []

    
    
    
    
    class func generateClasses(){
        
        let projectDir = "\(Folder.home.path)/Documents/tensorflowWorkspace/tensorflow/tensorflow/swift"
        let generatedFile = "Sources/GoOpWrapper.swift"
        
//        let opsTextFile = "\(Folder.home.path)/Documents/tensorflowWorkspace/tensorflow/tensorflow/core/ops/ops.pbtxt"
        let opsFile = "\(Folder.home.path)/Documents/tensorflowWorkspace/tensorflow/tensorflow/swift/misc/ops.pb"

        let stencilFile = "misc/OperationDefinitions.stencil"
        
        
        
        do {

            // This will correspond /align to go https://github.com/tensorflow/tensorflow/tree/master/tensorflow/go/genop
            // TODO - move this out of this library.
            //https://github.com/nubbel/swift-tensorflow/blob/master/RUNME.sh#L153
     
            //https://github.com/apple/swift-protobuf/issues/572#issuecomment-305857084 not working
//            let pbtxt = try String(contentsOfFile: opsFile, encoding: .utf8)
//            let msg = try? Tensorflow_OpList(textFormatString:pbtxt)
//            let opList = try? Tensorflow_OpList.init(seri:pbtxt)
            
            let opDefListData = try Data(contentsOf:URL(fileURLWithPath: opsFile))
            let opList = try? Tensorflow_OpList.init(serializedData:opDefListData)

            
            let stencilData = try Data(contentsOf:URL(fileURLWithPath: "\(projectDir)/\(stencilFile)"))
            if let stencilString = String(data: stencilData, encoding: String.Encoding.utf8){
                let template = StencilSwiftTemplate(templateString:stencilString,environment:stencilSwiftEnvironment())
                
                if let operations = opList?.op{
                     OperationsStencil.ops = operations
                    for op in operations{
                        let mOp = MutableTensorflow_OpDef(op: op)
                         OperationsStencil.mops.append(mOp)
                    }
                    updateOps()

                    let generated = try template.render(["operations": OperationsStencil.mops])
                    let newURL = URL(fileURLWithPath: projectDir + "/" + generatedFile)
                    try generated.data(using: .utf8)?.write(to: newURL)
                }
            }
        }catch {
            
            print("Project :",projectDir)
            print("error: \(error)")
            print("desc: \(error.localizedDescription)")
            exit(1)
        }
    }
    
    class  func updateOps(){
        var bShouldBreak = false
        for (index,op) in  OperationsStencil.ops.enumerated(){
            
            var str = op.description_p.replacingOccurrences(of: "*", with: " * ")
            str = str.replacingOccurrences(of: "\n", with: "\n// ")
            str = str.replacingOccurrences(of: "^", with: "// ^")
            OperationsStencil.ops[index].description_p = str
 
            
            if (bShouldBreak){
                bShouldBreak = false
                continue;
            }
            
            for (indexB,arg) in op.inputArg.enumerated(){
               print("arg:",arg)
                
            }
            print("op:",op)
            for (indexA,att) in op.attr.enumerated(){
                print(">",att.type)
               
                if (att.name == "T"){
                    if (att.type == "type"){
                        OperationsStencil.ops[index].attr[indexA].type = "Tensorflow_DataType"
                    }
//                    OperationsStencil.ops[index].attr.remove(at: indexA)
                    bShouldBreak = true
                    break;
                }
                
                if (att.name == "dtype"){
                    if (att.type == "type"){
                        OperationsStencil.ops[index].attr[indexA].type = "Tensorflow_DataType"
                    }else if(att.allowedValues.list.type.count > 0){
                      OperationsStencil.ops[index].attr[indexA].type = "[Any]"
                    }
                    
                }
               
                if (att.name == "type"){
                    OperationsStencil.ops[index].attr[indexA].type = "Tensorflow_DataType"
                    
                }
                
                if (att.type == "int"){
                    OperationsStencil.ops[index].attr[indexA].type = "UInt8"
                }else if (att.type == "bool"){
                    OperationsStencil.ops[index].attr[indexA].type = "Bool"
                }else if (att.type == "list(string)"){
                    OperationsStencil.ops[index].attr[indexA].type = "[Data]"
                }else if (att.type == "string"){
                    OperationsStencil.ops[index].attr[indexA].type = "String"
                }else if (att.type == "list(tensor)"){
                    OperationsStencil.ops[index].attr[indexA].type = "[Tensorflow_TensorProto]"
                }else if(att.type == "list(bool)"){
                    OperationsStencil.ops[index].attr[indexA].type = "[Bool]"
                }else if(att.type == "list(float)"){
                    OperationsStencil.ops[index].attr[indexA].type = "[Float]"
                }else if (att.type == "list(attr)"){
                    OperationsStencil.ops[index].attr[indexA].type = "[Tensorflow_NameAttrList]"
                }else if(att.type == "list(int)"){
                    OperationsStencil.ops[index].attr[indexA].type = "[Int64]"
                }else if (att.type == "list(type)"){
                    OperationsStencil.ops[index].attr[indexA].type = "[Tensorflow_DataType]"
                }else if(att.type == "list(shape)"){
                    OperationsStencil.ops[index].attr[indexA].type = "[Shape]"
                }
                
                
            }
            print("op:",OperationsStencil.ops[index].attr)

        }
    }
    
   
}
