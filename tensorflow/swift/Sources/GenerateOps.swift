import CTensorFlow
import protoTensorFlow
import Foundation
import IOSwift
import ByteTools
import protoTensorFlow
import StencilSwiftKit
import Stencil
import Files

// hack to allow stencil to correctly generate func types
public typealias TensorflowNameAttrList = Tensorflow_NameAttrList
public typealias TensorflowDataType = Tensorflow_DataType
public typealias TensorflowTensorShapeProto = Tensorflow_TensorShapeProto
public typealias TensorflowTensorProto = Tensorflow_TensorProto



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

        self.jsonString =  op.debugDescription
       
      
        
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
            var mAttr = MutableAttrDef.init(att:att )
            if(att.name == "t"){
                mAttr.name = "dataType"
            }
            
            if(att.type == "list(string)"){
                mAttr.type = "[Data]"
            }else if(att.type == "list(int)"){
                mAttr.type = "[Int64]"
            }else if(att.type == "list(float)"){
                mAttr.type = "[Float]"
            }else if(att.type == "list(bool)"){
                mAttr.type = "[Bool]"
            }else if(att.type == "list(type)"){
                mAttr.type = "[Tensorflow_DataType]"
            }else if(att.type == "list(shape)"){
                mAttr.type = "[Tensorflow_TensorShapeProto]"
            }else if(att.type == "list(tensor)"){
                mAttr.type = "[Tensorflow_TensorProto]"
            }else if(att.type == "list(attr)"){
                mAttr.type = "[Tensorflow_NameAttrList]"
            }
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
    
    //static var ops: [Tensorflow_OpDef] = []
    static var ops: [MutableTensorflow_OpDef] = []

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

                    for op in operations{
                        let mOp = MutableTensorflow_OpDef(op: op)
                         OperationsStencil.ops.append(mOp)
                    }
                    updateOps()

                    let generated = try template.render(["operations": OperationsStencil.ops])
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
 
           
            if (op.name.lowercased() == "where") {
               OperationsStencil.ops[index].name = "where_p"
            }else  if (op.name.lowercased() == "switch") {
               OperationsStencil.ops[index].name =  "switch_p"
            }
            
            
            if (bShouldBreak){
                bShouldBreak = false
                continue;
            }
            
            var allowedTypes:Tensorflow_AttrValue = Tensorflow_AttrValue()
            for (idx,arg) in op.inputArg.enumerated(){
                
                // DETERMINE THE ALLOWED TYPES
                 for (indexB,att) in op.attr.enumerated(){
                    if (att.name == "T"){
                        if(att.type == "type"){
                            allowedTypes = att.allowedValues
                        }
                    }
                }

                 print("allowedValues:",allowedTypes)
               print("arg:",arg)
                if (arg.type == Tensorflow_DataType.dtInvalid){
                    print("ok")
                    if (arg.typeAttr ==  "T"){
                        OperationsStencil.ops[idx].inputArg[idx].type = .dtFloat
                    }
                }

            }
            
            for (indexA,att) in op.attr.enumerated(){
                print("attr:",att)
             

                if let v = att.defaultValue.value{
                    OperationsStencil.ops[index].attr[indexA].type = "\(OperationsStencil.ops[index].attr[indexA].type) =\(v)"
                }
                if (att.name == "T"){
                    if (att.type == "type"){
                        OperationsStencil.ops[index].attr[indexA].type = "Tensorflow_DataType"
                    }
                    bShouldBreak = true
                    break;
                }else if (att.name == "dtype"){
                    if (att.type == "type"){
                        OperationsStencil.ops[index].attr[indexA].type = "Tensorflow_DataType"
                    }else if(att.allowedValues.list.type.count > 0){
                      OperationsStencil.ops[index].attr[indexA].type = "[Any]"
                    }
                }else if (att.type == "func"){
                    OperationsStencil.ops[index].attr[indexA].type = "Tensorflow_NameAttrList"
                }else if (att.name == "type"){
                    OperationsStencil.ops[index].attr[indexA].type = "Tensorflow_DataType"
                }else  if (att.type == "int"){
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
           

        }
         print("ops:",OperationsStencil.ops)
    }
    
   
}
