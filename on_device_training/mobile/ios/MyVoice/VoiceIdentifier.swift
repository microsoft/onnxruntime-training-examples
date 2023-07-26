import Foundation
import onnxruntime_training_objc

class VoiceIdentifier {
    
    private let ortEnv : ORTEnv
    private let ortSession: ORTSession
    private let kThresholdProbability: Float = 0.80
    
    enum VoiceIdentifierError: Error {
        case Error(_ message: String)
    }
    
    init() throws {
        ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)

        guard let libraryDirectory = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first else {
            throw VoiceIdentifierError.Error("Failed to find library directory ")
        }
        let modelPath = libraryDirectory.appendingPathComponent("inference_model.onnx").path

        if !FileManager.default.fileExists(atPath: modelPath) {
            throw VoiceIdentifierError.Error("Failed to find inference model file.")
        }
        ortSession = try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: nil)
    }
    
    private func isUser(logits: [Float]) -> Float {
        // apply softMax
        let maxInput = logits.max() ?? 0.0
        let expValues = logits.map { exp($0 - maxInput) } // Calculate e^(x - maxInput) for each element
        let expSum = expValues.reduce(0, +) // Sum of all e^(x - maxInput) values
        
        return expValues.map { $0 / expSum }[1] // Calculate the softmax probabilities
    }
    
    func evaluate(inputData: Data) -> Result<(Bool, Float), Error> {
        
        return Result<(Bool, Float), Error> { () -> (Bool, Float) in
            
            let inputShape: [NSNumber] = [1, inputData.count / MemoryLayout<Float>.stride as NSNumber]
            
            let input = try ORTValue(
                tensorData: NSMutableData(data: inputData),
                elementType: ORTTensorElementDataType.float,
                shape: inputShape)
            
            let outputs = try ortSession.run(
                withInputs: ["input": input],
                outputNames: ["output"],
                runOptions: nil)
            
            guard let output = outputs["output"] else {
                throw VoiceIdentifierError.Error("Failed to get model output from inference.")
            }
            
            let outputData = try output.tensorData() as Data
            let probUser = outputData.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) -> Float in
                let floatBuffer = buffer.bindMemory(to: Float.self)
                let logits = Array(UnsafeBufferPointer(start: floatBuffer.baseAddress, count: outputData.count/MemoryLayout<Float>.stride))
                return isUser(logits: logits)
            }
            
            return (probUser >= kThresholdProbability, probUser)
        }
    }
}
