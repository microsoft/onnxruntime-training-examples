import Foundation
import AVFoundation
import onnxruntime_training_objc

class Trainer {
    private let ortEnv: ORTEnv
    private let trainingSession: ORTTrainingSession
    private let checkpoint: ORTCheckpoint
    private var recordingCounter: Int
    private let sampleAudioRecordings: Int
    private let kEpoch: Int = 3
    
    let kUserIndex: Int64 = 1
    let kOtherIndex: Int64 = 0
    
    enum TrainerError: Error {
        case Error(_ message: String)
    }
    
    init(sampleAudioRecordings : Int) throws {
        ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
        
        // get path for artifacts
        guard let trainingModelPath = Bundle.main.path(forResource: "training_model", ofType: "onnx") else {
            throw TrainerError.Error("Failed to find training model file.")
        }
        
        guard let evalModelPath = Bundle.main.path(forResource: "eval_model",ofType: "onnx") else {
            throw TrainerError.Error("Failed to find eval model file.")
        }
        
        guard let optimizerPath = Bundle.main.path(forResource: "optimizer_model", ofType: "onnx") else {
            throw TrainerError.Error("Failed to find optimizer model file.")
        }
        
        guard let checkpointPath = Bundle.main.path(forResource: "checkpoint", ofType: nil) else {
            throw TrainerError.Error("Failed to find checkpoint file.")
        }
        
        checkpoint = try ORTCheckpoint(path: checkpointPath)
        
        trainingSession = try ORTTrainingSession(env: ortEnv, sessionOptions: ORTSessionOptions(), checkpoint: checkpoint, trainModelPath: trainingModelPath, evalModelPath: evalModelPath, optimizerModelPath: optimizerPath)
        
        self.sampleAudioRecordings = sampleAudioRecordings
        recordingCounter = 0
    }
    
    func exportModelForInference() throws {
        guard let libraryDirectory = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first else {
            throw TrainerError.Error("Failed to find library directory ")
        }
        
        let modelPath = libraryDirectory.appendingPathComponent("inference_model.onnx").path
        try trainingSession.exportModelForInference(withOutputPath: modelPath, graphOutputNames: ["output"])
    }
    
    func train(audio: Data)  -> Result<Void, Error> {
        return Result<Void, Error> { ()  in
            for _ in 0..<kEpoch {
                let (buffer, wavFileData) = try getDataFromWavFile(fileName: "other_\(recordingCounter)")
                try trainStep(inputData: [audio, wavFileData], label: [kUserIndex, kOtherIndex])
            }
            
            recordingCounter = min(recordingCounter + 1, sampleAudioRecordings - 1)
        }
    }
    
    func trainStep(inputData: [Data], label: [Int64]) throws  {
        
        let inputs = [try getORTValue(dataList: inputData), try getORTValue(lables: label)]
        try trainingSession.trainStep(withInputValues: inputs)
        
        // update the model params
        try trainingSession.optimizerStep()
        
        // reset the gradients
        try trainingSession.lazyResetGrad()
    }
    
    private func getORTValue(dataList: [Data]) throws -> ORTValue {
        let tensorData = NSMutableData()
        dataList.forEach {data in tensorData.append(data)}
        let inputShape: [NSNumber] = [dataList.count as NSNumber, dataList[0].count / MemoryLayout<Float>.stride as NSNumber]
        
        return try ORTValue(
            tensorData: tensorData, elementType: ORTTensorElementDataType.float, shape: inputShape
        )
    }
    
    private func getORTValue(lables: [Int64]) throws -> ORTValue {
        let tensorData = NSMutableData(bytes: lables, length: lables.count * MemoryLayout<Int64>.stride)
        let inputShape: [NSNumber] = [lables.count as NSNumber]
        
        return try ORTValue (
            tensorData: tensorData, elementType: ORTTensorElementDataType.int64, shape: inputShape
        )
    }
    
    private func getDataFromWavFile(fileName: String) throws -> (AVAudioBuffer, Data) {
        guard let fileUrl = Bundle.main.url(forResource: fileName, withExtension:"wav") else {
            throw TrainerError.Error("Failed to find wav file: \(fileName).")
        }
        
        let audioFile = try AVAudioFile(forReading: fileUrl)
        
        let format = audioFile.processingFormat
        
        let totalFrames = AVAudioFrameCount(audioFile.length)
        
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: totalFrames) else {
            throw TrainerError.Error("Failed to create audio buffer.")
        }
        
        try audioFile.read(into: buffer)
        
        guard let floatChannelData = buffer.floatChannelData else {
            throw TrainerError.Error("Failed to get float channel data.")
        }
        
        let data = Data(
            bytesNoCopy: floatChannelData[0],
            count: Int(buffer.frameLength) * MemoryLayout<Float>.size,
            deallocator: .none
        )
        
        return (buffer, data)
    }
}
