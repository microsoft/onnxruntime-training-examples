import SwiftUI

struct TrainView: View {
    
    private static let sentences = [
        "The gentle rustling of leaves in the breeze creates a soothing melody in the tranquil forest.",
        "With courage in their hearts and determination in their eyes, the brave explorers set forth on an epic quest.",
        "In the warmth of each other's embrace, they found solace and a love that transcended all boundaries.",
        "The rapid advancement of technology continues to reshape our world and the way we interact with it.",
        "Amidst the eerie darkness, whispers of an enigmatic presence sent shivers down their spines.",
        "Echoes of the past resonate through ancient ruins, telling tales of civilizations long gone.",
        "Among the stars and galaxies, the vastness of the cosmos reminds us of our place in the universe.",
        "Through laughter and tears, their unbreakable bond of friendship grew stronger with each passing day.",
        "With skilled brushstrokes and vivid colors, the artist brought life to a canvas, telling a thousand stories.",
        "In the face of adversity, they stood firm, displaying unwavering courage to overcome life's challenges.",
        "The tantalizing aroma of freshly baked bread wafted through the air, enticing all nearby.",
        "In the realm of dreams, possibilities are limitless, and the ordinary becomes extraordinary.",
        "The harmonious symphony of instruments blended together, evoking emotions that words cannot express.",
        "Through meticulous research and experimentation, scientists unravel the mysteries of the natural world.",
        "Even in the darkest times, a glimmer of hope can ignite a flame that illuminates the path ahead."
    ]
    
    private let knumRecording = 7
    private let audioRecorder = AudioRecorder()
    private let trainer = try! Trainer(sampleAudioRecordings: sentences.count)
    
    @State private var currentSentenceIndex = 0
    @State private var readyToRecord: Bool = true
    @State private var isTrainingComplete = false
    
    private func recordVoice() {
        audioRecorder.record { recordResult in
            let trainResult = recordResult.flatMap { recordingBufferAndData in
                return trainer.train(audio: recordingBufferAndData.data)
            }
            endRecord(trainResult)
        }
    }
    
    private func endRecord(_ result: Result<Void, Error>) {
        DispatchQueue.main.async {
            switch result {
            case .success:
                print("Successfully completed Train Step ")
            case .failure(let error):
                print("Error: \(error)")
            }
            readyToRecord = true
            
            if currentSentenceIndex < knumRecording - 1 {
                currentSentenceIndex += 1
                
            } else {
                isTrainingComplete = true
                try! trainer.exportModelForInference()
            }
        }
    }
    
    var body: some View {
        VStack {
            if !isTrainingComplete {
                Spacer()
                Text(TrainView.sentences[currentSentenceIndex])
                    .font(.title)
                    .padding()
                    .multilineTextAlignment(.center)
                    .fontDesign(.monospaced)
                
                Spacer()
                
                ZStack(alignment: .center) {
                    Image(systemName: "mic.fill")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 100, height: 100)
                        .foregroundColor( readyToRecord ? .gray: .red)
                        .transition(.scale)
                        .animation(.easeIn, value: 1)
                }
                
                Spacer()
                
                Button(action: {
                    readyToRecord = false
                    recordVoice()
                }) {
                    Text(readyToRecord ? "Record" : "Recording ...")
                        .font(.title)
                        .padding()
                        .background(readyToRecord ? .green : .gray)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }.disabled(!readyToRecord)
                
            } else {
                Spacer()
                Text("Training successfully finished!")
                    .font(.title)
                    .padding()
                    .multilineTextAlignment(.center)
                    .fontDesign(.monospaced)
                
                Spacer()
                NavigationLink(destination: InferView()) {
                    Text("Infer")
                        .font(.title)
                        .padding()
                        .background(.purple)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                .padding(.leading, 20)
            }
            
            Spacer()
        }
        .padding()
        .navigationTitle("Train")
    }
}

struct TrainView_Previews: PreviewProvider {
    static var previews: some View {
        TrainView()
    }
}
