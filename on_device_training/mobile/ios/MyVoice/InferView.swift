import SwiftUI

struct InferView: View {
    
    enum InferResult {
        case user, other, notSet
    }
    
    private let audioRecorder = AudioRecorder()
    
    @State private var voiceIdentifier: VoiceIdentifier? = nil
    @State private var readyToRecord: Bool = true
    
    @State private var inferResult: InferResult = InferResult.notSet
    @State private var probUser: Float = 0.0
    
    @State private var showAlert = false
    @State private var alertMessage = ""

    private func recordVoice() {
        audioRecorder.record { recordResult in
            let recognizeResult = recordResult.flatMap { recordingData in
                return voiceIdentifier!.evaluate(inputData: recordingData)
            }
            endRecord(recognizeResult)
        }
    }
    
    private func endRecord(_ result: Result<(Bool, Float), Error>) {
        DispatchQueue.main.async {
            switch result {
            case .success(let (isMatch, confidence)):
                print("Your Voice with confidence: \(isMatch),  \(confidence)")
                inferResult = isMatch ? .user : .other
                probUser = confidence
            case .failure(let error):
                print("Error: \(error)")
            }
            readyToRecord = true
        }
    }
    
    var body: some View {
        VStack {
            Spacer()
            
            ZStack(alignment: .center) {
                Image(systemName: "mic.fill")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 100, height: 100)
                    .foregroundColor( readyToRecord ? .gray: .red)
                    .transition(.scale)
                    .animation(.easeInOut, value: 1)
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
                
            }.disabled(voiceIdentifier == nil || !readyToRecord)
                .opacity(voiceIdentifier == nil ? 0.5: 1.0)
            
            if  inferResult != .notSet {
                Spacer()
                ZStack (alignment: .center) {
                    Image(systemName: inferResult == .user ? "person.crop.circle.fill.badge.checkmark": "person.crop.circle.fill.badge.xmark")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 100, height: 100)
                        .foregroundColor(inferResult == .user ? .green : .red)
                        .animation(.easeInOut, value: 2)
                    
                }
                
                Text("Probability of User : \(String(format: "%.2f", probUser*100.0))%")
                    .multilineTextAlignment(.center)
                    .fontDesign(.monospaced)
            }
            
            Spacer()
        }
        .padding()
        .navigationTitle("Infer")
        .onAppear {
            do {
                voiceIdentifier = try  VoiceIdentifier()
                
            } catch {
                alertMessage = "Error initializing inference session, make sure that training is completed: \(error)"
                showAlert = true
            }
            
        }
        .alert(isPresented: $showAlert) {
            Alert(title: Text("Error"), message: Text(alertMessage), dismissButton: .default(Text("OK")))
        }
    }
}

struct InferView_Previews: PreviewProvider {
    static var previews: some View {
        InferView()
    }
}
