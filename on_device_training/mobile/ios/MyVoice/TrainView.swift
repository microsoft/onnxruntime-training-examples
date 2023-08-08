import SwiftUI

struct TrainView: View {
    
    enum ViewState {
        case recording, trainingInProgress, trainingComplete
    }
    
    private static let sentences = [
        "In the embrace of nature's beauty, I find peace and tranquility. The gentle rustling of leaves soothes my soul, and the soft sunlight kisses my skin. As I breathe in the fresh air, I am reminded of the interconnectedness of all living things, and I feel a sense of oneness with the world around me.",
        "Under the starlit sky, I gaze in wonder at the vastness of the universe. Each twinkle represents a story yet untold, a dream yet to be realized. With every new dawn, I am filled with hope and excitement for the opportunities that lie ahead. I embrace each day as a chance to grow, to learn, and to create beautiful memories.",
        "A warm hug from a loved one is a precious gift that warms my heart. In that tender embrace, I feel a sense of belonging and security. Laughter and tears shared with dear friends create a bond that withstands the test of time. These connections enrich my life and remind me of the power of human relationships.",
        "Life's journey is like a beautiful melody, with each note representing a unique experience. As I take each step, I harmonize with the rhythm of existence. Challenges may come my way, but I face them with resilience and determination, knowing they are opportunities for growth and self-discovery.",
        "With every page turned in a book, I open the door to new worlds and ideas. The written words carry the wisdom of countless souls, and I am humbled by the knowledge they offer. In stories, I find a mirror to my own experiences and a beacon of hope for a better tomorrow.",
        "Life's trials may bend me, but they will not break me. Through adversity, I discover the strength within my heart. Each obstacle is a chance to learn, to evolve, and to emerge as a better version of myself. I am grateful for every lesson, for they shape me into the person I am meant to be.",
        "The sky above is an ever-changing canvas of colors and clouds. In its vastness, I realize how small I am in the grand scheme of things, and yet, I know my actions can ripple through the universe. As I walk this Earth, I seek to leave behind a positive impact and a legacy of love and compassion.",
        "In the stillness of meditation, I connect with the depth of my soul. The external noise fades away, and I hear the whispers of my inner wisdom. With each breath, I release tension and embrace serenity. Meditation is my sanctuary, a place where I can find clarity and renewed energy.",
        "Kindness is a chain reaction that spreads like wildfire. A simple act of compassion can brighten someone's day and inspire them to pay it forward. Together, we can create a wave of goodness that knows no boundaries, reaching even the farthest corners of the world.",
        "As the sun rises on a new day, I am filled with gratitude for the gift of life. Every moment is a chance to make a difference, to love deeply, and to embrace joy. I welcome the adventures that await me and eagerly embrace the mysteries yet to be uncovered."
    ]

    
    private let knumRecordings = 5
    private let audioRecorder = AudioRecorder()
    private let trainer = try! Trainer()
    
    @State private var trainingData: [Data] = []
    
    @State private var viewState: ViewState = .recording
    @State private var readyToRecord: Bool = true
    @State private var trainingProgress: Double = 0.0
    
    private func recordVoice() {
        audioRecorder.record { recordResult in
           switch recordResult {
           case .success(let recordingData):
               trainingData.append(recordingData)
               print("Successfully completed Recording")
           case .failure(let error):
               print("Error: \(error)")
            }
            
            readyToRecord = true
            
            if trainingData.count == knumRecordings  {
                viewState = .trainingInProgress
                trainAndExportModel()
            }
        }
    }
    
    private func updateProgressBar(progress: Double) {
        DispatchQueue.main.async {
            trainingProgress = progress
        }
    }
    
    private func trainAndExportModel() {
        Task {
            do {
                try trainer.train(trainingData, progressCallback: updateProgressBar)
                try trainer.exportModelForInference()
                   
                DispatchQueue.main.async {
                    viewState = .trainingComplete
                    print("Training is complete")
                }
            } catch {
                DispatchQueue.main.async {
                    viewState = .trainingComplete
                    print("Training Failed: \(error)")
                }
            }
        }
    }
    
    
    var body: some View {
        VStack {
            Spacer()
            switch viewState {
            case .recording:
                Text(TrainView.sentences[trainingData.count % TrainView.sentences.count])
                    .font(.body)
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
                    
            case .trainingInProgress:
                VStack {
                    Spacer()
                    ProgressView(value: trainingProgress,
                                 total: 1.0,
                                 label: {Text("Training")},
                                 currentValueLabel: {Text(String(format: "%.0f%%", trainingProgress * 100))})
                    .padding()
                    Spacer()
                }
                    
            case .trainingComplete:
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
