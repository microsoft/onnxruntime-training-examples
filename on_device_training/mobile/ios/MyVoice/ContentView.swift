import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationView {
            VStack {
                
                Text("My Voice")
                    .font(.largeTitle)
                    .padding(.top, 50)
                
                Spacer()
                
                ZStack(alignment: .center) {
                    Image(systemName: "waveform.circle.fill")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 100, height: 100)
                        .foregroundColor(.purple)
                }
                
                Spacer()
                
                HStack {
                    NavigationLink(destination: TrainView()) {
                        Text("Train")
                            .font(.title)
                            .padding()
                            .background(Color.purple)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    .padding(.trailing, 20)
                    
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
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}




