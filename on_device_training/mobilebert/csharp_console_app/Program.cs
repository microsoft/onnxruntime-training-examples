// See https://aka.ms/new-console-template for more information
using csharp_console_app;
using Microsoft.ML.OnnxRuntime;
using System.Text.Json;

namespace ConsoleApp
{
    public class Program
    {
        static void Main(string[] args)
        {
            // Console.WriteLine("Hello, World!");
            // TODO: wrap in try catch ???
            // string test = Console.ReadLine();
            // Console.WriteLine("You entered \"" + test + "\" in the console!");
            string parentDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.Parent.FullName;
            string dataPath = Path.Combine(parentDir, "train_tokens.json");                   
            DataLoader dl = new DataLoader();

            DataSplit training = dl.ReadFile(dataPath);
            Console.WriteLine("InputIds: " + training.InputIds.Length);
            Console.WriteLine("InputShape: " + training.InputShape[0] + ", " + training.InputShape[1]);
            Console.WriteLine("TokenTypeIds: " + training.TokenTypeIds);
            Console.WriteLine("AttentionMask: " + training.AttentionMask);
            Console.WriteLine("SpecialTokens: " + training.SpecialTokensMask.Length);
             
            CreateTrainingSession(training);
            Console.WriteLine("finished");
        }

        public static void CreateTrainingSession(DataSplit ds)
        {
            // TODO: change the strings from being hardcoded to ... making more sense
            string parentDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.Parent.FullName;
            string checkpointPath = Path.Combine(parentDir, "training_artifacts", "mobilebert-uncased.ckpt");
            
            var state = new CheckpointState(checkpointPath);
            string trainingPath = Path.Combine(parentDir, "training_artifacts", "mobilebert-uncased_training.onnx");
            var trainingSession = new TrainingSession(state, trainingPath);

            var memInfo = OrtMemoryInfo.DefaultInstance;

            List<FixedBufferOnnxValue> inputs = new List<FixedBufferOnnxValue>();

            // order of inputs should follow same order as the inputs for your onnx model
            FixedBufferOnnxValue inputIds = FixedBufferOnnxValue.CreateFromMemory<Int64>(memInfo, 
                ds.InputIds, Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Int64, 
                ds.InputShape, ds.InputIds.Length * sizeof(Int64));

            inputs.Add(inputIds);

            FixedBufferOnnxValue attentionMask = FixedBufferOnnxValue.CreateFromMemory<Int32>(memInfo,
                ds.AttentionMask, Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Int32,
                ds.AttentionShape, ds.AttentionMask.Length * sizeof(Int32));

            inputs.Add(attentionMask);

            FixedBufferOnnxValue tokenTypeIds = FixedBufferOnnxValue.CreateFromMemory<Int32>(memInfo,
                ds.TokenTypeIds, Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Int32,
                ds.TokenShape, ds.TokenTypeIds.Length * sizeof(Int32));

            inputs.Add(tokenTypeIds);

            trainingSession.TrainStep(inputs);
        }
    }
}