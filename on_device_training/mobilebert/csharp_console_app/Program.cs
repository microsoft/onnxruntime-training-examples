// See https://aka.ms/new-console-template for more information
using Microsoft.ML.OnnxRuntime;

namespace ConsoleApp
{
    public class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
            // TODO: wrap in try catch ???
            string test = Console.ReadLine();
            Console.WriteLine("You entered \"" + test + "\" in the console!");

            CreateTrainingSession();
        }

        public static void CreateTrainingSession()
        {
            // TODO: change the strings from being hardcoded to ... making more sense
            string parentDir = Directory.GetParent(Directory.GetCurrentDirectory()).FullName;
            string checkpointPath = Path.Combine(parentDir, "training_artifacts_archive", "mobilebert-uncased.ckpt");
            var state = new CheckpointState(checkpointPath);
            string trainingPath = Path.Combine(parentDir, "training_artifacts_archive", "mobilebert-uncased_training.onnx");
            var trainingSession = new TrainingSession(state, trainingPath);
            
        }
    }
}