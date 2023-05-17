using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace mobilevit_console
{
    public class Program
    {
        static string PARENTDIR = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.Parent.Parent.FullName;

        static string CHECKPOINTPATH = Path.Combine(PARENTDIR, "checkpoint");
        static string TRAININGMODELPATH = Path.Combine(PARENTDIR, "training_model.onnx");
        static string EVALMODELPATH = Path.Combine(PARENTDIR, "eval_model.onnx");
        static string OPTIMIZERMODELPATH = Path.Combine(PARENTDIR, "optimizer_model.onnx");

        static string TRAINEDMODELFILE = "trained_mobilevit.onnx";

        static string DEFAULTINFERENCE1 = Path.Combine(PARENTDIR, "test1.png"); 
        static string DEFAULTINFERENCE2 = Path.Combine(PARENTDIR, "test3.png");

        static string HELPDIALOGUE = @"
    usage:   mobilevit_console [-h | --help] <path_to_FER_dataset> [<path_to_inferencing_images>]
            
        path_to_FER_dataset         The absolute file path to the FER dataset. Within this file path should be 7 folders containing emotions.

        path_to_inferencing_images  Optionally, absolute file paths to PNG images to be inferenced on.
            ";

        static void Main(string[] args)
        {
            if (args.Length < 1)
            {
                PrintHelpAndExit();
            }

            foreach (string arg in args)
            {
                if(arg.Equals("-h") || arg.Equals("--help"))
                {
                    PrintHelpAndExit();
                }
            }

            string dataPath = args[0];
            DataLoader dl = LoadDataset(dataPath);
            CreateAndRunTrainingSession(0, 8, dl);
            Console.WriteLine("Finished training and exporting");
            Console.WriteLine();

            string inferencePath = Path.Combine(Directory.GetCurrentDirectory(), TRAINEDMODELFILE);
            var inferenceSession = new InferenceSession(inferencePath);

            if (args.Length > 1)
            {
                // assume additional arguments are paths to inference images
                for (int i = 1; i < args.Length; i++)
                {
                    RunInferenceSession(inferenceSession, dl, args[i]);
                }
            }
            else
            {
                RunInferenceSession(inferenceSession, dl, DEFAULTINFERENCE1);
                RunInferenceSession(inferenceSession, dl, DEFAULTINFERENCE2);
            }
        }

        public static void PrintHelpAndExit()
        {
            Console.Write(HELPDIALOGUE);
            System.Environment.Exit(0);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="pathToFER">Path to the FER directory containing 7 unzipped folders labelled with 
        /// their corresponding emotion</param>
        /// <param name="trimDatasetTo">Optional parameter to trim the size of the dataset. If left as -1, 
        /// no trimming will happen and the full dataset will be used for training.</param>
        /// <returns></returns>
        public static DataLoader LoadDataset(string pathToFER, int trimDatasetTo = -1)
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            DataLoader dl = new DataLoader();
            dl.LoadFER(pathToFER);
            if (trimDatasetTo > 0)
            {
                dl.TrimFER(trimDatasetTo);
            }

            watch.Stop();
            Console.WriteLine($"Execution Time for loading images and shuffling: {watch.ElapsedMilliseconds} ms");

            return dl;
        }

        public static void CreateAndRunTrainingSession(int numEpochs, int batchSize, DataLoader dl)
        {
            string parentDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.Parent.Parent.FullName;

            var state = CheckpointState.LoadCheckpoint(CHECKPOINTPATH);

            using (TrainingSession ts = new TrainingSession(state, TRAININGMODELPATH, EVALMODELPATH, OPTIMIZERMODELPATH))
            {
                Train(ts, dl, numEpochs, batchSize);
                Console.WriteLine("############################################################### Training finished");

                var outputNames = new List<string> { "outputs" };

                ts.ExportModelForInferencing(TRAINEDMODELFILE, outputNames);
            }
        }

        public static void RunInferenceSession(InferenceSession inferenceSession, DataLoader dl, string pathToInference)
        {
            Console.WriteLine($"Inferencing now for {pathToInference}");

            var input = dl.ImageProcessingForInference(pathToInference);
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("pixel_values", input) };

            foreach (var r in inferenceSession.Run(inputs))
            {
                var resultsTensor = r.AsTensor<float>();

                PrintPredictedEmotion(resultsTensor);

                Console.WriteLine(string.Join(",     ", DataLoader.EMOTIONSLABELS));
                Console.WriteLine(resultsTensor.GetArrayString());
            }
            Console.WriteLine();
        }

        public static float Train(TrainingSession ts, DataLoader dl, int numEpochs, int batchSize)
        {
            float loss = 0;
            for (int i = 0; i < numEpochs; i++)
            {
                loss = TrainEpoch(ts, dl, batchSize);

                Console.WriteLine($"Average loss after epoch {i}: {loss}");
            }

            return loss;
        }

        public static float TrainEpoch(TrainingSession ts, DataLoader dl, int batchSize)
        {
            int steps = dl.GetNumSteps(batchSize);
            Console.WriteLine("steps: " + steps);

            float avgLoss = 0;

            for (int step = 0; step < steps; step++)
            {
                var inputs = dl.GenerateBatchInput(batchSize);
                ts.LazyResetGrad();

                var watch = System.Diagnostics.Stopwatch.StartNew();
                var outputs = ts.TrainStep(inputs);

                ts.OptimizerStep();
                watch.Stop();
                Console.WriteLine($"Execution Time for train step {step} out of {steps}: {watch.ElapsedMilliseconds} ms");

                avgLoss += outputs.ElementAtOrDefault(0).AsTensor<float>().GetValue(0);
            }

            return avgLoss / steps;
        }

        public static void PrintPredictedEmotion(Tensor<float> results)
        {
            var predictedEmotion = -1;
            var secondPredictedEmotion = -1;
            float maxSoFar = float.NegativeInfinity;
            float secondMaxSoFar = float.NegativeInfinity;
            for (int i = 0; i < results.Length; i++)
            {
                if (results.GetValue(i) > maxSoFar)
                {
                    predictedEmotion = i;
                    maxSoFar = results.GetValue(i);
                }
                else if (results.GetValue(i) > secondMaxSoFar)
                {
                    secondPredictedEmotion = i;
                    secondMaxSoFar = results.GetValue(i);
                }
            }

            Console.WriteLine($"Predicted emotion: {DataLoader.EMOTIONSLABELS[predictedEmotion]}");
            Console.WriteLine($"Second most likely emotion: {DataLoader.EMOTIONSLABELS[secondPredictedEmotion]}");
        }
    }
}