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
        
        static void Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.Write("Please supply the file path to the FER dataset as the first command line argument.");
                System.Environment.Exit(1);
            }

            string dataPath = args[0];
            DataLoader dl = loadDataset(dataPath);
            CreateAndRunTrainingSession(8, 8, dl);
            Console.WriteLine("Finished training and exporting");

            string inferencePath = Path.Combine(Directory.GetCurrentDirectory(), TRAINEDMODELFILE);
            var inferenceSession = new InferenceSession(inferencePath);

            if (args.Length > 1)
            {
                // assume additional arguments are paths to inference images
                for (int i = 1; i < args.Length; i++)
                {
                    runInferenceSession(inferenceSession, dl, args[i]);
                }
            }
            else
            {
                runInferenceSession(inferenceSession, dl, DEFAULTINFERENCE1);
                runInferenceSession(inferenceSession, dl, DEFAULTINFERENCE2);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="pathToFER">Path to the FER directory containing 7 unzipped folders labelled with 
        /// their corresponding emotion</param>
        /// <param name="trimDatasetTo">Optional parameter to trim the size of the dataset. If left as -1, 
        /// no trimming will happen and the full dataset will be used for training.</param>
        /// <returns></returns>
        public static DataLoader loadDataset(string pathToFER, int trimDatasetTo = -1)
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            DataLoader dl = new DataLoader();
            dl.loadFER(pathToFER);
            if (trimDatasetTo > 0)
            {
                dl.trimFER(trimDatasetTo);
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
                train(ts, dl, numEpochs, batchSize);
                Console.WriteLine("############################################################### Training finished");

                var outputNames = new List<string> { "outputs" };

                ts.ExportModelForInferencing(TRAINEDMODELFILE, outputNames);
            }
        }

        public static void runInferenceSession(InferenceSession inferenceSession, DataLoader dl, string pathToInference)
        {
            Console.WriteLine($"Inferencing now for {pathToInference}");

            var input = dl.imageProcessingForInference(pathToInference);
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("pixel_values", input) };

            foreach (var r in inferenceSession.Run(inputs))
            {
                var resultsTensor = r.AsTensor<float>();

                printPredictedEmotion(resultsTensor);

                Console.WriteLine(string.Join(",     ", DataLoader.EMOTIONSLABELS));
                Console.WriteLine(resultsTensor.GetArrayString());
            }
        }

        public static float train(TrainingSession ts, DataLoader dl, int numEpochs, int batchSize)
        {
            float loss = 0;
            for (int i = 0; i < numEpochs; i++)
            {
                loss = trainEpoch(ts, dl, batchSize);

                Console.WriteLine($"Loss after epoch {i}: {loss}");
            }

            return loss;
        }

        public static float trainEpoch(TrainingSession ts, DataLoader dl, int batchSize)
        {
            int steps = dl.getNumSteps(batchSize);
            Console.WriteLine("steps: " + steps);

            float loss = 0;

            for (int step = 0; step < steps; step++)
            {
                var inputs = dl.generateBatchInput(batchSize);
                ts.LazyResetGrad();

                var watch = System.Diagnostics.Stopwatch.StartNew();
                var outputs = ts.TrainStep(inputs);

                ts.OptimizerStep();
                watch.Stop();
                Console.WriteLine($"Execution Time for train step {step} out of {steps}: {watch.ElapsedMilliseconds} ms");

                if(step == steps - 1)
                {
                    // if at the last step, update the loss
                    loss = outputs.ElementAtOrDefault(0).AsTensor<float>().GetValue(0);
                }
            }

            return loss;
        }

        public static void printPredictedEmotion(Tensor<float> results)
        {
            var predictedEmotion = -1;
            float maxSoFar = 0;
            for (int i = 0; i < results.Length; i++)
            {
                if (results.GetValue(i) > maxSoFar)
                {
                    predictedEmotion = i;
                    maxSoFar = results.GetValue(i);
                }
            }

            Console.WriteLine($"Predicted emotion: {DataLoader.EMOTIONSLABELS[predictedEmotion]}");
        }
    }
}