using Microsoft.ML.OnnxRuntime;

namespace mobilevit_console
{
    public class Program
    {
        static void Main(string[] args)
        {
            CreateTrainingSession();
            Console.WriteLine("Finished.");
        }

        public static DataSplit loadDataSplit(string dataPath)
        {
            DataLoader dl = new DataLoader();
            return dl.ReadFile(dataPath);
        }

        // TODO: rename methods
        public static void CreateTrainingSession()
        {
            string parentDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.Parent.Parent.FullName;
            string dataPath = Path.Combine(parentDir, "mini_train.json");

            var watch = System.Diagnostics.Stopwatch.StartNew();
            DataSplit ds = loadDataSplit(dataPath);
            watch.Stop();
            Console.WriteLine($"Execution Time for loading data split: {watch.ElapsedMilliseconds} ms");

            string checkpointPath = Path.Combine(parentDir, "checkpoint");

            var state = new CheckpointState(checkpointPath);
            string trainingPath = Path.Combine(parentDir, "training_model_updated.onnx");
            string optimizerPath = Path.Combine(parentDir, "optimizer_model.onnx");

            TrainingSession ts = new TrainingSession(state, trainingPath, optimizerPath);
            trainEpoch(ts, ds, 8);

            Console.WriteLine("############################################################### training finished");

            string savedCheckpointPath = Path.Combine(parentDir, "saved_checkpoint.ckpt");

            // TODO: EVALUATE IF YOU NEED TO KEEP THIS BECAUSE YOURE JUST DOUBLE CHECKING THAT SAVEDCHECKPOINTPATH DOES WHAT
            // YOU WANT IT TO DO 
            //state.SaveCheckpoint(savedCheckpointPath, true);

            //var loadedState = new CheckpointState(savedCheckpointPath);
            //var newTrainingSession = new TrainingSession(loadedState, trainingPath);
            //trainEpoch(newTrainingSession, ds, 8);
        }

        public static void trainEpoch(TrainingSession ts, DataSplit ds, int batchSize)
        {
            var numSamples = ds.ImageShape[0];
            var imageSizeArr = ds.ImageShape[1..];

            var imageSizeNum = 1;

            foreach (int dimension in imageSizeArr)
            {
                imageSizeNum = imageSizeNum * dimension;
            }

            int steps = (int)(numSamples / batchSize);
            // int steps = 20;
            Console.WriteLine("steps: " + steps);
            long[] inputShape = (new long[] { batchSize }).Concat(imageSizeArr).ToArray();

            long startBatch = 0;
            long endBatch = batchSize * imageSizeNum;
            long startLabel = 0;
            long endLabel = batchSize;

            long[] labelShape = new long[] { batchSize };

            for (int step = 0; step < steps; step++)
            {
                if (endBatch - startBatch < inputShape[0] * imageSizeNum)
                {
                    inputShape[0] = (endBatch - startBatch) / imageSizeNum;
                }
                var watch = System.Diagnostics.Stopwatch.StartNew();
                // slices and creates inputs 
                var inputs = createInputsFromDataSplit(ds, startBatch, endBatch, inputShape, startLabel, endLabel, labelShape);

                ts.LazyResetGrad();

                var outputs = ts.TrainStep(inputs);

                Console.WriteLine("Loss: " + outputs.ElementAtOrDefault(0).AsTensor<float>().GetValue(0));

                ts.OptimizerStep();

                startBatch += batchSize;
                endBatch = Math.Min(endBatch + batchSize, (numSamples * imageSizeNum) - 1);
                watch.Stop();
                Console.WriteLine($"Execution Time for 1 train step: {watch.ElapsedMilliseconds} ms");
            }
        }

        public static List<FixedBufferOnnxValue> createInputsFromDataSplit(DataSplit ds, long start, long end, long[] shape, long startLabel, long endLabel, long[] labelShape)
        {
            List<FixedBufferOnnxValue> inputs = new List<FixedBufferOnnxValue>();
            inputs.Add(sliceDoubleArrayToBufferOnnx(ds.Image, start, end, shape));
            inputs.Add(sliceInt64ArrayToBufferOnnx(ds.Label, startLabel, endLabel, labelShape));
            return inputs;
        }

        public static FixedBufferOnnxValue sliceInt64ArrayToBufferOnnx(Int64[] array, long start,
            long end, long[] shape)
        {
            var memInfo = OrtMemoryInfo.DefaultInstance;
            Int64[] arraySlice = array[(int)start..(int)end];

            return FixedBufferOnnxValue.CreateFromMemory<Int64>(memInfo, arraySlice,
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Int64, shape,
                arraySlice.Length * sizeof(Int64));
        }

        public static FixedBufferOnnxValue sliceDoubleArrayToBufferOnnx(double[] array, long start,
            long end, long[] shape)
        {
            var memInfo = OrtMemoryInfo.DefaultInstance;
            double[] arraySlice = array[(int)start..(int)end];

            return FixedBufferOnnxValue.CreateFromMemory<double>(memInfo, arraySlice,
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float, shape,
                arraySlice.Length * sizeof(double));
        }
    }
}