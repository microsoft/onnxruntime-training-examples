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
            string parentDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.Parent.FullName;
            string dataPath = Path.Combine(parentDir, "train_tokens.json");

            var watch = System.Diagnostics.Stopwatch.StartNew();
            DataSplit ds = loadDataSplit(dataPath);

            watch.Stop();

            Console.WriteLine($"Execution Time for loading data split: {watch.ElapsedMilliseconds} ms");

            TrainingSession ts = CreateTrainingSession();
            trainEpoch(ts, ds, 4);
            Console.WriteLine("finished");
        }

        public static DataSplit loadDataSplit(string dataPath)
        {
            DataLoader dl = new DataLoader();
            return dl.ReadFile(dataPath);
        }

        public static TrainingSession CreateTrainingSession()
        {
            string parentDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.Parent.FullName;
            string checkpointPath = Path.Combine(parentDir, "training_artifacts", "mobilebert-uncased.ckpt");
            
            var state = CheckpointState.LoadCheckpoint (checkpointPath);
            string trainingPath = Path.Combine(parentDir, "training_artifacts", "mobilebert-uncased_training.onnx");
            string optimizerPath = Path.Combine(parentDir, "training_artifacts", "mobilebert-uncased_optimizer.onnx");
            
            return new TrainingSession(state, trainingPath, optimizerPath);
        }

        public static void trainEpoch(TrainingSession ts, DataSplit ds, int batchSize)
        {
            var numSamples = ds.InputShape[0];
            var seqLen = ds.InputShape[1];

            int steps = (int)(numSamples / batchSize);
            Console.WriteLine("steps: " + steps);
            long[] inputShape = { batchSize, seqLen };

            long startBatch = 0;
            long endBatch = batchSize * seqLen;

            for (int step = 0; step < steps; step++)
            {
                if (endBatch - startBatch < inputShape[0] * inputShape[1])
                {
                    inputShape[0] = (endBatch - startBatch) / seqLen;
                }

                // slices and creates inputs 
                var inputs = createInputsFromDataSplit(ds, startBatch, endBatch, inputShape);

                ts.LazyResetGrad();

                var outputs = ts.TrainStep(inputs);

                Console.WriteLine("Loss: " + outputs.ElementAtOrDefault(0).AsTensor<float>().GetValue(0));

                ts.OptimizerStep();

                startBatch += batchSize;
                endBatch = Math.Min(endBatch + batchSize, (numSamples * seqLen) - 1);
            }
        }

        public static List<FixedBufferOnnxValue> createInputsFromDataSplit(DataSplit ds, long start, long end, long[] shape)
        {
            List<FixedBufferOnnxValue> inputs = new List<FixedBufferOnnxValue>();
            inputs.Add(sliceInt64ArrayToBufferOnnx(ds.InputIds, start, end, shape));
            inputs.Add(sliceInt64ArrayToBufferOnnx(ds.AttentionMask, start, end, shape));
            inputs.Add(sliceInt64ArrayToBufferOnnx(ds.TokenTypeIds, start, end, shape));
            inputs.Add(sliceInt64ArrayToBufferOnnx(ds.Labels, start, end, shape));
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
    }
}