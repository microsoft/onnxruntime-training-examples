using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using static DragGANApp.Misc;
using System.Drawing.Printing;

namespace DragGANApp
{
    public class DragGAN
    {
        public class Info
        {
            public int Seed;
            public Bitmap Image;
            public Tensor<float> Latent;
        }

        private string Model;
        private Misc.ModelInfo ModelInfo;
        private string m_model_path = @".\ONNXModels";

        private SessionOptions m_options;
        private CheckpointState m_state;
        private TrainingSession m_session;

        private OrtValue m_snapshot;

        private InferenceSession m_mapper_session;
        Memory<float> m_pixels_memory;
        Tensor<float> m_image_tensor;
        Tensor<float> m_F_tensor;
        //string m_saved_seeds_folder;

        public int m_iterations = 10;

        public bool OptimizationShouldStop = false;
        public Info LatestResult;
        // -----------------------------------------------------
        public class OptimizationParameters
        {
            public List<ImageCoord> Handles;
            public List<ImageCoord> Targets;
            public Tensor<float> Latent;
            public Action<OptimizationIntermidiateInformation> IntermidiateImagesAction;
            public Action OptimizationEndedAction;
        }

        public class OptimizationIntermidiateInformation
        {
            public Bitmap Image;
            public int CurrentIteration;
            public int TotalIterations;
        }

        private static int FFeatureSize = 128;
        private static Size ImageSize = new Size(512, 512);

        // -----------------------------------------------------

        public class ImageCoord
        {
            public int X { get; set; }
            public int Y { get; set; }

            public ImageCoord(int x, int y)
            {
                X = x;
                Y = y;
            }

            public override string ToString()
            {
                return $"(X, Y) = ({X}, {Y})";
            }
            public int this[int index]
            {
                get
                {
                    switch (index)
                    {
                        case 0: return X;
                        case 1: return Y;
                        default: throw new Exception("index out of bounds");
                    }
                }
                set
                {
                    switch (index)
                    {
                        case 0: X = value; break;
                        case 1: Y = value; break;
                        default: throw new Exception("index out of bounds");
                    }
                }
            }

        }
        // -----------------------------------------------------
        public class Feature
        {
            float[] m_data;

            public Feature(float[] data)
            {
                int size = data.Length;
                m_data = new float[size];
                for (int i = 0; i < size; i++)
                    m_data[i] = data[i];
            }

            public Feature(List<float> data)
            {
                int size = data.Count;
                m_data = new float[size];
                for (int i = 0; i < size; i++)
                    m_data[i] = data[i];
            }

            public Feature(int size)
            {
                m_data = new float[size];
                for (int i = 0; i < size; i++)
                    m_data[i] = 0;
            }

            public override string ToString()
            {
                if (m_data != null)
                    return $"Feature of length {m_data.Length}";
                else
                    return $"Feature was not initialized";
            }
            public float this[int index]
            {
                get
                {
                    return m_data[index];
                }
                set
                {
                    m_data[index] = value;
                }
            }

            public int Length
            {
                get { return m_data.Length; }
            }

            public static float Distance(Feature a, Feature b)
            {
                int length = a.Length;

                float sum = 0.0f;
                for (int i = 0; i < length; i++)
                {
                    sum += Math.Abs(a[i] - b[i]);
                }

                return sum;
            }
        }
        // -----------------------------------------------------

        static public Bitmap GetImage(Tensor<float> image_data, Memory<float> pixels_memory, List<ImageCoord> handles, List<ImageCoord> targets)
        {
            var bmp = GetImage(image_data, pixels_memory);

            var size = 5;
            using (var g = Graphics.FromImage(bmp))
            {
                for (int i = 0; i < handles.Count; i++)
                {
                    var h = handles[i];
                    var t = targets[i];
                    using (Brush brsh = new SolidBrush(Color.Red))
                    {
                        g.FillEllipse(brsh, h[0] - size, h[1] - size, size * 2, size * 2);
                    }
                    using (Brush brsh = new SolidBrush(Color.Blue))
                    {
                        g.FillEllipse(brsh, t[0] - size, t[1] - size, size * 2, size * 2);
                    }
                }
            }
            return bmp;
        }

        static public Bitmap GetImage(Tensor<float> image_data, Memory<float> pixels_memory)
        {
            unsafe
            {
                int width = image_data.Dimensions[3];
                int height = image_data.Dimensions[2];
                var bmp = new Bitmap(width, height, PixelFormat.Format24bppRgb);
                var bitmapData = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);

                byte* bits = (byte*)bitmapData.Scan0;
                float[] pixels = pixels_memory.ToArray();
                fixed (float* p = pixels)
                {
                    // Avoid having those 2 multiplications in the loop
                    //int wtimesh1 = height * width * 1;
                    //int wtimesh2 = height * width * 2;
                    for (int j = 0; j < height; j++)
                    {
                        // Avoid multiplication in the loop
                        int jtimesw = j * width;
                        for (int i = 0; i < width; i++)
                        {
                            // unflattening 1D array using 3D coordinates
                            //      index = z *   yMax *  xMax + y *  xMax + x
                            int pixel = j * bitmapData.Stride + i * 3;
                            int r_index = j * height + i;
                            int g_index = height * width + r_index; ;
                            int b_index = height * width + g_index;
                            // Avoid recalculation
                            //int jtimeswplusi = jtimesw + i;
                            bits[pixel + 2] = (byte)MathUtils.Clamp<float>((p[r_index] * 127.5f + 127.5f), 0f, 255f);
                            bits[pixel + 1] = (byte)MathUtils.Clamp<float>((p[g_index] * 127.5f + 127.5f), 0f, 255f);
                            bits[pixel + 0] = (byte)MathUtils.Clamp<float>((p[b_index] * 127.5f + 127.5f), 0f, 255f);
                        }
                    }

                }
                bmp.UnlockBits(bitmapData);

                //bmp.Save("StyleGAN.png");
                return bmp;
            }
        }

        public static Bitmap ResizeImage(Bitmap imgToResize, Size size)
        {
            try
            {
                Bitmap b = new Bitmap(size.Width, size.Height);
                using (Graphics g = Graphics.FromImage((Image)b))
                {
                    g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                    g.DrawImage(imgToResize, 0, 0, size.Width, size.Height);
                }
                return b;
            }
            catch
            {
                Console.WriteLine("Bitmap could not be resized");
                return imgToResize;
            }
        }


        // -----------------------------------------------------
        static List<ImageCoord> CreateSquareMask(int height, int width, ImageCoord center, int radius)
        {
            var size = radius * 2 + 1;
            var res = new List<ImageCoord>(size * size);

            if (radius <= 0)
            {
                throw new Exception("radius must be a positive integer");
            }

            if (center[0] < radius ||
                 center[0] >= height - radius ||
                 center[1] < radius ||
                 center[1] >= width - radius)
            {
                throw new Exception("center and radius must be within the bounds of the mask");
            }

            for (int x = center[1] - radius; x <= center[1] + radius; x++)
            {
                for (int y = center[0] - radius; y <= center[0] + radius; y++)
                {
                    res.Add(new ImageCoord(y, x));
                }
            }

            return res;
        }

        public static int IndexOfMin(IList<float> self)
        {
            if (self == null)
            {
                throw new ArgumentNullException("self");
            }

            if (self.Count == 0)
            {
                throw new ArgumentException("List is empty.", "self");
            }

            var min = self[0];
            int minIndex = 0;

            for (int i = 1; i < self.Count; ++i)
            {
                if (self[i] < min)
                {
                    min = self[i];
                    minIndex = i;
                }
            }

            return minIndex;
        }

        static List<ImageCoord> PointTracking(Tensor<float> F, Tensor<float> F0, List<ImageCoord> handle_points, List<ImageCoord> handle_points_0, int r2)
        {
            var n = handle_points.Count;
            var height = F.Dimensions[2];
            var width = F.Dimensions[3];
            var f_size = F.Dimensions[1];
            var patch_size = (r2 * 2 + 1) * (r2 * 2 + 1);
            var res = new List<ImageCoord>();

            for (int i = 0; i < n; i++)
            {
                var center = handle_points[i];

                // Find indices of patch around center point
                var patch = CreateSquareMask(height, width, center, r2);

                // Extract features in the patch
                var F_qi = new List<Feature>(patch_size);
                foreach (var p in patch)
                {
                    var feature = new Feature(f_size);
                    for (int j = 0; j < f_size; j++)
                    {
                        feature[j] = F[0, j, p[1], p[0]];
                    }
                    F_qi.Add(feature);
                }

                // Extract feature of the initial handle point
                var f_i = new Feature(f_size);
                for (int j = 0; j < f_size; j++)
                {
                    f_i[j] = F0[0, j, handle_points_0[i][1], handle_points_0[i][0]];
                }

                // Compute the L1 distance between the patch features and the initial handle point feature
                //distances = LA.norm(F_qi - f_i[:, :, None], 1, axis = 1)
                var distances = new List<float>(patch_size);
                for (int j = 0; j < patch_size; j++)
                {
                    var f_qi = F_qi[j];

                    var dist = Feature.Distance(f_qi, f_i);
                    distances.Add(dist);
                }

                var min_index = IndexOfMin(distances);
                res.Add(patch[min_index]);
            }

            return res;
        }
        // -----------------------------------------------------



        public DragGAN()
        {

        }

        public void Init(string model, int iterations)
        {
            Model = model;
            m_model_path = Path.GetFullPath(Path.Combine(m_model_path, model));

            ModelInfo = Misc.LoadModelInfo(Path.Combine(m_model_path, "info.json"));
            
            // set global values
            ImageSize = new Size(ModelInfo.ImageSize[0], ModelInfo.ImageSize[0]);
            FFeatureSize = ModelInfo.FFeatureSize;


            m_iterations = iterations;

            m_options = SessionOptions.MakeSessionOptionWithCudaProvider(0);

            // stylegan part
            string mapper_filename = Path.Combine(m_model_path, "stylegan_mapper.onnx");
            m_mapper_session = new InferenceSession(mapper_filename, m_options);

            m_pixels_memory = new float[3 * ImageSize.Width * ImageSize.Height];
            m_image_tensor = new DenseTensor<float>(m_pixels_memory, new[] { 1, 3, ImageSize.Width, ImageSize.Height});
            m_F_tensor = new DenseTensor<float>(new[] { 1, FFeatureSize, ImageSize.Width, ImageSize.Height});

            string CHECKPOINTPATH = Path.Combine(m_model_path, "checkpoint");
            string TRAININGMODELPATH = Path.Combine(m_model_path, "training_model.onnx");
            string EVALMODELPATH = Path.Combine(m_model_path, "eval_model.onnx");
            string OPTIMIZERMODELPATH = Path.Combine(m_model_path, "optimizer_model.onnx");

            m_state = CheckpointState.LoadCheckpoint(CHECKPOINTPATH);
            m_session = new TrainingSession(m_options, m_state, TRAININGMODELPATH, EVALMODELPATH, OPTIMIZERMODELPATH);

            // snashot the initial state
            m_snapshot = m_session.ToBuffer(true);
        }

        public void Shutdown()
        {
            try
            {
                m_snapshot.Dispose();
                m_snapshot = null;

                m_mapper_session.Dispose();
                m_mapper_session = null;

                m_state.Dispose();
                m_state = null;

                m_session.Dispose();
                m_session = null;

                m_options.Dispose();
                m_options = null;
            }
            catch (Exception e)
            {
                Console.WriteLine($"Problem cleaning up ORT resources.\n{e.ToString()}");
            }
        }

        public Info GetImage(int seed)
        {
            var ws = new DenseTensor<float>(new int[] { 1, 18, 512 });

            // Create a random number generator
            Random rng = new Random(seed);

            // Generate a 1x512 array of float32 random numbers
            var z = new float[512];
            for (int i = 0; i < z.Length; i++)
                z[i] = (float)(rng.NextDouble() * 2 - 1);

            var zz = new DenseTensor<float>(z, new int[] { 1, 512 });

            var mapper_inputs = new List<NamedOnnxValue>()
            {
                NamedOnnxValue.CreateFromTensor<float>("input", zz)
            };

            var mapper_outputs = m_mapper_session.Run(mapper_inputs); //.ToList().Last().AsEnumerable<NamedOnnxValue>(); //.ToList().Last().AsEnumerable<NamedOnnxValue>();

            var w = mapper_outputs[0].Value as DenseTensor<float>;

            for (int i = 0; i < 18; i++)
            {
                for (int j = 0; j < 512; j++)
                {
                    ws[0, i, j] = w[0, j];
                }
            }

            Tensor<float> latent = ws;

            m_session.FromBuffer(m_snapshot, true);

            // Try to use the parameter update API
            {
                int index;

                var parameter_trainable = m_state.GetParameter("latent_trainable");
                var parameter_untrainable = m_state.GetParameter("latent_untrainable");

                var shape_info_trainable = parameter_trainable.GetTensorTypeAndShape();
                var shape_info_untrainable = parameter_untrainable.GetTensorTypeAndShape();

                var shape_trainable = shape_info_trainable.Shape;
                var shape_untrainable = shape_info_untrainable.Shape;

                // update the trainable part
                float[] updated_parameter_buffer_trainable = new float[shape_trainable[0] * shape_trainable[1] * shape_trainable[2]];
                index = 0;
                for (int i = 0; i < shape_trainable[1]; i++)
                {
                    for (int j = 0; j < shape_trainable[2]; j++)
                    {
                        updated_parameter_buffer_trainable[index++] = latent[0, i, j];
                    }
                }
                var updated_parameter_trainable = OrtValue.CreateTensorValueFromMemory(updated_parameter_buffer_trainable, shape_trainable);
                m_state.UpdateParameter("latent_trainable", updated_parameter_trainable);

                // update the untrainable part
                float[] updated_parameter_buffer_untrainable = new float[shape_untrainable[0] * shape_untrainable[1] * shape_untrainable[2]];
                index = 0;
                for (int i = 0; i < shape_untrainable[1]; i++)
                {
                    for (int j = 0; j < shape_untrainable[2]; j++)
                    {
                        updated_parameter_buffer_untrainable[index++] = latent[0, (int)(shape_trainable[1] + i), j];
                    }
                }
                var updated_parameter_untrainable = OrtValue.CreateTensorValueFromMemory(updated_parameter_buffer_untrainable, shape_untrainable);
                m_state.UpdateParameter("latent_untrainable", updated_parameter_untrainable);
            }


            var handles = new List<ImageCoord>() { new ImageCoord(200, 200) };
            var targets = new List<ImageCoord>() { new ImageCoord(200, 200) };

            var handles_0 = new List<ImageCoord>();
            handles_0.AddRange(handles);

            Tensor<float> handle_points = new DenseTensor<float>(new[] { 1, 2 });
            Tensor<float> target_points = new DenseTensor<float>(new[] { 1, 2 });

            foreach (var p in handles)
            {
                handle_points[0, 0] = p[1];
                handle_points[0, 1] = p[0];
            }
            foreach (var p in targets)
            {
                target_points[0, 0] = p[1];
                target_points[0, 1] = p[0];
            }
            var inputs = new List<FixedBufferOnnxValue> {
                FixedBufferOnnxValue.CreateFromTensor(handle_points),
                FixedBufferOnnxValue.CreateFromTensor(target_points)
            };

            //Memory<float> pixels_memory = new float[3 * 1024 * 1024];

            Tensor<float> out_loss = new DenseTensor<float>(new ReadOnlySpan<int>(new int[] { }));
            Tensor<float> out_img = new DenseTensor<float>(m_pixels_memory, new[] { 1, 3, ImageSize.Width, ImageSize.Height});
            Tensor<float> out_F0 = new DenseTensor<float>(new[] { 1, FFeatureSize, ImageSize.Width, ImageSize.Height });

            var outputs = new FixedBufferOnnxValue[] {
                FixedBufferOnnxValue.CreateFromTensor(out_loss),
                FixedBufferOnnxValue.CreateFromTensor(out_img),
                FixedBufferOnnxValue.CreateFromTensor(out_F0)
            };

            m_session.TrainStep(inputs, outputs);

            var bmp = DragGAN.GetImage(m_image_tensor, m_pixels_memory);

            // cleanup
            foreach (var i in inputs)
                i.Dispose();
            foreach (var i in outputs)
                i.Dispose();

            return new Info()
            {
                Seed = seed,
                Image = bmp,
                Latent = ws,
            };
        }


        public bool CloseEnough(List<ImageCoord> handles, List<ImageCoord> targets, float tol = 2.0f)
        {
            float total_dist = 0.0f;
            for(int i=0; i<handles.Count; i++)
            {
                var h = handles[i];
                var t = targets[i];

                float dist = (float)Math.Sqrt((float)(h.X - t.X) * (h.X - t.X) + (float)(h.Y - t.Y) * (h.Y - t.Y));
                total_dist += dist;
            }
            total_dist /= handles.Count;
            //Console.WriteLine($"Dist: {total_dist}");
            if (total_dist <= tol)
                return true;
            
            return false;
        }

        public void RunInThread(object obj)
        {
            OptimizationShouldStop = false;
            var p = obj as OptimizationParameters;
            LatestResult = Run(p.Handles, p.Targets, p.Latent, p.IntermidiateImagesAction);
            //return LatestResult;
            if (p.OptimizationEndedAction != null)
            {
                p.OptimizationEndedAction();
            }
        }

        public DragGAN.Info Run(List<ImageCoord> user_handles, List<ImageCoord> user_targets, Tensor<float> latent, Action<OptimizationIntermidiateInformation> intermidiate_images = null)
        {
            // restore initial state
            m_session.FromBuffer(m_snapshot, true);

            // Try to use the parameter update API
            {
                int index;

                var parameter_trainable = m_state.GetParameter("latent_trainable");
                var parameter_untrainable = m_state.GetParameter("latent_untrainable");
                
                var shape_info_trainable = parameter_trainable.GetTensorTypeAndShape();
                var shape_info_untrainable = parameter_untrainable.GetTensorTypeAndShape();

                var shape_trainable = shape_info_trainable.Shape;
                var shape_untrainable = shape_info_untrainable.Shape;

                // update the trainable part
                float[] updated_parameter_buffer_trainable = new float[shape_trainable[0] * shape_trainable[1] * shape_trainable[2]];
                index = 0;
                for(int i=0; i< shape_trainable[1]; i++)
                {
                    for(int j=0; j< shape_trainable[2]; j++)
                    {
                        updated_parameter_buffer_trainable[index++] = latent[0, i, j];
                    }
                }
                var updated_parameter_trainable = OrtValue.CreateTensorValueFromMemory(updated_parameter_buffer_trainable, shape_trainable);
                m_state.UpdateParameter("latent_trainable", updated_parameter_trainable);

                // update the untrainable part
                float[] updated_parameter_buffer_untrainable = new float[shape_untrainable[0] * shape_untrainable[1] * shape_untrainable[2]];
                index = 0;
                for (int i = 0; i < shape_untrainable[1]; i++)
                {
                    for (int j = 0; j < shape_untrainable[2]; j++)
                    {
                        updated_parameter_buffer_untrainable[index++] = latent[0, (int)(shape_trainable[1] + i), j];
                    }
                }
                var updated_parameter_untrainable = OrtValue.CreateTensorValueFromMemory(updated_parameter_buffer_untrainable, shape_untrainable);
                m_state.UpdateParameter("latent_untrainable", updated_parameter_untrainable);
            } 

             
            var n_iter = m_iterations;
            var tolerance = 2;
            float step_size = 0.002f;
            var r2 = 12;


            var handles = new List<ImageCoord>();
            handles.AddRange(user_handles);
            var targets = new List<ImageCoord>();
            targets.AddRange(user_targets);

            var handles_0 = new List<ImageCoord>();
            handles_0.AddRange(handles);

            //var out_path = @"C:\T\StyleGAN\ONNXTry";

            Tensor<float> handle_points = new DenseTensor<float>(new[] { 1, 2 });
            Tensor<float> target_points = new DenseTensor<float>(new[] { 1, 2 });

            foreach (var p in handles)
            {
                handle_points[0, 0] = p[1];
                handle_points[0, 1] = p[0];
            }
            foreach (var p in targets)
            {
                target_points[0, 0] = p[1];
                target_points[0, 1] = p[0];
            }
            var inputs = new List<FixedBufferOnnxValue> {
                FixedBufferOnnxValue.CreateFromTensor(handle_points),
                FixedBufferOnnxValue.CreateFromTensor(target_points)
            };


            Memory<float> pixels_memory = new float[3 * ImageSize.Width * ImageSize.Height];

            Tensor<float> out_loss = new DenseTensor<float>(new ReadOnlySpan<int>(new int[] { }));
            Tensor<float> out_img = new DenseTensor<float>(pixels_memory, new[] { 1, 3, ImageSize.Width, ImageSize.Height });
            Tensor<float> out_F0 = new DenseTensor<float>(new[] { 1, FFeatureSize, ImageSize.Width, ImageSize.Height });

            var outputs = new FixedBufferOnnxValue[] {
                FixedBufferOnnxValue.CreateFromTensor(out_loss),
                FixedBufferOnnxValue.CreateFromTensor(out_img),
                FixedBufferOnnxValue.CreateFromTensor(out_F0)
            };

            m_session.TrainStep(inputs, outputs);

            // cleanup
            //foreach (var i in inputs)
            //    i.Dispose();
            foreach (var i in outputs)
                i.Dispose();


            //
            // try the optimization loop
            //

            m_session.SetLearningRate(step_size);

            handle_points = new DenseTensor<float>(new[] { 1, 2 });
            target_points = new DenseTensor<float>(new[] { 1, 2 });

            Tensor<float> out2_loss = new DenseTensor<float>(new ReadOnlySpan<int>(new int[] { }));
            Tensor<float> out2_img = new DenseTensor<float>(pixels_memory, new[] { 1, 3, ImageSize.Width, ImageSize.Height });
            Tensor<float> out2_F = new DenseTensor<float>(new[] { 1, FFeatureSize, ImageSize.Width, ImageSize.Height });

            var outputs2 = new FixedBufferOnnxValue[] {
                    FixedBufferOnnxValue.CreateFromTensor(out2_loss),
                    FixedBufferOnnxValue.CreateFromTensor(out2_img),
                    FixedBufferOnnxValue.CreateFromTensor(out2_F)
                };

            for (int iter = 0; iter < n_iter; iter++)
            {
                if (CloseEnough(handles, targets, 4.0f))
                    break;

                if (OptimizationShouldStop)
                    break;

                var watch = System.Diagnostics.Stopwatch.StartNew();

                //if np.allclose(handle_points, target_points, atol = tolerance):
                //    break

                m_session.LazyResetGrad();


                foreach (var p in handles)
                {
                    handle_points[0, 0] = p[1];
                    handle_points[0, 1] = p[0];
                }
                foreach (var p in targets)
                {
                    target_points[0, 0] = p[1];
                    target_points[0, 1] = p[0];
                }
                inputs = new List<FixedBufferOnnxValue> {
                    FixedBufferOnnxValue.CreateFromTensor(handle_points),
                    FixedBufferOnnxValue.CreateFromTensor(target_points)
                };

                // Training Step
                m_session.TrainStep(inputs, outputs2);

                // optimization step
                m_session.OptimizerStep();

                watch.Stop();
                Console.WriteLine($"{iter} / {n_iter}: Loss: {out_loss}   {watch.ElapsedMilliseconds} ms");

                // save the original image
                if (intermidiate_images != null)
                {
                    var bmp_iter = GetImage(out_img, pixels_memory, handles, targets);
                    //bmp_iter.Save(Path.Combine(out_path, $"Iter_{iter}.png"));

                    var info = new OptimizationIntermidiateInformation()
                    {
                        Image = bmp_iter,
                        CurrentIteration = iter,
                        TotalIterations = n_iter,
                    };
                    intermidiate_images(info);
                    //intermidiate_images(bmp_iter);
                }

                // Update the handle points with point tracking
                handles = PointTracking(out2_F, out_F0, handles, handles_0, r2);


            }

            //// cleanup
            foreach (var i in inputs)
                i.Dispose();
            foreach (var i in outputs2)
                i.Dispose();

            Console.WriteLine("Optimization Loop Done!");

            // get final image
            var final_bmp = GetImage(out_img, pixels_memory);

            // get optimized latent vector
            var output_latent = new DenseTensor<float>(latent.Dimensions);
            {
                int index;

                var parameter_trainable = m_state.GetParameter("latent_trainable");
                var parameter_untrainable = m_state.GetParameter("latent_untrainable");

                var shape_info_trainable = parameter_trainable.GetTensorTypeAndShape();
                var shape_info_untrainable = parameter_untrainable.GetTensorTypeAndShape();

                var shape_trainable = shape_info_trainable.Shape;
                var shape_untrainable = shape_info_untrainable.Shape;

                var tensor_trainable = parameter_trainable.GetTensorDataAsSpan<float>().ToArray();
                var tensor_untrainable = parameter_untrainable.GetTensorDataAsSpan<float>().ToArray();

                // update the trainable part
                index = 0;
                for (int i = 0; i < shape_trainable[1]; i++)
                {
                    for (int j = 0; j < shape_trainable[2]; j++)
                    {
                        output_latent[0, i, j] = tensor_trainable[index++];
                    }
                }

                // update the untrainable part
                index = 0;
                for (int i = 0; i < shape_untrainable[1]; i++)
                {
                    for (int j = 0; j < shape_untrainable[2]; j++)
                    {
                        output_latent[0, (int)(shape_trainable[1] + i), j] = tensor_untrainable[index++];
                    }
                }
            }

            return new Info()
            {
                Seed = -1,
                Image = final_bmp,
                Latent = output_latent,
            };

        }



    }
}
