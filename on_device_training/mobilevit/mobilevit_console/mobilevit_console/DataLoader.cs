using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing.Printing;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using static System.Net.Mime.MediaTypeNames;

namespace mobilevit_console
{
    /// <summary>
    /// Responsible for processing the FER image dataset in batches into tensors.
    /// </summary>
    public class DataLoader
    {

        public static List<string> EMOTIONSLABELS = new List<string>() // reflects structure and labels from training data
        {
            "neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"
        };

        public List<string> images = new List<string>();
        public List<int> labels = new List<int>();

        int batchIndex = 0;

        static int IMAGEDIM = 256; // represents the width and height that images should be resized to
        // the IMAGEDIM should match the input dimensions representing image width and height that your training model was generated with
        static int MAXCOLORVALUE = 255; // used for scaling the RGB values

        /// <summary>
        /// Pulls the names of all the images and shuffles them into a list. A corresponding list of labels will be simultaneously created.
        /// - create two lists, one of the images, one of the labels
        /// - shuffle the indexes so that the two are shuffled in the same way
        /// </summary>
        /// <param name="dataFilePath">Path to the FER directory containing 7 unzipped folders labelled with their corresponding emotion</param>
        public void loadFER(string dataFilePath)
        {
            for (int i = 0; i < EMOTIONSLABELS.Count; i++)
            {
                loadFileNames(dataFilePath, EMOTIONSLABELS[i], i);
            }

            var indexes = Enumerable.Range(0, images.Count).ToList();
            var rand = new Random();
            var shuffled = indexes.OrderBy(i => rand.Next());

            images = shuffled.Select(index => images[index]).ToList();
            labels = shuffled.Select(index => labels[index]).ToList();
        }

        public void trimFER(int imagesNum)
        {
            images = images.GetRange(0, imagesNum);
            labels = labels.GetRange(0, imagesNum);
        }

        /// <summary>
        /// 
        /// Processes a single directory for an emotion for the loadFER method.
        /// Adds the corresponding image paths and labels to the images and labels fields.
        /// </summary>
        /// <param name="dataFilePath"></param>
        /// <param name="emotion"></param>
        /// <param name="label"></param>
        public void loadFileNames(string dataFilePath, string emotion, int label)
        {
            string emotionDir = Path.Combine(dataFilePath, emotion);
            string[] imagePaths = Directory.GetFiles(emotionDir, "*.png", SearchOption.TopDirectoryOnly);
            images.AddRange(imagePaths);
            labels.AddRange(Enumerable.Repeat(label, imagePaths.Length));
        }

        public Tensor<float> imageProcessingForInference(string filePath)
        {
            using (var image = new Bitmap(System.Drawing.Image.FromFile(filePath)))
            {
                var pixel = image.GetPixel(0, 0);
                var resized = ResizeImage(image, 256, 256);
                return convertImageToTensorForInference(resized);
            }
        }

        public int getNumSteps(int batchSize)
        {
            return images.Count / batchSize;
        }

        /// <summary>
        /// Processes the next [batchSize] number of images and converts them to a 
        /// list of FixedBufferOnnxValue to be fed into a model.
        /// </summary>
        /// <param name="batchSize"></param>
        /// <returns></returns>
        public List<FixedBufferOnnxValue> generateBatchInput(int batchSize)
        {
            Tensor<float> batchPixelVals = new DenseTensor<float>(new[] { batchSize, 3, IMAGEDIM, IMAGEDIM });
            Tensor<Int64> batchLabels = new DenseTensor<Int64>(new[] { batchSize });

            for (int i = 0; i < batchSize; i++)
            {
                pngPathToTensor(images[batchIndex], batchPixelVals, i);
                batchLabels[i] = labels[batchIndex];
                batchIndex += 1;

                if (batchIndex >= images.Count)
                {
                    batchIndex = 0;
                }
            }

            return new List<FixedBufferOnnxValue> { 
                FixedBufferOnnxValue.CreateFromTensor(batchPixelVals), 
                FixedBufferOnnxValue.CreateFromTensor(batchLabels)
            };
        }

        private void pngPathToTensor(string path, Tensor<float> tensor, int imageIndex)
        {
            using (var image = new Bitmap(System.Drawing.Image.FromFile(path)))
            {
                var resized = ResizeImage(image, IMAGEDIM, IMAGEDIM);
                populateTensorWithImage(resized, tensor, imageIndex);
            }
        }

        public Tensor<float> convertImageToTensorForInference(Bitmap img)
        {
            Tensor<float> inferenceTensor = new DenseTensor<float>(new[] { 1, 3, img.Height, img.Width });

            populateTensorWithImage(img, inferenceTensor, 0);

            return inferenceTensor;
        }

        public void populateTensorWithImage(Bitmap img, Tensor<float> tensor, int imageIndex)
        {
            for (int i = 0; i < img.Height; i++)
            {
                for (int j = 0; j < img.Width; j++)
                {
                    var pixel = img.GetPixel(i, j);
                    tensor[imageIndex, 0, i, j] = scaleColorValue(pixel.R);
                    tensor[imageIndex, 1, i, j] = scaleColorValue(pixel.G);
                    tensor[imageIndex, 2, i, j] = scaleColorValue(pixel.B);
                }
            }
        }

        public Bitmap ResizeImage(System.Drawing.Image image, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }

        public float scaleColorValue(byte val)
        {
            return val / MAXCOLORVALUE;
        }
    }
}