using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.IO;


namespace DragGANApp
{
    public class Misc
    {
        public class OptionsColor
        {
            public int R { get; set; }
            public int G { get; set; }
            public int B { get; set; }
        }


        public class Options
        {
            public OptionsColor BGColor { get; set; }
            public int InitialSeed { get; set; }
            public int Iterations { get; set; }
            public string Model { get; set; }
        }

        public class ModelInfo
        {
            public string ModelName { get; set; }
            public List<int> ImageSize { get; set; }
            public int FFeatureSize { get; set; }
        }


        public static void SaveExampleOptionsFile(string filename)
        {
            //var options = new JsonSerializerOptions() { Fields= true };

            var app_options = new Options()
            {
                BGColor = new OptionsColor() { R = 100, G = 100, B = 100 },
                Iterations = 100,
            };

            JsonSerializerOptions options = new JsonSerializerOptions
            {
                Converters ={
                    new JsonStringEnumConverter()
                }
            };
            string jsonString = JsonSerializer.Serialize(app_options);

            File.WriteAllText(filename, jsonString);
        }

        public static Options LoadAppOptions()
        {
            var res = JsonSerializer.Deserialize<Options>(File.ReadAllText("Options.json"));
            return res;
        }

        public static ModelInfo LoadModelInfo(string fn)
        {
            var res = JsonSerializer.Deserialize<ModelInfo>(File.ReadAllText(fn));
            return res;
        }

    }
}
