using Microsoft.ML.OnnxRuntime;
using System.Text.Json;

namespace csharp_console_app
{
    /// <summary>
    /// Parses JSON files following the DataSplit format into OnnxValues
    /// </summary>
    public class DataLoader
    {
        /// <summary>
        /// Creates a DataSplit object from the specified filePath
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public DataSplit ReadFile(string filePath)
        {
            string json = File.ReadAllText(filePath);
            return JsonSerializer.Deserialize<DataSplit>(json);
        }

       // public List<FixedBufferOnnxValue> 
    }
}
