using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace mobilevit_console
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
