using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace mobilevit_console
{
    public class DataSplit
    {
        [JsonPropertyName("image_shape")]
        public long[] ImageShape { get; set; }

        [JsonPropertyName("image")]
        public double[] Image { get; set; }

        [JsonPropertyName("label_shape")]
        public long[] LabelShape { get; set; }

        [JsonPropertyName("label")]
        public Int64[] Label { get; set; }
    }
}
