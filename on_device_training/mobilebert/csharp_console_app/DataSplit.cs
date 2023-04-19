using System.Text.Json.Serialization;

namespace csharp_console_app
{
    /// <summary>
    /// Data structure to store a split of the data (ie, train, test, validation)
    /// </summary>
    public class DataSplit
    {
        [JsonPropertyName("input_ids_shape")]
        public long[] InputShape { get; set; }

        [JsonPropertyName("input_ids")]
        public Int64[] InputIds { get; set; }
        
        [JsonPropertyName("token_type_ids_shape")]
        public long[] TokenShape { get; set; }

        [JsonPropertyName("token_type_ids")]
        public Int64[] TokenTypeIds { get; set; }

        [JsonPropertyName("attention_mask_shape")]
        public long[] AttentionShape { get; set; }

        [JsonPropertyName("attention_mask")]
        public Int64[] AttentionMask { get; set;}

        [JsonPropertyName("labels_shape")]
        public long[] LabelsShape { get; set; }

        [JsonPropertyName("labels")]
        public Int64[] Labels { get; set; }
    }
}
