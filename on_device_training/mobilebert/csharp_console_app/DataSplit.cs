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
        public Int32[] TokenTypeIds { get; set; }

        [JsonPropertyName("attention_mask_shape")]
        public long[] AttentionShape { get; set; }

        [JsonPropertyName("attention_mask")]
        public Int32[] AttentionMask { get; set;}

        [JsonPropertyName("special_tokens_mask_shape")]
        public long[] SpecialShape { get; set; }

        [JsonPropertyName("special_tokens_mask")]
        public Int32[] SpecialTokensMask { get; set; }
    }
}
