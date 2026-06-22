using System.Collections.Generic;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;
using Newtonsoft.Json.Linq;

namespace VisionFlow.Graph
{
    // ── JSON-RPC 2.0 envelope ───────────────────────────────────────────────

    public sealed class JsonRpcMessage
    {
        [JsonProperty("jsonrpc")] public string Jsonrpc { get; set; } = "2.0";
        [JsonProperty("method")]  public string Method  { get; set; }
        [JsonProperty("id")]      public string Id      { get; set; }
        [JsonProperty("params")]  public JToken Params { get; set; }
        [JsonProperty("result")]  public JToken Result { get; set; }
        [JsonProperty("error")]   public JsonRpcError Error  { get; set; }
    }

    public sealed class JsonRpcError
    {
        [JsonProperty("code")]    public int    Code    { get; set; }
        [JsonProperty("message")] public string Message { get; set; }
    }

    // ── Graph node ─────────────────────────────────────────────────────────

    /// <summary>3-D node as streamed by VisionClaw CUDA layout.</summary>
    public sealed class GraphNode
    {
        [JsonProperty("id")]          public string Id          { get; set; }
        [JsonProperty("label")]       public string Label       { get; set; }
        [JsonProperty("type")]        public string Type        { get; set; }  // OWL class URI
        [JsonProperty("x")]           public float  X           { get; set; }
        [JsonProperty("y")]           public float  Y           { get; set; }
        [JsonProperty("z")]           public float  Z           { get; set; }
        [JsonProperty("mass")]        public float  Mass        { get; set; } = 1f;
        [JsonProperty("community")]   public int    Community   { get; set; }
        [JsonProperty("pagerank")]    public float  PageRank    { get; set; }
        [JsonProperty("provenance")]  public string Provenance  { get; set; } // Nostr event id
        [JsonProperty("properties")]  public Dictionary<string, JToken> Properties { get; set; }
    }

    // ── Graph edge ─────────────────────────────────────────────────────────

    /// <summary>OWL relationship edge.</summary>
    public sealed class GraphEdge
    {
        [JsonProperty("id")]           public string Id           { get; set; }
        [JsonProperty("source")]       public string Source       { get; set; }
        [JsonProperty("target")]       public string Target       { get; set; }
        [JsonProperty("relation")]     public string Relation     { get; set; }  // e.g. "subClassOf"
        [JsonProperty("weight")]       public float  Weight       { get; set; } = 1f;
        [JsonProperty("provenance")]   public string Provenance   { get; set; }
    }

    // ── Graph snapshot ─────────────────────────────────────────────────────

    public sealed class GraphSnapshot
    {
        [JsonProperty("nodes")] public List<GraphNode> Nodes { get; set; } = new();
        [JsonProperty("edges")] public List<GraphEdge> Edges { get; set; } = new();
        [JsonProperty("tick")]  public long            Tick  { get; set; }
    }

    // ── Delta update (hot-path; only changed nodes) ────────────────────────

    public sealed class GraphDelta
    {
        [JsonProperty("updated_nodes")] public List<GraphNode> UpdatedNodes { get; set; } = new();
        [JsonProperty("updated_edges")] public List<GraphEdge> UpdatedEdges { get; set; } = new();
        [JsonProperty("removed_nodes")] public List<string>    RemovedNodes { get; set; } = new();
        [JsonProperty("removed_edges")] public List<string>    RemovedEdges { get; set; } = new();
        [JsonProperty("tick")]          public long            Tick         { get; set; }
    }

    // ── Shared JSON options (reuse to avoid repeated alloc) ───────────────

    public static class VisionFlowJson
    {
        public static readonly JsonSerializerSettings Settings = new()
        {
            NullValueHandling = NullValueHandling.Ignore,
            Formatting = Formatting.None,
        };

        public static T Deserialize<T>(string json) =>
            JsonConvert.DeserializeObject<T>(json, Settings);

        public static string Serialize<T>(T value) =>
            JsonConvert.SerializeObject(value, Settings);
    }
}
