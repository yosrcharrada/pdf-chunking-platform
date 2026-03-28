using System.ComponentModel.DataAnnotations;

namespace PdfChunkingPlatform.Models;

public enum ChunkingStrategy
{
    NLP,
    Semantic,
    Regex,
    Hybrid
}

public class ChunkingRequest
{
    [Required]
    public List<IFormFile> Files { get; set; } = new();
    
    public ChunkingStrategy Strategy { get; set; } = ChunkingStrategy.NLP;
    
    [Range(50, 5000)]
    public int MaxChunkSize { get; set; } = 500;
    
    [Range(0, 500)]
    public int Overlap { get; set; } = 50;
    
    public string? RegexPattern { get; set; }
}

public class TextChunk
{
    public int Index { get; set; }
    public string Text { get; set; } = string.Empty;
    public int WordCount { get; set; }
    public int CharCount { get; set; }
    public int PageNumber { get; set; }
}

public class DocumentChunkResult
{
    public string FileName { get; set; } = string.Empty;
    public int TotalPages { get; set; }
    public int TotalChunks { get; set; }
    public ChunkingStrategy Strategy { get; set; }
    public List<TextChunk> Chunks { get; set; } = new();
    public long ProcessingTimeMs { get; set; }
}

public class ChunkingViewModel
{
    public ChunkingRequest Request { get; set; } = new();
    public List<DocumentChunkResult> Results { get; set; } = new();
    public bool HasResults => Results.Any();
    public string? ErrorMessage { get; set; }
}
