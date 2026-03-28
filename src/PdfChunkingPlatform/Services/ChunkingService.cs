using System.Text.RegularExpressions;
using PdfChunkingPlatform.Models;

namespace PdfChunkingPlatform.Services;

public interface IChunkingService
{
    List<TextChunk> Chunk(string text, ChunkingStrategy strategy, int maxChunkSize, int overlap, string? regexPattern = null);
}

public class ChunkingService : IChunkingService
{
    public List<TextChunk> Chunk(string text, ChunkingStrategy strategy, int maxChunkSize, int overlap, string? regexPattern = null)
    {
        return strategy switch
        {
            ChunkingStrategy.NLP => ChunkByNlp(text, maxChunkSize, overlap),
            ChunkingStrategy.Semantic => ChunkBySemantic(text, maxChunkSize),
            ChunkingStrategy.Regex => ChunkByRegex(text, regexPattern, maxChunkSize),
            ChunkingStrategy.Hybrid => ChunkByHybrid(text, maxChunkSize, overlap),
            _ => ChunkByNlp(text, maxChunkSize, overlap)
        };
    }

    private List<TextChunk> ChunkByNlp(string text, int maxChunkSize, int overlap)
    {
        var sentencePattern = @"(?<=[.!?])\s+(?=[A-Z])";
        var sentences = Regex.Split(text.Trim(), sentencePattern)
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .Select(s => s.Trim())
            .ToList();

        return BuildChunksFromSegments(sentences, maxChunkSize, overlap);
    }

    private List<TextChunk> ChunkBySemantic(string text, int maxChunkSize)
    {
        var paragraphPattern = @"\n\s*\n|\r\n\s*\r\n";
        var paragraphs = Regex.Split(text.Trim(), paragraphPattern)
            .Where(p => !string.IsNullOrWhiteSpace(p))
            .Select(p => p.Trim())
            .ToList();

        var chunks = new List<TextChunk>();
        var currentChunk = new System.Text.StringBuilder();
        int index = 0;

        foreach (var paragraph in paragraphs)
        {
            if (currentChunk.Length + paragraph.Length > maxChunkSize && currentChunk.Length > 0)
            {
                var chunkText = currentChunk.ToString().Trim();
                chunks.Add(CreateChunk(index++, chunkText));
                currentChunk.Clear();
            }

            if (paragraph.Length > maxChunkSize)
            {
                var subChunks = ChunkByNlp(paragraph, maxChunkSize, 0);
                foreach (var sub in subChunks)
                {
                    sub.Index = index++;
                    chunks.Add(sub);
                }
            }
            else
            {
                if (currentChunk.Length > 0) currentChunk.Append("\n\n");
                currentChunk.Append(paragraph);
            }
        }

        if (currentChunk.Length > 0)
            chunks.Add(CreateChunk(index, currentChunk.ToString().Trim()));

        return chunks;
    }

    private List<TextChunk> ChunkByRegex(string text, string? pattern, int maxChunkSize)
    {
        if (string.IsNullOrWhiteSpace(pattern))
            return ChunkByNlp(text, maxChunkSize, 0);

        try
        {
            var segments = Regex.Split(text.Trim(), pattern)
                .Where(s => !string.IsNullOrWhiteSpace(s))
                .Select(s => s.Trim())
                .ToList();
            return BuildChunksFromSegments(segments, maxChunkSize, 0);
        }
        catch
        {
            return ChunkByNlp(text, maxChunkSize, 0);
        }
    }

    private List<TextChunk> ChunkByHybrid(string text, int maxChunkSize, int overlap)
    {
        var paragraphPattern = @"\n\s*\n|\r\n\s*\r\n";
        var paragraphs = Regex.Split(text.Trim(), paragraphPattern)
            .Where(p => !string.IsNullOrWhiteSpace(p))
            .Select(p => p.Trim())
            .ToList();

        var allChunks = new List<TextChunk>();
        int globalIndex = 0;

        foreach (var paragraph in paragraphs)
        {
            var subChunks = ChunkByNlp(paragraph, maxChunkSize, overlap);
            foreach (var chunk in subChunks)
            {
                chunk.Index = globalIndex++;
                allChunks.Add(chunk);
            }
        }

        return allChunks;
    }

    private List<TextChunk> BuildChunksFromSegments(List<string> segments, int maxChunkSize, int overlap)
    {
        var chunks = new List<TextChunk>();
        var currentChunk = new System.Text.StringBuilder();
        int index = 0;
        var overlapBuffer = new List<string>();

        foreach (var segment in segments)
        {
            if (currentChunk.Length + segment.Length + 1 > maxChunkSize && currentChunk.Length > 0)
            {
                var chunkText = currentChunk.ToString().Trim();
                chunks.Add(CreateChunk(index++, chunkText));
                
                currentChunk.Clear();
                if (overlap > 0 && overlapBuffer.Count > 0)
                {
                    var overlapText = string.Join(" ", overlapBuffer);
                    if (overlapText.Length <= overlap)
                    {
                        currentChunk.Append(overlapText);
                        currentChunk.Append(" ");
                    }
                }
                overlapBuffer.Clear();
            }

            if (currentChunk.Length > 0) currentChunk.Append(" ");
            currentChunk.Append(segment);
            overlapBuffer.Add(segment);

            while (string.Join(" ", overlapBuffer).Length > overlap && overlapBuffer.Count > 1)
                overlapBuffer.RemoveAt(0);
        }

        if (currentChunk.Length > 0)
            chunks.Add(CreateChunk(index, currentChunk.ToString().Trim()));

        return chunks;
    }

    private static TextChunk CreateChunk(int index, string text)
    {
        var words = text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
        return new TextChunk
        {
            Index = index,
            Text = text,
            WordCount = words.Length,
            CharCount = text.Length
        };
    }
}
