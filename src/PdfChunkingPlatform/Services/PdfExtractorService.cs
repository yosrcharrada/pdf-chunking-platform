using UglyToad.PdfPig;

namespace PdfChunkingPlatform.Services;

public interface IPdfExtractorService
{
    Task<(string FullText, int PageCount, Dictionary<int, string> PageTexts)> ExtractTextAsync(Stream pdfStream);
}

public class PdfExtractorService : IPdfExtractorService
{
    public Task<(string FullText, int PageCount, Dictionary<int, string> PageTexts)> ExtractTextAsync(Stream pdfStream)
    {
        var pageTexts = new Dictionary<int, string>();
        var fullTextBuilder = new System.Text.StringBuilder();

        using var document = PdfDocument.Open(pdfStream);
        foreach (var page in document.GetPages())
        {
            var pageText = string.Join(" ", page.GetWords().Select(w => w.Text));
            pageTexts[page.Number] = pageText;
            fullTextBuilder.AppendLine(pageText);
        }

        var fullText = fullTextBuilder.ToString();
        return Task.FromResult((fullText, document.NumberOfPages, pageTexts));
    }
}
