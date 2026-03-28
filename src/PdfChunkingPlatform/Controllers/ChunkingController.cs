using System.Diagnostics;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using PdfChunkingPlatform.Models;
using PdfChunkingPlatform.Services;

namespace PdfChunkingPlatform.Controllers;

public class ChunkingController : Controller
{
    private readonly IPdfExtractorService _pdfExtractor;
    private readonly IChunkingService _chunkingService;
    private readonly ILogger<ChunkingController> _logger;

    public ChunkingController(IPdfExtractorService pdfExtractor, IChunkingService chunkingService, ILogger<ChunkingController> logger)
    {
        _pdfExtractor = pdfExtractor;
        _chunkingService = chunkingService;
        _logger = logger;
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    [RequestSizeLimit(104857600)]
    public async Task<IActionResult> Process(ChunkingRequest request)
    {
        var viewModel = new ChunkingViewModel { Request = request };

        if (!ModelState.IsValid || request.Files == null || !request.Files.Any())
        {
            viewModel.ErrorMessage = "Please upload at least one PDF file.";
            return View("~/Views/Home/Index.cshtml", viewModel);
        }

        var pdfFiles = request.Files.Where(f => f.ContentType == "application/pdf" || 
            Path.GetExtension(f.FileName).ToLower() == ".pdf").ToList();

        if (!pdfFiles.Any())
        {
            viewModel.ErrorMessage = "Only PDF files are supported.";
            return View("~/Views/Home/Index.cshtml", viewModel);
        }

        foreach (var file in pdfFiles)
        {
            try
            {
                var sw = Stopwatch.StartNew();
                using var stream = file.OpenReadStream();
                var (fullText, pageCount, _) = await _pdfExtractor.ExtractTextAsync(stream);
                
                var chunks = _chunkingService.Chunk(fullText, request.Strategy, request.MaxChunkSize, request.Overlap, request.RegexPattern);
                sw.Stop();

                viewModel.Results.Add(new DocumentChunkResult
                {
                    FileName = file.FileName,
                    TotalPages = pageCount,
                    TotalChunks = chunks.Count,
                    Strategy = request.Strategy,
                    Chunks = chunks,
                    ProcessingTimeMs = sw.ElapsedMilliseconds
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing file {FileName}", file.FileName);
                viewModel.ErrorMessage = $"Error processing {file.FileName}: {ex.Message}";
            }
        }

        TempData["Results"] = JsonSerializer.Serialize(viewModel.Results);
        return View("~/Views/Home/Index.cshtml", viewModel);
    }

    [HttpGet]
    public IActionResult Export(int documentIndex)
    {
        if (TempData["Results"] is string json)
        {
            TempData.Keep("Results");
            var results = JsonSerializer.Deserialize<List<DocumentChunkResult>>(json);
            if (results != null && documentIndex < results.Count)
            {
                var result = results[documentIndex];
                var exportJson = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
                return File(System.Text.Encoding.UTF8.GetBytes(exportJson), "application/json", 
                    $"{Path.GetFileNameWithoutExtension(result.FileName)}_chunks.json");
            }
        }
        return RedirectToAction("Index", "Home");
    }
}
