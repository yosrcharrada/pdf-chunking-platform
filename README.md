# ChunkMaster — PDF Chunking Platform

An advanced .NET 8 ASP.NET Core web application for intelligently chunking PDF documents into smaller pieces for downstream use cases such as graph database construction, RAG pipelines, and text analysis.

## Features

- **Four chunking strategies**
  - **NLP** — Sentence-aware splitting with configurable overlap
  - **Semantic** — Topic/paragraph boundary detection
  - **Regex** — Custom pattern-based splitting
  - **Hybrid** — Combines semantic paragraph detection with NLP sentence splitting

- **Multi-file upload** — Upload a single PDF or multiple PDFs at once with drag-and-drop support
- **Chunk metadata** — Each chunk shows word count, character count, and preview text
- **JSON export** — Download chunking results per document as structured JSON
- **Professional UI** — Yellow (#F5C518), white, and black design

## Tech Stack

- .NET 8, ASP.NET Core MVC
- [PdfPig](https://github.com/UglyToad/PdfPig) for PDF text extraction
- Self-contained Razor view (no external CDN dependencies)

## Getting Started

```bash
cd src/PdfChunkingPlatform
dotnet run
```

Then open http://localhost:5000 in your browser.

## Project Structure

```
src/PdfChunkingPlatform/
├── Controllers/
│   ├── HomeController.cs       # Entry point
│   └── ChunkingController.cs   # Process + Export actions
├── Models/
│   └── ChunkingModels.cs       # Request/result view models
├── Services/
│   ├── PdfExtractorService.cs  # PDF text extraction
│   └── ChunkingService.cs      # All chunking strategies
└── Views/
    └── Home/Index.cshtml       # Single-page UI
```