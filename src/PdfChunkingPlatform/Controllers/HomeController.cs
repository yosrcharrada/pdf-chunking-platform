using Microsoft.AspNetCore.Mvc;
using PdfChunkingPlatform.Models;

namespace PdfChunkingPlatform.Controllers;

public class HomeController : Controller
{
    public IActionResult Index()
    {
        return View(new ChunkingViewModel());
    }
}
