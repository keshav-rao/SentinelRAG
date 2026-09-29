"""
CLI script to run document ingestion and chunking for SentinelRAG.
"""
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rich.console import Console
from rich.table import Table
from config.settings import settings
from src.ingestion.loaders import load_documents_from_directory
from src.ingestion.chunking import chunk_documents, save_chunks_to_json, ChunkingStrategy

console = Console()

def main():
    console.print("\n[bold cyan]=== SentinelRAG Ingestion Pipeline ===[/bold cyan]\n")
    console.print(f"Reading raw policies from: [green]{settings.raw_data_dir}[/green]")
    
    # 1. Load documents
    raw_docs = load_documents_from_directory(settings.raw_data_dir)
    if not raw_docs:
        console.print("[red]No documents found in raw directory.[/red]")
        return

    console.print(f"Loaded [bold yellow]{len(raw_docs)}[/bold yellow] raw document sections.\n")

    # 2. Chunk documents
    chunks = chunk_documents(
        documents=raw_docs,
        chunk_size=500,
        chunk_overlap=80,
        strategy=ChunkingStrategy.RECURSIVE
    )

    # 3. Display summary table
    table = Table(title="Ingestion & Chunking Summary")
    table.add_column("Document", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    table.add_column("Total Chars", justify="right", style="magenta")

    # Group by source
    doc_stats = {}
    for c in chunks:
        src = c.metadata.get("source", "Unknown")
        if src not in doc_stats:
            doc_stats[src] = {"count": 0, "chars": 0}
        doc_stats[src]["count"] += 1
        doc_stats[src]["chars"] += len(c.text)

    for doc_name, stats in doc_stats.items():
        table.add_row(doc_name, str(stats["count"]), str(stats["chars"]))

    console.print(table)

    # 4. Save to disk
    output_path = settings.processed_data_dir / "chunks.json"
    save_chunks_to_json(chunks, output_path)
    console.print(f"\n[bold green][OK] Successfully generated and saved {len(chunks)} chunks to:[/bold green] {output_path}\n")

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
