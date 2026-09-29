"""
Indexes processed chunks into local Qdrant vector store.
"""
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rich.console import Console
from config.settings import settings
from src.ingestion.chunking import load_chunks_from_json
from src.retrieval.vector_store import SentinelVectorStore

console = Console()

def main():
    console.print("\n[bold cyan]=== SentinelRAG Qdrant Indexer ===[/bold cyan]\n")
    chunks_path = settings.processed_data_dir / "chunks.json"
    
    if not chunks_path.exists():
        console.print(f"[red]Chunks file not found at: {chunks_path}[/red]")
        console.print("Please run: [yellow]python -m src.ingestion.run_ingestion[/yellow] first.")
        return

    chunks = load_chunks_from_json(chunks_path)
    console.print(f"Loaded [bold green]{len(chunks)}[/bold green] chunks from {chunks_path}")
    console.print(f"Connecting to Qdrant (local storage: {settings.qdrant_local_path})...")
    
    vector_store = SentinelVectorStore()
    console.print("Embedding documents using Ollama [bold magenta]nomic-embed-text[/bold magenta]...")
    count = vector_store.index_chunks(chunks)
    console.print(f"\n[bold green][OK] Successfully indexed {count} vectors into Qdrant collection '{settings.qdrant_collection_name}'![/bold green]\n")

if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
