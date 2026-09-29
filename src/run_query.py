"""
Interactive and CLI Query Entrypoint for SentinelRAG.
Demonstrates the full self-evaluating RAG pipeline:
Guardrail -> Hybrid Search -> Rerank -> LLM Generation -> Critic Node -> Answer / Fallback.
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.graph.rag_graph import SentinelRAGPipeline

console = Console()

def run(query: str):
    console.print(f"\n[bold cyan]=== SentinelRAG Enterprise Query ===[/bold cyan]")
    console.print(f"[yellow]User Question:[/yellow] [bold white]{query}[/bold white]\n")

    console.print("[dim]Initializing SentinelRAG Graph and Ollama models...[/dim]")
    pipeline = SentinelRAGPipeline()

    console.print("[dim]Executing LangGraph pipeline (Guardrail -> Retrieve -> Rerank -> Generate -> Critic)...[/dim]")
    result = pipeline.query(query)

    # 1. Guardrail Status
    if not result.get("guardrail_passed", True):
        console.print(Panel(
            f"[bold red]BLOCKED BY GUARDRAIL[/bold red]\n\nReason: {result.get('guardrail_reason')}",
            title="[red]Security Gate[/red]",
            border_style="red"
        ))
        return

    # 2. Retrieved Sources Table
    sources = result.get("sources", [])
    if sources:
        table = Table(title="Retrieved & Reranked Policy Citations")
        table.add_column("Document", style="cyan")
        table.add_column("Section", style="magenta")
        table.add_column("Chunk ID", style="green")

        for s in sources:
            table.add_row(s.get("document", ""), str(s.get("section", "")), s.get("chunk_id", ""))
        console.print(table)

    # 3. Critic Evaluation Panel
    score = result.get("critic_score", 0.0)
    decision = result.get("critic_decision", "N/A")
    grounded = result.get("critic_grounded", False)
    hallucination = result.get("critic_hallucination", False)
    reason = result.get("critic_reason", "")
    retries = result.get("retry_count", 0)

    color = "green" if decision == "PASS" else "yellow"
    critic_details = (
        f"[bold]Decision:[/bold] [{color}]{decision}[/{color}] | "
        f"[bold]Confidence Score:[/bold] {score:.2f} | "
        f"[bold]Grounded:[/bold] {grounded} | "
        f"[bold]Hallucination Detected:[/bold] {hallucination} | "
        f"[bold]Retries:[/bold] {retries}\n\n"
        f"[bold]Auditor Justification:[/bold]\n{reason}"
    )
    console.print(Panel(critic_details, title="[blue]Critic & Hallucination Audit[/blue]", border_style=color))

    # 4. Final Answer Panel
    final_status = result.get("final_status", "grounded")
    status_color = "green" if final_status == "grounded" else "yellow"
    title = f"[{status_color}]SentinelRAG Answer ({final_status.upper()})[/{status_color}]"

    console.print(Panel(
        result.get("generated_answer", "No answer generated."),
        title=title,
        border_style=status_color
    ))
    console.print()

def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Ask SentinelRAG enterprise knowledge base.")
    parser.add_argument("query", nargs="?", type=str, help="The query to ask.")
    args = parser.parse_args()

    if args.query:
        run(args.query)
    else:
        # Default showcase question from user prompt
        default_q = "What is the maximum hotel reimbursement for a business trip?"
        console.print(f"[dim]No question provided. Running showcase question:[/dim]")
        run(default_q)

if __name__ == "__main__":
    main()
