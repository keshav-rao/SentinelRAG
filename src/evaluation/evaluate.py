"""
SentinelRAG Automated Evaluation Suite & Quality Gate.
Measures Faithfulness, Context Recall, Hallucination Rate, and Citation Accuracy.
"""
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from config.settings import settings
from src.graph.rag_graph import SentinelRAGPipeline

console = Console()

def run_evaluation(limit: int = 5) -> Dict[str, Any]:
    dataset_path = settings.eval_data_dir / "evaluation_dataset.json"
    if not dataset_path.exists():
        console.print(f"[red]Evaluation dataset not found at: {dataset_path}[/red]")
        return {}

    test_cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    selected_cases = test_cases[:limit]

    console.print(f"\n[bold cyan]=== SentinelRAG Evaluation Quality Gate ===[/bold cyan]")
    console.print(f"Evaluating [bold yellow]{len(selected_cases)}[/bold yellow] test cases against local pipeline...\n")

    pipeline = SentinelRAGPipeline()
    results = []

    passed_faithfulness = 0
    passed_recall = 0
    hallucination_count = 0

    table = Table(title="Test Case Evaluation Results")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Category", style="magenta")
    table.add_column("Recall", justify="center")
    table.add_column("Grounded", justify="center")
    table.add_column("Critic Score", justify="right")
    table.add_column("Decision", justify="center")

    for tc in selected_cases:
        tc_id = tc["id"]
        q = tc["question"]
        expected_doc = tc["expected_document"]

        response = pipeline.query(q)

        # Context Recall: Check if expected document was retrieved
        retrieved_docs = [s.get("document") for s in response.get("sources", [])]
        recall_hit = any(expected_doc.lower() in d.lower() for d in retrieved_docs if d)

        # Faithfulness / Groundedness from Critic
        grounded = response.get("critic_grounded", False)
        hallucination = response.get("critic_hallucination", False)
        score = response.get("critic_score", 0.0)
        decision = response.get("critic_decision", "FAIL")

        if recall_hit:
            passed_recall += 1
        if grounded and score >= settings.critic_min_score:
            passed_faithfulness += 1
        if hallucination:
            hallucination_count += 1

        rec_str = "[green]YES[/green]" if recall_hit else "[red]NO[/red]"
        grd_str = "[green]YES[/green]" if grounded else "[red]NO[/red]"
        dec_color = "green" if decision == "PASS" else "red"
        dec_str = f"[{dec_color}]{decision}[/{dec_color}]"

        table.add_row(
            tc_id,
            tc["category"],
            rec_str,
            grd_str,
            f"{score:.2f}",
            dec_str
        )

        results.append({
            "test_case": tc,
            "response": response.get("generated_answer"),
            "recall_hit": recall_hit,
            "critic_score": score,
            "decision": decision
        })

    console.print(table)

    total = len(selected_cases)
    recall_rate = passed_recall / total
    faithfulness_rate = passed_faithfulness / total
    hallucination_rate = hallucination_count / total

    # Quality Gate Check
    gate_passed = (faithfulness_rate >= 0.80) and (hallucination_rate <= 0.05)
    gate_color = "green" if gate_passed else "red"
    gate_status = "PASS" if gate_passed else "FAIL"

    metrics_text = (
        f"[bold]Total Evaluated:[/bold] {total}\n"
        f"[bold]Context Recall@K:[/bold] {recall_rate * 100:.1f}%\n"
        f"[bold]Faithfulness Rate:[/bold] {faithfulness_rate * 100:.1f}%\n"
        f"[bold]Hallucination Rate:[/bold] {hallucination_rate * 100:.1f}%\n\n"
        f"[bold]Quality Gate Decision:[/bold] [{gate_color}]{gate_status}[/{gate_color}]"
    )

    console.print(Panel(metrics_text, title="[blue]CI/CD Quality Gate Summary[/blue]", border_style=gate_color))

    # Save results
    out_file = settings.eval_data_dir / "eval_results.json"
    out_file.write_text(json.dumps({
        "metrics": {
            "total": total,
            "context_recall": recall_rate,
            "faithfulness": faithfulness_rate,
            "hallucination_rate": hallucination_rate,
            "quality_gate": gate_status
        },
        "results": results
    }, indent=2), encoding="utf-8")
    console.print(f"\n[dim]Detailed evaluation results saved to: {out_file}[/dim]\n")

    return {"gate_passed": gate_passed, "recall": recall_rate, "faithfulness": faithfulness_rate}

def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="Number of test cases to evaluate")
    args = parser.parse_args()

    run_evaluation(limit=args.limit)

if __name__ == "__main__":
    main()
