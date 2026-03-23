"""TSC v2.0 CLI — Feature Evaluation from the command line."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

console = Console()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise from third-party libs
    for lib in ("httpx", "httpcore", "openai", "anthropic", "urllib3"):
        logging.getLogger(lib).setLevel(logging.WARNING)


@click.group()
@click.version_option(version="2.0.0", prog_name="TSC")
def main():
    """TSC v2.0 — Technology-Stakeholder-Consensus Feature Evaluation Pipeline."""
    pass


@main.command()
@click.option("--interviews", type=click.Path(exists=True), help="Customer interviews file")
@click.option("--support", type=click.Path(exists=True), help="Support tickets file (CSV)")
@click.option("--analytics", type=click.Path(exists=True), help="Analytics data file")
@click.option("--context", type=click.Path(exists=True), help="Company context JSON")
@click.option("--proposal", type=click.Path(exists=True), help="Feature proposal JSON")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "groq", "openrouter"]), help="LLM provider")
@click.option("--model", type=str, help="Model name (provider-specific)")
@click.option("--output", "-o", type=click.Path(), default="recommendation.json", help="Output file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def evaluate(
    interviews: str | None,
    support: str | None,
    analytics: str | None,
    context: str | None,
    proposal: str | None,
    provider: str | None,
    model: str | None,
    output: str,
    verbose: bool,
):
    """Run a complete feature evaluation."""
    setup_logging(verbose)

    if not any([interviews, support, analytics, context, proposal]):
        console.print("[red]Error:[/red] At least one input file is required.")
        console.print("Run [bold]tsc evaluate --help[/bold] for usage.")
        sys.exit(1)

    # Banner
    console.print()
    console.print(Panel.fit(
        "[bold cyan]TSC v2.0[/bold cyan] — Technology-Stakeholder-Consensus\n"
        "[dim]Feature Evaluation Pipeline[/dim]",
        border_style="cyan",
    ))

    # Show config
    table = Table(show_header=False, border_style="dim")
    table.add_column("Setting", style="dim")
    table.add_column("Value")
    if provider:
        table.add_row("Provider", provider)
    if model:
        table.add_row("Model", model)
    table.add_row("Output", output)
    if interviews:
        table.add_row("Interviews", interviews)
    if support:
        table.add_row("Support", support)
    if analytics:
        table.add_row("Analytics", analytics)
    if context:
        table.add_row("Context", context)
    if proposal:
        table.add_row("Proposal", proposal)
    console.print(table)
    console.print()

    # Override settings
    from tsc.config import LLMProvider, settings

    if provider:
        settings.llm_provider = LLMProvider(provider)
    if model:
        settings.llm_model = model

    # Run pipeline
    from tsc.pipeline.orchestrator import TSCPipeline

    pipeline = TSCPipeline()

    # Progress display
    layer_status: dict[int, str] = {}

    def on_progress(layer: int, name: str, status: str, details: dict):
        emoji = "✅" if status == "done" else "⏳"
        layer_status[layer] = f"{emoji} Layer {layer}/8: {name}"
        if details:
            info = ", ".join(f"{k}: {v}" for k, v in details.items())
            layer_status[layer] += f" [{info}]"
        # Clear and redraw
        console.print(layer_status[layer])

    pipeline.set_progress_callback(on_progress)

    try:
        result = asyncio.run(
            pipeline.evaluate(
                interviews=interviews,
                support=support,
                analytics=analytics,
                context=context,
                proposal=proposal,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Pipeline error:[/red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)

    # Save output
    out_path = Path(output)
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    # Summary
    console.print()
    verdict_color = {
        "APPROVED": "green",
        "CONDITIONAL_APPROVE": "yellow",
        "REJECTED": "red",
    }.get(result.final_verdict, "white")

    console.print(Panel(
        f"[bold {verdict_color}]{result.final_verdict}[/bold {verdict_color}]\n"
        f"Confidence: {result.overall_confidence:.0%}\n"
        f"Feature: {result.feature_name}\n"
        f"Time: {result.metadata.total_time_minutes:.1f} min\n"
        f"Tokens: {result.metadata.total_tokens_used:,}\n"
        f"\nSaved to: {out_path}",
        title="[bold]EVALUATION RESULT[/bold]",
        border_style=verdict_color,
    ))

    # Pillar verdicts
    if result.verdicts_by_pillar:
        pillar_table = Table(title="Pillar Verdicts", show_lines=True)
        pillar_table.add_column("Pillar", style="bold")
        pillar_table.add_column("Verdict")
        pillar_table.add_column("Score")
        for name, pillar in result.verdicts_by_pillar.items():
            pillar_table.add_row(name, pillar.verdict, f"{pillar.score:.2f}")
        console.print(pillar_table)

    # Leadership summary
    if result.summary_for_leadership:
        console.print(Panel(
            result.summary_for_leadership,
            title="[bold]Leadership Summary[/bold]",
            border_style="blue",
        ))

    console.print()


if __name__ == "__main__":
    main()
