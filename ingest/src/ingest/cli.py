"""Command-line interface for the ingest module."""

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ingest.config import IngestConfig
from ingest.pipeline import IngestPipeline
from ingest.utils.logging import setup_logging

app = typer.Typer(
    name="ingest",
    help="Book ingestion tool for the librarian system",
    add_completion=False,
)
console = Console()


def get_config(config_path: str | None = None) -> IngestConfig:
    """Load configuration from file or defaults.

    Args:
        config_path: Optional path to config file

    Returns:
        IngestConfig instance
    """
    if config_path:
        return IngestConfig.load(config_path)
    return IngestConfig()


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to book file or directory"),
    config: str = typer.Option(None, "--config", "-c", help="Path to config file"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r", help="Search subdirectories"),
    book_id: str = typer.Option(None, "--id", help="Custom book ID (single file only)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
) -> None:
    """Ingest a book or directory of books into the vector database."""
    log_level = "DEBUG" if debug else ("INFO" if verbose else "WARNING")
    setup_logging(log_level)

    cfg = get_config(config)
    pipeline = IngestPipeline(cfg)

    path_obj = Path(path)

    if path_obj.is_file():
        console.print(f"[bold blue]Ingesting file:[/] {path_obj.name}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing...", total=None)

            try:
                result = asyncio.run(pipeline.ingest_book(path_obj, book_id))
                progress.update(task, completed=True)

                console.print("\n[bold green]Success![/]")
                console.print(f"  Book ID: {result.id}")
                console.print(f"  Title: {result.analysis.bibliography.title}")
                console.print(f"  Chunks: {result.total_chunks}")
            except Exception as e:
                progress.update(task, completed=True)
                console.print(f"\n[bold red]Error:[/] {e}")
                raise typer.Exit(1) from None

    elif path_obj.is_dir():
        console.print(f"[bold blue]Ingesting directory:[/] {path_obj}")

        try:
            results = asyncio.run(pipeline.ingest_directory(path_obj, recursive))

            console.print(f"\n[bold green]Ingested {len(results)} books[/]")

            table = Table(title="Ingested Books")
            table.add_column("ID", style="cyan")
            table.add_column("Title", style="green")
            table.add_column("Chunks", justify="right")

            for result in results:
                table.add_row(
                    result.id[:8] + "...",
                    result.analysis.bibliography.title[:50],
                    str(result.total_chunks),
                )

            console.print(table)
        except Exception as e:
            console.print(f"\n[bold red]Error:[/] {e}")
            raise typer.Exit(1) from None

    else:
        console.print(f"[bold red]Error:[/] Path not found: {path}")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    config: str = typer.Option(None, "--config", "-c", help="Path to config file"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results"),
    books_only: bool = typer.Option(False, "--books", "-b", help="Search books only (not chunks)"),
    tag: list[str] = typer.Option(None, "--tag", "-t", help="Filter by tag"),  # noqa: B008
    genre: list[str] = typer.Option(None, "--genre", "-g", help="Filter by genre"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Search the book database."""
    setup_logging("WARNING")

    cfg = get_config(config)
    pipeline = IngestPipeline(cfg)

    filters = {}
    if tag:
        filters["tags"] = list(tag)
    if genre:
        filters["genres"] = list(genre)

    if books_only:
        results = pipeline.search_books(query, limit, filters or None)
    else:
        results = pipeline.search(query, limit, filters or None)

    if json_output:
        output = [
            {
                "id": r.id,
                "score": r.score,
                "title": r.title,
                "book_id": r.book_id,
                "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                "chapter": r.chapter_title,
            }
            for r in results
        ]
        console.print(json.dumps(output, indent=2))
    else:
        if not results:
            console.print("[yellow]No results found[/]")
            return

        table = Table(title=f"Search Results for: {query}")
        table.add_column("Score", style="cyan", justify="right")
        table.add_column("Title", style="green")
        table.add_column("Chapter", style="blue")
        table.add_column("Content Preview")

        for r in results:
            preview = r.content[:100] + "..." if len(r.content) > 100 else r.content
            table.add_row(
                f"{r.score:.3f}",
                r.title[:30] + "..." if len(r.title) > 30 else r.title,
                (r.chapter_title[:20] + "...") if r.chapter_title and len(r.chapter_title) > 20 else (r.chapter_title or "-"),
                preview,
            )

        console.print(table)


@app.command()
def list_books(
    config: str = typer.Option(None, "--config", "-c", help="Path to config file"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """List all ingested books."""
    setup_logging("WARNING")

    cfg = get_config(config)
    pipeline = IngestPipeline(cfg)

    books = pipeline.list_books()

    if json_output:
        console.print(json.dumps(books, indent=2))
    else:
        if not books:
            console.print("[yellow]No books in database[/]")
            return

        table = Table(title="Ingested Books")
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="green")
        table.add_column("Authors", style="blue")
        table.add_column("Tags")
        table.add_column("Chunks", justify="right")

        for book in books:
            table.add_row(
                book["book_id"][:8] + "...",
                book["title"][:40] + "..." if len(book.get("title", "")) > 40 else book.get("title", "Unknown"),
                ", ".join(book.get("authors", []))[:30],
                ", ".join(book.get("tags", [])[:3]),
                str(book.get("total_chunks", 0)),
            )

        console.print(table)


@app.command()
def delete(
    book_id: str = typer.Argument(..., help="Book ID to delete"),
    config: str = typer.Option(None, "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a book from the database."""
    setup_logging("WARNING")

    if not force:
        confirm = typer.confirm(f"Delete book {book_id}?")
        if not confirm:
            console.print("[yellow]Cancelled[/]")
            return

    cfg = get_config(config)
    pipeline = IngestPipeline(cfg)

    try:
        pipeline.delete_book(book_id)
        console.print(f"[green]Deleted book {book_id}[/]")
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1) from None


@app.command()
def stats(
    config: str = typer.Option(None, "--config", "-c", help="Path to config file"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Show database statistics."""
    setup_logging("WARNING")

    cfg = get_config(config)
    pipeline = IngestPipeline(cfg)

    stats_data = pipeline.get_stats()

    if json_output:
        console.print(json.dumps(stats_data, indent=2))
    else:
        console.print("\n[bold]Database Statistics[/]")
        console.print(f"  Collection: {stats_data['collection']['name']}")
        console.print(f"  Total Points: {stats_data['collection']['points_count']}")
        console.print(f"  Total Books: {stats_data['total_books']}")
        console.print(f"  Status: {stats_data['collection']['status']}")


@app.command()
def init_config(
    output: str = typer.Option("config.yaml", "--output", "-o", help="Output file path"),
) -> None:
    """Generate a sample configuration file."""
    cfg = IngestConfig()
    cfg.to_yaml(output)
    console.print(f"[green]Created config file:[/] {output}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
