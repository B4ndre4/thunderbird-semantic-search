#!/usr/bin/env python3
"""
LanceDB Browser - Interactively browse LanceDB database records.

Usage:
    python lancedb_browser.py <database_path>
    python lancedb_browser.py <database_path> --table <table_name>
"""

import sys
import os
import click
import lancedb
import pyarrow as pa
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.columns import Columns
from rich.rule import Rule
from rich import box

console = Console()


def format_value(value) -> str:
    """Format a value for display, truncating if too long."""
    if value is None:
        return "[dim]NULL[/dim]"
    s = str(value)
    if len(s) > 200:
        return s[:197] + "[dim]...[/dim]"
    return s


def display_record(record: dict, index: int, total: int, table_name: str):
    """Display a single record in a formatted panel."""
    console.clear()

    title_text = Text()
    title_text.append("⬡ LanceDB Browser", style="bold cyan")
    title_text.append(f"  │  table: ", style="dim")
    title_text.append(table_name, style="bold yellow")
    console.print(Panel(title_text, box=box.HORIZONTALS, style="cyan"))

    progress = f"[bold white]Record[/bold white] [cyan]{index + 1}[/cyan] [dim]of[/dim] [cyan]{total}[/cyan]"
    console.print(f"\n  {progress}\n")

    tbl = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        expand=True,
    )
    tbl.add_column("Field", style="bold yellow", no_wrap=True, min_width=20)
    tbl.add_column("Type", style="dim cyan", no_wrap=True, min_width=15)
    tbl.add_column("Value", style="white")

    for key, value in record.items():
        value_type = type(value).__name__
        if isinstance(value, (list, tuple)):
            value_type = f"list[{len(value)}]"
        elif hasattr(value, '__class__') and 'ndarray' in type(value).__name__:
            value_type = f"array{list(value.shape)}"
        tbl.add_row(key, value_type, format_value(value))

    console.print(tbl)

    console.print()
    console.print(Rule(style="dim"))
    cmds = Text()
    cmds.append("  [n] ", style="bold green")
    cmds.append("Next  ", style="dim")
    cmds.append("[p] ", style="bold green")
    cmds.append("Previous  ", style="dim")
    cmds.append("[g] ", style="bold green")
    cmds.append("Go to...  ", style="dim")
    cmds.append("[t] ", style="bold green")
    cmds.append("Change table  ", style="dim")
    cmds.append("[q] ", style="bold green")
    cmds.append("Quit", style="dim")
    console.print(cmds)
    console.print()


def select_table(db) -> str | None:
    """Show list of tables and ask user to choose one."""
    table_names = db.table_names()
    if not table_names:
        console.print("[red]No tables found in database.[/red]")
        return None

    console.clear()
    console.print(Panel("[bold cyan]⬡ LanceDB Browser[/bold cyan]  —  Table Selection", box=box.HORIZONTALS))
    console.print()

    tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    tbl.add_column("#", style="cyan", no_wrap=True, min_width=4)
    tbl.add_column("Table Name", style="bold white")

    for i, name in enumerate(table_names):
        tbl.add_row(str(i + 1), name)

    console.print(tbl)
    console.print()

    if len(table_names) == 1:
        console.print(f"[dim]Only available table: [bold]{table_names[0]}[/bold] — automatically selected.[/dim]\n")
        return table_names[0]

    while True:
        choice = Prompt.ask(
            f"[bold green]Choose table[/bold green] [dim](1-{len(table_names)})[/dim]",
            default="1",
        )
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(table_names):
                return table_names[idx]
        except ValueError:
            pass
        if choice in table_names:
            return choice
        console.print("[red]Invalid choice, try again.[/red]")


def browse_table(db, table_name: str):
    """Main navigation loop for a table."""
    try:
        tbl = db.open_table(table_name)
    except Exception as e:
        console.print(f"[red]Error opening table '{table_name}': {e}[/red]")
        return "quit"

    try:
        df = tbl.to_arrow()
        records = df.to_pydict()
        keys = list(records.keys())
        total = len(records[keys[0]]) if keys else 0
    except Exception as e:
        console.print(f"[red]Error reading data: {e}[/red]")
        return "quit"

    if total == 0:
        console.print(f"[yellow]Table '{table_name}' is empty.[/yellow]")
        Prompt.ask("[dim]Press Enter to continue[/dim]")
        return "change_table"

    def get_record(idx: int) -> dict:
        return {k: records[k][idx] for k in keys}

    index = 0

    while True:
        record = get_record(index)
        display_record(record, index, total, table_name)

        cmd = Prompt.ask("[bold green]Command[/bold green]", default="n").strip().lower()

        if cmd in ("q", "quit", "exit"):
            return "quit"

        elif cmd in ("n", "next", ""):
            if index < total - 1:
                index += 1
            else:
                console.print("[yellow]  → Already at last record.[/yellow]")
                import time; time.sleep(0.8)

        elif cmd in ("p", "previous"):
            if index > 0:
                index -= 1
            else:
                console.print("[yellow]  → Already at first record.[/yellow]")
                import time; time.sleep(0.8)

        elif cmd in ("g", "goto", "go"):
            dest = Prompt.ask(f"[bold green]Go to record[/bold green] [dim](1-{total})[/dim]")
            try:
                dest_idx = int(dest) - 1
                if 0 <= dest_idx < total:
                    index = dest_idx
                else:
                    console.print(f"[red]  Number out of range (1-{total}).[/red]")
                    import time; time.sleep(0.8)
            except ValueError:
                console.print("[red]  Invalid value.[/red]")
                import time; time.sleep(0.8)

        elif cmd in ("t", "table", "change_table"):
            return "change_table"

        else:
            console.print(f"[red]  Unknown command: '{cmd}'[/red]")
            import time; time.sleep(0.6)


@click.command()
@click.argument("db_path", type=click.Path(exists=True))
@click.option("--table", "-t", default=None, help="Table name to open directly.")
def main(db_path: str, table: str | None):
    """
    Interactive browser for LanceDB database.

    DB_PATH is the path to the LanceDB database directory.
    """
    console.print(f"\n[dim]Connecting to[/dim] [bold cyan]{db_path}[/bold cyan]...\n")

    try:
        db = lancedb.connect(db_path)
    except Exception as e:
        console.print(f"[bold red]Connection error:[/bold red] {e}")
        sys.exit(1)

    current_table = table

    while True:
        if current_table is None:
            current_table = select_table(db)
            if current_table is None:
                break

        result = browse_table(db, current_table)

        if result == "quit":
            break
        elif result == "change_table":
            current_table = None

    console.print("\n[bold cyan]Goodbye![/bold cyan]\n")


if __name__ == "__main__":
    main()