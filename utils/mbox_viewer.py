#!/usr/bin/env python3
"""Simple .mbox file viewer"""

import mailbox
import sys
import email.header
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()


def decode_header(value):
    """Decode an email header into a readable string."""
    if not value:
        return "(not available)"
    parts = email.header.decode_header(value)
    result = []
    for chunk, charset in parts:
        if isinstance(chunk, bytes):
            result.append(chunk.decode(charset or "utf-8", errors="replace"))
        else:
            result.append(chunk)
    return " ".join(result)


def get_body(msg):
    """Extract the email body text."""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))
            if content_type == "text/plain" and "attachment" not in disposition:
                charset = part.get_content_charset() or "utf-8"
                body = part.get_payload(decode=True).decode(charset, errors="replace")
                break
    else:
        charset = msg.get_content_charset() or "utf-8"
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode(charset, errors="replace")
    return body.strip() or "(empty body)"


def show_email(msg, index, total):
    """Display a formatted email with rich."""
    console.clear()

    sender    = decode_header(msg.get("From", ""))
    recipient = decode_header(msg.get("To", ""))
    subject   = decode_header(msg.get("Subject", ""))
    date      = decode_header(msg.get("Date", ""))
    body      = get_body(msg)

    header_text = Text()
    header_text.append("From:     ", style="bold cyan")
    header_text.append(sender + "\n")
    header_text.append("To:       ", style="bold cyan")
    header_text.append(recipient + "\n")
    header_text.append("Date:     ", style="bold cyan")
    header_text.append(date + "\n")
    header_text.append("Subject:  ", style="bold cyan")
    header_text.append(subject)

    console.print(Panel(
        header_text,
        title=f"[bold yellow]Email {index + 1} / {total}[/bold yellow]",
        box=box.ROUNDED,
        expand=True
    ))

    console.print(Panel(
        body,
        title="[bold green]Body[/bold green]",
        box=box.ROUNDED,
        expand=True
    ))

    console.print(
        "\n[bold]Navigation:[/bold] "
        "[cyan]n[/cyan] next  "
        "[cyan]p[/cyan] previous  "
        "[cyan]q[/cyan] quit"
    )


def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage: python mbox_viewer.py <file.mbox>[/red]")
        sys.exit(1)

    path = sys.argv[1]

    try:
        mbox = mailbox.mbox(path)
        messages = list(mbox)
    except Exception as e:
        console.print(f"[red]Error opening file: {e}[/red]")
        sys.exit(1)

    if not messages:
        console.print("[yellow]The mbox file is empty.[/yellow]")
        sys.exit(0)

    total = len(messages)
    index = 0

    while True:
        show_email(messages[index], index, total)
        cmd = input("\n> ").strip().lower()

        if cmd == "q":
            console.print("[bold]Exiting.[/bold]")
            break
        elif cmd == "n":
            if index < total - 1:
                index += 1
            else:
                console.print("[yellow]You are already at the last email.[/yellow]")
                input("Press Enter to continue...")
        elif cmd == "p":
            if index > 0:
                index -= 1
            else:
                console.print("[yellow]You are already at the first email.[/yellow]")
                input("Press Enter to continue...")
        else:
            console.print("[red]Unrecognized command. Use n, p, or q.[/red]")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()