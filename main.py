#!/usr/bin/env python3
"""
📓 Notebooks CLI - A comprehensive command-line interface for exploring your research notebooks

Features:
- Summary of each notebook with metadata and key insights
- Picture highlights from outputs (with imgcat support)
- Random inspiring quotes based on research themes
- Sticky notes system for annotations and reminders
- Interactive exploration of notebook contents
"""

import json
import random
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import hashlib

# Rich imports for beautiful terminal output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from rich.text import Text
from rich.prompt import Prompt
from rich.progress import track
from rich.align import Align
from rich.rule import Rule
from rich import box
from rich.tree import Tree

console = Console()


class NotebookAnalyzer:
    """Analyzes Jupyter notebooks to extract metadata and summaries"""

    def __init__(self, notebooks_dir: str = "."):
        self.notebooks_dir = Path(notebooks_dir)
        self.notebooks = self._discover_notebooks()

    def _discover_notebooks(self) -> List[Path]:
        """Discover all .ipynb files in the directory and subdirectories"""
        return list(self.notebooks_dir.glob("**/*.ipynb"))

    def analyze_notebook(self, notebook_path: Path) -> Dict:
        """Analyze a single notebook and extract key information"""
        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook_data = json.load(f)
        except Exception as e:
            return {"error": f"Failed to read notebook: {e}"}

        analysis = {
            "title": notebook_path.stem,
            "path": str(notebook_path),
            "size": notebook_path.stat().st_size,
            "modified": datetime.fromtimestamp(notebook_path.stat().st_mtime),
            "cells": len(notebook_data.get("cells", [])),
            "code_cells": 0,
            "markdown_cells": 0,
            "imports": set(),
            "topics": [],
            "outputs": [],
            "images": [],
            "summary": "",
            "key_concepts": [],
        }

        cells = notebook_data.get("cells", [])

        for cell in cells:
            cell_type = cell.get("cell_type", "")
            source = cell.get("source", [])

            if cell_type == "code":
                analysis["code_cells"] += 1
                # Extract imports
                for line in source:
                    if isinstance(line, str) and (
                        line.strip().startswith("import ")
                        or line.strip().startswith("from ")
                    ):
                        analysis["imports"].add(line.strip())

                # Check for outputs and images
                outputs = cell.get("outputs", [])
                for output in outputs:
                    if "data" in output:
                        data = output["data"]
                        if "image/png" in data:
                            analysis["images"].append("PNG image output")
                        if "text/plain" in data:
                            analysis["outputs"].append(str(data["text/plain"])[:100])

            elif cell_type == "markdown":
                analysis["markdown_cells"] += 1
                # Extract content for summary
                content = "".join(source) if isinstance(source, list) else source
                if content and len(content) > 50:
                    analysis["summary"] += content[:300] + "..."
                    break  # Use first substantial markdown cell as summary

        # Extract key concepts based on notebook content
        analysis["key_concepts"] = self._extract_key_concepts(
            analysis["title"], analysis["summary"]
        )
        analysis["topics"] = self._categorize_notebook(
            analysis["title"], analysis["summary"]
        )

        return analysis

    def _extract_key_concepts(self, title: str, summary: str) -> List[str]:
        """Extract key concepts from notebook title and summary"""
        concepts = []
        text = (title + " " + summary).lower()

        # Research domain concepts
        concept_patterns = {
            "machine_learning": ["neural", "network", "learning", "training", "model"],
            "physics": ["spacetime", "quantum", "relativity", "lorentz", "minkowski"],
            "mathematics": [
                "algorithm",
                "optimization",
                "matrix",
                "vector",
                "function",
            ],
            "attention": ["attention", "transformer", "causal", "interaction"],
            "theory": ["theory", "simulation", "constructor", "stochastic"],
            "hierarchy": ["hierarchy", "embedding", "space", "dimension"],
        }

        for concept, keywords in concept_patterns.items():
            if any(keyword in text for keyword in keywords):
                concepts.append(concept)

        return concepts

    def _categorize_notebook(self, title: str, summary: str) -> List[str]:
        """Categorize notebook based on content"""
        categories = []
        text = (title + " " + summary).lower()

        if any(word in text for word in ["attention", "transformer", "causal"]):
            categories.append("🧠 Neural Networks")
        if any(word in text for word in ["spacetime", "physics", "quantum", "lorentz"]):
            categories.append("🔬 Physics")
        if any(word in text for word in ["theory", "constructor", "simulation"]):
            categories.append("📚 Theory")
        if any(word in text for word in ["fractal", "interpolation", "function"]):
            categories.append("📐 Mathematics")
        if any(word in text for word in ["stochastic", "process", "random"]):
            categories.append("🎲 Probability")

        return categories if categories else ["📓 General"]


class StickyNotesManager:
    """Manages sticky notes for notebooks"""

    def __init__(self, notes_file: str = ".notebook_notes.json"):
        self.notes_file = Path(notes_file)
        self.notes = self._load_notes()

    def _load_notes(self) -> Dict:
        """Load sticky notes from file"""
        if self.notes_file.exists():
            try:
                with open(self.notes_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading notes: {e}")
        return {}

    def _save_notes(self):
        """Save sticky notes to file"""
        with open(self.notes_file, "w") as f:
            json.dump(self.notes, f, indent=2, default=str)

    def add_note(self, notebook: str, note: str, category: str = "general"):
        """Add a sticky note to a notebook"""
        if notebook not in self.notes:
            self.notes[notebook] = []

        note_data = {
            "id": hashlib.md5(f"{notebook}{note}{datetime.now()}".encode()).hexdigest()[
                :8
            ],
            "content": note,
            "category": category,
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
        }

        self.notes[notebook].append(note_data)
        self._save_notes()
        return note_data["id"]

    def get_notes(self, notebook: str) -> List[Dict]:
        """Get all notes for a notebook"""
        return self.notes.get(notebook, [])

    def update_note(self, notebook: str, note_id: str, new_content: str):
        """Update an existing note"""
        if notebook in self.notes:
            for note in self.notes[notebook]:
                if note["id"] == note_id:
                    note["content"] = new_content
                    note["modified"] = datetime.now().isoformat()
                    self._save_notes()
                    return True
        return False

    def delete_note(self, notebook: str, note_id: str):
        """Delete a note"""
        if notebook in self.notes:
            self.notes[notebook] = [
                n for n in self.notes[notebook] if n["id"] != note_id
            ]
            self._save_notes()
            return True
        return False


class QuoteGenerator:
    """Generates inspiring quotes based on research themes"""

    def __init__(self):
        self.quotes_by_theme = {
            "machine_learning": [
                "🧠 'The question of whether a computer can think is no more interesting than the question of whether a submarine can swim.' - Edsger Dijkstra",
                "🤖 'Machine intelligence is the last invention that humanity will ever need to make.' - Nick Bostrom",
                "⚡ 'The future is not about replacing humans with machines, but about augmenting human intelligence.' - Fei-Fei Li",
                "🔮 'In the world of artificial intelligence, we are all apprentices in a craft where no one is a master.' - Anonymous",
            ],
            "physics": [
                "🌌 'The most incomprehensible thing about the universe is that it is comprehensible.' - Albert Einstein",
                "⚛️ 'In physics, reality is often stranger than fiction.' - Stephen Hawking",
                "🔬 'Space and time are not conditions in which we exist, but modes in which we think.' - Albert Einstein",
                "🌠 'The universe is not only queerer than we suppose, but queerer than we can suppose.' - J.B.S. Haldane",
            ],
            "mathematics": [
                "📐 'Mathematics is the language with which God has written the universe.' - Galileo Galilei",
                "∞ 'In mathematics you don't understand things. You just get used to them.' - John von Neumann",
                "🔢 'Pure mathematics is, in its way, the poetry of logical ideas.' - Albert Einstein",
                "📊 'Mathematics is the most beautiful and most powerful creation of the human spirit.' - Stefan Banach",
            ],
            "theory": [
                "💡 'A theory is something nobody believes, except for the person who made it. An experiment is something everybody believes, except for the person who made it.' - Albert Einstein",
                "🧩 'Theory without practice is sterile; practice without theory is blind.' - Immanuel Kant",
                "🎯 'It is the theory that decides what can be observed.' - Albert Einstein",
                "🌟 'The best theory is inspired by practice and verified by experiment.' - Anonymous",
            ],
            "general": [
                "🚀 'The important thing is not to stop questioning. Curiosity has its own reason for existing.' - Albert Einstein",
                "📚 'Research is what I'm doing when I don't know what I'm doing.' - Wernher von Braun",
                "💎 'The only way to make sense out of change is to plunge into it, move with it, and join the dance.' - Alan Watts",
                "🎨 'Creativity is intelligence having fun.' - Albert Einstein",
                "🔍 'Discovery consists of seeing what everybody has seen and thinking what nobody has thought.' - Albert Szent-Györgyi",
            ],
        }

    def get_quote(
        self, themes: Optional[List[str]] = None, seed: Optional[int] = None
    ) -> str:
        """Get an inspiring quote based on themes"""
        if seed is not None:
            random.seed(seed)

        if themes:
            # Try to find quotes for specific themes
            for theme in themes:
                if theme in self.quotes_by_theme:
                    return random.choice(self.quotes_by_theme[theme])

        # Fallback to general quotes
        return random.choice(self.quotes_by_theme["general"])


class ImageDisplayer:
    """Handles image display with imgcat support"""

    @staticmethod
    def can_use_imgcat() -> bool:
        """Check if imgcat is available"""
        try:
            subprocess.run(["which", "imgcat"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @staticmethod
    def display_image(image_path: Path):
        """Display image using imgcat if available"""
        if ImageDisplayer.can_use_imgcat():
            try:
                subprocess.run(["imgcat", str(image_path)], check=True)
                return True
            except subprocess.CalledProcessError:
                pass

        console.print(f"🖼️  Image available: {image_path}")
        console.print(f"   📁 Path: {image_path.absolute()}")
        console.print(f"   📏 Size: {image_path.stat().st_size} bytes")
        return False


class NotebooksCLI:
    """Main CLI class with Rich-powered beautiful interface"""

    def __init__(self):
        self.analyzer = NotebookAnalyzer()
        self.notes_manager = StickyNotesManager()
        self.quote_generator = QuoteGenerator()
        self.image_displayer = ImageDisplayer()

    def print_header(self):
        """Print a beautiful header using Rich"""
        title = Text("📓 NOTEBOOKS CLI", style="bold magenta")
        subtitle = Text("Explore Your Research Journey", style="italic cyan")

        header_panel = Panel(
            Align.center(title + "\n" + subtitle),
            box=box.DOUBLE,
            border_style="bright_blue",
            padding=(1, 2),
        )

        console.print()
        console.print(header_panel)
        console.print()

    def print_notebook_summary(self, analysis: Dict):
        """Print a beautiful summary of a notebook using Rich"""
        # Create main info table
        info_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        info_table.add_column("Key", style="cyan bold", width=12)
        info_table.add_column("Value", style="white")

        info_table.add_row("📁 Path:", analysis["path"])
        info_table.add_row(
            "📅 Modified:", analysis["modified"].strftime("%Y-%m-%d %H:%M:%S")
        )
        info_table.add_row("📏 Size:", f"{analysis['size']:,} bytes")
        info_table.add_row(
            "🔢 Cells:",
            f"{analysis['cells']} total ({analysis['code_cells']} code, {analysis['markdown_cells']} markdown)",
        )

        if analysis["topics"]:
            info_table.add_row("🏷️  Topics:", " ".join(analysis["topics"]))

        if analysis["key_concepts"]:
            info_table.add_row("💡 Concepts:", ", ".join(analysis["key_concepts"]))

        # Create title with emoji based on concepts
        title_emoji = "📚"
        if "attention" in analysis.get("key_concepts", []):
            title_emoji = "🧠"
        elif "physics" in analysis.get("key_concepts", []):
            title_emoji = "🔬"
        elif "mathematics" in analysis.get("key_concepts", []):
            title_emoji = "📐"

        # Main panel
        title = f"{title_emoji} {analysis['title']}"
        main_panel = Panel(
            info_table,
            title=title,
            title_align="left",
            border_style="blue",
            box=box.ROUNDED,
        )

        console.print(main_panel)

        # Summary panel
        if analysis["summary"]:
            summary_panel = Panel(
                Text(analysis["summary"], style="italic"),
                title="📝 Summary",
                title_align="left",
                border_style="green",
                box=box.SIMPLE,
            )
            console.print(summary_panel)

        # Imports panel
        if analysis["imports"]:
            imports_text = "\n".join(
                [f"• {imp}" for imp in sorted(list(analysis["imports"])[:5])]
            )
            imports_panel = Panel(
                imports_text,
                title="📦 Key Imports",
                title_align="left",
                border_style="yellow",
                box=box.SIMPLE,
            )
            console.print(imports_panel)

        # Images indicator
        if analysis["images"]:
            console.print(
                f"🖼️  Contains {len(analysis['images'])} images", style="bright_magenta"
            )

        console.print()

    def show_images(self):
        """Show image highlights using Rich"""
        console.print(Panel("🖼️  IMAGE HIGHLIGHTS", style="bold yellow", box=box.DOUBLE))
        console.print()

        image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".svg"]
        images = []

        for ext in image_extensions:
            images.extend(list(Path(".").glob(f"**/*{ext}")))

        if not images:
            console.print("No images found in the current directory.", style="red")
            return

        console.print(f"Found [bold cyan]{len(images)}[/bold cyan] images:\n")

        # Create a table for images
        images_table = Table(show_header=True, header_style="bold magenta")
        images_table.add_column("🖼️  Image", style="cyan")
        images_table.add_column("📏 Size", justify="right", style="green")
        images_table.add_column("📅 Modified", style="yellow")
        images_table.add_column("📁 Path", style="dim")

        for img in sorted(images):
            size_mb = img.stat().st_size / 1024 / 1024
            size_str = (
                f"{size_mb:.2f} MB" if size_mb > 1 else f"{img.stat().st_size:,} bytes"
            )
            modified_str = datetime.fromtimestamp(img.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M"
            )

            images_table.add_row(img.name, size_str, modified_str, str(img.parent))

            # Try to display with imgcat
            if self.image_displayer.can_use_imgcat():
                try:
                    self.image_displayer.display_image(img)
                except Exception as e:
                    console.print(f"   ❌ Failed to display: {e}", style="red")

        console.print(images_table)
        console.print()

    def show_inspiring_quote(self, seed: Optional[int] = None):
        """Show an inspiring quote using Rich"""
        # Gather themes from all notebooks
        themes = set()
        for notebook in self.analyzer.notebooks:
            analysis = self.analyzer.analyze_notebook(notebook)
            themes.update(analysis.get("key_concepts", []))

        quote = self.quote_generator.get_quote(list(themes), seed)

        quote_panel = Panel(
            Align.center(Text(quote, style="italic bold")),
            title="✨ DAILY INSPIRATION",
            title_align="center",
            border_style="bright_yellow",
            box=box.DOUBLE_EDGE,
            padding=(1, 2),
        )

        console.print(quote_panel)
        console.print()

    def manage_sticky_notes(
        self, action: str, notebook: Optional[str] = None, **kwargs
    ):
        """Manage sticky notes with Rich interface"""
        if action == "list":
            if notebook:
                notes = self.notes_manager.get_notes(notebook)
                if notes:
                    notes_table = Table(show_header=True, header_style="bold cyan")
                    notes_table.add_column("🔖 ID", style="yellow", width=10)
                    notes_table.add_column("Category", style="magenta", width=12)
                    notes_table.add_column("Content", style="white")
                    notes_table.add_column("📅 Created", style="green", width=12)

                    for note in notes:
                        notes_table.add_row(
                            note["id"],
                            note["category"].upper(),
                            note["content"][:60]
                            + ("..." if len(note["content"]) > 60 else ""),
                            note["created"][:10],
                        )

                    panel = Panel(
                        notes_table,
                        title=f"📝 Sticky Notes for {notebook}",
                        border_style="cyan",
                    )
                    console.print(panel)
                else:
                    console.print(
                        f"No sticky notes found for [bold]{notebook}[/bold]",
                        style="yellow",
                    )
            else:
                # List all notes
                if not self.notes_manager.notes:
                    console.print("No sticky notes found.", style="yellow")
                    return

                tree = Tree("📝 All Sticky Notes", style="bold cyan")

                for nb, notes in self.notes_manager.notes.items():
                    if notes:
                        nb_branch = tree.add(f"📚 {nb}", style="blue")
                        for note in notes[:3]:  # Show first 3
                            content = note["content"][:40] + (
                                "..." if len(note["content"]) > 40 else ""
                            )
                            nb_branch.add(f"🔖 {content}", style="white")
                        if len(notes) > 3:
                            nb_branch.add(f"... and {len(notes) - 3} more", style="dim")

                console.print(tree)

        elif action == "add":
            if not notebook:
                console.print("Please specify a notebook name", style="red")
                return

            content = kwargs.get("content") or Prompt.ask("Enter note content")
            category = kwargs.get("category", "general")

            note_id = self.notes_manager.add_note(notebook, content, category)

            success_panel = Panel(
                f"✅ Added note [{note_id}] to {notebook}",
                style="bold green",
                box=box.SIMPLE,
            )
            console.print(success_panel)

    def run_interactive(self):
        """Run interactive mode with Rich interface"""
        while True:
            console.print(Rule("🎯 Main Menu", style="bright_blue"))

            choices = [
                "📚 List all notebooks",
                "🔍 Analyze specific notebook",
                "🖼️  Show image highlights",
                "✨ Get inspiring quote",
                "📝 Manage sticky notes",
                "🚪 Exit",
            ]

            for i, choice in enumerate(choices, 1):
                console.print(f"{i}. {choice}")

            choice = Prompt.ask(
                "\nEnter choice", choices=["1", "2", "3", "4", "5", "6"], default="1"
            )
            console.print()

            if choice == "1":
                self.list_notebooks()
            elif choice == "2":
                self.analyze_specific_notebook()
            elif choice == "3":
                self.show_images()
            elif choice == "4":
                seed = random.randint(1, 1000)
                self.show_inspiring_quote(seed)
            elif choice == "5":
                self.interactive_notes()
            elif choice == "6":
                console.print("👋 Happy researching!", style="bold green")
                break

    def list_notebooks(self):
        """List all notebooks with summaries using Rich"""
        if not self.analyzer.notebooks:
            console.print(
                "No Jupyter notebooks found in the current directory.", style="red"
            )
            return

        console.print(
            Panel("📚 NOTEBOOK LIBRARY", style="bold magenta", box=box.DOUBLE)
        )
        console.print()

        for notebook in track(
            sorted(self.analyzer.notebooks), description="Analyzing notebooks..."
        ):
            analysis = self.analyzer.analyze_notebook(notebook)
            if "error" not in analysis:
                self.print_notebook_summary(analysis)

    def analyze_specific_notebook(self):
        """Analyze a specific notebook with Rich interface"""
        if not self.analyzer.notebooks:
            console.print("No notebooks found!", style="red")
            return

        # Create choices table
        choices_table = Table(show_header=True, header_style="bold cyan")
        choices_table.add_column("#", style="yellow", width=4)
        choices_table.add_column("📚 Notebook", style="white")

        for i, nb in enumerate(self.analyzer.notebooks, 1):
            choices_table.add_row(str(i), nb.name)

        console.print(
            Panel(choices_table, title="Available Notebooks", border_style="blue")
        )

        choice = Prompt.ask(
            "Choose notebook",
            choices=[str(i) for i in range(1, len(self.analyzer.notebooks) + 1)],
        )

        notebook = self.analyzer.notebooks[int(choice) - 1]
        analysis = self.analyzer.analyze_notebook(notebook)
        console.print()
        self.print_notebook_summary(analysis)

    def interactive_notes(self):
        """Interactive sticky notes management with Rich"""
        console.print(
            Panel("📝 Sticky Notes Manager", style="bold cyan", box=box.DOUBLE)
        )

        choices = ["List all notes", "Add new note", "List notes for specific notebook"]

        for i, choice in enumerate(choices, 1):
            console.print(f"{i}. {choice}")

        choice = Prompt.ask("Choose option", choices=["1", "2", "3"])
        console.print()

        if choice == "1":
            self.manage_sticky_notes("list")
        elif choice == "2":
            if not self.analyzer.notebooks:
                console.print("No notebooks found!", style="red")
                return

            # Show notebook choices
            choices_table = Table(show_header=True, header_style="bold cyan")
            choices_table.add_column("#", style="yellow", width=4)
            choices_table.add_column("📚 Notebook", style="white")

            for i, nb in enumerate(self.analyzer.notebooks, 1):
                choices_table.add_row(str(i), nb.stem)

            console.print(
                Panel(choices_table, title="Choose Notebook", border_style="blue")
            )

            nb_choice = Prompt.ask(
                "Choose notebook",
                choices=[str(i) for i in range(1, len(self.analyzer.notebooks) + 1)],
            )

            notebook = self.analyzer.notebooks[int(nb_choice) - 1].stem
            content = Prompt.ask("Enter note content")
            category = Prompt.ask("Enter category", default="general")

            self.manage_sticky_notes(
                "add", notebook, content=content, category=category
            )
        elif choice == "3":
            notebook = Prompt.ask("Enter notebook name")
            self.manage_sticky_notes("list", notebook)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="📓 Notebooks CLI - Explore your research journey",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Interactive mode
  %(prog)s --list                    # List all notebooks
  %(prog)s --images                  # Show image highlights  
  %(prog)s --quote --seed 42         # Get inspiring quote with seed
  %(prog)s --notes list              # List all sticky notes
  %(prog)s --notes add notebook.ipynb "Important finding"
        """,
    )

    parser.add_argument(
        "--list", action="store_true", help="List all notebooks with summaries"
    )
    parser.add_argument("--images", action="store_true", help="Show image highlights")
    parser.add_argument("--quote", action="store_true", help="Show inspiring quote")
    parser.add_argument("--seed", type=int, help="Random seed for quote generation")
    parser.add_argument(
        "--notes", nargs="+", help="Sticky notes: list | add <notebook> <content>"
    )
    parser.add_argument("--notebook", help="Analyze specific notebook")

    args = parser.parse_args()

    cli = NotebooksCLI()
    cli.print_header()

    # Show daily quote by default
    if not any([args.list, args.images, args.quote, args.notes, args.notebook]):
        cli.show_inspiring_quote()

    if args.list:
        cli.list_notebooks()
    elif args.images:
        cli.show_images()
    elif args.quote:
        cli.show_inspiring_quote(args.seed)
    elif args.notes:
        if args.notes[0] == "list":
            notebook = args.notes[1] if len(args.notes) > 1 else None
            cli.manage_sticky_notes("list", notebook)
        elif args.notes[0] == "add" and len(args.notes) >= 3:
            notebook = args.notes[1]
            content = " ".join(args.notes[2:])
            cli.manage_sticky_notes("add", notebook, content=content)
        else:
            console.print(
                "Usage: --notes list [notebook] | --notes add <notebook> <content>",
                style="red",
            )
    elif args.notebook:
        # Analyze specific notebook
        notebook_path = Path(args.notebook)
        if notebook_path.exists():
            analysis = cli.analyzer.analyze_notebook(notebook_path)
            cli.print_notebook_summary(analysis)
        else:
            console.print(f"Notebook not found: {args.notebook}", style="red")
    else:
        # Interactive mode
        cli.run_interactive()


if __name__ == "__main__":
    main()
