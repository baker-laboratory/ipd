import re
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import ipd

@ipd.struct
class FormatHistory:
    """Tracks original and formatted code for all files being processed."""
    buffers: dict[str, dict[str, str]] = ipd.field(dict)

    def add(self, filename: str, original_code: str):
        """Initialize a new file in history with its original code."""
        self.buffers[filename] = {"original": original_code}

    def update(self, filename: str, new_code: str):
        """Update the formatted code for a given file."""
        self.buffers[filename]["formatted"] = new_code

    def get_original(self, filename: str) -> str:
        """Retrieve the original code."""
        return self.buffers[filename]["original"]

    def get_formatted(self, filename: str) -> str:
        """Retrieve the formatted code."""
        return self.buffers[filename]["formatted"]

@ipd.struct
class FormatStep(ABC):
    """Abstract base class for formatting steps in the processing pipeline."""

    @abstractmethod
    def apply_formatting(self, code: str, history: Optional[FormatHistory] = None) -> str:
        """Apply a transformation to the given code buffer."""
        pass

@ipd.struct
class CodeFormatter:
    """Formats Python files using a configurable pipeline of FormatStep actions."""

    actions: list[FormatStep] = ipd.field(list)
    history: FormatHistory = ipd.field(FormatHistory)

    def run(self, files: dict[str, str], dryrun=False) -> FormatHistory:
        """Process in-memory Python file contents and return formatted buffers."""

        # Initialize history with original files
        for filename, code in files.items():
            self.history.add(filename, code)

        # Process each file through the pipeline
        for filename in self.history.buffers:
            code = self.history.get_original(filename)
            for action in self.actions:
                if dryrun: print(f"Dry run: {action.__class__.__name__} on {filename}")
                else: code = action.apply_formatting(code, self.history)
            self.history.update(filename, code)

        return self.history

no_format_pattern = re.compile(r"^(\s*)(class|def|for|if|elif|else)\s+?.*: [^#].*")

@ipd.struct
class AddFmtMarkers(FormatStep):
    """Adds `#             fmt: off` / `#             fmt: on` markers around single-line if/elif/else statements."""
    FMT_OFF = "#             fmt: off"
    FMT_ON = "#             fmt: on"

    @ipd.timed
    def apply_formatting(self, code: str, history: Optional[FormatHistory] = None) -> str:
        lines = code.split("\n")
        new_lines = []
        for line in lines:
            match = no_format_pattern.match(line)
            if match:
                indent = match.group(1)
                new_lines.append(f"{indent}{self.FMT_OFF}")
                new_lines.append(line)
                new_lines.append(f"{indent}{self.FMT_ON}")
            else:
                new_lines.append(line)

        return "\n".join(new_lines)

@ipd.struct
class RemoveFmtMarkers(FormatStep):
    """Removes `#             fmt: off` / `#             fmt: on` markers."""

    FMT_OFF = "#             fmt: off"
    FMT_ON = "#             fmt: on"

    @ipd.timed
    def apply_formatting(self, code: str, history: Optional[FormatHistory] = None) -> str:
        lines = code.split("\n")
        return "\n".join(line for line in lines if line.strip() not in {self.FMT_OFF, self.FMT_ON})

@ipd.struct
class RuffFormat(FormatStep):
    """Runs `ruff format` on the in-memory code buffer."""

    @ipd.timed
    def apply_formatting(self, code: str, history: Optional[FormatHistory] = None) -> str:
        try:
            process = subprocess.run(
                ["ruff", "format", "-"],  # `-` tells ruff to read from stdin
                input=code,
                text=True,
                capture_output=True,
                check=True)
            return process.stdout
        except subprocess.CalledProcessError as e:
            print("Error running ruff format:", e.stderr)
            return code  # Return original if formatting fails

re_two_blank_lines = re.compile(r"\n\s*\n\s*\n")

@ipd.struct
class RemoveExtraBlankLines(FormatStep):
    """Replaces multiple consecutive blank lines with a single blank line."""

    def apply_formatting(self, code: str, history=None) -> str:
        return re.sub(re_two_blank_lines, "\n\n", code).strip()

def format_files(root_path: Path, dryrun: bool = False):
    """Reads files, runs CodeFormatter, and writes formatted content back."""
    file_map = {}

    # Read all .py files
    if root_path.is_file():
        file_map[str(root_path)] = root_path.read_text(encoding="utf-8")
    else:
        for file in root_path.rglob("*.py"):
            file_map[str(file)] = file.read_text(encoding="utf-8")

    # Format files
    formatter = CodeFormatter(actions=[AddFmtMarkers(), RuffFormat(), RemoveFmtMarkers()])
    formatted_history = formatter.run(file_map, dryrun=dryrun)

    # Write back results
    for filename, history in formatted_history.buffers.items():
        if history["original"] != history["formatted"]:
            Path(filename).write_text(history["formatted"], encoding="utf-8")
            print(f"Formatted: {filename}")

def format_buffer(buf):
    formatter = CodeFormatter(actions=[AddFmtMarkers(), RuffFormat(), RemoveFmtMarkers()])
    formatted_history = formatter.run(dict(buffer=buf))
    return formatted_history.buffers["buffer"]["formatted"]
