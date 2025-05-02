import re
import subprocess
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass, field
from evn.format import IdentifyFormattedBlocks, PythonLineTokenizer
import evn


@dataclass
class FormatHistory:
    """Tracks original and formatted code for all files being processed."""

    buffers: dict[str, dict[str, str]] = field(default_factory=dict)

    def add(self, filename: str, original_code: str):
        """Initialize a new file in history with its original code."""
        self.buffers[filename] = {'original': original_code}

    def update(self, filename: str, new_code: str):
        """Update the formatted code for a given file."""
        self.buffers[filename]['formatted'] = new_code

    def get_original(self, filename: str) -> str:
        """Retrieve the original code."""
        return self.buffers[filename]['original']

    def get_formatted(self, filename: str) -> str:
        """Retrieve the formatted code."""
        return self.buffers[filename]['formatted']


@dataclass
class FormatStep(ABC):
    """Abstract base class for formatting steps in the processing pipeline."""

    formatter: Optional['CodeFormatter'] = None

    @abstractmethod
    def apply_formatting(self,
                         code: str,
                         history: Optional[FormatHistory] = None) -> str:
        """Apply a transformation to the given code buffer."""
        pass


@dataclass
class CodeFormatter:
    """Formats Python files using a configurable pipeline of FormatStep actions."""

    actions: list[FormatStep]
    history: FormatHistory = field(default_factory=FormatHistory)
    cpp_mark: IdentifyFormattedBlocks = field(
        default_factory=IdentifyFormattedBlocks)
    cpp_aln: PythonLineTokenizer = field(default_factory=PythonLineTokenizer)

    def __post_init__(self):
        for action in self.actions:
            action.formatter = self

    def run(self,
            files: dict[str, str],
            dryrun=False,
            debug=False) -> FormatHistory:
        """Process in-memory Python file contents and return formatted buffers."""

        # Initialize history with original files
        for filename, code in files.items():
            self.history.add(filename, code)

        # Process each file through the pipeline
        for filename in self.history.buffers:
            code = self.history.get_original(filename)
            if debug:
                print('*************************************')
            if debug:
                print(code, '\n************ orig ****************')
            for action in self.actions:
                if debug:
                    print(action.__class__.__name__, flush=True)
                if dryrun:
                    print(
                        f'Dry run: {action.__class__.__name__} on {filename}')
                else:
                    code = action.apply_formatting(code, self.history)
                if debug:
                    print(
                        code,
                        f'\n************ {action.__class__.__name__} ****************'
                    )
            self.history.update(filename, code)

        return self.history


no_format_pattern = re.compile(
    r'^(\s*)(class|def|for|if|elif|else)\s+?.*: [^#].*')


@dataclass
class MarkHandFormattedBlocksCpp(FormatStep):
    """Adds `# fmt: off` / `# fmt: on` markers around "human-formatted" constructs"""

    def apply_formatting(self,
                         code: str,
                         history: Optional[FormatHistory] = None) -> str:
        return self.formatter.cpp_mark.mark_formtted_blocks(code, 5)


@dataclass
class UnmarkCpp(FormatStep):
    """Adds `# fmt: off` / `# fmt: on` markers around "human-formatted" constructs"""

    def apply_formatting(self,
                         code: str,
                         history: Optional[FormatHistory] = None) -> str:
        return self.formatter.cpp_mark.unmark(code)


@dataclass
class AlignTokensCpp(FormatStep):
    """Aligns on tokens in the code buffer."""

    def apply_formatting(self,
                         code: str,
                         history: Optional[FormatHistory] = None) -> str:
        return self.formatter.cpp_aln.reformat_buffer(code, add_fmt_tag=True)


@dataclass
class RuffFormat(FormatStep):
    """Runs `ruff format` on the in-memory code buffer."""

    def apply_formatting(self,
                         code: str,
                         history: Optional[FormatHistory] = None) -> str:
        try:
            cmd = (['ruff', '--config', evn.projconf, 'format', '-'],
                   )  # `-` tells ruff to read from stdin
            process = subprocess.run(*cmd,
                                     input=code,
                                     text=True,
                                     capture_output=True,
                                     check=True)
            return process.stdout
        except subprocess.CalledProcessError as e:
            print('Error running ruff format:', e.stderr)
            print('Original code:\n', code, flush=True)
            # return code  # Return original if formatting fails
            raise e from None


re_two_blank_lines = re.compile(r'\n\s*\n\s*\n')


@dataclass
class RemoveExtraBlankLines(FormatStep):
    """Replaces multiple consecutive blank lines with a single blank line."""

    def apply_formatting(self, code: str, history=None) -> str:
        return re.sub(re_two_blank_lines, '\n\n', code).strip()


# def format_files(root_path: Path, dryrun: bool = False):
#     """Reads files, runs CodeFormatter, and writes formatted content back."""
#     file_map = {}

#     # Read all .py files
#     if root_path.is_file():
#         file_map[str(root_path)] = root_path.read_text(encoding="utf-8")
#     else:
#         for file in root_path.rglob("*.py"):
#             file_map[str(file)] = file.read_text(encoding="utf-8")

#     # Format files
#     formatter = CodeFormatter([
#         MarkHandFormattedBlocksCpp(),
#         RuffFormat(),
#         UnmarkCpp(),
#     ])
#     formatted_history = formatter.run(file_map, dryrun=dryrun)

#     # Write back results
#     for filename, history in formatted_history.buffers.items():
#         if history["original"] != history["formatted"]:
#             Path(filename).write_text(history["formatted"], encoding="utf-8")
#             print(f"Formatted: {filename}")


def format_buffer(buf, **kw):
    formatter = CodeFormatter([
        # MarkHandFormattedBlocksCpp(),
        AlignTokensCpp(),
        RuffFormat(),
        UnmarkCpp(),
    ])
    formatted_history = formatter.run(dict(buffer=buf), **kw)
    return formatted_history.buffers['buffer']['formatted']
