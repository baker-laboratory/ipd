import os
import shutil
import mimetypes
from typing import final

import ipd

text_extensions = {
    '.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm', '.css', '.js', '.ts', '.py', '.java', '.c',
    '.cpp', '.h', '.hpp', '.cs', '.go', '.rs', '.rb', '.php', '.pl', '.swift', '.kt', '.kts', '.sh', '.bash',
    '.zsh', '.ini', '.cfg', '.conf', '.yaml', '.yml', '.toml', '.properties', '.gradle', '.sql', '.r', '.ps1',
    '.jsx', '.tsx', '.vue', '.scss', '.less', '.tf', '.tfvars', '.dart', '.lua', '.cmake', '.make', '.tex',
    '.rst', '.asciidoc', '.adoc', '.htaccess', '.bat', '.cmd', '.hs', '.scala', '.sbt', '.clj', '.erl', '.ex',
    '.exs', '.lisp', '.proto'
}

def is_text_file(file_path: str) -> bool:
    """
    Determine if a file is a text file.

    This uses multiple strategies:
    1. Check file extension against common text file extensions
    2. Use Python's mimetypes module
    3. Try to decode a sample of the file as UTF-8
    """
    # Common text file extensions (for source code, configs, etc.)
    # Check extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext in text_extensions:
        return True
    # Check mimetype
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith(('text/', 'application/json', 'application/xml')):
        return True
    # Try reading the file as text
    try:
        with open(file_path, 'rb') as f:
            sample = f.read(4096)  # Read first 4K
            # Check for null bytes which are unlikely in text files
            if b'\x00' in sample:
                return False
            # Try to decode as UTF-8
            sample.decode('utf-8')
            return True
    except UnicodeDecodeError:
        return False
    except Exception:
        return False
    return False

@final
class CloneTextFiles:
    """
    Clone a directory structure but only copy text files (source code, config files, etc.).

    Args:
        self.source_dir: Source directory path to clone
        self.target_dir: Target directory where to create the clone
        respect_gitignore: Whether to respect .self.gitignore patterns
        self.max_size_bytes: Maximum file size to copy (None means no limit)
        self.min_size_bytes: Minimum file size to copy
        self.include_small_binary: Whether to also copy small binary files
        self.small_binary_threshold: Size threshold in bytes for small binary files
        self.verbose: Print information about processed files

    Returns:
        Tuple of (files_copied, files_skipped) counts
    """

    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        isignored: ipd.Callable[[str], bool] = lambda x: False,
        max_size_bytes: int = 0,
        min_size_bytes: int = 0,
        include_small_binary: bool = False,
        small_binary_threshold: int = 10240,  # 10KB default
        verbose: bool = True,
        filelist: list[str] = [],
    ):
        self.source_dir = os.path.abspath(source_dir)
        self.target_dir = os.path.abspath(target_dir)
        self.isignored = isignored
        self.max_size_bytes = max_size_bytes
        self.min_size_bytes = min_size_bytes
        self.include_small_binary = include_small_binary
        self.small_binary_threshold = small_binary_threshold
        self.verbose = verbose
        os.makedirs(self.target_dir, exist_ok=True)
        if filelist:
            for file in filelist:
                self._process_file(os.path.join(self.source_dir, file))
        else:
            self._recurse_dirs(self.source_dir)

    def _recurse_dirs(self, root: str):
        if self.isignored(root): return
        rootbase = os.path.basename(root)
        if rootbase[0] == '.' and rootbase != '.github': return
        if rootbase == 'lib': return
        if root.startswith(self.target_dir): return

        # Create the corresponding subdirectory in the target
        rel_path = os.path.relpath(root, self.source_dir)
        target_subdir = os.path.join(self.target_dir, rel_path) if rel_path != '.' else self.target_dir
        # os.makedirs(target_subdir, exist_ok=True)

        for file in os.listdir(root):
            if file.endswith('.log'): continue
            source_file = os.path.join(root, file)

            if os.path.isdir(source_file):
                self._recurse_dirs(source_file)
                continue

            self._process_file(source_file)

    def _process_file(self, source_file: str):
        rel_file_path = os.path.relpath(source_file, self.source_dir)
        target_file = os.path.join(self.target_dir, rel_file_path)
        if self.isignored(rel_file_path): return
        file_size = os.path.getsize(source_file)
        if file_size > self.max_size_bytes: return
        if file_size < self.min_size_bytes: return
        if not is_text_file(source_file): return
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        shutil.copy2(source_file, target_file)
        # target = target_file.replace(self.target_dir, '').lstrip('/').replace('/', '__')
        # shutil.copy2(source_file, os.path.join(self.target_dir, target))
        print(source_file)
