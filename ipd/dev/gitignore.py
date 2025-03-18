import fnmatch
import os
from typing import List, Optional

class GitIgnore:
    """
    A class to handle .gitignore pattern matching.
    Reads a .gitignore file on construction and provides methods to check if paths should be ignored.
    """

    def __init__(self, gitignore_file: Optional[str] = None):
        """
        Initialize the GitIgnore object by loading patterns from a .gitignore file.

        Args:
            gitignore_file: Path to the .gitignore file. If None, no patterns are loaded.
        """
        self.patterns: List[str] = []

        if gitignore_file and os.path.isfile(gitignore_file):
            try:
                with open(gitignore_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        # Skip empty lines and comments
                        if line and not line.startswith('#'):
                            self.patterns.append(line)
            except Exception as e:
                print(f"Warning: Could not read .gitignore file: {e}")

    def is_ignored(self, file_path: str) -> bool:
        """
        Check if a file should be ignored based on .gitignore patterns.

        Args:
            file_path: Path to the file to check, relative to the repository root

        Returns:
            True if the file should be ignored, False otherwise
        """
        if not self.patterns:
            return False

        # Normalize path separator
        file_path = file_path.replace(os.sep, '/')
        is_directory = os.path.isdir(file_path)

        # Track if the file is explicitly included
        explicitly_included = False

        # Process patterns in order
        for pattern in self.patterns:
            negated = pattern.startswith('!')
            if negated:
                pattern = pattern[1:].strip()

            # Prepare the pattern for matching
            if pattern.startswith('/'):
                # Anchored pattern to repo root
                pattern = pattern[1:]
            elif not pattern.startswith('**'):
                # If not anchored and not a global pattern,
                # it can match at any level
                pattern = f"**/{pattern}"

            # For directory-only patterns
            if pattern.endswith('/'):
                if not is_directory:
                    continue
                pattern = pattern[:-1]

            # Convert gitignore pattern to a regex pattern
            # pattern = pattern.replace('.', '\\.')
            # pattern = pattern.replace('**', '__DOUBLE_STAR__')
            # pattern = pattern.replace('*', '[^/]*')
            # pattern = pattern.replace('__DOUBLE_STAR__', '.*')
            # pattern = pattern.replace('?', '.')
            # pattern = f"^{pattern}$|^{pattern}/"
            # Check if the file matches the pattern
            # if re.search(pattern, file_path) or re.search(pattern, f"{file_path}/"):
            if file_path.startswith('.nox'):
                print(f"DEBUG: {file_path} matches {pattern}")
                if 'nox' in pattern:
                    break
            if fnmatch.fnmatch(file_path, pattern):
                if negated:
                    explicitly_included = True
                else:
                    if not explicitly_included:
                        return True

        return False
