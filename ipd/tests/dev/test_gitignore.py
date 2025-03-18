import os
import tempfile
import unittest
from unittest.mock import patch
import shutil

import ipd
from ipd.dev.gitignore import GitIgnore

config_test = ipd.Bunch(
    re_only=[
        #
    ],
    re_exclude=[
        #
    ],
)

def main():
    ipd.tests.maintest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
    )

class TestGitIgnore(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for our test files
        self.temp_dir = tempfile.mkdtemp()

        # Create some test directories and files
        os.makedirs(os.path.join(self.temp_dir, "src/main/java"))
        os.makedirs(os.path.join(self.temp_dir, ".git"))
        os.makedirs(os.path.join(self.temp_dir, "node_modules"))
        os.makedirs(os.path.join(self.temp_dir, "build"))
        os.makedirs(os.path.join(self.temp_dir, ".nox/py38"))
        os.makedirs(os.path.join(self.temp_dir, "docs"))
        os.makedirs(os.path.join(self.temp_dir, "tests/unit"))

        # Create some test files
        self.create_file("README.md")
        self.create_file("src/main/java/Main.java")
        self.create_file("src/main/java/Test.java")
        self.create_file("build/output.log")
        self.create_file("docs/index.html")
        self.create_file("docs/spec.pdf")
        self.create_file("tests/unit/test_core.py")
        self.create_file(".nox/py38/lib/site-packages/some_module.py")
        self.create_file("node_modules/package/index.js")
        self.create_file(".DS_Store")
        self.create_file("config.yaml")
        self.create_file("data.csv")
        self.create_file("secrets.env")

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def create_file(self, relative_path):
        """Helper method to create test files"""
        file_path = os.path.join(self.temp_dir, relative_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(f"Test content for {relative_path}")

    def create_gitignore_file(self, patterns):
        """Create a .gitignore file with the given patterns"""
        gitignore_path = os.path.join(self.temp_dir, ".gitignore")
        with open(gitignore_path, 'w') as f:
            f.write('\n'.join(patterns))
        return gitignore_path

    def get_relative_path(self, path):
        """Get path relative to the temp directory"""
        return os.path.relpath(path, self.temp_dir)

    def test_initialization_with_nonexistent_file(self):
        """Test initializing with a non-existent file"""
        gitignore = GitIgnore("nonexistent_file.txt")
        self.assertEqual(len(gitignore.patterns), 0)

    def test_initialization_with_valid_file(self):
        """Test initializing with a valid .gitignore file"""
        patterns = ["*.log", "build/", "node_modules/"]
        gitignore_path = self.create_gitignore_file(patterns)
        gitignore = GitIgnore(gitignore_path)
        self.assertEqual(len(gitignore.patterns), 3)
        self.assertEqual(gitignore.patterns, patterns)

    def test_initialization_with_comments_and_empty_lines(self):
        """Test that comments and empty lines are ignored"""
        patterns = ["*.log", "", "# This is a comment", "build/", "  ", "node_modules/"]
        gitignore_path = self.create_gitignore_file(patterns)
        gitignore = GitIgnore(gitignore_path)
        self.assertEqual(len(gitignore.patterns), 3)
        self.assertEqual(gitignore.patterns, ["*.log", "build/", "node_modules/"])

    def test_simple_patterns(self):
        """Test simple wildcard patterns like *.log"""
        patterns = ["*.log", "*.pdf", "*.env"]
        gitignore_path = self.create_gitignore_file(patterns)
        gitignore = GitIgnore(gitignore_path)

        # Should be ignored
        self.assertTrue(gitignore.is_ignored("build/output.log"))
        self.assertTrue(gitignore.is_ignored("docs/spec.pdf"))
        self.assertTrue(gitignore.is_ignored("secrets.env"))

        # Should not be ignored
        self.assertFalse(gitignore.is_ignored("README.md"))
        self.assertFalse(gitignore.is_ignored("data.csv"))
        self.assertFalse(gitignore.is_ignored("config.yaml"))

    def test_directory_patterns(self):
        """Test patterns that match directories"""
        patterns = ["build/", "node_modules/", ".nox/"]
        gitignore_path = self.create_gitignore_file(patterns)
        gitignore = GitIgnore(gitignore_path)

        # Should be ignored (files within ignored directories)
        self.assertTrue(gitignore.is_ignored("build/output.log"))
        self.assertTrue(gitignore.is_ignored("node_modules/package/index.js"))
        self.assertTrue(gitignore.is_ignored(".nox/py38/lib/site-packages/some_module.py"))

        # Should not be ignored
        self.assertFalse(gitignore.is_ignored("README.md"))
        self.assertFalse(gitignore.is_ignored("src/main/java/Main.java"))

    def test_anchored_patterns(self):
        """Test patterns that are anchored to the root"""
        patterns = ["/README.md", "/config.yaml"]
        gitignore_path = self.create_gitignore_file(patterns)
        gitignore = GitIgnore(gitignore_path)

        # Should be ignored (exact matches at root)
        self.assertTrue(gitignore.is_ignored("README.md"))
        self.assertTrue(gitignore.is_ignored("config.yaml"))

        # Should not be ignored (not at root)
        self.assertFalse(gitignore.is_ignored("docs/README.md"))
        self.assertFalse(gitignore.is_ignored("src/config.yaml"))

    def test_negated_patterns(self):
        """Test negated patterns (patterns that start with !)"""
        patterns = ["*.log", "!build/special.log"]
        gitignore_path = self.create_gitignore_file(patterns)
        gitignore = GitIgnore(gitignore_path)

        # Create a special file that should not be ignored
        self.create_file("build/special.log")

        # Should be ignored
        self.assertTrue(gitignore.is_ignored("build/output.log"))
        self.assertTrue(gitignore.is_ignored("logs/server.log"))

        # Should not be ignored (negated pattern)
        self.assertFalse(gitignore.is_ignored("build/special.log"))

    def test_double_asterisk_patterns(self):
        """Test patterns with double asterisks (match across directories)"""
        patterns = ["**/java/**", "**/unit/**"]
        gitignore_path = self.create_gitignore_file(patterns)
        gitignore = GitIgnore(gitignore_path)

        # Should be ignored
        self.assertTrue(gitignore.is_ignored("src/main/java/Main.java"))
        self.assertTrue(gitignore.is_ignored("tests/unit/test_core.py"))

        # Should not be ignored
        self.assertFalse(gitignore.is_ignored("README.md"))
        self.assertFalse(gitignore.is_ignored("docs/index.html"))

    def test_question_mark_patterns(self):
        """Test patterns with question marks (match single character)"""
        patterns = ["test?.py", "Main.???a"]
        gitignore_path = self.create_gitignore_file(patterns)
        gitignore = GitIgnore(gitignore_path)

        # Create some test files for this specific test
        self.create_file("test1.py")
        self.create_file("test2.py")
        self.create_file("tests.py")
        self.create_file("Main.java")

        # Should be ignored
        self.assertTrue(gitignore.is_ignored("test1.py"))
        self.assertTrue(gitignore.is_ignored("test2.py"))
        self.assertTrue(gitignore.is_ignored("src/main/java/Main.java"))

        # Should not be ignored
        self.assertFalse(gitignore.is_ignored("tests.py"))  # More than one character
        self.assertFalse(gitignore.is_ignored("Main.cpp"))  # Different extension

    def test_bracket_patterns(self):
        """Test patterns with character classes [abc]"""
        patterns = ["*.[jp][pn]g", "data.[ct][sx][vt]"]
        gitignore_path = self.create_gitignore_file(patterns)
        gitignore = GitIgnore(gitignore_path)

        # Create some test files for this specific test
        self.create_file("image.jpg")
        self.create_file("icon.png")
        self.create_file("data.csv")
        self.create_file("data.tsv")
        self.create_file("data.txt")

        # Should be ignored
        self.assertTrue(gitignore.is_ignored("image.jpg"))
        self.assertTrue(gitignore.is_ignored("icon.png"))
        self.assertTrue(gitignore.is_ignored("data.csv"))
        self.assertTrue(gitignore.is_ignored("data.tsv"))

        # Should not be ignored
        self.assertFalse(gitignore.is_ignored("image.gif"))
        self.assertFalse(gitignore.is_ignored("data.txt"))

    def test_specific_cases(self):
        """Test specific cases mentioned in the requirements"""
        patterns = [
            "*.log",  # All log files
            "build/",  # The build directory
            "node_modules/",  # The node_modules directory
            ".nox/",  # The .nox directory
            ".DS_Store",  # macOS specific files
            "!important.log",  # Exception for an important log file
            "/config.yaml",  # Only config.yaml in the root
            "**/.git/**",  # All .git directories
            "*.py[cod]",  # Python bytecode files
            "src/**/*.java"  # All Java files in src
        ]
        gitignore_path = self.create_gitignore_file(patterns)
        gitignore = GitIgnore(gitignore_path)

        # Create some additional test files
        self.create_file("important.log")
        self.create_file("src/test/Test.java")
        self.create_file("test.pyc")

        # Should be ignored
        self.assertTrue(gitignore.is_ignored("build/output.log"))
        self.assertTrue(gitignore.is_ignored("node_modules/package/index.js"))
        self.assertTrue(gitignore.is_ignored(".nox/py38/lib/site-packages/some_module.py"))
        self.assertTrue(gitignore.is_ignored(".DS_Store"))
        self.assertTrue(gitignore.is_ignored("config.yaml"))
        self.assertTrue(gitignore.is_ignored(".git/config"))
        self.assertTrue(gitignore.is_ignored("test.pyc"))
        self.assertTrue(gitignore.is_ignored("src/main/java/Main.java"))

        # Should not be ignored
        self.assertFalse(gitignore.is_ignored("important.log"))
        self.assertFalse(gitignore.is_ignored("subdir/config.yaml"))
        self.assertFalse(gitignore.is_ignored("README.md"))

    def test_empty_patterns(self):
        """Test with no patterns"""
        gitignore = GitIgnore()
        self.assertFalse(gitignore.is_ignored("anything.txt"))

    @patch('builtins.open', side_effect=IOError("Permission denied"))
    def test_file_error_handling(self, mock_open):
        """Test handling of file errors"""
        # Should not raise an exception, just print a warning
        gitignore = GitIgnore("some_file.txt")
        self.assertEqual(len(gitignore.patterns), 0)

    def test_path_normalization(self):
        """Test path separator normalization"""
        patterns = ["docs/"]
        gitignore_path = self.create_gitignore_file(patterns)
        gitignore = GitIgnore(gitignore_path)

        # Should be ignored regardless of path separator
        self.assertTrue(gitignore.is_ignored("docs/index.html"))
        self.assertTrue(gitignore.is_ignored("docs\\index.html"))

if __name__ == '__main__':
    main()
