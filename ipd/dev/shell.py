import collections
import subprocess

BashResult = collections.namedtuple('BashResult', 'stdout, stderr, returncode')

def bash(cmd: str) -> BashResult:
    """Run a bash command and return the stdout, stderr, and return code."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return BashResult(result.stdout, result.stderr, result.returncode)

def run(command: str, echo: bool = False, errok: bool = False, strip: str | bool = 'auto') -> str:
    """Run a shell command and return the stdout.

    Args:
        command (str): The command to run.
        echo (bool): Whether to print the command before running it.
        errok (bool): Whether to raise an error if the command fails.
    Returns:
        str: The stdout of the command.
    """
    if echo: print(command, flush=True)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and not errok:
        raise RuntimeError(f'Error running command: {command}\n{result.stderr}')
    out = result.stdout
    if strip == 'auto': strip = out.count('\n') <= 1
    if strip: out = out.strip()
    return out
