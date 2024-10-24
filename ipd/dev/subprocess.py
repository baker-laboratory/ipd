import subprocess

def bash(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True)
    return result.stdout.decode(), result.stderr.decode(), result.returncode

def run(command: str, echo: bool = True):
    if echo: print(command, flush=True)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'Error running command: {command}\n{result.stderr}')
    if echo: print(result.stdout, end='')
    return result.stdout
