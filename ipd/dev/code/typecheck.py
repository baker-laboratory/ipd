import collections
import json
import sys
from pathlib import Path

import ipd

def get_pyright_errors(codedir: str) -> list[dict]:
    """Run pyright and return parsed error output."""
    try:
        result = ipd.dev.run(f'pyright --outputjson {codedir}', errok=True, echo=True)
        output = json.loads(result)
        # rich.print(output)
        return output.get('generalDiagnostics', [])
    except json.JSONDecodeError as e:
        print(f'Error parsing pyright output: {e}')
        sys.exit(1)

def add_type_ignore_comments(errors: list[dict]) -> None:
    """Add '# type: ignore' comments to lines with type errors."""
    files_to_modify = collections.defaultdict(set)
    for error in errors:
        fname = error.get('file', '')
        line_number = error.get('range', {}).get('start', {}).get('line', 0)
        files_to_modify[fname].add(line_number)

    for fname, lineno in files_to_modify.items():
        path = Path(fname)
        lines = path.read_text().splitlines()
        modified_lines = []
        for i, line in enumerate(lines):
            if i in lineno and '# type: ignore' not in line:
                assert line.strip()
                line = f'{line.rstrip()}  # type: ignore'
            modified_lines.append(line)

        # Write modified content back to file
        path.write_text('\n'.join(modified_lines) + '\n')
        print(f'Modified {fname}: added type: ignore to {len(lineno)} lines')
