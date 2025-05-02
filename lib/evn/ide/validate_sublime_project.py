from pathlib import Path
from jsonschema import validate
import json5 as json
schema = {
    "type": "object",
    "properties": {
        "folders": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string"
                    },
                    "name": {
                        "type": "string"
                    },
                },
                "required": ["path"]
            }
        },
        "settings": {
            "type": "object"
        },
    },
    "required": ["folders"]
}
def main():
    path = Path(__file__).parent / "evn.sublime-project"
    project_json = json.load(open(path))
    # import evn.tree.tree_format
    # evn.show(project_json, format='tree')
    validate(instance=project_json, schema=schema)
if __name__ == '__main__':
    main()
