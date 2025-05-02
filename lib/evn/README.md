# evn
Evn is like ruff, but more even. Why do all python formatters have to be such basic bitches? Ever seen what clang-format does to c++? It's all neat and *even* and lined up in columns. I like the hand-formatted code feel... why not a python formatter that does that?

```
operators = {
            "+",  "-",  "*",  "/",  "%", "**", "//", "==", "!=",
            "<",  ">",  "<=", ">=", "=", "->", "+=", "-=", "*=",
            "/=", "%=", "&",  "|",  "^", ">>", "<<", "~"};
```
```
keywords = {
            "False",  "None",     "True",  "and",    "as",       "assert",
            "async",  "await",    "break", "class",  "continue", "def",
            "del",    "elif",     "else",  "except", "finally",  "for",
            "from",   "global",   "if",    "import", "in",       "is",
            "lambda", "nonlocal", "not",   "or",     "pass",     "raise",
            "return", "try",      "while", "with",   "yield"};
```
```
struct LineInfo {
    int lineno;             // Line number.
    string line;            // Original line.
    string indent;          // Leading whitespace.
    string content;         // Line without indent.
    vector<string> tokens;  // Tokenized content.
    vector<string> pattern; // Token pattern (wildcards)
};
```
