#include "_common.hpp"

// Helper struct to store per–line data.
struct LineInfo {
    int lineno;             // Line number.
    string line;            // Original line.
    string indent;          // Leading whitespace.
    string content;         // Line without indent.
    vector<string> tokens;  // Tokenized content.
    vector<string> pattern; // Token pattern (wildcards)
};

class PythonLineTokenizer {
  public:
    // Reformat the given code buffer (as a string) into a new string.
    // Each line is processed, and consecutive lines that share the same
    // token pattern (by wildcard) and the same indentation are grouped and
    // aligned. If add_fmt_tag is true, formatting tags are added.
    string reformat_buffer(const string &code, bool add_fmt_tag = false,
                           bool debug = false) {
        istringstream stream(code);
        string line;
        vector<string> lines;
        while (getline(stream, line)) lines.push_back(line);
        vector<string> output = reformat_lines(lines, add_fmt_tag, debug);
        ostringstream result;
        for (const auto &outline : output) result << outline << "\n";
        return result.str();
    }

    // Process a vector of lines.
    vector<string> reformat_lines(const vector<string> &lines, bool add_fmt_tag = false,
                                  bool debug = false) {
        vector<LineInfo> infos = line_info(lines);
        vector<string> output;
        vector<LineInfo> block;
        const size_t length_threshold = 10;
        for (const auto &info : infos) {
            if (debug) cout << "reformat " << info.lineno << info.line << endl;
            // Blank lines are output as-is.
            if (info.content.empty()) {
                flush_block(block, output);
                output.push_back(rstrip(info.line));
                continue;
            }
            if (block.empty()) {
                block.push_back(info);
            } else {
                // Group lines if indent and token pattern match, and if lengths
                // are similar.
                try {
                    if (info.indent != block.at(0).indent ||
                        abs(static_cast<int>(info.line.size()) -
                            static_cast<int>(block.at(0).line.size())) >
                            length_threshold ||
                        info.pattern != block.at(0).pattern) {
                        flush_block(block, output, add_fmt_tag, debug);
                    }
                } catch (const out_of_range &e) {
                    throw runtime_error("Error grouping lines: " + string(e.what()));
                }
                block.push_back(info);
            }
        }
        flush_block(block, output, add_fmt_tag, debug);
        return output;
    }

    // Formats tokens by computing a delimiter for each token (except the
    // first). (This implementation is largely unchanged; error checking can be
    // added as needed.)
    vector<string> format_tokens(const vector<string> &tokens) {
        vector<string> formatted;
        if (tokens.empty()) return formatted;
        formatted.resize(tokens.size());
        formatted.at(0) = tokens.at(0); // first token: no preceding delimiter

        bool in_param_context = false;
        bool is_def = (tokens.at(0) == "def");
        bool is_lambda = (tokens.at(0) == "lambda");
        if (is_def) {
            in_param_context = false;
        } else if (is_lambda) {
            in_param_context = true;
        }

        int depth = 0;
        for (size_t i = 1; i < tokens.size(); i++) {
            string prev = tokens.at(i - 1);
            if (prev == "(") {
                depth++;
                if (is_def) in_param_context = true;
            } else if (prev == ")") {
                depth--;
                if (is_def && depth == 0) in_param_context = false;
            }
            if (is_lambda && tokens.at(i) == ":") { in_param_context = false; }
            string delim = delimiter(i - 1, i, tokens, in_param_context, depth);
            formatted.at(i) = delim + tokens.at(i);
        }
        return formatted;
    }

    // Joins tokens into a single string.
    // If skip_formatting is true, assumes tokens are already formatted.
    string join_tokens(const vector<string> &tokens,
                       const vector<int> &widths = vector<int>(),
                       const vector<char> &justifications = vector<char>(),
                       bool skip_formatting = false) {
        vector<string> formatted_tokens(tokens);
        if (!skip_formatting) formatted_tokens = format_tokens(tokens);
        if (!widths.empty() && widths.size() == formatted_tokens.size() &&
            !justifications.empty() && justifications.size() == formatted_tokens.size()) {
            for (size_t i = 0; i < formatted_tokens.size(); i++) {
                if (widths.at(i) > 0) {
                    int token_len = static_cast<int>(formatted_tokens.at(i).size());
                    int padding = static_cast<int>(widths.at(i)) - token_len;
                    if (padding > 0) {
                        char just = justifications.at(i);
                        if (just == 'L' || just == 'l') {
                            formatted_tokens.at(i).append(padding, ' ');
                        } else if (just == 'R' || just == 'r') {
                            formatted_tokens.at(i).insert(0, padding, ' ');
                        } else if (just == 'C' || just == 'c') {
                            int pad_left = padding / 2;
                            int pad_right = padding - pad_left;
                            formatted_tokens.at(i).insert(0, pad_left, ' ');
                            formatted_tokens.at(i).append(pad_right, ' ');
                        }
                    }
                }
            }
        }
        string result;
        for (const auto &tok : formatted_tokens) result += tok;
        return rstrip(result);
    }

    // Returns a vector of LineInfo for each line.
    vector<LineInfo> line_info(const vector<string> &lines) {
        vector<LineInfo> infos;
        for (int i = 0; i < lines.size(); i++) {
            LineInfo info;
            info.lineno = i;
            info.line = lines[i];
            size_t pos = info.line.find_first_not_of(" \t");
            info.indent = (pos == string::npos) ? info.line : info.line.substr(0, pos);
            info.content = (pos == string::npos) ? "" : info.line.substr(pos);
            if (!info.content.empty()) {
                info.tokens = tokenize(info.content);
                info.pattern = get_token_pattern(info.tokens);
            }
            infos.push_back(info);
        }
        return infos;
    }

    // Flushes a block of LineInfo objects into output.
    void flush_block(vector<LineInfo> &block, vector<string> &output,
                     bool add_fmt_tag = false, bool debug = false) {
        if (block.empty()) return;
        if (block.size() == 1) {
            LineInfo const &info = block.at(0);
            if (is_oneline_statement(info.tokens)) {
                output.push_back(info.indent + "#             fmt: off");
                output.push_back(rstrip(info.line));
                output.push_back(info.indent + "#             fmt: on");
            } else {
                output.push_back(rstrip(info.line));
            }
        } else {
            vector<vector<string>> token_lines;
            for (const auto &info : block) token_lines.push_back(info.tokens);
            vector<vector<string>> formatted_lines;
            for (auto &tokens : token_lines)
                formatted_lines.push_back(format_tokens(tokens));
            size_t nTokens = 0;
            for (auto &tokens : formatted_lines) nTokens = max(nTokens, tokens.size());
            vector<int> max_width(nTokens, 0);
            for (auto &tokens : formatted_lines) {
                for (size_t j = 0; j < tokens.size(); j++) {
                    max_width.at(j) =
                        max(max_width.at(j), static_cast<int>(tokens.at(j).size()));
                }
            }
            vector<char> justifications(nTokens, 'L');
            if (add_fmt_tag)
                output.push_back(block.at(0).indent + "#             fmt: off");
            for (auto &tokens : formatted_lines) {
                string joined = join_tokens(tokens, max_width, justifications, true);
                output.push_back(block.at(0).indent + joined);
            }
            if (add_fmt_tag)
                output.push_back(block.at(0).indent + "#             fmt: on");
        }
        block.clear();
    }
};

PYBIND11_MODULE(_token_column_format, m) {
    m.doc() = "A module that wraps PythonLineTokenizer using pybind11";
    py::class_<PythonLineTokenizer>(m, "PythonLineTokenizer")
        .def(py::init<>())
        .def("format_tokens", &PythonLineTokenizer::format_tokens,
             "Format tokens by prepending delimiters based on Black-like "
             "spacing heuristics")
        .def(
            "join_tokens",
            static_cast<string (PythonLineTokenizer::*)(
                const vector<string> &, const vector<int> &, const vector<char> &, bool)>(
                &PythonLineTokenizer::join_tokens),
            py::arg("tokens"), py::arg("widths") = vector<int>(),
            py::arg("justifications") = vector<char>(),
            py::arg("skip_formatting") = false,
            "Join tokens into a valid Python code line using Black-like "
            "heuristics. If skip_formatting is true, assume tokens are already "
            "formatted.")
        .def("reformat_buffer", &PythonLineTokenizer::reformat_buffer, py::arg("code"),
             py::arg("add_fmt_tag") = false, py::arg("debug") = false,
             "Reformat a code buffer, grouping lines with matching token "
             "patterns and indentation into blocks and aligning them into evn "
             "columns.")
        .def("reformat_lines", &PythonLineTokenizer::reformat_lines, py::arg("lines"),
             py::arg("add_fmt_tag") = false, py::arg("debug") = false,
             "Reformat a code buffer (given as a vector of lines) by grouping "
             "lines with matching token patterns and indentation into blocks "
             "and  inorkeywords.begin(), keywords.end(), <stcolumns.");

    m.def("tokenize", &tokenize, "Tokenize a single line of Python code");
    m.def("tokens_match", &tokens_match,
          "Compare two token vectors using wildcards for identifiers, "
          "strings, and numerics");
    m.def("is_oneline_statement", &is_oneline_statement, py::arg("tokens"),
          "Check if a line is an oneline statement");
}
