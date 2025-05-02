#include "_common.hpp"

// Character group indices for substitution matrix
enum CharGroup {
    UPPERCASE = 0,
    LOWERCASE = 1,
    DIGIT = 2,
    WHITESPACE = 3,
    // All Python punctuation characters as separate groups
    PAREN_OPEN = 4,    // (
    PAREN_CLOSE = 5,   // )
    BRACKET_OPEN = 6,  // [
    BRACKET_CLOSE = 7, // ]
    BRACE_OPEN = 8,    // {
    BRACE_CLOSE = 9,   // }
    DOT = 10,          // .
    COMMA = 11,        // ,
    COLON = 12,        // :
    SEMICOLON = 13,    // ;
    PLUS = 14,         // +
    MINUS = 15,        // -
    ASTERISK = 16,     // *
    SLASH = 17,        // /
    BACKSLASH = 18,    //
    VERTICAL_BAR = 19, // |
    AMPERSAND = 20,    // &
    LESS_THAN = 21,    // <
    GREATER_THAN = 22, // >
    EQUAL = 23,        // =
    PERCENT = 24,      // %
    HASH = 25,         // #
    AT_SIGN = 26,      // @
    EXCLAMATION = 27,  // !
    QUESTION = 28,     // ?
    CARET = 29,        // ^
    TILDE = 30,        // ~
    BACKTICK = 31,     // `
    QUOTE_SINGLE = 32, // '
    QUOTE_DOUBLE = 33, // "
    UNDERSCORE = 34,   // _
    DOLLAR = 35,       // $
    OTHER = 36,        // Other characters
    NUM_GROUPS
};

// Get character group for substitution matrix
CharGroup get_char_group(char c) {
    if (isupper(c)) return UPPERCASE;
    if (islower(c)) return LOWERCASE;
    if (isdigit(c)) return DIGIT;
    if (isspace(c)) return WHITESPACE;

    // Check for specific punctuation
    switch (c) {
    case '(':  return PAREN_OPEN;
    case ')':  return PAREN_CLOSE;
    case '[':  return BRACKET_OPEN;
    case ']':  return BRACKET_CLOSE;
    case '{':  return BRACE_OPEN;
    case '}':  return BRACE_CLOSE;
    case '.':  return DOT;
    case ',':  return COMMA;
    case ':':  return COLON;
    case ';':  return SEMICOLON;
    case '+':  return PLUS;
    case '-':  return MINUS;
    case '*':  return ASTERISK;
    case '/':  return SLASH;
    case '\\': return BACKSLASH;
    case '|':  return VERTICAL_BAR;
    case '&':  return AMPERSAND;
    case '<':  return LESS_THAN;
    case '>':  return GREATER_THAN;
    case '=':  return EQUAL;
    case '%':  return PERCENT;
    case '#':  return HASH;
    case '@':  return AT_SIGN;
    case '!':  return EXCLAMATION;
    case '?':  return QUESTION;
    case '^':  return CARET;
    case '~':  return TILDE;
    case '`':  return BACKTICK;
    case '\'': return QUOTE_SINGLE;
    case '"':  return QUOTE_DOUBLE;
    case '_':  return UNDERSCORE;
    case '$':  return DOLLAR;
    default:   return OTHER;
    }
}

// Default substitution matrix (higher score = more similar)
array<array<float, NUM_GROUPS>, NUM_GROUPS> create_default_submatrix() {
    array<array<float, NUM_GROUPS>, NUM_GROUPS> matrix{};

    // Initialize with zeroes
    for (int i = 0; i < NUM_GROUPS; i++) {
        for (int j = 0; j < NUM_GROUPS; j++) { matrix[i][j] = 0.0f; }
    }

    // Exact matches get 1.0
    for (int i = 0; i < NUM_GROUPS; i++) matrix[i][i] = 1.0f;

    const vector<CharGroup> keyGroups = {EQUAL, COLON, COMMA,    BRACKET_OPEN, PAREN_OPEN,
                                         PLUS,  MINUS, ASTERISK, SLASH,        UPPERCASE};

    for (const auto &group : keyGroups) matrix[group][group] = 5.0;
    matrix[EQUAL][EQUAL] = 10.0;

    // Letter case transitions get 0.9
    matrix[UPPERCASE][LOWERCASE] = 0.3f;
    matrix[LOWERCASE][UPPERCASE] = 0.3f;

    // Letters to digits get 0.5
    matrix[UPPERCASE][DIGIT] = 0.2f;
    matrix[LOWERCASE][DIGIT] = 0.2f;
    matrix[DIGIT][UPPERCASE] = 0.2f;
    matrix[DIGIT][LOWERCASE] = 0.2f;

    // Brackets/parentheses/braces are somewhat similar (0.3)
    matrix[PAREN_OPEN][BRACKET_OPEN] = 0.3f;
    matrix[PAREN_OPEN][BRACE_OPEN] = 0.3f;
    matrix[BRACKET_OPEN][PAREN_OPEN] = 0.3f;
    matrix[BRACKET_OPEN][BRACE_OPEN] = 0.3f;
    matrix[BRACE_OPEN][PAREN_OPEN] = 0.3f;
    matrix[BRACE_OPEN][BRACKET_OPEN] = 0.3f;

    matrix[PAREN_CLOSE][BRACKET_CLOSE] = 0.3f;
    matrix[PAREN_CLOSE][BRACE_CLOSE] = 0.3f;
    matrix[BRACKET_CLOSE][PAREN_CLOSE] = 0.3f;
    matrix[BRACKET_CLOSE][BRACE_CLOSE] = 0.3f;
    matrix[BRACE_CLOSE][PAREN_CLOSE] = 0.3f;
    matrix[BRACE_CLOSE][BRACKET_CLOSE] = 0.3f;

    // Operators have some similarity (0.4)
    matrix[PLUS][MINUS] = 0.4f;
    matrix[MINUS][PLUS] = 0.4f;
    matrix[ASTERISK][SLASH] = 0.4f;
    matrix[SLASH][ASTERISK] = 0.4f;
    matrix[LESS_THAN][GREATER_THAN] = 0.4f;
    matrix[GREATER_THAN][LESS_THAN] = 0.4f;

    // Quotes have similarity
    // matrix[QUOTE_SINGLE][QUOTE_DOUBLE] = 0.7f;
    // matrix[QUOTE_DOUBLE][QUOTE_SINGLE] = 0.7f;

    return matrix;
}

class IdentifyFormattedBlocks {
  public:
    array<array<float, NUM_GROUPS>, NUM_GROUPS> sub_matrix;
    bool in_formatted_block = false;
    vector<string> lines, output;
    vector<float> scores;
    size_t consecutive_high_scores = 0;
    float threshold = 5.0f;

    IdentifyFormattedBlocks(float threshold = 5.0f) : threshold(threshold) {
        sub_matrix = create_default_submatrix();
    }

    void set_substitution_matrix(CharGroup i, CharGroup j, float val) {
        sub_matrix[i][j] = val;
    }

    // Compute similarity score between two lines
    float compute_similarity_score(string const &line1, string const &line2) {
        if (debug) cerr << "compute_similarity_score " << line1 << " " << line2 << endl;
        if (line1.empty() || line2.empty()) return 0.0f;
        size_t indent1 = line1.find_first_not_of(" \t");
        size_t indent2 = line2.find_first_not_of(" \t");
        if (indent1 != indent2) return 0.0f;
        float alignmentScore = 0.0f;
        size_t len1 = line1.size();
        size_t len2 = line2.size();

        // Score character by character for alignment
        for (size_t i = 0; i < min(len1, len2); i++) {
            if (isalnum(static_cast<unsigned char>(line1[i])) &&
                isalnum(static_cast<unsigned char>(line2[i])) && line1[i] != line2[i])
                continue;
            CharGroup g1 = get_char_group(line1[i]);
            CharGroup g2 = get_char_group(line2[i]);
            if (debug) cerr << i << " g1 " << g1 << " g2 " << g2 << endl;
            alignmentScore += sub_matrix[g1][g2];
        }
        if (debug) cerr << "adject for len" << endl;
        float maxlen = static_cast<float>(max(line1.size(), line2.size()));
        alignmentScore = alignmentScore / sqrt(maxlen);
        float lengthPenalty =
            1.0f - (abs(static_cast<int>(len1) - static_cast<int>(len2)) /
                    static_cast<float>(max(len1, len2)));
        if (debug)
            cerr << "alignmentScore " << alignmentScore << " lengthPenalty "
                 << lengthPenalty << endl;
        return 0.7f * alignmentScore + 0.3f * lengthPenalty;
    }

    string unmark(string const &code) {
        start_new_code(code);
        if (lines.empty()) return code;

        for (string const &line : lines) {
            if (line.find("#             fmt:") != string::npos) continue;
            if (is_whitespace(line) && output.size() && is_whitespace(output.back()))
                continue;
            output.push_back(line);
        }
        ostringstream result;
        for (string const &line : output) { result << line << endl; }
        return result.str();
    }

    void start_new_code(string const &code) {
        lines.clear();
        output.clear();
        scores.clear();
        istringstream stream(code);
        string line;
        while (getline(stream, line)) lines.push_back(line);
        in_formatted_block = false;
    }
    string finish_code() {
        ostringstream result;
        for (const string &line : output) { result << line << endl; }
        return result.str();
    }

    // Process code to identify and mark well-formatted blocks
    string mark_formtted_blocks(string const &code, float thresh = 0) {
        start_new_code(code);
        if (thresh > 0) threshold = thresh;
        if (lines.empty()) return code;
        output.push_back(lines[0]);

        consecutive_high_scores = 0;
        for (size_t i = 1; i < lines.size(); i++) {
            if (is_multiline(lines[i - 1]) || is_multiline(lines[i])) {
                if (debug) cerr << "multiline " << lines[i] << endl;
                maybe_close_formatted_block();
                output.push_back(lines[i]);
                continue;
            }
            string i_indent = get_indentation(lines[i]);
            if (!in_formatted_block && is_oneline_statement_string(lines[i])) {
                if (debug) cerr << "oneline " << lines[i] << endl;
                maybe_close_formatted_block();
                // cout << "single " << lines[i] << endl;
                output.push_back(i_indent + "#             fmt: off");
                output.push_back(lines[i]);
                output.push_back(i_indent + "#             fmt: on");
                continue;
            }
            scores.push_back(compute_similarity_score(lines[i - 1], lines[i]));
            if (scores.back() >= threshold) {
                if (debug) cerr << "block " << scores.back() << " " << lines[i] << endl;
                consecutive_high_scores++;
                if (consecutive_high_scores >= 1 && !in_formatted_block) {
                    in_formatted_block = true;
                    string tmp = output.back();
                    output.back() = i_indent + "#             fmt: off";
                    output.push_back(tmp);
                    output.push_back(lines[i]);
                    continue;
                }
            } else {
                maybe_close_formatted_block();
            }
            output.push_back(lines[i]);
        }
        maybe_close_formatted_block(true);
        return finish_code();
    }
    void maybe_close_formatted_block(bool at_end = false) {
        if (!in_formatted_block) return;
        if (debug) cerr << "maybe close block" << endl;
        consecutive_high_scores = 0;
        in_formatted_block = false;
        string indent = "!!";
        assert(output.size());
        for (size_t i = output.size() - 1; i > 0; --i) {
            if (output[i].find("#             fmt:") == string::npos) {
                indent = get_indentation(output[i]);
                break;
            }
        }
        output.push_back(indent + "#             fmt: on");
        if (debug) cerr << "block closed" << endl;
    }
};

PYBIND11_MODULE(_detect_formatted_blocks, m) {
    m.doc() = "Identifies and marks well-formatted code blocks with fmt: off/on "
              "markers";

    py::class_<IdentifyFormattedBlocks>(m, "IdentifyFormattedBlocks")
        .def(py::init<>(), "Default constructor which initializes the "
                           "substitution matrix.")
        .def("set_substitution_matrix", &IdentifyFormattedBlocks::set_substitution_matrix,
             py::arg("i"), py::arg("j"), py::arg("val"),
             "Set a value in the substitution matrix at indices (i, j).")
        .def("compute_similarity_score",
             &IdentifyFormattedBlocks::compute_similarity_score, py::arg("line1"),
             py::arg("line2"), "Compute similarity score between two lines")
        .def("mark_formtted_blocks", &IdentifyFormattedBlocks::mark_formtted_blocks,
             py::arg("code"), py::arg("threshold") = 0.7f,
             "Process the input code and mark formatted blocks based on a "
             "similarity threshold.")
        .def("unmark", &IdentifyFormattedBlocks::unmark, py::arg("code"),
             "remove marks.");

    py::enum_<CharGroup>(m, "CharGroup")
        .value("UPPERCASE", UPPERCASE)
        .value("LOWERCASE", LOWERCASE)
        .value("DIGIT", DIGIT)
        .value("WHITESPACE", WHITESPACE)
        .value("PAREN_OPEN", PAREN_OPEN)
        .value("PAREN_CLOSE", PAREN_CLOSE)
        .value("BRACKET_OPEN", BRACKET_OPEN)
        .value("BRACKET_CLOSE", BRACKET_CLOSE)
        .value("BRACE_OPEN", BRACE_OPEN)
        .value("BRACE_CLOSE", BRACE_CLOSE)
        .value("DOT", DOT)
        .value("COMMA", COMMA)
        .value("COLON", COLON)
        .value("SEMICOLON", SEMICOLON)
        .value("PLUS", PLUS)
        .value("MINUS", MINUS)
        .value("ASTERISK", ASTERISK)
        .value("SLASH", SLASH)
        .value("BACKSLASH", BACKSLASH)
        .value("VERTICAL_BAR", VERTICAL_BAR)
        .value("AMPERSAND", AMPERSAND)
        .value("LESS_THAN", LESS_THAN)
        .value("GREATER_THAN", GREATER_THAN)
        .value("EQUAL", EQUAL)
        .value("PERCENT", PERCENT)
        .value("HASH", HASH)
        .value("AT_SIGN", AT_SIGN)
        .value("EXCLAMATION", EXCLAMATION)
        .value("QUESTION", QUESTION)
        .value("CARET", CARET)
        .value("TILDE", TILDE)
        .value("BACKTICK", BACKTICK)
        .value("QUOTE_SINGLE", QUOTE_SINGLE)
        .value("QUOTE_DOUBLE", QUOTE_DOUBLE)
        .value("UNDERSCORE", UNDERSCORE)
        .value("DOLLAR", DOLLAR)
        .value("OTHER", OTHER)
        .value("NUM_GROUPS", NUM_GROUPS)
        .export_values(); // Export values to the module scope
}
