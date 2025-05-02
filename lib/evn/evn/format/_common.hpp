// format_identifier.cpp
#include <algorithm>
#include <cctype>
#include <iostream>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <ranges>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace std;
bool debug = false;

enum class TokenType {
    Identifier,
    String,
    Numeric,
    Exact // Keywords, punctuation, comments, etc.
};
// Get indentation level of a line
string get_indentation(string const &line) {
    auto nonWhitespace = line.find_first_not_of(" \t");
    if (nonWhitespace == string::npos) { return ""; }
    return line.substr(0, nonWhitespace);
}

bool is_whitespace(const std::string &str) {
    return str.empty() || std::all_of(str.begin(), str.end(),
                                      [](unsigned char c) { return std::isspace(c); });
}

// Returns the index of the first non-whitespace character from the end of the
// string or std::string::npos if the string contains only whitespace.
size_t find_last_non_whitespace(const std::string &str) {
    for (std::size_t i = str.size(); i > 0; --i) {
        if (!std::isspace(static_cast<unsigned char>(str[i - 1]))) { return i - 1; }
    }
    return std::string::npos;
}

bool is_multiline(string const &line) {
    size_t i = find_last_non_whitespace(line);
    if (i == string::npos) return false; // Empty line
    return line[i] == '\\';
}

bool is_opener(const string &token) {
    return token == "(" || token == "[" || token == "{";
}

bool is_closer(const string &token) {
    return token == ")" || token == "]" || token == "}";
}

bool is_operator(const string &token) {
    static const unordered_set<string> operators = {
        "+", "-",  "*",  "/",  "%",  "**", "//", "==", "!=", "<", ">",  "<=", ">=",
        "=", "->", "+=", "-=", "*=", "/=", "%=", "&",  "|",  "^", ">>", "<<", "~"};
    return operators.find(token) != operators.end();
}

bool is_keyword(const string &token) {
    static const unordered_set<string> python_keywords = {
        "False",  "None",   "True",    "and",      "as",       "assert", "async",
        "await",  "break",  "class",   "continue", "def",      "del",    "elif",
        "else",   "except", "finally", "for",      "from",     "global", "if",
        "import", "in",     "is",      "lambda",   "nonlocal", "not",    "or",
        "pass",   "raise",  "return",  "try",      "while",    "with",   "yield"};
    return python_keywords.find(token) != python_keywords.end();
}

string rstrip(const string &str) {
    string trimmed_str = str;
    auto it = find_if(trimmed_str.rbegin(), trimmed_str.rend(),
                      [](unsigned char ch) { return !isspace(ch); });
    trimmed_str.erase(it.base(), trimmed_str.end());
    return trimmed_str;
}

// Helper functions for token type checking.
bool is_string_literal(const string &token) {
    if (token.empty()) return false;
    if (token.at(0) == '\'' || token.at(0) == '"') return true;
    if (token.size() >= 2 && (token.at(0) == 'f' || token.at(0) == 'F') &&
        (token.at(1) == '\'' || token.at(1) == '"'))
        return true;
    return false;
}

bool is_identifier(const string &token) {
    if (token.empty()) return false;
    if (!isalpha(static_cast<unsigned char>(token.at(0))) && token.at(0) != '_')
        return false;
    for (size_t i = 1; i < token.size(); i++) {
        if (!isalnum(static_cast<unsigned char>(token.at(i))) && token.at(i) != '_')
            return false;
    }
    return true;
}
TokenType get_token_type(const string &token) {
    if (is_string_literal(token)) return TokenType::String;
    if (is_identifier(token)) {
        if (is_keyword(token)) return TokenType::Exact;
        return TokenType::Identifier;
    }
    if (!token.empty() && isdigit(static_cast<unsigned char>(token.at(0))))
        return TokenType::Numeric;
    return TokenType::Exact;
}

bool is_identifier_or_literal(const string &token) {
    TokenType t = get_token_type(token);
    return (t == TokenType::Identifier || t == TokenType::String ||
            t == TokenType::Numeric);
}

bool is_oneline_statement_string(string const &line) {
    if (line.empty()) { return false; }
    string trimmed = line;
    size_t firstNonSpace = trimmed.find_first_not_of(" \t");
    if (firstNonSpace == string::npos) return false; // Empty line
    trimmed = trimmed.substr(firstNonSpace);
    if (trimmed[0] == '#') { return false; }
    const vector<string> keywords = {"if ",    "elif ", "else:",  "for ",
                                     "while ", "def ",  "class ", "with "};
    bool foundKeyword = false;
    string keywordFound;
    for (const auto &keyword : keywords) {
        if (trimmed.compare(0, keyword.length(), keyword) == 0) {
            foundKeyword = true;
            keywordFound = keyword;
            break;
        }
    }
    if (!foundKeyword) return false;

    // Now we need to find the colon that ends the statement header
    size_t colonPos = 0;
    bool inString = false;
    char stringDelimiter = 0;
    bool escaped = false;
    int parenLevel = 0;

    // For else:, we already know the colon position
    if (keywordFound == "else:") {
        colonPos = firstNonSpace + 4; // "else" length
    } else {
        // For other keywords, we need to find the colon
        for (size_t i = 0; i < trimmed.length(); i++) {
            char c = trimmed[i];

            // Handle string delimiters
            if ((c == '"' || c == '\'') && !escaped) {
                if (!inString) {
                    inString = true;
                    stringDelimiter = c;
                } else if (c == stringDelimiter) {
                    inString = false;
                }
            }

            // Handle escaping
            if (c == '\\' && !escaped) {
                escaped = true;
                continue;
            } else {
                escaped = false;
            }

            // Track parentheses level (ignore if in string)
            if (!inString) {
                if (c == '(' || c == '[' || c == '{') {
                    parenLevel++;
                } else if (c == ')' || c == ']' || c == '}') {
                    parenLevel--;
                } else if (c == ':' && parenLevel == 0) {
                    colonPos = firstNonSpace + i;
                    break;
                }
            }
        }
    }

    // If we couldn't find a proper colon, it's not a valid statement
    if (colonPos == 0 || colonPos >= line.length() - 1) { return false; }

    // Now check if there's an action after the colon
    string afterColon = line.substr(colonPos + 1);
    size_t actionStart = afterColon.find_first_not_of(" \t");

    // If there's nothing after the colon or just a comment, it's not an inline
    // action
    if (actionStart == string::npos || afterColon[actionStart] == '#') { return false; }

    return true;
}

bool is_oneline_statement(vector<string> const &tokens) {
    if (tokens.empty()) return false;
    static const vector<string> keywords = {"if",    "elif", "else",  "for",
                                            "while", "def",  "class", "with"};
    if (find(keywords.begin(), keywords.end(), tokens[0]) == keywords.end()) return false;
    for (int i = 1; i < tokens.size(); ++i)
        if (tokens[i] == ":") {
            if (i == tokens.size() - 1) return false;
            if (tokens[i + 1][0] == '#') return false;
            return true;
        }
    return false;
}

// Delimiter helper: returns the delimiter to insert before the current
// token.
string delimiter(size_t prev_index, size_t curr_index, const vector<string> &tokens,
                 bool in_param_context, int depth) {
    const string &prev = tokens.at(prev_index);
    const string &next = tokens.at(curr_index);
    if (in_param_context && (prev == "=" || next == "=")) return "";
    if (is_operator(prev) || is_operator(next)) {
        if (depth > 1 && (prev == "+" || prev == "-" || next == "+" || next == "-"))
            return "";
        return " ";
    }
    if (is_opener(prev)) return "";
    if (is_closer(next)) return "";
    if (next == "," || next == ":" || next == ";") return "";
    if (next == "(" && is_identifier_or_literal(prev) && !is_keyword(prev)) return "";
    return " ";
}

// Parses a string literal from the given line starting at index i.
string parse_string_literal(const string &line, size_t &i, bool is_f_string) {
    size_t start = i;
    if (is_f_string) ++i; // skip the 'f' or 'F'
    if (i >= line.size()) throw out_of_range("String literal start index out of range");
    char quote = line.at(i);
    bool triple = false;
    if (i + 2 < line.size() && line.at(i) == line.at(i + 1) &&
        line.at(i) == line.at(i + 2)) {
        triple = true;
        i += 3;
    } else {
        ++i;
    }
    while (i < line.size()) {
        if (line.at(i) == '\\') {
            i += 2;
        } else if (triple) {
            if (i + 2 < line.size() && line.at(i) == quote && line.at(i + 1) == quote &&
                line.at(i + 2) == quote) {
                i += 3;
                break;
            } else {
                ++i;
            }
        } else {
            if (line.at(i) == quote) {
                ++i;
                break;
            } else {
                ++i;
            }
        }
    }
    return line.substr(start, i - start);
}

// Tokenizes a single line of Python code.
vector<string> tokenize(const string &line) {
    vector<string> tokens;
    size_t i = 0;
    while (i < line.size()) {
        // Skip whitespace.
        if (isspace(static_cast<unsigned char>(line.at(i)))) {
            ++i;
            continue;
        }
        // Handle comments: rest of the line is one token.
        if (line.at(i) == '#') {
            tokens.push_back(line.substr(i));
            break;
        }
        // Check for an f-string literal.
        if ((line.at(i) == 'f' || line.at(i) == 'F') && (i + 1 < line.size()) &&
            (line.at(i + 1) == '\'' || line.at(i + 1) == '"')) {
            tokens.push_back(parse_string_literal(line, i, true));
            continue;
        }
        // Check for a normal string literal.
        if (line.at(i) == '\'' || line.at(i) == '"') {
            tokens.push_back(parse_string_literal(line, i, false));
            continue;
        }
        // Check for an identifier or keyword.
        if (isalpha(static_cast<unsigned char>(line.at(i))) || line.at(i) == '_') {
            size_t start = i;
            while (i < line.size() && (isalnum(static_cast<unsigned char>(line.at(i))) ||
                                       line.at(i) == '_')) {
                ++i;
            }
            tokens.push_back(line.substr(start, i - start));
            continue;
        }
        // Handle numeric literals in a basic way.
        if (isdigit(static_cast<unsigned char>(line.at(i)))) {
            size_t start = i;
            while (i < line.size() &&
                   (isdigit(static_cast<unsigned char>(line.at(i))) ||
                    line.at(i) == '.' || line.at(i) == 'e' || line.at(i) == 'E' ||
                    line.at(i) == '+' || line.at(i) == '-')) {
                ++i;
            }
            tokens.push_back(line.substr(start, i - start));
            continue;
        }
        // Check for multi-character punctuation/operators.
        bool multi_matched = false;
        static const vector<string> multi_tokens = {"...", "==", "!=", "<=", ">=", "//",
                                                    "**",  "->", "+=", "-=", "*=", "/=",
                                                    "%=",  "&=", "|=", "^=", ">>", "<<"};
        for (const auto &tok : multi_tokens) {
            if (line.compare(i, tok.size(), tok) == 0) {
                tokens.push_back(tok);
                i += tok.size();
                multi_matched = true;
                break;
            }
        }
        if (multi_matched) continue;
        // Single-character punctuation.
        try {
            tokens.push_back(string(1, line.at(i)));
        } catch (const out_of_range &e) {
            throw runtime_error("Index error in tokenize at position " + to_string(i));
        }
        ++i;
    }
    return tokens;
}

// Returns a token pattern for grouping.
vector<string> get_token_pattern(const vector<string> &tokens) {
    vector<string> pattern;
    for (const auto &tok : tokens) {
        if (is_string_literal(tok))
            pattern.push_back("STR");
        else if (is_identifier(tok) && !is_keyword(tok))
            pattern.push_back("ID");
        else if (!tok.empty() && isdigit(static_cast<unsigned char>(tok.at(0))))
            pattern.push_back("NUM");
        else
            pattern.push_back(tok);
    }
    return pattern;
}

// Compares two token vectors using wildcard rules.
bool tokens_match(const vector<string> &tokens1, const vector<string> &tokens2) {
    if (tokens1.size() != tokens2.size()) return false;
    for (size_t i = 0; i < tokens1.size(); i++) {
        TokenType type1 = get_token_type(tokens1.at(i));
        TokenType type2 = get_token_type(tokens2.at(i));
        if (type1 != type2) return false;
        if (type1 == TokenType::Exact && tokens1.at(i) != tokens2.at(i)) return false;
    }
    return true;
}
