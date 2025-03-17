#include <algorithm>
#include <iostream>
#include <string>

#include "tokenizer.hpp"

using namespace autodiff;

Tokenizer::Tokenizer(const std::string& expr) : expr(expr), cur_pos(0) {}

std::vector<std::string> Tokenizer::tokenize() {
    cur_pos = 0;
    std::vector<std::string> tokens;
    while (cur_pos < expr.size()) {
        char c = expr[cur_pos];
        if ((c == '-') && (cur_pos == 0 || expr[cur_pos - 1] == '(')) {
            tokens.push_back("-" + getNumber());
        }
        else if (isDigit(c)) {
            tokens.push_back(getNumber());
        } else if (isLetter(c)) {
            tokens.push_back(getLetters());
        } else if (isOperator(c)) {
            tokens.push_back(getOperator());
        } else {
            ++cur_pos;
        }
    }
    return tokens;
}

std::vector<std::string> Tokenizer::getVariables() {
    cur_pos = 0;
    std::vector<std::string> vars;
    while (cur_pos < expr.size()) {
        char c = expr[cur_pos];
        if (isLetter(c)) {
            std::string var = getLetters();
            if (!isFunction(var) && std::find(vars.begin(), vars.end(), var) == vars.end()) {
                vars.push_back(var);
            }
        } else {
            ++cur_pos;
        }
    }
    return vars;
}

std::string Tokenizer::getNumber() {
    std::string number;
    while (cur_pos < expr.size() && isDigit(expr[cur_pos])) {
        number += expr[cur_pos++];
    }
    return number;
}

std::string Tokenizer::getLetters() {
    std::string var;
    while (cur_pos < expr.size() && isLetter(expr[cur_pos])) {
        var += expr[cur_pos++];
    }
    return var;
}

std::string Tokenizer::getOperator() {
    std::string op;
    op += expr[cur_pos++];
    return op;
}

bool autodiff::isDigit(char c) {
    return c >= '0' && c <= '9';
}

bool autodiff::isLetter(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

bool autodiff::isOperator(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/'
        || c == '^' || c == '(' || c == ')' || c == ',';
}

bool autodiff::isFunction(const std::string& s) {
    return s == "sin" || s == "cos" || s == "tan" || s == "exp"
        || s == "log" || s == "ln" || s == "pow";
}