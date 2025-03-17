#include <iostream>

#include "expression_builder.hpp"

using namespace autodiff;

ExpressionBuilder::ExpressionBuilder(const std::vector<std::string>& tokens) :tokens(tokens), cur_index(0) {}

ExprNodePtr ExpressionBuilder::build() {
    return parseExpression();
}

ExprNodePtr ExpressionBuilder::parseExpression() {
    ExprNodePtr left = parseTerm();

    while (isTokenAvailable()) {
        std::string op = peekToken();
        if (op == "+" || op == "-") {
            std::string token = consumeToken();
            OperatorType opType = getOperatorType(token);
            ExprNodePtr right = parseTerm();
            left = buildOperator(opType, std::move(left), std::move(right));
        } else {
            break;
        }
    }

    return left;
}

ExprNodePtr ExpressionBuilder::parseTerm() {
    ExprNodePtr left = parseFactor();

    while (isTokenAvailable()) {
        std::string op = peekToken();
        if (op == "*" || op == "/") {
            std::string token = consumeToken();
            OperatorType opType = getOperatorType(token);
            ExprNodePtr right = parseFactor();
            left = buildOperator(opType, std::move(left), std::move(right));
        } else {
            break;
        }
    }

    return left;
}

ExprNodePtr ExpressionBuilder::parseFactor() {
    ExprNodePtr left = parsePrimary();
    if (isTokenAvailable() && peekToken() == "^") {
        consumeToken();
        ExprNodePtr right = parseFactor();
        return buildOperator(OperatorType::POW, std::move(left), std::move(right));
    }
    return left;
}


ExprNodePtr ExpressionBuilder::parsePrimary() {
    if (!isTokenAvailable()) {
        std::cerr << "Error: Unexpected end of tokens in parsePrimary()" << std::endl;
        return nullptr; 
    }
    std::string token = consumeToken();
    if (getTokenType(token) == NodeType::NUMBER) {
        return buildNumber(token);
    } else if (getTokenType(token) == NodeType::VARIABLE) {
        return buildVariable(token);
    } else if (token == "(") {
        ExprNodePtr expression = parseExpression();
        if (isTokenAvailable() && peekToken() == ")") {
            consumeToken(); // ')'
            return expression;
        } else {
            std::cerr << "Error: Missing closing parenthesis ')'" << std::endl;
            return nullptr;
        }
    } else if (getTokenType(token) == NodeType::FUNCTION) {
        FunctionType funcType = getFunctionType(token);
        if (isTokenAvailable() && consumeToken() == "(") {
            if (funcType == FunctionType::LOG || funcType == FunctionType::POW_FUNC) {
                ExprNodePtr arg1 = parseExpression();
                if (isTokenAvailable() && consumeToken() == ",") {
                    ExprNodePtr arg2 = parseExpression();
                    if (isTokenAvailable() && consumeToken() == ")") {
                        return buildFunction(token, funcType, std::move(arg1), std::move(arg2));
                    }
                }
            } else { // ln, cos, sin, tan, exp
                ExprNodePtr arg = parseExpression();
                if (isTokenAvailable() && consumeToken() == ")") {
                    return buildFunction(token, funcType, std::move(arg));
                }
            }
        }
        std::cerr << "Error: Invalid function call for " << token << std::endl;
        return nullptr;
    } else {
        std::cerr << "Error: Unexpected token: " << token << std::endl;
        return nullptr;
    }
}

std::string ExpressionBuilder::peekToken() const {
    if (cur_index < tokens.size()) {
        return tokens[cur_index];
    }
    return "";
}

std::string ExpressionBuilder::consumeToken() {
    if (cur_index < tokens.size()) {
        return tokens[cur_index++];
    }
    return "";
}

bool ExpressionBuilder::isTokenAvailable() const {
    return cur_index < tokens.size();
}

NodeType ExpressionBuilder::getTokenType(const std::string& token) const {
    if (isDigit(token[0]) ||
        (token[0] == '-' && token.size() > 1 && isDigit(token[1]))) {
        return NodeType::NUMBER;
    } else if (isOperator(token[0])) {
        return NodeType::OPERATOR;
    } else if (isFunction(token)) {
        return NodeType::FUNCTION;
    }
    return NodeType::VARIABLE;
}

OperatorType ExpressionBuilder::getOperatorType(const std::string& token) const {
    switch (token[0]) {
    case '+':
        return OperatorType::ADD;
    case '-':
        return OperatorType::SUB;
    case '*':
        return OperatorType::MUL;
    case '/':
        return OperatorType::DIV;
    case '^':
        return OperatorType::POW;
    default:
        return OperatorType::NONE_OP;
    }
}

FunctionType ExpressionBuilder::getFunctionType(const std::string& token) const {
    if (token == "ln") {
        return FunctionType::LN;
    } else if (token == "log") {
        return FunctionType::LOG;
    } else if (token == "cos") {
        return FunctionType::COS;
    } else if (token == "sin") {
        return FunctionType::SIN;
    } else if (token == "tan") {
        return FunctionType::TAN;
    } else if (token == "exp") {
        return FunctionType::EXP;
    } else if (token == "pow") {
        return FunctionType::POW_FUNC;
    } else {
        return FunctionType::NONE_FUNC;
    }
}

ExprNodePtr autodiff::buildNumber(std::string token) {
    return std::make_unique<ExprNode>(NodeType::NUMBER, token);
}

ExprNodePtr autodiff::buildVariable(std::string token) {
    return std::make_unique<ExprNode>(NodeType::VARIABLE, token);
}

ExprNodePtr autodiff::buildOperator(OperatorType opType) {
    return std::make_unique<ExprNode>(NodeType::OPERATOR, opType);
}

ExprNodePtr autodiff::buildOperator(OperatorType opType, ExprNodePtr arg1, ExprNodePtr arg2) {
    return std::make_unique<ExprNode>(NodeType::OPERATOR, opType, std::move(arg1), std::move(arg2));
}

ExprNodePtr autodiff::buildFunction(std::string token, FunctionType funcType, ExprNodePtr arg) {
    return std::make_unique<ExprNode>(NodeType::FUNCTION, funcType, std::move(arg));
}

ExprNodePtr autodiff::buildFunction(std::string token, FunctionType funcType, ExprNodePtr arg1, ExprNodePtr arg2) {
    return std::make_unique<ExprNode>(NodeType::FUNCTION, funcType, std::move(arg1), std::move(arg2));
}