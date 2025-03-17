#include <iostream>
#include <string>
#include <vector>

#include "tree_printer.hpp"
#include "expr_node.hpp"

using namespace autodiff;

std::string TreePrinter::print(const ExprNodePtr& node) const {
    return printNode(node);
}

std::string TreePrinter::printNode(const ExprNodePtr& node) const {
    if (!node) {
        return "";
    }
    switch (node->type) {
        case NodeType::NUMBER:
        case NodeType::VARIABLE:
            return node->value;
        case NodeType::OPERATOR:
            return printOperator(node);
        case NodeType::FUNCTION:
            return printFunction(node);
    }
    return "";
}

std::string TreePrinter::printOperator(const ExprNodePtr& node) const {
    std::string leftStr = printNode(node->left);
    std::string rightStr = printNode(node->right);
    std::string opStr = getOperatorString(node->opType);

    bool leftParen = needParentheses(node, node->left, false);
    bool rightParen = needParentheses(node, node->right, true);

    if (leftParen) {
        leftStr = "(" + leftStr + ")";
    }
    if (rightParen) {
        rightStr = "(" + rightStr + ")";
    }

    return leftStr + opStr + rightStr;
}

std::string TreePrinter::printFunction(const ExprNodePtr& node) const {
    std::string funcStr = getFunctionString(node->funcType);
    FunctionType funcType = node->funcType;
    if (funcType == FunctionType::LOG || funcType == FunctionType::POW_FUNC) {
        std::string leftStr = printNode(node->left);
        std::string rightStr = printNode(node->right);
        return funcStr + "(" + leftStr + "," + rightStr + ")";
    } else {
        std::string argStr = printNode(node->left);
        return funcStr + "(" + argStr + ")";
    }
}

int TreePrinter::getPrecedence(const ExprNodePtr& node) const {
    if (node->type == NodeType::OPERATOR) {
        switch (node->opType) {
            case OperatorType::ADD:
            case OperatorType::SUB:
                return 1;
            case OperatorType::MUL:
            case OperatorType::DIV:
                return 2;
            case OperatorType::POW:
                return 3;
        }
    }
    return 0;
}

bool TreePrinter::needParentheses(const ExprNodePtr& parent, const ExprNodePtr& child, bool isRight) const {
    if (!child) {
        return false;
    }
    if (parent->type == NodeType::OPERATOR && child->type == NodeType::OPERATOR) {
        int parentPrec = getPrecedence(parent);
        int childPrec = getPrecedence(child);
        if (parentPrec == childPrec) {
            if (isRight) {
                return true;
            }
        } else if (parentPrec > childPrec) {
            return true;
        }
    }
    return false;
}

std::string TreePrinter::getOperatorString(OperatorType op) const {
    switch (op) {
        case OperatorType::ADD:
            return "+";
        case OperatorType::SUB:
            return "-";
        case OperatorType::MUL:
            return "*";
        case OperatorType::DIV:
            return "/";
        case OperatorType::POW:
            return "^";
    }
    return "";
}

std::string TreePrinter::getFunctionString(FunctionType func) const {
    switch (func) {
        case FunctionType::SIN:
            return "sin";
        case FunctionType::COS:
            return "cos";
        case FunctionType::TAN:
            return "tan";
        case FunctionType::LOG:
            return "log";
        case FunctionType::LN:
            return "ln";
        case FunctionType::EXP:
            return "exp";
        case FunctionType::POW_FUNC:
            return "pow";
    }
    return "";
}