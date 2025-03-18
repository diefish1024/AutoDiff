#include <iostream>
#include <cmath>
#include <string>

#include "simplifier.hpp"
#include "expr_node.hpp"

using namespace autodiff;

ExprNodePtr Simplifier::simplify(ExprNodePtr node) {
    if (!node) {
        return nullptr;
    }
    return simplifyNode(std::move(node));
}

ExprNodePtr Simplifier::simplifyNode(ExprNodePtr node) {
    if (!node) {
        return nullptr;
    }
    node->left = simplifyNode(std::move(node->left));
    node->right = simplifyNode(std::move(node->right));

    switch (node->type) {
        case NodeType::OPERATOR:
            switch (node->opType) {
                case OperatorType::ADD:
                    return simplifyAdd(std::move(node));
                case OperatorType::SUB:
                    return simplifySub(std::move(node));
                case OperatorType::MUL:
                    return simplifyMul(std::move(node));
                case OperatorType::DIV:
                    return simplifyDiv(std::move(node));
                case OperatorType::POW:
                    return simplifyPow(std::move(node));
                default:
                    return node;
            }
            break;
        default:
            return node;
    }
}

ExprNodePtr Simplifier::simplifyAdd(ExprNodePtr node) {
    ExprNodePtr& left = node->left;
    ExprNodePtr& right = node->right;
    if (isZero(left)) { // 0 + x = x
        return std::move(right);
    }
    if (isZero(right)) { // x + 0 = x
        return std::move(left);
    }
    if (left->type == NodeType::NUMBER && right->type == NodeType::NUMBER) {
        double leftValue = std::stod(left->value);
        double rightValue = std::stod(right->value);
        return buildNumber(std::to_string(leftValue + rightValue));
    }
    return node;
}

ExprNodePtr Simplifier::simplifySub(ExprNodePtr node) {
    ExprNodePtr& left = node->left;
    ExprNodePtr& right = node->right;
    if (isZero(left)) { // 0 - x = -x
        double value = std::stod(right->value);
        return buildNumber(std::to_string(-value));
    }
    if (isZero(right)) { // x - 0 = x
        return std::move(left);
    }
    if (left->type == NodeType::NUMBER && right->type == NodeType::NUMBER) {
        double leftValue = std::stod(left->value);
        double rightValue = std::stod(right->value);
        return buildNumber(std::to_string(leftValue - rightValue));
    }
    return node;
}

ExprNodePtr Simplifier::simplifyMul(ExprNodePtr node) {
    ExprNodePtr& left = node->left;
    ExprNodePtr& right = node->right;
    if (isZero(left) || isZero(right)) { // 0 * x = 0 or x * 0 = 0
        return buildNumber("0");
    }
    if (isOne(left)) { // 1 * x = x
        return std::move(right);
    }
    if (isOne(right)) { // x * 1 = x
        return std::move(left);
    }
    if (left->type == NodeType::NUMBER && right->type == NodeType::NUMBER) {
        double leftValue = std::stod(left->value);
        double rightValue = std::stod(right->value);
        return buildNumber(std::to_string(leftValue * rightValue));
    }
    return node;
}

ExprNodePtr Simplifier::simplifyDiv(ExprNodePtr node) {
    ExprNodePtr& left = node->left;
    ExprNodePtr& right = node->right;
    if (isZero(left)) { // 0 / x = 0
        return buildNumber("0");
    }
    if (isOne(right)) { // x / 1 = x
        return std::move(left);
    }
    if (left->type == NodeType::NUMBER && right->type == NodeType::NUMBER) {
        double leftValue = std::stod(left->value);
        double rightValue = std::stod(right->value);
        return buildNumber(std::to_string(leftValue / rightValue));
    }
    return node;
}

ExprNodePtr Simplifier::simplifyPow(ExprNodePtr node) {
    ExprNodePtr& left = node->left;
    ExprNodePtr& right = node->right;
    if (isZero(right)) { // x^0 = 1
        return buildNumber("1");
    }
    if (isOne(right)) { // x^1 = x
        return std::move(left);
    }
    if (left->type == NodeType::NUMBER && right->type == NodeType::NUMBER) {
        double leftValue = std::stod(left->value);
        double rightValue = std::stod(right->value);
        return buildNumber(std::to_string(std::pow(leftValue, rightValue)));
    }
    return node;
}

bool Simplifier::isNumberNode(const ExprNodePtr& expr, const std::string& value) const {
    return expr->type == NodeType::NUMBER && expr->value == value;
}

bool Simplifier::isZero(const ExprNodePtr& expr) const {
    return isNumberNode(expr, "0");
}

bool Simplifier::isOne(const ExprNodePtr& expr) const {
    return isNumberNode(expr, "1");
}