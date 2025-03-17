#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include "differentiator.hpp"
#include "expression_builder.hpp"

using namespace autodiff;

ExprNodePtr Differentiator::differentiate(const ExprNodePtr& expr, const std::string& var) {
    if (!expr) {
        return nullptr;
    }
    switch (expr->type) {
        case NodeType::NUMBER:
            return buildNumber(std::string("0"));
        case NodeType::VARIABLE:
            if (expr->value == var) {
                return buildNumber(std::string("1"));
            } else {
                return buildNumber(std::string("0"));
            }
        case NodeType::OPERATOR:
            return diffOperator(expr, var);
        case NodeType::FUNCTION:
            return diffFunction(expr, var);
        default:
            std::cerr << "Error: Unknown NodeType in differentiate" << std::endl;
            return nullptr;
    }
}

ExprNodePtr Differentiator::diffOperator(const ExprNodePtr& expr, const std::string& var) {
    OperatorType opType = expr->opType;
    ExprNodePtr left = expr->left ? differentiate(expr->left, var) : nullptr;
    ExprNodePtr right = expr->right ? differentiate(expr->right, var) : nullptr;
    ExprNodePtr originalLeft = expr->left ? cloneSubtree(expr->left.get()) : nullptr;
    ExprNodePtr originalRight = expr->right ? cloneSubtree(expr->right.get()) : nullptr;
    switch (opType) {
        case OperatorType::ADD:
            return buildOperator(OperatorType::ADD, std::move(left), std::move(right)); // (u+v)' = u' + v'
        case OperatorType::SUB:
            return buildOperator(OperatorType::SUB, std::move(left), std::move(right)); // (u-v)' = u' - v'
        case OperatorType::MUL: // (u*v)' = u'v + uv'
            return buildOperator(OperatorType::ADD,
                buildOperator(OperatorType::MUL, std::move(left), std::move(originalRight)),
                buildOperator(OperatorType::MUL, std::move(originalLeft), std::move(right)));
        case OperatorType::DIV: { // (u/v)' = (u'v - uv') / v^2
            ExprNodePtr originalRightCopy = cloneSubtree(originalRight.get());
            return buildOperator(OperatorType::DIV,
                buildOperator(OperatorType::SUB,
                    buildOperator(OperatorType::MUL, std::move(left), std::move(originalRightCopy)),
                    buildOperator(OperatorType::MUL, std::move(originalLeft), std::move(right))),
                    buildOperator(OperatorType::POW, std::move(originalRight), buildNumber(std::string("2"))));
        }
        case OperatorType::POW: { // (u^v)' = (v * u^(v-1) * u') + (ln(u) * u^v * v')
            ExprNodePtr originalRightCopy = cloneSubtree(originalRight.get());
            ExprNodePtr originalLeftCopy = cloneSubtree(originalLeft.get());
            ExprNodePtr vMinus1 = buildOperator(OperatorType::SUB, std::move(originalRight), buildNumber(std::string("1")));
            ExprNodePtr uPowVMinus1 = buildOperator(OperatorType::POW, std::move(originalLeft), std::move(vMinus1));
            ExprNodePtr term1 = buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::MUL, std::move(originalRightCopy), std::move(uPowVMinus1)),
                std::move(left));
            ExprNodePtr term2 = buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::MUL, buildFunction(std::string("ln"), FunctionType::LN, std::move(originalLeftCopy)),
                    cloneSubtree(expr.get())),
                std::move(right));
            return buildOperator(OperatorType::ADD, std::move(term1), std::move(term2));
        }
        default:
            std::cerr << "Error: Unknown OperatorType in diffOperator" << std::endl;
            return nullptr;
    }
}

ExprNodePtr Differentiator::diffFunction(const ExprNodePtr& expr, const std::string& var) {
    FunctionType funcType = expr->funcType;
    ExprNodePtr left = expr->left ? differentiate(expr->left, var) : nullptr;
    ExprNodePtr right = expr->right ? differentiate(expr->right, var) : nullptr;
    ExprNodePtr originalLeft = expr->left ? cloneSubtree(expr->left.get()) : nullptr;
    ExprNodePtr originalRight = expr->right ? cloneSubtree(expr->right.get()) : nullptr;

    switch (funcType) {
        case FunctionType::LN: // (ln(u))' = (1/u) * u'
            return buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::DIV, buildNumber(std::string("1")), std::move(originalLeft)),
                std::move(left));
        case FunctionType::LOG: { // Assuming base is constant.
            ExprNodePtr base = std::move(originalLeft);
            ExprNodePtr value = std::move(originalRight);
            ExprNodePtr valueDeriv = std::move(right);
            ExprNodePtr lnBase = buildFunction(std::string("ln"), FunctionType::LN, std::move(base));
            ExprNodePtr denominator = buildOperator(OperatorType::MUL, std::move(value), std::move(lnBase));
            return buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::DIV, buildNumber(std::string("1")), std::move(denominator)),
                std::move(valueDeriv));
        }
        case FunctionType::COS: // (cos(u))' = -sin(u) * u'
            return buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::MUL, buildNumber(std::string("-1")),
                    buildFunction(std::string("sin"), FunctionType::SIN, std::move(originalLeft))),
                std::move(left));
        case FunctionType::SIN: // (sin(u))' = cos(u) * u'
            return buildOperator(OperatorType::MUL,
                buildFunction(std::string("cos"), FunctionType::COS, std::move(originalLeft)),
                std::move(left));
        case FunctionType::TAN: // (tan(u))' = (1/cos^2(u)) * u'
            return buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::DIV, buildNumber(std::string("1")),
                    buildOperator(OperatorType::POW, buildFunction(std::string("cos"), FunctionType::COS, std::move(originalLeft)),
                        buildNumber(std::string("2")))),
                std::move(left));
        case FunctionType::EXP: // (exp(u))' = exp(u) * u'
            return buildOperator(OperatorType::MUL,
                buildFunction(std::string("exp"), FunctionType::EXP, std::move(originalLeft)),
                std::move(left));
        case FunctionType::POW_FUNC: { // pow(u, v)' = (u^v)'
            // (u^v)' = (v * u^(v-1) * u') + (ln(u) * u^v * v')
            ExprNodePtr originalLeftCopy = cloneSubtree(originalLeft.get());
            ExprNodePtr originalRightCopy = cloneSubtree(originalRight.get());
            ExprNodePtr vMinus1 = buildOperator(OperatorType::SUB, std::move(originalRight), buildNumber(std::string("1")));
            ExprNodePtr uPowVMinus1 = buildOperator(OperatorType::POW, std::move(originalLeft), std::move(vMinus1));
            ExprNodePtr term1 = buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::MUL, std::move(originalRightCopy), std::move(uPowVMinus1)),
                std::move(left));

            ExprNodePtr lnU = buildFunction(std::string("ln"), FunctionType::LN, std::move(originalLeftCopy));
            ExprNodePtr powUV = buildOperator(OperatorType::POW, std::move(originalLeft), std::move(originalRight));

            ExprNodePtr term2 = buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::MUL, std::move(lnU), std::move(powUV)),
                std::move(right));

            return buildOperator(OperatorType::ADD, std::move(term1), std::move(term2));
        }
        default:
            std::cerr << "Error: Unknown FunctionType in diffFunction" << std::endl;
            return nullptr;
    }
}


ExprNodePtr autodiff::cloneSubtree(const ExprNode* node) {
    if (!node) {
        return nullptr;
    }

    ExprNodePtr newNode;
    switch (node->type) {
        case NodeType::NUMBER:
            newNode = buildNumber(node->value);
            break;
        case NodeType::VARIABLE:
            newNode = buildVariable(node->value);
            break;
        case NodeType::OPERATOR:
            newNode = buildOperator(node->opType, cloneSubtree(node->left.get()), cloneSubtree(node->right.get()));
            break;
        case NodeType::FUNCTION: {
            if (node->right) {
                newNode = buildFunction(node->value, node->funcType, cloneSubtree(node->left.get()), cloneSubtree(node->right.get()));
            } else {
                newNode = buildFunction(node->value, node->funcType, cloneSubtree(node->left.get()));
            }
            break;
        }
        default:
            return nullptr;
    }
    return newNode;
}
