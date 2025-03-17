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
    ExprNodePtr originalLeft = expr->left ? cloneSubtree(expr->left) : nullptr;
    ExprNodePtr originalRight = expr->right ? cloneSubtree(expr->right) : nullptr;
    switch (opType) {
        case OperatorType::ADD:
            return buildOperator(OperatorType::ADD, std::move(left), std::move(right)); // (u+v)' = u' + v'
        case OperatorType::SUB:
            return buildOperator(OperatorType::SUB, std::move(left), std::move(right)); // (u-v)' = u' - v'
        case OperatorType::MUL: // (u*v)' = u'v + uv'
            return buildOperator(OperatorType::ADD,
                buildOperator(OperatorType::MUL, std::move(left), std::move(originalRight)),
                buildOperator(OperatorType::MUL, std::move(originalLeft), std::move(right)));
        case OperatorType::DIV: // (u/v)' = (u'v - uv') / v^2
            return buildOperator(OperatorType::DIV,
                buildOperator(OperatorType::SUB,
                    buildOperator(OperatorType::MUL, std::move(left), std::move(originalRight)),
                    buildOperator(OperatorType::MUL, std::move(originalLeft), std::move(right))),
                buildOperator(OperatorType::POW, std::move(originalRight), buildNumber(std::string("2"))));
        case OperatorType::POW: { // (u^v)' = (v * u^(v-1) * u') + (ln(u) * u^v * v')
            ExprNodePtr v_minus_1 = buildOperator(OperatorType::SUB, std::move(originalRight), buildNumber(std::string("1")));
            ExprNodePtr u_pow_v_minus_1 = buildOperator(OperatorType::POW, std::move(originalLeft), std::move(v_minus_1));
            ExprNodePtr term1 = buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::MUL, std::move(originalRight), std::move(u_pow_v_minus_1)),
                std::move(left));
            ExprNodePtr term2 = buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::MUL, buildFunction(std::string("ln"), FunctionType::LN, std::move(originalLeft)), cloneSubtree(expr)),
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
    ExprNodePtr arg = expr->left ? differentiate(expr->left, var) : nullptr;
    ExprNodePtr originalArg = expr->left ? cloneSubtree(expr->left) : nullptr;
    ExprNodePtr arg1_deriv = expr->left ? differentiate(expr->left, var) : nullptr;
    ExprNodePtr arg2_deriv = expr->right ? differentiate(expr->right, var) : nullptr;
    ExprNodePtr originalArg1 = expr->left ? cloneSubtree(expr->left) : nullptr;
    ExprNodePtr originalArg2 = expr->right ? cloneSubtree(expr->right) : nullptr;
    switch (funcType) {
        case FunctionType::LN: // (ln(u))' = (1/u) * u'
            return buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::DIV, buildNumber(std::string("1")), std::move(originalArg)),
                std::move(arg));
        case FunctionType::LOG: {
            // todo: now the base is constant, need to handle the case when base is a variable
            ExprNodePtr base = std::move(originalArg1);
            ExprNodePtr value = std::move(originalArg2);
            ExprNodePtr value_deriv = std::move(arg2_deriv);
            ExprNodePtr ln_base = buildFunction(std::string("ln"), FunctionType::LN, std::move(base));
            ExprNodePtr denominator = buildOperator(OperatorType::MUL, std::move(value), std::move(ln_base));
            return buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::DIV, buildNumber(std::string("1")), std::move(denominator)),
                    std::move(value_deriv));
        }
        case FunctionType::COS: // (cos(u))' = -sin(u) * u'
            return buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::MUL, buildNumber(std::string("-1")), buildFunction(std::string("sin"),
                    FunctionType::SIN, std::move(originalArg))),
                std::move(arg));
        case FunctionType::SIN: // (sin(u))' = cos(u) * u'
            return buildOperator(OperatorType::MUL,
                buildFunction(std::string("cos"), FunctionType::COS, std::move(originalArg)),
                std::move(arg));
        case FunctionType::TAN: // (tan(u))' = (1/cos^2(u)) * u'
            return buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::DIV, buildNumber(std::string("1")),
                    buildOperator(OperatorType::POW, buildFunction(std::string("cos"), FunctionType::COS, std::move(originalArg)),
                        buildNumber(std::string("2")))),
                std::move(arg));
        case FunctionType::EXP: // (exp(u))' = exp(u) * u'
            return buildOperator(OperatorType::MUL,
                buildFunction(std::string("exp"), FunctionType::EXP, std::move(originalArg)),
                std::move(arg));
        case FunctionType::POW_FUNC: // pow(u, v)' = (u^v)'  (与 OperatorType::POW 相同)
             { // (u^v)' = (v * u^(v-1) * u') + (ln(u) * u^v * v')
                ExprNodePtr u = std::move(originalArg1);
                ExprNodePtr v = std::move(originalArg2);
                ExprNodePtr u_deriv = std::move(arg1_deriv);
                ExprNodePtr v_deriv = std::move(arg2_deriv);
                ExprNodePtr v_minus_1 = buildOperator(OperatorType::SUB, std::move(v), buildNumber(std::string("1")));
                ExprNodePtr u_pow_v_minus_1 = buildOperator(OperatorType::POW, std::move(u), std::move(v_minus_1));
                ExprNodePtr term1 = buildOperator(OperatorType::MUL,
                    buildOperator(OperatorType::MUL, std::move(v), std::move(u_pow_v_minus_1)),
                    std::move(u_deriv));
                ExprNodePtr term2 = buildOperator(OperatorType::MUL,
                    buildOperator(OperatorType::MUL, buildFunction(std::string("ln"), FunctionType::LN, std::move(u)),
                        buildFunction(std::string("pow"), FunctionType::POW_FUNC, std::move(u), std::move(v))), std::move(v_deriv));
                return buildOperator(OperatorType::ADD, std::move(term1), std::move(term2));
            }
        default:
            std::cerr << "Error: Unknown FunctionType in diffFunction" << std::endl;
            return nullptr;
    }
}

ExprNodePtr autodiff::cloneSubtree(const ExprNodePtr& node) {
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
            newNode = buildOperator(node->opType, cloneSubtree(node->left), cloneSubtree(node->right));
            break;
        case NodeType::FUNCTION: {
            if (node->right) {
                newNode = buildFunction(node->value, node->funcType, cloneSubtree(node->left), cloneSubtree(node->right));
            } else {
                newNode = buildFunction(node->value, node->funcType, cloneSubtree(node->left));
            }
            break;
        }
        default:
            return nullptr;
    }
    return newNode;
}
