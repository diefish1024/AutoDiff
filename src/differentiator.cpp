#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include "differentiator.hpp"

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
    ExprNodePtr leftDerivative = expr->left ? differentiate(expr->left, var) : nullptr;
    ExprNodePtr rightDerivative = expr->right ? differentiate(expr->right, var) : nullptr;

    switch (opType) {
        case OperatorType::ADD:
            return buildOperator(OperatorType::ADD, leftDerivative, rightDerivative); 
        case OperatorType::SUB:
            return buildOperator(OperatorType::SUB, leftDerivative, rightDerivative); 
        case OperatorType::MUL: // (u*v)' = u'v + uv'
            return buildOperator(OperatorType::ADD,
                buildOperator(OperatorType::MUL, leftDerivative, expr->right),
                buildOperator(OperatorType::MUL, expr->left, rightDerivative));
        case OperatorType::DIV: { // (u/v)' = (u'v - uv') / v^2
            return buildOperator(OperatorType::DIV,
                buildOperator(OperatorType::SUB,
                    buildOperator(OperatorType::MUL, leftDerivative, expr->right),
                    buildOperator(OperatorType::MUL, expr->left, rightDerivative)),
                buildOperator(OperatorType::POW, expr->right, buildNumber(std::string("2"))));
        }
        case OperatorType::POW: { // (u^v)' = (v * u^(v-1) * u') + (ln(u) * u^v * v')
            ExprNodePtr vMinus1 = buildOperator(OperatorType::SUB, expr->right, buildNumber(std::string("1")));
            ExprNodePtr uPowVMinus1 = buildOperator(OperatorType::POW, expr->left, vMinus1);
            ExprNodePtr term1 = buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::MUL, expr->right, uPowVMinus1),
                leftDerivative);
            ExprNodePtr term2 = buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::MUL, buildFunction(FunctionType::LN, expr->left),
                    cloneSubtree(expr.get())), // Keep cloneSubtree for expr itself
                rightDerivative);
            return buildOperator(OperatorType::ADD, term1, term2);
        }
        default:
            std::cerr << "Error: Unknown OperatorType in diffOperator" << std::endl;
            return nullptr;
    }
}


ExprNodePtr Differentiator::diffFunction(const ExprNodePtr& expr, const std::string& var) {
    FunctionType funcType = expr->funcType;
    ExprNodePtr leftDerivative = expr->left ? differentiate(expr->left, var) : nullptr;
    ExprNodePtr rightDerivative = expr->right ? differentiate(expr->right, var) : nullptr;

    switch (funcType) {
        case FunctionType::LN: // (ln(u))' = (1/u) * u'
            return buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::DIV, buildNumber(std::string("1")), expr->left),
                leftDerivative);
        case FunctionType::LOG: { // log_base(value) = ln(value) / ln(base)
            ExprNodePtr lnValue = buildFunction(FunctionType::LN, expr->right);
            ExprNodePtr lnBase = buildFunction(FunctionType::LN, expr->left);
            
            ExprNodePtr numerator = buildOperator(OperatorType::SUB,
                buildOperator(OperatorType::MUL,buildOperator(OperatorType::DIV, rightDerivative, expr->right),
                    expr->left),
                buildOperator(OperatorType::MUL, expr->right,
                    buildOperator(OperatorType::DIV, leftDerivative, expr->left)));
            
            ExprNodePtr denominator = buildOperator(OperatorType::POW, lnBase, buildNumber(std::string("2")));
            return buildOperator(OperatorType::DIV, numerator, denominator);
        }                
        case FunctionType::COS: // (cos(u))' = -sin(u) * u'
            return buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::MUL, buildNumber(std::string("-1")),
                    buildFunction(FunctionType::SIN, expr->left)),
                leftDerivative);
        case FunctionType::SIN: // (sin(u))' = cos(u) * u'
            return buildOperator(OperatorType::MUL,
                buildFunction(FunctionType::COS, expr->left),
                leftDerivative);
        case FunctionType::TAN: // (tan(u))' = (1/cos^2(u)) * u'
            return buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::DIV, buildNumber(std::string("1")),
                    buildOperator(OperatorType::POW, buildFunction(FunctionType::COS, expr->left),
                        buildNumber(std::string("2")))),
                leftDerivative);
        case FunctionType::EXP: // (exp(u))' = exp(u) * u'
            return buildOperator(OperatorType::MUL,
                buildFunction(FunctionType::EXP, expr->left),
                leftDerivative);
        case FunctionType::POW_FUNC: { // pow(u, v)' = (u^v)'
            // (u^v)' = (v * u^(v-1) * u') + (ln(u) * u^v * v')
            ExprNodePtr vMinus1 = buildOperator(OperatorType::SUB, expr->right, buildNumber(std::string("1")));
            ExprNodePtr uPowVMinus1 = buildOperator(OperatorType::POW, expr->left, vMinus1);
            ExprNodePtr term1 = buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::MUL, expr->right, uPowVMinus1),
                leftDerivative);

            ExprNodePtr lnU = buildFunction(FunctionType::LN, expr->left);
            ExprNodePtr powUV = buildOperator(OperatorType::POW, expr->left, expr->right);

            ExprNodePtr term2 = buildOperator(OperatorType::MUL,
                buildOperator(OperatorType::MUL, lnU, powUV),
                rightDerivative);

            return buildOperator(OperatorType::ADD, term1, term2);
        }
        default:
            std::cerr << "Error: Unknown FunctionType in diffFunction" << std::endl;
            return nullptr;
    }
}

