#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "expr_node.hpp"
#include "tokenizer.hpp"
#include "expression_builder.hpp"
#include "differentiator.hpp"
#include "tree_printer.hpp"

using namespace autodiff;

int main() {
    std::string expr;
    // std::cout << "Enter an expression: ";
    std::getline(std::cin, expr);

    Tokenizer tokenizer(expr);
    std::vector<std::string> tokens = tokenizer.tokenize();

    ExpressionBuilder builder(tokens);
    ExprNodePtr root = builder.build();

    Differentiator differentiator;
    TreePrinter printer;

    std::vector<std::string> vars = tokenizer.getVariables();
    std::sort(vars.begin(), vars.end());

    for (const std::string& var : vars) {
        ExprNodePtr diff = differentiator.differentiate(root, var);
        std::string derivativeExpr = printer.print(diff);
        std::cout << var << ": " << derivativeExpr << std::endl;
    }

    return 0;
}