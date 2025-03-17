#ifndef TREE_PRINTER_HPP
#define TREE_PRINTER_HPP

#include <string>
#include <memory>

#include "expr_node.hpp"

namespace autodiff {
    class TreePrinter {
    public:
        std::string print(const ExprNodePtr& node) const;
    private:
        std::string printNode(const ExprNodePtr& node) const;
        std::string printOperator(const ExprNodePtr& node) const;
        std::string printFunction(const ExprNodePtr& node) const;

        int getPrecedence(const ExprNodePtr& node) const;
        bool needParentheses(const ExprNodePtr& parent, const ExprNodePtr& child, bool isRight) const;

        std::string getOperatorString(OperatorType op) const;
        std::string getFunctionString(FunctionType func) const;
    };

}; // namespace autodiff

#endif // TREE_PRINTER_HPP