#ifndef EXPR_NODE_HPP
#define EXPR_NODE_HPP

#include <string>
#include <memory>

namespace autodiff {
    enum class NodeType {
        NUMBER,
        VARIABLE,
        OPERATOR,
        FUNCTION
    };
    enum class OperatorType {
        ADD, SUB, MUL, DIV, POW,
        NONE_OP
    };
    enum class FunctionType {
        LN, LOG, COS, SIN, TAN, POW_FUNC, EXP,
        NONE_FUNC
    };

    struct ExprNode {
        NodeType type;
        std::string value;
        OperatorType opType;
        FunctionType funcType;
        std::unique_ptr<ExprNode> left;
        std::unique_ptr<ExprNode> right;

        ExprNode(NodeType t);
        ExprNode(NodeType t, std::string val); // NUMBER and VARIABLE
        ExprNode(NodeType t, OperatorType op); // OPERATOR
        ExprNode(NodeType t, FunctionType func); // FUNCTION
        // FUNCTION with single argument
        ExprNode(NodeType t, FunctionType func, std::unique_ptr<ExprNode> arg);
        // FUNCTION with two arguments
        ExprNode(NodeType t, FunctionType func, std::unique_ptr<ExprNode> arg1, std::unique_ptr<ExprNode> arg2);
        // OPERATOR with two arguments
        ExprNode(NodeType t, OperatorType op, std::unique_ptr<ExprNode> left, std::unique_ptr<ExprNode> right);
    };

    typedef std::unique_ptr<ExprNode> ExprNodePtr;

}; // namespace autodiff

#endif // EXPR_NODE_HPP