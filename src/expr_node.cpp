#include "expr_node.hpp"

using namespace autodiff;

ExprNode::ExprNode(NodeType t) : type(t) {}
ExprNode::ExprNode(NodeType t, std::string val) : type(t), value(val) {}
ExprNode::ExprNode(NodeType t, OperatorType op) : type(t), opType(op) {}
ExprNode::ExprNode(NodeType t, FunctionType func) : type(t), funcType(func) {}
ExprNode::ExprNode(NodeType t, FunctionType func, std::unique_ptr<ExprNode> arg) :
    type(t), funcType(func), left(std::move(arg)) {}
ExprNode::ExprNode(NodeType t, FunctionType func, std::unique_ptr<ExprNode> arg1, std::unique_ptr<ExprNode> arg2) :
    type(t), funcType(func), left(std::move(arg1)), right(std::move(arg2)) {}
ExprNode::ExprNode(NodeType t, OperatorType op, std::unique_ptr<ExprNode> l, std::unique_ptr<ExprNode> r) :
    type(t), opType(op), left(std::move(l)), right(std::move(r)) {}