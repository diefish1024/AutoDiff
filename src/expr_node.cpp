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

ExprNodePtr autodiff::buildFunction(FunctionType funcType, ExprNodePtr arg) {
    return std::make_unique<ExprNode>(NodeType::FUNCTION, funcType, std::move(arg));
}

ExprNodePtr autodiff::buildFunction(FunctionType funcType, ExprNodePtr arg1, ExprNodePtr arg2) {
    return std::make_unique<ExprNode>(NodeType::FUNCTION, funcType, std::move(arg1), std::move(arg2));
}