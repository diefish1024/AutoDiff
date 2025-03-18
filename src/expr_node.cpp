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

ExprNodePtr autodiff::buildOperator(OperatorType opType, const ExprNodePtr& arg1, const ExprNodePtr& arg2) {
    return std::make_unique<ExprNode>(NodeType::OPERATOR, opType, cloneSubtree(arg1.get()), cloneSubtree(arg2.get()));
}

ExprNodePtr autodiff::buildFunction(FunctionType funcType, const ExprNodePtr& arg) {
    return std::make_unique<ExprNode>(NodeType::FUNCTION, funcType, cloneSubtree(arg.get()));
}

ExprNodePtr autodiff::buildFunction(FunctionType funcType, const ExprNodePtr& arg1, const ExprNodePtr& arg2) {
    return std::make_unique<ExprNode>(NodeType::FUNCTION, funcType, cloneSubtree(arg1.get()), cloneSubtree(arg2.get()));
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
                newNode = buildFunction(node->funcType, cloneSubtree(node->left.get()), cloneSubtree(node->right.get()));
            } else {
                newNode = buildFunction(node->funcType, cloneSubtree(node->left.get()));
            }
            break;
        }
        default:
            return nullptr;
    }
    return newNode;
}