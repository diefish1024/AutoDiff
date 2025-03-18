#ifndef SIMPLIFIER_HPP
#define SIMPLIFIER_HPP

#include "expr_node.hpp"

namespace autodiff {
    class Simplifier {
    public:
        ExprNodePtr simplify(ExprNodePtr node);
    private:
        ExprNodePtr simplifyNode(ExprNodePtr node);

        ExprNodePtr simplifyAdd(ExprNodePtr node);
        ExprNodePtr simplifySub(ExprNodePtr node);
        ExprNodePtr simplifyMul(ExprNodePtr node);
        ExprNodePtr simplifyDiv(ExprNodePtr node);
        ExprNodePtr simplifyPow(ExprNodePtr node);

        bool isNumberNode(const ExprNodePtr& expr, const std::string& value) const;
        bool isZero(const ExprNodePtr& expr) const;
        bool isOne(const ExprNodePtr& expr) const;
    };
};

#endif // SIMPLIFIER_HPP