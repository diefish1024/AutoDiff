#ifndef DIFFERENTIATOR_HPP
#define DIFFERENTIATOR_HPP

#include <string>
#include <memory>

#include "expr_node.hpp"

namespace autodiff {
    class Differentiator {
    public:
        ExprNodePtr differentiate(const ExprNodePtr& expr, const std::string& var);

    private:
        ExprNodePtr diffOperator(const ExprNodePtr& expr, const std::string& var);
        ExprNodePtr diffFunction(const ExprNodePtr& expr, const std::string& var);
    };
    
    ExprNodePtr cloneSubtree(const ExprNodePtr& expr);

}; // namespace autodiff

#endif // DIFFERENTIATOR_HPP