#ifndef EXPRESSION_BUILDER_HPP
#define EXPRESSION_BUILDER_HPP

#include <string>
#include <vector>
#include <memory>

#include "expr_node.hpp"
#include "tokenizer.hpp"

namespace autodiff {
    class ExpressionBuilder {
    public:
        ExpressionBuilder(const std::vector<std::string>& tokens);
        ExprNodePtr build();
    private:
        const std::vector<std::string> tokens;
        int cur_index;

        ExprNodePtr parseExpression(); // fourth
        ExprNodePtr parseTerm(); // third
        ExprNodePtr parseFactor(); // second
        ExprNodePtr parsePrimary(); // first

        std::string peekToken() const;
        std::string consumeToken();
        bool isTokenAvailable() const;


        NodeType getTokenType(const std::string& token) const;
        OperatorType getOperatorType(const std::string& token) const;
        FunctionType getFunctionType(const std::string& token) const;

    };

}; // namespace autodiff

#endif // EXPRESSION_BUILDER_HPP