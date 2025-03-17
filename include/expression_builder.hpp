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
    ExprNodePtr buildNumber(std::string token);
    ExprNodePtr buildVariable(std::string token);
    ExprNodePtr buildOperator(OperatorType opType);
    ExprNodePtr buildOperator(OperatorType opType, ExprNodePtr arg1, ExprNodePtr arg2);
    ExprNodePtr buildFunction(std::string token, FunctionType funcType, ExprNodePtr arg);
    ExprNodePtr buildFunction(std::string token, FunctionType funcType, ExprNodePtr arg1, ExprNodePtr arg2);

}; // namespace autodiff

#endif // EXPRESSION_BUILDER_HPP