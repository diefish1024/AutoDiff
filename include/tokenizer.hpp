#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <vector>

namespace autodiff {
    class Tokenizer {
    public:
        Tokenizer(const std::string& expr);
        std::vector<std::string> tokenize();
        std::vector<std::string> getVariables();

    private:
        std::string expr;
        int cur_pos;

        std::string getNumber();
        std::string getLetters();
        std::string getOperator();
        std::string getFunction();
    };

    bool isDigit(char c);
    bool isLetter(char c);
    bool isOperator(char c);
    bool isFunction(const std::string& s);

}; // namespace autodiff

#endif // TOKENIZER_HPP