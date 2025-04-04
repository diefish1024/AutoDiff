自动微分技术在诸如Torch和TensorFlow等框架中的应用，极大地便利了人们基于反向传播的深度学习算法的实现和训练。现在，我们希望实现一个针对代数表达式的自动微分程序。

### 输入格式

首先，输入一个由以下符号组成的中缀表达式。

#### 运算符

| 类型        | 示例 | 备注                             |
| ----------- | ---- | -------------------------------- |
| 括号        | ( )  |                                  |
| 幂运算      | ^    |                                  |
| 乘法 & 除法 | * /  |                                  |
| 加法 & 减法 | + -  |                                  |
| 参数分隔符  | ,    | 可选，仅用于多元函数中的参数分隔 |

上述运算符按优先级从高到低排列。例如，`a+b^c*d` 将被视为与 `a + ( (b ^ c) * d )` 相同。

#### 数学函数（加分项）

| 函数                 | 描述                                                         |
| -------------------- | ------------------------------------------------------------ |
| ln(A) log(A, B)      | 对数函数。`ln(A)` 表示A的自然对数，`log(A, B)` 表示以A为底的B的对数。 |
| cos(A) sin(A) tan(A) | 基本三角函数。                                               |
| pow(A, B) exp(A)     | 指数函数。`pow(A, B)` 表示A的B次方，`exp(A)` 表示A的自然指数。 |

#### 操作数

| 类型     | 示例            | 备注                                                         |
| -------- | --------------- | ------------------------------------------------------------ |
| 字面常量 | 2 3 0 -5        | 仅考虑由纯数字和负号组成的整数。                             |
| 变量     | ex cosval xy xx | 考虑到上述“数学函数”为保留字，非保留字的标识符（小写英文字母组成的字符串）称为变量。 |

### 输出方式

对于表达式中出现的每个变量（如上定义），使用输入形式中定义的运算符、数学函数和操作数，描述一个表示输入代数表达式对该变量的导数的算术表达式。

输出按变量的字典序排列。每行打印两个字符串，分别是每个变量及其对应的导数函数。两个字符串之间用 `:` 分隔。

### 要求

处理给定的包含运算符和操作数的代数表达式，并输出原表达式对每个变量的导数函数的代数表达式。

只需确保输出表达式与正确的导数函数相同，无需对表达式进行简化或展开。

使用C/C++实现，仅可使用标准库。

代码具有高可读性或注释充分。请确保源文件能以utf-8编码正确打开。

### 加分项

— 不使用STL中提供的复杂容器  
— 支持包含数学函数的表达式  
— 通过应用至少两条规则简化代数表达式以减少结果的长度。

### 提示

从中缀表达式构建表达式树。

#### 简单输入案例：

输入：

```
a+b^c*d
```

输出：

```
a: 1
b: c*b^(c-1)*d
c: ln(b)*b^c*d
d: b^c
```

输入：

```
a*10*b+2^a/a
```

输出：

```
a: 10*b-2^a/a^2+2^a*ln(2)/a
b: a*10
```

输入：

```
xx^2/xy*xy+a^a （注意xx和xy分别被视为单个变量）
```

输出：

```
a: a^a*(1+ln(a))
xx: 2*xx
xy: 0
```

#### 更多输入案例

— `x*ln(y)`  
— `x*ln(x*y)+y*cos(x)+y*sin(2*x)`  
— `log(a,b)/log(c,a)`

### 评分标准：

第1章：引言（6分）  
第2章：算法规范（12分）  
第3章：测试结果（20分）  
第4章：分析与评论（10分）  
编写程序（50分），需有充分的注释。  
文档的整体风格（2分）  

注：任何在解决加分问题上表现出色的人将获得额外5%的分数。