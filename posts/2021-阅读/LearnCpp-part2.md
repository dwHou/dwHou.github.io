#### 11.1 函数重载

##### 函数重载的简介

函数重载允许我们创建多个具有相同名称的函数，只要每个相同名称的函数具有不同的参数类型（或者可以通过其他方式区分函数）。 每个共享名称（在同一范围内）的函数称为<font color="brown">重载函数</font>（有时简称为重载）。

```cpp
int add(int x, int y) // 重载函数，integer version
{
    return x + y;
}

double add(double x, double y) // 重载函数，floating point version
{
    return x + y;
}

int main()
{
  	std::cout << add(1, 2); // 重载解析，calls add(int, int)
    std::cout << '\n';
    std::cout << add(1.2, 3.4); // 重载解析，calls add(double, double)
    return 0;
}
```

上面的程序将正常编译。 尽管您可能期望这些函数会导致命名冲突，但这里的情况并非如此。 由于<font color="blue">这些函数的参数类型不同，编译器能够区分这些函数</font>，并将它们视为恰好共享名称的单独函数。

##### 重载解析的简介

此外，当对已重载的函数进行函数调用时，编译器将尝试<font color=blue>根据函数调用中使用的参数将函数调用与适当的重载相匹配</font>。 这称为重载解析。



> 使其编译
>
> 为了使使用重载函数的程序能够编译，必须满足以下两点：
>
> - 每个重载函数都必须与其他函数区分开来。 我们在第 10.11 课讨论。
> - 对重载函数的每次调用都必须解析为重载函数。 我们在第 10.12 课——函数重载解析和模糊匹配中讨论编译器如何将函数调用与重载函数相匹配。

#### 11.2 函数重载区分

前文我们引入了函数重载的概念，它允许我们创建多个具有相同名称的函数，只要每个同名函数具有不同的参数类型（或者可以通过其他方式区分函数）。

在本节中，我们将仔细研究重载函数是如何区分的。 未正确区分的重载函数将导致编译器发出编译错误。

|                      |                               |                                                              |
| :------------------- | :---------------------------- | :----------------------------------------------------------- |
| Function property    | Used for differentiation      | Notes                                                        |
| Number of parameters | Yes                           |                                                              |
| Type of parameters   | Yes                           | Excludes typedefs, type aliases, and const qualifier on value parameters. Includes ellipses. |
| Return type          | <font color="brown">No</font> |                                                              |

- 基于参数数量的重载

- 根据参数类型重载

  ⚠️不包括 typedef、类型别名和 const 限定符的差异。但包括省略号，它被视为是一种独特的参数类型：

  ```cpp
  typedef int Height; // typedef
  using Age = int; // type alias
  
  void print(int value);
  void print(Age value); // not differentiated from print(int)
  void print(Height value); // not differentiated from print(int)
  
  void print(int);
  void print(const int); // not differentiated from print(int)
  
  void foo(int x, int y);
  void foo(int x, ...); // differentiated from foo(int, int)
  ```

- 不能根据返回类型区分

  这是合理的。 如果你是编译器，你看到了这样的语句：

  ```cpp
  int getRandomValue();
  double getRandomValue();
  
  getRandomValue(); // 你会调用这两个重载函数中的哪一个？ 没法区分。这背后是语法设计之道。
  ```

- 类型签名

  函数的类型签名（通常称为签名）被定义为函数头中用于区分函数的部分。 在 C++ 中，这包括函数名称、参数数量、参数类型和函数级限定符（成员函数的）。 而它不包括返回类型。

> 作为旁白：
>
> 当编译器编译函数时，它会执行<font color="brown">名称重整</font>，这意味着函数的编译名称会根据各种标准（例如参数的数量和类型）进行更改（“重整”），以便链接时具有唯一的名称。
>
> 例如，某些原型 int fcn() 的函数可能会编译为名称 __fcn_v，而 int fcn(int) 可能会编译为名称 __fcn_i。 因此，虽然在源代码中，两个重载函数共享一个名称，但在编译代码中，这些名称实际上是唯一的。
>
> 对于如何重整名称没有标准化，因此不同的编译器会产生不同的重整名称。

#### 11.3 函数重载解析和不明确的匹配

在上一课（11.2——函数重载区分）中，我们讨论了函数的哪些属性用于区分重载函数。 如果重载函数无法与同名的其他重载正确区分，则编译器将发出编译错误。

然而，拥有一组差异化的重载函数只是问题的一半。 当进行任何函数调用时，编译器还必须确保可以找到匹配的函数声明。

对于非重载函数（具有唯一名称的函数），只有一个函数可能与函数调用匹配。 该函数要么匹配（或者可以在应用类型转换后匹配），要么不匹配（并导致编译错误）。 对于重载函数，可能有许多函数可能与函数调用匹配。 由于函数调用<font color="brown">只能解析其中一个</font>，因此编译器必须确定哪个重载函数是最佳匹配。 **将函数调用与特定重载函数相匹配的过程称为<font color="brown">重载解析</font>**。

- 在函数实数的类型和函数形参的类型完全匹配的简单情况下，这（通常）很简单

- 实参形参类型不匹配时的调用时，编译器将逐步执行一系列规则来确定哪个重载函数（如果有）是最佳匹配。

  在每个步骤中，编译器都会对函数调用中的参数应用一堆不同的类型转换。 

  1. 步骤 1) 编译器尝试找到<font color="blue">完全匹配</font>。 这分两个阶段发生。 首先，编译器将查看是否存在重载函数，其中函数调用中的参数类型与重载函数中的参数类型完全匹配。

     其次，编译器将对函数调用中的参数应用许多<font color="blue">简单转换</font>。 简单转换是一组特定的转换规则，它们将修改类型（而不修改值）以查找匹配项。 例如，非常量类型（non-const）可以简单地转换为常量（const）类型：

  2. 步骤 2) 如果未找到完全匹配，编译器会尝试通过对参数应用<font color="blue">数字提升</font>来查找匹配。如果在数字提升之后找到匹配项，则函数调用将得到解决。

  3. 步骤 3) 如果通过数字提升未找到匹配项，编译器会尝试通过对参数应用<font color="blue">数字转换</font>（10.3 -- 数字转换）来查找匹配项。

     >通过应用数字升级进行的匹配优先于通过应用数字转换进行的任何匹配。

  4. 步骤 4) 如果通过数值转换未找到匹配项，编译器将尝试通过任何<font color="blue">用户定义的转换</font>（以后会介绍）来查找匹配项。 

  5. 步骤 5) 如果仍未找到匹配项，编译器将查找使用<font color="blue">省略号</font>的匹配函数。

  6. 步骤 6) 如果仍未找到匹配项，编译器将放弃并报出编译错误。

  对于应用的每个转换，编译器都会检查任何重载函数现在是否匹配。 应用所有不同的类型转换并检查匹配后，该步骤就完成了。 结果将是以下三种可能结果之一：

  1. 没有找到匹配的函数。 编译器按顺序移动到下一<font color="brown">步骤</font>。

     > 如果编译器到达整个序列的末尾而没有找到匹配项，则会生成编译错误，指出无法为该函数调用找到匹配的重载函数。

  2. 找到了一个匹配的函数。 该函数被认为是最佳匹配。 至此匹配过程完成，后续步骤不再执行。

  3. 找到多个匹配函数。 编译器将发出<font color="brown">不明确的匹配</font>编译错误。 我们稍后会进一步讨论这个案例。

- 不明确的匹配

  意味着给定步骤中的任何匹配都不会被认为比同一步骤中的任何其他匹配更好。如：

  ```cpp
  void print(int)
  {
  }
  void print(double)
  {
  }
  int main()
  {
      print(5L); // 5L is type long
  /* 这个调用会报错，重载函数的不明确的匹配。
  因为这个例子会走到 步骤 3)，然后long可以数字转换为int或double，二者并不存在谁更优先。
  */
      return 0;
  }
  
  void print(unsigned int)
  {
  }
  void print(float)
  {
  }
  int main()
  {
      print(0); // int can be numerically converted to unsigned int or to float
      print(3.14159); // double can be numerically converted to unsigned int or to float
   /*尽管您可能期望 0 解析为 print(unsigned int) 且 3.14159 解析为 print(float)，但这两个调用都会导致不明确的匹配。 int 值 0 可以数字转换为无符号 int 或浮点型，因此任一重载都同等地匹配，结果是不明确的函数调用。
   */
      return 0;
  }
  ```

  > 可以看出来，步骤 3) 数字转换，由于本身很少存在优先级，所以这里很容易被程序员误判，出现不明确的匹配。
  >
  > 遇到不明确匹配的几种解决方案：
  >
  > ① 定义新的重载函数，它的参数恰好是调用用到的。
  >
  > ②  调用时对实参进行静态强制转换（static_cast）

- 匹配多个参数的函数

  所选择的函数必须为至少一个参数提供比所有其他候选函数更好的匹配，并且对于所有其他参数不更差。

  如果找到这样的函数，那么它显然是最好的选择。 如果找不到这样的函数，则调用将被视为不明确（或不匹配）。

  ```cpp
  #include <iostream>
  
  void print(char, int)
  {
  	std::cout << 'a' << '\n';
  }
  
  void print(char, double)
  {
  	std::cout << 'b' << '\n';
  }
  
  void print(char, float)
  {
  	std::cout << 'c' << '\n';
  }
  
  int main()
  {
  	print('x', 'a');
    // all functions match the first argument exactly. However, the top function matches the second parameter via promotion, whereas the other functions require a conversion. Therefore, print(char, int) is unambiguously the best match.
  	return 0;
  }
  ```

#### 11.4 删除函数

```cpp
#include <iostream>

void printInt(int x)
{
    std::cout << x << '\n';
}

int main()
{
    printInt(5);    // okay: prints 5
    printInt('a');  // 数字提升 prints 97 -- does this make sense?
    printInt(true); // 数字提升 print 1 -- does this make sense?

    return 0;
}
```

假设我们认为使用 char 或 bool 类型的值调用 printInt() 没有意义。 我们可以做什么？

##### 使用=delete说明符删除函数

如果我们明确不想调用一个函数，我们可以使用 = delete 说明符将该函数定义为已删除。 如果编译器将函数调用与已删除的函数相匹配，则编译将因编译错误而停止。

使用此语法更新代码：

```cpp
#include <iostream>

void printInt(int x)
{
    std::cout << x << '\n';
}

void printInt(char) = delete; // calls to this function will halt compilation
void printInt(bool) = delete; // calls to this function will halt compilation

int main()
{
    printInt(97);   // okay

    printInt('a');  // compile error: function deleted
    printInt(true); // compile error: function deleted

    printInt(5.0);  // compile error: ambiguous match

    return 0;
}
/*
让我们快速浏览一下其中的一些内容。 首先，printInt('a') 与 printInt(char) 直接匹配，后者被删除。 编译器因此产生编译错误。 printInt(true) 与 printInt(bool) 直接匹配，后者被删除，因此也会产生编译错误。

printInt(5.0) 是一个有趣的例子，可能会产生意想不到的结果。 首先，编译器检查是否存在完全匹配的 printInt(double)。 不存在。 接下来，编译器尝试找到最佳匹配。 尽管 printInt(int) 是唯一未删除的函数，但已删除的函数仍被视为函数重载决策中的候选函数。 由于这些函数都不是明确的最佳匹配，因此编译器将发出模糊匹配编译错误。
*/
```

>= delete 的意思是“我禁止这个”，而不是“这个不存在”。

##### 删除所有不匹配的重载

删除一堆单独的函数重载是能行的，但可能很冗长。 我们可以通过使用函数模板（在即将到来的第 11.6 课——函数模板 中介绍）来完成此操作：

```cpp
#include <iostream>

// This function will take precedence for arguments of type int
void printInt(int x)
{
    std::cout << x << '\n';
}

// This function template will take precedence for arguments of other types
// Since this function template is deleted, calls to it will halt compilation
template <typename T>
void printInt(T x) = delete;

int main()
{
    printInt(97);   // okay
    printInt('a');  // compile error
    printInt(true); // compile error

    return 0;
}
```

#### 11.5 默认参数

默认参数是为函数参数提供的默认值。

```cpp
// 请注意，必须使用等号来指定默认参数。使用括号或大括号初始化不起作用
#include <iostream>

void print(int x, int y=4) // 4 is the default argument
{
    std::cout << "x: " << x << '\n';
    std::cout << "y: " << y << '\n';
}

int main()
{
    print(1, 2); // y will use user-supplied argument 2
    print(3); // y will use default argument 4, as if we had called print(3, 4)

    return 0;
}
```

> 也许令人惊讶的是，默认参数由编译器在调用点处理。 在上面的例子中，当编译器看到 print(3) 时，它会将这个函数调用重写为 print(3, 4)，以便参数的数量与参数的数量相匹配。 重写后的函数调用将像往常一样工作。

##### 何时使用默认参数

当函数需要一个具有合理默认值的值，但您希望让调用者根据需要进行覆盖时，默认参数是一个很好的选择。

>作者注：
>
>由于用户可以选择是否提供特定的参数值或使用默认值，因此提供默认值的参数有时称为可选参数。 但是，术语“可选参数”也用于指代其他几种类型的参数（包括通过地址传递的参数和使用 std::Optional 的参数），因此我们建议避免使用该术语。

##### 多个默认参数

一个函数可以有多个带有默认参数的参数：

```cpp
#include <iostream>

void print(int x=10, int y=20, int z=30)
{
    std::cout << "Values: " << x << " " << y << " " << z << '\n';
}

int main()
{
    print(1, 2, 3); // all explicit arguments
    print(1, 2); // rightmost argument defaulted
    print(1); // two rightmost arguments defaulted
    print(); // all arguments defaulted

    return 0;
}
```

C++不支持例如 print(,,3)的函数调用语法（ x 和 y 用默认参数，而为 z 提供显式值）。这有两个主要后果 ：

1. 如果为参数指定了默认参数，则所有后续参数（右侧）也必须指定为默认参数。
2. 如果多个参数具有默认参数，则越靠左边的参数应该是越有可能由用户显式设置的参数。

##### 默认参数不能重新声明

一旦声明，默认参数就不能重新声明（在同一文件中）。 这意味着对于具有前向声明和函数定义的函数，默认参数可以在前向声明或函数定义中声明，但不能同时在两者中声明。

> 最佳实践是在前向声明中而不是在函数定义中声明默认参数，因为前向声明更有可能被其他文件看到（特别是在头文件中）。

##### 默认参数和函数重载

具有默认值的参数既有可能区分函数重载，但也可能会导致潜在的不明确的函数调用。 两个例子：

```cpp
#include <string>

void print(std::string)
{
}

void print(char=' ')
{
}

int main()
{
    print("Hello, world"); // resolves to print(std::string)
    print('a'); // resolves to print(char)
    print(); // resolves to print(char)

    return 0;
}
```

```cpp
void print(int x);
void print(int x, int y = 10);
void print(int x, double y = 20.5);
print(1, 2); // will resolve to print(int, int)
print(1, 2.5); // will resolve to print(int, double)
print(1); // ambiguous function call
```

总结：默认参数提供了一种有用的机制来指定用户可能想要或不想覆盖的参数值。 它们在 C++ 中经常使用，您将经常看到它们。

#### 11.6 函数模板

考虑为下面的函数支持int, double, long, long double 甚至是您自己创建的新类型等。且实现代码与 int 版本 max 的实现代码完全相同：

```cpp
int max(int x, int y)
{
    return (x < y) ? y : x;
    // Note: we use < instead of > because std::max uses <
}
```

必须为我们想要支持的每组参数类型创建具有相同实现的重载函数，这样维护起来令人头疼，容易出错，并且明显违反了 DRY（don’t repeat yourself）原则。这里还有一个不太明显的挑战：使用 max 函数的程序员可能希望使用 max 函数的作者没有预料到的参数类型来调用它（因此作者没有为其编写重载函数）。

我们真正缺少的是某种编写 max <font color="brown">单一版本</font>的方法，<font color="brown">它可以处理任何类型的参数</font>（甚至是编写 max 代码时可能没有预料到的类型）。 普通功能根本无法胜任这里的任务。 幸运的是，C++ 支持另一个专门为解决此类问题而设计的功能。

欢迎来到 <font color="brown">**C++ 模板**</font>的世界。

##### C++模板的介绍

在 C++ 中，模板系统旨在简化创建能够使用不同数据类型的函数（或类）的过程。

我们不是手动创建一堆几乎相同的函数或类（每组不同类型一个），而是创建一个模板。 就像普通的定义一样，模板描述了函数或类的样子。 与普通定义（必须指定所有类型）不同，在模板中我们可以使用一种或多种占位符（placeholder）类型。 <font color="brown">占位符类型</font>表示在编写模板时未知的某种类型，但稍后将提供。

<font color="blue">一旦定义了模板，编译器就可以使用模板根据需要生成任意数量的重载函数（或类），每个重载函数（或类）使用不同的实际类型！</font>

最终结果是相同的——我们最终得到了一堆几乎相同的函数或类（每组不同类型一个）。 但我们只需要创建和维护一个模板，编译器就会为我们完成所有艰难的工作。

>关键见解：编译器可以使用单个模板来生成一系列相关的函数或类，每个函数或类使用一组不同的类型。

小讲堂：由于实际类型直到在程序中使用模板时（而不是在编写模板时）才确定，因此模板的作者不必尝试预测可能使用的所有实际类型。 这意味着模板代码可以与编写模板时甚至不存在的类型一起使用！ 稍后当我们开始探索 <font color="brown">C++ 标准库</font>时，我们将看到它如何派上用场，其中绝对<font color="brown">充满了模板代码</font>！

>关键见解：模板可以使用在编写模板时甚至不存在的类型。 这有助于使模板代码既灵活又面向未来！

##### 函数模板

函数模板是一种类函数的定义，用于生成一个或多个重载函数，每个重载函数具有一组不同的实际类型。 这将使我们能够创建可以与许多不同类型一起使用的函数。

创建函数模板时，我们对任何参数类型、返回类型或稍后要指定的函数体中使用的类型使用占位符类型，也称为类型模板参数（type template parameters）或模板类型（template types）。

函数模板最好通过示例来教授，因此让我们将上面示例中的普通 max(int, int) 函数转换为函数模板。 这非常简单，我们将解释一路上发生的事情。

