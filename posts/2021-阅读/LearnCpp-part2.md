## 函数重载和函数模板

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

```cpp
int max(int x, int y)
{
    return (x < y) ? y : x;
}
```

##### 创建模板化 max 函数

1. 请注意，我们在此函数中使用了三次 int 类型：一次用于参数 x，一次用于参数 y，一次用于函数的返回类型。要创建函数模板，我们要做两件事。 首先，我们将用类型模板参数替换特定类型。 在这种情况下，因为我们只有一种需要替换的类型（int），所以我们只需要一个类型模板参数（我们将其称为 T）。这是我们使用单一模板类型的新函数：

   ```cpp
   T max(T x, T y) // won't compile because we haven't defined T
   {
       return (x < y) ? y : x;
   }
   ```

   这是一个好的开始——但是，它不会编译，因为编译器不知道 T 是什么！ 而且这仍然是一个普通函数，而不是函数模板。

2. 我们将告诉编译器这是一个函数模板，并且 T 是一个类型模板参数，它是任何类型的占位符。 这是使用所谓的<font color="brown">模板参数声明</font>来完成的。 模板参数声明的范围仅限于后面的函数模板（或类模板）。 因此，每个函数模板（或类）都需要有自己的模板参数声明。

   ```cpp
   template <typename T> // this is the template parameter declaration
   T max(T x, T y) // this is the function template definition for max<T>
   {
       return (x < y) ? y : x;
   }
   ```

   在模板参数声明中，我们从**关键字 template** 开始，它告诉编译器我们正在创建一个模板。 接下来，我们在尖括号 (<>) 内指定模板将使用的所有模板参数。 对于每个类型模板参数，我们使用关键字 typename 或 class，后跟类型模板参数的名称（例如 T）。

> 1. 我们在第 11.8 课中讨论如何创建具有多种模板类型的函数模板
> 2. 在这种情况下，typename 和 class 关键字没有区别。 您经常会看到人们使用 class 关键字，因为它是较早引入到语言中的。 但是，我们更喜欢较新的 typename 关键字，因为它更清楚地表明类型模板参数可以替换为任何类型（例如基本类型），而不仅仅是类类型。

不管你信不信，我们已经完成了！ 我们创建了 max 函数的模板版本，它现在可以接受不同类型的参数。

因为该函数模板有一个名为 T 的模板类型，所以我们将其称为 max<T>。 在下一课中，我们将了解如何使用 max<T> 函数模板生成一个或多个具有不同类型参数的 max() 函数。

##### 命名模板参数

略 ： ① 简单情况约定用T ② 如果类型模板参数具有某些要求，可以用（T前缀+）大写字母开头的名字

#### 11.7 函数模板实例化

```cpp
template <typename T>
T max(T x, T y)
{
    return (x < y) ? y : x;
}
```

本节，我们将重点介绍如何使用函数模板。

##### 使用函数模板

函数模板实际上并不是函数——它们的代码不会直接编译或执行。 相反，函数模板只有一项工作：生成函数（被编译和执行）。

要使用 max<T> 函数模板，我们可以使用以下语法进行函数调用：

```cpp
max<actual_type>(arg1, arg2); // actual_type is some actual type, like int or double
```

这看起来很像普通的函数调用——主要区别是在尖括号中添加了类型（称为模板实参），它指定将用于代替模板类型 T 的实际类型。

```cpp
#include <iostream>

template <typename T>
T max(T x, T y)
{
    return (x < y) ? y : x;
}

int main()
{
    std::cout << max<int>(1, 2) << '\n';    // instantiates and calls function max<int>(int, int)
    std::cout << max<int>(4, 3) << '\n';    // calls already instantiated function max<int>(int, int)
    std::cout << max<double>(1, 2) << '\n'; // instantiates and calls function max<double>(double, double)
    return 0;
}
/*
当编译器遇到函数调用 max<int>(1, 2) 时，它将确定 max<int>(int, int) 的函数定义尚不存在。 因此，编译器将使用我们的 max<T> 函数模板来创建一个。
从函数模板（具有模板类型）创建函数（具有特定类型）的过程称为函数模板实例化（或简称实例化）。 当这个过程由于函数调用而发生时，称为隐式实例化。 实例化的函数通常称为函数实例（简称实例）或模板函数。 函数实例在所有方面都是普通函数。
*/
```

上述程序里，我们的函数模板将用于生成两个函数：一次将 T 替换为 int，另一次将 T 替换为 double。 所有实例化之后，程序将如下所示：

> 将隐式实例化写为显式形式

```cpp
#include <iostream>

// a declaration for our function template (we don't need the definition any more)
template <typename T>
T max(T x, T y);

template<>
int max<int>(int x, int y) // the generated function max<int>(int, int)
{
    return (x < y) ? y : x;
}

template<>
double max<double>(double x, double y) // the generated function max<double>(double, double)
{
    return (x < y) ? y : x;
}

int main()
{
    std::cout << max<int>(1, 2) << '\n';    // instantiates and calls function max<int>(int, int)
    std::cout << max<int>(4, 3) << '\n';    // calls already instantiated function max<int>(int, int)
    std::cout << max<double>(1, 2) << '\n'; // instantiates and calls function max<double>(double, double)
  /*
  注：当我们实例化 max<double> 时，实例化的函数具有 double 类型的形参。 因为我们提供了 int 实参，所以这些参数将隐式转换为 double。
  */
    return 0;
}
```

##### 模板参数推导

如果参数的类型与我们想要的实际类型匹配，我们不需要指定实际类型 - 相反，我们可以使用模板参数推导来让编译器在函数调用中<font color="brown">从参数类型中推导应该使用的实际类型</font>。

例如，不用进行这样的函数调用：

```cpp
std::cout << max<int>(1, 2) << '\n'; // specifying we want to call max<int>
```

我们可以改为执行以下操作之一：

```cpp
std::cout << max<>(1, 2) << '\n';
std::cout << max(1, 2) << '\n';

/*
两种情况之间的差异与编译器如何从一组重载函数解析函数调用有关。 在最上面的情况下（带有空尖括号），编译器在确定要调用哪个重载函数时将仅考虑 max<int> 模板函数重载。 在最下面的情况下（没有尖括号），编译器将同时考虑 max<int> 模板函数重载和 max 非模板函数重载。
*/
```

请注意底例（`std::cout << max(1, 2) << '\n';`）中的<font color="brown">语法看起来与普通函数调用相同！</font> <font color="blue">最佳实践：</font> 在大多数情况下，我们将使用这种正常的函数调用语法来调用从函数模板实例化的函数。

原因如下：

- 语法更加简洁。

- 我们很少会同时拥有匹配的非模板函数和函数模板。

- 如果我们确实有一个匹配的非模板函数和一个匹配的函数模板，我们通常会更喜欢调用非模板函数。

  > 最后一点可能并不明显。 函数模板具有适用于多种类型的实现——但因此，它必须是通用的。 非模板函数仅处理特定的类型组合。 它可以有一个比函数模板版本更优化或更专门针对这些特定类型的实现。 

##### 具有非模板参数的函数模板

```cpp
// T is a type template parameter
// double is a non-template parameter
template <typename T>
int someFcn (T, double)
{
    return 5;
}

int main()
{
    someFcn(1, 3.4); // matches someFcn(int, double)
    someFcn(1, 3.4f); // matches someFcn(int, double) -- the float is promoted to a double
    someFcn(1.2, 3.4); // matches someFcn(double, double)
    someFcn(1.2f, 3.4); // matches someFcn(float, double)
    someFcn(1.2f, 3.4f); // matches someFcn(float, double) -- the float is promoted to a double

    return 0;
}
```

##### 实例化函数可能并不总是可以编译

##### 实例化函数可能并不总是在语义上有意义

```cpp
#include <iostream>
#include <string>

template <typename T>
T addOne(T x);
{
    return x + 1;
}

int main()
{
    std::string hello{ "Hello, world!" };
    std::cout << addOne(hello) << '\n'; // 编译失败，x是std::string时，没法进行x+1
    std::cout << addOne("Hello, world!") << '\n'; // 虽然C++语法上允许将整数值添加到字符串字面量，但语义上没有意义
    return 0;
}
```

我们可以告诉编译器不允许使用某些参数实例化函数模板。 这是通过使用函数模板专门化来完成的，用到了 <font color="brown">`= delete`</font> 来删除函数。

```cpp
// Use function template specialization to tell the compiler that addOne(const char*) should emit a compilation error
template <>
const char* addOne(const char* x) = delete;

int main()
{
    std::cout << addOne("Hello, world!") << '\n'; // compile error
    return 0;
}
```

##### 在多文件中使用函数模板

考虑以下程序，该程序<font color="brown">无法正常工作</font>：

```cpp
//代码位于文件main.cpp
#include <iostream>

template <typename T>
T addOne(T x); // function template forward declaration

int main()
{
    std::cout << addOne(1) << '\n';
    std::cout << addOne(2.3) << '\n';

    return 0;
}
```

```cpp
//代码位于文件add.cpp
template <typename T>
T addOne(T x) // function template definition
{
    return x + 1;
}
```

如果 addOne 是非模板函数，则此程序可以正常工作：在 main.cpp 中，编译器会对 addOne 的前向声明感到满意，并且链接器会将 main.cpp 中对 addOne() 的调用连接到该函数 定义在add.cpp中。

但是因为 addOne 是一个模板，所以这个程序不起作用，我们得到一个链接器错误`...error LNK2019: unresolved external symbol "int __cdecl addOne<int>(int) ...`

**这里编译器的行为是：**

> 在 main.cpp 中，我们调用 addOne<int> 和 addOne<double>。 但是，由于编译器看不到函数模板 addOne 的定义，因此无法在 main.cpp 中实例化这些函数。 不过，它确实看到了 addOne 的前向声明，并且会假设这些函数存在于其他地方，并将在稍后链接。
>
> 当编译器去编译add.cpp时，它会看到函数模板addOne的定义。 但是，add.cpp 中没有使用此模板，因此编译器不会实例化任何内容。 最终结果是链接器无法将对 main.cpp 中的 addOne<int> 和 addOne<double> 的调用连接到实际函数，因为这些函数从未实例化。

<font color="blue">最佳实践：</font>解决此问题的最传统方法是将<font color="brown">所有模板代码放入头文件 (.h)</font>，而不是源文件 (.cpp)：

```cpp
//代码位于文件add.h
//在main.cpp里再 #include "add.h"
#ifndef ADD_H
#define ADD_H

template <typename T>
T addOne(T x) // function template definition
{
    return x + 1;
}

#endif
```

您可能想知道为什么这不会导致违反单一定义规则 (one-definition rule)。 ODR 规定类型、模板、内联函数和内联变量允许在不同文件中具有相同的定义。 因此，如果将模板定义复制到多个文件中（只要每个定义相同），就没有问题。

但是实例化函数本身又如何呢？ 如果一个函数在多个文件中实例化，如何不导致违反 ODR？ 答案是从模板隐式实例化的函数是隐式内联的。 如您所知，内联函数可以在多个文件中定义，只要每个文件中的定义相同即可。

>关键见解：
>
>模板定义不受单一定义规则的约束，该规则要求每个程序只需要一个定义，因此将相同的模板定义#included 到多个源文件中不是问题。 从函数模板隐式实例化的函数是隐式内联的，因此它们可以在多个文件中定义，只要每个定义都是相同的。
>
>模板本身不是内联的，因为内联的概念仅适用于变量和函数。

##### 泛型编程

由于模板类型可以替换为任何实际类型，因此模板类型有时称为泛型类型。 由于模板的编写可以与特定类型无关，因此使用模板进行编程有时称为泛型编程。 C++ 通常非常关注类型和类型检查，相比之下，泛型编程让我们专注于算法逻辑和数据结构设计，而不必过多担心类型信息。

##### 总结

一旦习惯了编写函数模板，您就会发现它们实际上并不比编写具有实际类型的函数花费更长的时间。 函数模板可以通过最大限度地减少需要编写和维护的代码量来显着减少代码维护和错误。

函数模板确实有一些缺点，如果我们不提及它们，那就太失职了。 首先，编译器将为每个函数调用创建（并编译）一个具有唯一参数类型集的函数。 因此，虽然函数模板编写起来很紧凑，但它们可能会扩展为大量代码，从而导致代码膨胀和编译时间变慢。 函数模板的更大缺点是它们往往会产生看起来疯狂的、几乎无法阅读的错误消息，这些错误消息比常规函数更难破译。 这些错误消息可能非常令人生畏，但是一旦您了解了它们想要告诉您的内容，它们所指出的问题通常就很容易解决。

与模板为编程工具包带来的强大功能和安全性相比，这些缺点相当小，因此在需要类型灵活性的任何地方都可以自由使用模板！ 一个好的经验法则是首先创建普通函数，然后如果您发现需要不同参数类型的重载，则将它们转换为函数模板。

#### 11.8 具有多种模板类型的函数模板

下面的程序会编译失败，

```cpp
#include <iostream>

template <typename T>
T max(T x, T y)
{
    return (x < y) ? y : x;
}

int main()
{
    std::cout << max(2, 3.5) << '\n';  // compile error

    return 0;
}
```

在函数调用 max(2, 3.5) 中，我们传递两种不同类型的参数：一种 int 和一种 double。 因为我们在不使用尖括号来指定实际类型的情况下进行函数调用，所以编译器将首先查看 max(int, double) 是否存在非模板匹配。 它不会找到一个。

接下来，编译器将查看是否可以找到函数模板匹配（使用模板参数推导）。 然而，这也会失败，原因很简单：T 只能代表单一类型。 T 没有任何类型允许编译器将函数模板 max<T>(T, T) 实例化为具有两种不同参数类型的函数。 换句话说，由于函数模板中的两个参数都是 T 类型，因此它们必须解析为相同的实际类型。

由于未找到非模板匹配，并且未找到模板匹配，因此函数调用无法解析，并且我们收到编译错误。

您可能想知道为什么编译器不生成函数 max<double>(double, double)，然后使用数值转换将 int 参数类型转换为 double。 答案很简单：<font color="brown">类型转换仅在解决函数重载时完成，而不是在执行模板参数推导时完成。</font>

> 这种类型转换的缺乏是有意为之的，至少有两个原因。 
>
> 首先，它有助于使事情变得简单：我们要么找到函数调用参数和模板类型参数之间的精确匹配，要么找不到。 其次，它允许我们设计需要确保两个或多个参数具有相同类型的函数模板。

我们必须找到另一个解决方案。 幸运的是，我们可以通过（至少）三种方式解决这个问题。

##### 1. 使用 static_cast 将实参转换为匹配类型

第一个解决方案是让调用者承担将参数转换为匹配类型的负担。 例如：

```cpp
std::cout << max(static_cast<double>(2), 3.5) << '\n'; // convert our int to a double so we can call max(double, double)
```

然而，这个解决方案很笨拙并且难以阅读。

##### 2.提供显式类型模板参数

幸运的是，如果我们指定要使用的显式类型模板参数，则不必使用模板参数推导：

```cpp
// we've explicitly specified type double, so the compiler won't use template argument deduction
std::cout << max<double>(2, 3.5) << '\n';
```

在上面的例子中，我们调用 max<double>(2, 3.5)。 因为我们已经明确指定 T 应替换为 double，所以编译器不会使用模板参数推导。 相反，它只会实例化函数 max<double>(double, double)，然后对任何不匹配的参数进行类型转换。 我们的 int 参数将隐式转换为 double。

虽然这比使用 static_cast 更具可读性，但如果我们在对 max 进行函数调用时根本不需要考虑类型，那就更好了。

##### 3.具有多个模板类型参数的函数模板

问题的根源（root cause）在于我们只为函数模板定义了单一模板类型 (T)，然后指定两个参数必须是同一类型。

解决这个问题的最好方法是重写我们的函数模板，使我们的参数可以解析为不同的类型。 我们现在将使用两个（T 和 U），而不是使用一个模板类型参数 T：

```cpp
#include <iostream>

template <typename T, typename U> // We're using two template type parameters named T and U
auto max(T x, U y) // x can resolve to type T, and y can resolve to type U
{
    return (x < y) ? y : x; // uh oh, we have a narrowing conversion problem here
}

int main()
{
    std::cout << max(2, 3.5) << '\n';
    return 0;
}
/*
小tips：为了防止函数的返回经历数字变窄，我们使用了自动返回类型——我们将让编译器从 return 语句中推断出返回类型应该是什么。 
*/
```

##### 缩写函数模板 `C++20`

C++20引入了auto关键字的新用法：当auto关键字在普通函数中用作参数类型时，编译器会自动将函数转换为函数模板，每个auto参数成为独立的模板类型参数。 这种创建函数模板的方法称为缩写函数模板。

```cpp
auto max(auto x, auto y)
{
    return (x < y) ? y : x;
}
```

是 C++20 中以下内容的简写：

```cpp
template <typename T, typename U>
auto max(T x, U y)
{
    return (x < y) ? y : x;
}
```

如果您希望每个模板类型参数都是独立类型，可以首选此形式，因为它更加简洁和可读。但当您希望多个自动参数为同一类型时，则没有一个简单的缩写函数模板可以实现这样的功能。

#### 11.9 非类型模板参数

在前面的课程中，我们讨论了如何创建使用类型模板参数的函数模板。 类型模板参数充当作为模板实参传入的实际类型的占位符。

虽然类型模板参数是迄今为止最常用的模板参数类型，但还有另一种值得了解的模板参数：非类型模板参数。

##### 非类型模板参数

 ～是constexpr 值的占位符。例子如下：

```cpp
#include <iostream>

template <int N> // declare a non-type template parameter of type int named N
void print()
{
    std::cout << N << '\n'; // use value of N here
}

int main()
{
    print<5>(); // 5 is our non-type template argument

    return 0;
}
/* 该程序会打印：5 */
```

>最佳实践：就像 T 通常用作第一个类型模板参数的名称一样，N 通常用作 int 非类型模板参数的名称。

##### 非类型模版的用途

从 C++20 开始，函数参数不能为 constexpr。 对于普通函数、constexpr 函数（这是有道理的，因为它们必须能够在运行时执行），甚至可能令人惊讶的是 consteval 函数，都是如此。

例如，类类型 std::bitset 使用非类型模板参数来定义要存储的位数，因为位数必须是 constexpr 值。

>作者注：必须使用非类型模板参数来规避函数参数不能为 constexpr 的限制并不好。 有很多不同的提案正在评估中，以帮助解决此类情况。 我预计我们可能会在未来的 C++ 语言标准中看到更好的解决方案。

#### 11.x 第十一章总结

函数模板可能看起来相当复杂，但它们是使代码与不同类型的对象一起工作的非常强大的方法。 我们将在以后的章节中看到更多模板内容，所以请做好准备。

## 复合类型：引用和指针

#### 12.1 复合数据类型简介

在第 4.1 课——基本数据类型简介中，我们介绍了基本数据类型，它们是 C++ 作为核心语言的一部分提供的基本数据类型。

到目前为止，我们在程序中已经大量使用了这些基本类型，尤其是 int 数据类型。 虽然这些基本类型对于简单的使用非常有用，但当我们开始做更复杂的事情时，它们并不能满足我们的全部需求。

##### 复合数据类型

<font color="blue">复合数据类型（有时也称为组合数据类型）是可以从基本数据类型（或其他复合数据类型）构造的数据类型。 每种复合数据类型也有其独特的属性。</font>

我们可以使用复合数据类型来优雅地解决一些基本类型难以解决的挑战。

C++ 支持以下复合类型：

- Functions
- Arrays
- Pointer types:
  - Pointer to object
  - Pointer to function
- Pointer to member types:
  - Pointer to data member
  - Pointer to member function
- Reference types:
  - L-value references
  - R-value references
- Enumerated types:
  - Unscoped enumerations
  - Scoped enumerations
- Class types:
  - Structs
  - Classes
  - Unions

您已经经常使用一种复合类型：函数（functions）。例如，考虑这个函数：

```cpp
void doSomething(int x, double y)
{
}
```

该函数的类型为 `void(int, double)`。 请注意，该类型由基本类型组成，因此是复合类型。 当然，函数也有自己的特殊行为（例如可调用）。

#### 12.2 值类别（左值和右值）

在我们讨论第一个复合类型（左值引用）之前，我们先绕道讨论一下什么是<font color="brown">左值</font>。

前面5.4节，介绍过自增/自减运算符的副作用：一个函数或表达式如果存在超过它生命的影响，则被称为有副作用的。

```cpp
#include <iostream>
int main()
{
    int x { 5 };
    ++x; // This expression statement has the side-effect of incrementing x
    std::cout << x << '\n'; // prints 6
    return 0;
}
```

在上面的程序中，表达式 ++x 递增 x 的值，并且即使在表达式完成计算后该值仍保持更改。

##### 表达式的属性

为了帮助确定表达式应如何计算以及可以在何处使用它们，C++ 中的所有表达式都有两个属性：**类型（a type）和 值类别（a value category）**。

**表达式的类型**

表达式的类型等同于由计算表达式得出的值、对象或函数的类型。







