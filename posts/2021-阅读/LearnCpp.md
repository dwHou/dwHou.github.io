[TOC]

## 介绍与起步学习

#### 0.1 关于本教程

**课程组织：**

本章会介绍C++的方方面面，它是如何诞生的。接下来的第一章会广泛但浅显地学习C++最基础的概念。后面的章节再深入这些概念，或者引入新的概念。

**课程的学习目标：**

学C++这门语言的同时，学习编程的思想和方法。

通过大量清晰、简洁的代码例子，实践所学的概念。避免*magic hand wave*，就是说跳跃到让学生一头雾水。也有一些练习用的程序，可以自己尝试，再对比参考答案。

最重要的，编程得开心。这才是来到一个正确位置的心态。



#### 0.2 关于编程语言(PL)

机器码 → 汇编语言 → 高级语言

高级语言翻译成计算机能运行的机器码，有两种主要方式：编译 & 解释

**编译器：**可以读入源码，产生一个可执行程序。早期的时候，编译器很简单，产生的机器码没有优化、很慢。现在，编译器已经能产出优化得非常好的机器码了，甚至比程序员写的汇编更加好。

**解释器：**不需要编译一个可执行程序。更加灵活，但效率较低，因为每次执行都需要解释一次。

传统的C/C++/Pascal都是编译型的，而一些“脚本”语言比如Perl/JS倾向于解释型，而有的语言比如Java，会混合使用二者。



#### 0.3 关于C和C++

C和Unix的命运息息相关。当初创造C语言是为了重写Unix（原本是汇编语言版的），增加可移植性，让它可以编译在各种机器上。

C++也诞生于贝尔实验室。1998年被标准化，03年进行了一次大的更新，之后有四个年份也进行了大的更新 (**C++11**, C++14, C++17, 和C++20) 。尤其是C++11被视为一个新的baseline版本，之后便是三年一更。

C和C++的设计哲学：相信开发者。

**C++擅长什么：**

```
C++ 在需要高性能和精确控制内存和其他资源的情况下表现出色。以下是一些最有可能用 C++ 编写的常见应用程序类型：
Video games
Real-time systems (e.g. for transportation, manufacturing, etc…)
High-performance financial applications (e.g. high frequency trading)
Graphical applications and simulations
Productivity / office applications
Embedded software
Audio and video processing
Artificial intelligence and neural networks
```

有一位德国人说：AI is a ressource eater before the lord, as we germans say.

https://ai.stackexchange.com/questions/6185/why-does-c-seem-less-widely-used-than-python-in-ai

其中C++主要是用在 Keras, Torch, TensorFlow等框架的底层。



#### 0.4 关于C++开发

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20210824162102685.png" alt="image-20210824162102685" style="zoom:50%;" />



#### 0.5 关于编译器，链接器和库

也就是上面的<font color="red">Step 4~7</font>

**Step 4** 编译源码

做两件事：① 查错（不符合C++语法的），② 翻译为机器码（并保存在目标文件*name.o* 或 *name.obj*）

**Step 5** 链接目标文件和库

在编译器产生一个或多个目标文件之后，链接器做三件事：

① 链接这些目标文件，合并为一个单一的可执行程序

② 链接器还可以链接库文件。库文件指一些为了**复用**而事先打包好的预编译代码的集合。比如标准库（**Standard Library**）。

一般来说，标准库不用你操心，只要你用到了其中任何一部分，链接器将自动连上。

但之后我们会学习链接其他的库，和如何自己制作库。

③ 确保跨文件的倚赖都是正确的。

**更高阶的：**

有些复杂的项目，会使用makefile，这是一种描述如何build程序的文件（比如编译哪个、链接哪个）。

**Step 6 & 7** 测试和调试

所以步骤3/4/5/7都用到了软件：编辑器、编译器、链接器、调试器。有一类把它们集成到一起的软件包叫做**IDE**。



#### 0.6 安装IDE

#### 0.7 编译你的第一个程序

项目是一个容器，包含了产生一个程序所需的所有文件，也有IDE编译和链接的配置，甚至上次打开项目时的状态。**一个项目对应一个程序。**

但项目文件一般是针对<font color="red">特定IDE</font>的。所以用另一个IDE时要重新创建。

本教程里都属于控制台程序，就是可以从操作系统的控制台运行的。没有用户交互界面(GUI)。

默认情况下，许多IDE会在新建项目时，也将项目加入到一个工作区或解决方案中（"workplace" or "solution"）。

工作区或解决方案是一种容器，包含一个或多个相关的项目。比如一款游戏，如果分为单人版和多人版两个程序，它们应该作为同一个工作区的两个项目。

但这个教程里，我们还是基本建议为每个程序建立新的工作区。减少不必要的程序出错的可能。



#### 0.8 C++常见的一点问题

#### 0.9 编译的配置

build configuration（配置）是指一个项目的设置的集合，它决定了IDE如何生成你的项目。

比如包括：可执行文件叫什么？什么路径去找库文件或其他代码？调试信息是否保留？编译器要多大力气优化你的程序？等等

一般你可以用默认的，除非有特定的需求去修改它。

新建项目时，IDE会建立两种配置：**debug配置**和**release配置**。

前者会关闭所有优化，但保留所有调试信息，所以很大、很慢，但有助于调试。前者会被设置为默认配置。

**Xcode**

Choose **Product** -> **Scheme** -> **Edit Scheme**. Change the `Build Configuration` under the `Info` tab.



#### 0.10 编译的扩展

编译器的扩展**compiler extensions**. 指的是标准的规则之外，每个编译器的特别行为(compiler-specific behaviors)。

这部分可能导致与C++标准，或者其他编译器不兼容的程序。

而这些扩展又不是必要的，我们建议关掉这些编译扩展。



#### 0.11 配置你的编译器：warning和error的层级

抛出错误，会不通过编译。抛出警告，不会停止编译，还是因为“相信开发者”的哲学。但我们应当习惯将每一个警告也解决掉。

warning是可以定制层级的。



#### 0.12 配置你的编译器：选择语言标准

一般编译器会默认的标准不会是最新的。

C++98, C++03, C++11, C++14, C++17, C++20, etc… 都可以选。

- c++1x = C++11
- c++1y = C++14
- c++1z = C++17
- c++2a = C++20
- c++2b = C++23

在某一代标准还没结束时，会暂时使用类似c++2a的名称。一般我们会选择早于最新标准一、两代的标准，比如C++20出来后，就倾向使用C++14和C++17。

这样一方面编译器还要一段时间针对最新标准打磨优化，另一方面前两代的标准在不同平台的兼容性和支持度也更高。

![Xcode](/Users/DevonnHou/Library/Application Support/typora-user-images/image-20210830083338204.png)



## C++基础学习

#### 1.1 程序的语句和结构

1.语句

就和自然语言是由语句构成一样，C++也是由语句构成。大多语句都以分号<font color="red">；</font>结尾。

在高级语言中一条语句往往可以编译为多条机器指令。

2.函数和主函数

在C++中，语句通常组成函数。一个函数是许多语句顺序执行的集合。每个C++都有一个特殊的函数，称作主(**main**)函数。

```c++
/*预处理指令，告诉编译器要用到iostream(标准库的一部分)的内容*/
#include <iostream> 
/*函数头*/
int main()
{
  /*函数体*/
  /*<<将Hello,world传递给控制台*/
   std::cout << "Hello world!";
   return 0;
}
```



#### 1.2 注释

1.单行注释：**//**   

2.多行注释：**/***  和  ***/ ** 不能嵌套，不然 /* 只会匹配左数出现的第一个 */ 导致错误

小tips：

- At the library, program, or function level, use comments to describe *what*.
- Inside the library, program, or function, use comments to describe *how*.
- At the statement level, use comments to describe *why*.

① 好的注释是解释why，而不是描述what。前者表达你的思想、决策，后者只能说明你的代码易读性差，要重新书写。当然对于初学者或者出于教学目的，可以写what型的注释。

② 更好的实践是用更长的变量名称，让它表达自己的含义，即**self documenting code**。

③ 注释掉代码：这个也是常用的做法。不过如果遇到需要嵌套使用多行注释时，也可以考虑使用 **#if 0** 预处理语句，后面会讲到。



#### 1.3 介绍对象和变量

前面提到语句构成函数，来运行产生结果。那结果怎么来的呢？必然要操纵（读、改、写）数据。**数据**就是一切可以被计算机移动、处理或存储的信息。

所有计算机都有内存，称作**RAM**。存在内存里的数据也叫值。一些老的语言，比如Apple Basic，是可以直接存取某某号内存的。但在C++中，是不允许直接访存的，而是间接地用一个对象（object）。它是一个内存区域，包括了值和属性。

即：

Apple Basic：*go get the value stored in mailbox number 7532.*

C++：*go get the value stored by this object*.

意味着我们可以使用对象来存储和检索值，而不用操心到底是放在哪一号内存里。

**对象**可以被命名或者未命名。一个被命名的对象被称作**变量**，它的名字也叫标识符。在我们的程序里大多数对象都是这种变量。

对象在程序运行期间会被实例化，也就是创建并分配内存地址；一个被实例化的对象称作**实例**。

**数据类型**告诉编译器，变量将存储什么类型的值。除了内置的类型，C++也支持用户定义的类型。这是C++强大的原因之一。所以我们介绍了变量的三个很基础的要素： 标识符，类型 和 值



#### 1.4 变量的分配和初始化

把定义（**define**）和分配（**assign**）一起做，就称作初始化（**initialize**）。

```c++
int a; // no initializer
int b = 5; // initializer after equals sign
int c( 6 ); // initializer in parenthesis
int d { 7 }; // initializer in braces
```

其中使用赋值运算符的，也称作拷贝初始化（**copy initialization**），这个很熟悉，是沿袭C语言而来的；

其中使用圆括号的，称作直接初始化（**direct initialization**）；

其中使用花括号的，称作大括号初始化（**list initialization** (also sometimes called **uniform initialization** or **brace initialization**））。

> 简单的数据类型，使用拷贝初始化就ok了，但复杂的数据类型，还是直接初始化效率更高。不过直接初始化不支持列表类型，所以提出列表初始化这样一个统一的形式。

最佳实践建议：

① 只要有机会就使用大括号初始化。不过另一方面对于单独的分配而言，C++只有拷贝分配，没有所谓的直接分配和大括号分配。

② 创建变量时就做初始化。除非故意，最好还是别只定义一个未分配的变量。



#### 1.5 介绍iostream

io库是C++标准库的一部分。

std::cout << 

还可以多个 << 连用，把一串控制台输出连起来，如：

```cpp
int x{ 5 };
std::cout << "x is equal to: " << x;
```

std::endl

换行。\n 也是换行，而且效率更高。因为endl多一个刷新输出的工作，而这个不是必要的，并且cout也会做这个工作。

std::cin >>

从键盘得到的输入必须存在一个变量中。还可以多个 >> 连用，接收多个输入，中间由空格相隔，如：

```cpp
std::cin >> x >> y; // get two numbers and store in variable x and y respectively
```

C++ I/O库不支持一种不需要按回车就能从键盘接收输入的方式。不过一些第三方库有实现这个函数功能。

小**tips**：<<、>>并不难记，它们表明了数据的传递方向。



#### 1.6 未初始化的变量和未定义的行为

不像其他编程语言，C++并不会自动初始化一个给定的值（比如0）。未初始化意味着，默认的值会变成所分配内存里本来存的一些无用（garbage）的值。

历史渊源：

早期计算机速度很慢，由于初始化每个变量会影响速度，而且大多时候这些变量的初始值是会被写覆盖的。所以C语言默认就不进行初始化了（C++继承了这点）。当然，以现在计算机的性能已经几乎不用考虑这点资源消耗，除非你在需要极致优化的时候故意这么做。

未定义的行为(**UB**)指执行结果没有被C++语言定义的行为，未初始化变量就是其中一种。它可能出现许多症状，比如：

程序每次运行结果不定；程序崩溃；有的编译器可以正常编译，有的却不行；你修改代码一个不相干的地方，却影响了执行结果；等等

所以务必要避免**未定义行为**。



#### 1.7 关键字和如何命名标识符

C++ 20 有92个关键字，也称保留字。

标识符的命名**规则**：① 不能是关键字 ② 由字母、数字、下划线组成 ③ 首位只能是字母或下划线 ④ 大小写敏感

标识符的命名**习惯**：① 变量的首位用小写字母 ② 函数的首位用小写字母，接着蛇形或**驼峰**命名法 ③ 用户定义的类型（如结构体、类、枚举）首位采用大写字母

<font color="red">注1：</font>不过如果你要在一个现有代码上进行工作，更好地还是延续这份代码的命名风格，而不是生硬地照搬之前的习惯。

<font color="red">注2：</font>避免用下划线开头的标识符，这一般是留给操作系统、库和编译器用的。

<font color="red">注3：</font>令标识符有含义，并且琐碎的、不重要的标识符用短一点的名字如*i*；广泛用到的标识符用长一点的、描述性的名字如*openFileOnDisk*；

<font color="red">注3’：</font>避免使用缩略词，虽然能减少你写代码的时间，但是易读性会大大降低，令你更难维护。**代码被读的次数会比写的次数多。**IDE的自动补全照样可以帮你写快。

> Code is read more often than it is written, the time you saved while writing the code is time that every reader, including the future you, wastes when reading it. 



#### 1.8 空格和格式

空格是用来组成格式的。包括了spaces，tabs 和 newlines。

编译器会无视空格，所以我们称C++是空格无关（independent）的语言。

如果一个很长的语句被分为多行，操作符应该放在前面：

```cpp
std::cout << 3 + 4
    + 5 + 6
    * 7 * 8;
```

漂亮的写法：

```cpp
cost          = 57;
pricePerItem  = 24;
value         = 5;
numberOfItems = 17;
```

养成好习惯（second nature）

Code -> Preferences -> Keyboard Shortcuts 可以找到VS Code关于**auto-format**的快捷键，Mac上默认是 Option/ALT + Shift + F.



#### 1.9 字面量，操作符

在计算机科学中，**字面量（literals）**就是指这个量本身，比如字面量3。也就是指3。字面量是相对变量常量等定义的。

string x=“ABC” 意思是把字面量”ABC” 赋值给变量 x。const string y=”cbd”. 意思是把字面量”cbd” 赋值给了常量y。字面量，即自己描述自己的量。

有的**操作符（operators）**是一个符号（+、*、=），有的是多个符号（>>、==），有的是词语（new、delete、throw）。按操作数个数，又可以分为一元、二元、三元操作符。



#### 1.10 表达式

表达式是字面量、变量、操作符和显式函数调用的组合，这个组合应当输出一个值。



#### 1.11 程序

不要试图一次写完。可以写一部分，编译通过，再添加一部分代码。也不是一次写得漂亮，写完能正常工作后，再进行优化。



#### 1.x 第一章总结



## 函数和文件



#### 2.1 函数的介绍

前面介绍过一个函数是许多语句顺序执行的集合。但这个定义没有提供函数的用处，这里更新一下定义：**函数是设计用于完成特定工作的可复用语句序列。**

```cpp
return-type identifier() // 函数头
{
// Your code here 括号连同里面的代码称为函数体
}
```

函数不能嵌套定义，即函数不可以在另一个函数里定义。

小tips：词语“foo”常用来占位作为一个不重要、讲解概念用的函数的名称，它本身没有啥意义。

<font color="red">**可复用**</font>既是定义也是目的。<font color="red">Follow the DRY best practice: “don’t repeat yourself”.</font> 



#### 2.2 函数的返回值

返回值不一定是字面量，**可以是任何表达式。**但要与你的返回值类型吻合，不然造成未定义行为了。

当函数不需要返回值时，使用返回值类型void，然后就不要写return语句。

main函数的返回值也称作状态码，因为它能反映程序有没有成功执行。一般返回**0**表示正常运行。

C++不允许显式地调用main()函数。



#### 2.3 函数的参数

形参 parameter 实参 argument  

当函数被调用时，所有形参会被创建为变量，并且实参的值会传递给形参。



#### 2.4 局部

函数的形参和定义在函数体中的变量，都称作局部变量。

```cpp
int add(int x, int y) // function parameters x and y are local variables
{
    int z{ x + y }; // z is a local variable too
    return z;
} // z, y, and x destroyed here
```

大多数时候，局部变量是在进入函数时创建，在离开函数时销毁。但有的特别的编译器可以决定更早创建和更晚销毁（注：但不会改变后文提到的local scope），来达到优化的目的。

```cpp
#include <iostream>

void doSomething()
{
    std::cout << "Hello!\n";
}

int main()
{
    int x{ 0 }; // x's lifetime begins here

    doSomething(); // x is still alive during this function call

    return 0;
} // x's lifetime ends here
```

**Local scope 局部范围**

一个标识符的范围 决定了源码中这个标识符在哪处可以访问到。

这是编译时就确定的属性，如果尝试在该范围外用到某个标识符，便会报错。

好习惯：最好在尽可能接近要使用的地方定义局部变量。



#### 2.5 函数的用处

Organization、Reusability、Testing、Extensibility、Abstraction



#### 2.6 前向声明

**前向声明**（**Forward Declaration**），是指声明标识符(表示编程的实体，如数据类型、变量、函数)时还没有给出完整的定义。

因为：如果函数B调用函数A，那编译器必须要先知道A是什么，所以A要先定义。但如果A和B相互调用咋办（循环依赖）？→ 前向声明

前向声明函数时，只需要函数原型：返回值、名字、形参，不需要函数体，直接分号结束。

```cpp
int add(int x, int y); // forward declaration of add() (using a function prototype)
int add(int, int); // valid function prototype
```

而且可以不用写形参的名字（但习惯上还是会写，为了易读性）

前向声明函数是最常见的，也可以用于变量、用户定义的数据类型。语法有一点差别，在之后的章节会介绍。



<font color="red">**冷知识：**</font>所有的定义也是声明。

声明能满足编译器，但光只是声明不能满足链接器。

定义能满足编译器（所以它也是声明）、链接器。

对于变量，如`int x;`，就既是定义也是声明。

> 我们平时语境下说的声明就是纯粹的声明 pure declarations，不是定义，不能满足链接器的。

只能**一处定义**，可以多处声明（但多处是冗余的）。

对于同一标识符、不同参数的函数，是视为不同函数的。遇到这种情况不叫重复定义，不会报错的。这个叫**重载**。



#### 2.7 包含多个文件的程序

大型程序都会包含多个文件，得以更好地组织和复用。

面对多文件的项目代码，IDE会很方便。

1. 用前向声明，不同文件里的函数就能够互相调用。
2. 不同的文件是独立编译的，不存在先后顺序。
3. 文件要加到项目里才算。



#### 2.8 命名空间

前面提过，不同的文件是独立编译的。

但是在链接的时候，如果有同名的函数，就会报错。哪怕都编译通过了。只能**一处定义**。



**命名冲突**大多发生在函数和全局变量上。

**命名空间**（namespace）可以解决这个问题。某个命名空间声明的标识符，不会被误认为是声明在另一个范围的同名标识符。

**The global namespace**

在C++中，任何没有定义在一个类、函数或命名空间中的标识符，会被认为是在全局命名空间。比如main函数就通常在全局命名空间。

**The std namespace**

起初发明C++语言时，标准库是在全局命名空间的，不需要使用std::。可想而知，带了非常多的麻烦。就改成了现在的样子。

使用方式1：

```cpp
#include <iostream>

int main()
{
    std::cout << "Hello world!"; // when we say cout, we mean the cout defined in the std namespace
    return 0;
}
```

*std::cout* 可以念做 “the *cout* that lives in namespace *std*“

使用方式2：

```cpp
#include <iostream>

using namespace std; // this is a using directive telling the compiler to check the std namespace when resolving identifiers with no prefix

int main()
{
    cout << "Hello world!"; // cout has no prefix, so the compiler will check to see if cout is defined locally or in namespace std
    return 0;
}
```

不推荐使用方式2，那就重蹈覆辙，C++语言发明之处的那些麻烦又要经历一遍。最糟糕的是现在不报错，未来在用C++新版时却报错，仅仅因为标准新增了几个标识符。



#### 2.9 预处理

在编译之前，其实还进行了一个称作**translation**的操作。而translation中最值得注意的是它涉及到了预处理**preprocessor**。预处理指令都以#开头。

预处理也是短暂地在内存中进行的，它并不会改变原来的代码文本。



常见的预处理指令（他们许多和C++的语法不同）：

- **Includes**

  语法示例：

  ```c++
  #include <iostream>
  ```

  作用：#include指令将该处替换为具体文件的内容。几乎都是用于头文件。

  

- **Macro defines**

  语法示例：

  ```cpp
  #define identifier substitution_text
  #define identifier
  ```

  作用：

  Function-like macros比较危险，尽量不使用，而且普通函数都能取代它，这里就不讨论了。

  Object-like macros with substitution text这里的标识符一般全用大写字母，预处理后全部被替换文本。如`#define MY_NAME "Alex"` 。它过去被作为常数（constant variables）的一种便捷的替代方法。除了一些遗留代码，现在基本不这么用了。

  Object-like macros without substitution text会将标识符替换成空白，也就是去掉这个内容。在接下来介绍的**Conditional compilation**中可以发挥特别的用处。

  <font color="red">Tips:</font>

  宏指令只替换C++代码中的标识符，不会替换其他预处理指令出现的该标识符。

  

- **Conditional compilation**

  语法示例：

  ```cpp
  #include <iostream>
  
  #define PRINT_JOE
  
  int main()
  {
  #ifdef PRINT_JOE
      std::cout << "Joe\n"; // if PRINT_JOE is defined, compile this code
  #endif
  
  #ifdef PRINT_BOB
      std::cout << "Bob\n"; // if PRINT_BOB is defined, compile this code
  #endif
    
  #ifndef PRINT_BOB
      std::cout << "Bob\n"; // if PRINT_BOB is not defined, compile this code
  #endif
  
      return 0;
  }
  ```

  可以控制哪些部分编译，哪些部分不编译。

  #ifdef PRINT_BOB与#ifndef PRINT_BOB也可以写成 

  #if defined(PRINT_BOB)与#if !defined(PRINT_BOB)

  **#if 0**也属于条件编译的预处理指令，它可以当作一种特殊的注释方式。避免了多重注释不能嵌套的问题。

  ```cpp
  #include <iostream>
  
  int main()
  {
      std::cout << "Joe\n";
  
  #if 0 // Don't compile anything starting here
      std::cout << "Bob\n";
      /* Some
       * multi-line
       * comment here
       */
      std::cout << "Steve\n";
  #endif // until this point
  
      return 0;
  }
  ```

最后，预处理都会在编译之前结束，宏指令的标识符则会被丢弃。所以一个文件里定义的宏指令，另一个文件是感知不到的。



#### 2.10 头文件

**Headers**

当程序越来越大，文件越来越多。需要前向声明的函数就会非常冗长。

有没有一种方法，就是**将前向声明都放在一个位置**，然后任何要用到的地方引用它就好？

这就是C++中第二种最常见的文件：**头文件**（后缀.h，也有.hpp或无后缀的，如`iostream`）

头文件帮助我们省下了很多打重复代码的精力。

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20211014125548688.png" alt="image-20211014125548688" style="zoom:50%;" />

**最佳实践：**

① 头文件一般不要出现函数和变量的定义，以免日后违背“一处定义”的问题。

② 源代码一般都会#include它自个儿的头文件。这样编译器能在编译时就发现问题，而不是链接时才发现。

比如

something.h:

```cpp
int something(int); // return type of forward declaration is int
```

something.cpp:

```cpp
#include "something.h"

void something(int) // error: wrong return type
{
}
```

就能在编译时发现问题了。

③ 虽然include的头文件很可能会include其他头文件。这样“传递”声明。但还是建议显式地include所有需要的头文件，而不是倚赖传递。

**Q: I didn’t include <someheader.h> and my program worked anyway! Why?**

这种情形是可能发生的，就是当头文件“传递”声明发生时。但这可能导致某个程序在你的机器能运行，但在别人的机器无法运行。



**冷知识一**

为什么既有尖括号（#include <iostream>），又有双引号（#include "add.h"）的形式。

因为出现头文件的位置，既可能是项目路径（current directory），又可能是系统环境（include directories）。尖括号 vs双引号 可以更好地引导编译器去哪儿寻找头文件。

尖括号用于非用户编写的头文件，编译器会直截了当去include directories找。双引号用于用户编写的头文件，编译器会先在include directories找。

> Use double quotes to include header files that you’ve written or are expected to be found in the current directory. Use angled brackets to include headers that come with your compiler, OS, or third-party libraries you’ve installed elsewhere on your system.

**冷知识二**

为什么标准库的头文件没有.h后缀？

其实同时存在无后缀的iostream和iostream.h的头文件，但二者不是一回事。 这是由于历史原因，起初所有的标准库头文件都有.h后缀。但在进入美国国标时，更规范地要求函数在std命名空间，以免和用户定义的函数冲突。

此时如果重写标准库的代码，一些旧的程序就没法运行了。为了解决这个问题，新使用了无后缀的头文件，所有在std命名空间的函数在这里声明。而那些旧的程序仍然可以使用.h后缀的头文件，而不需要重写。

此外，许多继承自C语言的库，还会给出一个c前缀，比如stdlib.h变为cstdlib。同样地，这部分库也被移到了std命名空间。

**关于include其他路径下的头文件**

```cpp
#include "headers/myHeader.h"
#include "../moreHeaders/myOtherHeader.h"
```

上面这种写相对路径的不是良好的办法。万一改动文件结构，就没法用了。

更好的措施是：设置编译器、IDE的环境路径或者叫搜索路径。*include path* or *search directory*

**关于include各种头文件的顺序**

如果头文件写得规范，每个都有齐全的声明，那么主程序include的顺序就不成问题，任何顺序都没事。

但如果写得不规范，出现互相倚赖，就需要调整顺序了。不过发现这类错误是好事情，我们可以fix掉，而不是留有隐患。

所以<font color="red">最佳实践</font>推荐这么排序：

1. 和源码成对的头文件
2. 项目的其他头文件
3. 第三方头文件
4. 标准库头文件

这样当用户定义的头文件需要倚赖第三方或标准库头文件时，可以很快发现编译错误并且fix。



#### 2.11 重复定义的问题

还是上节提到的**最佳实践：**① 头文件一般不要出现函数和变量的定义，以免日后违背“一处定义”的问题。

比如：

square.h:

```cpp
// We shouldn't be including function definitions in header files
// But for the sake of this example, we will
int getSquareSides()
{
    return 4;
}
```

geometry.h:

```cpp
#include "square.h"
```

main.cpp:

```cpp
#include "square.h"
#include "geometry.h"

int main()
{
    return 0;
}
```

就会出问题，这预处理后相当于：

```cpp
int getSquareSides()  // from square.h
{
    return 4;
}

int getSquareSides() // from geometry.h (via square.h)
{
    return 4;
}

int main()
{
    return 0;
}
```

**头文件保护符**

好消息是我们可以利用头文件保护符（**header guard**）的机制来避免上面的问题。标准库的头文件全加上了它。

头文件保护符属于条件编译的指令，写法如下：

```cpp
#ifndef SOME_UNIQUE_NAME_HERE
#define SOME_UNIQUE_NAME_HERE

// your declarations (and certain types of definitions) here

#endif
```

很多编译器也都支持#pragma once，来作为头文件保护符，就一行，更为简单。但这不是标准里的，所以保险起见还是用#ifndef。

```cpp
#pragma once

// your code here
```

<font color="red">**#ifndef**</font> 

当SOME_UNIQUE_NAME_HERE已经定义过，编译器就会忽略它，避免重复定义。如果没定义过，就可以给它定义。

**大写字母+下划线**：一般SOME_UNIQUE_NAME_HERE会直接写为该头文件的名称，对整个头文件预防重复定义。格式是全大写，且标点或空格改为下划线。

例如square.h:

```cpp
#ifndef SQUARE_H
#define SQUARE_H

int getSquareSides()
{
    return 4;
}

#endif
```

但头文件保护符只能防止同一个文件内不要出现重复定义，你在a.cpp和main.cpp分别定义同一个函数，仍然会导致编译成功、链接失败。所以最终还是希望能遵守<font color="red">**最佳实践**</font>，不要在头文件里定义东西。



#### 2.12 设计第一个程序

在很多方面，编程就像做建筑。一开始需要蓝图。

① **Define your goal** 

② **Define requirements** 

③ **Define your tools, targets, and backup plan** 

④ **Break hard problems down into easy problems** 

⑤ **Figure out the sequence of events** 

实现也是先框架后细节的。

① **Outlining your main function** ② **Implement each function** ③ **Final testing**



#### 2.x 第二章总结





## 调试程序

bug、软件错误是非常常见的。关键是我们用什么方法去处理它。

学会找到和解决bug是成为一名出色程序员的重要技能。

#### 3.1 语法和语义错误

编程具有挑战，而C++又是一个古怪的语言。二者放一起，能出现的bug会是五花八门。

主要分为两类：**语法错误**，**语义错误**（也叫逻辑错误）

语法错误容易排查，编译器会指明。虽然现代的高级编译器可以检查出个别类型的语义错误，但大部分的语义错误是没法检查出的。毕竟编译器设计的初衷就是解析语法，而不是程序的意图。

除了特别简单的语义错误能一眼看出来，大部分是没法轻松目测出来的。<font color="red">**调试**</font>技术就显现出了用处！



#### 3.2 调试的过程

bug的出现一般有这么个简单前提：

> Something that you thought was correct, isn’t. 

找到问题根源 ➡️ 尝试理解问题 ➡️ 确定解决办法 ➡️ 修复问题 ➡️ 重新测试



#### 3.3 调试的战略

方式一：检查代码

但遇到复杂的项目时，方式一难度大、效率低，并且很枯燥。

方式二：通过运行来诊断：

- 重现问题：你首先要亲眼看一下错误的发生
- 收集信息，缩小范围：比如根据错误类型，甚至根据直觉
- 反复进行上面的过程



#### 3.4 调试的基本战术（手动篇）

战术1:  注释掉代码

战术2:  查看调用次数、顺序，在函数的开头print函数名。

<font color="red">注：</font>这里print得用std::cerr，因为std::cout是有缓冲的，也就是在你希望它输出，到它实际输出有时间间隔。如果这中间程序挂掉了，就会误导你。而std::cerr是无缓冲的（只是性能差点，但debugging的时候咱们不在意性能）。

战术3:  输出变量的值 

<font color="red">注：</font>也使用std::cerr

> 但这种输出语句来调试不太好，除非是手头没有称手的debugger。它的弊端是 调试语句 ① 让代码更杂乱 ② 让输出更杂乱 ③ 调试完毕要手动删除，也没法复用 ④ 需要编辑代码，有时误编辑带来新的bug。
>



#### 3.5 调试的进阶战术（自动篇）

上一章讲到一些调试的方法，会带来些麻烦。调试语句要手动加和删。

**1.** 使用预处理指令

更好的办法是配合预处理指令，让程序自动判断用不用调试语句：

```cpp
#include <iostream>

#define ENABLE_DEBUG // comment out to disable debugging

int getUserInput()
{
#ifdef ENABLE_DEBUG
std::cerr << "getUserInput() called\n";
#endif
	std::cout << "Enter a number: ";
	int x{};
	std::cin >> x;
	return x;
}

int main()
{
#ifdef ENABLE_DEBUG
std::cerr << "main() called\n";
#endif
    int x{ getUserInput() };
    std::cout << "You entered: " << x;

    return 0;
}
```

就可以通过是否注释*#define ENABLE_DEBUG* 这行来决定调试与否了。

**2. **使用日志

这是很常用的方法，好处也多多。

有很多第三方的日志工具，使用哪种取决于我们自己啦，这里以plog为例：

```cpp
#include <iostream>
#include <plog/Log.h> // Step 1: include the logger headers
#include <plog/Initializers/RollingFileInitializer.h>

int getUserInput()
{
	PLOGD << "getUserInput() called"; // PLOGD is defined by the plog library

	std::cout << "Enter a number: ";
	int x{};
	std::cin >> x;
	return x;
}

int main()
{
	plog::init(plog::debug, "Logfile.txt"); // Step 2: initialize the logger

	PLOGD << "main() called"; // Step 3: Output to the log as if you were writing to the console

	int x{ getUserInput() };
	std::cout << "You entered: " << x;

	return 0;
}
```

而且开启、关闭也很方便：

```cpp
plog::init(plog::none , "Logfile.txt"); // plog::none eliminates writing of most messages, essentially turning logging off
```

这样就关闭logger了。很多logger都提供了不同模式、档位，来减少甚至停止向日志输出内容。



#### 3.6 使用集成调试器（integrated debugger）

上面3.4～3.5节的方法都是假设我们没法暂停一个运行的程序。但现代IDE提供了我们<font color="red">**调试器**</font>，打破这一假设。

##### 3.6.0 调试器（debugger）

调试器：一种能  控制另一个程序的执行过程和检查另一个程序的状态  的程序。



##### 3.6.1 单步调试（Stepping）

是允许我们逐语句执行的功能。它包含以下一些命令：

**Step into**

按程序的正常执行顺序，运行下一条语句。如果该语句包含一个函数调用，则会在这个被调用函数的开头停下。

而且你会看到某种标记，用来指示要运行的下一行。（注：调用函数，和返回，会指示两次。）

**Step over**

不会进入调用的函数一行行执行，而是直接执行完整个函数。

它可以帮忙在debug时跳过那些你认为没问题或者不感兴趣的函数。

**Step out**

会直接执行当前函数的剩余未执行部分，然后在函数返回的位置停下。

它可以帮忙在debug时跳过那些你认为没问题或者不感兴趣，但是不小心step into进去的函数。

**Step back**

一般来说单步调试只能前进不能后退。点快、错过了，只能从头来，细心一点。

但现在有些调试器比如Visual Studio企业版，能支持返回上一个状态。不过开启这个功能是有代价的，要额外存一份独立的程序状态。大多数调试器都还不支持。

##### 3.6.2 运行和断点（Running and breakpoints）

##### 3.6.3 跟踪变量（Watching variables）

##### 3.6.4 调用堆栈（The call stack）



























































































