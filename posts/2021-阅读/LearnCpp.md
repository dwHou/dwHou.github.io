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

不要试图一次写完。可以写一部分，编译通过，再添加一部分代码。



#### 1.x 第一章总结













































































































































