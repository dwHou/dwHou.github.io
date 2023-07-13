http://chuquan.me/2021/12/03/understand-concepts-in-xcode/



点击蓝色的project，可以在building setting里找到search path，也就是该项目会在哪些路径找头文件。有些子项目会属于另一个项目，这里的search path有时会显示$inherited。

新增的头文件（而非修改），如果不添加到工程，哪怕在search path里也找不到。所以可以用拖拽的方式，将其添加到工程。

Xcode左边栏目一排图标要认识下，

<img src="../../images/typora-images/image-20230324212936687.png" alt="image-20230324212936687" style="zoom:50%;" />

悬浮可以看到分别是：

<font color="blue">项目导航，版本管理导航，符号导航，搜索，错误和警告，测试导航，调试导航，断点导航，报告导航</font>

比如，我们错误先看 错误和警告，再点进去详细看 报告导航，学会快速准确地debug代码。

C++常见的内存错误及解决方法

（1）内存分配未成功，却使用了它。
解决方法：在使用内存之前先检查指针是否是NULL。如果是用malloc来申请内存，应该用if(p == NULL)或if（p != NULL）进行防错处理。如果是new来申请内存，申请失败会抛出异常，所以应该捕捉异常来进行防错处理。
（2）内存虽然分配成功，但尚未初始化就引用它。
解决办法： 尽管有时候缺省时会自动初始化，但无论什么时候创建对象均要对其进行初始化，即使是赋0值也是不可忽略的。
（3）内存分配成功，但访问越界
解决方法：对数组for循环时要把握越界，否则可能会导致数组越界。
（4）忘记释放内存，导致内存泄漏
解决办法：动态内存的申请和释放必须配对，new-delete和malloc-free其使用次数必须相等。
（5）已经释放内存还在使用它
free或delete后 ，没有将指针设为NULL，产生“野指针”。



Xcode → Product → Profile 里面有这些工具：

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20230712110012273.png" alt="image-20230712110012273" style="zoom:50%;" />

其中比较常用的
 ·Time Profiler：分析代码的执行时间，执行对系统的CPU上运行的进程低负载时间为基础采样

