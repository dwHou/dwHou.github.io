https://www.3blue1brown.com/

#### 第一章 微积分的本质

目标：微积分里的概念和公式很多。而本视频的目的是，让观众看完之后感觉自己也能发明微积分。

> [!TIP]
>
> "The art of doing mathematics is finding that *special case* that contains all the germs of generality."
> \- David Hilbert

###### 面积问题

圆的面积 $Area = \pi R^2$

###### 解开圆环

![img](https://www.3blue1brown.com/content/lessons/2017/essence-of-calculus/figure-2.39.svg)

本着使用标准微积分符号的精神，我们将其中一个环的厚度称为$dr$

一个圆环拉直，近似于矩形面积 $Area = 2\pi rdr$ .  

> 从圆周中古人最早发现$\pi$的定义。

> [!NOTE]
>
> 由于圆环的外圆周长会略大于内圆周长，所以拉直后近似于矩形是不准确的。更类似梯形。但当$dr$越小时，错误也会越来越小。
>
> 你可能会疑问，虽然知道$dr$足够小时，误差可以忽略。但所有圆环面积的误差累积起来会不会显著呢？如果想更精确，也可以证明每个圆环面积的误差受限（bounded）于常数倍的$(dr)^2$，圆环的总数为$\frac{R}{dr}$，所以误差累积受限于常数倍的$\frac{R}{dr}(dr)^2 = Rdr$ 也是趋近于0的。

###### 可视化数学

>  [!NOTE]
>
> “Think to Graph” 是一种学习策略，也是一种哲学：用图像思维来深入理解数学。

![img](https://www.3blue1brown.com/content/lessons/2017/essence-of-calculus/figure-4.14-4.57.svg)



作为一名大胆的数学家，你可能会预感到，将这个过程推向极致，实际上可能会让事情变得更容易，而不是更难。

将这些圆环拉直得到的矩形竖直地并排放在一个水平轴上，会发现组成了一个三角形，面积是 $Area = \frac{1}{2} R (2 \pi R) = \pi R^2$

这是圆面积的公式。无论你是谁，无论你对数学有什么看法，它都美极了。



###### 方法推广

作为一名数学家，你不仅关心寻找答案，还关心开发通用的解决问题的工具和技巧。所以，花点时间思考一下刚才发生了什么，以及为什么它有效，因为我们从近似值到精确值的过渡其实非常微妙，并且深刻地触及了微积分的本质。

