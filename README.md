## 用EM路由实现的矩阵胶囊（中文译本）
        
Geoffrey Hinton, Sara Sabour, Nicholas Frosst{geoffhinton, sasabour, frosst}@google.com
谷歌大脑   多伦多, 加拿大

### 摘要
一个胶囊是一组神经元，其输出表征同一实体的不同属性。一个胶囊网络的每层含有多个胶囊。我们描述一种胶囊版本，其中每个胶囊有一个逻辑单元来表明一个实体的存在性和一个4x4矩阵，这个矩阵能够习得表征那个实体与观察者（姿态）的关系。每层的一个胶囊对上层的多个不同胶囊构成的姿态矩阵进行投票，方法是它的姿态矩阵与可训练的视点不变的能够习得表征局部-整体关系的变换矩阵相乘。每张选票通过一个分配的系数进行加权。每张图片采用Expectation-Maximization algorithm对这些系数进行迭代更新，这样，每个胶囊的输出路由到接受一组相似选票的上层的一个胶囊。变换矩阵的训练不同，是在每对相邻胶囊层之间采用展开式迭代的EM算法（unrolled iterations of EM）进行反向传播。通过smallNORB评测, 与最好记录相比，胶囊减少了45%的测试错误率。同时显示出比标准的CNN对白盒对抗攻击具有超强的抵抗力。

------
### 1 简介
卷积神经网络是构建于这样的简单事实，即一个视觉系统要对图片中的所以位置采用同样的知识。这是通过绑定特征探测器的权重，以便在一处习得的特征在别处有效。卷积胶囊网络扩展位置知识共享到包括局部-整体关系的知识，这种关系通过一个熟悉的图形表示。视点变化对像素强度有复杂效果，但对表征对象或对象局部和观察者之间的关系的姿态矩阵有简单线性效果。胶囊网络意图利用好这一底层线性关系，处理视点变化和提升分割决定力。胶囊网络利用高维度巧合过滤：通过寻找给姿态矩阵投票的协议，探测出一个熟悉对象。这些票来自于已经探测出的对象局部。一个对象局部产生一票，方法是它的姿态矩阵乘以一个习得的变换矩阵，其表征视点不变的局部和整体关系。随着视点变化，局部和整体的姿态矩阵会以一种协调方式改变，这样，来自不同局部的投票间的任何协议都会保持。在一堆不相关选票中寻找高维选票的紧致集群是一种解决局部整体归属问题的办法。这是不同寻常的，因为我们不能用对低维度翻译空间网格化以利卷积那样，对高维度体态空间网格化。对于这个挑战，我们采用称为协议路由“routingby-agreement”的快速迭代处理方法，即对一个局部属于一个整体的概率进行更新，这是基于来自于那个局部的选票接近于来自属于那个整体的其它局部的选票。这是一个强大的分割原则，其允许采用熟悉的图形知识派生分割，而不是仅仅使用如颜色或速度的近似值或一致性等低级方法。胶囊网络和标准神经网络的一个重要区别在于，一个胶囊的激活是基于一种在多个输入体态预测之间的比较，而标准神经网络是基于在一个单一输入活动向量和一个习得的权重向量的比较。

### 2 胶囊网络如何工作
典型地，神经网络应用简单的非线性，即应用一个非线性函数进行一个线性过滤器的标量输出。也可以用softmax的非线性将一个全部logits向量转化成一个概率向量。
胶囊网络用一个复杂得多的非线性方法将一层中的全部激活概率和胶囊姿态集合转化为下一层的激活概率和胶囊姿态。一个胶囊网络包含几个胶囊层。在L层的这组胶囊用$Ω_L$表示。每个胶囊有一个4x4姿态矩阵，M，和一个激活概率，a。它们就像一个标准神经网络的活动：依赖于当前输入，不被保存。在L层的每个胶囊i和L+1层的每个胶囊j之间有一个4x4的可训练的转换矩阵，$W_{ij}$。这些$ W_{ij} $（和每个胶囊两个习得的偏置项）是唯一保存参数，而且它们是分别习得的。胶囊i的姿态矩阵由$W_{ij}$进行变换，即对胶囊j的姿态矩阵投一票$V_{ij} = M_iW_{ij}$。以$V_{ij}$ 和 $a_i（i ∈ Ω_L, j ∈ Ω_{L+1}）$为输入，用非线性路由程序对L+1层的全部胶囊的姿态和激活进行计算。这个非线性程序是EM程序的一个版本。它迭代地调整L+1层胶囊的均值，变化和激活概率，以及在所有$i ∈ Ω_L, j ∈ Ω_{L+1}$之间的分配概率。在附录1，我们给协议路由（routing-by-agreement）一个中性的直观介绍，并详细描述它与拟合高斯混合的EM算法之间的关系。

### 3 用EM实现路由协议
假定，我们已经确定了一层中所有胶囊的姿态和激活概率，现在，我们想要确定上层有哪些激活胶囊？以及如何将每个激活的低层胶囊分配给一个激活的高层胶囊。高层中的每个胶囊对应一个高斯（Gaussian），低层（转成了一个向量）的每个激活胶囊的姿态对应一个数据点（或者数据点的片断，胶囊部分激活的情况下）利用最短描述长度原则，当决定是否激活一个高层胶囊时，我们面临一种选择。

选择0: 如果不激活它，要描述所有分配给高层胶囊的低层胶囊的姿态，我们必须为每个数据点付出$−βu$的固定开销。在非适当均匀先验分布的情况下，这个开销是数据点的负对数概率密度。对于片段分配，我们付出固定开销的片段。

选择1: 如果激活更高级别的胶囊，我们必须付出$−βa$的固定开销来编码它的均值和方差，以及它是激活的事实，然后支付额外的费用，并按照分配概率进行比例分配，以描述较低级别的均值和预测值之间的差异，这是当更高级别胶囊的均值通过转换矩阵的逆向方法来预测它们时需要的。计算描述一个数据点成本的简单得多的方法是，在无论归属哪个高层胶囊拟合的高斯分布下，使用那个数据点的选票的负对数概率密度。

对于附录1解释的理由，是不正确的，不过，我们用它是因为它需要更少的计算（在附录中也有说明）。
选择0和1在开销方面的区别是，在每次迭代中通过逻辑函数确定更高级别胶囊的激活概率。附录1解释了逻辑函数是正确选择的原因。
使用我们针对上述选择1的高效近似值，通过使用有轴对齐协方差矩阵的活动胶囊j来解释整个数据点i产生的增量成本，简单地说，就是解释投票$V_{ij}$的每个维度h的全维开销总和。简单描述为$−ln(P^h_{i|j}) $ 其中 $P^h_{i|j}$是矢量化选票$V_{ij}$的$h^{th}$组件的概率密度，这是在j对维度h的高斯模型中，有方差$(σ^h_j)^2$和均值$µ^h_j$，其中$µ_j$是j的姿态矩阵$M_j$的矢量化版本。

$P^h_{i|j} = \frac{1}{\sqrt{2π(σ^h_j)^2}}exp\Bigl(−\frac{(V^h_{ij} − µ^h_j)^2}{2(σ^h_j)^2}\Bigr),
ln(P^h_{i|j}) = \frac{(V^h_{ij} − µ^h_j)^2}{2(σ^h_j)^2} − ln(σ^h_j) − ln(2π)/2$

对j的一个单个维度h，计算其全部低层胶囊的总和，得到：

$$cost{^h_j} = \sum_i−r_{ij} ln(P^h_{i|j})$$

$$=\frac{\sum_ir_{ij} (V^h_{ij} − µ^h_j)^2}{2(σ^h_j)^2}+(ln(σ^h_j) + \frac{ln(2π)}2)\sum_ir_{ij}           (1)$$
$$= \biggl(ln(σ^h_j) + \frac{1+ln(2π)}2\biggr) \sum_ir_{ij} $$

其中$\sum_ir_{ij}$ 是分配给j的数据量，$V^h_{ij}$是$V_{ij}$在维度h上的值。开启胶囊j提高了分配给j的较低级别胶囊均值的描述长度，从每个较低级别胶囊的$-βu$到$-βa$加上所有维度的成本总和，所以我们定义胶囊j的激活功能为：

$$a_j = logistic\biggl(λ\bigl(βa − βu\sum_ir_{ij} −\sum_hcost^h_j\bigr)\biggr) (2)$$

其中，$βa$对于所有胶囊而言是一样的，而且，$λ$是一个逆向温度参数. 我们通过差异方法学习$βa$和$βu$，并将$λ$设定为超参数。要完成L + 1层胶囊的姿态参数和激活，我们在L层已经完成姿态参数和激活之后运行几轮EM算法迭代（通常为3轮）。由整个胶囊层实现的非线性算法是一种使用EM算法的集群发现形式，所以我们称之为EM路由。

----------

程序1路由算法返回层L+1中的胶囊的激活和姿势，在给出胶囊在层L中的激活和投票情况下。$V^h_{ij}$是来自从L层的激活$a_i$胶囊i到L+1层的胶囊j的第h维选票. $βa，βu$是区别习得的，并且逆向温度λ在每次迭代中按固定时间表提高。

----------
1: **procedure** EM ROUTING$(a, V )$

2: $∀i ∈ Ω_L, j ∈ Ω_L+1: R_{ij} ← 1/|Ω_L+1|$

3: for $t$ iterations do

4: $∀j ∈ Ω_L+1$: M-STEP$(a, R, V , j)$

5: $∀i ∈ Ω_L$: E-STEP$(µ, σ, a, V , i)$

return $a, M$


1: **procedure** M-STEP$(a, R, V , j)$ . 》for one higher-level capsule, j

2: $∀i ∈ Ω_L: R_{ij} ← R_{ij} ∗ a_i$

3: $∀h: µ^h_j ←\frac{\sum_i R_{ij}V^h_{ij}}{\sum_i R_{ij}}$

4: $∀h: (σ^h_j)^2 ←\frac{\sum_i Rij (V^h_{ij}−µ^h_j)^2}{\sum_i Rij}$

5: $cost^h ←\bigl(βu + log(σ^h_j)\bigr)\sum_i R_{ij}$

6: $a_j ← logistic(λ(β_a −\sum_hcost^h))$


1: **procedure** E-STEP(µ, σ, a, V , i) . 》for one lower-level capsule, i

2: $∀j ∈ Ω_{L+1}: p_j ← \frac{1}{\sqrt{\prod^H_h2π(σ^h_j)2}}exp\bigl(−\sum^H_h\frac{(V^h_{ij}−µ^h_j)2}{2(σ^h_j)2}\bigr)$

3: $∀j ∈ Ω_{L+1}: R_{ij} ← \frac{a_j p_j}{\sum_{k∈Ω_{L+1}}a_kp_k}$


### 4 胶囊网络架构
模型总的架构如图1所示。
模型由一个5x5带32通道(A=32)，用ReLU非线性函数，步长为2的卷积层开始。所有其它层是胶囊层，始于主胶囊层。
B=32主胶囊类型的每个胶囊的4x4姿态是一个习得的所有的低层的在那个中心weiReLU的线性转换。

![架构图](https://github.com/humor250/matrixcapsules/blob/master/cape.png)

图1：胶囊网络架构有一个ReLU卷积层，后面跟一个主卷积胶囊层和两个其它卷积胶囊层。

The activations of the primary capsules are produced by applying the sigmoid function
to the weighted sums of the same set of lower-layer ReLUs.
主胶囊的激活是利用sigmoid函数处理同组的低层ReLU的权重总和产生。

The primary capsules are followed by two 3x3 convolutional capsule layers (K=3), each with 32
capsule types (C=D=32) with strides of 2 and one, respectively. The last layer of convolutional
capsules is connected to the final capsule layer which has one capsule per output class.

When connecting the last convolutional capsule layer to the final layer we do not want to throw
away information about the location of the convolutional capsules but we also want to make use of
the fact that all capsules of the same type are extracting the same entity at different positions. 

We therefore share the transformation matrices between different positions of the same capsule type and
add the scaled coordinate (row, column) of the center of the receptive field of each capsule to the first
two elements of the right-hand column of its vote matrix. We refer to this technique as Coordinate
Addition. 

This should encourage the shared final transformations to produce values for those two
elements that represent the fine position of the entity relative to the center of the capsule’s receptive
field.

主胶囊之后是两个3x3卷积胶囊层（K = 3），每个都有32个胶囊类型（C = D = 32），步幅分别为2和1。最后一层卷积胶囊连接到每个输出级都有一个胶囊的最终胶囊层。将最后一个卷积胶囊层连接到最后一层时，我们不想丢弃远离有关卷积胶囊位置的信息，我们也想利用所有同一类型的胶囊都在不同位置提取同一个实体的事实。为此，我们分享同一胶囊类型的不同位置的变换矩阵，然后将每个胶囊的接受域中心的缩放坐标（行，列）添加到它的投票矩阵右侧栏中的头两个元素。我们称这种技术为坐标加成。这应该有助于这个共享的最终转换产生价值，因为这两个元素表示了这个相对于胶囊接受域中心的实体的精确位置。

The routing procedure is used between each adjacent pair of capsule layers. For convolutional capsules,
each capsule in layer L + 1 sends feedback only to capsules within its receptive field in layer
L. 

Therefore each convolutional instance of a capsule in layer L receives at most kernel size X kernel
size feedback from each capsule type in layer L + 1. 

The instances closer to the border of the
image receive fewer feedbacks with corner ones receiving only one feedback per capsule type in
layer L + 1.

路由程序在每对相邻的胶囊层之间使用。对于卷积胶囊，L+1层的每个胶囊只将反馈发送到L层中其接受域内的胶囊。因此，L层的一个胶囊的每个卷积实例以最大核尺寸X接收来自L+1层的每个胶囊类型的核尺寸反馈。越接近图像边界的实例接收较少的反馈，如角落的实例接收仅仅一个L+1层的一个反馈。

### 4.1 SPREAD LOSS传播损失
In order to make the training less sensitive to the initialization and hyper-parameters of the model,
we use “spread loss” to directly maximize the gap between the activation of the target class (at) and
the activation of the other classes. If the activation of a wrong class, ai
, is closer than the margin,
m, to at then it is penalized by the squared distance to the margin:
$Li = (max(0, m − (a_t − a_i))2, L =\sum_{i \neq t}L_i (3)$
By starting with a small margin of 0.2 and linearly increasing it during training to 0.9, we avoid
dead capsules in the earlier layers. Spread loss is equivalent to squared Hinge loss with m = 1.
Guermeur & Monfrini (2011) studies a variant of this loss in the context of multi class SVMs.

为了降低训练对模型的初始参数和超参数敏感性，我们使用“传播损失”来直接最大化目标类（at）激活和其他类激活之间的差距。如果错误类别ai的激活比余量m更近，那么它受到距离平方的处罚：Li =（max（0，m - （at-ai））^2，L = Xi6 = tLi（3）从0.2的小余量开始，在训练过程中将其线性增加到0.9，我们避免了早期层中的死胶囊。扩散损失相当于m = 1时的平方Hinge损失。Guermeur＆Monfrini（ 2011）研究了在多类SVM的背景下这种损失的变化。
