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

主胶囊的激活是利用sigmoid函数处理同组的低层ReLU的权重总和产生。主胶囊之后是两个3x3卷积胶囊层（K = 3），每个都有32个胶囊类型（C = D = 32），步幅分别为2和1。最后一层卷积胶囊连接到每个输出级都有一个胶囊的最终胶囊层。将最后一个卷积胶囊层连接到最后一层时，我们不想丢弃远离有关卷积胶囊位置的信息，我们也想利用所有同一类型的胶囊都在不同位置提取同一个实体的事实。为此，我们分享同一胶囊类型的不同位置的变换矩阵，然后将每个胶囊的接受域中心的缩放坐标（行，列）添加到它的投票矩阵右侧栏中的头两个元素。我们称这种技术为坐标加成。这应该有助于这个共享的最终转换产生价值，因为这两个元素表示了这个相对于胶囊接受域中心的实体的精确位置。

路由程序在每对相邻的胶囊层之间使用。对于卷积胶囊，L+1层的每个胶囊只将反馈发送到L层中其接受域内的胶囊。因此，L层的一个胶囊的每个卷积实例以最大核尺寸X接收来自L+1层的每个胶囊类型的核尺寸反馈。越接近图像边界的实例接收较少的反馈，如角落实例接收仅一个来自L+1层的一个反馈。

### 4.1 SPREAD LOSS传播损失
In order to make the training less sensitive to the initialization and hyper-parameters of the model,
we use “spread loss” to directly maximize the gap between the activation of the target class (at) and
the activation of the other classes. If the activation of a wrong class, ai
, is closer than the margin,
m, to at then it is penalized by the squared distance to the margin:
$Li = (max(0, m − (a_t − a_i))^2, L =\sum_{i \neq t}L_i (3)$
By starting with a small margin of 0.2 and linearly increasing it during training to 0.9, we avoid
dead capsules in the earlier layers. Spread loss is equivalent to squared Hinge loss with m = 1.
Guermeur & Monfrini (2011) studies a variant of this loss in the context of multi class SVMs.

为了降低训练对模型的初始参数和超参数敏感度，我们使用“传播损失”来直接最大化目标类（$a_t$）激活和其他类激活之间的间距。如果错误类别$a_i$的激活比对$a_t$余量m更近，那么它的罚额是距离平方：$$L_i = (max(0, m − (a_t − a_i))^2, L =\sum_{i \neq t}L_i (3)$$ 从0.2的小幅度开始，在训练过程中将其线性增加到0.9，我们避免了早期层中的死胶囊。传播损失相当于m = 1时的Hinge损失值的平方。Guermeur＆Monfrini（ 2011）研究了在多类SVM背景下这种损失的一个变体。

5 实验

smallNORB数据集（LeCun et al.（2004））有5种玩具的灰度立体图像：飞机，汽车，卡车，人类和动物，每种有10个涂哑光绿色的物理实例。每种的5个物理实例为训练数据，另外5个为测试数据。每个玩具都有18个不同的方位角（0-340），9个高度和6种光照条件，所以训练和测试数据集均包含24,300个96x96图像的立体对。我们选择smallNORB作为开发胶囊系统的基准，因为它是专为一种纯粹的图形识别任务而进行的细致设计，不受上下文和颜色干扰，但它比MNIST更接近自然图像。

表1：我们的胶囊架构的不同组件对smallNORB的影响。
![表一](https://github.com/humor250/matrixcapsules/blob/master/table1_matrixcapsules.png)

我们将smallNORB缩减为48×48像素，每幅图像正常化为有零均值和单位差异。
在训练过程中，我们随机裁剪出32×32小图片并添加随机亮度和与裁剪的图像形成对比。
在测试过程中，我们从图像中心剪下一个32×32的小图片，并在smallNORB上实现1.8％的测试错误。如果我们平均化测试时多个裁剪的种类激活，我们达到了1.4％。在不使用元数据的情况下，smallNORB上最好的报告结果是2.56％（Ciresfort等（2011））。为了实现这一点，他们添加了两个额外的立体图像对输入，图像是通过中心滤波器和离心滤波器创建的。他们也对图像应用仿射失真。我们的工作还击败了Sabour等人（2017）在smallNORB上达到2.7％的胶囊网络工作。我们还在NORB上测试了我们的模型，这是一个带增加背景的smallNORB的一个抖动版本，我们实现了2.6％的错误率，看齐了2.7％的最好记录（Ciresan et al.（2012））。

作为我们对新视角进行总结的实验基准，我们训练了一个CNN，带有两个分别具有32和64通道的卷积层。两层都有一个内核大小
为5，步幅为1，带一个2×2最大化池。第三层是1024个单元的带丢失（dropout）的全连接层，并连接到5路softmax输出层。所有隐藏的单元使用ReLU
非线性算法。对CNN基准，我们准备了上述对胶囊网络相同的图片。我们的基准CNN是广泛的超参搜索（过滤器大小，通道数量和学习率）的结果。
CNN基准在smallNORB上达到5.2％的测试错误率，有4.2M参数量。我们推断Ciresfort等人（2011）网络拥有2.7M参数。通过使用小矩阵乘法，与基准CNN相比，我们将参数数量减少了15到310K（和Ciresfort等人（2011）的9倍因子）。 一个只有68K可训练参数的A=64，B=8，C=D=16的小胶囊网络，达到了2.2％的测试错误率，这也击败了之前的最优的测试错误率。

Fig. 2 shows how EM routing adjusts the vote assignments and the capsule means to find the tight
clusters in the votes. 
The histograms show the distribution of vote distances to the mean (pose) of
each class capsule during routing iterations. 

At the first iteration, votes are distributed equally between
5 final layer capsules. Therefore, all capsules receive votes closer than 0.05 to their calculated
mean. In the second iteration, the assignment probability for agreeing votes increases. Therefore,
most of the votes are assigned to the detected clusters, the animal and human class in the middle
row, and the other capsules only receive scattered votes which are further than 0.05 from the calculated
mean. The zoomed-out version of Fig. 2 in the Appendix shows the full distribution of vote
distances at each routing iteration.
Instead of using our MDL-derived capsule activation term which computes a separate activation
probability per capsule, we could view the capsule activations like the mixing proportions in a
mixture of Gaussians and set them to be proportional to the sum of the assignment probabilities
of a capsule and to sum to 1 over all the capsules in a layer. This increases the test error rate on

图2 显示了EM路由如何调整投票分配和胶囊均值，以找到选票中的密集群。
直方图显示投票距离的均值（姿态）分布在路由迭代期间每个分类胶囊。在第一次迭代中，投票之间平均分配5最后一层胶囊。因此，所有胶囊的计算结果都会得到接近0.05的选票意思。在第二次迭代中，同意投票的分配概率增加。因此，大多数选票都被分配到检测到的集群，中间是动物和人类排，而其他胶囊只能得到比计算结果更多的零散投票意思。附录中图2的缩小版显示了投票的完整分配每次路由迭代的距离。而不是使用我们的MDL派生的胶囊激活术语来计算单独的激活每个胶囊的概率，我们可以观察胶囊激活，如a中的混合比例高斯混合，并将它们设置为与分配概率的总和成比例
的胶囊，并且在一层中的所有胶囊上总计为1。这增加了测试错误率

![图2](https://github.com/humor250/matrixcapsules/blob/master/pic2_matrixcapsules.png)
Figure 2: Histogram of distances of votes to the mean of each of the 5 final capsules after each
routing iteration. Each distance point is weighted by its assignment probability. All three images
are selected from the smallNORB test set. The routing procedure correctly routes the votes in the
truck and the human example. The plane example shows a rare failure case of the model where the
plane is confused with a car in the third routing iteration. The histograms are zoomed-in to visualize
only votes with distances less than 0.05. Fig. B.2 shows the complete histograms for the ”human”
capsule without clipping the x-axis or fixing the scale of the y-axis.

图2：投票距离每个胶囊5个胶囊后均值的直方图路由迭代。 每个距离点由其分配概率加权。 所有三个图像是从smallNORB测试集中选择的。 路由程序正确路由中的投票卡车和人的例子。该平面示例显示了一个罕见的模型失败案例飞机在第三次路线迭代中与汽车混淆。直方图被放大以可视化只有距离小于0.05的选票。图B.2显了“人”的完整直方图，胶囊没有剪裁x轴或固定y轴的比例。

Table 2: A comparison of the smallNORB test error rate of the baseline CNN and the capsules model
on novel viewpoints when both models are matched on error rate for familiar viewpoints.

表2：在熟悉视角下两模型误差率相同时，新视角下基线CNN和胶囊模型的smalNORB测试错误率的比较
![表2](https://github.com/humor250/matrixcapsules/blob/master/table2_matrixcapsules.png)

smallNORB to 4.5%. Tab. 1 summarizes the effects of the number of routing iterations, the type of
loss, and the use of matrices rather than vectors for the poses.
The same capsules architecture as Fig. 1 achieves 0.44% test error rate on MNIST. If the number
of channels in the first hidden layer is increased to 256, it achieves 11.9% test error rate on Cifar10
(Krizhevsky & Hinton (2009)).

smallNORB降至4.5％。 标签.1统计了路由迭代次数的影响，即类型损失，以及使用矩阵而不是向量来表示姿势。与图1相同的胶囊架构在MNIST上达到了0.44％的测试错误率。 如果在第一隐层的通道数量增加到256个，在Cifar10上实现了11.9％的测试错误率（Krizhevsky＆Hinton（2009））。

5.1 GENERALIZATION TO NOVEL VIEWPOINTS
A more severe test of generalization is to use a limited range of viewpoints for training and to test on
a much wider range. We trained both our convolutional baseline and our capsule model on one-third
of the training data containing azimuths of (300, 320, 340, 0, 20, 40) and tested on the two-thirds of
the test data that contained azimuths from 60 to 280. In a separate experiment, we trained on the 3
smaller elevations and tested on the 6 larger elevations.
It is hard to decide if the capsules model is better at generalizing to novel viewpoints because it
achieves better test accuracy on all viewpoints. To eliminate this confounding factor, we stopped
training the capsule model when its performance matched the baseline CNN on the third of the
test set that used the training viewpoints. Then, we compared these matched models on the twothirds
of the test set with novel viewpoints. Results in Tab. 2 show that compared with the baseline
CNN capsules with matched performance on familiar viewpoints reduce the test error rate on novel
viewpoints by about 30% for both novel azimuths and novel elevations.

5.1新视角概述
更严格的总的测试是使用有限范围的观点进行训练和测试范围更广泛。我们对卷积基线和胶囊模型进行了三分之一的训练的训练数据包含方位角（300,320,340,0,20,40），并在三分之二的包含方位角从60到280的测试数据。在另一个实验中，我们对3进行了训练较小的海拔高度和6个较大的海拔高度进行测试。很难决定胶囊模型是否更好地推广到新颖的观点，因为它在所有视点上实现更好的测试准确性。为了消除这个混杂因素，我们停了下来训练胶囊模型时，其性能与基线CNN的三分之一相符测试集使用了培训观点。然后，我们将这些匹配的模型在两三位进行比较具有新颖观点的测试集。结果在Tab.2显示与基线相比,在熟悉的视点上具有匹配性能的CNN胶囊，在新视角，对于新方位角和新高程，减少了约为30％测试错误率。

6 ADVERSARIAL ROBUSTNESS
There is growing interest in the vulnerability of neural networks to adversarial examples; inputs
that have been slightly changed by an attacker to trick a neural net classifier into making the wrong
classification. These inputs can be created in a variety of ways, but straightforward strategies such as
FGSM (Goodfellow et al. (2014)) have been shown to drastically decrease accuracy in convolutional
neural networks on image classification tasks. We compare our capsule model and a traditional
convolutional model on their ability to withstand such attacks.
FGSM computes the gradient of the loss w.r.t. each pixel intensity and then changes the pixel
intensity by a fixed amount  in the direction that increases the loss. So the changes only depend on
the sign of the gradient at each pixel. This can be extended to a targeted attack by updating the input
to maximize the classification probability of a particular wrong class. We generated adversarial
attacks using FGSM because it has only one hyper-parameter and it is easy to compare models
that have very different gradient magnitudes. To test the robustness of our model, we generated
adversarial images from the test set using a fully trained model. We then reported the accuracy of
the model on these images.
We found that our model is significantly less vulnerable to both general and targeted FGSM adversarial
attacks; a small  can be used to reduce a convolutional model’s accuracy much more than an
equivalent  can on the capsule model (Fig. 3). It should also be noted that the capsule model’s accuracy
after the untargeted attack never drops below chance (20%) whereas the convolutional model’s
accuracy is reduced to significantly below chance with an  as small as 0.2.
We also tested our model on the slightly more sophisticated adversarial attack of the Basic Iterative
Method (Kurakin et al. (2016)), which is simply the aforementioned attack except it takes multiple
smaller steps when creating the adversarial image. Here too we find that our model is much more
robust to the attack than the traditional convolutional model.

6 抵御的鲁棒性
人们越来越感兴趣的是神经网络对敌对的例子。输入
攻击者已经稍微改变了这一点，以欺骗神经网络分类器制造错误
分类。这些输入可以通过各种方式创建，但直接的策略如
FGSM（Goodfellow et al。（2014））已经显示大大降低了卷积的准确性
神经网络对图像分类任务。我们比较我们的胶囊模型和传统模型
卷积模型对其抵御这种攻击的能力。
FGSM计算损失w.r.t的梯度。每个像素的亮度然后改变像素
强度按固定量计算？在增加损失的方向上。所以这些变化只依赖于
每个像素处渐变的符号。这可以通过更新输入扩展到有针对性的攻击
以最大化特定错误类别的分类概率。我们产生了敌对情绪
使用FGSM的攻击，因为它只有一个超参数，并且很容易比较模型
它们具有非常不同的梯度大小。为了测试我们的模型的鲁棒性，我们生成了
使用完全训练的模型从测试集中获得敌对图像。然后我们报告了准确性
这些图像上的模型。
我们发现我们的模型对于普通和有针对性的FGSM敌手都不那么脆弱
攻击;一个小的 ？可以用来减少卷积模型的精度远远超过一个
相当于？可以在胶囊模型上（图3）。还应该指出，胶囊模型的准确性
在非目标攻击之后永远不会低于机会（20％）而卷积模型
准确性被降低到显着低于一个？小到0.2。
我们还测试了我们的基础迭代稍微复杂的敌对攻击模型
方法（Kurakin et al。（2016）），这只是上述攻击，除了它需要多次
创建敌对图片时步幅较小。我们也发现我们的模型更多
比传统的卷积模型更能抵御攻击。

![图3](https://github.com/humor250/matrixcapsules/blob/master/pic3_matrixcapsules.png)

Figure 3: Accuracy against  after an adversarial attack (left) and Success Rate after a targeted
adversarial attack (right). The targeted attack results were evaluated by averaging the success rate
after the attack for each of the 5 possible classes.
图3：对抗攻击（左）后的$\epsilon$准确性和目标攻击（右）后的成功率。目标攻击结果，是对5个可能种类的每一个的攻击后，通过平均成功率进行评估。

It has been shown that some robustness to adversarial attacks in models can be due to simple numerical
instability in the calculation of the gradient Brendel & Bethge (2017). To ensure that this
was not the sole cause of our model’s robustness, we calculated the percentage of zero values in the
gradient with respect to the image in the capsule model and found it to be smaller than that of the
CNN. Furthermore, the capsule gradients, although smaller that those of the CNN, are only smaller
by 2 orders of magnitude, as opposed to 16 orders of magnitude seen in Brendel & Bethge (2017)’s
work.
Finally we tested our model’s robustness to black box attacks by generating adversarial examples
with a CNN and testing them on both our capsule model and a different CNN. We found that the
capsule model did not perform noticeably better at this task than the CNN.

已经表明，模型中对抗攻击的一些鲁棒性可能归因于简单的数值
计算梯度Brendel＆Bethge（2017）的不稳定性。为了确保这一点
并不是我们模型稳健性的唯一原因，我们计算了零值的百分比
相对于胶囊模型中的图像的梯度，并且发现其小于
CNN。此外，胶囊梯度虽然小于CNN，但只有较小的梯度
增加2个数量级，而Brendel＆Bethge（2017）的数据则为16个数量级
工作。
最后，我们通过生成敌对的例子来测试我们的模型对黑匣子攻击的鲁棒性
与CNN并在我们的胶囊模型和不同的CNN上测试它们。我们发现了
胶囊模型在这项任务上的表现并不比CNN好得多。

7 RELATED WORK
Among the multiple recent attempts at improving the ability of neural networks to deal with viewpoint
variations, there are two main streams. One stream attempts to achieve viewpoint invariance
and the other aims for viewpoint equivariance. The work presented by Jaderberg et al. (2015)), Spatial
Transformer Networks, seeks viewpoint invariance by changing the sampling of CNNs according
to a selection of affine transformations. De Brabandere et al. (2016) extends spatial transformer
networks where the filters are adapted during inference depending on the input. They generate different
filters for each locality in the feature map rather than applying the same transformation to all
filters. Their approach is a step toward input covariance detection from traditional pattern matching
frameworks like standard CNNs (LeCun et al. (1990)). Dai et al. (2017) improves upon spatial
transformer networks by generalizing the sampling method of filters. Our work differs substantially
in that a unit is not activated based on the matching score with a filter (either fixed or dynamically
changing during inference). In our case, a capsule is activated only if the transformed poses coming
from the layer below match each other. This is a more effective way to capture covariance and leads
to models with many fewer parameters that generalize better.
The success of CNNs has motivated many researchers to extend the translational equivariance built
in to CNNs to include rotational equivariance (Cohen & Welling (2016), Dieleman et al. (2016),
Oyallon & Mallat (2015)). The recent approach in Harmonic Networks (Worrall et al. (2017))
achieves rotation equivariant feature maps by using circular harmonic filters and returning both the
maximal response and orientation using complex numbers. This shares the basic representational
idea of capsules: By assuming that there is only one instance of the entity at a location, we can
use several different numbers to represent its properties. They use a fixed number of streams of
rotation orders. By enforcing the equality of the sum of rotation orders along any path, they achieve
patch-wise rotation equivariance. This approach is more parameter-efficient than data augmentation
approaches, duplicating feature maps, or duplicating filters (Fasel & Gatica-Perez (2006), Laptev
et al. (2016)). Our approach encodes general viewpoint equivariance rather than only affine 2D
rotations. Symmetry networks (Gens & Domingos (2014)) use iterative Lucas-Kanade optimization
to find poses that are supported by the most low-level features. Their key weakness is that the
iterative algorithm always starts at the same pose, rather than the mean of the bottom-up votes.

Lenc & Vedaldi (2016) proposes a feature detection mechanism (DetNet) that is equivariant to affine
transformations. DetNet is designed to detect the same points in the image under different viewpoint
variations. This effort is orthogonal to our work but DetNet might be a good way to implement the
de-rendering first-stage that activates the layer of primary capsules.
Our routing algorithm can be seen as an attention mechanism. In this view, it is related to the work of
Gregor et al. (2015), where they improved the decoder performance in a generative model by using
Gaussian kernels to attend to different parts of the feature map generated by the encoder. Vaswani
et al. (2017) uses a softmax attention mechanism to match parts of the query sequence to parts of
the input sequence for the translation task and when generating an encoding for the query. They
show improvement upon previous translation efforts using recurrent architectures. Our algorithm
has attention in the opposite direction. The competition is not between the lower-level capsules that
a higher-level capsule might attend to. It is between the higher-level capsules that a lower-level
capsule might send its vote to.

7相关工作
最近多次尝试提高神经网络处理视点的能力
变化，有两个主要流。一个流尝试实现视点不变性
另一个目标是视点等同性。 Jaderberg等人提出的工作。 （2015）），Spatial
变压器网络通过改变CNN的采样来寻求视点不变性
到选择仿射变换。 De Brabandere等人。 （2016年）扩展了空间变压器
根据输入推理过滤器的网络。他们产生不同的
为特征映射中的每个位置过滤，而不是对所有位置应用相同的转换
过滤器。他们的方法是从传统模式匹配向输入协方差检测迈出的一步
像标准CNN这样的框架（LeCun et al。（1990））。 Dai et al。 （2017年）改善空间
变压器网络通过推广滤波器的抽样方法。我们的工作差别很大
因为根据过滤器的匹配分数（固定或动态），单位不会被激活
在推断期间改变）。在我们的例子中，只有在转换后的姿势到来时才会激活胶囊
从下面的图层互相匹配。这是捕获协方差和线索的更有效方法
到具有更少泛化参数的模型更好。
CNN的成功促使许多研究人员扩展了所建立的平移等价性
包括旋转等变量（Cohen＆Welling（2016），Dieleman等（2016），
Oyallon＆Mallat（2015））。 Harmonic Networks最近的做法（Worrall et al。（2017））
通过使用圆谐波滤波器实现旋转等变特征映射并返回两者
最大响应和使用复数的方向。这共享基本代表性
胶囊的想法：通过假设一个位置只有一个实体实例，我们可以
使用几个不同的数字来表示它的属性。他们使用固定数量的流
轮换订单。通过沿着任何路径执行轮换顺序总和的等式，它们可以实现
补丁式旋转等变性。这种方法比数据增强更具参数效率
方法，复制特征地图或复制过滤器（Fasel＆Gatica-Perez（2006），Laptev
等人。 （2016））。我们的方法编码一般视点等变量而不是仅仿射二维
旋转。对称网络（Gens＆Domingos（2014））使用迭代Lucas-Kanade优化
找到最低级别功能支持的姿势。他们的关键弱点是，
迭代算法始终始于相同的姿势，而不是自下而上的投票的意思。

Lenc和Vedaldi（2016）提出了一个与仿射等同的特征检测机制（DetNet）
TRANSF


7.1 PREVIOUS WORK ON CAPSULES
Hinton et al. (2011) used a transformation matrix in a transforming autoencoder that learned to
transform a stereo pair of images into a stereo pair from a slightly different viewpoint. However,
that system requires the transformation matrix to be supplied externally. More recently, routing-byagreement
was shown to be effective for segmenting highly overlapping digits (Sabour et al. (2017)),
but that system has several deficiencies that we have overcome in this paper:
1. It uses the length of the pose vector to represent the probability that the entity represented by
a capsule is present. To keep the length less than 1, requires an unprincipled non-linearity
and this prevents the existence of any sensible objective function that is minimized by the
iterative routing procedure.
2. It uses the cosine of the angle between two pose vectors to measure their agreement. Unlike
the negative log variance of a Gaussian cluster, the cosine saturates at 1, which makes it
insensitive to the difference between a quite good agreement and a very good agreement.
3. It uses a vector of length n rather than a matrix with n elements to represent a pose, so its
transformation matrices have n
2 parameters rather than just n.

7.1以前的胶囊工作
Hinton等人（2011）在学习过的变换自编码器中使用了变换矩阵
将一对立体图像从略微不同的视点转换为立体对。然而，
该系统需要从外部提供变换矩阵。最近，路由 - 通过协议
被证明对分割高度重叠的数字是有效的（Sabour等（2017）），
但是这个系统有几个缺陷，我们在本文中已经解决了这个问题：
它使用姿态向量的长度来表示由实体表示的实体的概率
存在胶囊。要保持长度小于1，需要一个无原则的非线性
并且这可以防止任何由此最小化的明智的目标函数的存在
迭代路由过程。
它使用两个姿态矢量之间角度的余弦来衡量它们的一致性。不像
高斯簇的负对数方差，余弦在1处饱和，这使得它
对相当好的协议和非常好的协议之间的区别不敏感。
它使用一个长度为n的矢量，而不是一个有n个元素的矩阵来表示一个姿势，所以它是
变换矩阵有n个
2个参数而不仅仅是n。

8 CONCLUSION
Building on the work of Sabour et al. (2017), we have proposed a new type of capsule system in
which each capsule has a logistic unit to represent the presence of an entity and a 4x4 pose matrix
to represent the pose of that entity. We also introduced a new iterative routing procedure between
capsule layers, based on the EM algorithm, which allows the output of each lower-level capsule
to be routed to a capsule in the layer above in such a way that active capsules receive a cluster of
similar pose votes. This new system achieves significantly better accuracy on the smallNORB data
set than the state-of-the-art CNN, reducing the number of errors by 45%. We have also shown it to
be significantly more robust to white box adversarial attacks than a baseline CNN.
SmallNORB is an ideal data-set for developing new shape-recognition models precisely because it
lacks many of the additional features of images in the wild. Now that our capsules model works
well on NORB, we plan to implement an efficient version to test much larger models on much larger
data-sets such as ImageNet.
ACKNOWLEDGMENTS Thanks to Robert Gens, Eric Langlois, Taco Cohen and anonymous
commentators for helpful discussions and to everyone who made TensorFlow.

8结论
以Sabour等人的工作为基础。 （2017年），我们提出了一种新型胶囊系统
其中每个胶囊具有用于表示实体和4×4姿态矩阵的存在的逻辑单元
以表示该实体的姿态。我们还介绍了一种新的迭代路由程序
基于EM算法的胶囊层，其允许每个较低级别胶囊的输出
被路由到上面的层中的胶囊，使得活性胶囊接收一簇
类似的姿势投票。这个新系统在smallNORB数据上实现了更高的精度
比最先进的CNN设置，减少了45％的错误数量。我们也展示了它
对于白盒对抗性攻击比基线CNN更强大。
SmallNORB是开发新型形状识别模型的理想数据集，因为它恰恰是
缺乏野外图像的许多附加功能。现在我们的胶囊模型起作用了
在NORB上，我们计划实施一个高效的版本来测试更大的模型
数据集如ImageNet。
致谢感谢Robert Gens，Eric Langlois，Taco Cohen和匿名
有帮助的讨论评论员和每个制作TensorFlow的人。
