# SNN-snntorch-classification

主题：构建脉冲神经网络，实现分类任务。

具体实现：实现了一个监督学习算法，利用标准静态MNIST数据集，使用梯度下降训练多层全连接脉冲神经网络，实现手写数字图像的10分类。

实现平台：pytoch、snnTorch

一、背景知识

（1）snnTorch

snnTorch是一个Python包，用于脉冲神经网络执行基于梯度的学习。它以PyTorch为基础，并扩展了PyTorch的功能。它能够利用GPU加速张量的计算，并将其应用于脉冲神经元网络。snntorch包含了脉冲神经元库（snntorch）、代理梯度函数库（snntorch.surrogate）、反向传播库（snntorch.backprop）、神经形态的数据集（snntorch.spikevision）等多种库，便于各类脉冲神经网络的构建和训练。

（2）神经元模型

1.Hodgkin-Huxley（HH）模型

HH模型是一组描述神经元细胞膜的电生理现象的非线性微分方程，直接反映了细胞膜上离子通道的开闭情况。HH模型因为精确地描绘了膜电压的生物特性，因此能很好地与生物神经元的电生理实验结果相吻合，但是它的运算量较高，难以实现大规模神经网络的实时仿真。

2.人工神经元模型

即绝大多数神经网络应用的神经元模型，它的工作原理是输入乘以相应的权重并通过激活函数。这种简化使深度学习在计算机视觉、自然语言处理和许多其他机器学习领域的任务中实现了相当好的效果。

3.Leaky Integrate and Fire(LIF)模型

为了解决HH模型运算量的问题，有学者提出了一种简化的生物学模型，即LIF模型，这也是本次建模中使用到的神经元模型。它将细胞膜的电特性看成电阻和电容的组合，随着时间的推移对输入进行积分（很像 RC 电路），如果积分值超过阈值，LIF 神经元就会发出电压尖峰。如下图是一个阶跃输入刺激下的LIF神经元的充放电过程和脉冲输出。

<img width="180" alt="image" src="https://user-images.githubusercontent.com/82455250/203706870-a608d915-be1a-422a-8a09-aa4a2437eabf.png">


（3）针对脉冲的不可微性的反向传播算法

如下图可以看到，由于LIF神经元的输出不是具体的数值而是脉冲，因此在反向传播时会出现不可微的情况。即在求导时，S对U的偏导总是0或者无穷大。为了克服这种问题，采用了代理梯度的方法对其进行反向求梯度。

<img width="131" alt="image" src="https://user-images.githubusercontent.com/82455250/203707080-9b0c4227-ab95-48e9-8f7e-c857699edeb5.png">
<img width="193" alt="image" src="https://user-images.githubusercontent.com/82455250/203707090-8bc2fa3a-ad94-4936-8a59-8e6f591d845a.png">


Spike Operator算法将梯度计算分为两部分：如果S不是脉冲，那么其梯度项是0；如果S是脉冲，那么梯度项是1。利用这种方法可以对S的梯度进行近似。

（4）时间的反向传播

由于脉冲神经元的输入输出和时间有着密切的关系，即数据只有在时间步上累计才是有意义的，因此采用反向传播时间（BPTT）算法计算损失，它可以计算指定时间步内的后代的梯度并将它们加在一起。

<img width="156" alt="image" src="https://user-images.githubusercontent.com/82455250/203707186-fbfeb883-9a80-41bd-8529-ea1e71f30e82.png">

二、网络构建

（1）网络结构的设置

利用LIF神经元构建类似BP算法的全连接网络SNN，LIF神经元的作用类似于人工神经元，即对数据加权求和并做非线性激活，只不过其非线性的输出是一个个脉冲。如果膜超过阈值，那么神经元会发出一个输出脉冲信号：
<img width="148" alt="image" src="https://user-images.githubusercontent.com/82455250/203707273-e266b4c5-3e72-4c68-b874-e7179d59d724.png">

 
如果触发脉冲，则应重置膜电位。减法重置机制的模型如下：
<img width="172" alt="image" src="https://user-images.githubusercontent.com/82455250/203707287-09edad96-47a3-4d6b-aa35-06ec34caf75e.png">


其中，W是可学习的参数，Beta是衰减率，也是我们对LIF神经元唯一要设置的超参数（阈值U通常为1	）

其余部分类似于BP网络，需要输入层，隐藏层和输出层。本次建模构建了一个三层全连接网络，对于每一批数据的输入，设置一定长度的时间步对LIF神经元进行计算，并采用代理梯度的方法计算反向梯度。

（2）损失函数的构建

由于SNN网络的输出是一组脉冲（在指定的时间步下），因此其分类的标准是，取放电率（或脉冲数）最高的神经元作为预测类。基于这一点，可以设置如下损失函数：
<img width="217" alt="image" src="https://user-images.githubusercontent.com/82455250/203707577-0cdb940d-1384-404c-b6a7-77b21b93d16d.png">


它的含义是是鼓励正确类别对应的输出神经元结点膜电位增加，而错误类别对应的输出神经元结点膜电位降低。实际上，这意味着鼓励神经元结点在每一个时间步下都触发正确类，同时抑制错误类。
该损失函数应用于每个时间步，因此也会在时间t上产生损失。因此，需要在这一批数据训练结束时将这些损失加在一起：
<img width="95" alt="image" src="https://user-images.githubusercontent.com/82455250/203707594-468b5578-c39f-43c1-ba08-8b769b4648c1.png">


三、训练结果

数据集采用了MNIST标准数据集（60000训练集、10000测试集）。
采用梯度反向传播的方法对网络进行训练。梯度的计算采用了代理梯度的方法（snntorch对LIF脉冲神经元已经内置，不用再额外设置），权重的更新利用了Adam优化器进行计算。通过设置网络隐藏层结点个数，调整网络超参数（LIF结点的衰减率beta、时间步长）和训练超参数（学习率lr、迭代代数epoch、batchsize）得到最佳的网络性能。

（1）不同衰减率下的训练效果

<img width="235" alt="image" src="https://user-images.githubusercontent.com/82455250/203707625-9d478659-4e91-4a9b-82ca-1deb286cbb83.png">

图1 Beta=0.5

<img width="235" alt="image" src="https://user-images.githubusercontent.com/82455250/203707857-3817070f-7e27-42db-b6ea-92e8417bc802.png">

图2 Beta=0.7

<img width="228" alt="image" src="https://user-images.githubusercontent.com/82455250/203708013-1f293242-5b27-444c-ae72-9ef3032f8874.png">


图3 Beta=0.9

可以发现提高衰减率对于网络有更好的效果。

（2）不同隐藏层结点个数的训练效果：

<img width="240" alt="image" src="https://user-images.githubusercontent.com/82455250/203706239-a5ce01a6-613a-4c25-860b-8272df7aa15b.png">

图4 Node=200

<img width="242" alt="image" src="https://user-images.githubusercontent.com/82455250/203708078-ad112f4f-ead8-401c-9106-f1c0530f9fd2.png">

图5 Node=500

<img width="244" alt="image" src="https://user-images.githubusercontent.com/82455250/203708115-3878fb93-0b1a-4774-8bad-1f0a9bf65849.png">

图6 Node=1000

提高结点个数在一定程度上能够更好的拟合数据，但到一定程度，提高结点个数也不会优化网络，反而增加了训练计算时间。
最终，在beta=0.95，学习率为1e-3,结点个数为500时有着最优的网络性能。在10000张测试集上可以达到95.78%的准确率。

<img width="281" alt="image" src="https://user-images.githubusercontent.com/82455250/203708188-76e801ee-6a2f-41d6-82c2-e3cd3a555ec5.png">


四、小结

通过本次建模练习，我充分理解了基于生物学基础的LIF脉冲神经元的原理及它在神经网络的构建中是如何应用的。虽然本次建模实现的功能并不复杂，但是却是计算神经科学在利用神经网络实现感知功能的一次很好的应用。目前绝大部分用于图像分类的网络都是利用人工神经元进行的，输入静态图像，通过比较神经元结点的输出，实现分类；LIF神经元的输出是一个个离散的脉冲，因此如果输入是一个静态的数据集，需要在一定的时间步上进行输出脉冲数量的累加，来作为它们的输出，这是与一般神经元最不同的特征之一。实验结果表明，即使是三层全连接这样简单的网络，用LIF神经元代替人工神经元也能取得很好的分类效果。

如果要实现更加复杂的数据集的分类，还可以将SNN与卷积神经网络结合起来，进行更加充分和有效的特征提取，我将会继续进行这方面的探索和尝试。
