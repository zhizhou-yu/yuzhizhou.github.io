---
author: Zhizhou Yu
mode: immersive
header:
  theme: dark
article_header:
  type: overlay
  theme: dark
  background_color: '#203028'
  background_image:
    gradient: 'linear-gradient(135deg, rgba(34, 139, 87 , .4), rgba(139, 34, 139, .4))'
    src: /assets/header.jpg
title: "Self-supervised Learning: Generative or Contrastive 自监督学习2020综述"
date: 2020-06-26 12:00:00
tags: self-supervised-learning
key: 2020-06-27-Self-supervised-Learning-Survey
---

>这是一篇清华大学 [Tang Jie](https://scholar.google.com/citations?user=n1zDCkQAAAAJ&hl=en&oi=ao) 等人在 2020 年 6 月提出的自监督学习（self-supervised learning）最新的综述文章，本文进行简单的内容梳理。


<!--more-->
近十年来，深度监督学习取得了巨大的成功。然而，它依赖于手工标签，并且易受攻击的弱点促使学者们探索更好的解决方案。近年来，**自监督学习**作为一种新的学习方法，在表征学习方面取得了骄人的成绩并吸引了越来越多的注意。自监督表示学习利用输入数据本身作为监督信号，几乎有利于所有不同类型的下游任务。本文主要介绍自监督学习在**计算机视觉，自然语言处理，和图学习（Graph Learning）**领域的方法，并对现有的方法进行了全面的回顾，根据不同方法的目标归纳为生成式（generative）、对比式（contrastive）和对比-生成式 / 对抗式（adversarial）三大类。最后，简要讨论了自监督学习的开放问题和未来的发展方向。

* TOC
{:toc}

## Introduction

监督学习在过去取得了巨大的成功，然而监督学习的研究进入了瓶颈期，因其依赖于昂贵的人工标签，却饱受泛化错误、伪相关和对抗攻击（generalization error、spurious correlations and adversarial attacks）的困扰。自监督学习以其良好的数据利用效率和泛化能力引起了人们的广泛关注。本文将全面研究最新的自监督学习模型的发展，并讨论其理论上的合理性，包括**预训练语言模型（Pretrained Language Model，PTM）、生成对抗网络（GAN）、自动编码器及其拓展、最大化互信息（Deep Infomax，DIM）以及对比编码（Contrastive Coding）**。

自监督学习与无监督学习的区别主要在于，无监督学习专注于检测特定的数据模式，如聚类、社区发现或异常检测，而自监督学习的目标是恢复（recovering），仍处于监督学习的范式中。下图 1 展示了两者之间的区别，自监督中的“related information” 可以来自其他模态、输入的其他部分以及输入的不同形式。

![不同类型的学习之间差异]({{ '/assets/images/differences-of-supervised-unsupervised-and-self-supervised-learning.png' | relative_url }})
{: style="width: 50%; margin: 0 auto;"}
***Fig. 1.** 监督、无监督和自监督学习的区别. (Image source: [Jie Tang et al.](https://arxiv.org/pdf/2006.08218))*

## Generative Self-supervised Learning 生成式自监督学习

### Auto-regressive (AR) Model
在自回归模型中，联合分布可以被分解为条件的乘积，每个变量概率依赖于之前的变量：

$$
\max _{\theta} p_{\theta}(\mathbf{x})=\sum_{t=1}^{T} \log p_{\theta}\left(x_{t} \mid \mathbf{x}_{1: t-1}\right)
$$

在自然语言处理中，自回归语言模型的目标通常是最大化正向自回归因子分解的似然。例如 GPT、GPT-2 使用 Transformer 解码器结构进行建模；在计算机视觉中，自回归模型用于逐像素建模图像，例如在 PixelRNN 和 PixelCNN 中，下方(右侧)像素是根据上方(左侧)像素生成的；而在图学习中，则可以通过深度自回归模型来生成图，例如 GraphRNN 的目标为最大化观察到的图生成序列的似然。

自回归模型的优点是能够很好地建模上下文依赖关系。然而，其缺点是每个位置的 token 只能从一个方向访问它的上下文。

### Auto-encoding (AE) Model
自动编码模型的目标是从(损坏的)输入中重构(部分)输入。标准的自动编码器，首先一个编码器 $$h = f_{enc}(x)$$ 对输入进行编码得到隐表示 $$h$$，再通过解码器 $$x' = f_{dec}(h)$$ 重构输入，目标是使得输入 $$x$$ 和 $$x'$$ 尽可能的相近。

- **1. Denoising AE，去噪自动编码器**
  
  Denoising AE 认为表示应该对引入的噪声具有鲁棒性，MLM（masked language model）就可以看作是去噪自动编码器模型，MLM 通过对序列中的 token 随机替换为 mask token，并根据上下文预测它们。同时为了解决下游任务中输入不存在 mask token 的问题，在实际使用中，替换 token 时可以选择以一定概率随机替换为其他单词或是保持不变。与自回归模型相比，在 MLM 模型中，所预测的 token 可以访问双向的上下文信息。然而，MLM 假设预测的 token 是相互独立的。


- **2. Variational AE，变分自动编码器**
  
  变分自动编码模型假设数据是从潜在变量表示中生成的。给定数据 $X$，潜在变量 $$Z=\{z_1,z_2,...,z_n\}$$ 上的后验分布 $p(z\|x)$ 通过 $q(z\|x)$ 来逼近。根据变分推断：
  
  $$
  \log p(x) \geq-D_{K L}(q(z \mid x) \| p(z))+\mathbb{E}_{\sim q(z \mid x)}[\log p(x \mid z)]$$  
  
  从自动编码器视角来看，第一项可以看作是正则化项保证后验分布 $q(z\|x)$ 逼近先验分布 $p(z)$，第二项就是根据潜在变量重构输入的似然。VAE 在图像生成领域中有所应用，代表模型有向量量化 VAE 模型 VQ-VAE，而在图学习领域，则有变分图自动编码器 VGAE。


### Hybrid model
结合 AR 模型和 AE 模型的各自优点，MADE 模型对 AE 做了一个简单的修改，使得 AE 模型的参数遵循 AR 模型的约束。具体来说，对于原始的 AE，相邻两层之间的神经元通过 MLPs 完全连接。然而，在 MADE 中，相邻层之间的一些连接被 mask 了，以确保输入的每个维度仅从之前的维度来重构。

![PLM]({{ '/assets/images/PLM.png' | relative_url }})
{: style="width: 75%; margin: 0 auto;"}
***Fig. 2.** PLM实例示意图. (Image source: [Z Yang et al.](https://papers.nips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf))*

在 NLP 领域中，PLM（Permutation Language Model ）模型则是一个代表性的混合方法，XLNet 引入 PLM，是一种生成自回归预训练模型。XLNet 通过最大化因数分解顺序的所有排列的期望似然来实现学习双向上下文关系（见图 2）。具体的，假设 $$\mathcal{Z}_{\mathcal{T}}$$ 是长度为 $$T$$ 的所有可能的序列排序，则 PLM 的优化目标为：

$$
\max _{\theta} \mathbb{E}_{\mathbf{z} \sim \mathcal{Z}_{\mathcal{T}}}\left[\sum_{t=1}^{T} \log p_{\theta}\left(x_{z_{t}} \mid \mathbf{x}_{\mathbf{z}_{<t}}\right)\right]
$$

实际上，对于每个文本序列，都采样了不同的分解因子。因此，每个 token 都可以从双向看到它的上下文信息。在此基础上，XLNet 还对模型进行了位置重参数化，使模型知道需要预测的位置，并引入了特殊的双流自注意机制来实现目标感知预测。此外，与 BERT 不同的是，XLNet 受到 AR 模型最新改进的启发，提出了结合片段递归机制和相对位置编码机制的 [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf?fbclid=IwAR3nwzQA7VyD36J6u8nEOatG0CeW4FwEU_upvvrgXSES1f0Kd-xGTS0MFfY) 集成到预训练中，比 Transformer 更好地建模了长距离依赖关系。

## Contrastive Self-supervised Learning 对比式自监督学习

大多数表示学习任务都希望对样本 x 之间的关系建模。因此长期以来，人们认为生成模型是表征学习的唯一选择。然而，随着 Deep Infomax、MoCo 和 SimCLR 等模型的提出，**对比学习（contrastive learning）**取得了突破性进展，如图 3 所示，在 ImageNet 上，自监督模型的 Top-1 准确率已经接近监督方法（ResNet50），揭示了判别式模型的潜力。

![Contrastive learning]({{ '/assets/images/Contrastive-Learning.png' | relative_url }})
{: style="width: 60%; margin: 0 auto;"}
***Fig. 3.** 自监督学习模型在 ImageNet 上 Top-1 准确率对比. (Image source: [Jie Tang et al.](https://arxiv.org/pdf/2006.08218))*

对比学习的宗旨在于“learn to compare”，通过设定噪声对比估计（Noise Contrastive Estimation，NCE）来实现这个目标：

$$
\mathcal{L}=\mathbb{E}_{x, x^{+}, x^{-}}\left[-\log \left( \frac{e^{f(x)^{T} f\left(x^{+}\right)}}{e^{f(x)^{T} f\left(x^{+}\right)}+e^{f(x)^{T} f\left(x^{-}\right)}}\right)\right]
$$

其中，$x^{+}$ 与 $x$ 相似，而 $x^{-}$ 与 $x$ 不相似，$f$ 是编码器函数（表示学习方程）。相似度量和编码函数可能会根据具体的任务有所不同，但是整体的框架仍然类似。当有更多的不相似样本对时，我们可以得到 InfoNCE：

$$
\mathcal{L}=\mathbb{E}_{x, x^{+}, x^{k}}\left[-\log \left( \frac{e^{f(x)^{T} f\left(x^{+}\right)}}{e^{f(x)^{T} f\left(x^{+}\right) }+ \sum_{k=1}^{K} e^{f(x)^{T} f\left(x^{k}\right)}}\right)\right]
$$

在这里，根据最近的对比学习模型框架，具体可细分为两大类：实例-上下文对比（context-instance contrast）和实例-实例对比（instance-instance contrast）。他们都在下游任务中有不俗的表现，尤其是分类任务。

### Context-Instance Contrast 
Context-Instance Contrast，又称为 Global-local Contrast，重点建模样本局部特征与全局上下文表示之间的归属关系，其初衷在于当我们学习局部特征的表示时，我们希望它与全局内容的表示相关联，比如条纹与老虎，句子与段落，节点与相邻节点。

这一类方法又可以细分为两大类：预测相对位置（Predict Relative Position，PRP）和最大化互信息（Maximize Mutual Information，MI）。**两者的区别在于：1）**PRP 关注于学习局部组件之间的相对位置。全局上下文是预测这些关系的隐含要求(例如，了解大象的长相对于预测它头和尾巴的相对位置至关重要)；**2）**MI 侧重于学习局部组件和全局上下文之间的明确归属关系。局部之间的相对位置被忽略。

- **1. Predict Relative Position**

  许多数据蕴含丰富的时空关系，例如图 4 中，大象的头在其尾巴的右边，而在文本中，“Nice to meet you” 很可能出现在“Nice to meet you, too”的前面。许多模型都把识别其各部分之间的相对位置作为辅助任务（pretext task），例如图 4 中，预测两个 patch 之间的相对位置，或是预测图像的旋转角，又或者是“解拼图”。

  ![Contrastive learning]({{ '/assets/images/pretext-task.png' | relative_url }})
  {: style="width: 67%; margin: 0 auto;"}
  ***Fig. 4.** Pretext Task示例. (Image source: [Jie Tang et al.](https://arxiv.org/pdf/2006.08218))*

  在 NLP 领域中，BERT 率先使用的 NSP（Next Sentence Prediction）也属于类似的辅助任务，NSP 要求模型判断两个句子是否具有上下文关系，然而最近的研究方法证明 NSP 可能对模型没什么帮助甚至降低了性能。为了替换 NSP，ALBERT 提出了 SOP（Sentence Order Prediction）辅助任务，其认为 NSP 中随机采样的句子可能来自不同的文章，主题不同则难度自然很低，而 SOP 中，两个互换位置的句子被认为是负样本，这使得模型将会集中在语义的连贯性学习。

- **2. Maximize Mutual Information**
  
  MI 类任务利用统计学中的互信息，通常来说，模型的优化目标为：

  $$
  \max _{g_{1} \in \mathcal{G}_{1}, g_{2} \in \mathcal{G}_{1}} I\left(g_{1}\left(x_{1}\right), g_{2}\left(x_{2}\right)\right)
  $$

  其中，$g_i$ 表示表示编码器，$\mathcal{G}_{i}$ 则表示满足约束的一类编码器，$I(\cdot, \cdot)$ 表示对于真实互信息的采样估计。在应用中，MI 计算困难，一个常见的做法是通过 NCE 来最大化互信息 $I$ 的下界。

  Deep Infomax(DIM) 是第一个通过对比学习任务显式建模互信息的算法，它最大化了局部 patch 与其全局上下文之间的 MI。在实际应用中，以图像为例，我们可以把一张狗的图像 $x$ 编码为 $f(x) \in \mathbb{R}^{M\times M \times d}$，随后取出一个局部向量 $v \in \mathbb{R}^d$，为了进行实例和上下文之间的对比:

    - 首先需要通过一个总结函数（summary function） $g: \mathbb{R}^{M\times M \times d} \rightarrow \mathbb{R}^d$，来生成上下文向量 $s=g(f(x)) \in \mathbb{R}^d$
    {:.p-1}
    - 其次需要一个猫图像作为 $x^-$，并得到其上下文向量 $s^- = g(f(x^-))$
    {:.p-1}

  这样就可以得到对比学习的目标函数：

  $$
  \mathcal{L}=\mathbb{E}_{v, x}\left[-\log \left(\frac{e^{v^{T} \cdot s}}{e^{v^{T \cdot s}}+e^{v^{T} \cdot s^{-}}}\right)\right]
  $$

  Deep Infomax 给出了自监督学习的新范式，这方面工作较有影响力的跟进，最先有在 speech recognition 领域的 CPC 模型，CPC 将音频片段与其上下文音频之间的互信息最大化。为了提高数据利用效率，它需要同时使用几个负的上下文向量，CPC 随后也被应用到图像分类中。AMDIM 模型提出通过随机生成一张图像的不同 views（截断、变色等）用于生成局部特征向量和上下文向量，其与 Deep Infomax 的对比如下图 5 所示，Deep Infomax 首先通过编码器得到图像的特征图，随后通过 readout（summary function）得到上下文向量，而 AMDIM 则通过随机选择图像的另一个 view 来生成上下文向量。

  ![Deep Infomax and AMDIM]({{ '/assets/images/deep-infomax-and-amdim.png' | relative_url }})
  {: style="width: 50%; margin: 0 auto;"}
  ***Fig. 5.** Deep Infomax 与 AMDIM 示意图. (Image source: [Jie Tang et al.](https://arxiv.org/pdf/2006.08218))*


  在 NLP 领域，InfoWord 模型提出最大化句子的整体表示和句子的 n-grams 之间的互信息。而在图学习领域则有 DGI（Deep Graph Infomax）模型，DGI 以节点表示作为局部特征，以随机采样的二跳邻居节点的均值作为上下文向量，注意图学习中的负样本难以构造，DGI 提出保留原有上下文节点的结构但打乱顺序来生成负样本。

  ![Deep Graph Infomax]({{ '/assets/images/deep-graph-infomax.png' | relative_url }})
  {: style="width: 60%; margin: 0 auto;"}
  ***Fig. 6.** Deep Graph Infomax 示意图. (Image source: [Jie Tang et al.](https://arxiv.org/pdf/2006.08218))*
  

### Instance-Instance Contrast 
尽管基于最大化互信息的模型取得了成功，但是一些最近的研究对优化 MI 带来的实际增益表示怀疑。[M Tschannen et al.](https://arxiv.org/pdf/1907.13625)证明基于最大化互信息的模型的成功与 MI 本身联系并不大。相反，它们的成功应该更多地归因于编码器架构和以及与度量学习相关的负采样策略。度量学习的一个重点是在提高负采样效率的同时执行困难样本采样（hard positive sampling），而且它们对于基于最大化互信息的模型可能发挥了更关键的作用。

而最新的相关研究 [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf) 和 [SimCLR](https://arxiv.org/pdf/2002.05709) 也进一步证实了上述观点，它们的性能优于 context-instance 类方法，并通过 instance-instance 级的直接对比，相比监督方法取得了具有竞争力的结果。

- **1. Cluster-based Discrimination**
  
  instance-instance contrast 类的方法最早的研究方向是基于聚类的方法，DeepCluster 取得了接近 AlexNet 的成绩。图像分类任务要求模型正确地对图像进行分类，并且希望在同一个类别中图像的表示应该是相似的。因此，目标是在嵌入空间内保持相似的图像表示相接近。在监督学习中，这一目标通过标签监督来实现; 然而，在自监督学习中，由于没有标签，DeepCluster 通过聚类生成伪标签，再通过判别器预测伪标签。

  最近，局部聚合（Local Aggregation, LA）方法已经突破了基于聚类的方法的边界。LA 指出了 DeepCluster 算法的不足，并进行了相应的优化。首先，在 DeepCluster 中，样本被分配到互斥的类中，而 LA 对于每个样本邻居的识别是分开的。第二，DeepCluster 优化交叉熵的判别损失，而 LA 使用一个目标函数直接优化局部软聚类度量。这两个变化在根本上提高了 LA 表示在下游任务中的性能。

  ![DeepCluster and LA]({{ '/assets/images/deepcluster-and-la.png' | relative_url }})
  {: style="width: 60%; margin: 0 auto;"}
  ***Fig. 7.** DeepCluster 和 LA 方法示意图. (Image source: [Jie Tang et al.](https://arxiv.org/pdf/2006.08218))*

  基于聚类的判别也可以帮助预训练模型提升泛化能力，更好地将模型从辅助任务目标转移到实际任务中。传统的表示学习模型只有两个阶段: 一个是预训练阶段，另一个是评估阶段。ClusterFit 在上述两个阶段之间引入了一个与 DeepCluster 相似的聚类预测微调阶段，提高了表示在下游分类评估中的性能。

- **2. Instance Discrimination**
  
  以实例判别（instance discrimination）作为辅助任务的模型原型是 InstDisc，在其基础之上，CMC 提出将一幅图像的多个不同 view 作为正样本，将另一幅图像作为负样本。在嵌入空间中，CMC 将使一幅图像的多个 view 尽可能靠近，并拉开与其他样本的距离。然而，它在某种程度上受到了 Deep InfoMax 思想的限制，仅仅对每个正样本采样一个负样本。

  在 [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf) 中，通过 momentum contrast 进一步发展了实例判别的思想，MoCo 极大的增加了负样本的数量。对于一个给定的图像 $x$，MoCo 希望通过一个 query encoder $f_q(\cdot)$ 得到一个可以区分于任何其他图像的 instinct 表示 $q=f_q(x)$。因此，对于其他图像集中的 $x_i$，异步更新 key encoder $f_k(\cdot)$ 并得到 $k_+ = f_k(x)$ 以及 $k_i = f_k(x_i)$，并优化下面的目标函数：

  $$
  \mathcal{L}=-\log \frac{\exp \left(q \cdot k_{+} / \tau\right)}{\sum_{i=0}^{K} \exp \left(q \cdot k_{i} / \tau\right)}
  $$

  其中，$K$ 表示负样本的数量，这个式子就是 InfoNCE 的一种表达。同时，MoCo 还提出了其他两个在提升负采样效率方面至关重要的思想：

    - 首先，摒弃了传统的端到端训练，采用带有两个编码器（query and key encoder）的 momentum contrast 学习来避免训练初始阶段的损失的波动。
    - 其次，为了扩大负样本的容量，MoCo 采用队列（K=65536）的方式将最近编码过的批样本保存为负样本。这大大提高了负采样效率。
  
  ![moco]({{ '/assets/images/moco.png' | relative_url }})
  {: style="width: 100%; margin: 0 auto;"}
  ***Fig. 8.** MoCo 示意图. (Image source: [Kaiming He et al.](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf))*

  MoCo 还采用了一些辅助的技术来保证训练的收敛性，如批随机化以及 temperature 超参数 $\tau$ 的调整。然而，MoCo 采用了一种过于简单的正样本策略：正对表示来自同一样本，没有任何变换或增强，使得正对太容易区分。PIRL 添加了上述提到的”解拼图“的困难正样本（hard positive example）增强。为了产生一个 pretext-invariant 的表示，PIRL 要求编码器把一个图像和它的拼图作为相似的对。

  在 [SimCLR](https://arxiv.org/pdf/2002.05709) 中，作者通过引入 10 种形式的数据增强进一步说明了困难正样本（hard positive example）策略的重要性。这种数据增强类似于 CMC，它利用几个不同的 view 来增加正对。SimCLR 遵循端到端的训练框架，而不是 MoCo 的 momentum contrast，为处理大规模负样本问题，SimCLR 选择将 batch 大小 N 扩展为 8196。`有钱任性!`{: .error}

  ![simclr]({{ '/assets/images/simclr.png' | relative_url }})
  {: style="width: 75%; margin: 0 auto;"}
  ***Fig. 9.** SimCLR 数据增强示意图. (Image source: [G Hinton et al.](https://arxiv.org/pdf/2002.05709))*


  具体来看，minibatch 中的 N 个样本将会被增强为 2N 个样本 $\hat{x}_j(j=1,2,...,2N)$。对于一个正样本对 $\hat{x}_i$ 和 $\hat{x}_j$，其他的 2(N-1) 个样本将会被视为负样本，我们可以得到成对对比损失 NT-Xent loss：

  $$
  l_{i, j}=-\log \frac{\exp \left(\operatorname{sim}\left(\hat{x}_{i}, \hat{x}_{j}\right) / \tau\right)}{\sum_{k=1}^{2 N} \mathbb{I}_{[k \neq i]} \exp \left(\operatorname{sim}\left(\hat{x}_{i}, \hat{x}_{k}\right) / \tau\right)}
  $$

  注意，$l_{i, j}$ 是非对称的，此处 $sim(\cdot, \cdot)$ 为正弦函数可以归一化表示，最终的和损失为：

  $$
  \mathcal{L}=\frac{1}{2 N} \sum_{k=1}^{N}\left[l_{2 i-1,2 i}+l_{2 i, 2 i-1}\right]
  $$

  SimCLR 还提供了一些其他有用的技术，包括 representation 和 contrast loss 之间的可学习的非线性转换，更多的训练步数，和更深的神经网络。[Kaiming He et al.](https://arxiv.org/pdf/2003.04297) 进行消融分析研究表明，SimCLR 中的技术也可以进一步提高 MoCo 的性能。
  
  在图学习中，图对比编码（Graph Contrastive Coding, GCC）率先将实例判别作为结构信息预训练的辅助任务。对于每个节点，通过随机游走独立采样两个不同的子图，并使用它们的归一化图 Laplacian 矩阵中的 top 特征向量作为节点的初始表示，然后使用 GNN 对它们进行编码，并像 MoCo 和 SimCLR 那样计算 InfoNCE loss，其中相同节点(在不同的子图中)的节点嵌入被视为相似正样本，其余的视为不相似的负样本。结果表明，GCC 比以前的工作，如 struc2vec、GraphWave 和 ProNE，学习了更好的可迁移的结构知识。


## Generative-Contrastive (Adversarial) Self-supervised Learning 对比-生成式（对抗式）自监督学习

### Why Generative-Contrastive (Adversarial)?
生成式模型往往会优化下面的目标似然方程：

$$
\mathcal{L}_{M L E}=-\sum_{x} \log p(x \mid c)
$$

其中，$x$ 是我们希望建模的样本，$c$ 则是条件约束例如上下文信息，这个方程一般都会使用最大似然估计 MLE 来优化，然而 MLE 存在两个缺陷：

  - **Sensitive and Conservative Distribution（敏感的保守分布）**：当 $p(x \mid c) \rightarrow 0$ 时，$\mathcal{L}_{M L E}$ 会变得非常大，使得模型对于罕见样本会十分敏感，这会导致生成模型拟合的数据分布非常保守，性能受到限制。
  - **Low-level Abstraction Objective（低级抽象目标）**：在使用 MLE 的过程中，表示分布建模的对象是样本 $x$ 级别，例如图像中的像素，文本中的单词以及图中的节点。然而，大部分分类任务需要更高层次的抽象表示，例如目标检测、长段落理解以及社区分类。

以上两个缺陷严重的限制了生成式自监督模型的发展，而使用判别式或对比式目标函数可以很好的解决上述问题。以自动编码器和 GAN 为例，自动编码器由于使用了一个 pointwise 的 $L_2$ 重构损失，因此可能建模的是样本中的 pixel-level 模式而不是样本分布，而 GAN 利用了对比式目标函数直接对比生成的样本和真实样本，建模的是 semantic-level 从而避免了这个问题。

与对比学习不同，对抗学习仍然保留由编码器和解码器组成的生成器结构，而对比学习则放弃了解码器（如下图 10 所示）。这点区别非常重要，因为一方面，生成器将生成式模型所特有的强大的表达能力赋予了对抗学习; 另一方面，这也使得对抗式方法的学习目标相比与对比式方法更具挑战性，也导致了收敛不稳定。在对抗式学习的设置中，解码器的存在要求表示具有“重构性”，换句话说，表示应当包含所有必要的信息来构键输入。而在对比式学习的设置中，我们只需要学习“可区分的”信息就可以区分不同的样本。

![ae-gan-contrastive]({{ '/assets/images/ae-gan-contrastive.png' | relative_url }})
{: style="width: 67%; margin: 0 auto;"}
***Fig. 10.** AE、GAN 和 Contrastive 方法的区别，总体上它们都包含两个大的组件----生成器和判别器。生成器可以进一步分解为编码器和解码器，区别在于：1）隐编码表示 $z$ 在 AE 和 Contrastive 方法中是显式的；2）AE 没有判别器是纯生成式模型，Contrastive 方法的判别器参数少于 GAN；3）AE 学习目标是生成器，而 GAN 和 Contrastive 方法的学习目标是判别器。 (Image source: [Jie Tang et al.](https://arxiv.org/pdf/2006.08218))*

综上所述，对抗式方法吸收了生成式方法和对比式方法各自的优点，但同时也有一些缺点。在我们需要拟合隐分布的情况下，使用对抗式方法会是一个更好的选择。下面将具体讨论它在表示学习中的各种应用。

### Generate with Complete Input
本小节将介绍 GAN 及其变种如何应用到表示学习中，这些方法关注于捕捉样本的完整信息。GAN 在图像生成中占据主导地位，然而数据的生成与表示还是存在 gap 的，这是因为在 AE 和 Contrastive 方法中，隐编码表示 $z$ 是显式建模的，而 GAN 是隐式的，因此我们需要提取出 GAN 的隐分布 $p_z(z)$。

[AAE](https://arxiv.org/pdf/1511.05644.pdf%5D) 率先尝试弥补这个 gap，AAE 参考 AE 的思想，为了提取隐分布 $p_z(z)$，可以将 GAN 中的生成器替换为显式建模的 VAE，回忆一下前文提到的 VAE 损失函数：

$$
\mathcal{L}_{\mathrm{VAE}}=-\mathbb{E}_{q(z \mid x)}(-\log (p(x \mid z))+\mathrm{KL}(q(z \mid x) \| p(z))
$$

我们说过，AE 由于使用了 $L_2$ 损失导致建模对象可能是 pixel-level，而 GAN 使用的对比式损失可以更好地建模 high-level 表示，因此为了缓解这个问题，AAE 将上面的损失函数中的 KL 项替换为了一个判别损失：

$$
\mathcal{L}_{\text {Disc }}= \text{CrossEntropy} (q(z), p(z))
$$

它要求判别器区分来自编码器的表示和先验分布。尽管如此，AAE 仍然保留了重构损失与 GAN 的核心思想还是存在矛盾。基于 AAE，BiGAN 和 ALI 模型主张保留对抗性学习，并提出新的框架如下图 11 所示。

![bigan]({{ '/assets/images/bigan.png' | relative_url }})
{: style="width: 67%; margin: 0 auto;"}
***Fig. 11.** BiGAN 和 ALI 的框架图。 (Image source: [J Donahue et al.](https://arxiv.org/pdf/1605.09782))*

给定一个真实样本 $x$:

  - 生成器 $G$：这里生成器实际上扮演解码器的角色，其会根据先验隐分布 $z$ 生成 fake 样本 $x' = G(z)$
  - 编码器 $E$：新引入的单元，其将真实样本 $x$ 映射为表示 $z' = E(x)$，而这正是我们想要的表示
  - 判别器 $D$：给定输入 $[z, G(z)]$ 和 $[x, E(x)]$，判断哪一个来自真实的样本分布。

容易看出，训练的目标是 $E=G^{-1}$，即编码器 $E$ 应当学会”转换“生成器 $G$。看起来似乎和 AE 很像，但是区别在于编码表示的分布不作任何关于数据的假设，完全由判别器来决定，可以捕捉 semantic-level 的差异。

### Pre-trained Language Model
在很长一段时间内，预训练语言模型 PTM 都关注于根据辅助任务来最大化似然估计，因为判别式的目标函数由于语言的生动丰富性被认为是无助的。然而，最近的一些研究显示了 PTM 在对比学习方面的良好性能和潜力。

这方面的先驱工作有 [ELECTRA](https://arxiv.org/pdf/2003.10555)，在相同的算力下取得了超越 BERT 的表现。如下图 12 所示，ELECTRA 提出了 RTD（Replaced Token Detection），利用 GAN 的结构构建了一个预训练语言模型，ELECTRA 的生成器可以看作一个小型的 Masked Language Model，将句子中的 masked token 替换为正常的单词，而判别器 D 预测哪些单词被替换掉，这里的替换的意思是与原始单词不一致。

![ELECTRA]({{ '/assets/images/ELECTRA.png' | relative_url }})
{: style="width: 67%; margin: 0 auto;"}
***Fig. 12.** ELECTRA 框架图。 (Image source: [K Clark et al.](https://arxiv.org/pdf/2003.10555))*

其训练过程包含两个阶段：

  - 生成器”热启动“：按照 MLM 的辅助任务 $$\mathcal{L}_{\mathrm{MLM}}\left(\boldsymbol{x}, \theta_{\mathrm{G}}\right)$$ 训练 G 一些步骤以对 G 的参数进行”预热“。
  - 训练判别器：判别器的参数将会初始化为 G 的参数，并按照交叉熵判别损失 $$\mathcal{L}_{\mathrm{Disc}}\left(\boldsymbol{x}, \theta_{\mathrm{D}}\right)$$ 进行训练，此时生成器的参数将会被冻结。

最终的目标函数为：

$$
\min _{\theta_{\mathrm{G}}, \theta_{\mathrm{D}}} \sum_{\boldsymbol{x} \in \mathcal{X}} \mathcal{L}_{\mathrm{MLM}}\left(\boldsymbol{x}, \theta_{\mathrm{G}}\right)+\lambda \mathcal{L}_{\mathrm{Disc}}\left(\boldsymbol{x}, \theta_{\mathrm{D}}\right)
$$

尽管 ELECTRA 的设计是仿照 GAN 的，但是其训练方式不符合 GAN，这是由于图像数据连续的，而文本数据是离散的阻止了梯度的传播。另一方面，ELECTRA 实际上将多分类转化为了二分类，使得其计算量降低，但也可能因此降低性能。

### Graph Learning
在图学习领域中，也有利用对抗学习的实例，但是不同模型之间的区别很大。大部分方法都跟随 BiGAN 和 ALI 选择让判别器区分生成的表示和先验分布，例如 [ANE](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/16498/15927) 的训练分为两步：1）生成器 G 将采样的图编码为嵌入表示，并计算 NCE 损失；2）判别器 D 区分生成的嵌入表示和先验分布。最终的损失是对抗损失加上 NCE 损失，最终 G 可以返回一个更好的表示。

[GraphGAN](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/16611/15969) 建模链路预测任务，并且遵循 GAN 的原始形似，直接判别节点而非表示。如下图 13 所示，GraphGAN 表示分类任务中很多的错误都是由于边缘节点造成的，通常情况下同一聚簇内的节点会在嵌入空间内聚集，聚簇之间会存在包含少量节点的 density gap，作者证明如果能够在 density gap 中生成足够多的 fake 节点，就能够改善分类的性能。在训练过程中，GraphGAN 利用生成器在 density gap 中生成节点，并利用判别器判断节点的原始类别和生成的 fake 节点的类别，在测试阶段，fake 节点将会被移除，此时分类的性能应当得到提升。

![graphgan]({{ '/assets/images/graphgan.png' | relative_url }})
{: style="width: 67%; margin: 0 auto;"}
***Fig. 13.** GraphGAN 示意图。 (Image source: [H Wang et al.](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/16611/15969))*


## Discussions and Future Directions

- **Theoretical Foundation：**
  
  尽管自监督学习的方法众多，但是对于自监督学习背后的理论基础的探讨很少，理论分析十分重要，它可以避免让我们误入歧途。[S Arora et al.](https://arxiv.org/pdf/1902.09229) 探讨了对比学习目标函数的泛化能力，而 [M Tschannen et al.](https://arxiv.org/pdf/1907.13625) 则指出互信息与几种基于互信息的方法的成功关系不大，反而是其中采样策略和架构设计可能更重要。这类工作对于自监督学习形成坚实的基础至关重要，我们亟需更多的理论分析工作。

- **Transferring to downstream tasks：**

  在预训练和下游任务之间有一个较大的 gap。研究人员精心设计各种辅助任务（pretext task），以帮助模型学习数据集的一些重要特征，使得这些特征可以转移到其他下游任务中，但有时可能也很难实现。此外，选择辅助任务的过程似乎过于启发式，没有模式可循。对于预训练的任务选择问题，一个可能令人兴奋的方向是像神经架构搜索（Neural Architecture Search）那样为特定的下游任务自动设计预训练任务。

- **Transferring across datasets：**

  这个问题也被称为如何学习归纳偏差或归纳学习。传统上，我们将数据集分为用于学习模型参数的训练集和用于评估的测试集。这种学习范例的前提是现实世界中的数据符合我们训练数据集中的分布。然而，这种假设往往与实践中不符。自监督表示学习解决了部分问题，特别是在自然语言处理领域。大量的语料库用于语言模型的预训练，有助于涵盖大多数语言模式，因此有助于 PTM 在各种语言任务中的成功。然而，这是基于同一语言文本共享相同的嵌入空间这一事实。对于像机器翻译这样的任务和图学习这样的领域，不同数据集的嵌入空间不同，如何有效地学习可迁移的归纳偏差仍然是一个问题。

- **Exploring potential of sampling strategies：**
  
  [M Tschannen et al.](https://arxiv.org/pdf/1907.13625) 指出采样策略对于基于互信息的模型贡献较大，MoCo 和 SimCLR 以及一系列对比学习方法也支持了这一观点，他们都提出要尽可能的增大负样本的数量并增强正样本，这一点在深度度量学习（metric learning）中有所探讨。如何进一步提升采样带来的性能，仍然是一个有待解决而又引人注目的问题。

- **Early Degeneration for Contrastive Learning：**

  尽管 MoCo 和 SimCLR 逼近了监督学习的性能，然而，它们通常仅限于分类问题。同时，语言模型预训练的对比-生成式方法 ELECTRA 在几个标准的 NLP 基准测试中也以更少的模型参数优于其他的生成式方法。然而，一些评论表明，ELECTRA 在语言生成和神经实体提取方面的表现并没有达到预期。这一问题可能是由于对比式目标函数经常陷入嵌入空间的提前退化（early degeneration）问题，即模型过早地适应了辅助任务（pretext task），从而失去了泛化能力。我们期望在保持对比学习优势的同时，会有新的技术或范式来解决提前退化问题。