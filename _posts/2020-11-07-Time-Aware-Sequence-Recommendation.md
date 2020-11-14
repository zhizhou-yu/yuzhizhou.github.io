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
title: "Time Interval Aware Self-Attention for Sequential Recommendation 论文阅读"
date: 2020-11-07 12:00:00
tags: exploration
key: Time-Aware-Sequence-Recommendation
---

>在实际的推荐场景中，大部分的推荐算法都忽略了用户行为序列的 time interval，或者也只是简单的把 time interval 加入序列中的一个特征（目前我们的模型也是这种做法，稍微复杂点的也就是把 time interval 输入到一个 gate 中并输出序列中的不同 item 的重要性 score），导致模型在时序细粒度感知上存在瓶颈。因此，构建一个合适的 Time-Aware 序列推荐模型十分重要。本篇主要解读下 WSDM 2020 的论文 [Time Interval Aware Self-Attention for Sequential Recommendation](https://cseweb.ucsd.edu/~jmcauley/pdfs/wsdm20b.pdf)

<!--more-->

* TOC
{:toc}

## 摘要

Sequential Recommned System 试图利用用户历史行为序列（点击、购买等），来预测用户的下一步行动。经典方法如马尔可夫链（Markov Chains）、递归神经网络（RNN）和自注意力（Self-Attention）由于具有捕捉序列动态模式的能力而被广泛应用。但是，大多数这些模型都做了一个简化假设：仅将历史行为序列记录视为有序序列，而不考虑每次行为之间的时间间隔（time interval），换句话说，它们是在对时序进行建模而不对实际的时间戳进行建模。在本文中，我们试图在序列推荐框架中显式地建模序列内行为之间的时间戳，以探索不同时间间隔对 next item 预测的影响。我们提出了具有时间间隔感知的自注意序列推荐模型 TiSASRec(Time Interval aware Self-attention based sequential recommendation），该模型可以同时对 item 的绝对位置（position）以及它们之间的时间间隔进行建模。实验结果表明，我们的方法在稀疏和密集数据集上表现均优于各种最新的序列推荐模型。


## 引言

挖掘用户历史行为序列的重要工作主要有两种模式：时间推荐（temporal recommendation）和序列推荐（sequential recommendation）。时间推荐着重于建模绝对时间戳以捕捉用户和 item 的时间动态关系。 例如，某个商品的受欢迎程度可能在不同的时段内发生变化，或者用户的平均评分可能会随着时间的推移而增加或减少。 这些模型在探索数据集中的时间变化时很有用。不同于序列推荐，时间推荐仅考虑取决于时间的时间模式。

而大多数已有的序列推荐模型都按照行为时间戳对 item 进行排序，并专注于序列模式的挖掘以预测 next item。一种典型的解决方案是基于 Markov Chains 的方法，它们在高度稀疏的数据中表现良好，但在更复杂的场景中可能无法捕获复杂的用户行为动态。递归神经网络（RNN）也广泛应用于序列推荐，尽管 RNN 模型具有所谓的“长记忆力”（long-term memory），但它们需要大量数据（尤其是密集数据）才能取得一个较好的表现。为了解决 Markov Chains 模型和基于 RNN 的模型的缺陷，受 Transformer 启发，将自注意力机制应用于序列推荐问题。基于自注意力的模型显著优于基于 Markov Chains / CNN / RNN 等序列推荐方法。

通常，现有的序列推荐模型会丢弃具体的时间戳，仅保留序列行为的时序，也就是说，这些方法（隐式）假定序列中的所有相邻 item 都具有相同的时间间隔，影响 next item 的因素仅是前一项的位置和标识。但是，直觉上，时间戳较近的项目对下一个项目的影响更大。例如，两个用户具有相同的行为序列，但是其中一个用户在一天内生成了这些行为，而另一个用户在一个月内完成了这些行为，因此即使他们的历史行为在序列中具有相同的绝对位置，但这两个用户的行为应该对下一个项目有不同的影响。此时，现有的大部分序列推荐模型因为只考虑行为序列的绝对位置，会将这两种行为序列看作是相同的输入，如下图 1 所示，我们希望模型对于在绝对位置上完全一致，但时间间隔不同的序列，我们应当给用户推荐出不同的结果。

![不同的时间间隔的相同行为序列]({{ '/assets/images/different-seq.png' | relative_url }})
{: style="width: 75%; margin: 0 auto;"}
*Fig. 1. 在时序上完全一致，但时间间隔不同的序列，一个好的推荐系统应当给予用户不同的推荐结果. (Image source: [Jiacheng Li et al.](https://cseweb.ucsd.edu/~jmcauley/pdfs/wsdm20b.pdf))*

在本文中，我们认为用户行为序列应该被建模为具有不同时间间隔的序列。图 2 显示了对于不同的数据集，用户的行为序列具有不同的时间间隔，以往的研究忽略了这些时间间隔的相互作用及其对预测 next item 的影响。因此，为了解决上述局限，我们提出了一个时间感知的自注意力推荐模型，其灵感来自于[22]相对位置表征的自我注意。该模型不仅考虑了行为的绝对位置，还考虑了任意两个行为之间的相对时间间隔。

![不同的时间间隔的相同行为序列]({{ '/assets/images/datasets-with-diffrent-time-intervals.png' | relative_url }})
{: style="width: 75%; margin: 0 auto;"}
*Fig. 2. 不同的数据集中统计出的用户行为序列时间间隔是有显著差异的，因此模型应当合理利用这个特征. (Image source: [Jiacheng Li et al.](https://cseweb.ucsd.edu/~jmcauley/pdfs/wsdm20b.pdf))*

## 方法

TiSASRec 包括个性化的时间间隔处理模块，一个嵌入层，时间感知自注意力模块，和一个预测层。模型会将 item 的绝对位置和相对时间间隔进行嵌入，并基于此计算注意力权重，使用交叉熵损失作为我们的目标函数。TiSASRec 的目标是捕获序列模式，并探索时间间隔对 next item 推荐的影响。