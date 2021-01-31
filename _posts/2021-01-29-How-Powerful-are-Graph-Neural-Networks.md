---
title: How Powerful are Graph Neural Networks?
date: 2021-02-02 22:00:00 +0900
category:
- paper review
tag:
- WL test

published: false
---

Graph Isomorpihsm Network 이해하기





## Before



GNN 의 expressive power 에  대한 연구는 크게 두 가지 방향으로 이루어집니다.

첫 번째 방법은 이 논문과 같이 Weisfeiler-Lehman (WL) graph isomorphism test 를 통해



WL test 와의 비교를 통해 GNN 의 limitation 



두 번째 방법은 expressive power 를 continuous function 을 근사할 수 있는지 

특히 그래프에서 다루는 함수들은 주로 permutation invariant 또는 equivariant 하므로, 

permutation invariant / equivariant function 들에 대한 universality 



### Related Papers

|                                                              |                      |
| ------------------------------------------------------------ | -------------------- |
| [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://arxiv.org/pdf/1810.02244.pdf) | Morris et al., 2018  |
| [Provably Powerful Graph Networks](https://arxiv.org/pdf/1905.11136.pdf) | Maron et al., 2019   |
| [On the Universality of Invariant Networks](https://arxiv.org/pdf/1901.09342.pdf) | Maron et al., 2019   |
| [Universal Invariant and Equivariant Graph Neural Networks](https://arxiv.org/pdf/1905.04943.pdf) | Keriven et al., 2019 |
| [On the equivalence between graph isomorphism testing and function approximation with GNNs](https://arxiv.org/pdf/1905.12560.pdf) | Chen et al., 20      |
|                                                              |                      |








## Introduction



잘 알려져 있는 Graph Convolutional Network, GraphSAGE, Graph Attention Network, Gated Graph Neural Netowork 등 대부분의 GNN 은 recursive neighborhood aggregation (message passing) 방식을 사용합니다 [2]. 이런 network 들을 Message Passing Neural Network (MPNN) 이라 부릅니다. 



MPNN 은 각 node 마다 주변 neighborhood 의 feature vector (representation) 를 수집하여, 새로운 feature vector 를 만들어냅니다. $$k$$ 번의 iteration 후, 각 node 들은 $$k$$-hop neighborhood 의 feature vector 들을 모을 수 있고, 이는 그래프의 구조에 대한 정보를 포함한다고 해석할 수 있습니다.

MPNN 에 속하는 network 들은 서로 다른 neighborhood aggregation scheme 을 사용합니다.



특히 node classification, link prediction, graph classification 등 다양한 task 에 대해 state-of-the-art 성능을 보여줍니다. 하지만, 이런 GNN 을 설계하는 것은 주로 경험적인 직관 혹은 실험을 통한 시행 착오를 통해 이루어집니다. 위에서 설명했듯이, 이 논문에서는 GNN 의 expressive power 를 Weisfeiler-Lehman (WL) graph isomorphism test 를 통해 분석합니다. 



WL test 는 graph isomorphism 



WL test 또한 MPNN 과 같이 매 iteratin 마다 주변 neighborhood 의 feature vector 를 수집해 각 node 의 feature vector 를 update 합니다.



비록 regular graph 와 같은 그래프들을 완벽히 구분하지는 못하지만, 

대부분의 그래프에 대해 

특히 단순한 알고리즘 덕분에 computationally efficient 



WL test 가 강력한 이유는 neighborhood aggregation 이후 node 의 feature vetor 를 update 하는 과정이 injective 하기 때문입니다. 그래프의 두 node 가 서로 다른 neighborhood 를 가지고 있다면, 서로 다른 feature vector 를 가지게 됩니다.



GNN 또한 neighborhood aggregation scheme 



각 node 의 neighborhood 를 multiset 으로 본다면, GNN 의 neighborhood aggregation scheme 은 multiset 에 대한 함수로 생각할 수 있습니다. GNN 이 WL test 와 같이 그래프를 구분할 수 있는 능력 (discriminative power) 이 좋아지려면, 함수는 서로 다른 multiset 에 대해 embedding space 에서 서로 다른 위치로 보내주어야 합니다. 

따라서, GNN 의 expressive power 를 multiset function 을 통해 나타낼 수 있습니다. 







## Preliminaries



논문에서 다루는 GNN 은 MPNN 으로 매 iteration 마다 각 node 의 neighborhood feature vector 를 수집해 새로운 feature vector 로 update 합니다. 이 과정은 크게 두 단계로 나눌 수 있습니다.



$$k$$ 번째 iteration 에서 node $$v$$ 의 feature vector 를 $$h_v^{(k)}$$ 



첫번 째는 AGGREGATE 입니다. 각 node 의 neighborhood feature vector 를 수집하는 단계로, $$v$$ 의 neighborhood $$N(v)$$ 안의 $$u$$ 에 대해 feature vector $$h_u^{(k-1)}$$ 들을 모아줍니다.

$$
a_v^{(k)} = \text{AGGREGATE}^{(k)}\left(\left\{\!\!\left\{h_u^{(k-1)}:u\in N(v)\right\}\!\!\right\}\right)
$$


$$\text{AGGREGATE}$$ 함수는 multiset 에 대해 정의된 함수이며, GraphSAGE 에서와 같이 max-pooling 또는 mean-pooling 등을 사용할 수 있습니다.



두번 째는 COMBINE 입니다. 전 단계에서 수집한 정보 $$a_v^{(k)}$$ 와 현재의 feature vector $$h_v^{(k-1)}$$ 를 사용해 feature vector 를 update 합니다.
$$
h_v^{(k)} = \text{COMBINE}^{(k)}\left(h_v^{(k-1)},a_v^{(k)}\right)
$$
GraphSAGE 에서는 vector concatenation $$[\,\cdot\,]$$ 이후 weight matrix $$W$$ 를 이용한 linear mapping 을 통해 다음과 같은 $$\text{COMBINE}$$ 함수를 사용했습니다.
$$
\text{COMBINE}^{(k)}\left(h_v^{(k-1)},a_v^{(k)}\right)
= W \cdot \left[ h_v^{(k-1)},a_v^{(k)} \right]
$$


정리하면, MPNN 의 $$k$$ 번째 layer 는 다음과 같이 나타낼 수 있습니다. 


$$
\begin{align}
& a_v^{(k)} = \text{AGGREGATE}^{(k)}\left(\left\{\!\!\left\{h_u^{(k-1)}:u\in N(v)\right\}\!\!\right\}\right) \\
& h_v^{(k)} = \text{COMBINE}^{(k)}\left(h_v^{(k-1)},a_v^{(k)}\right)
\end{align}
$$



GNN 에서 $$\text{AGGREGATE}$$ 와 $$\text{COMBINE}$$ 함수의 선택에 따라 성능이 



Node classification 에서는 GNN 의 마지막 layer 에서 얻은 feature vector $$h_v^{(K)}$$ 들로 prediction 을 수행합니다.  Graph classificaiton 의 경우,  마지막 layer 에서 얻은 feature vector 들을 모아 $$\text{READOUT}$$ 함수를 통해 그래프의 representation $$h_G$$ 를 표현하고, 이를 통해 prediction 을 수행합니다.
$$
h_G = \text{READOUT}\left( \left\{\!\!\left\{ h_v^{(K)}:v\in V \right\}\!\!\right\} \right)
$$
그래프의 representation 이 점의 ordering 에 달라지지 않길 바라기 때문에, $$\text{READOUT}$$  함수로 permutation invariant function 을 사용합니다. 간단한 예로 feature 들을 모두 더하는 summation 이 있습니다.






## Theoretical Framework



Multiset $$X$$ 는 $$X$$ 의 서로 다른 원소들로 이루어진 집합 $$S$$ 와, $$S$$ 의 각 원소들을 multiplicity 로 mapping 해주는 함수 $$m:S\rightarrow \mathbb{N}$$ 으로 표현할 수 있습니다. 예를 들어, multiset $$X = \{\!\!\{ 1,3,2,3,1,2 \}\!\!\}$$ 에 대해
$$
S = \{1,2,3\}\\
m: 1\mapsto 2, 2\mapsto 2, 3\mapsto 2
$$
$$X = (S=\{1,2,3\},   m: 1\mapsto 2, 2\mapsto 2, 3\mapsto 2)$$ 와 같이 나타낼 수 있습니다.



이상적인 GNN 은 서로 다른 두 node 가 같은 neighborhood (multiset) 를 가질 때만 같은 representation 을 줍니다. 따라서, neighborhood aggregation scheme 은 multiset 에 대해 injective 해야합니다.





## Building Powerful Graph Neural Networks



lemma 2, 3

GIN

lemma 5, 6

READOUT as sum



## Less Powerful But Still Interesting GNNs



1-layer perceptron not enough : GCN



mean and max pooling not injective



examples that confuse mean max pooling



## Experiment & Result







## Reference



1. Xu, K., Hu, W., Leskovec, J., and Jegelka, S. (2019). [How powerful are graph neural networks?](https://arxiv.org/pdf/1810.00826.pdf) In
   International Conference on Learning Representations.



2. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and Dahl, G. E. (2017). [Neural
   message passing for quantum chemistry](https://arxiv.org/pdf/1704.01212.pdf). In International Conference on Machine Learning, pages 1263–1272.







