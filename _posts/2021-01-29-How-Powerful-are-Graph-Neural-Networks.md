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
h_v^{(k)} 
&= \text{COMBINE}^{(k)}\left(h_v^{(k-1)},a_v^{(k)}\right) \\
&= \text{COMBINE}^{(k)}\left(h_v^{(k-1)},\text{AGGREGATE}^{(k)}\left(\left\{\!\!\left\{h_u^{(k-1)}:u\in N(v)\right\}\!\!\right\}\right)\right) 
\tag{1}
\end{align}
$$



GNN 에서 $$\text{AGGREGATE}$$ 와 $$\text{COMBINE}$$ 함수의 선택에 따라 성능이 



Node classification 에서는 GNN 의 마지막 layer 에서 얻은 feature vector $$h_v^{(K)}$$ 들로 prediction 을 수행합니다.  Graph classificaiton 의 경우,  마지막 layer 에서 얻은 feature vector 들을 모아 $$\text{READOUT}$$ 함수를 통해 그래프의 representation $$h_G$$ 를 표현하고, 이를 통해 prediction 을 수행합니다.

$$
h_G = \text{READOUT}\left( \left\{\!\!\left\{ h_v^{(K)}:v\in V \right\}\!\!\right\} \right)
\tag{2}
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



이를 토대로 MPNN 보다 강력한 GNN 을 만드는 것이 목표입니다.



## Building Powerful Graph Neural Networks



### Weisfeiler-Lehman Test



> **Lemma 2.** 
>
> Let $$G_1$$ and $$G_2$$ be any two non-isomorphic graphs. If a graph neural network $$\mathcal{A}:\mathcal{G}\rightarrow\mathbb{R}^d$$ maps $$G_1$$ and $$G_2$$ to different embeddings, the Weisfeiler-Lehman graph isomorphism test also decides $$G_1$$ and $$G_2$$ are not isomorphic.



따라서 neighborhood aggregation scheme 을 사용하는 GNN 은, graph Isomorphism 을 구분하는 능력에 있어 최대 WL test 만큼 강력하다는 것입니다.



Lemma 2 의 증명은 





> **Theorem 3.**
>
> Let $$\mathcal{A}:\mathcal{G}\rightarrow\mathbb{R}^d$$ be a GNN. With a sufficient number of GNN layers, $$\mathcal{A}$$ maps any graphs $$G_1$$ and $$G_2$$ that the Weisfeiler-Lehman test of isomorphism decides as non-isomorphic, to different embeddings if the following conditions hold:
>
> a) $$\mathcal{A}$$ aggregates and updates node features iteratively with
> 
> $$
> h_v^{(k)} = \phi\left( h_v^{(k-1)},f\left(\left\{\!\!\left\{ h_u^{(k-1)}:u\in N(v) \right\}\!\!\right\}\right) \right)
> $$
> 
> where the functions $$f$$, which operates on multisets, and $$\phi$$ are injective.
>
> b) $$\mathcal{A}$$ 's graph-level readout, which operates on the multiset of node features $$\left\{\!\!\left\{  h_v^{(k)}\right\}\!\!\right\}$$, is injective 



Theorem 3 에서 함수 $$f$$ 와 $$\phi$$ 는 각각 위에서 설명한 $$\text{AGGREGATE}$$ 와 $$\text{COMBINE}$$ 함수에 해당하며, graph-level readout 은 $$\text{READOUT}$$ 을 의미합니다. 즉 $$\text{AGGREGATE}$$, $$\text{COMBINE}$$ 과 $$\text{READOUT}$$ 이 injective 일 경우, GNN 은 WL test 와 같은 성능을 낼 수 있습니다.



그래프를 구분하는 능력에 있어 GNN 이 최대 WL test 만큼 성능을 낼 수 있다면,  GNN 을 쓰는 이유가 무엇인지에 대해 생각해보아야 합니다.



GNN 의 가장 큰 장점은 바로 그래프의 구조에 대해 학습할 수 있다는 것입니다. 



### Graph Isomorphism Network



node 의 input feature space $$\chi$$ 가 countable 



> **Lemma 5.**
>
> Assume $$\chi$$ is countable. There exists a function $$f:\chi \rightarrow\mathbb{R}^n$$ so that $$h(X)=\sum_{x\in X}f(x)$$ is unique for each multiset $$X\subset\chi$$ of bounded size. Moreover, any multiset function $$g$$ can be decomposed as $$g(X)=\phi\left(\sum_{x\in X}f(x)\right)$$ for some function $$\phi$$.



> **Corollary 6.**
>
> Assume $$\chi$$ is countable. There exists a function $$f:\chi \rightarrow\mathbb{R}^n$$ so that for infinitely many choices of $$\epsilon$$, including all irrational numbers, $$h(c,X)=(1+\epsilon)f(c) + \sum_{x\in X}f(x)$$ is unique for each pair $$(c,X)$$ where $$c\in\chi$$ and $$X\subset\chi$$ is a multiset of bounded size. Moreover, any function $$g$$ over such pairs can be decomposed as $$g(c,X)=\varphi\left( (1+\epsilon)f(c)+\sum_{x\in X}f(x) \right)$$ for some function $$\varphi$$.



Lemma 5 와 Corollary 6 의 증명에서 핵심은, countable $$\chi$$ 의 enumeration $$Z: \chi\rightarrow\mathbb{N}$$ 와 bounded multiset $$X$$ 에 대해 $$\vert X\vert<N$$ 를 만족하는 $$N$$ 을 사용해 $$f(x) = N^{-Z(x)}$$ 를 정의하는 것입니다. 모든 bounded multiset $$X$$ 에 대해 $$h(X)$$ 가 injective 하게 만들기 위해, countable $$\chi$$ 의 각 원소들을 나열하고 각 원소가 포함되었는지 아닌지를 $$N$$ 진법으로 표현하는 것입니다.



$$(1)$$ 에 Corollary 6 의 결과를 사용하면, 각 layer $$k=1,\cdots,K$$ 에 대해 다음을 만족하는 함수 $$f^{(k)}$$ 와 $$\varphi^{(k)}$$ 가 존재합니다.

$$
h_v^{(k)} = \varphi^{(k-1)}\left( (1+\epsilon)\;f^{(k-1)}\left(h_v^{(k-1)}\right)+\sum_{u\in N(v)}f^{(k-1)}\left(h_u^{(k-1)}\right) \right)
\tag{3}
$$

$$(3)$$ 에서 양변에 $$f^{(k)}$$ 를 취해주면 다음과 같습니다.

$$
f^{(k)}\left(h_v^{(k)}\right) = f^{(k)}\circ\varphi^{(k-1)}\left( (1+\epsilon)\;f^{(k-1)}\left(h_v^{(k-1)}\right)+\sum_{u\in N(v)}f^{(k-1)}\left(h_u^{(k-1)}\right) \right)
\tag{4}
$$

$$k$$ 번째 layer 에서 각 node 의 feature vector 를 $$f^{(k)}\left(h_v^{(k)}\right)$$ 로 생각한다면, $$(4)$$ 를 간단히 쓸 수 있습니다.

$$
h_v^{(k)} = f^{(k)}\circ\varphi^{(k-1)}\left( (1+\epsilon)\;h_v^{(k-1)}+\sum_{u\in N(v)}h_u^{(k-1)} \right)
\tag{5}
$$


Universal approximation theorem 덕분에 multi-layer perceptrons (MLPs) 를 이용하면, 두 함수의 composition $$f^{(k)}\circ\varphi^{(k-1)}$$ 을 근사할 수 있습니다. 또한 $$\epsilon$$ 을 학습 가능한 parameter $$\epsilon^{(k)}$$ 으로 설정한다면, $$(5)$$ 를 다음과 같이 MLP 를 사용해 표현할 수 있습니다.

$$
h_v^{(k)} = \text{MLP}^{(k)}\left( \left(1+\epsilon^{(k)}\right)\;h_v^{(k-1)}+\sum_{u\in N(v)}h_u^{(k-1)}\right)
\tag{6}
$$


Graph Isomorphism Network (GIN) 은 $$(6)$$ 을 layer-wise propagation rule 로 사용합니다. Theorem 3 로 인해 GIN 은 maximally powerful GNN 이라는 것을 알 수 있습니다. 즉 간단한 구조를 가지며 powerful 하다는 것이 GIN 의 가장 큰 장점입니다.



Node classification 에는 $$(6)$$ 을 활용한 GIN 을 사용하면 되지만, graph classification 의 경우 graph-level readout function 이 필요합니다. Readout function 의 input 인 node 의 representation 은 layer 를 거칠수록 local 에서 점점 global 하게 변합니다. Layer 의 수가 너무 많다면, global 한 특성만 남을 것이고, layer 의 수가 너무 적다면 local 한 특성만 남게 됩니다. 따라서, 그래프를 구분하기 위해서는 적당한 수의 layer 를 거쳐야 합니다. 



이런 특성을 반영하기 위해, GIN 에서 각 layer 의 graph representation (i.e. $$\text{READOUT}(\,\cdot\,)$$ 의 output) 을 concatenation 으로 모두 합쳐줍니다.  

$$
h_G = \text{CONCAT}\left( \text{READOUT} \left(\left\{\!\!\left\{h_v^{(k)} \right\}\!\!\right\}\right) \,:\, k=0,1,\cdots,K\right)
\tag{7}
$$
$$(7)$$ 은 각 layer 마다 나타나는 그래프의 구조적 정보를 모두 포함하고 있습니다. 이 때, graph representation 을 모든 node 의 feature vector 를 더해주는 summation 으로 정의한다면,
$$
\text{READOUT} \left(\left\{\!\!\left\{h_v^{(k)} \right\}\!\!\right\}\right) = \sum_{v\in V} f^{(k)}\left(h_v^{(k)}\right)
$$

graph-level readout function 은 node feature vector 로 이루어진 multiset 에 대해 injective 합니다.



따라서, Theorem 3 에 의해 graph classification 의 GIN 또한 maximally powerful GNN 임을 확인할 수 있습니다.





논문에서는 node 의 input feature space $$\chi$$ 가 countable 인 상황만 고려했지만, 실제로 그래프의 input data 가 countable space 라고 보장할 수 없습니다. $$\chi$$ 가 continuous space 일 때 이론적인 연구가 필요해보입니다.



## Less Powerful But Still Interesting GNNs



논문에서는 $$(6)$$ 의 두 가지 특징, MLP 와 feature vector summation, 에 대한 ablation study 를 보여줍니다.

다음의 두 가지 변화를 통해 GNN 의 성능이 떨어짐을 확인합니다.

1. MLPs 대신 1-layer perceptrons
2. Summation 대신 mean-pooling 또는 max-pooling

 

### 1-Layer Perceptrons instead of MLPs



GCN 의 layer-wise propagation rule 은 다음과 같습니다.

$$
h_v^{(k)} = \text{ReLU}\left( W\cdot\text{MEAN}\left\{\!\!\left\{ h_u^{(k-1)} \,:\, u\in N(v)\cup\{v\}\right\}\!\!\right\}  \right)
\tag{8}
$$

$$(6)$$ 과 비교해보면, $$\text{MLP}$$ 대신 1-layer perceptron $$\sigma\circ W$$ 를 사용했음을 알 수 있습니다. Universal approximation theorem 은 MLP 에 대해 성립하지만, 일반적으로 1-layer perceptron 에 대해서는 성립하지 않습니다.  다음의 Lemma 7 은 1-layer perceptron 을 사용한 GNN 이 구분하지 못하는 non-isomorphic 그래프들이 존재함을 보여줍니다. 즉, 1-layer perceptron 으로는 충분하지 않다는 뜻입니다.



> **Lemma 7.**
>
> There exist finite multisets $$X_1\neq X_2$$ so that for any linear mapping $$W$$, 
> 
> $$
> \sum_{x\in X_1} \text{ReLU}(Wx) = \sum_{x\in X_2} \text{ReLU}(Wx)
> $$



### Mean / Max-Pooling instead of Summation



Aggregator $$h$$ 를 사용한 GraphSAGE [4] 의 layer-wise propagation rule 은 다음과 같습니다.

$$
h_v^{(k)} = \text{ReLU}\left( W\cdot \text{CONCAT}\left( h_v^{(k-1)}, h\left( \left\{\!\!\left\{ h_u^{(k-1)} \,:\, u\in N(v) \right\}\!\!\right\} \right) \right) \right)
$$


Max-pooling 과 mean-pooling 의 경우 aggregator $$h$$ 는 $$f(x) = \text{ReLU}\left(Wx\right)$$ 에 대해 다음과 같습니다.

$$
\begin{align}
& h_{max}\left( \left\{\!\!\left\{ h_u^{(k-1)} \,:\, u\in N(v) \right\}\!\!\right\} \right) 
= \text{MAX}\left( \left\{\!\!\left\{ f\left(h_u^{(k-1)}\right) \,:\, u\in N(v) \right\}\!\!\right\}  \right) \\
\\
& h_{mean}\left( \left\{\!\!\left\{ h_u^{(k-1)} \,:\, u\in N(v) \right\}\!\!\right\} \right) 
= \text{MEAN}\left( \left\{\!\!\left\{f\left(h_u^{(k-1)}\right) \,:\, u\in N(v) \right\}\!\!\right\}  \right)
\end{align}
$$

여기서 $$\text{MAX}$$ 와 $$\text{MEAN}$$ 은 element-wise max 와 mean operaotr 입니다.



$$h_{max}$$ 와 $$h_{mean}$$ 모두 multiset 에 대해 정의되며, permutation invariant 하기 때문에, aggregator 로써 역할을 잘 수행합니다. 하지만, 두 함수 모두 multiset 에 대해 injective 하지 않습니다. 다음의 예시를 통해 확인해보겠습니다.



<p align='center'>
    <img src = '/assets/post/How-Powerful-are-Graph-Neural-Networks/eg.PNG' style = 'max-width: 100%; height: auto'>
</p>



Figure 3 에서 node 의 색은 feature vector 를 의미합니다. 같은 색을 가지면, 같은 feature vector 를 가집니다. $$f(red) > f(blue)>f(green)$$ 을 만족한다고 가정하겠습니다. Figure 3-(a) 를 보면 non-isomorphic 한 두 그래프 모두 $$h_{max}$$ 와 $$h_{mean}$$ 의 결과가 $$f(blue)$$ 로 같습니다. Figure 3-(c) 도 마찬가지로 non-isomorphic 한 두 그래프 모두 $$h_{max}=f(red)$$, $$h_{mean}=\frac{1}{2}(f(red)+f(green))$$ 으로 결과가 같습니다. Figure 3-(b) 의 경우 $$h_{mean}$$ 은 값이 다르지만, $$h_{max}$$ 의 값은 같습니다. 



논문에서는 $$(6)$$ 에서와 같이 summation 

$$
h_{sum}\left( \left\{\!\!\left\{ h_u^{(k-1)} \,:\, u\in N(v) \right\}\!\!\right\} \right) 
= \sum_{u\in N(v)}f\left(h_u^{(k-1)}\right) 
$$

$$h_{sum}$$ 이 multiset 전체를 injective 하게 표현할 수 있고, $$h_{mean}$$ 의 경우 multiset 의 distribution 을, $$h_{max}$$ 의 경우 multiset 의 서로다른 원소들로 이루어진 set 을 표현할 수 있다고 설명합니다.

<p align='center'>
    <img src = '/assets/post/How-Powerful-are-Graph-Neural-Networks/rank.PNG' style = 'max-width: 100%; height: auto'>
</p>



따라서, max-pooling 과 mean-pooling 을 사용한 GraphSAGE 같은 경우 GIN 보다 representation power 가 떨어진다고 볼 수 있습니다. 



## Experiment & Result



<p align='center'>
    <img src = '/assets/post/How-Powerful-are-Graph-Neural-Networks/train.PNG' style = 'max-width: 100%; height: auto'>
</p>





<p align='center'>
    <img src = '/assets/post/How-Powerful-are-Graph-Neural-Networks/test.PNG' style = 'max-width: 100%; height: auto'>
</p>







## Reference



1. Xu, K., Hu, W., Leskovec, J., and Jegelka, S. (2019). [How powerful are graph neural networks?](https://arxiv.org/pdf/1810.00826.pdf) In
   International Conference on Learning Representations.



2. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and Dahl, G. E. (2017). [Neural
   message passing for quantum chemistry](https://arxiv.org/pdf/1704.01212.pdf). In International Conference on Machine Learning, pages 1263–1272.



3.  Zhengdao Chen, Soledad Villar, Lei Chen, and Joan Bruna. [On the equivalence between graph isomorphism testing and function approximation with GNNs](https://arxiv.org/pdf/1905.12560.pdf). In Advances in Neural Information Processing Systems, pages 15868–15876, 2019.



4. William L Hamilton, Rex Ying, and Jure Leskovec. [Inductive representation learning on large graphs](https://arxiv.org/pdf/1706.02216.pdf).
   In Advances in Neural Information Processing Systems (NIPS), pp. 1025–1035, 2017a.



