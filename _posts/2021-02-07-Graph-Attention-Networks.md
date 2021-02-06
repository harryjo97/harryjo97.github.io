---
title: Graph Attention Networks
date: 2021-02-07 22:00:00 +0900
category:
- paper review
tag:
- GAT
published: false
---

[paper review] : Graph Attention Networks 이해하기



## Introduction



CNN 은 image classification, semantic segmentation, machine translation 등 다양한 분야에 적용되어 좋은 성능을 보여주었습니다. 

local filter 의 weight sharing 덕분에 효율적이며 computationally efficient 합니다.



CNN 의 핵심인 convolution 은 주어진 data 의 구조가 grid-like (Euclidean domain) 특성을 가지는 상황에서 정의되고, 이를 irregular data 에 일반화



3D mesh, social network, biological network 등과 같이 data 의 구조가 irregular 한 경우, data 는 그래프의 형태로 잘 표현 됩니다. 그렇기 때문에, 그래프 data 에 대해 CNN 을 적용하기 위한 다양한 연구가 이루어지고 있으며, 크게 두 가지로 나눌 수 있습니다.



첫번째 방법은, Fourier domain 에서 정의된 convolution 을 이용한 spectral approach 입니다. ChebNet 과 GCN 과 같이 



두번째 방법은, 공간적으로 가까운 neighbor 를 통해 그래프에서 직접 정의된 convolution 을 사용하는 non-spectral approach 입니다. 







## GAT Architecture

&nbsp;

GAT 를 이루는 graph attentional layer 는 



$$N$$ 개의 node 를 가지는 그래프에 대해, 각 node 의 feature 를 vector $$h_i\in\mathbb{R}^F$$ 로 나타냅니다. 여기서 $$F$$ 는 feature 들의 개수이며, node 의 ordering 에는 의미가 없고 단순히 node 를 구분하기 위한 notation 입니다. Node 의 feature vector 를 $$h'_i\in\mathbb{R}^{F'}$$ 으로 update 한다면 



모든 node 들에 대해 공통된 weight matrix $$W\in\mathbb{R}^{F\times F'}$$ 를 통해 선형 변환 이후, 

공통된 attentional mechanism $$\mathcal{A}:\mathbb{R}^{F'}\times\mathbb{R}^{F'}\rightarrow\mathbb{R}$$ 을 통해 attention coefficients $$e_{ij}$$ 를 다음과 같이 계산합니다.
$$
e_{ij} = \mathcal{A}\left( Wh_i, Wh_j \right)
\tag{1}
$$
attention coefficient $$e_{ij}$$ 는 node $$i$$ 에 대한 node $$j$$ 의 중요도 (importance) 를 의미합니다. 



만약 모든 node 들의 쌍에 대한 attention coefficient 를 사용한다면, 그래프의 구조적인 특성을 살릴 수 없습니다. GAT 의 핵심은, 바로 node $$i$$ 의 neighborhood $$N_i$$ 에 속하는 node $$j$$ 들에 대한 $$e_{ij}$$ 를 사용하는 것입니다. 이렇게 masked attention 을 통해 그래프의 구조에 대한 정보를 이용할 수 있습니다.



각 node 들에 대해 $$(1)$$ 의 coefficient 들을 비교할 수 있도록, 정규화가 필요합니다. 

[4] 의 attention mechanism 과 같이 softmax 함수를 통해 normalize 해줍니다.
$$
\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k\in N_i}\exp(e_{ik})}
\tag{2}
$$


이 논문에서는 attentional mechansim $$\mathcal{A}$$ 를 weight vector $$a\in\mathbb{R}^{2F'}$$ 와 activation function 으로 LeakyReLU 를 사용해 다음과 같이 정의합니다.
$$
\mathcal{A}\left( Wh_i, Wh_j \right)
= \text{LeakyReLU}\left( a^T\left[ Wh_i\vert\vert Wh_j \right] \right)
\tag{3}
$$


<p align='center'>
    <img src = '../assets/post/Graph-Attention-Networks/attention.PNG' style = 'max-width: 100%; height: auto'>
</p>

$$(2)$$ 를 다시 쓰면, 
$$
\alpha_{ij} = \frac{\exp\left( \text{LeakyReLU}\left( a^T\left[ Wh_i\,\Vert\, Wh_j \right] \right) \right)}{\sum_{k\in N_i}\exp\left(\text{LeakyReLU}\left( a^T\left[ Wh_i\,\Vert\, Wh_k \right] \right)\right)}
\tag{4}
$$


$$(4)$$ 를 통해 계산한 정규화된 attention coefficient 들을 통해 다음과 같이 node $$i$$ 의 feature vector 를 update 해줍니다.
$$
h^{(l+1)}_i = \sigma\left( \sum_{j\in N_i}\alpha_{ij}Wh^{(l)}_j \right)
\tag{5}
$$


assigning different importances to to nodes of same neighbors

GCN layer-wise propagation rule
$$
h^{(l+1)} = \sigma\left( \sum_{j\in N_i}\frac{1}{c_{ij}}Wh^{(l)}_j \right)
$$






Self-attention 의 학습 과정을 안정화시키기 위해, multi-head attention 을 적용했습니다.



<p align='center'>
    <img src = '../assets/post/Graph-Attention-Networks/multihead.PNG' style = 'max-width: 100%; height: auto'>
</p>

각 layer 마다 $$K$$ 개의 independent 한 attention mechanism $$(5)$$ 들을 사용해 

다음과 같이 새로운 layer-wise propagation rule 을 제시합니다.
$$
h^{(l+1)} = \Big\Vert^{K}_{k=1} \sigma\left( \sum_{j\in N_i}\alpha^{k}_{ij}W^kh^{(l)}_j \right)
\tag{6}
$$
 $$l$$ 번째 layer 의 output feature vector $$h^{(l+1)}$$ 은 $$KF^{(l+1)}$$ 개의 feature 들을 가지게 됩니다.



마지막 layer 에서는 concatenation 의 의미가 없으므로, 대신 feature 들의 averaging 을 통해 final output 을 만들어줍니다.


$$
h^{(L)} = \sigma\left( \frac{1}{K}\sum^K_{k=1}\sum_{j\in N_i}\alpha^{k}_{ij}W^kh^{(L-1)}_j \right)
\tag{7}
$$


highly efficient : self-attentional layer 모든 edge 들에 대해 parallelized , 

각 node 들의 feature vector update 또한 각 node 들에 대해 parallelized

특히 GCN 과 같이 eigenvalue decomposition 혹은 그와 비슷한 복잡한 행렬 연산이 필요하지 않습니다.

$$(6)$$ 에서 $$K=1$$ 일 때의 computational complexity $$O\left(\vert V\vert FF' + \vert E\vert F'\right)$$ 

GCN 과 비슷한 복잡도





### GAT vs GCN



GCN 과의 가장 큰 차이는 위에서 설명했듯이, 동일한 neighborhood 내의 node 들에 대해 다른 importance 를 부여할 수 있다는 점입니다. 특히 모델을 학습시킨 후, 학습된 attentional weight 는 모델에 대한 해석에 큰 도움을 줄 수 있습니다.



또한 GCN 은 Fourier domain 에서 정의된 convolution 

graph Laplacian 

이는 학습 전에 그래프의 구조에 대한 정보를 알고 있어야합니다.

결국 GCN 과 다르게 바로 inductive learning 에 적용할 수 있습니다.



### GAT vs GraphSAGE



sample fixed-size neighborhood due to computational footprint consistency

추론을 진행할 때, neighborhood 전체에 대한 정보를 사용할 수 없습니다.

특히 LSTM 을 aggregator function 으로 사용한 GraphSAGE 와 다르게, node 의 ordering 과 상관 없다









## Evaluation



<p align='center'>
    <img src = '../assets/post/Graph-Attention-Networks/transductive.PNG' style = 'max-width: 100%; height: auto'>
</p>



<p align='center'>
    <img src = '../assets/post/Graph-Attention-Networks/inductive.PNG' style = 'max-width: 100%; height: auto'>
</p>



<p align='center'>
    <img src = '../assets/post/Graph-Attention-Networks/t-sne.PNG' style = 'max-width: 100%; height: auto'>
</p>





## Conclusion



## Reference



1. Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua
   Bengio. [Graph attention networks](https://arxiv.org/pdf/1710.10903.pdf). arXiv preprint arXiv:1710.10903, 2017.



2. Thomas N Kipf and Max Welling. [Semi-supervised classification with graph convolutional networks](). arXiv preprint arXiv:1609.02907, 2016.



3. Will Hamilton, Zhitao Ying, and Jure Leskovec. [Inductive representation learning on large graphs](https://arxiv.org/pdf/1706.02216.pdf). In Advances in Neural Information Processing Systems, pages 1024–1034, 2017.



4. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. [Neural machine translation by jointly
   learning to align and translate](https://arxiv.org/pdf/1409.0473.pdf). International Conference on Learning Representations (ICLR),
   2015.