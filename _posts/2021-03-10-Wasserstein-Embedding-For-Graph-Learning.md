---
title: Wasserstein Embedding For Graph Learning
date: 2021-03-10
category:
- paper review
tag:
- Optimal Transportation
published: false
---

[paper review] : WEGL 이해하기



## Related Study

&nbsp;

주어진 그래프들을 분류하는 graph classification 은 화학, 생물학, 사회인문학 등 다양한 분야에서 사용됩니다. 하지만 두 그래프가 얼마나 다른지를 측정하는 것은 생각보다 어려운 과제입니다. 



단순한 방법으로는 두 그래프가 같은 그래프인지를 graph isomorphism 을 통해 구분할 수 있습니다. Graph isomorphism 에 대해서는 다양한 연구가 이루어졌으며, 대표적인 방법으로는 Weisfeiler-Lehman test 가 있습니다. 특히 GIN 은 graph isomorphism 과 GNN 의 expressive power 



하지만 graph isomorphism 을 통해 알 수 있는 것은 두 그래프가 동일한지 아닌지 뿐입니다. 두 그래프가 다르다면 얼마나 다른지에 대한 정보를 얻을 수 없습니다. 최근 들어 optimal transportation 을 통해 그래프 사이의 dissimilarity 를 수치화하는 연구가 이루어지고 있습니다 [1,4,5]. 

&nbsp;

## Introduction

&nbsp;

기존에 graph-structured data 를 분석하는 방식에는 크게 두 가지가 있습니다. 첫 번째는 GNN 을 사용하는 방법입니다.  GNN 은 feature aggregation, graph pooling, classification 세 가지를 거쳐 그래프의 representation 을 학습합니다. 다양한 domain 에서 좋은 성능을 보여주고 있습니다.



두 번째 방법은 graph kernel 을 이용해 두 그래프 사이의 similarity 를 표현하여 SVM 과 같은 classifier 를 통해 그래프를 학습하는 방법입니다. 대표적인 예로 random walk kernel, Weisfeiler-Lehman kernel 등이 있으며, 최근에는 Wasserstein distance 를 활용한 WWL kernel [4] 에 대한 연구가 진행되었습니다.



기존의 GNN 과 graph kernel 모두 그래프의 크기가 커질수록 사용하기 힘들어지는 단점이 있습니다. GNN 은 그래프의 크기가 클수록 학습시 필요한 계산량이 많아져 학습이 어려워집니다. 또한 graph kernel 의 경우 그래프 쌍마다 similarity 를 계산해야하기 때문에 마찬가지로 크기가 큰 그래프 dataset 에 사용하기 어렵습니다.



논문에서는 그래프에 LOT framework [2] 를 적용해 이런 문제를 해결하려고 합니다. GNN 과 graph kernel 의 장점을 모두 사용하기 위해, node embedding 과 LOT framework 를 결합한 Wasserstein Embedding for Graph Learning (WEGL) 를 제시합니다.

&nbsp;

## Background

&nbsp;



### Wasserstein Distances



$$\mathcal{Z}\subset\mathbb{R}^d$$ 에서 정의된 두 probability measure $$\mu_i$$ 와 $$\mu_j$$ 사이의 2-Wasserstein distance 는 다음과 같이 정의합니다 [7].
$$
\mathcal{W}_2(\mu_i,\mu_j) = \left( \inf_{\pi\in\Pi(\mu_i,\mu_j)} \int_{\mathcal{Z}\times\mathcal{Z}}\vert z_i-z_j\vert^2\,d\pi(z_i,z_j) \right)^{1/2}
\tag{1}
$$



여기서 $$\Pi(\mu_i,\mu_j)$$ 는 transport plan $$\pi$$ 들의 집합으로, 각각의 transport plan $$\pi$$  는 모든 Borel subset $$A$$ 와 $$B$$ 에 대해 $$\pi(A\times\mathcal{Z})=\mu_i(A)$$  와 $$\pi(\mathcal{Z}\times B)=\mu_j(B)$$ 를 만족합니다. 



특히 $$\mu_i$$ 가 Lebesgue measure 에 대해 absolutely continuous 하다면, Brenier theorem [7]  에 의해 $$(1)$$ 의 정의는 다음과 동일합니다.
$$
\mathcal{W}_2(\mu_i,\mu_j) = \left( \inf_{f\in MP(\mu_i,\mu_j)} \int_{\mathcal{Z}}\vert z-f(z)\vert^2\,d\mu_i(z) \right)^{1/2}
\tag{2}
$$


여기서 $$MP(\mu_i,\mu_j)=\left\{ f:\mathcal{Z}\rightarrow\mathcal{Z} \mid \mu_j(B)=\mu_i(f^{-1}(B)) \;\; \text{for any Borel set B} \right\}$$ 으로 정의하며, $$f\in MP(\mu_i,\mu_j)$$ 를 transport map 이라고 부릅니다. 특히 Brenier theorem 에 의해 optimal transport plan $$\pi^{\ast}$$ 는 유일하게 존재하며, $$(2)$$ 를 만족하는 optimal transport map 또한 유일하게 존재합니다. 이를 만족하는 optimal transport map 을 Monge map 이라고 정의합니다.





### LOT Framework



Linear Optimal Transportation (LOT) framework [2] 는 기존의 optimal transport metric 을 더 빠르게 계산하기 위해 제시된 방법입니다. 기존의 방법은 $$M$$ 개의 그래프들에 대해 그래프의 각 쌍마다 2-Wasserstein distance 를 구하려면 총 $$M(M-1)/2$$ 번의 거리 계산이 필요하기 때문에, large-scale dataset 에 적용하기 힘듭니다.



이를 해결하기 위해, LOT 는 isometric linear embedding 을 통해 



LOT 는 manifold 위에서의 geodesic distance 를 직접 계산하는 것이 아니라, manifold 에 대한 tangent space 에서의 거리를 계산합니다. 







LOT 는 크게 두 가지 장점이 있습니다. 첫 번째는 두 probability measure 의 2-Wasserstein distance 와 tangent space 에서 projection 들의  Euclidean distance 가 같기 때문에, measure 사이의 거리 계산이 쉬워집니다. 특히 기존의 방법으로는 $$M$$ 개의 그래프에 대해 $$M(M-1)/2$$ 개의 LP 를 풀어야했지만, LOT 를 통해 $$M$$ 개의 LP 만을 풀면 됩니다.



또한 LOT 를 통해 tangent space 로의 linear embedding 을 얻을 수 있습니다. 

embedding 에 대해 Euclidean structure 에 적용할 수 있는 PCA 혹은 LDA 와 같은 알고리즘들을 사용할 수 있습니다.







&nbsp;

## Linear Wasserstein Embedding

&nbsp;

Lebesgue measure 에 대해 absolutely continuous 한 reference measure $$\mu_0$$ 에 대해,



$$f_i$$ 를 $$\mu_0$$ 로부터 $$\mu_i$$ 로의 Monge map 이라고 하면, 정의에 의해 $$f_i$$ 는 다음과 같이 쓸 수 있습니다.
$$
f_i = \underset{f\in MP(\mu_0,\mu_i)}{\arg\!\min} \int_{\mathcal{Z}} \vert z-f(z)\vert^2\,d\mu_0(z)
$$



$$(0)$$ 의 projection 을 살짝 수정해 다음과 같이 projection $$\phi$$ 를 정의하면,
$$
\phi(\mu_i) := (f_i-Id)\sqrt{p_i}
$$

$$\phi$$ 는 다음과 같은 특징을 가집니다.



>1. $$\phi$$ provides an isometric embedding for probability measures
>2. $$\phi(\mu_0)=0$$
>3. $$\Vert \phi(\mu_i)-\phi(\mu_0)\Vert_2 = \Vert\phi(\mu_i)\Vert_2=\mathcal{W}_2(\mu_i,\mu_0)$$
>4. $$\Vert\phi(\mu_i)-\phi(\mu_j)\Vert_2\approx\mathcal{W}_2(\mu_i,\mu_j)$$ [3]



다음의 그림을 통해 projection $$\phi$$ 를 이해할 수 있습니다.

<p align='center'>
    <img src='../assets/post/Wasserstein-Embedding-For-Graph-Learning/projection.PNG' style='max-width: 60%; height:auto'>
</p>








probability distribution $$\{p_i\}^{M}_{i=1}$$ 들에 대해, distribution $$p_i$$ 로부터 뽑은 $$N_i$$ 개의 i.i.d sample 들로 이루어진 $$Z_i = \left[ z^i_1,\cdots,z^i_{N_i} \right]^T\in\mathbb{R}^{N_i\times d}$$  와 reference distribution $$p_0$$ 로부터 뽑은 $$N$$ 개의 i.i.d sample 들로 이루어진 $$Z_0 = \left[ z^0_1,\cdots,z^0_{N} \right]^T\in\mathbb{R}^{N\times d}$$ 를 



$$p_i$$ 와 $$p_0$$ 사이의 optimal transport plan $$\pi^{\ast}_i$$ 는 다음의 LP 를 통해 구할 수 있습니다.


$$
\pi^{\ast}_{i}
= \underset{\pi\in\Pi_i}{\arg\!\min} \sum^{N}_{j=1}\sum^{N_i}_{k=1} \pi_{jk}\Vert z^0_j - z^i_k\Vert^2_2
\tag{}
$$

$$
\text{for }\;\;\;
\Pi_i = \left\{ \pi\in\mathbb{R}^{N\times N_i} \mid N_i\sum^N_{j=1}\pi_{jk} = N\sum^{N_i}_{k=1}\pi_{jk}=1,\;\forall k\in\{1,\cdots,N_i\},\;\forall j\in\{1,\cdots,N\} \right\}
$$







barycentric projection 을 통해 $$(0)$$ 의 Monge map 을 근사하면, Appendix D

$$p_i$$ 와 $$p_0$$ 사이의 Monge map 은 다음과 같습니다.


$$
F_i = N\left( \pi^{\ast}_iZ_i \right)\in\mathbb{R}^{N\times d}
\tag{}
$$



$$()$$ 로부터 $$Z_i$$ 의 linear Wasserstein embedding $$\phi(Z_i)$$ 를 다음과 같이 계산할 수 있습니다.
$$
\phi(Z_i) = (F_i-Z_0)/\sqrt{N}\in\mathbb{R}^{N\times d}
\tag{}
$$



&nbsp;

## WEGL : A Linear Wasserstein Embedding for Graphs

&nbsp;

논문에서는 graph classification task 를 위한 Wasserstein Embedding for Graph Learning (WEGL) 를 제시합니다. WEGL 은 크게 5 개의 단계로 나눌 수 있습니다. 먼저 주어진 $$M$$ 개의 독립적인 그래프 $$\{G_i=(\mathcal{V}_i,\mathcal{E}_i)\}^{M}_{i=1}$$ 들은 diffusion layer 들을 거쳐 node embedding $$\{Z_i\}^{M}_{i=1}$$ 로 바뀝니다. 이 후 $$\{Z_i\}^{M}_{i=1}$$ 로부터 reference node embedding $$Z_0$$ 를 계산하고, $$Z_0$$ 에 대한 linear Wasserstein embedding $$\{\phi(Z_i)\}^{M}_{i=1}$$ 을 구합니다. 마지막으로, 최종적인 embedding $$\{\phi(Z_i)\}^{M}_{i=1}$$ 들은 classifier 를 통해 분류됩니다.



정리하면 WEGL 모델의 input 은 graph dataset, diffusion layer 의 수, final node embedding 의 local pooling 함수, 그리고 classifier 의 종류이며, output 은 그래프의 classification 결과입니다. 다음은 WEGL 의 과정을 표현한 그림입니다.

<p align='center'>
    <img src='../assets/post/Wasserstein-Embedding-For-Graph-Learning/WEGL.PNG' style='max-width: 100%; height:auto'>
</p>


### Node embedding



WEGL 의 첫 번째 단계는 그래프들의 node embedding 을 구하는 것입니다. 그래프의 node embedding 에는 다양한 방법이 존재하며 크게 parametric encoder 와 non-parametric encoder 로 나눌 수 있습니다. 만약 parametric encoder 를 사용한다면, 학습 과정에서 node embedding 이 달라질 때마다 새로 linear Wasserstein embedding 을 계산해야합니다. 따라서 WEGL 에서는 복잡한 계산을 줄이기 위해 non-parametric diffusion layer 를 사용합니다.



주어진 그래프 $$G=(\mathcal{V},\mathcal{E})$$ 의 node feature $$\{x_v\}_{v\in\mathcal{V}}$$ 들과 scalar edge feature $$\{w_{uv}\}_{(u,v)\in\mathcal{E}}$$ 에 대해, diffusion layer 는 다음과 같이 정의됩니다.
$$
x^{(l)}_v = \sum_{u\in N(v)\cup\{v\}}\frac{w_{uv}}{\sqrt{\text{deg}(u)\text{deg}(v)}}\,x^{(l-1)}_u
\tag{}
$$


Self-loop $$(v,v)$$ 를 포함해 scalar edge feature 가 주어지지 않은 edge $$(u,v)$$ 들에 대해서는 모두 1 로 설정해줍니다. 특히 $$(0)$$ 에서 $$\sqrt{\text{deg}(u)\text{deg}(v)}$$ 를 통해 noramlize 해주는 방법은  GCN 의 propagation rule 에서도 볼 수 있습니다.



만약 edge feature 가 scalar 가 아닌 multiple features $$w_{uv}\in\mathbb{R}^{F}$$ 로 주어진다면, $$()$$ 의 diffusion layer 를 다음과 같이 바꿔줍니다. 여기서 $$\text{deg}_f(u) = \sum_{v\in\mathcal{V}}w_{uv,f}$$ 로 정의합니다.

$$
x^{(l)}_v = \sum_{u\in N(v)\cup\{v\}}\left( \sum^F_{f=1}\frac{w_{uv,f}}{\sqrt{\text{deg}_f(u)\text{deg}_f(v)}}\right)\,x^{(l-1)}_u
\tag{}
$$



마지막으로 diffusion layer 들을 거친 node feature $$\left\{x^{(l)}_v\right\}^{L}_{l=0}$$ 들에 대한 local pooling $$g$$ 
$$
z_v = g\left( \left\{x^{(l)}_v\right\}^{L}_{l=0} \right) \in\mathbb{R}^d
$$


local pooling $$g$$ 로는 concatenation 또는 averaging 을 사용합니다.





$$
h\left(G_i\right) = Z_i = \begin{bmatrix}
z_1,\;\cdots\;,\;z_{\vert \mathcal{V}_i\vert}
\end{bmatrix}^T \in \mathbb{R}^{\vert\mathcal{V}_i\vert\times d}
$$

$$\left\{G_i=(\mathcal{V}_i,\mathcal{E}_i)\right\}^M_{i=1}$$ 들의 node embedding $$\left\{Z_i\right\}^M_{i=1}$$ 







### Reference Distribution





논문에서는 그래프들의 node embeddings $$\cup^M_{i=1} Z_i$$ 에 대해 $$N=\left\lfloor\frac{1}{M}\sum^M_{i=1}N_i \right\rfloor$$ 개의 centroid 들을 가지도록 $$k$$-means clustering 을 통해 reference node embedding $$Z_0$$ 을 계산합니다.



다른 방법으로는 node embedding 들에 대한 Wasserstein barycenter 혹은 normal distribution 으로부터 뽑은 $$N$$ 개의 sample 들로 reference node embedding 을 구성할 수 있습니다. 



linear Wasserstein embedding 의 결과는 reference 에 따라 달라지지만, 실험적으로 WEGL 의 성능은 reference 에 따라 큰 차이를 보이지 않습니다.



### Linear Wasserstein Embedding



$$()$$ 를 통해 reference embedding $$Z_0$$ 에 대한 linear Wasserstein embedding $$\phi(Z_i)\in\mathbb{R}^{N\times d}$$ 를 계산합니다.



그래프들의 최종적인 embedding 들은 차원이 모두 동일합니다.





### Classifier



linear Wasserstein embedding 을 통해 얻은 graph embedding $$\{\phi(Z_i)\}^{M}_{i=1}$$ 



WEGL 의 장점 중 하나는 task 에 맞는 classifier 를 선택할 수 있다는 점입니다.



논문에서는 classifier 로 AuotML, random forest, RBF kernel 을 이용한 SVM 을 사용했습니다.







&nbsp;

## Experimental Evaluation

&nbsp;

논문에서는 2 가지의 graph classification task 에 대해 WEGL 의 성능을 평가했습니다.



### Molecular Property Prediction



첫 번째 task 는 molecular property prediction task 입니다. 



ogbg-molhiv dataset 은 Open Graph Benchmark 의 일부로 dataset 의 각각의 그래프는 분자를 나타냅니다. 그래프의 node 는 원자를, edge 는 원자들 사이의 결합을 표현하며, 이로부터 각각의 분자가 HIV 를 억제하는지에 대해 이진분류하는 것이 목표입니다.



ogbg-molhiv dataset 에 대해 GNN baseline 모델들과 성능을 비교합니다. GNN baseline 으로는 GCN, GIN, DeeperGCN, HIMP 그리고 GCN 과

성능은 ROC-AUC 로 평가했습니다.



또한 ogbg-molhiv dataset 에 대해, 특별히 virtual node 를 사용하는 방법이 좋은 성능을 보여주기 때문에, 각각의 모델들의 virtual node variant 



실험에서는 모든 그래프들의 node embedding 의 평균과 분산을 통해 standardize 시켜 사용합니다.

계산을 줄이기 위해 node embedding 들에 대한 20차원 PCA 를 적용합니다.





ogbg-molhiv dataset 에 대한 실험 결과는 다음과 같습니다.

<p align='center'>
    <img src='../assets/post/Wasserstein-Embedding-For-Graph-Learning/ogbg-molhiv.PNG' style='max-width: 100%; height:auto'>
</p>
WEGL 에 AutoML classifier 를 사용했을 때 state-of-the-art performance 를 보여주며, random forest classifier 를 사용했을 때도 준수한 성능을 보여줍니다. 특히 GNN 모델들과 같이 end-to-end 학습 없이도 large-scale graph dataset 에 적용될 수 있음을 알 수 있습니다.



또한 linear Wasserstein embedding 의 효과를 입증하기 위한 ablatin study 를 진행했습니다. WEGL 에서 Wasserstein embedding 대신 global average pooling (GAP) 를 사용한 경우 test ROC-AUC 가 확연히 줄어드는 것을 확인할 수 있습니다.




### TUD Benchmark



두 번째 task 는 social network, bioinformatics, 그리고 molecular graph dataset 들에 대해 실험을 진행했습니다. 



GNN baseline 뿐만 아니라, graph classification 에서 좋은 성능을 보여주는 graph kernel 들을 함께 비교했습니다. WEGL 의 classifier 로는 random forest, RBF kernel 을 이용한 SVM, 그리고 GBDT 를 사용했습니다.



각각의 dataset 들에 대한 graph classification accuracy 는 다음의 표에서 확인할 수 있습니다.

<p align='center'>
    <img src='../assets/post/Wasserstein-Embedding-For-Graph-Learning/TUD.PNG' style='max-width: 100%; height:auto'>
</p>
WEGL 은 거의 state-of-the-art performance 에 근접한 성능을 가지며, 특히 모든 dataset 에 대해 top-3 performance 를 보여줍니다. 이로부터 다양한 domain 에서의 graph 들을 잘 학습할 수 있음을 보여줍니다.






### Computation Time



마지막으로 WEGL 의 computational efficiency 를 확인하기 위해 학습과 추론에서의 wall-clock time 을 GIN 과 WWL kernel [4] graph kernel 과 비교했습니다. 

dataset 의 그래프 수와 그래프들의 평균적인 node, edge 수에 따른 학습 / 추론 시간을 측정했습니다.



학습과 추론 시간 결과는 다음의 그래프에서 확인할 수 있습니다. 

<p align='center'>
    <img src='../assets/post/Wasserstein-Embedding-For-Graph-Learning/time.PNG' style='max-width: 100%; height:auto'>
</p>
WEGL 은 다른 모델들과 비교해 비슷하거나 훨씬 좋은 성능을 보여줍니다. 특히 그래프의 수가 많아질수록 GIN, WWL 과 비교해 학습 시간이 짧았습니다. GPU 를 사용한 GIN 과 비교해 추론 시간은 조금 길었지만, WEGL 이 CPU 를 사용한 점을 감안하면 그 차이는 크지 않습니다. 



  

&nbsp;

## Future Work

&nbsp;

end-to-end 학습이 불가능



optimal distance 의 가장 큰 약점은 rescaling, translation, rotation 과 같은 transformation 들에 대해 invariant 하지 못하다는 것입니다. 이를 보완하기 위해 나온 개념이 Gromov-Wasserstein distance 입니다.



Wasserstein distance 대신 Gromov-Wasserstein distance 를 사용한다면, permutation 에 대해 invariant 한 모델을 만들 수 있다고 기대합니다. 





entropy-regularized transport problem




&nbsp;

## Appendix

&nbsp;



### B. Wasserstein Space



wasserstein space definition



set of measures = Riemannian manifold



geodesic distance on the manifold = 2-Wasserstein distance 





### C. LOT Framework



고정된 reference measure $$\sigma$$ 에 대한 tangent space 로의 projection 을 $$P$$ 라고 하겠습니다.

주어진 measure $$\mu$$ 에 대해, $$\sigma$$ 와 $$\mu$$ 사이의 optimal transport map 을 $$\psi$$ 라 한다면, projection $$P$$ 를 다음과 같이 정의합니다. 
$$
P(\mu) = \psi-Id
$$




2-Wasserstein space 의 기하학적 특징에 의해, $$P(\sigma)$$ 로부터 $$P(\mu)$$ 까지의 거리는  manifold 에서의 $$\sigma$$ 와 $$\mu$$ 의 geodesic distance, 즉 2-Wasserstein distance 와 같습니다. 즉 projection $$P$$ 는 reference measure $$\sigma$$ 로부터의 거리를 보존하는 isometric mapping 이며, 이를 equidistant azimuthal projection 이라고도 부릅니다.



다음의 그림을 통해 manifold 위에서의 measure 들과 tangent space 에서의 projection 들에의 관계를 이해할 수 있습니다.

<p align='center'>
    <img src='../assets/post/Wasserstein-Embedding-For-Graph-Learning/LOT.PNG' style='max-width: 80%; height:auto'>
</p>



이 때 reference measure $$\sigma$$ 에 대한 LOT distance $$d_{LOT,\,\sigma}$$ 는 다음과 같이 정의합니다.
$$
d_{LOT,\,\sigma}(\mu,\nu) = \Vert P(\mu)-P(\nu)\Vert_{\sigma} \approx \mathcal{W}_2(\mu,\nu)
$$






> Linearized LOT 


$$
d_{LOT,\,\sigma}(\mu,\nu) = \left( \inf_{\gamma\in\Gamma(\mu,\nu,\sigma)} \int_{\mathcal{Z}\times\mathcal{Z}\times\mathcal{Z}} \vert z-z' \vert^2\,d\gamma(z,z',z'') \right)^{1/2}
$$


> Barycentric projection



<p align='center'>
    <img src='../assets/post/Wasserstein-Embedding-For-Graph-Learning/barycentric.PNG' style='max-width: 100%; height:auto'>
</p>



### D. Approximation of Monge Map



discrete measure 

$$\mu_0=\frac{1}{N}\sum^N_{l=1}\delta_{z^0_l}$$ 

$$\mu_i=\frac{1}{N_i}\sum^{N_j}_{n=1}\delta_{z^i_n}$$

$$\mu_0$$ 와 $$\mu_i$$ 사이의 optimal transportation plan $$\pi^{\ast}_i$$ 에 대해, 

$$\frac{1}{N}\delta_{z^0_j}$$ 의 


$$
\bar{z^0_j} = N\sum^{N_i}_{k=1}\pi^{\ast}_{jk}Z^i_k
\tag{}
$$


따라서 $$()$$ 로부터 Monge map 은 다음과 같이 쓸 수 있습니다.
$$
F_i = N(\pi^{\ast}_iZ_i)
$$


&nbsp;

## Reference

&nbsp;

1. Soheil Kolouri, Navid Naderializadeh, Gustavo K Rohde, and Heiko Hoffmann. [Wasserstein embedding for graph learning](https://arxiv.org/pdf/2006.09430.pdf). arXiv preprint arXiv:2006.09430, 2020.



2. Wei Wang, Dejan Slepcev, Saurav Basu, John A Ozolek, and Gustavo K Rohde. [A linear optimal transportation framework for quantifying and visualizing variations in sets of images](https://link.springer.com/article/10.1007/s11263-012-0566-z). International Journal of Computer Vision, 101(2):254–269, 2013.



3. Caroline Moosmüller and Alexander Cloninger. [Linear optimal transport embedding: Provable
   fast wasserstein distance computation and classification for nonlinear problems](https://arxiv.org/pdf/2008.09165.pdf). arXiv preprint
   arXiv:2008.09165, 2020.



4. Matteo Togninalli, Elisabetta Ghisu, Felipe Llinares-López, Bastian Rieck, and Karsten Borgwardt.
   [Wasserstein Weisfeiler-Lehman graph kernels](https://arxiv.org/pdf/1906.01277.pdf). In Advances in Neural Information Processing
   Systems, pp. 6436–6446, 2019.



5. G. Bécigneul, O.-E. Ganea, B. Chen, R. Barzilay, and T. Jaakkola. [Optimal Transport Graph Neural Networks](https://arxiv.org/pdf/2006.04804.pdf). arXiv preprint arXiv:2006.04804



6. WEGL Github code :  [https://github.com/navid-naderi/WEGL](https://github.com/navid-naderi/WEGL)



7. Peyré, G., Cuturi, M., et al. [Computational optimal transport](https://arxiv.org/pdf/1803.00567.pdf). Foundations and Trends® in Machine Learning, 11(5-6):355–607, 2019.

   



