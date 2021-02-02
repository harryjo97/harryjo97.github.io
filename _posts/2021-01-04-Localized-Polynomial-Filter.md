---
title: Localized Polynomial Filter
date: 2021-01-04 20:00:00 +0900
category:
- theory
tag:
- GCN
---

Graph Convolutional Network 이해하기 : (4) Localized Polynomial Filter



## Localization of Graph Laplacian



그래프 $$G$$ 의 vertex $$i$$ 와 $$j$$ 에 대해, 두 vertex 사이의 거리 $$d_G(i,j)$$ 를 $$i$$ 와 $$j$$ 를 연결하는 모든 path 들 중 edge 들의 수가 가장 적은 path 의 길이로 정의합니다. 


**(Lemma 1)**  그래프 $$G$$ 의 adjacency matrix $$A$$ 에 대해 $$\tilde{A}$$ 를 다음과 같다면,

$$
\tilde{A} = \begin{cases}
A_{ij} &\mbox{ if } i\neq j \\
1 &\mbox{ if } i=j
\end{cases}
$$

임의의 양의 정수 $$s$$ 에 대해, $$\left( \tilde{A}^s \right)_{ij}$$ 은 vertex $$i$$ 와 $$j$$ 를 연결하는 path 들 중 길이가  $$s$$ 이하인 path 들의 수와 일치합니다.


**(Lemma 2)**  $$N\times N$$ matrix $$A$$ , $$B$$ 와 모든 $$1\leq m,n\leq N$$ 에 대해, $$B_{mn}=0$$ 이면 $$A_{mn}=0$$ 을 만족한다면, 임의의 양의 정수 $$s$$ 에 대해서도 $$\left( B^s \right)_{mn}=0$$ 이면  $$\left( A^s \right)_{mn}=0$$ 이 성립합니다.


위의 두 lemma 를 사용하면, graph Laplacian $$L$$ 의 localization 을 보일 수 있습니다. 


**(Localization of graph Laplacian)**  그래프 $$G$$ 의 vertex $$i, \;j$$ 와 $$d_G(i,j)$$  보다 작은 모든 $$s$$ 에 대해 다음이 성립합니다.

$$
\left(L^s\right)_{ij} = 0
\tag{1}
$$

&nbsp;

## Localized Polynomial Filter



Filter $$g_{\theta}$$ 에 대한 spectral convolution 의 결과 $$f_{out}$$ 은 다음과 같습니다.

$$
f_{out} = Ug_{\theta}(\Lambda)U^T\;f_{in} 
\tag{2}
$$



만약 filter $$g_{\theta}$$ 가 order $$K$$ polynomial $$\sum^K_{k=0} a_k x^k$$ 이라면, $$(2)$$ 를 다음과 같이 정리할 수 있습니다.

$$
\begin{align}
f_{out}
&= U\left(\sum^{N-1}_{k=0} a_k\Lambda^k \right)U^T f_{in} \\
&= \sum^{K}_{k=0} a_k\left( U\Lambda U^T \right)^k f_{in} 
= g_{\theta}(L)f_{in}
\tag{3}
\end{align}
$$

$$(3)$$ 에서 볼 수 있듯이, Fourier basis $$U$$ 를 직접 계산하지 않고도 $$(2)$$ 의 결과를 얻을 수 있습니다. 

특히 vertex $$i$$ 에 대해서만 자세히 살펴보겠습니다.

$$
f_{out}(i) 
= (g_{\theta}(L) f_{in})(i) 
= \sum^{K}_{k=0}\sum^{N}_{j=1} a_{k} \left(L^k\right)_{ij} f_{in}(j) 
\tag{4}
$$


Vertex $$i$$ 로부터 거리가 $$K$$ 이하인 vertex 들의 집합을 $$N(i,K)$$ 라고 하고, 이를 $$i$$ 의 $$K$$- hop local neighborhood 라고 부르겠습니다. 만약 vertex $$j$$ 가 $$N(i,K)$$ 의 원소가 아니라면, $$(1)$$ 에 의해 $$(L^k)_{ij}=0$$ 입니다.

따라서, $$(4)$$ 를 정리하면 다음과 같습니다.

$$
\begin{align}
f_{out}(i) 
&= \sum^N_{j=1}\sum^K_{k=0} a_k(L^k)_{ij}f_{in}(j) \\
&= \sum_{j\in N(i,K)}\left[\sum^K_{k=0} a_k(L^k)_{ij}\right]f_{in}(j) \\
&= \sum_{j\in N(i,K)} b_{ij}f_{in}(j)
\tag{5}
\end{align}
$$


결국 $$f_{out}(i)$$ 는 $$i$$ 의 $$K$$- hop local neighborhood 원소들만을 이용해서 표현할 수 있습니다. 즉 $$f_{out}(i)$$ 를 계산하기 위해 모든 vertex 들의 정보를 사용하지 않아도 된다는 뜻입니다. CNN 의 convolutional fiter 가 각 픽셀을 중심으로 주변의 픽셀 값만을 사용하는 것과 같은 맥락입니다.

&nbsp;

## Reference

1. David K Hammond, Pierre Vandergheynst, and Remi Gribonval. [Wavelets on graphs via spectral
   graph theory](https://arxiv.org/pdf/0912.3848.pdf). Applied and Computational Harmonic Analysis, 30(2):129–150, 2011.



2. D. Shuman, S. Narang, P. Frossard, A. Ortega, and P. Vandergheynst. [The Emerging Field of Signal
   Processing on Graphs: Extending High-Dimensional Data Analysis to Networks and other Irregular Domains](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6494675). _IEEE Signal Processing Magazine_, 30(3):83–98, 2013.



