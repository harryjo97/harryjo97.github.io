---
title: Localized Polynomial Filter
date: 2021-01-04 20:00:00 +0900
category:
- theory
tag:
- spectral filtering
- polynomial filter
---

Graph convolutional Network 이해하기 : (4) Localized Polynomial Filter



## Localization of Graph Laplacian



그래프 $$G$$ 의 vertex $$i$$ 와 $$j$$ 에 대해, $$d_G(i,j)$$ 를 $$i$$ 와 $$j$$ 를 연결하는 모든 path 들 중 edge 들의 수가 가장 적은 path 의 길이로 정의합니다. 



> Lemma 1.
>
> 그래프 $$G$$ 의 adjacency matrix $$A$$ 에 대해 $$\tilde{A}$$ 를 다음과 같이 정의하겠습니다.
> $$
> \tilde{A} = \begin{cases}
> A_{ij} &\mbox{ if } i\neq j \\
> 1 &\mbox{ if } i=j
> \end{cases}
> $$
> 그러면 $$s>0$$ 에 대해, $$\left( \tilde{A}^s \right)_{ij}$$ 은 vertex $$i$$ 와 $$j$$ 를 연결하는 path 들 중 길이가  $$s$$ 이하인 path 들의 수와 같습니다.



> Lemma 2. 
>
> $$N\times N$$ matrix $$A$$ 와 $$B$$ 에 대해 $$B_{mn}=0$$ 이면 $$A_{mn}=0$$ 을 만족한다고 가정하겠습니다. 그러면 모든 $$s>0$$ 에 대해 $$\left( B^s \right)_{mn}=0$$ 이면  $$\left( A^s \right)_{mn}=0$$ 이 성립합니다.



위의 두 lemma 를 사용하면, 다음의 $$L$$ 의 localization 을 보일 수 있습니다. 

> 그래프 $$G$$ 의 vertex $$i, \;j$$ 와 $$d_G(i,j)$$  보다 작은 모든 $$s$$ 에 대해 다음이 성립합니다.
> $$
> \left(L^s\right)_{ij} = 0
> \tag{1}
> $$
> 





$$N\times N$$ matrix $$B$$ 를 다음과 같이 정의하겠습니다.
$$
B_{ij} = \begin{cases}
1 &\mbox{ if } L_{ij}\neq 0 \\
0 &\mbox{ if } L_{ij}= 0
\end{cases}
$$
$$L$$ 의 정의에 의해 $$B$$ 는 그래프 $$G$$ 에 대한 adjacency matrix $$A$$ 를 변형한 $$\tilde{A}$$ 와 같습니다. 그러므로, Lemma 1 에 의해 $$\left( B^s \right)_{ij}=0$$ 입니다. Lemma 2 를 사용하면 $$\left( B^s \right)_{ij}=0$$ 이므로 $$\left( L^s \right)_{ij}=0$$ 임을 알 수 있습니다.  



&nbsp;

## Localized Polynomial Filter



$$g_{\theta}$$ 에 대한 spectral filtering 의 결과 $$f_{out}$$ 은 다음과 같습니다.

$$
f_{out} = Ug_{\theta}(\Lambda)U^T\;f_{in} \tag{2}
$$



만약 $$g_{\theta}$$ 가 order $$K$$ polynomial 이라면, $$g_{\theta}(x) = \sum^K_{k=0} a_k x^k$$ 를 $$(1)$$ 에 넣어 정리할 수 있습니다.

$$
\begin{align}
f_{out}
&= U\left(\sum^{N-1}_{k=0} a_k\Lambda^k \right)U^T f_{in} \\
&= \sum^{K}_{k=0} a_k\left( U\Lambda U^T \right)^k f_{in} 
= g_{\theta}(L)f_{in}
\tag{3}
\end{align}
$$

$$(3)$$ 에서 볼 수 있듯이, graph Laplacian $$L$$ 의 eigenvector 를 계산하지 않고도 spectral filtering 의 결과를 구할 수 있습니다. 특히 vertex $$i$$ 에 대해서 보면,

$$
f_{out}(i) 
= (g_{\theta}(L) f_{in})(i) 
= \sum^{K}_{k=0}\sum^{N}_{j=1} a_{k} \left(L^k\right)_{ij} f_{in}(j) 
\tag{$4$}
$$

graph Laplacian 의 localization $$(1)$$ 을 사용하면, $$(4)$$ 를  vertex $$i$$ 의 $$K$$ - hop local neighborhood $$N(i,K)$$ 에 대해 표현할 수 있습니다. 

$$
\begin{align}
f_{out}(i) 
&= \sum^{N}_{j=1} \sum^{K}_{k=d_G(i,j)} a_k\left(L^k\right)_{ij} f_{in}(j) \\
\\
&= \sum_{j\in N(i,K)} b_{ij} f_{in}(j)
\end{align}
\tag{5}
$$

즉 $$f_{out}(i)$$ 는 $$i$$ 의 K - localized neighborhood 의 vertices $$j$$ 에 대해 $$f_{in}(j)$$ 들의 합으로 표현할 수 있습니다. 



따라서, spectral filter $$g_{\theta}$$ 가 order $$K$$ polynomial 이라면 filter 가 vertex domain 에서 $$K$$ - localized 된다는 것을 확인할 수 있습니다.



## Reference

1. David K Hammond, Pierre Vandergheynst, and Remi Gribonval. [Wavelets on graphs via spectral
   graph theory](https://arxiv.org/pdf/0912.3848.pdf). Applied and Computational Harmonic Analysis, 30(2):129–150, 2011.



2. D. Shuman, S. Narang, P. Frossard, A. Ortega, and P. Vandergheynst. [The Emerging Field of Signal
   Processing on Graphs: Extending High-Dimensional Data Analysis to Networks and other Irregular Domains](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6494675). _IEEE Signal Processing Magazine_, 30(3):83–98, 2013.



