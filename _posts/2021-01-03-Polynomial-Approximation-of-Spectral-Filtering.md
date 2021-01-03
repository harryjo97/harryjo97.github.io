---
title: Polynomial Approximation of Spectral Filtering
description: based on "Wavelets on Graphs via Spectral Graph Theory"
date: 2021-01-04 18:00:00 +0900
category:
- theory
tag:
- gnn
- polynomial approximation
- spectral filtering
---



Chebyshev polynomial 을 이용한 spectral filtering 의 polynomial approximation.



## 0. Graph Convolution and Filtering

Graph convolution 과 spectral filtering 에 대해 자세히 설명한 [포스트](https://harryjo97.github.io/theory/Graph-Convolution-and-Filtering/) 를 보고 오시면, 이번 포스트를 이해하는데 큰 도움이 될 것입니다. Graph signal processing 과 spectral graph wavelet transform 에 대해 더 공부하고 싶다면 [Wavelets on Graphs via Spectral Graph Theory](https://arxiv.org/pdf/0912.3848.pdf) 를 참고하기 바랍니다.



## 1. Polynomial Approximation



입력 신호 $$f_{in}$$ 의 filter $$g_{\theta}$$ 에 대한 spectral filtering 의 결과 $$f_{out}$$ 은 다음과 같습니다.

$$
\begin{align}

f_{out} 
= U\hat{g}_{\theta}(\Lambda)U^T\;f_{in} \tag{$1$}

\end{align}
$$



이 때, $$(1)$$ 을 계산하기 위해서는  graph Laplacian $$L$$ 의 eigenvector 를 모두 찾아야 합니다. $$N$$ 개의 node 를 가지는 그래프에 대해서, QR decomposition 의 computational complexity 가 $$O(N^3)$$ 이기 때문에, node 가 수천개 수만개 이상인 그래프에 대해서는 직접 $$(1)$$ 을 계산하기는 힘듭니다.



따라서, 그래프의 크기가 큰 경우에는 $$(1)$$ 을 근사할 수 있는 효율적인 방법이 필요합니다 $$(\ast)$$. [Wavelets on Graphs via Spectral Graph Theory](https://arxiv.org/pdf/0912.3848.pdf) 에서는 Chebyshev polynomial 을 이용한 해결법을  제시합니다. 



주어진 그래프의 graph Laplacian $$L$$ 과 주어진 filter $$g_{\theta}$$ 에 대해, 우리는 $$U\hat{g}_{\theta}(\Lambda)U^T$$ 를 approximate 할 수 있는 polynomial $$p$$ 를 찾으려고 합니다. 만약 $$\hat{g}_{\theta}$$ 의 polynomial approximant $$p$$ 가 $$L$$ 의 spectrum 에 대한 upper bound $$\lambda_{max}$$ 에 대해 다음의 조건을 만족한다면,

$$
\left\vert \hat{g}_{\theta}(x) - p(x) \right\vert \leq B < \infty
\;\;\text{ for all }\;\; x\in [0,\lambda_{max}]
\tag{$2$}
$$

$$\tilde{f}_{out} = p(L)f$$ 에 대해 다음의 부등식이 성립합니다.


$$
\begin{align}
\vert f_{out}(i) - \tilde{f}_{out}(i) \vert 
&\leq \left\vert \sum_{l} \hat{g}_{\theta}(\lambda_l)\hat{f}(\lambda_l)u_l(i) - \sum_{l} p(\lambda_l)\hat{f}(\lambda_l)u_l(i) \right\vert \\
&\leq \sum_{l} \vert \hat{g}_{\theta}(\lambda_l) - p(\lambda_l) \vert \left\vert \hat{f}(\lambda_l)u_l(i) \right\vert \\
&\leq B \left( \sum_l \left\vert \hat{f}(\lambda_l) \right\vert^2\sum_l \vert u_l(i) \vert^2 \right)^{1/2} 
= B\;\|f\| 
\tag{$3$}
\end{align}
$$


$$(2)$$ 에서 등장하는 spectrum 의 upper bound $$\lambda_{max}$$ 는 Arnoldi iteration 혹은 Jacobi-Davidson method 등을 사용하면, 전체 spectrum 을 찾는 것에 비해서 훨씬 쉽게 구할 수 있습니다.



$$f_{out}$$ 과 $$\tilde{f}_{out}$$ 에 대한 오차 $$(3)$$ 을 줄이기 위해서는, $$\hat{g}_{\theta}$$ 와 $$p$$ 에 대한 $$L_{\infty}$$ error $$(2)$$ 를  최소화해야 합니다. 

만약  $$p$$ 가 polynomial of order $$M$$ 이라면, $$(2)$$ 는 $$p$$ 가 minimax polynomial of order $$M$$ 일 때 최소가 됩니다.



truncated Chebyshev expansion 으로 minimax polynomial 에 대한 approximate 



Chebyshev polynomial 은 다음과 같은 점화식[^1]으로 정의됩니다.

$$
T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)
\;\; \text{ with } \;\; T_0(x) = 1 ,\; T_1(x) = x
\tag{4}
$$

특히 Chebyshev polynomial 은 $$L^2\left( [-1,1],\, \frac{dx}{\sqrt{1-x^2}} \right)$$[^2] 의 orthogonal basis 를 이루기 때문에 $$h\in L^2\left( [-1,1],\, \frac{dx}{\sqrt{1-x^2}} \right)$$ 에 대해 uniformly convergent 한 Chebyshev 

$$
h(x) = \frac{1}{2}c_0 + \sum^{\infty}_{k=1} c_kT_k(x)
$$


Chebyshev coefficeint $$c_k$$ 는 다음과 같습니다.

$$
c_k = \frac{2}{\pi}\int^1_{-1} \frac{T_k(x)h(x)}{\sqrt{1-x^2}}dx
$$



$$
\hat{g}_{\theta}(\Lambda) \;\approx \; \frac{1}{2}c_0I + \sum^{K}_{k=1} c_kT_k(\tilde{\Lambda})
$$







## 2. Localized Polynomial Filter





## 3. Lanczos Algorithm 

$$(\ast)$$ 의 다른 해결 방법으로는 Lanczos Algorithm 이 있습니다. [LanczosNet: Multi-Scale Deep Graph Convolutional Networks](https://arxiv.org/pdf/1901.01484.pdf) 의 paper review 포스트를 통해 더 자세히 설명하겠습니다.




## 4. Next

다음 포스트에서는 [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf) 의 paper review 를 하겠습니다.





[^1]: recurrence relation.
[^2]: Hilbert space of square integrable functions

