---
title: Polynomial Approximation of Spectral Filtering
description: based on "Wavelets on Graphs via Spectral Graph Theory"
date: 2021-01-03 18:00:00 +0900
category:
- theory
tag:
- gnn
- polynomial approximation
- spectral filtering
---



Chebyshev polynomial 을 이용한 spectral filtering 의 polynomial approximation.



## 0. Graph Convolution and Filtering

Graph convolution 과 spectral filtering 에 대해 자세히 설명한 [포스트](https://harryjo97.github.io/theory/Graph-Convolution-and-Filtering/) 를 보고 오시면, 이번 포스트를 이해하는데 큰 도움이 될 것입니다. 이번 포스트는 Spectral Graph Wavelet Transform 에 대해 다루는 [Wavelets on Graphs via Spectral Graph Theory](https://arxiv.org/pdf/0912.3848.pdf) paper 에서 polynomial approximation 파트를 자세히 설명하였습니다. 특히 graph signal processing 관점에 대해 더 공부하고 싶다면 이 paper 를 참고하기 바랍니다.



## 1. Polynomial Approximation



입력 신호 $$f_{in}$$ 의 filter $$g_{\theta}$$ 에 대한 spectral filtering 의 결과 $$f_{out}$$ 은 다음과 같습니다.

$$
\begin{align}

f_{out} 
= U\hat{g}_{\theta}(\Lambda)U^T\;f_{in} \tag{$1$}

\end{align}
$$



이 때, $$(1)$$ 을 계산하기 위해서는  graph Laplacian $$L$$ 의 eigenvector 를 모두 찾아야 합니다. $$N$$ 개의 node 를 가지는 그래프에 대해서, QR decomposition 의 computational complexity 가 $$O(N^3)$$ 이기 때문에, node 가 수천개 수만개 이상인 그래프에 대해서는 $$(1)$$ 을 계산하기 힘듭니다.



따라서, 그래프의 크기가 큰 경우에는 $$(1)$$ 을 근사할 수 있는 효율적인 방법이 필요합니다 $$(\ast)$$. [Wavelets on Graphs via Spectral Graph Theory](https://arxiv.org/pdf/0912.3848.pdf) 에서는 Chebyshev polynomial 과 polynomial approximation 을 이용한 해결법을  제시합니다. 



주어진 그래프의 graph Laplacian $$L$$ 과 주어진 filter $$g_{\theta}$$ 에 대해, $$g_{\theta}$$ 의 polynomial approximant $$p$$ 가 다음의 조건을 만족한다고 가정합니다.
$$
\left\vert g(\theta x) - p(x) \right\vert \leq B < \infty
\;\;\text{ for all }\;\; x\in [0,\lambda_{max}]
\tag{$2$}
$$

$$\lambda_{max}$$ 는 $$L$$ 의 spectrum 에 대한 upper bound 입니다. $$\lambda_{max} \geq \lambda_{N-1}$$ 

$$\tilde{f}_{out} = p(L)f$$ 에 대해 다음의 부등식이 성립합니다.


$$
\begin{align}
\vert f_{out}(i) - \tilde{f}_{out}(i) \vert 
&\leq \left\vert \sum_{l} g(\theta\lambda_l)\hat{f}(\lambda_l)u_l(i) - \sum_{l} p(\lambda_l)\hat{f}(\lambda_l)u_l(i) \right\vert \\
&\leq \sum_{l} \vert g(\theta\lambda_l) - p(\lambda_l) \vert \left\vert \hat{f}(\lambda_l)u_l(i) \right\vert \\
&\leq B \left( \sum_l \left\vert \hat{f}(\lambda_l) \right\vert^2\sum_l \vert u_l(i) \vert^2 \right)^{1/2} 
= B\;\|f\| 
\tag{$3$}
\end{align}
$$


Spectrum 의 upper bound $$\lambda_{max}$$ 는 Arnoldi iteration 혹은 Jacobi-Davidson method 등을 사용하면 쉽게 구할 수 있습니다.



$$p(x)$$ 가 $$M$$ 차 다항식이라면,

$$(3)$$ 의 approximation error 는 $$p(x)$$ 가 $$M$$ 차 minimax polynomial 일 때 최소가 됩니다.





## 2. Localized Polynomial Filter





## 3. Lanczos Algorithm 

$$(\ast)$$ 의 다른 해결 방법으로는 Lanczos Algorithm 이 있습니다. [LanczosNet: Multi-Scale Deep Graph Convolutional Networks](https://arxiv.org/pdf/1901.01484.pdf) 의 paper review 포스트를 통해 더 자세히 설명하겠습니다.




## 4. Next

다음 포스트에서는 [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf) 의 paper review 를 하겠습니다.