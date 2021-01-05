---
title: Polynomial Approximation of Spectral Filtering
description: based on "Wavelets on Graphs via Spectral Graph Theory"
date: 2021-01-04 20:00:00 +0900
category:
- theory
tag:
- gnn
- polynomial approximation
- spectral filtering
---



Truncated Chebyshev expansion 을 통한 spectral filtering 의 polynomial approximation.



## 0. Graph Convolution and Spectral Filtering

Graph convolution 과 spectral filtering 에 대해 자세히 설명한 [포스트](https://harryjo97.github.io/theory/Graph-Convolution-and-Filtering/) 를 보고 오시면, 이번 포스트를 이해하는데 큰 도움이 될 것입니다. Graph signal processing 과 spectral graph wavelet transform 에 대해 더 공부하고 싶다면 [Wavelets on Graphs via Spectral Graph Theory](https://arxiv.org/pdf/0912.3848.pdf) 를 참고하기 바랍니다.



## 1. Chebyshev Polynomial



Chebyshev polynomial 은 다음과 같이 점화식[^1]으로 정의됩니다.

$$
T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)
\;\; \text{ with } \;\; T_0(x) = 1 ,\; T_1(x) = x
\tag{a}
$$



특히 Chebyshev polynomial 은 $$L^2\left( [-1,1],\, \frac{dx}{\sqrt{1-x^2}} \right)$$[^2] 의 orthogonal basis 를 이루기 때문에 $$h\in L^2\left( [-1,1],\, \frac{dx}{\sqrt{1-x^2}} \right)$$ 에 대해 uniformly convergent 한 Chebyshev expansion 이 존재합니다.

$$
h(x) = \frac{1}{2}c_0 + \sum^{\infty}_{k=1} c_kT_k(x)
\tag{b}
$$


$$(b)$$ 에서 Chebyshev coefficeint $$c_k$$ 는 다음과 같이 계산할 수 있습니다.

$$
c_k = \frac{2}{\pi}\int^1_{-1} \frac{T_k(x)h(x)}{\sqrt{1-x^2}}dx
$$

&nbsp;

## 2. Polynomial Approximation and Localization



입력 신호 $$f_{in}$$ 의 filter $$g$$ 에 대한 spectral filtering 의 결과 $$f_{out}$$ 은 다음과 같습니다.

$$
\begin{align}

f_{out} 
= U\hat{g}(\Lambda)U^T\;f_{in} \tag{$1$}

\end{align}
$$

이 포스트에서는 편의상 $$\hat{g}$$ 을 $$g_{\theta}$$ 로 대신 쓰겠습니다. 



> Localization of graph Laplacian



그래프 $$G$$ 의 vertex $$i$$ 와 $$j$$ 에 대해, $$d_G(i,j)$$ 를 $$i$$ 와 $$j$$ 를 연결하는 모든 path 들 중 edge 들의 수가 가장 적은 path 의 길이로 정의합니다. 

이 때, 그래프 $$G$$ 의 vertex $$i, \;j$$ 와 $$d_G(i,j)$$  보다 작은 모든 $$s$$ 에 대해 다음이 성립합니다.

$$
\left(L^s\right)_{ij} = 0
\tag{$2$}
$$

증명 과정에 대해 자세히 알고 싶다면,  [Wavelets on Graphs via Spectral Graph Theory](https://arxiv.org/pdf/0912.3848.pdf) 의 lemma 5.4 를 참고하기 바랍니다. 




> Localized filter in vertex domain



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

$$(3)$$ 에서 볼 수 있듯이, graph Laplacian $$L$$ 의 eigenvector 를 계산하지 않고도 spectral filtering 의 결과를 구할 수 있습니다.

특히 vertex $$i$$ 에 대한 spectral filtering 의 결과는 다음과 같습니다.

$$
f_{out}(i) 
= (g_{\theta}(L) f_{in})(i) 
= \sum^{K}_{k=0}\sum^{N}_{j=1} a_{k} \left(L^k\right)_{ij} f_{in}(j) 
\tag{$4$}
$$

graph Laplacian 의 localization $$(2)$$ 를 사용하면, $$(4)$$ 를  vertex $$i$$ 의 $$K$$ - hop local neighborhood $$N(i,K)$$ 에 대해 표현할 수 있습니다. 

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

&nbsp;

## 3. Truncated Chebyshev Expansion



$$(1)$$ 을 계산하기 위해서는  graph Laplacian $$L$$ 의 eigenvector 를 모두 찾아야 합니다. $$N$$ 개의 node 를 가지는 그래프에 대해서, QR decomposition 의 computational complexity 가 $$O(N^3)$$ 이기 때문에, node 가 수천개 혹은 수만개 이상인 그래프에 대해서는 직접 $$(1)$$ 을 계산하기는 힘듭니다. 따라서, 그래프의 크기가 큰 경우에는 $$(1)$$ 을 근사할 수 있는 효율적인 방법이 필요합니다. 저희의 목표는 $$(1)$$ 을 효율적으로 근사할 수 있는 $$g_{\theta}$$ 의 polynomial approximant $$p$$ 를 찾는 것입니다.



만약 $$p$$ 가 $$L$$ 의 spectrum 에 대한 upper bound $$\lambda_{max}$$[^3] 에 대해 다음의 조건을 만족한다면,

$$
\left\vert g_{\theta}(x) - p(x) \right\vert \leq B < \infty
\;\;\text{ for all }\;\; x\in [0,\lambda_{max}]
\tag{6}
$$



$$(3)$$ 을 참고하면, polynomial filter $$p(L)$$ 의 결과 $$\tilde{f}_{out} = p(L)f_{in}$$ 을 통해 다음과 같이 $$f_{out}$$ 을 근사할 수 있습니다. 

$$
\begin{align}
\vert f_{out}(i) - \tilde{f}_{out}(i) \vert 
&= \left\vert \sum_{l} g_{\theta}(\lambda_l)\hat{f}(\lambda_l)u_l(i) - \sum_{l} p(\lambda_l)\hat{f}(\lambda_l)u_l(i) \right\vert \\
\\
&\leq \sum_{l} \vert g_{\theta}(\lambda_l) - p(\lambda_l) \vert \left\vert \hat{f}(\lambda_l)u_l(i) \right\vert \\
&\leq B \left( \sum_l \left\vert \hat{f}(\lambda_l) \right\vert^2\sum_l \vert u_l(i) \vert^2 \right)^{1/2} 
= B\;\|f\| 
\tag{7}
\end{align}
$$


이 때 $$f_{out}$$ 과 $$\tilde{f}_{out}$$ 에 대한 오차 $$(7)$$ 을 줄이기 위해서는, $$g_{\theta}$$ 와 $$p$$ 에 대한 $$L_{\infty}$$ error $$(6)$$ 를  최소화해야 합니다. 만약  $$p$$ 가 order $$K$$ polynomial 이라면, $$(6)$$ 은 $$p$$ 가 minimax polynomial of order $$K$$ 일 때 최소가 됩니다. 이 때 truncated Chebyshev expansion 을 통해 minimax polynomial 에 대한 근사가 가능합니다. 



저희가 찾고자 하는  $$p$$ 로 $$g_{\theta}$$ 의 truncated Chebyshev expansion 을 선택할 수 있습니다. 하지만 $$g_{\theta}$$ 는 spectral domain 에서 정의된 함수이기 때문에, $$(b)$$ 를 적용하기 위해서는 domain 의 변환이 필요합니다. $$L$$ 의 eigenvalue 들은 모두 $$[0, \lambda_{max}]$$ 구간에 속하기 때문에 $$h_{\theta}$$ 를 다음과 같이 정의하면 $$g_{\theta}$$ 를 $$[-1,1]$$ 에서 정의된 함수로 바꿀 수 있습니다.

$$
h_{\theta}(x) = g_{\theta}\left( \frac{\lambda_{\max}}{2}(x+1) \right)
$$

$$(b)$$ 를 $$h_{\theta}$$ 에 적용하고 order $$K$$ 까지의 truncation 을 생각하면,

$$
h_{\theta}(x) \approx \frac{1}{2}c_0 + \sum^{K}_{k=1} c_kT_k(x)
$$

$$\tilde{L} = \frac{2}{\lambda_{max}}L - I$$ 에 대해 $$p$$ 를 다음과 같이 정의하면,

$$
p(\tilde{L}) = \frac{1}{2}c_0I + \sum^{\infty}_{k=1} c_kT_k(\tilde{L})
\tag{8}
$$

복잡한 eigenvector 의 계산 없이 spectral filtering 에 대한 근사가 가능합니다.

$$
\begin{align}
Ug_{\theta}(\Lambda)U^T 
&= Uh_{\theta}(\tilde{\Lambda})U^T  \\
&\approx Up(\tilde{\Lambda})U^T = p(\tilde{L})
\end{align}
$$

정리하면, spectral filtering 의 결과 $$f_{out}$$ 은 다음과 같이 근사할 수 있습니다.

$$
f_{out} = Ug_{\theta}(\Lambda)U^Tf_{in} \approx p(\tilde{L})f_{in}
\tag{9}
$$

&nbsp;

## 4. Advantage of using Chebyshev polynomial



$$(9)$$ 과 같이 truncated Chebyshev expansion 을 통한 spectral filtering 의 근사는 다음과 같은 세 가지 이점이 있습니다.




> Fast filtering using recurrence relation

Chebyshev polynomial 의 중요한 특성은 $$(a)$$ 의 점화식을 통해 얻을 수 있다는 것입니다. Graph Laplacian $$L$$ 에서부터 시작해 재귀적 연산으로 order $$K$$ polynomial $$T_K$$ 까지 구하는 computational cost 는 $$L$$ 이 sparse matrix  [^4]일 때 $$O(K\vert E\vert)$$ 입니다. 




> Localized in vertex domain

$$(5)$$ 에서 보았듯이 $$g_{\theta}$$ 대신 polynomial approximant 를 사용한 filter 는 vertex domain 에서 localized 되어있습니다. 이를 통해 CNN 의 중요한 특성인 locality 가 그래프에서 일반화될 수 있습니다.




> Learnable filter

$$(8)$$ 의 coefficient $$c_k$$ 를 parameter 로 학습하는 neural network 를 만들 수 있습니다. [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/pdf/1606.09375.pdf) 에 이 방법이 자세히 설명되어 있습니다.

&nbsp;

## 5. Lanczos Algorithm 

$$(1)$$ 을 효율적으로 계산하는 다른 해결 방법으로는 Lanczos Algorithm 이 있습니다. [LanczosNet: Multi-Scale Deep Graph Convolutional Networks](https://arxiv.org/pdf/1901.01484.pdf) 의 paper review 를 통해 더 자세히 설명하겠습니다.


&nbsp;

## 6. Next

다음 포스트에서는 [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf) 의 paper review 를 하겠습니다.


&nbsp;
&nbsp;


[^1]: recurrence relation.
[^2]: Hilbert space of square integrable functions
[^3]: Spectrum 의 upper bound $$\lambda_{max}$$ 는 Arnoldi iteration 혹은 Jacobi-Davidson method 등을 사용하면 $$L$$ 의 spectrum 전체를 찾는 것에 비해서 훨씬 쉽게 구할 수 있습니다.

[^4]: 많은 경우 graph Laplacian 은 sparse matrix 로 표현 가능합니다.

