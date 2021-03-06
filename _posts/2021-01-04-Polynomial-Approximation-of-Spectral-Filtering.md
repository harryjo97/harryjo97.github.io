---
title: Polynomial Approximation Using Chebyshev Expansion
date: 2021-01-04 22:00:00 +0900
category:
- theory
tag:
- GCN
---

Graph Convolutional Network 이해하기 : (5) Polynomial approximation using Chebyshev expansion



## Chebyshev Polynomial



Chebyshev polynomial $$\{T_{k}(x)\}_{k\geq 0}$$ 는 다음과 같이 점화식으로 정의됩니다.

$$
T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)
\;\; \text{ with } \;\; T_0(x) = 1 ,\; T_1(x) = x
\tag{1}
$$



특히 Chebyshev polynomial $$\{T_{k}(x)\}_{k\geq 0}$$ 는 $$L^2\left( [-1,1],\, \frac{dx}{\sqrt{1-x^2}} \right)$$ 의 orthogonal basis 를 이루기 때문에 $$h\in L^2\left( [-1,1],\, \frac{dx}{\sqrt{1-x^2}} \right)$$ 에 대해, 다음과 같이 uniformly convergent 한 Chebyshev expansion 이 존재합니다.

$$
h(x) = \frac{1}{2}c_0 + \sum^{\infty}_{k=1} c_kT_k(x)
\tag{2}
$$


$$(2)$$ 에서 Chebyshev coefficeint $$c_k$$ 는 다음과 같이 계산할 수 있습니다.

$$
c_k = \frac{2}{\pi}\int^1_{-1} \frac{T_k(x)h(x)}{\sqrt{1-x^2}}dx
\tag{$\ast$}
$$

&nbsp;

## Truncated Chebyshev Expansion



Filter $$g_{\theta}$$ 에 대한 spectral convolution 의 결과 $$f_{out}$$ 은 다음과 같습니다.

$$
f_{out} 
= Ug_{\theta}(\Lambda)U^T\;f_{in}
\tag{3}
$$

$$(3)$$ 을 계산하기 위해서는 Fourier basis $$U$$ 가 필요하기 때문에, graph Laplacian $$L$$ 의 eigenvector 를 모두 찾아야 합니다. $$N$$ 개의 node 를 가지는 그래프에 대해서, QR decomposition 과 같은 eigenvalue decomposition 의 computational complexity 의 $$O(N^3)$$ 입니다. 즉 node 가 수천개 혹은 수만개 이상인 그래프에 대해서 직접 $$(3)$$ 을 계산하는 것은 현실적으로 불가능합니다.




따라서, 그래프의 크기가 큰 경우에는 $$(3)$$ 을 근사할 수 있는 효율적인 방법이 필요합니다. 만약 $$g_{\theta}$$ 가 order $$K$$ polynomial $$\sum^K_{k=0} a_k x^k$$ 이라면, $$(3)$$ 을 다음과 같이 쓸 수 있습니다.

$$
\begin{align}
f_{out}
&= U\left(\sum^{N-1}_{k=0} a_k\Lambda^k \right)U^T f_{in} \\
&= \sum^{K}_{k=0} a_k\left( U\Lambda U^T \right)^k f_{in} 
= g_{\theta}(L)f_{in}
\tag{4}
\end{align}
$$

$$(4)$$ 에서 볼 수 있듯이, Fourier basis $$U$$  없이도 $$(3)$$ 의 결과를 얻을 수 있습니다. 만약 $$g_{\theta}$$ 에 대한 polynomial approximant $$p$$ 를 찾을 수 있다면, $$p$$ 를 사용해 $$(4)$$ 와 같이 $$(3)$$ 을 효율적으로 근사할 수 있습니다.




만약 $$p$$ 가 $$L$$ 의 spectrum 에 대한 upper bound $$\lambda_{max}$$ 에 대해 다음의 조건을 만족한다면,

$$
\left\vert g_{\theta}(x) - p(x) \right\vert \leq B < \infty
\;\;\text{ for all }\;\; x\in [0,\lambda_{max}]
\tag{5}
$$


Polynomial $$p(L)$$ 과의 spectral convolution $$\tilde{f}_{out} = p(L)f_{in}$$ 을 통해, 다음과 같이 $$(3)$$ 을 근사할 수 있습니다 [3].

$$
\begin{align}
\vert f_{out}(i) - \tilde{f}_{out}(i) \vert 
&= \left\vert \sum_{l} g_{\theta}(\lambda_l)\hat{f}(\lambda_l)u_l(i) - \sum_{l} p(\lambda_l)\hat{f}(\lambda_l)u_l(i) \right\vert \\
\\
&\leq \sum_{l} \vert g_{\theta}(\lambda_l) - p(\lambda_l) \vert \left\vert \hat{f}(\lambda_l)u_l(i) \right\vert \\
&\leq B \left( \sum_l \left\vert \hat{f}(\lambda_l) \right\vert^2\sum_l \vert u_l(i) \vert^2 \right)^{1/2} 
= B\;\|f\| 
\tag{6}
\end{align}
$$

이 때 $$f_{out}$$ 과 $$\tilde{f}_{out}$$ 에 대한 오차 $$(6)$$ 을 줄이기 위해서는, $$g_{\theta}$$ 와 $$p$$ 에 대한 $$L_{\infty}$$ error $$(5)$$ 를  최소화해야 합니다. 만약  $$p$$ 가 order $$K$$ polynomial 이라면, $$(5)$$ 는 $$p$$ 가 minimax polynomial of order $$K$$ 일 때 최소가 됩니다. 더 나아가, minimax polynomial 은 truncated Chebyshev expansion 을 통해 충분히 근사할 수 있습니다.




따라서, $$(6)$$ 의 오차를 줄이기 위해 $$p$$ 로 $$g_{\theta}$$ 의 truncated Chebyshev expansion 을 선택할 수 있습니다. 하지만  $$g_{\theta}$$ 는 $$L$$ 의 spectrum 을 포함하는 domain 에서 정의된 함수이기 때문에, $$(2)$$ 를 적용하기 위해서는 domain 의 변환이 필요합니다. $$L$$ 의 eigenvalue 들은 모두 $$[0, \lambda_{max}]$$ 구간에 속하기 때문에 $$h_{\theta}$$ 를 다음과 같이 정의하면 $$g_{\theta}$$ 를 $$[-1,1]$$ 에서 정의된 함수로 바꿀 수 있습니다.

$$
h_{\theta}(x) = g_{\theta}\left( \frac{\lambda_{\max}}{2}(x+1) \right)
$$

$$(2)$$ 를 $$h_{\theta}$$ 에 적용하고 order $$K$$ 까지의 truncation 을 생각하면, 다음과 같이 $$h_{\theta}$$ 를 근사할 수 있습니다.

$$
h_{\theta}(x) \approx \frac{1}{2}c_0 + \sum^{K}_{k=1} c_kT_k(x)
$$

$$\tilde{L} = \frac{2}{\lambda_{max}}L - I$$ 에 대해 $$g_{\theta}(L) = h_{\theta}(\tilde{L})$$ 를 만족하기 때문에, $$p$$ 를 다음과 같이 정의합니다.

$$
p(\tilde{L}) = h_{\theta}(\tilde{L}) =  \frac{1}{2}c_0I + \sum^{\infty}_{k=1} c_kT_k(\tilde{L})
\tag{7}
$$

$$(7)$$ 을 사용하면, Fourier basis $$U$$ 를 이용하지 않고도 $$(3)$$ 에 대한 근사가 가능합니다.

$$
\begin{align}
& Ug_{\theta}(\Lambda)U^T 
= Uh_{\theta}(\tilde{\Lambda})U^T 
\approx Up(\tilde{\Lambda})U^T = p(\tilde{L}) \\
\\
& f_{out} = Ug_{\theta}(\Lambda)U^Tf_{in} \approx p(\tilde{L})f_{in}
\tag{8}
\end{align}
$$



&nbsp;

마지막으로, 두 가지 확인해야할 것이 있습니다. 첫번 째로, $$\tilde{L}$$ 을 계산하기 위해서 $$\lambda_{max}$$ 에 대한 정보가 필요합니다. Spectrum 의 upper bound $$\lambda_{max}$$ 는 Arnoldi iteration 혹은 Jacobi-Davidson method 등을 사용하면 $$L$$ 의 전체 spectrum 을 찾는 것에 비해서 훨씬 쉽게 구할 수 있습니다. 



두번 째로, $$p(\tilde{L})$$ 을 계산하기 위해서는 Chebyshev coefficient $$c_k$$ 에 대해 알아야합니다. 이는 $$(\ast)$$ 를 통해 이론적으로 계산할 수 있지만, 현실적으로 도움이 되지 않습니다. 여기서, neural network 가 등장합니다. Universal approximation theorem 에 의해 $$(7)$$ 을 근사할 수 있는 neural network 가 존재합니다. 따라서, coefficient $$c_k$$ 를 parameter 로 학습하는 neural network 가 바로 ChebNet 입니다 [1].

&nbsp;

## Advantage of Using Chebyshev Expansion



$$(8)$$ 과 같이 truncated Chebyshev expansion 을 통한 근사는 다음과 같은 두 가지 이점이 있습니다.



### Fast filtering using recurrence relation

Chebyshev polynomial 의 중요한 특성은 $$(1)$$ 의 점화식을 통해 재귀적으로 계사할 수 있다는 것입니다. Graph Laplacian $$L$$ 에서부터 시작해 재귀적 연산으로 order $$K$$ polynomial $$T_K$$ 까지 구하는 computational cost 는 $$L$$ 이 sparse matrix 일 때 $$O(K\vert E\vert)$$ 입니다. 



### Localized Filter

$$(8)$$ 의 결과 $$p(\tilde{L})f_{in}$$ 은 각 vertex $$i$$ 에 대해 $$i$$ 의 $$K$$- hop local neighborhood 만을 이용해 표현할 수 있습니다. 이를 통해 CNN 의 중요한 특성인 locality 가 그래프에서 일반화될 수 있습니다.

&nbsp;

## Lanczos Algorithm 

$$(3)$$ 을 효율적으로 계산하는 다른 해결 방법으로는 Lanczos Algorithm 이 있습니다. [LanczosNet: Multi-Scale Deep Graph Convolutional Networks](https://arxiv.org/pdf/1901.01484.pdf) 의 paper review 를 통해 더 자세히 설명하겠습니다.


&nbsp;

## Reference

1. Michael Defferrard, Xavier Bresson, and Pierre Vandergheynst. [Convolutional neural networks on
   graphs with fast localized spectral filtering](https://arxiv.org/pdf/1606.09375.pdf). In Advances in neural information processing systems
   (NIPS), 2016.



2. Thomas N. Kipf and Max Welling. [Semi-supervised classification with graph convolutional networks](https://arxiv.org/pdf/1609.02907.pdf).
   In International Conference on Learning Representations (ICLR), 2017.



3. David K Hammond, Pierre Vandergheynst, and Remi Gribonval. [Wavelets on graphs via spectral
   graph theory](https://arxiv.org/pdf/0912.3848.pdf). Applied and Computational Harmonic Analysis, 30(2):129–150, 2011.





[^3]: Spectrum 의 upper bound $$\lambda_{max}$$ 는 Arnoldi iteration 혹은 Jacobi-Davidson method 등을 사용하면 $$L$$ 의 spectrum 전체를 찾는 것에 비해서 훨씬 쉽게 구할 수 있습니다.

