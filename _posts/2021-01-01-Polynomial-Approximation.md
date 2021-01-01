---
title: Polynomial Approximation
description: based on "Wavelets on Graphs via Spectral Graph Theory"
date: 2021-01-01 11:00:00 +0900
category:
- theory
tag:
- gnn
- Polynomial approximation
---

##0. 

## 1. Polynomial Approximation



Filter $$g_{\theta}$$ 를 거친 입력 신호 $$f_{in}$$ 의 결과 $$f_{out}$$ 을 계산하기 위해서는




$$
f_{out} = g_{\theta} \ast f_{in}
$$

$$f_{out}$$ 의 Fourier transform 을 보면

$$
\hat{f}_{out}(\lambda_l) = \hat{g}_t(\lambda_l)\hat{f}_{in}(\lambda_l)
$$



Inverse Fourier transform 을 통해 $$f_{out}$$ 을 복원하면
$$
\begin{align}

f_{out} 
&= \sum^{N-1}_{l=0} \hat{f}(\lambda_l)u_l
= 
\begin{bmatrix}
\big| & \big| & \cdots & \big| \\
u_0 & u_1 & \cdots & u_{N-1} \\
\big| & \big| & \cdots & \big|
\end{bmatrix}
\begin{bmatrix}
\hat{f}_{out}(\lambda_0) \\
\vdots \\
\hat{f}_{out}(\lambda_{N-1})
\end{bmatrix} 
= U
\begin{bmatrix}
\hat{g}(\lambda_0)\hat{f}_{in}(\lambda_0) \\
\vdots \\
\hat{g}(\lambda_{N-1})\hat{f}_{in}(\lambda_{N-1})
\end{bmatrix} \\
\\
&= U
\begin{bmatrix}
\hat{g}(\lambda_0) & 0 & \cdots & 0 \\
0 & \hat{g}(\lambda_1) & \cdots & 0 \\
\vdots &  & \ddots & \\
0 & 0 & 0 & \hat{g}(\lambda_{N-1})
\end{bmatrix}
\begin{bmatrix}
\hat{f}_{in}(\lambda_0) \\
\vdots \\
\hat{f}_{in}(\lambda_{N-1})
\end{bmatrix}
= U\hat{g}_t(\Lambda)U^T\;f_{in} \tag{$1$}

\end{align} 
$$



즉 filter $$g_{\theta}$$ 의 결과를 확인하기 위해서는  graph Laplacian $$L$$ 의 eigenvector 가 모두 필요합니다. 하지만, eigenvalue 와 eigenvector 를 계산하기 위해 많이 사용하는 QR decomposition 의 computational complexity 가 $$O(N^3)$$ 이므로 vertices 가 수천개 이상인 그래프에서는 적용하기에는 힘듭니다.

따라서, 그래프의 크기가 큰 경우에는 $$(1)$$ 을 근사할 수 있는 효율적인 방법이 필요합니다.  



[Hammond et al.](https://arxiv.org/pdf/0912.3848.pdf) 에서는 (truncated) Chebyshev polynomial 을 이용한 polynomial approximation 을  제시합니다. 



Lanczos algorithm 



__Lemma 6.1.__

주어진 그래프의 graph Laplacian $$L$$ 과 주어진 filter $$g_{\theta}$$ 에 대해
polynomial approximant $$p$$ 가 다음의 조건을 만족한다고 가정합니다.
$$
\left\vert g(\theta x) - p(x) \right\vert \leq B < \infty
\;\;\text{ for all }\;\; x\in [0,\lambda_{max}]
\tag{$2$}
$$

$$\lambda_{max}$$ 는 $$L$$ 의 spectrum 에 대한 upper bound 입니다. $$\lambda_{max} \geq \lambda_{N-1}$$ 

$$\tilde{f}_{out} = p(L)f$$ 에 대해 다음의 부등식이 성립합니다.
$$
\vert f_{out}(i)- \tilde{f}_{out}(i) \vert \leq B \; \|f\|
$$

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


## 4. Next

다음 포스트에서는 [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf) 의 paper review 를 하겠습니다.