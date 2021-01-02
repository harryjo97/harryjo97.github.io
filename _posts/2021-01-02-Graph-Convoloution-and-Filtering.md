---
title: Graph Convolution and Filtering
description: based on "The Emerging Field of Signal Processing on Graphs"
date: 2021-01-01 22:00:00 +0900
category:
- theory
tag:
- gnn
- graph convolution
- filtering
---



Graph Signal Processing 의 관점에서 graph convolution 과 filtering 의 이해.



## 0. Graph Fourier Transform



Graph Fourier transform 에 대해 자세히 설명한 [포스트](https://harryjo97.github.io/theory/Graph-Fourier-Transform/) 를 보고 오시면, graph convolution 을 이해하는데 큰 도움이 될 것입니다.



## 1. Classical Convolution









## 2. Graph Convolution



$$
g\ast f = \sum^{N-1}_{l=0} \hat{g}(\lambda_l) \hat{f}(\lambda_l)u_l
$$


graph Laplacian eigenvector $$\{u_l\}^{N-1}_{l=0}$$ orthonormal


$$
\widehat{g\ast f}(\lambda_l) = \hat{g}(\lambda_l)\hat{f}(\lambda_l)
$$


vertex  domain 에서의 convolution 은 spectral domain 에서의 multiplication 과 동일합니다.





## 3. Filtering 



> Filtering in spectral domain



spectral filtering


$$
f_{out} = g\ast f_{in}
$$

$$f_{out}$$ 의 Fourier transform 을 보면

$$
\hat{f}_{out}(\lambda_l) = \hat{g}(\lambda_l)\hat{f}_{in}(\lambda_l)
$$

Inverse Fourier transform 을 통해 $$f_{out}$$ 을 복원하면,

$$
\begin{align}

f_{out} 
&= \sum^{N-1}_{l=0} \hat{f}_{out}(\lambda_l)u_l
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
= U\hat{g}(\Lambda)U^T\;f_{in} \tag{$1$}

\end{align}
$$






> Filtering in vertex domain



vertex $$i$$ 에서의 함수값 $$f_{out}(i)$$ 를 $$i$$ 의 $$K$$-hop local neighborhood $$N(i,K)$$ 에 대한 입력 함수 $$f_{in}$$ 의 합


$$
f_{out}(i) = b_{ii}f_{in}(i) + \sum_{j\in N(i,K)}b_{ij} f_{in}(j)
$$






## 4. Next

다음 포스트에서는 Polynomial Approximation of Spectral Filtering 대해 설명하겠습니다.

