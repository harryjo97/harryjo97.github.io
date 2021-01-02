---
title: Graph Convolution and Filtering
description: based on "The Emerging Field of Signal Processing on Graphs"
date: 2021-01-03 22:00:00 +0900
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



$$
(g \ast f) (t) 
= \int_{\mathbb{R}}\hat{g}(\xi)\hat{f}(\xi)e^{2\pi i\xi t}d\xi
= \int_{\mathbb{R}} g(\tau)f(t-\tau)d\tau 
\tag{1}
$$






## 2. Graph Convolution





기존의 convolution 과 같이 graph convolution 또한 Fourier transform 에 대해 다음의 조건을 만족하도록 convolution operator $$\ast$$ 를 정의하려고 합니다. 


$$
\widehat{g\ast f}(\lambda_l) = \hat{g}(\lambda_l)\hat{f}(\lambda_l)
\tag{2}
$$


즉 vertex  domain 에서의 convolution 과 spectral domain 에서의 multiplication 이 일치하도록 만들고 싶습니다. 

$$(2)$$ 에 대해 inverse Fourier transform 을 적용하면, 다음의 결과를 얻게 됩니다.


$$
g\ast f = \sum^{N-1}_{l=0} \hat{g}(\lambda_l) \hat{f}(\lambda_l)u_l
\tag{3}
$$



따라서, vertex domain 에서 정의된 두 함수 $$f$$ 와 $$g$$ 에 대해 convolution operator $$\ast$$ 는 $$(3)$$ 과 같이 정의합니다.

이는 $$(1)$$ 에서 $$\left\{e^{2\pi i\xi t}\right\}_{\xi\in\mathbb{R}}$$ 대신 graph Laplacian eigenvector $$\{u_l\}^{N-1}_{l=0}$$  을 사용했다고 이해할 수 있습니다. 

( $$\left\{e^{2\pi i\xi t}\right\}_{\xi\in\mathbb{R}}$$ 와 $$\{u_l\}^{N-1}_{l=0}$$ 의 관계에 대해서는 [포스트](https://harryjo97.github.io/theory/Graph-Laplacian/) 를 참고해주세요 )



 [포스트](https://harryjo97.github.io/theory/Graph-Fourier-Transform/) 의 Why Do We Need Fourier Transform? 에서 설명했듯이, vertex domain 에서 직접 convolution operator 를 정의할 수 없기 때문에, Fourier transform 을 활용하여 spectral domain 을 통해 간접적으로 정의한 것입니다.





## 3. Filtering 



> Filtering in spectral domain



frequency filtering 을 그래프 영역에 일반화 시키면, spectral filtering 을 다음과 같이 정의할 수 있습니다.

$$
\hat{f}_{out}(\lambda_l) = \hat{g}(\lambda_l)\hat{f}_{in}(\lambda_l)
$$

Inverse Fourier transform 을 통해 $$f_{out}$$ 을 복원하면,

$$
\begin{align}

f_{out} 
&= \sum^{N-1}_{l=0} \hat{f}_{out}(\lambda_l)u_l \\
&= 
\begin{bmatrix}
\big| & \big| &  & \big| \\
u_0 & u_1 & \cdots & u_{N-1} \\
\big| & \big| &  & \big|
\end{bmatrix}
\begin{bmatrix}
\hat{f}_{out}(\lambda_0) \\
\vdots \\
\hat{f}_{out}(\lambda_{N-1})
\end{bmatrix} \\
\\
&= U
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
= U\hat{g}(\Lambda)U^T\;f_{in} \tag{4}

\end{align}
$$








> Filtering in vertex domain



vertex $$i$$ 에서의 함수값 $$f_{out}(i)$$ 를 $$i$$ 의 $$K$$-hop local neighborhood $$N(i,K)$$ 에 대한 입력 함수 $$f_{in}$$ 의 합


$$
f_{out}(i) = b_{ii}f_{in}(i) + \sum_{j\in N(i,K)}b_{ij} f_{in}(j)
$$






## 4. Next

다음 포스트에서는 Polynomial Approximation of Spectral Filtering 대해 설명하겠습니다.

