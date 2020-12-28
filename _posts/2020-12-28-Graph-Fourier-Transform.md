---
title: Graph Fourier Transform
description: with Graph Signal Processing
date: 2020-12-28 22:00:00 +0900
category:
- GNN
---



Graph Signal Processing 의 관점에서 Graph Fourier Transform 의 이해.



## 0. Graph Laplacian



Graph Laplacian 에 대해 자세히 설명한 [포스트](https://harryjo97.github.io/gnn/Graph-Laplacian/) 를 보고 오시면, graph Fourier Transform 을 이해하는데 큰 도움이 될 것입니다. 



## 1. Classical Fourier Transform



Fourier transform 은 주어진 신호를 서로 다른 frequency 를 가지는 주기 함수들로 분해하여 표현합니다. 

![](/assets/post/Graph-Fourier-Transform/fourier.jpg)



> Fourier transform

임의의 실수 $$\xi$$ 에 대한 integrable function $$ f : \mathbb{R} \rightarrow \mathbb{C} $$ 의 Fourier transform 은 다음과 같이 정의합니다.

$$
\hat{f}(\xi) = \langle f, e^{2\pi i\xi t} \rangle = \int_{\mathbb{R}} f(t) e^{-2\pi i\xi t}dt \tag{$1$}
$$


> Inverse Fourier transform

Fourier transform 의 역과정인 inverse Fourier transform 은 다음과 같습니다.

$$
f(x) = \int_{\mathbb{R}} \hat{f}(\xi)e^{2\pi i\xi t}d\xi \tag{$2$}
$$




> Laplacian operator

Complex exponentials $$ e^{2\pi i \xi t}$$ 들은 1 차원 Laplacian operator $$ \Delta $$ 의 eigenfunction 으로

$$
\Delta(e^{2\pi i \xi t}) = \frac{\partial^2}{\partial t^2}e^{2\pi i \xi t} 
= -(2\pi\xi)^2 e^{2\pi i \xi t}
$$


Fourier transform 은 Laplacian operator $$\Delta$$ 의 eigenfunction 들의 합으로 표현하는 것입니다.



## 2. Graph Fourier Transform



> Graph Fourier transform

[포스트](https://harryjo97.github.io/gnn/Graph-Laplacian/) 의 4번째 파트에서 설명했듯이, Euclidean space 의 Laplacian operator $$\Delta$$ 는 그래프에서의 graph Laplacian $$L$$ 에 해당합니다. 즉 그래프에서도 graph Laplacian $$L$$ 을 사용해 Fourier transform 을 정의할 수 있습니다.

Complex exponentials $$ e^{2\pi i \xi t}$$ 들은 1 차원 Laplacian operator $$ \Delta $$ 의 eigenfunction

이에 대응하는 것이 $$L$$ 의 eigenvector 들입니다. $$L$$ 의 eigenvalue $$ \lambda_l $$ 에 대한 eigenvector 를 $$ u_l $$ 이라 하면,  $$(1)$$ 과 같이graph Fourier transform 을 정의할 수 있습니다.

$$
\hat{f}(\lambda_l) = \langle u_l, f\rangle =  \sum^N_{i=1} f(i)u^{\ast}_l(i) 
\tag{$3$}
$$

$$L$$ 의 eigenvector 들을 column 으로 가지는 행렬 $$U$$ 에 대해

$$
U = \begin{bmatrix}
\bigg| & \bigg| & & \bigg| \\
u_0 & u_1 & \cdots & u_{N-1} \\
\bigg| & \bigg| & & \bigg|
\end{bmatrix}
$$

$$(3)$$ 을 vector 의 형태로 표현하면,

$$
\hat{f} = U^Tf
$$


> Inverse graph Fourier transform

Graph Laplacian $$L$$ 의 eigenvector 들은 orthonormal[^1] 하기 때문에, $$(2)$$ 와 같이 inverse graph Fourier transform 을 정의할 수 있습니다.

$$
f(i) = \sum^{N-1}_{l=0} \hat{f}(\lambda_l)u_l(i) 
\tag{$4$}
$$


> Parseval relation

Classical Fourier transform 에서와 마찬가지로 Parseval relation[^2] 을 만족합니다.

$$
\begin{align}
	\langle f, h\rangle 
	&= \langle \sum^{N-1}_{l=0}\hat{f}(l)u_l, \sum^{N-1}_{l'=0}\hat{g}(l')u_{l'} \rangle \\
	&= \sum_{l, l'} \hat{f}(l)\hat{g}(l')\langle u_l, u_{l'} \rangle \\
	&= \sum_{l} \hat{f}(l)\hat{g}(l) = \langle \hat{f}, \hat{g} \rangle
\end{align}
$$

## 3. Fourier Domain[^3]







## 4. Next



다음 포스트에서는 graph Fourier transform 을 사용한 graph Convolution 에 대해 설명하겠습니다.



[^1]: Eigenvector 들을 unit vector 로 설정하면 orthonormal 하게 됩니다.
[^2]: $$ \langle f, h \rangle = \langle \hat{f}, \hat{g} \rangle$$ 의 identity 를 의미합니다.
[^3]: frequency domain, spectral domain 등의 이름으로도 불립니다.