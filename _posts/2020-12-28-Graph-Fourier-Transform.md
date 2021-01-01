---
title: Graph Fourier Transform
description: with Graph Signal Processing
date: 2020-12-28 22:00:00 +0900
category: 
- theory
tag:
- gnn
- Graph Fourier Transform
---



Graph Signal Processing 의 관점에서 Graph Fourier Transform 의 이해.



## 0. Graph Laplacian



Graph Laplacian 에 대해 자세히 설명한 [포스트](https://harryjo97.github.io/gnn/Graph-Laplacian/) 를 보고 오시면, graph Fourier Transform 을 이해하는데 큰 도움이 될 것입니다. 



## 1. Classical Fourier Transform



Fourier transform 은 주어진 신호 (함수) 를 서로 다른 frequency 를 가지는 정현파들로 분해하는 변환입니다.  



<p align='center'>
    <img src = '/assets/post/Graph-Fourier-Transform/fourier.jpg' style = 'max-width: 100%; height: auto'>
</p>
Graph Fourier transform 과 구분 짓기 위해 이를 "Classical" Fourier transform 이라고 부르겠습니다.




> Fourier transform

임의의 실수 $$\xi$$ 에 대한 integrable function $$ f : \mathbb{R} \rightarrow \mathbb{C} $$ 의 Fourier transform 은 다음과 같이 정의합니다.

$$
\hat{f}(\xi) = \langle f, e^{2\pi i\xi t} \rangle = \int_{\mathbb{R}} f(t) e^{-2\pi i\xi t}dt \tag{$1$}
$$



$$(1)$$ 의 정의를 살펴보면, Fourier transform 은 time domain $$t\in\mathbb{R}$$ 에서 정의된 함수 $$f(t)$$ 를 frequency domain $$\xi\in\mathbb{C}$$ 에서 정의된 함수 $$\hat{f}(\xi)$$ 로 변환시켜줍니다.




> Inverse Fourier transform

Fourier transform 의 역과정인 inverse Fourier transform 은 다음과 같습니다.

$$
f(x) = \int_{\mathbb{R}} \hat{f}(\xi)e^{2\pi i\xi t}d\xi \tag{$2$}
$$




> Laplacian operator

Fourier transform 에 등장하는 complex exponentials $$ \left\{ e^{2\pi i \xi t} \right\}_{\xi\in\mathbb{C}} $$ 는 1 차원 Laplacian operator $$ \Delta $$ 의 eigenfunction 입니다.

$$
\Delta(e^{2\pi i \xi t}) = \frac{\partial^2}{\partial t^2}e^{2\pi i \xi t} 
= -(2\pi\xi)^2 e^{2\pi i \xi t}
$$


그렇기 때문에 Classical Fourier transform 은 Laplacian operator $$\Delta$$ 의 eigenfunction 들의 합으로 분해하는 것입니다.



## 2. Graph Fourier Transform



> Graph Fourier transform

[포스트](https://harryjo97.github.io/gnn/Graph-Laplacian/) 의 4번째 파트에서 설명했듯이, Euclidean space 의 Laplacian operator $$\Delta$$ 는 그래프에서의 graph Laplacian $$L$$ 에 해당합니다. 즉 그래프에서도 graph Laplacian $$L$$ 을 사용해 Fourier transform 을 정의할 수 있습니다. 



Complex exponentials $$ \left\{ e^{2\pi i \xi t} \right\}_{\xi\in\mathbb{C}} $$ 이 1 차원 Laplacian operator $$ \Delta $$ 의 eigenfunction 이고, 그래프에서 이에 대응하는 것은 $$L$$ 의 eigenvector $$ \left\{ u_l \right\}^{N-1}_{l=0} $$ [^1]입니다. 즉 그래프에서의 Fourier transform 은 그래프에서의 함수 $$ f \in\mathbb{R}^N $$ 를 $$L$$ 의 eigenvector $$ \left\{ u_l \right\}^{N-1}_{l=0} $$ 들의 합으로 분해하는 변환입니다.  



$$f$$에서 $$u_l$$ 이 차지하는 부분은 두 vector 의 inner product 로 얻을 수 있습니다. 따라서, $$(1)$$ 과 같이 graph Fourier transform 을 정의할 수 있습니다.
$$
\hat{f}(\lambda_l) = \langle u_l, f\rangle =  \sum^N_{i=1} f(i)u^{\ast}_l(i) 
\tag{$3$}
$$

여기서 $$f$$ 에 대한 Fourier transform 의 결과인 $$\hat{f}$$ 는 $$L$$ 의 spectrum 에서만 정의되는 함수입니다. $$\hat{f}$$ 을 다음과 같이 $$\mathbb{R}^N$$ 의 vector 로 보겠습니다.

$$
\hat{f} 
= \begin{bmatrix}
\hat{f}(\lambda_0) \\
\vdots \\
\hat{f}(\lambda_{N-1})
\end{bmatrix}
$$

$$L$$ 의 eigenvector 들을 column 으로 가지는 행렬 $$U$$ 에 대해

$$
U = \begin{bmatrix}
\bigg| & \bigg| & & \bigg| \\
u_0 & u_1 & \cdots & u_{N-1} \\
\bigg| & \bigg| & & \bigg|
\end{bmatrix}
$$

$$(3)$$ 을 다음과 같이 표현할 수 있습니다.

$$
\hat{f} 
= \begin{bmatrix}
\hat{f}(\lambda_0) \\
\vdots \\
\hat{f}(\lambda_{N-1})
\end{bmatrix}
= \begin{bmatrix}
- & u_0^{\ast} & - \\

& \vdots & \\
- & u_{N-1}^{\ast} & -
\end{bmatrix}
\begin{bmatrix}
f(1) \\
\vdots \\
f(N)
\end{bmatrix}
= U^{\ast}f
$$




> Inverse graph Fourier transform

$$\hat{f}(\lambda_l)$$ 은 그래프에서의 함수 $$f$$ 의 $$u_l$$ 에 대한 성분으로 이해할 수 있습니다. $$ \left\{ u_l \right\}^{N-1}_{l=0} $$ 은 $$\mathbb{R}^N$$ 의 orthonormal basis[^2] 를 이루기 때문에, Fourier transform 의 결과인 $$\hat{f}$$ 으로 부터 원래의 함수 $$f$$ 를 얻어내려면 각 성분들을 모두 더해주면 됩니다. 따라서 $$(2)$$ 와 같이 inverse graph Fourier transform 을 정의할 수 있습니다.

$$
f(i) = \sum^{N-1}_{l=0} \hat{f}(\lambda_l)u_l(i) 
\tag{$4$}
$$



$$(4)$$ 를 $$\mathbb{R}^N$$ 의 vector 로 표현하면,


$$
f 
= \begin{bmatrix}
f(1) \\
\vdots \\
f(N)
\end{bmatrix}
= \begin{bmatrix}
\bigg| & \bigg| & & \bigg| \\
u_0 & u_1 & \cdots & u_{N-1} \\
\bigg| & \bigg| & & \bigg|
\end{bmatrix} 
\begin{bmatrix}
\hat{f}(\lambda_0) \\
\vdots \\
\hat{f}(\lambda_{N-1})
\end{bmatrix}
= U\hat{f}
$$





> Parseval relation

Classical Fourier transform 에서와 마찬가지로 Parseval relation[^3] 을 만족합니다.

$$
\begin{align}
	\langle f, h\rangle 
	&= \langle \sum^{N-1}_{l=0}\hat{f}(\lambda_l)u_l, \sum^{N-1}_{l'=0}\hat{g}(\lambda_{l'})u_{l'} \rangle \\
	&= \sum_{l, l'} \hat{f}(\lambda_l)\hat{g}(\lambda_{l'})\langle u_l, u_{l'} \rangle \\
	&= \sum_{l} \hat{f}(\lambda_l)\hat{g}(\lambda_l) = \langle \hat{f}, \hat{g} \rangle
\end{align}
$$

&nbsp;



## 3. Why Do We Need Fourier Transform?







## 4. Next



다음 포스트에서는 graph Fourier transform 을 사용한 graph Convolution 과 polynomial approximation 에 대해 설명하겠습니다.



&nbsp;

&nbsp;



[^1]:$$L$$ 의 eigenvalue $$ \lambda_l $$ 에 대한 eigenvector 를 $$ u_l $$ 이라 하겠습니다.
[^2]: Eigenvector 들을 unit vector 로 설정하면 orthonormal 하게 됩니다.
[^3]: $$ \langle f, h \rangle = \langle \hat{f}, \hat{g} \rangle$$ 의 identity 를 의미합니다.