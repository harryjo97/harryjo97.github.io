---
title: Graph Fourier Transform
description: with Graph Signal Processing
date: 2021-01-01 19:00:00 +0900
category: 
- theory
tag:
- fourier transform
- graph fourier transform
---



Graph Signal Processing 의 관점에서 Graph Fourier Transform 의 이해.



## 0. Graph Laplacian



Graph Laplacian 에 대해 자세히 설명한 [포스트](https://harryjo97.github.io/gnn/Graph-Laplacian/) 를 보고 오시면, graph Fourier Transform 을 이해하는데 큰 도움이 될 것입니다. 



## 1. Classical Fourier Transform




> Fourier transform

Integrable function $$ f : \mathbb{R} \rightarrow \mathbb{C} $$ 의 Fourier transform 은 다음과 같이 정의합니다.

$$
\hat{f}(\xi) = \langle f, e^{2\pi i\xi t} \rangle = \int_{\mathbb{R}} f(t) e^{-2\pi i\xi t}dt \tag{$1$}
$$



$$(1)$$ 을 살펴보면, Fourier transform 은 time domain $$t\in\mathbb{R}$$ 에서 정의된 함수 $$f(t)$$ 를 frequency domain $$\xi\in\mathbb{C}$$ 에서 정의된 함수 $$\hat{f}(\xi)$$ 로 변환시켜준다는 것을 알 수 있습니다.

<p align='center'>
    <img src = '/assets/post/Graph-Fourier-Transform/fourier.jpg' style = 'max-width: 100%; height: auto'>
</p>

위의 그림은 time domain 으로부터 frequency domain 으로의 Fourier transform 을 잘 나타내어 줍니다.




> Inverse Fourier transform

$$(1)$$ 에서 정의된 Fourier transform 의 역과정인 inverse Fourier transform 은 다음과 같습니다.

$$
f(x) = \int_{\mathbb{R}} \hat{f}(\xi)e^{2\pi i\xi t}d\xi \tag{$2$}
$$



Invere Fourier transform 은 Fourier transform 과 반대로 frequency domain 에서 정의된 함수 $$\hat{f}$$ 을 time domain 에서의 함수 $$f$$ 로 변환시켜줍니다.

$$(2)$$ 는 주어진 $$\hat{f}$$ 으로부터 원래의 함수 $$f$$ 를 복원하는 과정입니다. Inverse Fourier transform 을 통해 각 성분별 amplitude $$\hat{f}(\xi)$$ 가 주어졌을 때, 원래의 함수는 각 성분들의 합으로 표현할 수 있습니다.




> Laplacian operator

$$(1)$$ 과 $$(2)$$ 에서 등장하는 complex exponentials $$ \left\{ e^{2\pi i \xi t} \right\}_{\xi\in\mathbb{C}} $$ 는 1 차원 Laplacian operator $$ \Delta $$ 의 eigenfunction 입니다.

$$
\Delta(e^{2\pi i \xi t}) = \frac{\partial^2}{\partial t^2}e^{2\pi i \xi t} 
= -(2\pi\xi)^2 e^{2\pi i \xi t}
$$


즉 Fourier transform 은 Laplacian operator $$\Delta$$ 의 eigenfunction 들의 합으로 분해하는 변환으로 생각 할 수 있습니다.



## 2. Graph Fourier Transform



> Graph Fourier transform

[포스트](https://harryjo97.github.io/gnn/Graph-Laplacian/) 의 4번째 파트에서 설명했듯이, Euclidean space 의 Laplacian operator $$\Delta$$ 는 그래프에서의 graph Laplacian $$L$$ 에 해당합니다. 그렇기 때문에, 그래프에서도 graph Laplacian $$L$$ 을 사용해 Fourier transform 을 정의할 수 있습니다. 

Complex exponentials $$ \left\{ e^{2\pi i \xi t} \right\}_{\xi\in\mathbb{C}} $$ 은 1 차원 Laplacian operator $$ \Delta $$ 의 eigenfunction 이고, 그래프에서 이에 대응하는 것은 $$L$$ 의 eigenvector $$ \left\{ u_l \right\}^{N-1}_{l=0} $$ [^1]입니다. 즉 그래프에서의 Fourier transform 은 그래프에서의 함수 $$ f \in\mathbb{R}^N $$ 를 $$L$$ 의 eigenvector $$ \left\{ u_l \right\}^{N-1}_{l=0} $$ 들의 합으로 분해하는 변환으로 이해할 수 있습니다.  

$$f$$에서 $$u_l$$ 성분은 두 vector 의 inner product 로 계산할 수 있습니다. 따라서, $$(1)$$ 과 같이 graph Fourier transform 을 정의할 수 있습니다.

$$
\hat{f}(\lambda_l) = \langle u_l, f\rangle =  \sum^N_{i=1} f(i)u^{\ast}_l(i) 
\tag{$3$}
$$

여기서 $$f$$ 에 대한 Fourier transform 의 결과인 $$\hat{f}$$ 는 $$L$$ 의 spectrum 에서만 정의되는 함수입니다. 그렇기 때문에 $$\hat{f}$$ 가 정의된 domain 을 spectral domain[^2] 이라고 부릅니다.

$$\hat{f}$$ 을 다음과 같이 $$\mathbb{R}^N$$ 의 vector 로 보겠습니다.
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

$$\hat{f}(\lambda_l)$$ 은 그래프에서의 함수 $$f$$ 의 $$u_l$$ 에 대한 성분으로 볼 수 있습니다. 이 때 $$ \left\{ u_l \right\}^{N-1}_{l=0} $$ 은 $$\mathbb{R}^N$$ 의 orthonormal basis[^3] 를 이루기 때문에, Fourier transform 의 결과인 $$\hat{f}$$ 으로 부터 원래의 함수 $$f$$ 를 얻어내기 위해서는 각 성분들을 모두 더해주면 됩니다. 

$$(2)$$ 와 같이 inverse graph Fourier transform 을 정의할 수 있습니다.

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

Classical Fourier transform 에서와 마찬가지로, graph Fourier transform 은 Parseval relation[^4] 을 만족합니다.

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



그래프의 vertex domain $$V$$ 는 discrete 하기 때문에, 그래프의 함수 $$f:V\rightarrow \mathbb{R}^N$$ 를 vertex domain 에서 다루기 까다롭습니다. 다음의 예시를 들어 생각해보겠습니다.

$$\mathbb{R}$$ 에서 정의된 함수 $$f$$ 와 실수 $$s, t$$ 에 대한 operation : translation  $$ T_{t} $$ 과 scaling  $$T^{s}$$  은 다음과 같이 정의됩니다.

$$
\begin{align}
T_{t}f(x) = f(x+t)\;, \;\;T^{s}f(x) = f(sx)
\tag{$5$}
\end{align}
$$



하지만 vertex domain 은 discrete 하기 때문에, vertex $$i$$ 에 대해 $$i+t$$ 와 $$si$$ 의 의미를 알 수 없습니다. 그렇기에 그래프의 함수에 대해서는 $$(5)$$ 의 translation 과 scaling 을 적용할 수 없습니다. 

이런 문제를 해결하기 위해서  discrete 한 vertex domain 대신 continuous 한 spectral domain 을 이용합니다. Fourier transform 을 통해 vertex domain 에서 정의된 함수 $$f$$ 를 spectral domain 에서의 함수 $$\hat{f}$$ 로 변환해 준 후, $$\hat{f}$$ 에 대한 $$(5)$$ 의 translation 과 scaling 을 적용해주는 것입니다.



<p align='center'>
    <img src = '/assets/post/Graph-Fourier-Transform/transform-diagram.PNG' style = 'max-width: 50%; height: auto'>
</p>




Vertex domain 에서 정의하고 싶은 operation 은 위의 다이어그램과 같이 spectral domain 에서의 operation 을 통해 정의할 수 있습니다. 이 때, vertex domain 과 spectral domain 을 연결해주는 것이 바로 Fourier transform 입니다.  






## 4. Next



다음 포스트에서는 graph convolution 과 filtering 에 대해 설명하겠습니다.



&nbsp;

&nbsp;



[^1]:$$L$$ 의 eigenvalue $$ \lambda_l $$ 에 대한 eigenvector 를 $$ u_l $$ 이라 하겠습니다.
[^2]: Fourier domain, frequency domain 으로도 불립니다.
[^3]: Eigenvector 들을 unit vector 로 설정하면 orthonormal 하게 됩니다.
[^4]: $$ \langle f, h \rangle = \langle \hat{f}, \hat{g} \rangle$$ 의 identity 를 의미합니다.
