---
title: Graph Convolution and Spectral Filtering
description: based on "The Emerging Field of Signal Processing on Graphs"
date: 2021-01-03 17:00:00 +0900
category:
- theory
tag:
- gnn
- graph convolution
- spectral filtering
---



Graph Signal Processing 의 관점에서 graph convolution 과 spectral filtering 의 이해.



## 0. Graph Fourier Transform



Graph Fourier transform 에 대해 자세히 설명한 [포스트](https://harryjo97.github.io/theory/Graph-Fourier-Transform/) 를 보고 오시면, graph convolution 을 이해하는데 큰 도움이 될 것입니다.




## 1. Graph Convolution



Vertex domain 에서 직접 convolution operator 를 정의할 수 없기 때문에, Fourier transform 을 이용하여 spectral domain 을 통해 vertex domain 의 함수에 대해 convolution operator 를 정의하겠습니다.



기존의 convolution 과 같이 graph convolution 또한 Fourier transform 에 대해 다음의 조건을 만족해야 합니다. 

$$
\widehat{g\ast f}(\lambda_l) = \hat{g}(\lambda_l)\hat{f}(\lambda_l)
\tag{1}
$$


즉 vertex  domain 에서의 convolution 과 spectral domain 에서의 multiplication 이 일치하도록 만들고 싶습니다. $$(1)$$ 에 대해 inverse Fourier transform 을 적용하면, 다음의 결과를 얻게 됩니다.

$$
g\ast f = \sum^{N-1}_{l=0} \hat{g}(\lambda_l) \hat{f}(\lambda_l)u_l
\tag{2}
$$

따라서, vertex domain 에서 정의된 두 함수 $$f$$ 와 $$g$$ 에 대해 convolution operator $$\ast$$ 는 $$(2)$$ 과 같이 정의합니다. 이는 기존의 convolution 에서 $$\left\{e^{2\pi i\xi t}\right\}_{\xi\in\mathbb{R}}$$ 대신 graph Laplacian eigenvector $$\{u_l\}^{N-1}_{l=0}$$  을 사용했다고 이해할 수 있습니다. ( $$\left\{e^{2\pi i\xi t}\right\}_{\xi\in\mathbb{R}}$$ 와 $$\{u_l\}^{N-1}_{l=0}$$ 의 관계에 대해서는 [포스트](https://harryjo97.github.io/theory/Graph-Laplacian/) 를 참고해주세요 )



또한 $$(2)$$ 는 Hadamard product $$\odot$$ 와  $$\{u_l\}^{N-1}_{l=0}$$ 을 column vector 로 가지는 matrix $$U$$ 를 사용해 다음과 같은 형태로도 표현할 수 있습니다.


$$
g \ast f = U((U^Tg) \odot (U^Tf))
$$






## 2. Spectral Filtering 



Spectral graph theory 와 같이 복잡한 이론을 통해 그래프에서 convolution 을 정의한 이유는, 바로 CNN 을 그래프에 적용하기 위해서입니다. CNN 은 large-scale high dimensional 데이터로 부터 local structure 를 학습하여 의미있는 패턴을 잘 찾아냅니다. Local feature 들은 convolutional filter 로 표현되며, filter 는 translation-invariant 이기 때문에 공간적인 위치나 데이터의 크기에 상관없이 같은 feature 를 뽑아낼 수 있습니다. 



하지만, 그래프와 같이 irregular (non-Euclidean) domain 에서는 직접 translation 과 convolution 을 정의할 수 없기 때문에,  CNN 을 그래프에 바로 적용할 수 없습니다. 따라서, 그래프에 맞는 filtering 이 필요하며, 이를 spectral filtering 이라고 부릅니다.



다음과 같이 convolution 을 사용해 그래프의 함수 $$f_{in}$$ 의 filter $$g$$ 에 대한 spectral filtering 을 정의할 수 있습니다. 


$$
f_{out} = g\ast f_{in}
$$

$$(1)$$ 을 이용하면,

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
\end{bmatrix} \\
\\
&= U\hat{g}(\Lambda)U^T\;f_{in} \tag{4}

\end{align}
$$

&nbsp;

$$(4)$$ 에서 $$\hat{g}(\Lambda)$$ 는 다음과 같이 정의합니다.

$$
\hat{g}(\Lambda) =
\begin{bmatrix}
\hat{g}(\lambda_0) & 0 & \cdots & 0 \\
0 & \hat{g}(\lambda_1) & \cdots & 0 \\
\vdots &  & \ddots & \\
0 & 0 & 0 & \hat{g}(\lambda_{N-1})
\end{bmatrix}
$$



정리하자면, filter $$g$$  에 대한 spectral graph filtering 은 다음과 같습니다.

$$
f_{out} = U\hat{g}(\Lambda)U^T f_{in}
\tag{5}
$$



$$(5)$$ 를 통해 graph convolutional network 의 convolutional layer 를 만들 수 있습니다.

&nbsp;

## 3. Next

다음 포스트에서는 $$(5)$$ 를 효율적으로 계산할 수 있는 Polynomial Approximation of Spectral Filtering 대해 설명하겠습니다.





