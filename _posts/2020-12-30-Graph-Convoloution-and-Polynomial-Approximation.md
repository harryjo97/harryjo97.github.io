---
title: Graph Convoloution and Polynomial Approximation
description: based on "Wavelets on Graphs via Spectral Graph Theory"
date: 2020-12-30 15:00:00 +0900
category:
- theory
tag:
- gnn
- Graph convolution
- Polynomial approximation
---



based on the paper [Wavelets on Graphs via Spectral Graph Theory](https://arxiv.org/pdf/0912.3848.pdf)



## 0. Graph Fourier Transform



Graph Fourier transform 에 대해 자세히 설명한 [포스트](https://harryjo97.github.io/theory/Graph-Fourier-Transform/) 를 보고 오시면, graph convolution 을 이해하는데 큰 도움이 될 것입니다.



## 1. Classical Convolution









## 2. Graph Convolution







## 3. Spectral Graph Wavelet Transform



kernel function $$ g:\mathbb{R}^+ \rightarrow \mathbb{R}^+$$ bandpass filter

$$g(0) = 0$$ and $$ \lim_{x\rightarrow\infty}g(x) = 0$$ 



wavelet operator $$T_g = g(L)$$ 를 다음과 같이 정의합니다.


$$
\widehat{T_gf}(\lambda_l) = g(\lambda_l)\hat{f}(\lambda_l)
$$






## 4. Polynomial Approximation




$$
\hat{f}_{out}(\lambda_l) = \hat{g}_t(\lambda_l)\hat{f}_{in}(\lambda_l)
$$

$$
f_{out} = U\hat{f}_{out}
= U\hat{g}_t^T\hat{f}_{in} 
= U\hat{g}_t^TU^T (U\hat{f}_{in})
= U\hat{g}_t^TU^T\;f_{in}
$$




## 4. Next

다음 포스트에서는 [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf) 의 paper review 를 하겠습니다.

