---
title: Graph Convoloution and Spectral Graph Wavelet Transform
description: based on "Wavelets on Graphs via Spectral Graph Theory"
date: 2020-12-30 15:00:00 +0900
category:
- theory
tag:
- gnn
- Graph convolution
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







## 4. Next

다음 포스트에서는 



Polynomial approximation

