---
title: Graph Laplacian as Operator
description : with Spectral Graph Theory
date: 2021-02-20 23:00:00 +0900
category: 
- theory
tag:
- graph laplacian
published: false
---



Graph Laplacian 과 Laplacian opeartor 의 관계



이번 포스트에서는 graph Laplacian 과 Laplacian operator $$\Delta$$ 의 관계에 대해 설명하려고 합니다. $$\Delta$$ 는 $$n$$ 차원 Euclidean space 에서 정의된 second-order differential operator 로 $$f$$ 의 gradient $$\nabla f$$ 에 대한 divergence 로 정의됩니다.
$$
\Delta f = div(\nabla f)
$$

이를 그래프에 적용하기 위해서는 그래프에서의 function, gradient, 그리고 divergence 크게 세 가지를 정의해야 합니다.



## Function for graph



그래프에서의 함수는 각각의 vertex 를 feature 에 매칭해주는 역할을 가집니다. 이 때 feature space 를 $$F$$ 차원의 vector 로 표현할 수 있으므로, 그래프 영역에서의 함수는 $$f : V \rightarrow \mathbb{R}^{F}$$ 으로 정의할 수 있습니다. 이와 같은 접근법을 Graph signal processing 이라고 하면, 자세한 설명은 Reference 의 [The Emerging Field of Signal Processing on Graphs](https://arxiv.org/pdf/1211.0053.pdf) 을 참고하기 바랍니다.



## Gradient for graph



Euclidean space 에서의 gradient 는 함수의 방향에 따른 도함수를 의미합니다. 그래프와 같이 discrete 한 경우에는 도함수를 함수값의 difference 로 해석할 수 있습니다. Euclidean space 에서 방향의 개념을 그래프에서의 edge 로 생각한다면, edge $$e = (i,j)$$ 에 대한 함수 $$f$$ 의 gradient 를 $$f(i)-f(j)$$  로 정의할 수 있습니다.

edge 의 시작점에 $$+1$$, 끝점에 $$-1$$ 을 부여하는 incidence matrix $$K$$ 를 통해 











그래프의 모든 edge 에 대해 gradient 를 표현하면, 다음과 같습니다.
$$
\nabla f = K^T f
$$

즉 gradient 를 edge 들에 대한 함수로 해석할 수 있습니다.



## Divergence for graph



Euclidean space 에서의 divergence 는 한 점에 대한 "vector field" 의 "net outward flux" 를 의미합니다.  그래프에서의 vector field 는 gradient 이고 edge 들에 대한 함수입니다. 

따라서 vertex $$i$$ 에 대한 gradient 의 "net outward flux" 는 $$i$$ 와 연결된 edge 들에 대한 gradient 함수값의 합이라고 생각할 수 있습니다. 이를 vertex $$i$$ 에 대해  

$$
\sum_{e=(i,j)\in E} g(e) - \sum_{e' = (j,i)\in E} g(e') = \sum_{e\in E}K_{ie}g
$$

정리하면, gradient $$g$$ 에 대한 divergence 를 다음과 같이 표현됩니다.

$$
div(g) = Kg
$$



## Graph Laplacian as Operator



위의 결과들을 종합하면 그래프에서의 Laplacian operator 는 다음과 같습니다.

$$
\Delta f = div(\nabla f) = KK^T f
$$


이 때 행렬 $$KK^T$$ 를 살펴 보면,

$$
(KK^T)_{ij} = \sum^E_{s=1} K_{is}K^T_{sj}
$$

이제 off-diagonal 의 원소와 diagonal 의 원소를 나눠서 계산해보겠습니다.

만약 $$i\neq j$$ 라면,

$$
K_{is}K^T_{sj} = \begin{cases}
-1 &\mbox{ if }\; e_s = (i,j)  \\
0 &\mbox{ otherwise }
\end{cases}
$$

만약 $$i=j$$ 라면,

$$
K_{is}K^T_{si} = \begin{cases}
1 &\mbox{ if }\; i\; \mbox{ in edge }\; e_s\\
0 &\mbox{ otherwise }
\end{cases}
$$

따라서,

$$
(KK^T)_{ij} = \begin{cases}
-1 &\mbox{ if }\; i\neq j \\
deg(i) &\mbox{ if }\; i=j
\end{cases}
$$

결론적으로 $$KK^T$$ 는 그래프의 adjacency matrix $$W$$ 에 대한 unnormalized graph Laplacian $$L = D-W$$ 와 일치합니다. 그렇기 때문에 그래프에서의 Laplacian operator 는

$$
\Delta f = Lf
$$

으로 정의할 수 있습니다.

이를 weighted graph 에 적용하면, 임의의 $$f\in\mathbb{R}^{N\times F}$$ 에 대해 weighted graph 의 graph Laplacian 은 

$$
(Lf)(i) 
= \left( (D-W)
\begin{bmatrix}
f(1) \\ \vdots \\ f(N) 
\end{bmatrix} 
\right)(i)
= \sum^{N}_{j=1} W_{ij} (f(i)-f(j))
$$

를 만족하는 difference operator 입니다.



## Smoothness



마지막으로, graph Laplacian 의 의미에 대해서 얘기해보겠습니다. Laplacian operator 는 second-order differential operator 로 함수가 얼마나 "매끄러운지" 를 알려줍니다. 이를 그래프 관점에서 보면, "매끄러운" 함수란 edge 로 연결된 점들에 대한 함숫값이 많이 변하지 않는다는 것입니다. edge 의 양 끝점의 함숫값의 차이가 작아야 한다는 것을 Mean Square Error 의 형태를 사용하면 

$$
\sum_{e=(i,j)\in E} W_{ij} \|f(i)-f(j)\|^2
$$

로 표현할 수 있습니다.

$$(1)$$의 식을 다시 보면,

$$
f^TLf = \frac{1}{2} \sum_{i,j} W_{ij}\|f(i)-f(j)\|^2
$$

edge 가 없는 두 vertex $$i$$ 와 $$j$$ 에 대해 $$W_{ij}=0$$ 이기 때문에 위의 두 식은 동일합니다. 

따라서 graph Laplacian $$L$$ 은 operator 의 관점에서, 그래프에서 정의된 함수가 얼마나 "매끄러운지"를 알려줍니다. 





## Reference

1.  D. Shuman, S. Narang, P. Frossard, A. Ortega, and P. Vandergheynst. [The Emerging Field of Signal
   Processing on Graphs: Extending High-Dimensional Data Analysis to Networks and other Irregular Domains](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6494675). _IEEE Signal Processing Magazine_, 30(3):83–98, 2013.

   

2. Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. [Spectral networks and locally
   connected networks on graphs](https://arxiv.org/pdf/1312.6203.pdf). In _International Conference on Learning Representations (ICLR)_,
   2014.



3. Mikhail Belkin and Partha Niyogi. [Laplacian eigenmaps and spectral techniques for embedding
   and clustering](https://papers.nips.cc/paper/2001/file/f106b7f99d2cb30c3db1c3cc0fde9ccb-Paper.pdf). In NIPS, volume 14, pages 585–591, 2001.



4. F. R. K. Chung. Spectral Graph Theory, volume 92. American Mathematical Society, 1997.

