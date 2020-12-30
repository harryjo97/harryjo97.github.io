---
title: Graph Laplacian
description : with Spectral Graph Theory
date: 2020-12-27 22:00:00 +0900
category: 
- theory
tag:
- gnn
- Graph Laplacian
---



Spectral Graph Theory 이용한 Graph Laplacian 의 이해.



## 1. Adjacency Matrix, Degree Matrix 



주어진 undirected weighted graph $$G = (V,E,W)$$ 는 vertices 의 집합 $$V$$, edges 의 집합 $$E$$, 그리고 weighted adjacency matrix $$W$$ 로 이루어집니다. 이 포스트에서는 $$\vert V\vert= N < \infty $$ 을 가정합니다. 

$$E$$ 의 원소 $$ e = (i,j) $$ 는 vertex $$i$$ 와 $$j$$ 를 연결하는 edge 를 나타내고,  $$W_{ij}$$ 는 edge $$e = (i,j)$$ 의 weight 을 의미합니다. 만약 $$i$$ 와 $$j$$를 연결하는 edge 가 없다면 $$W_{ij}=0$$ 으로 설정합니다. $$W$$ 는 모든 vertex pair 마다 정의되며 그래프가 undirected 이므로, $$W$$ 는 $$N\times N$$ symmetric matrix 입니다.



쉽게 생각할 수 있는 weighted adjacency matrix 로 adjacency matrix[^1] 가 있습니다. Adjacency matrix 는 $$i$$ 와 $$j$$ 를 연결하는 edge 가 있다면 $$W_{ij} = 1$$, 없다면 $$W_{ij}=0$$ 입니다. 아래는 그래프 (labeled) 에 대한 adjacency matrix 의 예시입니다. 



<p align='center'>
    <img src = '/assets/post/Graph-Laplacian/adjacency.gif' style = 'max-width: 100%; height: auto'>
</p>



주어진 weighted adjacency matrix $$W$$ 에 대해 degree matrix  $$D$$ 는 

$$
D_{ii} = \sum^{N}_{j=1} W_{ij}
$$

를 만족하는 diagonal matrix 로 정의합니다. 쉽게 말해, $$D_{ii}$$ 는 vertex $$i$$ 를 끝점으로 가지는 edge 들의 weight 를 모두 더한 값과 같습니다. 위의 예시처럼 edge 마다 weight 를 1로 설정한다면, degree matrix 의 diagonal element 는 각 vertex 의 degree 를 의미하기 때문에 degree matrix 라는 명칭이 붙었습니다.



그래프의 edge weight 가 주어지지 않은 경우에는 threshold Gaussian kernel weighting function을 사용해 아래와 같이 weighted adjacency matrix $$W$$ 를 정의할 수 있습니다. 그래프를 사용한 semi-supervised learning 에서는 $$dist(i,j)$$ 로 vertex $$i$$ 와 $$j$$ 의 feature vector 사이의 Euclidean distance 를 사용합니다.

$$
W_{ij} = 
\begin{cases}
\exp \left( \frac{dist(i,j)^2}{2\theta^2} \right) &\mbox{ if }\; dist(i,j)\leq k \\
0 &\mbox{ otherwise }
\end{cases}
$$

&nbsp;

## 2. Unnormalized Graph Laplacian



주어진 undirected weighted graph $$ G = (V,E,W) $$ 에 대해  unnormalized graph Laplacian[^2] 은

$$
L = D-W
$$

로 정의합니다. $$D$$ 는 앞서 정의한 degree matrix 입니다. 

아래의 그림은 그래프 (labeled) 에 대한 adjacency matrix, degree matrix, 그리고 graph Laplacian matrix 의 예시입니다.



<p align='center'>
    <img src = '/assets/post/Graph-Laplacian/graph-eg.jpg' style = 'max-width: 100%; height: auto'>
</p>



> Real symmetric matrix

Undirected weighted graph $$G$$ 에 대해 $$W$$ 와 $$D$$ 는 real symmetric matrix 이므로, $$L=D-W$$ 또한 real symmetric matrix 입니다. 


> Positive semi-definite

임의의 $$x\in\mathbb{R}^N$$ 에 대해,

$$
\begin{align}
x^TLx 
&= x^TDx - x^TWx = \sum_{i} D_{ii}x_i^2 - \sum_{i,j}x_iW_{ij}x_j \\
&= \frac{1}{2} \left( 2\sum_{i} D_{ii}x_i^2 - 2\sum_{i,j} W_{ij}x_ix_j \right) \\
&\overset{\mathrm{(1)}}{=} \frac{1}{2}\left( 2\sum_{i}\left\{\sum_{j}W_{ij}\right\} x_i^2 - 2\sum_{i,j} W_{ij}x_ix_j \right) \\
&\overset{\mathrm{(2)}}{=} \frac{1}{2}\left( \sum_{i,j}W_{ij}x_i^2 + \sum_{i,j}W_{ij}x_j^2  - 2\sum_{i,j} W_{ij}x_ix_j \right) \\
&= \frac{1}{2}\sum_{i,j} W_{ij}(x_i-x_j)^2 \geq 0 \tag{$\dagger$}
\end{align}
$$

을 만족합니다. 유도 과정에서의 (1) 은 degree matrix 의 정의를 사용하였고, (2) 는 $$W$$ 가 symmetric 이기 때문에 성립합니다.

따라서 $$L$$ 은 positive semi-definite matrix 입니다.



> Eigenvalue and eigenvector

$$L$$ 이 real symmetric, positive semi-definite matrix 이므로, $$L$$ 은 non-negative real eigenvalue 들을 가집니다. 

서로 다른 eigenvalue $$\lambda$$, $$\mu$$ 와 이에 해당하는 eigenvector $$u$$, $$v$$ 에 대해

$$
\lambda u^Tv = (\lambda u)^Tv = (Lu)^Tv = u^TLv = u^T(\mu v) = \mu u^Tv
$$

이기 때문에 $$u^Tv = 0$$ 이어야 하고, $$u$$ 와 $$v$$ 는 orthogonal 합니다.

따라서 $$L$$ 의 eigenvector 들은 서로 orthogonal 합니다.



> Zero as eigenvalue

특히 $$u_0 = \frac{1}{\sqrt{N}}\begin{pmatrix} 1 & 1 & \cdots & 1 \end{pmatrix}^T$$ 에 대해 

$$
Lu_0 = (D-W)u_0 = \mathbf{0}
$$

을 만족하기 때문에 $$L$$ 은 0 을 eigenvalue 로 가지며, 0 에 해당하는 eigenvector[^3] 는 $$u_0$$ 입니다. 여기서 기억해야 할 점은, 주어진 그래프와 상관 없이 $$L$$ 은 0 을 eigenvalue 로 가지고, eigenvector $$u_0 = \frac{1}{\sqrt{N}}\begin{pmatrix} 1 & 1 & \cdots & 1 \end{pmatrix}^T$$ 또한 변하지 않는다는 것입니다.



> Multiplicity of eigenvalue zero

$$L$$ 이 positive semi-definite 임을 증명하는 과정에서 $$(\dagger)$$ 의 결과를 이용하면,

$$
0 = u^TLu = \sum_{i,j}W_{ij}(u(i) - u(j))^2
$$

$$W_{ij}\neq 0$$ 인 모든 vertices $$i$$ 와 $$j$$ , 즉 edge로 연결된 $$i$$ 와 $$j$$ 에 대해 $$u(i) = u(j)$$를 만족합니다 (#) . 

따라서 $$k$$ 개의 connected components 를 가지는 그래프 $$G$$ 의 graph Laplacian $$L$$ 은 

$$
L = \begin{bmatrix}
L_1 & & & \\
 & L_2 & & \\
 & & \ddots & \\
 & & & L_k
\end{bmatrix}
$$

sub-Laplacian $$L_i$$ 들로 이루어진 block matrix 로 볼 수 있습니다.

각각의 sub-Laplacian 들은 모두 0 을 eigenvalue 로 가지고, eigenvector $$u$$ 는 (#)으로 인해 유일하게 결정되기 때문에 각각의 sub-Laplacian 들은 정확히 1개의 0 eigenvalue 를 가집니다.  그렇기 때문에 $$L$$ 은 정확히 $$k$$ 개의 0 eigenvalue 들을 가집니다.

정리하면, graph Laplacian $$L$$ 의 eigenvalue 0 의 multiplicity 는 주어진 그래프 $$G$$ 의 connected components 의 개수와 일치합니다.



주어진 그래프를 connected 라고 가정하고 $$L$$ 의 eigenvalue 들을

$$
0 = \lambda_0 < \lambda_1 \leq \cdots \leq \lambda_{N-1}
$$

으로 나타낼 수 있습니다. $$L$$ 의 spectrum[^4]을 연구하는 분야가 spectral graph theory  입니다. Spectral graph theory 에서 다루는 Fideler vector, Cheegar Constant, Laplacian embedding, NCut 등에 대해서는, 기회가 생기면 다른 포스트를 통해 자세히 설명하겠습니다.



&nbsp;



## 3. Other Graph Laplacians 



> Normalized graph Laplacian

Normalized graph Laplacian $$L^{norm}$$ 은 다음과 같이 정의합니다.

$$
L^{norm} = D^{-1/2}\;L\;D^{-1/2} = I -  D^{-1/2}\;W\;D^{-1/2}
$$

$$L$$ 과 같이 symmetric positive semi-definite matrix 입니다. 따라서 [Unnormalized Graph Laplacian](#unnormalized-graph-laplacian) 파트에서 설명한 eigenvalue 에 대한 성질이 동일하게 적용됩니다.  하지만 $$L$$ 과 $$L^{norm}$$ 은 similar matrices 가 아니기 때문에 다른 eigenvector 를 가집니다. $$L$$ 의 eigenvalue 0 에 대한 eigenvector $$u_0$$ 는 그래프에 상관 없이 일정하지만, $$L^{norm}$$ 의 경우 그래프에 따라 변합니다. 그렇기 때문에, 다음의 포스트에서 설명할 graph Fourier transform 의 결과 또한 달라집니다.

Normalized graph Laplacian 의 특징으로는, eigenvalue 들이 $$[0,2]$$ 에 속한다는 것입니다. 특히, 그래프 $$G$$ 가 bipartite graph 일 때만 $$L^{norm}$$ 의 가장 큰 eigenvalue 가 2가 됩니다.   



> Random walk graph Laplacian

Random walk graph Laplacian $$L^{rw}$$ 는 다음과 같이 정의합니다. 

$$
L^{rw} = D^{-1}L = I - D^{-1}W
$$

여기서 $$D^{-1}W$$ 는 random walk matrix 로 그래프 $$G$$ 에서의 Markov random walk 를 나타내어 줍니다. $$L^{rw}$$ 는 $$L$$, $$L^{norm}$$ 과 마찬가지로 positive semi-definite matrix 이지만, symmetric 이 보장되지 않습니다. 



&nbsp;



## 4. Graph Laplacian as Operator



마지막 파트에서는 graph Laplacian 과 Laplacian operator $$\Delta$$ 의 관계에 대해 설명하려고 합니다. $$\Delta$$ 는 $$n$$ 차원 Euclidean space 에서 정의된 second-order differential operator 로 $$f$$ 의 gradient $$\nabla f$$ 에 대한 divergence 로 정의됩니다.

$$
\Delta f = div(\nabla f)
$$

이를 그래프에 적용하기 위해서는 그래프에서의 function, gradient, 그리고 divergence 크게 세 가지를 정의해야 합니다.



> Function for graph

그래프에서의 함수는 각각의 vertex 를 feature 에 매칭해주는 역할을 가진다고 자연스럽게 생각 할 수 있습니다. 이 때 feature space 를 $$N$$ 차원의 vector 로 표현할 수 있으므로, 그래프 영역에서의 함수는 $$f : V \rightarrow \mathbb{R}^N$$ 으로 정의할 수 있습니다. 이와 같은 접근법을 Graph Signal 이라고 합니다. 자세한 설명은 [The Emerging Field of Signal Processing on Graphs](https://arxiv.org/pdf/1211.0053.pdf) 을 참고하시면 큰 도움이 될 것 같습니다. 



> Gradient for graph

Euclidean space 에서의 gradient 는 함수의 방향에 따른 도함수를 의미합니다. 그래프와 같이 이산적인 경우에는 도함수를 함수값의 difference 로 해석할 수 있습니다. Euclidean space 에서의 방향의 개념을 그래프에서의 edge 로 생각한다면, edge $$e = (i,j)$$ 에 대한 함수 $$f$$ 의 gradient 를 $$f(i)-f(j)$$  로 정의할 수 있습니다.

edge 의 시작점에 $$+1$$, 끝점에 $$-1$$ 을 부여하는 incidence matrix $$K$$ 를 통해 그래프의 모든 edge 에 대해 gradient 를 표현하면, 

$$
\nabla f = K^T f
$$

즉 gradient 를 edge 들에 대한 함수로 해석할 수 있습니다.




> Divergence for graph

Euclidean space 에서의 divergence 는 한 점에 대한 "vector field" 의 "net outward flux" 를 의미합니다.  그래프에서의 vector field 는 gradient 이고 edge 들에 대한 함수입니다. 

따라서 vertex $$i$$ 에 대한 gradient 의 "net outward flux" 는 $$i$$ 와 연결된 edge 들에 대한 gradient 함수값의 합이라고 생각할 수 있습니다. 이를 vertex $$i$$ 에 대해  

$$
\sum_{e=(i,j)\in E} g(e) - \sum_{e' = (j,i)\in E} g(e') = \sum_{e\in E}K_{ie}g
$$

정리하면, gradient $$g$$ 에 대한 divergence 를 다음과 같이 표현됩니다.

$$
div(g) = Kg
$$




> Graph Laplacian (unnormalized)

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

이를 weighted graph 에 적용하면, 임의의 $$f\in\mathbb{R}^N$$ 에 대해 weighted graph 의 graph Laplacian 은 

$$
(Lf)(i) 
= \left( (D-W)
\begin{bmatrix}
f(1) \\ \vdots \\ f(N) 
\end{bmatrix} 
\right)(i)
= \sum_{i\sim j} W_{ij} (f(i)-f(j))
$$

를 만족하는 difference operator 입니다.



마지막으로, graph Laplacian 의 의미에 대해서 얘기해보겠습니다. Laplacian operator 는 second-order differential operator 로 함수가 얼마나 "매끄러운지" [^5]를 알려줍니다. 이를 그래프 관점에서 보면, "매끄러운" 함수란 edge 로 연결된 점들에 대한 함숫값이 많이 변하지 않는다는 것입니다. edge 의 양 끝점의 함숫값의 차이가 작아야 한다는 것을 Mean Square Error 의 형태를 사용하면 

$$
\sum_{e=(i,j)\in E} W_{ij} (f(i)-f(j))^2
$$

로 표현할 수 있습니다.

[Unnormalized Graph Laplacian](#unnormalized-graph-laplacian) 파트의 $$(\dagger)$$ 를 다시 보면,

$$
f^TLf = \frac{1}{2} \sum_{i,j} W_{ij}(f(i)-f(j))^2
$$

edge 가 없는 두 vertex $$i$$ 와 $$j$$ 에 대해 $$W_{ij}=0$$ 이기 때문에 위의 두 식은 동일합니다. 

따라서 graph Laplacian $$L$$ 은 operator 의 관점에서, 그래프에서 정의된 함수가 얼마나 "매끄러운지"를 알려줍니다. 



&nbsp;



## 5.  Next



다음 포스트에서는 graph Fourier transform 에 대해 설명하겠습니다.



&nbsp;

&nbsp;





[^1]: adjacency matrix 는 보통 $$A$$ 로 나타내며, edge 의 weight 가 0 또는 1 입니다.
[^2]: combinatorial graph Laplacian 으로도 불립니다. 
[^3]: $$u_0$$ 는 유일한 eigenvector 가 아닐 수 있습니다.
[^4]:  eigenvalue 들을 spectrum 이라고 부릅니다.
[^5]: "smooth" 