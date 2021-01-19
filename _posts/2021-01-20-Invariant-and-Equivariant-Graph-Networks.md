---
title: Invariant and Equivariant Graph Networks
date: 2021-01-20 11:00:00 +0900
category:
- paper review
tag:
- invariant graph networks
---

k-Invariant Graph Networks 이해하기 : (1) Permutation Invariant / Equivariant



## Motive



CNN 의 translation invariant 한 특성 



### Translation Invariance of CNN





translation invariant 한 함수 $$f$$ 를 근사하기 위해  Multi-layer perceptron (MLP) 

가장 기본적인 형태는 각 layer $$\mathcal{L}_i$$ 가 non-linear function $$\sigma$$ 와 linear function $$L(x) = Ax+b$$ 에 대해 다음과 같이 표현
$$
\mathcal{L}(x) = \sigma(L(x))
$$


<p align='center'>
    <img src = 'C:/Users/NGCN/Desktop/blog/assets/post/Invariant-and-Equivariant-Graph-Networks/mlp.PNG' style = 'max-width: 40%; height: auto'>
</p>





<p align='center'>
    <img src = 'C:/Users/NGCN/Desktop/blog/assets/post/Invariant-and-Equivariant-Graph-Networks/mlp-invariant.PNG' style = 'max-width: 50%; height: auto'>
</p>





이를 일반화시켜 invariant operator 를 equivariant operator 로 바꾼



<p align='center'>
    <img src = 'C:/Users/NGCN/Desktop/blog/assets/post/Invariant-and-Equivariant-Graph-Networks/mlp-equivariant.PNG' style = 'max-width: 50%; height: auto'>
</p>



Invariant Graph Networks





## Basic Definition



hypergraph

multi-set $$\{\{a,b,\cdots\}\}$$ 

$$[n]=\{1,2,\cdots,n\}$$ 

vec( )

[vec( )]





## Graphs as Tensors



$$n$$ 개의 node 를 가진 그래프를 생각해봅시다. 각 node $$i$$ 마다 value $$x_i\in\mathbb{R}$$ 를 가지고 각 edge $$(i,j)$$ 마다 value $$x_{ij}\in\mathbb{R}$$ 를 가진다면,  이는 $$X\in\mathbb{R}^{n\times n}$$ tensor 를 통해 다음과 같이 표현할 수 있습니다.
$$
X_{ij} = \begin{cases}
x_i &\mbox{ if }\; i=j \\
x_{ij} & \mbox{ if }\; i\neq j
\end{cases}
\tag{1}
$$


<p align='center'>
    <img src = 'C:/Users/NGCN/Desktop/blog/assets/post/Invariant-and-Equivariant-Graph-Networks/2-tensor.PNG' style = 'max-width: 50%; height: auto'>
</p>



$$(1)$$ 의 표현법을 hypergraph 에 대해서도 일반화할 수 있습니다. $$n$$ 개의 node 를 가지는 hypergraph 에 대해, 각 hyper-edge 는  $$(i_1,\cdots,i_k)\in [n]^k$$  의 형태로 나타낼 수 있습니다. 따라서 $$(1)$$ 과 마찬가지로, hypergraph 또한 $$X\in\mathbb{R}^{n^k}$$ tensor 로 표현할 수 있습니다.

<p align='center'>
    <img src = 'C:/Users/NGCN/Desktop/blog/assets/post/Invariant-and-Equivariant-Graph-Networks/k-tensor.PNG' style = 'max-width: 50%; height: auto'>
</p>



만약 hyper-edge 들이 $$a$$ 차원의 value 를 가진다면, $$X\in\mathbb{R}^{n^k\times a}$$  tensor 로 표현할 수 있습니다.



## Permutation Invariant / Equivariant



이미지에서의 translation 은 symmetry 의 한 종류입니다. 그래프에 있어 symmetry 는 node 순서의 재배열 (re-ordering) 을 통해 해석할 수 있습니다.

<p align='center'>
    <img src = 'C:/Users/NGCN/Desktop/blog/assets/post/Invariant-and-Equivariant-Graph-Networks/symmetry.PNG' style = 'max-width: 50%; height: auto'>
</p>





### Permutation of Tensors



그래프 node 가 재배열되면, 그에 따라 그래프 tensor 또한 변하게 됩니다. 

<p align='center'>
    <img src = 'C:/Users/NGCN/Desktop/blog/assets/post/Invariant-and-Equivariant-Graph-Networks/permutation.PNG' style = 'max-width: 70%; height: auto'>
</p>

Permutation $$p\in S_n$$ 와 $$k$$-tensor $$X\in\mathbb{R}^{n^k}$$ 에 대해, $$X$$ 에 대한 permutation $$p$$ 는 각 hyper-edge $$(i_1,\cdots,i_k)\in [n]^k$$ 에 대해 다음과 같이 쓸수 있습니다. 
$$
(p\cdot X)_{i_1,\cdots,i_k} = X_{p^{-1}(i_1),\cdots,p^{-1}(i_k)}
\tag{2}
$$


### Permutation Invariant



함수 $$f$$ 가 permutation invariant 하다는 것은, input element 들의 순서와 상관 없이 output 이 같다는 뜻입니다. $$f$$ 의 input 이 tensor 일 경우, permutation invariant 는 다음과 같이 나타낼 수 있습니다.
$$
f(p\cdot A) = f(A)
\tag{3}
$$




### Permutation Equivariant



함수 $$f$$ 가 permutation equivariant 하다는 것은, 임의의 permutation $$p$$ 에 대해 $$p$$ 와 $$f$$  가 commute 함을 의미합니다. $$f$$ 의 input 이 tensor 일 경우, permutation equivariant 는 다음과 같이 나타낼 수 있습니다.
$$
f(p\cdot A) = p\cdot f(A)
\tag{4}
$$




## Characterization



모든 permutation invaraint  / equivariant 한 linear operator 들을 찾아내는 것이 목표입니다.







## Reference







http://irregulardeep.org/An-introduction-to-Invariant-Graph-Networks-(1-2)/