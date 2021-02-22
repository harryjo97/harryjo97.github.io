---
title: Continuous Graph Neural Networks
date: 2021-02-24 22:00:00 +0900
category:
- paper review
tag:
- neural ode
published: false
---

[paper review] CGNN 이해하기



## Introduction

&nbsp;

GCN 의 propagation rule 은 neighbor node representation 들을 통해 node representation 을 update 합니다 [3]. 이 때 update 과정에서 각 node 와 주변의 node 들의 representation 이 점점 비슷해집니다. GCN layer 를 많이 쌓을수록 node 들의 representation 이 같은 값으로 수렴하는 over-smoothing 이 발생하기 쉽고, 이는 GCN 의 performance 를 저해하게 됩니다 [4]. 깊은 GNN 모델은 node feature 들의 복잡한 상호작용을 표현할 수 있기 때문에, over-smoothing 을 해결하는 것이 GNN 의 성능 향상에 중요합니다. 



또한 기존의 GNN 은 node representation 의 이산적 변화만을 모델링할 수 있습니다.

최근 [2] 에서 ODE solver 를 활용해 





논문에서는 node representation 의 변화를 ODE 를 통해 정의하여, node representation 의 시간에 따른 연속적인 변화를 표현할 수 있는 모델 CGNN 을 제시합니다. 이를 통해 over-smoothing 문제를 해결하고, node 들의 long-range dependency 를 학습할 수 있도록 합니다. 

&nbsp;

## Preliminaries

&nbsp;

simple undirected graph $$G=(V,E)$$ 에 대해 adjacent matrix 를 $$Adj$$ 라고 하겠습니다. [3] 에서는 degree matrix $$D$$ 와 normalized adjacent matrix $$D^{-1/2}Adj\,D^{-1/2}$$ 를 통해 정의된 matrix $$I + D^{-1/2}Adj\,D^{-1/2}$$ 를 사용하여 GCN 의 propagation rule 을 정의합니다. 



이 때 $$I + D^{-1/2}Adj\,D^{-1/2}$$ 의 eigenvalue 는 $$[0,2]$$ 구간에 존재하기 때문에, $$I + D^{-1/2}Adj\,D^{-1/2}$$ 를 사용한 layer 들을 많이 거칠 경우 exploding / vanishing gradient 와 같은 instability 가 발생할 수 있습니다 [3]. 따라서 [3] 에서는 renormalization trick 을 통해 

$$I + D^{-1/2}Adj\,D^{-1/2}$$ 대신 $$\hat{A} = \tilde{D}^{-1/2}\tilde{A}\,\tilde{D}^{-1/2}$$ 
$$
H_{n+1} = \hat{A}\,H_n\Theta
$$




논문에서는 $$I + D^{-1/2}Adj\,D^{-1/2}$$ 의 eigenvalue 크기를 조절하기 위해, renormalization trick 대신 hyperparameter $$\alpha\in (0,1)$$ 을 사용해, 다음과 같은 regularized adjacency matrix $$A$$ 를 사용합니다.

$$
A = \frac{\alpha}{2}\left(I + D^{-1/2}Adj\,D^{-1/2} \right)
\tag{1}
$$






&nbsp;

## Related Works

&nbsp;

### Neural ODE

[2] 에서는 ODE solver 를 통해 hidden representation 의 연속적인 변화를 모델링합니다. 

이 때 ODE solver 를 black box 로 

adjoint sensitivity method 를 통해 gradient 를 계산해줍니다. Adjoint sensitivity method 를 사용하면 다음과 같은 장점이 있습니다.

- Memory efficiency
- Adaptive computation
- Scalable and invertible normalizing flows
- Continuous time-series models





&nbsp;

## Model

&nbsp;

CGNN 은 크게 encoder, ODE, decoder 세 가지 스텝으로 이루어져 있습니다. 

먼저 encoder (fully connected layer) $$\mathcal{E}$$ 를 통해 각 node 의 feature 를 latent space 로 보내줍니다. 즉 node feature matrix $$X\in\mathbb{R}^{|V|\times|F|}$$ 를 $$E = \mathcal{E}(X)$$ 로 변환해줍니다. 그 후 initial value $$H(0)$$ 를 $$E$$ 로 설정하고, node representation 의 연속적인 변화를 정의하는 ODE 를 만들어줍니다. 이 때 ODE 는 node 들의 long-term dependency 를 모델링할 수 있어야합니다. 마지막으로, 시간 $$t_1$$ 에서의 node representation $$H(t_1)$$ 은 decoder (fully connected layer) $$\mathcal{D}$$ 와 softmax 함수를 통해 node 들이 labeling 됩니다. 

다음의 그림을 통해 CGNN 의 구조를 이해할 수 있습니다. 그림에서 빨간색 화살표는 message passing 을 나타냅니다.

<p align='center'>
    <img src='../assets/post/Continuous-Graph-Neural-Networks/architecture.PNG' style='max-width: 80%; height: auto'>
</p>



CGNN 에서 가장 스텝은 바로 node 들의 관계를 모델링해주는

node representation 의 연속적인 변화를 정의해 줄 올바른 ODE 를 만드는 것입니다. 



논문에서는 두 가지 ODE 를 제시합니다.



### Case 1 : Independent Feature Channels

node representation 을 찾기 위해서는 node 들의 연결성을 반영해야하기 때문에, ODE 는 그래프의 구조를 고려해야합니다.  

논문에서는 PageRank 와 같은 그래프에서의 diffusion-based method 로부터 다음의 propogation rule 을 정의합니다. $$A$$ 는 $$(1)$$ 에서의 정의를 사용합니다.
$$
H_{n+1} = AH_n + H_0
\tag{2}
$$
$$(2)$$ 를 자기 자신의 처음 feature 를 기억하며 주변 node 들의 feature 를 통해 node representation 을 update 하는 과정으로 이해할 수 있습니다. 즉 원래의 node feature 를 잊어버리지 않으며 그래프의 구조를 학습할 수 있습니다. $$(2)$$ 를 통해 다음과 같이 $$H_n$$ 을 직접 표현할 수 있습니다.
$$
H_n = \left(\sum^n_{i=0} A^i\right)H_0
\tag{3}
$$
$$(3)$$ 을 통해 representation $$H_n$$ 은 $$n$$ 번째 layer 까지 propagated 된 정보를 포함한다는 것을 알 수 있습니다.



$$A$$ 의 정의에 의해 diagonalizable 하며, eigenvalue 는 $$(0,1)$$ 구간에 포함됩니다. $$A=U\Lambda U^T$$ 로 표현한다면 $$A-I = U(\Lambda-I)U^T$$ 이며, $$\Lambda-I$$ 의 diagonal element 들은 모두 0 보다 작기 때문에 $$A-I$$ 는 invertible 합니다. 따라서, $$(3)$$ 을 다음과 같이 정리할 수 있습니다.
$$
H_n = \left(\sum^n_{i=0} A^i\right)H_0 
= \left(A-I\right)^{-1}\left(A^{n+1}-I\right)H_0
\tag{4}
$$
 

discrete propagation 과정인 $$(3)$$ 을 continuous propagation 으로 확장하기 위해, $$(3)$$ 을 시간 $$t=0$$ 부터 $$t=n$$ 까지의 적분을 나타내는 Riemann sum 으로 생각합니다. 
$$
\sum^{n+1}_{i=1} A^{(i-1)\Delta t}E\Delta t
\tag{5}
$$
$$t=n$$ 이며 $$\Delta t=\frac{t+1}{n+1}$$ 일 때



$$ n\rightarrow\infty$$ 이면 $$(5)$$ 는 다음의 적분으로 
$$
H(t) = \int^{t+1}_{0} A^sE\,ds
\tag{6}
$$
$$(6)$$ 의 양변에 미분을 취하면,
$$
\frac{dH(t)}{dt} = A^{t+1}E
\tag{7}
$$
$$t$$ 가 정수가 아니라면 $$A^{t+1}$$ 을 직접 계산할 수 없기 때문에, 한 번 더 미분을 취해줍니다.
$$
\frac{d^2H(t)}{dt^2} = \ln A\,A^{t+1}E = \ln A\frac{dH(t)}{dt}
\tag{8}
$$
$$(8)$$ 의 양변을 다시 적분해줌으로써 다음의 결과를 얻습니다.
$$
\frac{dH(t)}{dt} = \ln A\,H(t) + const
\tag{9}
$$
$$(6)$$ 으로부터 $$t=0$$ 일 때, $$H(0)$$ 의 값을 구할 수 있습니다.
$$
H(0) = \int^1_0 A^sE\,ds = \left(\ln A\right)^{-1}\left( A-I\right)E
\tag{10}
$$
$$(7)$$ 과 $$(10)$$ 으로부터 $$(9)$$ 의 적분상수 $$const$$ 를 구할 수 있습니다.
$$
AE = \left.\frac{dH(t)}{dt}\right|_{t=0} = \ln A\,H(0) + const
$$

$$
const 
= AE - (A-I)E = E
$$

즉 $$(9)$$ 를 다시 쓸 수 있습니다.
$$
\frac{dH(t)}{dt} = \ln A\,H(t) + E
\tag{11}
$$


따라서 다음의 Proposition 1 을 얻을 수 있습니다.



> __Proposition 1.__
>
> The discrete dynamic in $$(3)$$ is a discretisation of the following ODE :
> $$
> \frac{dH(t)}{dt} = \ln A\,H(t)+E
> $$
> with the initial value $$H(0)=\left(\ln A\right)^{-1}\left( A-I\right)E$$

 

$$(11)$$ 의 ODE 는 epidemic model 의 관점에서 이해할 수 있습니다.







$$(11)$$ 의 ODE 는 analytical 해를 가집니다.



> __Proposition 2.__
>
> 



### Case 2 : Modelling the Interaction of Feature Channels







&nbsp;

## Discussion

&nbsp;







&nbsp;

## Experiment

&nbsp;





&nbsp;

## Appendix

&nbsp;





&nbsp;

## Reference

1. L.-P. A. Xhonneux, M. Qu, and J. Tang. [Continuous graph neural networks](https://arxiv.org/pdf/1912.00967.pdf). arXiv preprint arXiv:1912.00967, 2019.



2. Tian Qi Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. [Neural ordinary differential equations](https://arxiv.org/pdf/1806.07366.pdf). In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems 31, pages 6571–6583. Curran Associates, Inc., 2018.



3. Thomas N Kipf and Max Welling. [Semi-supervised classification with graph convolutional networks](). arXiv preprint arXiv:1609.02907, 2016. 



4. Qimai Li, Zhichao Han, and Xiao-Ming Wu. [Deeper insights into graph convolutional networks for semisupervised learning](https://arxiv.org/pdf/1801.07606.pdf). In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.