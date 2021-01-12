---
title: Semi-Supervised Classification with Graph Convolutional Networks
date: 2021-01-13 23:00:00 +0900
category: 
- paper review
tag
- gcn
---

Graph Convolutional Network



## Preliminary





## Introduction



논문에서 다루는 문제는 다음과 같습니다.

> Classifying nodes in a graph where labels are only available for a small subset of nodes.

즉 그래프의 node 들 중 label 이 주어진 node 들의 수가 적을 때의 상황을 고려합니다.



GCN 을 사용하기 이전에는 주로 explicit graph-based regularization [(Zhu et al., 2003)](https://www.aaai.org/Papers/ICML/2003/ICML03-118.pdf) 을 이용하여 문제에 접근하였습니다. 이 방법은 다음과 같이 supervised loss $$\mathcal{L}_0$$ 에 graph Laplacian regularization 항 $$\mathcal{L}_{reg}$$ 을 더한 loss function 을 이용하여 학습합니다.

$$
\mathcal{L} = \mathcal{L}_0 + \lambda\mathcal{L}_{reg},\;\;\; \mathcal{L}_{reg} = f(X)^T\,L\,f(X)
\tag{1}
$$
$$(1)$$ 에서 $$f$$ 는 neural network 와 같이 differentiable 함수이며 $$X$$ 는 feature vector 들의 matrix , 그리고 $$L$$ 은 unnormalized graph Laplacian 입니다. 그래프의 adjacency matrix $$A$$ 에 대해 $$ f(X)^T\;L\;f(X) = \sum_{i,j} A_{ij} \|f(X_i)-f(X_j)\|^2$$ 를 만족하기 때문에, $$\mathcal{L}_{reg}$$ 를 이용하는 것은 그래프의 인접한 node 들은 비슷한 feature 를 가질 것이라는 가정을 전제로 합니다. 



Explicit graph-based regularization 을 사용하지 않기 위해, 그래프의 구조를 포함하도록 neural network model 로 $$f(X,A)$$ 를 사용합니다. 





&nbsp;

## Fast Approximate Convolutions On Graphs



### Spectral Graph Convolution

Graph signal $$x\in\mathbb{R}^N$$ 와 filter $$g_{\theta}=\text{diag}(\theta)$$ 에 대해 spectral convolution 은 다음과 같이 주어집니다.

$$
g_{\theta}\ast x = Ug_{\theta}U^Tx
\tag{2}
$$

여기서 $$U$$  는 normalized graph Laplacian $$L = I - D^{-1/2}AD^{-1/2}$$ 의 eigenvector 로 이루어진 Fourier basis 입니다. Filter $$g_{\theta}$$ 는 다음과 같이 $$L$$ 의 eigenvalue 들의 함수로 생각할 수 있습니다.
$$
g_{\theta}(\Lambda) =
\begin{bmatrix}
g_{\theta}(\lambda_0) & & & \\
 & g_{\theta}(\lambda_1) & & \\
  & & \ddots & \\
  & & & g_{\theta}(\lambda_{N-1})
\end{bmatrix}
$$

$$(2)$$ 을 계산하기 위해서는 $$U$$ 와 matrix multiplication 을 수행해야하며, 이는 $$O(N^2)$$ 으로 상당히 복잡하 연산입니다. 또한 $$U$$ 를 구하기 위한 eigendecomposition 연산은 $$O(N^3)$$ 이므로 node 의 개수가 수천 수만개인 그래프에 대해서 $$(2)$$ 를 계산하는 것은 굉장히 힘듭니다.



이를 해결하기 위해, truncated Chebyshev expansion 을 통해 $$g_{\theta}(\Lambda)$$ 를 다음과 같이 근사합니다.

$$
g_{\theta'}(\Lambda) \approx \sum^K_{k=0} \theta'_{k}T_k(\tilde{\Lambda})
\tag{3}
$$

여기서 $$\tilde{\Lambda} = \frac{2}{\lambda_{max}}\Lambda - I$$ 로 Chebyshev expansion 을 위해 scale 을 맞춰준 것이고, $$\theta'\in\mathbb{R}^K$$ 은 Chebyshev coefficient 들의 vector 입니다.



$$(3)$$ 의 근사를 $$(2)$$ 에 대입하면 $$\tilde{L} = \frac{2}{\lambda_{max}}L - I$$ 에 대해 다음의 결과를 얻을 수 있습니다.

$$
g_{\theta'}\ast x \approx \sum^K_{k=0} \theta'_kT_k(\tilde{L})x
\tag{4}
$$

$$(4)$$ 는 graph Laplacian 의 localization 을 



### Layer-wise Linear Model

overfitting 





$$(4)$$ 에서 $$K=1$$ 로 제한을 두면 다음과 같이 두 개의 parameter $$\theta'_0$$ 와 $$\theta'_1$$ 을 통해 $$(2)$$ 를 표현할 수 있습니다.
$$
g_{\theta'}\ast x \approx \theta'_0x + \theta'_1(L-I)x = \theta'_0x - \theta'_1D^{-1/2}AD^{-1/2}x
\tag{5}
$$


더 나아가 하나의 parameter $$\theta = \theta'_0 = -\theta'_1$$ 만을 사용하면,
$$
g_{\theta}\ast x \approx \theta(I + D^{-1/2}AD^{-1/2})x
\tag{6}
$$
이 때 $$I + D^{-1/2}AD^{-1/2}$$ 의 eigenvalue 는 $$[0,2]$$ 에 속하기 때문에, 여러개의 layer 를 쌓아 deep model 을 만든다면 exploding / vanishing gradient problem 과 같이 불안정한 학습이 이루어질 수 있습니다.



따라서, $$I + D^{-1/2}AD^{-1/2}$$ 를 $$\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$$ 로 바꾸어주는 renormalization trick 을 적용합니다.



$$C$$ 개의 input channel 을 가지는 signal $$X\in\mathbb{R}^{N\times C}$$ 과 $$F$$ 개의 feature map 에 대해서 일반화시키면 다음의 결과를 얻을 수 있습니다.
$$
Z = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}X\Theta
$$
$$\Theta\in\mathbb{R}^{C\times F}$$ 는 filter의 parameter matrix 이고 $$Z\in\mathbb{R}^{N\times F}$$ 가 filtering 의 결과



&nbsp;


## Semi-Supervised Node Classification









## Related Work







## Experiments







## Results









## Discussion









## Conclusion









## Appendix







## Reference







