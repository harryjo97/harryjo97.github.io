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

GCN 을 사용하기 이전에는 주로 explicit graph-based regularization [(Zhu et al., 2003)](https://www.aaai.org/Papers/ICML/2003/ICML03-118.pdf) 을 이용하여 문제에 접근하였습니다. 

이 방법은 다음과 같이 supervised loss $$\mathcal{L}_0$$ 에 graph Laplacian regularization 항 $$\mathcal{L}_{reg}$$ 을 더한 loss function 을 이용하여 학습합니다.
$$
\mathcal{L} = \mathcal{L}_0 + \lambda\mathcal{L}_{reg},\;\;\; \mathcal{L}_{reg} = f(X)^T\,L\,f(X)
\tag{1}
$$
$$(1)$$ 에서 $$f$$ 는 neural network 와 같이 differentiable 함수이며 $$X$$ 는 feature vector 들의 matrix , 그리고 $$L$$ 은 unnormalized graph Laplacian 입니다.



그래프의 adjacency matrix $$A$$ 에 대해 $$ f(X)^T\;L\;f(X) = \sum_{i,j} A_{ij} \|f(X_i)-f(X_j)\|^2$$ 를 만족하기 때문에, $$\mathcal{L}_{reg}$$ 는 $$f$$ 가 얼마나 smooth 한지 나타내어줍니다. 즉, $$(1)$$ 은 그래프의 인접한 node 들은 비슷한 feature 를 가질 것이라는 가정을 전제로 합니다. 











## Fast Approximate Convolutions On Graphs









## Semi-Supervised Node Classification









## Related Work







## Experiments







## Results









## Discussion









## Conclusion









## Appendix







