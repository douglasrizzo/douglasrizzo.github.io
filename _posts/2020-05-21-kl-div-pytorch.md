---
layout: post
title: Solving the mistery of the KL divergence
categories: colab python machine-learning pytorch programming
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f2NnBBRUjrZYLdJC9-MPn1TqXdxW-s2e?usp=sharing)

In this notebook, I try to understand how the KL divergence works, specifically the one from PyTorch.

Relevant docs are here: https://pytorch.org/docs/stable/nn.html#torch.nn.KLDivLoss

Basically, given an $N \times \ast $ tensor `x`, where $\ast$ represents any number of dimensions besides the first one, the first dimension of `x` will hold $N$ tensors. Each one of these tensors symbolizes a (discrete) probability distribution. This means that each of the tensors must sum to 1 (`x.sum(0) = [1.0 ,1.0 ,1.0 ,...]`).

An easy way to do that to the output of a neural network is to use the softmax function. Another is to divide each value inside the tensor by the sum of all values.

```python
import torch
import torch.nn.functional as F
```

In this function, I calculate the KL divergence betwwen `a1` and `a2` both by hand as well as by using PyTorch's `kl_div()` function. My goals were to get the same results from both and to understand the different behaviors of the function depending on the value of the `reduction` parameter.

First, both tensors must have the same dimensions and every single tensor after dimension 0 must sum to 1, _i.e._ dimension 0 is the batch dimension and each individual tensor in this dimension represents a (discrete) probability distribution. Applying `x.softmax(0)` accomplishes this.

Furthermore, we need to apply the log to the values in the
first collection. `log_softmax(0)` accomplishes both at the same time.

```python
def kl_div(a1, a2):
    # the individual terms of the KL divergence can be calculated like this
    manual_kl = (a2.softmax(0) * (a2.log_softmax(0) - a1.log_softmax(0)))

    # applying necessary transformations
    a1ready = a1.log_softmax(0)
    a2ready = a2.softmax(0)

    print('\nSums')
    print(manual_kl.sum())
    print(F.kl_div(a1ready, a2ready, reduction='none').sum())
    print(F.kl_div(a1ready, a2ready, reduction='sum'))

    print('\nMeans')
    print(manual_kl.mean())
    print(F.kl_div(a1ready, a2ready, reduction='none').mean())
    print(F.kl_div(a1ready, a2ready, reduction='mean'))

    print('\nBatchmean')
    print(manual_kl.mean(0).sum())
    print(F.kl_div(a1ready, a2ready, reduction='batchmean'))
```

Here I apply the above function on 2D tensors.

```python
torch.manual_seed(1)
dist = torch.distributions.uniform.Uniform(0,10)
a1 = dist.sample((5 ,2))
a2 = dist.sample((5, 2))

print((a1/a1.sum()).sum())

kl_div(a1, a2)
```

    tensor(1.0000)
    
    Sums
    tensor(4.3808)
    tensor(4.3808)
    tensor(4.3808)
    
    Means
    tensor(0.4381)
    tensor(0.4381)
    tensor(0.4381)
    
    Batchmean
    tensor(0.8762)
    tensor(0.8762)

    /home/user/.anaconda3/lib/python3.7/site-packages/torch/nn/functional.py:2247: UserWarning: reduction: 'mean' divides the total loss by both the batch size and the support size.'batchmean' divides only by the batch size, and aligns with the KL div math definition.'mean' will be changed to behave the same as 'batchmean' in the next major release.
      warnings.warn("reduction: 'mean' divides the total loss by both the batch size and the support size."

Here I apply the above function on 3D tensors.

```python
torch.manual_seed(1)
dist = torch.distributions.uniform.Uniform(0,10)
a1 = dist.sample((10, 6, 4))
a2 = dist.sample((10, 6, 4))
a1s = a1.softmax(2)
a2s = a2.softmax(2)

kl_div(a1, a2)
```

    
    Sums
    tensor(90.7882)
    tensor(90.7882)
    tensor(90.7882)
    
    Means
    tensor(0.3783)
    tensor(0.3783)
    tensor(0.3783)
    
    Batchmean
    tensor(9.0788)
    tensor(9.0788)

