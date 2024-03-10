---
layout: post
title: Reverse engineering a step decay for learning rate
categories: deep-learning pytorch tutorial programming
---

<!-- TOC -->

- [The theory behind the problem](#the-theory-behind-the-problem)
- [Problem setting](#problem-setting)
- [Finding the unknowns](#finding-the-unknowns)
- [A concrete example](#a-concrete-example)
- [A PyTorch example](#a-pytorch-example)
- [Closing remarks](#closing-remarks)

<!-- /TOC -->

## The theory behind the problem

When training a neural network using stochastic gradient descent, the learning rate is a parameter that controls the magnitude of updates to the neural network weights. Using this parameter is important since, when working with minibatch training, updates may not always be in the correct direction and magnitude to minimize the loss function, since the minibatch represents only a fraction of the training data.

For that reason, we start with a large learning rate, which allows the network to make large updates to its weights, learning a lot faster. When training is reaching its end, we slowly decrease the learning rate, forcing the network to make smaller updates towards convergence.

## Problem setting

We would like to implement a decaying strategy for the learning rate $\eta$ of your neural network. One option is to implement step decay for $\eta$, which multiplies $\eta$ by a factor $\gamma$ after every $n$ epochs. This is implemented in libraries such as PyTorch/TensorFlow as `ExponentialLR`, `StepLR` or `MultiStepLR`. These implementations usually expect you to provide the values for the initial learning rate $\eta_0$, $\gamma$ and $n$, but they don't allow you to explicitly select the final value for $\eta$ at the end of training.

We'd like to start at a reasonably high $\eta_0$ and end in an equally reasonable final value $\eta_m$ after $m$ updates. If $\eta_m$ is too low, there is the risk that additional training epochs will not have any effect in the network. If it is too high, our network may not reap the benefits of a low learning rate, which stabilizes convergence. In short:

> Having full control of the range of values the learning rate assumes would be very beneficial, but the decay strategies provided by DL frameworks don't make this task so easy to achieve, because they allow us to use a less obvious set of variables that indirectly affect the values assumed by the learning rate during training.

This setting may be familiar in deep reinforcement learning, in which a neural network may need to be trained for many epochs, to let the agent generate new transitions with updated policies, while also learning faster at the beginning of training.

In this post, given the number of updates we will apply to the learning rate (which we have called $n$), we'll see:

- how we can select the range of values the learning rate may assume ($\eta_0 \geq \eta \geq \eta_m$);
- select an appropriate value for the step decay parameter $\gamma$ that keeps $\eta$ in that range.

## Finding the unknowns

At update step $m$, the value of $\eta_m=\eta_0 \cdot \gamma^m$. If we take $\eta_m$ as the final possible value of $\eta$ and, by consequence, its lowest one, and select a reasonable value for it, we can then solve for $\gamma$:

\\[ \eta_m=\eta_0 \cdot \gamma^m \\]

\\[ \frac{\eta_m}{\eta_0}=\gamma^m \\]

\\[ \gamma=\sqrt[m]{\frac{\eta_m}{\eta_0}} \\]

The value of $m$ can be taken as the number of times $\eta_0$ will be multiplied by $\gamma$ until the end of training. Bear in mind that, while optimally, we would like to update $\eta$ after every epoch (making $m=M$), this can introduce numerical errors, since $\gamma$ would need to be too close to 1 for updates to be small enough. Instead, we set $m < M$ and tell PyTorch/TensorFlow to update $\eta$ after a number of steps $n$ such that, by the end of training, $\eta$ has been updated $m$ times.

This value $n$ can be found by dividing the total number of epochs $M$ by the number of updates $m$.

## A concrete example

- Number of training epochs $M=1000000=10^6$
- Initial learning rate $\eta_0=0.1$
- Final learning rate $\eta_m=0.0001$
- Number of updates to $\eta$ until the end of training $m=10000=10^4$
- Step decay $\gamma=\sqrt[m]{\frac{\eta_M}{\eta_0}}=\sqrt[10000]{\frac{0.0001}{0.1}} \approx 0.999309463003$
- Number of training steps between updates $n=\frac{M}{m}=\frac{10^6}{10^4}=100$

## A PyTorch example

In this example, I create an SGD optimizer for my network and show how we can accomplish our bounded learning rate using a `StepLR`, `MultiStepLR` or `ExponentialLR` scheduler available in PyTorch.

```python
max_steps  = int(1E6)                                 # M
n_updates  = int(1E4)                                 # n
initial_lr = 0.1                                      # alpha_0
final_lr   = .0001                                    # alpha_M
step_size  = max_steps // n_updates                   # n
lr_gamma   = (final_lr / initial_lr)**(1 / n_updates) # gamma

optimizer  = optim.SGD(model.parameters(),
                       lr=initial_lr)

# using our step size n
scheduler  = StepLR(optimizer, step_size=step_size, gamma=lr_gamma)

# or using a list of milestone steps
scheduler  = MultiStepLR(optimizer,
                         milestones=list(range(step_size, max_steps, step_size)),
                         gamma=lr_gamma)

# or using the "close to 1" unstable gamma
unstable_gamma = (final_lr / initial_lr)**(1 / max_steps)
scheduler  = ExponentialLR(optimizer, gamma=unstable_gamma)
```

## Closing remarks

In this post, I have shown you how I have solved the problem of bounding the learning rate of a neural network between a maximum and minimum value, by analitically finding the correct multiplicative decay factor for a given number of updates that will be applied to the learning rate until the end of training.

I hope this post helps other people. Bye.
