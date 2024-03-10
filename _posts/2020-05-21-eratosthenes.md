---
layout: post
title: Sieve of Eratosthenes
categories: colab python programming mathematics
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1F8L9vW8PCrZCpzaKnIgcMM2MBhHJRXuU?usp=sharing)

In this notebook, I implement a few versions of the [sieve of Eratosthenes](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes) for finding prime numbers.

Starting from $n=2$ and given the set $P$ of prime numbers up until $n-1$, we check if $n$ is divisible by any $p \in P$. If not, $P \leftarrow P + \\\{n\\\}$.

```python
from math import sqrt

from math import sqrt, pi
import numpy as np

cached_primes = [2]
evaluated = 2
```

The first version of the sieve works just like explained at the top.

```python
def eratosthenes(n):
    primes = []
    for i in range(2, n):
        for p in primes:
            if i % p == 0:
                break
        else:
            primes.append(i)

    return primes
```

The second version starts with the number 2 (the only even prime) already in the set of primes and only checks the primality of odd numbers. It should be twice as fast.

```python
def eratosthenes_no_evens(n):
    primes = [2]
    for i in range(3, n, 2):
        for p in primes:
            if i % p == 0:
                break
        else:
            primes.append(i)

    return primes
```

The last version uses all the gimmicks from the previous ones, but also caches the primes found in previous runs and reuses them in subsequent calls of the function.

```python
def eratosthenes_cached(n):
   global evaluated, cached_primes

   if evaluated < n:
      start = evaluated if evaluated % 2 != 0 else evaluated + 1
      for i in range(start, n + 1, 2):
         for p in cached_primes:
            if i % p == 0:
              break
         else:
            cached_primes.append(i)

   evaluated = n

   for i in range(len(cached_primes)):
     if cached_primes[i]>=n:
       return cached_primes[:i]

   return cached_primes
```

Check if the output of all functions is equal for an arbitrary number of input integers.

```python
from numpy.random import randint

for v in randint(5000, size=100):
  a = eratosthenes(v)
  b = eratosthenes_no_evens(v)
  c = eratosthenes_cached(v)

  if not a == b == c:
    raise RuntimeError('not equal')
```

Let's check the performance of the three functions.

```python
%timeit -r 5 -n 30 [eratosthenes(v) for v in randint(5000, size=100)]
%timeit -r 5 -n 30 [eratosthenes_no_evens(v) for v in randint(5000, size=100)]
%timeit -r 5 -n 30 [eratosthenes_cached(v) for v in randint(5000, size=100)]
```

    30 loops, best of 5: 430 ms per loop
    30 loops, best of 5: 404 ms per loop
    30 loops, best of 5: 200 ms per loop

We can see that, because the third function caches primes from previous runs, it is twice as fast than the other two.
