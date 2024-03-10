---
layout: post
title: Free online resources to study reinforcement learning and deep RL
categories: awesome-list reinforcement-learning deep-reinforcement-learning self-study
---

Over the last few years and months, I gathered some material that I used to study reinforcement learning and deep reinforcement learning, mostly by myself, as part of my PhD formation.

I hope these links can help others as much as they've helped me. The list also serves the purpose of remembering me that this material exists, since memory loss is a thing during a PhD.

<!-- TOC -->

- [Books and surveys](#books-and-surveys)
- [Courses](#courses)
- [Miscellaneous online material](#miscellaneous-online-material)
- [Algorithm implementations](#algorithm-implementations)

<!-- /TOC -->

## Books and surveys

The current textbook for the area of reinforcement learning in general is

> R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. Cambridge, Mass: The MIT Press, 2018.

It is freely and legally available [here](http://www.incompleteideas.net/book/the-book.html) and you can find my name in [the errata](http://www.incompleteideas.net/book/errata.html).

A textual reference for the overall area of deep RL is hard to pinpoint, but personally, I recommend the following survey:

> Y. Li, “Deep Reinforcement Learning, ” arXiv:1810.06339 [cs, stat], Oct. 2018, Accessed: May 08, 2020. [Online]. Available: http://arxiv.org/abs/1810.06339.

## Courses

* [Reinforcement Learning Specialization on Coursera](https://www.coursera.org/specializations/reinforcement-learning): I believe that, as of now, this is the most educational and informative resource available online to learn the fundamentals of RL from scratch. It is ministered by Martha and David White from UAlberta with special appearances by many prominent researchers.

  While the specialization follows Sutton's book, it also adds its own value by giving additional step-by-step examples and providing additional definitions or explanations of the definitions from the book. You also learn a lot from the quizzes and practical tests, which are done in Jupyter Notebooks. I definitely recommend this as a starting point for anyone who wants to dig deep into RL, as the specialization focuses on the foundations of the area.
  
  A disclaimer: the course is not actually free. You can watch all the lectures and work on all non-graded exercises, but all the graded ones (which are needed to get the certificate) are locked. You also need to pay for each one of the 4 certificates in order to finish the specialization. One option to circumvent this is to see if you are eligible for [Coursera for university and students](https://www.coursera.org/for-university-and-college-students) (must have a `.edu` email) or for Financial Aid [[1]](https://learner.coursera.help/hc/en-us/articles/209819033-Apply-for-Financial-Aid-or-a-Scholarship) [[2]](https://blog.coursera.org/courseras-financial-aid-what-it-is-and-who-is/). In the case of Financial Aid, you must ask it for each course in the specialization.

* [Introduction to reinforcement learning by David Silver](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ): Before the RL specialization, I believe this playlist was the go-to reference to study RL. David Silver is also a contemporary reference in the area and his lectures also use Sutton's book as reference. My only grudge with the course is that it assumes considerable previous knowledge in many areas, at times sounding more like a graduate course e.g, by encouraging students to prove statements while skipping $k$-armed bandits, for example. [Slides available](https://www.davidsilver.uk/teaching/).

* [Deep RL Bootcamp](https://www.youtube.com/playlist?list=PLsuq9stvuZe4pc8T4NutncqaYzpmwBJtg): lectures with some of the scientists behind the groundbreaking DRL algorithms created in the last years. Good explanations of DQN and policy gradient methods from their creators.  [Slides available](https://sites.google.com/view/deep-rl-bootcamp/lectures).

* [CS 285 at UC Berkeley](http://rail.eecs.berkeley.edu/deeprlcourse/): a full DRL course. This looks like the real deal. Unfortunately, it was not the material I used to learn what I know, but people highly recommend it, so I'll keep it in the list.

* [Practical_RL](https://github.com/yandexdataschool/Practical_RL): *An open course on reinforcement learning in the wild [...] maintained to be friendly to online students (both english and russian).* This one is mostly maintained on GitHub. Unfortunately, lectures are in Russian, but there are slides and links to more material.

* [Reinforcement Learning Course \| DeepMind & UCL](https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb): A YouTube playlist with 10 lectures averaging 1h40min each lecture

## Miscellaneous online material

* [Spinning Up in Deep RL](https://spinningup.openai.com/): *an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning.* The website has great summaries about the policy gradient algorithms created by OpenAI, as well as references to the original papers.

* Arthur Juliani's [Simple Reinforcement learning series][ajsrl].

## Algorithm implementations

* [My personal repository](https://github.com/douglasrizzo/machine_learning/blob/master/include/GridWorld.hpp) where I have implemented policy iteration, value iteration, Monte Carlo ES, Sarsa and Q-Learning in C++ and applied it in a grid world.

* MorvanZhou's simple and educational [Python implementations of classical RL algorithms](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/). I used them on a personal project and compiled them into a single, [documented](https://douglasrizzo.com.br/sc2qsr/rl.html) Python module available [here](https://github.com/douglasrizzo/sc2qsr/blob/master/sc2qsr/rl.py) under the same license.

* Implementations of model-free and policy gradient methods in PyTorch 0.4, in a very instructive way:

    * [higgsfield/RL-Adventure](https://github.com/higgsfield/RL-Adventure)
    * [higgsfield/RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2)

* [OpenAI baselines](https://github.com/openai/baselines): Baseline implementations of most famous DRL algorithms by OpenAI (since the ones they found online were riddled with bugs).

* [Stable baselines](https://github.com/hill-a/stable-baselines): a repository that is maintained more frequently than OpenAI's. Their code is PEP8 compliant and actually documented.

* Arthur Juliani also has complete implementations over at his GitHub, as well as commented ones in his [aforementioned series][ajsrl].

* Denny Britz has [a very famous repository](https://github.com/dennybritz/reinforcement-learning) of reinforcement algorithms over at GitHub.

* [PyTorch official DQN tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html): this is the intermediate step in the PyTorch tutorials so, if you fancy learning some PyTorch, I believe this is the most straightforward way to implement and debug your first DRL algorithm.

[ajsrl]: https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149
