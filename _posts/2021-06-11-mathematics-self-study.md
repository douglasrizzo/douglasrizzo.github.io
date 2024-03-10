---
layout: post
title: Resources to self-study mathematics for machine learning
categories: mathematics self-study machine-learning
---

> Two plus two is four, minus one, that's three. Quick maths.
>
> [Michael Dapaah](https://youtu.be/3M_5oYU-IsU?t=63)

<!-- TOC -->

- [Introduction](#introduction)
- [Study areas](#study-areas)
  - [Algebra and pre-calculus](#algebra-and-pre-calculus)
  - [Calculus](#calculus)
  - [Linear Algebra](#linear-algebra)
    - [Advanced Linear Algebra](#advanced-linear-algebra)
- [Software and apps](#software-and-apps)
- [Online communities](#online-communities)
- [My failures](#my-failures)
- [Final thoughts](#final-thoughts)
- [More resources and moving forward](#more-resources-and-moving-forward)

<!-- /TOC -->

## Introduction

When I started my PhD, I didn't know at first if my research would take me in a path whose foundations I was comfortable with, such as algorithms, logic and other discrete areas, or if fate would take me towards the more mathematical path. As I started to get really interested in the applications of neural networks for reinforcement learning, I realized little by little that I would not only need to remember all the mathematics I had learned before in my life, but also learn and get comfortable with some new stuff.

In this article, I go over:

- what topics I've been studying;
- in which order I study them;
- what material worked for me;
- where to find them online (the free ones, at least);
- which software, online services and apps I use;
- where to find online communities to post questions and answers and to study together in real time;
- where I failed in my studies;
- some final tips;
- links to more resources.

## Study areas

### Algebra and pre-calculus

I recommend the [free algebra book by James Brennan](https://jamesbrennan.org/algebra/) and the [free precalculus book by Stitz and Zaeger](https://www.stitz-zeager.com/), which I found out about in the very helpful [/r/learnmath](https://reddit.com/r/learnmath) subreddit. My recommendation would be to evaluate which topics you remember and just skim over them, while giving yourself more time over the stuff you don't remember or have never learned.

Personally, I made a lot of progress by just following both books until I felt I was comfortable with my knowledge. I tried not to skip anything, but I also felt like I could work fast as I had already learned most of this stuff before.

### Calculus

To study calculus, I used the [books by Stewart](https://stewartcalculus.com/media/16_home.php), 8th edition. I knew I was prepared to start calculus when I had studied enough algebra to fulfill the prerequisites at the start of the book. I also worked through the two review lists available in Stewart's website [[1]](https://stewartcalculus.com/data/CALCULUS_8E/upfiles/6e_reviewofalgebra.pdf) [[2]](https://stewartcalculus.com/data/CALCULUS_8E/upfiles/6e_reviewofanalgeom.pdf). Those lists helped me to prioritize what I needed to focus on my aforementioned precalculus quest.

### Linear Algebra

I decided to start studying linear algebra alongside calculus. I had an easier time with linear algebra as I already had some familiarity with matrix operations, not only because I was already working with neural networks for quite some time, but I also implemented [my own matrix class in C++](douglasrizzo.github.io/matrix) at one point.

I started with a [Brazilian undergrad text book from 1986](https://www.google.com.br/books/edition/Algebra_linear/M8CNGwAACAAJ?hl=en), but also searched many concepts on YouTube, since I did not get the intuition behind some of them. I ended up ditching the book after getting acquainted with [Larson's book](https://www.amazon.com/gp/product/B019EB9S4O/), which presented the material in a much nore contemporary fashion.

#### Advanced Linear Algebra

I haven't gotten here yet, but after learning all the basic stuff about matrices, vector spaces, linear transforms and eigenthingies, I was recommended the book [Linear Algebra Done Right](https://www.amazon.com/Linear-Algebra-Right-Sheldon-Axler/dp/3319110799/), by Sheldon Axler to move on to more advanced stuff.

## Software and apps

- **Solvers:** These are services that do things like simplifying expressions, solving for $x$, finding derivatives, minima and maxima values of functions and so on. You name it, they do it. Just be careful that, sometimes, the answers are either more convoluted than they need to be, or simpler than you'd want them to be.
  - [Wolfram Alpha](https://www.wolframalpha.com/) is in this category. Instead of paying for a subscription to the website, a cheaper solution is to buy [the smartphone app](https://play.google.com/store/apps/details?id=com.wolfram.android.alpha), which is a one-time purchase that gives you the coveted "step-by-step solution" functionality.
  - A free alternative which is equally impressive is [Microsoft Math Solver](https://math.microsoft.com/), which also has a more user-friendly [mobile app](https://play.google.com/store/apps/details?id=com.microsoft.math).
- **Graphical calculators:** you're gonna need these ones often, to visualize functions, derivatives, systems of equations etc. I recommend the Desmos Graphical Calculator, a very famous graphical calculator, which is freely available [online](https://www.desmos.com/calculator) and as [an app](https://play.google.com/store/apps/details?id=com.desmos.calculator). It only works in two dimensions, I don't really know if there is anything fancier than that.
- **Actual calculators:** On Linux, I use [SpeedCrunch](http://speedcrunch.org/) for [quick maths](https://youtu.be/3M_5oYU-IsU?t=63) or [Qalculate!](https://qalculate.github.io/) for more symbolic stuff (with fractions, units of measurement etc). On mobile, I'd recommend [Desmos Scientific Calculator](https://play.google.com/store/apps/details?id=com.desmos.scientific). I know some people like to use "calculator simulators" but I haven't had the need for that.
- **Linear algebra special:** instead on depending on MATLAB or some other obnoxious monolithic proprietary software, you can represent systems of linear equations as matrices in Python using [NumPy](https://numpy.org/) and manipulate/solve them using [`numpy.linalg`](https://numpy.org/devdocs/reference/routines.linalg.html). You can load `.mat` files into Python variables using [SciPy's `loadmat` function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html).

## Online communities

It came a point in which I started having the kind of questions that I couldn't quite google my way to the answer. When I started to look for a place to post my questions, I found the extremely active and welcoming [/r/learnmath](https://reddit.com/r/learnmath) subreddit. The people there are really nice and welcoming of noobs and they are a lot less strict than other sites, such as the Math Stack Exchange.

Another place I suggest people to look are Discord servers, where people gather to talk about homework. An example is [Homework Help](https://discord.gg/USVc7XX). There are also study servers on Discord in which people can keep you company in mute video calls. I highly recommend the [Study Lions](https://discord.gg/s9tEhQPw8A), but there are others.

Overall, my advice with regards to communities is to both **seek and provide help**. Seeing people ask questions to which I knew the answer to made me realize how much I had actually learned in my journey, which really motivated me. Also, I could help a random stranger online, which is nice.

## My failures

Humble time. Since I didn't know where to start, I decided to enroll in online university courses and let the lectures guide me. I had had a great experience with Coursera and the machine learning course by Andrew Ng, so I thought I'd have an equally positive experience with Coursera's math courses[^calc-courses].

However, unlike the previous courses I took, I couldn't watch the lectures or complete the assignments before or after the correct dates, which really frustrated me.

Another problem I faced was that I just couldn't follow the professors in some of the lectures. Sometimes, the exercises after a lesson seemed to have nothing to do with the lesson itself. Since I had no other material to follow, I felt stuck in some lessons and couldn't meet the deadlines.

## Final thoughts

- **Don't force it:** Studying mathematics ended up becoming a hobby of mine, one that I could spend an entire day doing, after I had started. There were days that I would eat up over 20 pages of a book, a pretty decent number if you ask me, since I'd be able to end a 400 page book in 20 days (not that I did that). My main problem was, and still is, consistency. I can't really do the whole "study everyday for 30 minutes" thing. I usually studied for the whole day, twice a week.

  I'll admit I did not study any of the books from cover to cover and I am still working through them, but I never felt hindered or delayed by the method I chose to study mathematics. My only limitations are time, since I have other responsibilities, and motivation.

- **Find the format that is best for you:** since I did very well in online AI courses [[1]](https://www.coursera.org/verify/BJWU9J3F9J4X) [[2]](http://coursera.org/verify/specialization/7YLWXWJ5GUKJ), I thought the same would be true for online math courses. However, when it came to mathematics, I felt like books were a much better medium, as they present the material in both a rigorous and linear fashion, delving into the theory and definitions and following up with lots of examples and exercises which could mostly be completed with the material presented right before. Text books are also good when I just wanted to review some old algebra, in which case I am able to quicly skip to whatever concept I am interested in, while videos would be much harder to sift through.

- **YouTube is your friend:** I tend to forget basic stuff very often, such as how to complete the square, trigonometric identities ([there are lots of them](https://en.wikipedia.org/wiki/List_of_trigonometric_identities)) or the binomial theorem. In these cases, looking up a YouTube video is much faster than searching for what you need in books. Math videos usually go directly to the point and you don't waste any time.

## More resources and moving forward

This video my the Math Sorcerer lists books from every area of Mathematics, starting from basic logic and algebra and going all the way up to the crazy stuff.

<iframe width="560" height="315" src="https://www.youtube.com/embed/pTnEG_WGd2Q" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

This video by Aleph 0 has some book recommendations for undergraduate level courses in pure mathematics, with accompanying video lectures on YouTube.

<iframe width="560" height="315" src="https://www.youtube.com/embed/fo-alw2q-BU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---

[^calc-courses]: I enrolled in the UC Irvine precalculus course (which is not available on Coursera anymore, but can be found [here](https://www.math.uci.edu/node/22608)) and later, in the University of Ohio calculus course (which also is not available anymore, but a few pointers to where the material is currently available are [[link]](https://mooculus.osu.edu/lectures) [[link]](https://www.youtube.com/user/kisonecat/playlists) [[link]](https://www.coursera.org/instructor/jimfowler)).
