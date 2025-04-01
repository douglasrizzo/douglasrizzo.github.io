---
layout: post
title: Building a content-based movie recommendation system using the MovieLens dataset
category: recommendations-systems
tags: python open-source recommendation-systems content-based-systems kaggle
toc:
  sidebar: left
---

## Introduction

In this project, we will build a movie recommendation system using content-based filtering. We will use the [MovieLens dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset), provided in Kaggle, which contains movie and user data, to recommend movies to an user based on the genres of movies he has liked the most.

### Resources

- GitHub repository containing this project and a list of dependencies [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/douglasrizzo/recsys-movies/blob/master/src/content-based-filtering.ipynb)
- Open this page in Google Colab [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/douglasrizzo/recsys-movies/blob/master/src/content-based-filtering.ipynb)
- [Read the contents in my website](https://douglasrizzo.com.br/blog/2025/03/content-based-filtering/)

The video below contains an overview of the method we will implement. However, unlike the example above, they use a method which gives more weight to genres a user has rated more movies in. We will fix that mistake and show the differences in both methods.

<iframe width="560" height="315" src="https://www.youtube.com/embed/YMZmLx-AUvY?si=jz2I8XVknb9X2Y3a" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Content-based filtering overview

The idea of content-based filtering is simple.

1. represent items with a set of features
2. discover a user's preferences to each of those features, based on items they have already rated/consumed/liked that contain the features
3. infer the user's preference to unseen items based on those item features.

Unlike more sophisticated recommendation systems, such as collaborative filtering, content-based filtering only uses data provided by a single user to model their preferences. Thus it is much simpler to implement.

### A concrete example

In our project, we will build a movie recommendation system, so let's build an example using this theme. I will number the steps below and you can look for references to these tables in the Python implementation that comes right after.

1. We begin with a list of ratings a user has given to the movies they have watched.

   | Movie ID | Movie Title  | User Rating |
   | -------- | ------------ | ----------- |
   | 1        | Inception    | 5.0         |
   | 2        | The Matrix   | 4.5         |
   | 3        | Interstellar | 4.0         |

2. Each of the movies belongs to a list of genres.

   | Movie ID | Movie Title  | Genres           |
   | -------- | ------------ | ---------------- |
   | 1        | Inception    | Sci-Fi, Thriller |
   | 2        | The Matrix   | Sci-Fi, Action   |
   | 3        | Interstellar | Sci-Fi, Drama    |

3. If we average the ratings the user has given for each movie _by genre_, we get the user's ratings for each genre.

   - Sci-Fi appears in all three movies, so its average rating is:
     $$ \frac{(5.0 + 4.5 + 4.0)}{3} = 4.5 $$
   - Thriller appears only in _Inception_, so its average rating is: **5.0**
   - Action appears only in _The Matrix_, so its average rating is: **4.5**
   - Drama appears only in _Interstellar_, so its average rating is: **4.0**

   This gives us the ratings by genre for that particular user.

   | Genre    | Average Rating |
   | -------- | -------------- |
   | Sci-Fi   | 4.5            |
   | Thriller | 5.0            |
   | Action   | 4.5            |
   | Drama    | 4.0            |

   In technical jargon, this is called the **_user profile_** and the component in your system that generates this is the **_user profiler_**.

4. Now, let's consider a new list of movies this user has not yet watched.

   | Movie ID | Movie Title        | Genres                   |
   | -------- | ------------------ | ------------------------ |
   | 4        | Blade Runner 2049  | Sci-Fi, Thriller         |
   | 5        | Mad Max: Fury Road | Action, Sci-Fi           |
   | 6        | The Social Network | Drama                    |
   | 7        | The Crow           | Action, Horror           |
   | 8        | La La Land         | Musical, Romantic Comedy |

When we select and preprocess item features, keeping them ready to match against user profiles, we performing **_content analysis_**. In our example, we use movie genres as their features, transforming them into what are called **_tags_**, and we are not considering other information such as the movie description, its release date or its director. Some of these could also be transformed into tags, while others need to be processed using NLP techniques.

5. To estimate the current user's ratings for these unwatched movies, we take the **average of the user's ratings for the movie's genres**:

   - **Blade Runner 2049 (Sci-Fi, Thriller)**  
     $$ \frac{(4.5 + 5.0)}{2} = 4.75 $$
   - **Mad Max: Fury Road (Action, Sci-Fi)**  
     $$ \frac{(4.5 + 4.5)}{2} = 4.5 $$
   - **The Social Network (Drama)**
     - Drama’s average rating is **4.0**, so the estimated rating is **4.0**.
   - **The Crow (Action, Horror)**
     - Because the user has not ranked any horror movies yet, we can average over the ratings of the genres we do have. In this case, only Action, whose estimated rating is **4.5**.
   - **La La Land (Musical, Romantic Comedy)**
     - The user has not rated any romantic comedies or musicals yet, so we are unable to estimate a rating for _La La Land_ and thus we are unable to recommend it to this user.

   | Movie ID | Movie Title        | Inferred Rating |
   | -------- | ------------------ | --------------- |
   | 4        | Blade Runner 2049  | 4.75            |
   | 5        | Mad Max: Fury Road | 4.5             |
   | 6        | The Social Network | 4.0             |
   | 7        | The Crow           | 4.5             |
   | 8        | La La Land         | ---             |

Now we know that, if we were to recommend new movies for that user to watch, we would recommend _"Blade Runner 2049"_ first, then _"Mad Max: Fury Road"_ or _"The Crow"_ and _"The Social Network"_ last.

The process of selecting ranked items for a given users is called **_content retrieval_**.

Maybe we would like to prioritize _"Mad Max: Fury Road"_ over _"The Crow"_ since we have more user preference information to estimate the rating of the first movie, but that is not strictly necessary and we can consider them tied until the user rates more movies.

## Downloading the dataset

As you have seen above, we need a dataset that contains:

1. movies with their genres
2. user ratings for movies

Fortunately, the MovieLens dataset has both of these pieces of data. We will download it using the [`kagglehub`](https://github.com/Kaggle/kagglehub) package.

```python
import pathlib as pl

import kagglehub
import pandas as pd

(mvls_links_path, mvls_genometags_path, mvls_movies_path, mvls_genomescores_path, mvls_tags_path, mvls_ratings_path) = (
  list(pl.Path(kagglehub.dataset_download("grouplens/movielens-20m-dataset")).iterdir())
)
print(
  mvls_links_path, mvls_genometags_path, mvls_movies_path, mvls_genomescores_path, mvls_tags_path, mvls_ratings_path
)
```

    /home/dodo/.cache/kagglehub/datasets/grouplens/movielens-20m-dataset/versions/1/link.csv /home/dodo/.cache/kagglehub/datasets/grouplens/movielens-20m-dataset/versions/1/genome_tags.csv /home/dodo/.cache/kagglehub/datasets/grouplens/movielens-20m-dataset/versions/1/movie.csv /home/dodo/.cache/kagglehub/datasets/grouplens/movielens-20m-dataset/versions/1/genome_scores.csv /home/dodo/.cache/kagglehub/datasets/grouplens/movielens-20m-dataset/versions/1/tag.csv /home/dodo/.cache/kagglehub/datasets/grouplens/movielens-20m-dataset/versions/1/rating.csv

MovieLens contains a collection of CSV files. You can see below the movie-related data, which contains the movie and year of release in one column, and all of its genres on another. This equivalent to table 2 in the example.

```python
movies = pd.read_csv(mvls_movies_path).set_index("movieId").sort_index()
movies.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table class="post-body">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>genres</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>

The first thing we will do is represent movie genres by one-hot encoded columns. We will also keep a record of the names of these columns for later.

```python
# get a single list of all unique genres for all movies
genres = list(set(movies["genres"].str.split("|").sum()))
genres = sorted(genres)
movies = movies.join(movies["genres"].str.get_dummies(sep="|")).drop("genres", axis=1)
print(f"Unique genres: {genres}")
print(f"Number of genres: {len(genres)}")
movies.head()
```

    Unique genres: ['(no genres listed)', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    Number of genres: 20

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table class="post-body">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>(no genres listed)</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>...</th>
      <th>Film-Noir</th>
      <th>Horror</th>
      <th>IMAX</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Toy Story (1995)</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jumanji (1995)</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Grumpier Old Men (1995)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Waiting to Exhale (1995)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Father of the Bride Part II (1995)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>

Regarding user ratings to movies, MovieLens provides us with 20 million ratings. In the cell below, you can see a few basic characteristics of user ratings. Most importantly, they range from 0.5 to 5 in intervals of 0.5

```python
ratings = pd.read_csv(mvls_ratings_path)
ratings["rating"].value_counts()
```

    rating
    4.0    5561926
    3.0    4291193
    5.0    2898660
    3.5    2200156
    4.5    1534824
    2.0    1430997
    2.5     883398
    1.0     680732
    1.5     279252
    0.5     239125
    Name: count, dtype: int64

A user rating to a movie ties a user ID to a movie ID and contains the 0.5--5 rating the user has given the movie. In this project, we will work only with the user whose ID is 1. Let's peek at some movies he has seen and their ratings.

We can see that user 1 has watched and rated 175 movies. That is a lot of movies. Also, the minimum rating user 1 has given to a movie is 3, far from the 0.5 it is allowed to give.

```python
user_ratings = ratings[ratings["userId"] == 1]
ax = user_ratings["rating"].hist(bins=10, range=(0, 5))
ax.bar_label(ax.containers[0]);
```

![png](/assets/img/content-based-filtering_9_0.png)

This is what the user-movie rating data actually looks like. This is equivalent to table 1 in the example.

```python
user_ratings.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table class="post-body">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3.5</td>
      <td>2005-04-02 23:53:47</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>29</td>
      <td>3.5</td>
      <td>2005-04-02 23:31:16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>32</td>
      <td>3.5</td>
      <td>2005-04-02 23:33:39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>3.5</td>
      <td>2005-04-02 23:32:07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>3.5</td>
      <td>2005-04-02 23:29:40</td>
    </tr>
  </tbody>
</table>
</div>

## Computing user-genre preferences

Our goal now will be to generate the user's features, which will be derived from the genres of the movies they have watched and their ratings.

To make things easier for us, we will join user 1's movie ratings with the genres of those movies.

```python
user_ratings = user_ratings[["userId", "movieId", "rating"]].merge(movies, on="movieId")
user_ratings.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table class="post-body">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>title</th>
      <th>(no genres listed)</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>...</th>
      <th>Film-Noir</th>
      <th>Horror</th>
      <th>IMAX</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3.5</td>
      <td>Jumanji (1995)</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>29</td>
      <td>3.5</td>
      <td>City of Lost Children, The (Cité des enfants p...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>32</td>
      <td>3.5</td>
      <td>Twelve Monkeys (a.k.a. 12 Monkeys) (1995)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>3.5</td>
      <td>Seven (a.k.a. Se7en) (1995)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>3.5</td>
      <td>Usual Suspects, The (1995)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>

Now, we will propagate the user's ratings to the genres of each movie. by multiplying the ratings user 1 has given to a movie with the genre columns of each movie. You will see that the one-hot encoding of that movie will become the actual ratings user 1 has given to the movie.

```python
# multiply the genre columns by the rating
user_ratings[genres] = user_ratings[genres].multiply(user_ratings["rating"], axis="index")
user_ratings.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table class="post-body">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>title</th>
      <th>(no genres listed)</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>...</th>
      <th>Film-Noir</th>
      <th>Horror</th>
      <th>IMAX</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3.5</td>
      <td>Jumanji (1995)</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>29</td>
      <td>3.5</td>
      <td>City of Lost Children, The (Cité des enfants p...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>32</td>
      <td>3.5</td>
      <td>Twelve Monkeys (a.k.a. 12 Monkeys) (1995)</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>3.5</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>3.5</td>
      <td>Seven (a.k.a. Se7en) (1995)</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>3.5</td>
      <td>Usual Suspects, The (1995)</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>

Now, if we sum all the ratings user 1 has given to the movies they have watched _by genre_, we get a proxy value that denotes how much user 1 enjoys each of the genres.

```python
user_genre_preferences = pd.DataFrame(
  data={"Movies": (user_ratings[genres] != 0).sum(), "Sum of ratings": user_ratings[genres].sum()}
)
user_genre_preferences.sort_values("Sum of ratings", ascending=False)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table class="post-body">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movies</th>
      <th>Sum of ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adventure</th>
      <td>73</td>
      <td>276.5</td>
    </tr>
    <tr>
      <th>Fantasy</th>
      <td>69</td>
      <td>261.5</td>
    </tr>
    <tr>
      <th>Action</th>
      <td>66</td>
      <td>246.0</td>
    </tr>
    <tr>
      <th>Horror</th>
      <td>45</td>
      <td>168.5</td>
    </tr>
    <tr>
      <th>Drama</th>
      <td>43</td>
      <td>162.0</td>
    </tr>
    <tr>
      <th>Thriller</th>
      <td>42</td>
      <td>158.0</td>
    </tr>
    <tr>
      <th>Comedy</th>
      <td>41</td>
      <td>153.0</td>
    </tr>
    <tr>
      <th>Sci-Fi</th>
      <td>40</td>
      <td>148.5</td>
    </tr>
    <tr>
      <th>Crime</th>
      <td>21</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>Children</th>
      <td>19</td>
      <td>68.5</td>
    </tr>
    <tr>
      <th>Mystery</th>
      <td>18</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>Romance</th>
      <td>11</td>
      <td>43.5</td>
    </tr>
    <tr>
      <th>Animation</th>
      <td>10</td>
      <td>36.5</td>
    </tr>
    <tr>
      <th>War</th>
      <td>9</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>Western</th>
      <td>4</td>
      <td>13.5</td>
    </tr>
    <tr>
      <th>Musical</th>
      <td>3</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>IMAX</th>
      <td>2</td>
      <td>8.5</td>
    </tr>
    <tr>
      <th>(no genres listed)</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Documentary</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Film-Noir</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

Because the magnitudes of these sums of ratings can get out of hand, we can divide them by the sum of all ratings and get neat values that all add up to 1.

```python
user_genre_preferences["Normalized sum of ratings"] = (
  user_genre_preferences["Sum of ratings"] / user_genre_preferences["Sum of ratings"].max()
)
user_genre_preferences.sort_values("Normalized sum of ratings", ascending=False)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table class="post-body">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movies</th>
      <th>Sum of ratings</th>
      <th>Normalized sum of ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adventure</th>
      <td>73</td>
      <td>276.5</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Fantasy</th>
      <td>69</td>
      <td>261.5</td>
      <td>0.945750</td>
    </tr>
    <tr>
      <th>Action</th>
      <td>66</td>
      <td>246.0</td>
      <td>0.889693</td>
    </tr>
    <tr>
      <th>Horror</th>
      <td>45</td>
      <td>168.5</td>
      <td>0.609403</td>
    </tr>
    <tr>
      <th>Drama</th>
      <td>43</td>
      <td>162.0</td>
      <td>0.585895</td>
    </tr>
    <tr>
      <th>Thriller</th>
      <td>42</td>
      <td>158.0</td>
      <td>0.571429</td>
    </tr>
    <tr>
      <th>Comedy</th>
      <td>41</td>
      <td>153.0</td>
      <td>0.553345</td>
    </tr>
    <tr>
      <th>Sci-Fi</th>
      <td>40</td>
      <td>148.5</td>
      <td>0.537071</td>
    </tr>
    <tr>
      <th>Crime</th>
      <td>21</td>
      <td>80.0</td>
      <td>0.289331</td>
    </tr>
    <tr>
      <th>Children</th>
      <td>19</td>
      <td>68.5</td>
      <td>0.247740</td>
    </tr>
    <tr>
      <th>Mystery</th>
      <td>18</td>
      <td>65.0</td>
      <td>0.235081</td>
    </tr>
    <tr>
      <th>Romance</th>
      <td>11</td>
      <td>43.5</td>
      <td>0.157324</td>
    </tr>
    <tr>
      <th>Animation</th>
      <td>10</td>
      <td>36.5</td>
      <td>0.132007</td>
    </tr>
    <tr>
      <th>War</th>
      <td>9</td>
      <td>33.0</td>
      <td>0.119349</td>
    </tr>
    <tr>
      <th>Western</th>
      <td>4</td>
      <td>13.5</td>
      <td>0.048825</td>
    </tr>
    <tr>
      <th>Musical</th>
      <td>3</td>
      <td>11.0</td>
      <td>0.039783</td>
    </tr>
    <tr>
      <th>IMAX</th>
      <td>2</td>
      <td>8.5</td>
      <td>0.030741</td>
    </tr>
    <tr>
      <th>(no genres listed)</th>
      <td>0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Documentary</th>
      <td>0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Film-Noir</th>
      <td>0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>

These values are interesting and they may be used as an indication of how much a user prefers certain movie genres. This is the way that is taught in the video at the beginning of the page.

However, because this method just sums all ratings the user has given to a collection of movies, it favors those genres in which the user has given more ratings, regardless of how low the ratings are.

For example, if a user watches 9 action movies and gives a rating of 1 to all of them, the Action genre will have a weight of 9 in our preference vector:

```
1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 9
```

If that same user watches 2 musicals (I know, who likes musicals) but gives a rating of 4 to both of them, the Musical genre will have a weight of 8 in our fetaure vector, which is less than the weight of the Action genre.

```
4 + 4 = 8
```

But we can clearly see that this user hates action movies but loves musicals!

Our hypothesis is corroborated in the chart below, in which our sum of ratings for a genre grows as the user watches more movies.

```python
user_genre_preferences[["Movies", "Sum of ratings"]].sort_values("Movies", ascending=False).plot.barh()
```

    <Axes: >

![png](/assets/img/content-based-filtering_21_1.png)

By looking at the correlations between number of movies watched, sum of ratings and the normalized sum of ratings, we can see a pretty hgh positive correlation between all of them.

```python
user_genre_preferences.corr()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table class="post-body">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movies</th>
      <th>Sum of ratings</th>
      <th>Normalized sum of ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Movies</th>
      <td>1.000000</td>
      <td>0.999871</td>
      <td>0.999871</td>
    </tr>
    <tr>
      <th>Sum of ratings</th>
      <td>0.999871</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Normalized sum of ratings</th>
      <td>0.999871</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

One way we can mitigate this is by taking the _average_ of the ratings a user gives to the movies of a particular genre as their preference for that genre. The table below is the one presented in step 3 of the example.

```python
user_genre_preferences["Normalized by movies in genre"] = (
  user_genre_preferences["Sum of ratings"] / user_genre_preferences["Movies"]
).fillna(0)
user_genre_preferences = user_genre_preferences.sort_values("Normalized by movies in genre", ascending=False)
user_genre_preferences
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table class="post-body">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movies</th>
      <th>Sum of ratings</th>
      <th>Normalized sum of ratings</th>
      <th>Normalized by movies in genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IMAX</th>
      <td>2</td>
      <td>8.5</td>
      <td>0.030741</td>
      <td>4.250000</td>
    </tr>
    <tr>
      <th>Romance</th>
      <td>11</td>
      <td>43.5</td>
      <td>0.157324</td>
      <td>3.954545</td>
    </tr>
    <tr>
      <th>Crime</th>
      <td>21</td>
      <td>80.0</td>
      <td>0.289331</td>
      <td>3.809524</td>
    </tr>
    <tr>
      <th>Fantasy</th>
      <td>69</td>
      <td>261.5</td>
      <td>0.945750</td>
      <td>3.789855</td>
    </tr>
    <tr>
      <th>Adventure</th>
      <td>73</td>
      <td>276.5</td>
      <td>1.000000</td>
      <td>3.787671</td>
    </tr>
    <tr>
      <th>Drama</th>
      <td>43</td>
      <td>162.0</td>
      <td>0.585895</td>
      <td>3.767442</td>
    </tr>
    <tr>
      <th>Thriller</th>
      <td>42</td>
      <td>158.0</td>
      <td>0.571429</td>
      <td>3.761905</td>
    </tr>
    <tr>
      <th>Horror</th>
      <td>45</td>
      <td>168.5</td>
      <td>0.609403</td>
      <td>3.744444</td>
    </tr>
    <tr>
      <th>Comedy</th>
      <td>41</td>
      <td>153.0</td>
      <td>0.553345</td>
      <td>3.731707</td>
    </tr>
    <tr>
      <th>Action</th>
      <td>66</td>
      <td>246.0</td>
      <td>0.889693</td>
      <td>3.727273</td>
    </tr>
    <tr>
      <th>Sci-Fi</th>
      <td>40</td>
      <td>148.5</td>
      <td>0.537071</td>
      <td>3.712500</td>
    </tr>
    <tr>
      <th>War</th>
      <td>9</td>
      <td>33.0</td>
      <td>0.119349</td>
      <td>3.666667</td>
    </tr>
    <tr>
      <th>Musical</th>
      <td>3</td>
      <td>11.0</td>
      <td>0.039783</td>
      <td>3.666667</td>
    </tr>
    <tr>
      <th>Animation</th>
      <td>10</td>
      <td>36.5</td>
      <td>0.132007</td>
      <td>3.650000</td>
    </tr>
    <tr>
      <th>Mystery</th>
      <td>18</td>
      <td>65.0</td>
      <td>0.235081</td>
      <td>3.611111</td>
    </tr>
    <tr>
      <th>Children</th>
      <td>19</td>
      <td>68.5</td>
      <td>0.247740</td>
      <td>3.605263</td>
    </tr>
    <tr>
      <th>Western</th>
      <td>4</td>
      <td>13.5</td>
      <td>0.048825</td>
      <td>3.375000</td>
    </tr>
    <tr>
      <th>(no genres listed)</th>
      <td>0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Documentary</th>
      <td>0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Film-Noir</th>
      <td>0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>

If we now look at the correlation of the two feature sets implemented, we can see they are mostly uncorrelated, which means they are two completely different ways of expressing user preference.

```python
user_genre_preferences[["Normalized sum of ratings", "Normalized by movies in genre"]].corr()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table class="post-body">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Normalized sum of ratings</th>
      <th>Normalized by movies in genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Normalized sum of ratings</th>
      <td>1.000000</td>
      <td>0.451363</td>
    </tr>
    <tr>
      <th>Normalized by movies in genre</th>
      <td>0.451363</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

As you can see above, the second choice of features is very interesting. While before, we thought that the user enjoyed the Adventure, Fantasy Action and Horror genres, now we belive they enjoy IMAX movies, romances and crimes, even though they have watched less movies in that genre.

One thing to note is that not all genres have inferred ratings for the given user. In cases where the user has not rated any movie of a particular genre, that genre will not have a rating. In our case, since MovieLens ratings start at 0.5, we use 0 to denote the absence of ratings.

```python
genres_with_ratings = (user_genre_preferences["Normalized by movies in genre"] > 0).sum()
genres_without_ratings = len(user_genre_preferences) - genres_with_ratings

print(f"Total genres: {len(user_genre_preferences)}\nNumber of genres with ratings for user: {genres_with_ratings}")
```

    Total genres: 20
    Number of genres with ratings for user: 17

## Inferring preferences for unwatched movies

Now that we know how much the user prefers each genre, we will compute the ratings for movies that user has not watched yet.

First, let's get all the movies they have not watched from MovieLens. This would be the table from step 4 in the example.

```python
# get the movies that the user has not seen
unwatched_movies = movies.loc[~movies.index.isin(user_ratings["movieId"])].copy()
unwatched_movies
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table class="post-body">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>(no genres listed)</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>...</th>
      <th>Film-Noir</th>
      <th>Horror</th>
      <th>IMAX</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Toy Story (1995)</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Grumpier Old Men (1995)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Waiting to Exhale (1995)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Father of the Bride Part II (1995)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Heat (1995)</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>131254</th>
      <td>Kein Bund für's Leben (2007)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>131256</th>
      <td>Feuer, Eis &amp; Dosenbier (2002)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>131258</th>
      <td>The Pirates (2014)</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>131260</th>
      <td>Rentun Ruusu (2001)</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>131262</th>
      <td>Innocence (2014)</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>27103 rows × 21 columns</p>
</div>

As an intermediate step, we will to propagate the user's genre preferences to the preferences of these unwatched movies.

```python
movie_genre_ratings = unwatched_movies[genres].multiply(user_genre_preferences["Normalized by movies in genre"])
movie_genre_ratings
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table class="post-body">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>(no genres listed)</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Fantasy</th>
      <th>Film-Noir</th>
      <th>Horror</th>
      <th>IMAX</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>3.787671</td>
      <td>3.65</td>
      <td>3.605263</td>
      <td>3.731707</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>3.789855</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>3.731707</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.954545</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>3.731707</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>3.767442</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.954545</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>3.731707</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>3.727273</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.809524</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>3.761905</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>131254</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>3.731707</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>131256</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>3.731707</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>131258</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>3.787671</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>131260</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>131262</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>3.787671</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>3.789855</td>
      <td>0.0</td>
      <td>3.744444</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>27103 rows × 20 columns</p>
</div>

We then sum these preferences _by movie_ and divide by the number of genres each movie has, to get the user preferences for each unwatched movie.

If we sort the movies by these inferred preferences, we get a list in descending order of the movies user 1 has not seen but might enjoy, based on their genres and the ratings user 1 has given to the movies they have seen.

```python
denominator = movie_genre_ratings != 0
numerator = movie_genre_ratings.sum(axis=1)
content_based_ratings = numerator / denominator.sum(axis=1)
unwatched_movies["content_based_rating"] = content_based_ratings
unwatched_movies[["title", "content_based_rating"]].sort_values("content_based_rating", ascending=False)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table class="post-body">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>content_based_rating</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4858</th>
      <td>Hail Columbia! (1982)</td>
      <td>4.25</td>
    </tr>
    <tr>
      <th>4861</th>
      <td>Mission to Mir (1997)</td>
      <td>4.25</td>
    </tr>
    <tr>
      <th>4459</th>
      <td>Alaska: Spirit of the Wild (1997)</td>
      <td>4.25</td>
    </tr>
    <tr>
      <th>4460</th>
      <td>Encounter in the Third Dimension (1999)</td>
      <td>4.25</td>
    </tr>
    <tr>
      <th>4461</th>
      <td>Siegfried &amp; Roy: The Magic Box (1999)</td>
      <td>4.25</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>131108</th>
      <td>The Fearless Four (1997)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>131110</th>
      <td>A House of Secrets: Exploring 'Dragonwyck' (2008)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>131166</th>
      <td>WWII IN HD (2009)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>131172</th>
      <td>Closed Curtain (2013)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>131260</th>
      <td>Rentun Ruusu (2001)</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>27103 rows × 2 columns</p>
</div>

Notice that this step already takes care of situations in which a movie belongs to a certain genre but the user has no rating for that genre. For example, this user has a rating for the Action gente, but not for Western. If movie belongs to both genres, the average will be taken only over one genre, since `content_based_ratings.loc[movie_id, "<genre without user rating>"]` will be equal to 0 and ignored when accounting for our denominator.

We can also compute the alternate user-genre preferences, using normalized sums of ratings, which are highly correlated with the number of movies watched in each genre. You can see we get a different list of movies, although some movies remain in the list and the first movie is the same.

```python
content_based_ratings = unwatched_movies[genres].multiply(user_genre_preferences["Normalized sum of ratings"])
content_based_ratings = content_based_ratings.sum(axis=1)
unwatched_movies["content_based_rating2"] = content_based_ratings
unwatched_movies[["title", "content_based_rating2"]].sort_values("content_based_rating2", ascending=False)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table class="post-body">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>content_based_rating2</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>81132</th>
      <td>Rubber (2010)</td>
      <td>4.783002</td>
    </tr>
    <tr>
      <th>49593</th>
      <td>She (1965)</td>
      <td>4.725136</td>
    </tr>
    <tr>
      <th>5018</th>
      <td>Motorama (1991)</td>
      <td>4.717902</td>
    </tr>
    <tr>
      <th>71999</th>
      <td>Aelita: The Queen of Mars (Aelita) (1924)</td>
      <td>4.687161</td>
    </tr>
    <tr>
      <th>72165</th>
      <td>Cirque du Freak: The Vampire's Assistant (2009)</td>
      <td>4.569620</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>100246</th>
      <td>Pretty Sweet (2012)</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Heidi Fleiss: Hollywood Madam (1995)</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>100266</th>
      <td>Day Is Done (2011)</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>100287</th>
      <td>Head Games (2012)</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>96842</th>
      <td>Behind the Burly Q: The Story of Burlesque in ...</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>27103 rows × 2 columns</p>
</div>

Both of the tables above are equivalent to the table presented in step 5 of the example.

## Discussion

One disadvantage of content-based filtering is that we can only recommend an item for a user if that user has rated at least one other item that contains at least one feature in common with the item in question.

In our case, we can only recommend movies to a user if that movie contains at least one genre in common with another movie that the user has already rated.

The fewer features an unrated item has with rated ones, the less granular its inferred rating will be.

The diagrams below illustrate a situation in which a user has ratings for 17 genres, computed from the ratings of movies that contain those genres. In the first example, a movie with 5 genres has 2 genres in common with rated genres for the user, so we can compute an inferred rating for the movie.

In the second example, a movie with 5 genres has 0 genres in common with the rated genres for the user, so we cannot directly compute an inferred rating for it.

```python
from matplotlib_venn import venn2
import matplotlib.pyplot as plt

movie_genres_count = 5
intersection = 2
set1 = genres_with_ratings - intersection
set2 = movie_genres_count - intersection
venn2(subsets=(set1, set2, intersection), set_labels=("Features with\nuser ratings", "Movie features"), alpha=0.5)
plt.title(f"Movie that can be rated based on {intersection} features")
plt.show()
venn2(
  subsets=(genres_with_ratings, movie_genres_count, 0),
  set_labels=("Features with\nuser ratings", "Movie features"),
  alpha=0.5,
)
plt.title("Movie that cannot be rated")
plt.show()

```

![png](/assets/img/content-based-filtering_39_0.png)

![png](/assets/img/content-based-filtering_39_1.png)

Because in our example, this user has watched over 100 movies, we are able to recommend over 90% of unwatched movies to them.

```python
movies_with_inferred_ratings = (~unwatched_movies["content_based_rating"].isna()).sum()
print(f"Number of unwatched movies: {len(unwatched_movies)}")
print(f"Number of unwatched movies with inferred ratings: {movies_with_inferred_ratings}")
print(
  f"Percentage of unwatched movies with inferred ratings: {movies_with_inferred_ratings / len(unwatched_movies):.2%}"
)
```

    Number of unwatched movies: 27103
    Number of unwatched movies with inferred ratings: 24901
    Percentage of unwatched movies with inferred ratings: 91.88%

Another disadvantage is that, depending on the number of features and technique employed in computing inferred ratings, the granularity of ratings can be very poor.

In our case, although we inferred ratings for over 24,000 movies, over 50% of them received one of 10 ratings.

```python
inferred_ratings_count = unwatched_movies["content_based_rating"].value_counts()
topk = 10

print(f"Unique inferred ratings: {len(inferred_ratings_count)}")
print(
  f"Percentage of unwatched movies in the top {topk} inferred ratings: {(inferred_ratings_count.iloc[:topk].sum() / movies_with_inferred_ratings):.2%}"
)
```

    Unique inferred ratings: 1182
    Percentage of unwatched movies in the top 10 inferred ratings: 51.42%

This happens because, while we have $$n$$ total features to represent user taste and items (the movie genres), the current user only has ratings for a subset $$m \leq n$$ of the features.

Using a simple average over genre ratings, a movie's inferred rating can only come from one of $$2^m-1$$ possible values.

If the number of genres of the movie is known, say $$k$$, then this value is restricted to $$\binom{m}{k}$$, assuming that each possible rating is different.

The table below shows the number of possible ratings a movie may have, given the number of genres it has.

```python
from scipy.special import comb

genres_count = unwatched_movies[genres].sum(axis=1).value_counts()
possible_ratings = pd.DataFrame(
  index=genres_count.index,
  data={
    "Number of Movies": genres_count,
    "Possible ratings": [comb(genres_with_ratings, i) for i in genres_count.index],
  },
)
possible_ratings["Movies per possible rating"] = (
  possible_ratings["Number of Movies"] / possible_ratings["Possible ratings"]
)
possible_ratings.index.name = "Number of genres"
possible_ratings
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table class="post-body">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Number of Movies</th>
      <th>Possible ratings</th>
      <th>Movies per possible rating</th>
    </tr>
    <tr>
      <th>Number of genres</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>10816</td>
      <td>17.0</td>
      <td>636.235294</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8766</td>
      <td>136.0</td>
      <td>64.455882</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5260</td>
      <td>680.0</td>
      <td>7.735294</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1685</td>
      <td>2380.0</td>
      <td>0.707983</td>
    </tr>
    <tr>
      <th>5</th>
      <td>468</td>
      <td>6188.0</td>
      <td>0.075630</td>
    </tr>
    <tr>
      <th>6</th>
      <td>82</td>
      <td>12376.0</td>
      <td>0.006626</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20</td>
      <td>19448.0</td>
      <td>0.001028</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>24310.0</td>
      <td>0.000206</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>19448.0</td>
      <td>0.000051</td>
    </tr>
  </tbody>
</table>
</div>

## Conclusion

In this project, we have built a movie recommendation system based on content-based filtering.

We went through the theory of content-based filtering as well as an example, then used a publicly available dataset to implement the system. We compared two ways of building user features and saw one of the downsides of content-based filtering, which is the inability of recommending new items to a user if those items don't have intersecting features with the items the user has already rated.

See you guys next time!
