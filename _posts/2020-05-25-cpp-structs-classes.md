---
layout: post
title: In C++, classes and structs are the same thing
categories: cpp programming
---

**Disclaimer:** I am by no means a C or C++ expert. I have never programmed in C++ professionally, only academically. You can view my C++ work [here](https://github.com/douglasrizzo?tab=repositories&language=c%2B%2B). My reference book for C++ is [Deitel](https://www.amazon.com/How-Program-10th-Paul-Deitel/dp/0134448235). A good playlist to learn C++ is [The Cherno's](https://www.youtube.com/playlist?list=PLlrATfBNZ98dudnM48yfGUldqGD0S4FFb). Examples were expanded from The Cherno's [YouTube video](https://www.youtube.com/watch?v=fLgTtaqqJp0). Code is available [on GitHub](https://gist.github.com/douglasrizzo/b1375881d1afb70cd1fe76fefa47d3f0).

---

<!-- TOC -->

- [Introduction](#introduction)
- [A versatile entry point (`main.cpp`)](#a-versatile-entry-point-maincpp)
- [class_class.hpp](#class_classhpp)
- [struct_struct.hpp](#struct_structhpp)
- [The inbred brethren](#the-inbred-brethren)
    - [struct_class.hpp](#struct_classhpp)
    - [class_struct.hpp](#class_structhpp)
- [Access modifiers in structs (`struct_access_mod.cpp`)](#access-modifiers-in-structs-struct_access_modcpp)
- [What a valid C struct looks like (`c_struct.c`)](#what-a-valid-c-struct-looks-like-c_structc)
- [Conclusion](#conclusion)

<!-- /TOC -->

## Introduction

The C language came before C++. C has structs, which is a composite data type. Basically, it is a variable that groups more variables together and stores them in a contiguous manner in memory.

When C++ came about, it covered all possible use cases developers could have for structs by introducing classes. However, in order to maintain backwards compatibility, C++ still allowed the creation of structs.

Recently, I discovered that structs in C++ are much more powerful than they are in C. For example, a struct in C++ can have access modifiers (think `public` and `private`), just like classes. They also allow for functions to be declared inside them, just like... classes. Lastly, a struct can *inherit* all the members from another struct, similar to what other C++ declaration? You guessed it, **classes**.

C++ structs still maintain backwards compatibility with C, because you can still declare a struct as a [passive data structure](https://en.wikipedia.org/wiki/Passive_data_structure). But, in order to keep language implementation simple (this is my hunch, not gospel), **in C++, structs and classes are the same thing**. The only difference being that, in classes, all access modifiers are private by default, while, in structs, they are public by default, just like in C.

If, like me, you have been taught a single, monolithic concept of structs (that they are "simpler than classes" or what have you) and do not know exactly how they behave or may be employed in C++, check the examples below in which I show that:

1. structs can have functions inside of them, just like classes have methods;
2. structs may have private members, unlike in C;
3. structs and classes can inherit from one another, the only difference being the need to explicitly set the visibility of private struct members and public class members.

I'll create two "things" called `Vec2` and `Vec3`. `Vec3` will inherit from `Vec2`, access is attributes and even call one of its methods. Our `main()` function will instantiate two `Vec3` objects, add their coordinates and display them to stdout.

The catch is, I have created four different files with different implementations for `Vec2` and `Vec3`, showcasing both structs and classes inheriting from structs and classes. All we need to do is uncomment which header file we want to use and the code will work in every case, outputting `4 4 4`.

## A versatile entry point (`main.cpp`)

```cpp
#include <iostream>
#include <struct_struct.hpp>
// #include <class_class.hpp>
// #include <struct_class.hpp>
// #include <class_struct.hpp>

int main() {
  Vec3 v1, v2;

  v1.x = 1;
  v1.y = 2;
  v1.z = 3;
  v2.x = 3;
  v2.y = 2;
  v2.z = 1;

  v1.add(v2);

  std::cout << v1.x << " " << v1.y << " " << v1.z << std::endl;
}
```

## class_class.hpp

Anyone who is familiar with object oriented programming knows this: classes inherit from classes. There really isn't any surprises here. Just notice how we need to set the class attributes to `public` so that we can access them in `main()`.

```cpp
class Vec2 {
 public:
  float x, y;

  void add(Vec2 other) {
    x += other.x;
    y += other.y;
  }
};

class Vec3 : public Vec2 {
 public:
  float z;

  void add(Vec3 other) {
    Vec2::add(other);
    z += other.z;
  }
};
```

## struct_struct.hpp

Here, things can get a little more esoteric to those that have learned that "structs are simpler than classes" (they are not) or that their only purpose is to create [passive data structures](https://en.wikipedia.org/wiki/Passive_data_structure) (it is not). Structs may have methods, inherit from other structs and even have access modifiers such as `private`.

Remember that this behavior is not present in C structs. That's where most of the misconceptions or best practices on when to use structs , even in C++, come from. There may be some best practices dictating that classes should be preferred when dealing with objects that have complex behavior, while structs should be used when creating plain old data objects, but technically, this example shows that both classes and structs are interchangeable in C++.

One thing to note is that here, we don't need to set the visibility of the struct members to `public`, as that is their natural visibility.

```cpp
struct Vec2 {
  float x, y;

  void add(Vec2 other) {
    x += other.x;
    y += other.y;
  }
};

struct Vec3 : public Vec2 {
  float z;

  void add(Vec3 other) {
    Vec2::add(other);
    z += other.z;
  }
};
```

## The inbred brethren

In the following two examples, I show that it is possible for structs and classes to inherit from one another in C++.

### struct_class.hpp

```cpp
struct Vec2 {
  float x, y;

  void add(Vec2 other) {
    x += other.x;
    y += other.y;
  }
};

class Vec3 : public Vec2 {
 public:
  float z;

  void add(Vec3 other) {
    Vec2::add(other);
    z += other.z;
  }
};
```

### class_struct.hpp

```cpp
class Vec2 {
 public:
  float x, y;

  void add(Vec2 other) {
    x += other.x;
    y += other.y;
  }
};

struct Vec3 : public Vec2 {
  float z;

  void add(Vec3 other) {
    Vec2::add(other);
    z += other.z;
  }
};
```

## Access modifiers in structs (`struct_access_mod.cpp`)

A last example that wasn't covered in the previous ones. In C++, structs may have private members, *i.e.* inner variables that may not be accessed outside of the struct. This behavior does not exist in C structs.

Notice the `private` access modifier for the Boolean variable inside the `Vec2` struct below.

```cpp
#include <iostream>

struct Vec2 {
  float x, y;

  void add(Vec2 other) {
    x += other.x;
    y += other.y;
  }

 private:
  bool inverted;
};

int main(void) {
  Vec2 v;
  std::cout << v.inverted;
}
```

When I try to compile this file with `g++`, I get an error, since I am trying to access the `inverted` attribute outside of the struct.

```
struct_access_mod.cpp: In function ‘int main()’:
struct_access_mod.cpp:17:18: error: ‘bool Vec2::inverted’ is private within this context
   17 |   std::cout << v.inverted;
      |                  ^~~~~~~~
struct_access_mod.cpp:12:8: note: declared private here
   12 |   bool inverted;
      |        ^~~~~~~~
```

## What a valid C struct looks like (`c_struct.c`)

This last example shows what a valid struct looks like in C: no access modifiers, no inheritance and no methods. Also notice the need to identify `v` as a `struct` during its declaration, something that has been made optional in C++. Compiling this file with either `gcc` or `g++` results in success.

It is important to know that, while C structs don't allow for functions to be declared inside of them, they may still store function pointers, but I don't have enough experience with plain C to know if this is actually useful.

```cpp
#include <stdio.h>

struct Vec2 {
  float x, y;
};

int main() {
  struct Vec2 v;

  v.x = 1;
  v.y = 2;

  printf("%f %f", v.x, v.y);
}
```

## Conclusion

The first conclusion is that, while C++ still supports structs for backwards compatibility with C, it implements them in a whole different way, making them basically an alias for classes.

A second, more personal conclusion is that, studying this topic helped me get a firmer grasp on exactly what C and C++ are, what are their boundaries, where one stops and the other begins. I had this erroneous thought that "structs are a thing from C" and that C++, being a superset of C, inherits all of the functionality and limitations structs had, since they came from a limited subset of instructions. Now I know:

* what are the limitations and use cases of structs in C;
* exactly what the differences between C and C++ structs are;
* that everything I know about classes in C++ and other languages can be transferred to structs, but only in C++ and not in C.

Now I know that C++ can expand C in ways I have never considered before.
