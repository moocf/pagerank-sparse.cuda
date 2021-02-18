#pragma once
#include <vector>
#include <iterator>
#include <algorithm>
#include "ceilDiv.h"
#include "transform.h"
#include <stdio.h>

using std::random_access_iterator_tag;
using std::vector;
using std::max;




template <class T>
int rangeSize(T v, T V, T DV=1) {
  return max(0, (int) ceilDiv(V-v, DV));
}


template <class T>
int rangeLast(T v, T V, T DV=1) {
  return v + DV*(rangeSize(v, V, DV) - 1);
}


template <class T>
T* rangeData(T v, T V, T DV=1) {
  int N = rangeSize(v, V, DV);
  T *a = new T[N];
  for (int i=0; v<V; i++, v+=DV)
    a[i] = v;
  return a;
}

template <class T>
T* rangeData(T V) {
  return rangeDate(0, V);
}


template <class T>
vector<T> rangeVector(T v, T V, T DV=1) {
  int N = rangeSize(v, V, DV);
  vector<T> a;
  a.reserve(N);
  for (; v<V; v+=DV)
    a.push_back(v);
  return a;
}

template <class T>
vector<T> rangeVector(T V) {
  return rangeVector(0, V);
}




template <class T>
class RangeIterable {
  const T N;

  public:
  class Iterator {
    T n;

    public:
    using iterator_category = random_access_iterator_tag;
    using difference_type = const T;
    using value_type = T;
    using pointer = value_type*;
    using reference = value_type;

    Iterator(T n) : n(n) {}
    ITERATOR_DEREF(Iterator, n, n+i, NULL)
    ITERATOR_NEXT(Iterator, ++n, --n)
    ITERATOR_ADVANCE(Iterator, i, n += i, n -= i)
    ITERATOR_ARITHMETICP(Iterator, a, b, a.n+b)
    ITERATOR_ARITHMETICN(Iterator, a, b, a.n-b)
    ITERATOR_COMPARISION(Iterator, a, b, a.n, b.n)
  };

  RangeIterable(T N) : N(N) {}
  Iterator begin() { return Iterator(0); }
  Iterator end() { return Iterator(N); }
  Iterator rbegin() { return Iterator(N-1); }
  Iterator rend() { return Iterator(-1); }
  ITERABLE_SIZE(N)
};


template <class T>
auto range(T v, T V, T DV=1) {
  int N = rangeSize(v, V, DV);
  auto r = RangeIterable<T>(N);
  return transform(r, [=](int n) { return v+DV*n; });
}

template <class T>
auto range(T V) {
  return RangeIterable<T>(V);
}
