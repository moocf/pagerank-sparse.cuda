#pragma once




template <class I>
class SliceIterable {
  const I ib, ie;

  public:
  SliceIterable(I ib, I ie) : ib(ib), ie(ie) {}
  auto begin() { return ib; }
  auto end()   { return ie; }
};

template <class I>
auto slice(I ib, I ie) {
  return SliceIterable<I>(ib, ie);
}

template <class T>
auto slice(vector<T>& x, int i=0) {
  using I = decltype(x.begin());
  return SliceIterable<I>(x.begin()+i, x.end());
}
