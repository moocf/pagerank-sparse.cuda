#pragma once




template <class C, class I>
bool contains(C& x, I&& vs) {
  for (auto v : vs)
    if (x.count(v) == 0) return false;
  return true;
}
