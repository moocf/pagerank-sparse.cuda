#pragma once
#include "_support.h"
#include "transform.h"




template <class I>
auto sliceIterable(I x, int i=0, int I=x.size()) {
  return transform(x.begin()+i, x.begin()+I, IDENTITY);
}
