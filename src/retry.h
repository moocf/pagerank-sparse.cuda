#pragma once




template <class G, class F>
void retry(F fn, G& x, int L=10) {
  int n = x.order(), m = x.size();
  for (int l=0; l<L; l++) {
    fn();
    if (x.order()!=n || x.size()!=m) break;
  }
}
