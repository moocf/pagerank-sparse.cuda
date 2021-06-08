#include <iostream>
#include <stdio.h>
#include "src/main.h"

using namespace std;




template <class G, class H>
void runPageRankPush(G& x, H& xt, bool all) {
  float t;
  auto r1 = pageRankPush(t, x, xt);
  printf("[%09.3f ms] pageRankPush\n", t); if (all) println(r1);
}


template <class G, class H, class D>
void runPageRankCuda(G& x, H& xt, bool all, PageRankFlags F, D& r1) {
  typedef PageRankMode Mode; float t, R;
  if (!isValid(F)) return;
  auto r2 = pageRankCuda(t, R, x, xt, {Mode::BLOCK, F});
  printf("[%09.3f ms; %03dR] [%.4e] pageRankCuda {block}    ", t, (int) R, absError(r1, r2)); cout << stringify(F) << "\n"; if (all) println(r2);
  auto r3 = pageRankCuda(t, R, x, xt, {Mode::THREAD, F});
  printf("[%09.3f ms; %03dR] [%.4e] pageRankCuda {thread}   ", t, (int) R, absError(r1, r3)); cout << stringify(F) << "\n"; if (all) println(r3);
  if (!isValidSwitched(F)) return;
  auto r4 = pageRankCuda(t, R, x, xt, {Mode::SWITCHED, F});
  printf("[%09.3f ms; %03dR] [%.4e] pageRankCuda {switched} ", t, (int) R, absError(r1, r4)); cout << stringify(F) << "\n"; if (all) println(r4);
}


template <class G>
void runPageRank(G& x, bool all) {
  typedef PageRankFlags Flags; float t;
  auto xt = transposeWithDegree(x); print(xt); printf(" (transposeWithDegree)\n");
  auto xn = transposeForNvgraph(x); print(xn); printf(" (transposeForNvgraph)\n");
  auto r1 = pageRankNvgraph(t, xn);
  printf("[%09.3f ms; %03dR] [%.4e] pageRankNvgraph\n", t, 0, absError(r1, r1)); if (all) println(r1);
  for (int o=0; o<1; o++)
    runPageRankCuda(x, xt, all, Flags(o), r1);
}


int main(int argc, char **argv) {
  char *file = argv[1];
  bool all = argc > 2;
  printf("Loading graph %s ...\n", file);
  auto x = readMtx(file); println(x);
  runPageRank(x, all);
  printf("\n");
  return 0;
}
