#include <iostream>
#include <stdio.h>
#include "src/main.h"
#include "setups.h"
#include "tests.h"
#include "runs.h"

using namespace std;




template <class G, class H>
void runPageRankPush(G& x, H& xt, bool all) {
  float t;
  auto r1 = pageRankPush(t, x, xt);
  printf("[%09.3f ms] pageRankPush\n", t); if (all) println(r1);
}


template <class G, class H, class C, class D>
void runPageRankCuda(G& x, H& xt, C& cs, C& id, C& ch, bool all, PageRankFlags F, D& r1) {
  typedef PageRankMode Mode; float t;
  if (!isValid(F)) return;
  auto r2 = pageRankCuda(t, x, xt, cs, id, ch, {Mode::BLOCK, F});
  printf("[%09.3f ms] [%.4e] pageRankCuda {block}    ", t, absError(r1, r2)); cout << stringify(F) << "\n"; if (all) println(r2);
  auto r3 = pageRankCuda(t, x, xt, cs, id, ch, {Mode::THREAD, F});
  printf("[%09.3f ms] [%.4e] pageRankCuda {thread}   ", t, absError(r1, r3)); cout << stringify(F) << "\n"; if (all) println(r3);
  if (!isValidSwitched(F)) return;
  auto r4 = pageRankCuda(t, x, xt, cs, id, ch, {Mode::SWITCHED, F});
  printf("[%09.3f ms] [%.4e] pageRankCuda {switched} ", t, absError(r1, r4)); cout << stringify(F) << "\n"; if (all) println(r4);
}


template <class G, class H, class C, class D>
void runPageRankSteppedCuda(G& x, H& xt, C& cs, C& id, C& ch, bool all, PageRankFlags F, D& r1) {
  typedef PageRankMode Mode; float t;
  if (!isValid(F) || !isValidStepped(F)) return;
  auto r5 = pageRankSteppedCuda(t, x, xt, cs, id, ch, {Mode::BLOCK, F});
  printf("[%09.3f ms] [%.4e] pageRankSteppedCuda {block}    ", t, absError(r1, r5)); cout << stringify(F) << "\n"; if (all) println(r5);
  auto r6 = pageRankSteppedCuda(t, x, xt, cs, id, ch, {Mode::THREAD, F});
  printf("[%09.3f ms] [%.4e] pageRankSteppedCuda {thread}   ", t, absError(r1, r6)); cout << stringify(F) << "\n"; if (all) println(r6);
  if (!isValidSwitched(F)) return;
  auto r7 = pageRankSteppedCuda(t, x, xt, cs, id, ch, {Mode::SWITCHED, F});
  printf("[%09.3f ms] [%.4e] pageRankSteppedCuda {switched} ", t, absError(r1, r7)); cout << stringify(F) << "\n"; if (all) println(r7);
}


template <class G>
void runPageRank(G& x, bool all) {
  typedef PageRankFlags Flags; float t;
  auto xt = transposeWithDegree(x); print(xt); printf(" (transposeWithDegree)\n");
  auto xn = transposeForNvgraph(x); print(xn); printf(" (transposeForNvgraph)\n");
  auto cs = components(x, xt);   auto de = deadEnds(x);
  printf("components: %zu deadEnds: %zu\n", cs.size(), de.size());
  auto id = inIdenticals(x, xt); auto ch = chains(x, xt);
  printf("inIdenticals: %zu chains: %zu\n", id.size(), ch.size());
  auto r1 = pageRankNvgraph(t, xn);
  printf("[%09.3f ms] [%.4e] pageRankNvgraph\n", t, absError(r1, r1)); if (all) println(r1);
  for (int o=0; o<128; o++)
    runPageRankCuda(x, xt, cs, id, ch, all, Flags(o), r1);
  for (int o=0; o<128; o++)
    runPageRankSteppedCuda(x, xt, cs, id, ch, all, Flags(o), r1);
}


int main(int argc, char **argv) {
  char *file = argv[1];
  bool all = argc > 2;
  setupAll(); testAll();
  printf("Loading graph %s ...\n", file);
  auto x = readMtx(file); println(x);
  runPageRank(x, all);
  printf("\n");
  return 0;
}
