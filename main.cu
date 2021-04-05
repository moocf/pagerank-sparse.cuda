#include <iostream>
#include <stdio.h>
#include "src/main.h"
#include "setups.h"
#include "tests.h"
#include "runs.h"

using namespace std;




template <class G, class H>
void runPageRankPush(G& g, H& gt, bool all) {
  float t;
  auto r1 = pageRankPush(t, g, gt);
  printf("[%07.1f ms] pageRankPush\n", t); if (all) print(r1);
}


template <class G, class H>
void runPageRank(G& g, H& gt, bool all) {
  typedef PageRankMode  Mode;
  typedef PageRankFlags Flags;
  float t; Flags f;
  for (int i=0; i<2; i++) {
    f.skipConverged = true; // i&1 == 1;
    auto r1 = pageRank(t, g, gt);
    printf("[%07.1f ms] pageRank\n", t); if (all) print(r1);
    auto r2 = pageRankCuda(t, g, gt, {Mode::BLOCK, f});
    printf("[%07.1f ms] pageRankCuda {block}    ", t); cout << stringify(f) << "\n"; if (all) print(r2);
    auto r3 = pageRankCuda(t, g, gt, {Mode::THREAD, f});
    printf("[%07.1f ms] pageRankCuda {thread}   ", t); cout << stringify(f) << "\n"; if (all) print(r3);
    auto r4 = pageRankCuda(t, g, gt, {Mode::SWITCHED, f});
    printf("[%07.1f ms] pageRankCuda {switched} ", t); cout << stringify(f) << "\n"; if (all) print(r4);
  }
  // auto r5 = pageRankSticdCuda(t, g, gt);
  // printf("[%07.1f ms] pageRankSticdCuda      \n", t); if (all) print(r5);
}


int main(int argc, char **argv) {
  char *file = argv[1];
  bool all = argc > 2;

  setupAll();
  testAll();
  printf("Loading graph %s ...\n", file);
  auto g = readMtx(file);
  print(g);
  auto gt = transposeWithDegree(g);
  print(gt);
  // runPageRankPush(g, gt, all);
  runPageRank(g, gt, all);
  // runAdd();
  // runFill();
  // runSum();
  // runErrorAbs();
  // runDotProduct();
  printf("\n");
  return 0;
}
