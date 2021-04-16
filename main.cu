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


template <class G, class H, class C>
void runPageRankCuda(G& g, H& gt, bool all, PageRankFlags F, C& r1) {
  typedef PageRankMode Mode; float t;
  if (!isValid(F)) return;
  auto r2 = pageRankCuda(t, g, gt, {Mode::BLOCK, F});
  printf("[%07.1f ms] [%.4e] pageRankCuda {block}    ", t, absError(r1, r2)); cout << stringify(F) << "\n"; if (all) print(r2);
  auto r3 = pageRankCuda(t, g, gt, {Mode::THREAD, F});
  printf("[%07.1f ms] [%.4e] pageRankCuda {thread}   ", t, absError(r1, r3)); cout << stringify(F) << "\n"; if (all) print(r3);
  if (!isValidSwitched(F)) return;
  auto r4 = pageRankCuda(t, g, gt, {Mode::SWITCHED, F});
  printf("[%07.1f ms] [%.4e] pageRankCuda {switched} ", t, absError(r1, r4)); cout << stringify(F) << "\n"; if (all) print(r4);
}


template <class G, class H, class C>
void runPageRankSteppedCuda(G& g, H& gt, bool all, PageRankFlags F, C& r1) {
  typedef PageRankMode Mode; float t;
  if (!isValid(F) || !isValidStepped(F)) return;
  auto r5 = pageRankSteppedCuda(t, g, gt, {Mode::BLOCK, F});
  printf("[%07.1f ms] [%.4e] pageRankSteppedCuda {block}    ", t, absError(r1, r5)); cout << stringify(F) << "\n"; if (all) print(r5);
  auto r6 = pageRankSteppedCuda(t, g, gt, {Mode::THREAD, F});
  printf("[%07.1f ms] [%.4e] pageRankSteppedCuda {thread}   ", t, absError(r1, r6)); cout << stringify(F) << "\n"; if (all) print(r6);
  if (!isValidSwitched(F)) return;
  auto r7 = pageRankSteppedCuda(t, g, gt, {Mode::SWITCHED, F});
  printf("[%07.1f ms] [%.4e] pageRankSteppedCuda {switched} ", t, absError(r1, r7)); cout << stringify(F) << "\n"; if (all) print(r7);
}


template <class G, class H>
void runPageRank(G& g, H& gt, bool all) {
  typedef PageRankFlags Flags;
  float t;
  auto r1 = pageRank(t, g, gt);
  printf("[%07.1f ms] [%.4e] pageRank\n", t, absError(r1, r1)); if (all) print(r1);
  for (int o=0; o<128; o++)
    runPageRankCuda(g, gt, all, Flags(o), r1);
  for (int o=0; o<128; o++)
    runPageRankSteppedCuda(g, gt, all, Flags(o), r1);
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
