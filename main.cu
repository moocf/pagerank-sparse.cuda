#include "src/main.h"
#include "setups.h"
#include "tests.h"
#include "runs.h"

using namespace std;




template <class G>
void runPageRankPush(G& g, bool all) {
  float t;
  auto ranks = pageRankPush(t, g);
  printf("[%07.1f ms] pageRankPush\n", t); if (all) print(ranks);
}


template <class G, class H>
void runPageRank(G& g, H& gt, bool all) {
  typedef PageRankMode Mode;
  float t;
  auto r1 = pageRank(t, gt);
  printf("[%07.1f ms] pageRank               \n", t); if (all) print(r1);
  auto r2 = pageRankCuda(t, gt, {Mode::BLOCK});
  printf("[%07.1f ms] pageRankCuda {block}   \n", t); if (all) print(r2);
  auto r3 = pageRankCuda(t, gt, {Mode::THREAD});
  printf("[%07.1f ms] pageRankCuda {thread}  \n", t); if (all) print(r3);
  auto r4 = pageRankCuda(t, gt, {Mode::SWITCHED});
  printf("[%07.1f ms] pageRankCuda {switched}\n", t); if (all) print(r4);
  auto r5 = pageRankSticdCuda(t, g, gt);
  printf("[%07.1f ms] pageRankSticdCuda      \n", t); if (all) print(r5);
}


int main(int argc, char **argv) {
  DiGraph<> g;
  DiGraph<int, int> h;
  char *file = argv[1];
  bool all = argc > 2;

  setupAll();
  testAll();
  printf("Loading graph %s ...\n", file);
  readMtx(file, g);
  print(g);
  transposeWithDegree(g, h);
  print(h);
  // runPageRankPush(g, all);
  runPageRank(g, h, all);
  // runAdd();
  // runFill();
  // runSum();
  // runErrorAbs();
  // runDotProduct();
  printf("\n");
  return 0;
}
