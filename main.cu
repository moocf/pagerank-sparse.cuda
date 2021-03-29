#include "src/main.h"
#include "tests.h"
#include "runs.h"

using namespace std;




template <class G>
void runPageRankPush(G& g, bool all) {
  float t;
  auto ranks = pageRankPush(t, g);
  printf("[%07.1f ms] pageRankPush\n", t); if (all) print(ranks);
}


template <class G>
void runPageRank(G& g, bool all) {
  typedef PageRankMode Mode;
  float t;
  auto ranks1 = pageRank(t, g);
  printf("[%07.1f ms] pageRank               \n", t); if (all) print(ranks1);
  auto ranks2 = pageRankCuda(t, g, {Mode::BLOCK});
  printf("[%07.1f ms] pageRankCuda {block}   \n", t); if (all) print(ranks2);
  auto ranks3 = pageRankCuda(t, g, {Mode::THREAD});
  printf("[%07.1f ms] pageRankCuda {thread}  \n", t); if (all) print(ranks3);
  auto ranks4 = pageRankCuda(t, g, {Mode::SWITCHED});
  printf("[%07.1f ms] pageRankCuda {switched}\n", t); if (all) print(ranks4);
}


int main(int argc, char **argv) {
  DiGraph<> g;
  DiGraph<int, int> h;
  char *file = argv[1];
  bool all = argc > 2;

  testAll();
  printf("Loading graph %s ...\n", file);
  readMtx(file, g);
  print(g);
  transposeWithDegree(g, h);
  print(h);
  // runPageRankPush(g, all);
  runPageRank(h, all);
  // runAdd();
  // runFill();
  // runSum();
  // runErrorAbs();
  // runDotProduct();
  printf("\n");
  return 0;
}
