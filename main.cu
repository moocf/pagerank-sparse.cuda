#include "src/main.h"
#include "tests.h"
#include "runs.h"

using namespace std;




template <class G>
void runPageRankPush(G& g) {
  float t;
  auto ranks = pageRankPush(t, g);
  printf("[%07.1f ms] pageRankPush\n", t); print(ranks);
}


template <class G>
void runPageRank(G& g) {
  float t;
  auto ranks1 = pageRank(t, g);
  printf("[%07.1f ms] pageRank\n", t); print(ranks1);
  auto ranks2 = pageRankCuda(t, g);
  printf("[%07.1f ms] pageRankCuda \n", t); print(ranks2);
}


int main(int argc, char **argv) {
  testAll();
  printf("Loading graph ...\n");
  DiGraph<> g;
  DiGraph<int, int> h;
  readMtx(argv[1], g);
  print(g);
  transposeWithDegree(g, h);
  print(h);
  // runAdd();
  // runFill();
  // runSum();
  // runErrorAbs();
  // runDotProduct();
  runPageRankPush(g);
  runPageRank(h);
  return 0;
}
