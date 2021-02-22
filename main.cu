#include "src/main.h"
#include "tests.h"
#include "runs.h"

using namespace std;




int main(int argc, char **argv) {
  testAll();
  printf("Loading graph ...\n");
  DiGraph<> g;
  DiGraph<int, int> h;
  readMtx(argv[1], g);
  print(g);
  transposeWithDegree(g, h);
  print(h);
  runAdd();
  runFill();
  runSum();
  runErrorAbs();
  runDotProduct();
  runPageRank(g);
  runPageRankPull(h);
  return 0;
}
