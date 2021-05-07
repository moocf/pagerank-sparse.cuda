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
void runPageRankCuda(G& x, H& xt, H& xe, H& xf, C& cs, C& id, C& ch, bool all, PageRankFlags F, D& r1) {
  typedef PageRankMode Mode; float t, R;
  if (!isValid(F)) return;
  auto r2 = pageRankCuda(t, R, x, xt, cs, id, ch, {Mode::BLOCK, F});
  printf("[%09.3f ms; %03dR] [%.4e] pageRankCuda {block}    ", t, (int) R, absError(r1, r2)); cout << stringify(F) << "\n"; if (all) println(r2);
  auto r3 = pageRankCuda(t, R, x, xt, cs, id, ch, {Mode::THREAD, F});
  printf("[%09.3f ms; %03dR] [%.4e] pageRankCuda {thread}   ", t, (int) R, absError(r1, r3)); cout << stringify(F) << "\n"; if (all) println(r3);
  if (!isValidSwitched(F)) return;
  auto r4 = pageRankCuda(t, R, x, xt, cs, id, ch, {Mode::SWITCHED, F});
  printf("[%09.3f ms; %03dR] [%.4e] pageRankCuda {switched} ", t, (int) R, absError(r1, r4)); cout << stringify(F) << "\n"; if (all) println(r4);
}


template <class G, class H, class C, class D>
void runPageRankSteppedCuda(G& x, H& xt, H& xe, H& xf, C& cs, C& id, C& ch, bool all, PageRankFlags F, D& r1) {
  typedef PageRankMode Mode; float t, R;
  if (!isValidStepped(F)) return;
  auto r5 = pageRankSteppedCuda(t, R, x, xt, xe, xf, cs, id, ch, {Mode::BLOCK, F});
  printf("[%09.3f ms; %03dR] [%.4e] pageRankSteppedCuda {block}    ", t, (int) R, absError(r1, r5)); cout << stringify(F) << "\n"; if (all) println(r5);
  auto r6 = pageRankSteppedCuda(t, R, x, xt, xe, xf, cs, id, ch, {Mode::THREAD, F});
  printf("[%09.3f ms; %03dR] [%.4e] pageRankSteppedCuda {thread}   ", t, (int) R, absError(r1, r6)); cout << stringify(F) << "\n"; if (all) println(r6);
  if (!isValidSwitched(F)) return;
  auto r7 = pageRankSteppedCuda(t, R, x, xt, xe, xf, cs, id, ch, {Mode::SWITCHED, F});
  printf("[%09.3f ms; %03dR] [%.4e] pageRankSteppedCuda {switched} ", t, (int) R, absError(r1, r7)); cout << stringify(F) << "\n"; if (all) println(r7);
}


template <class G>
void runPageRank(G& x, bool all) {
  typedef PageRankFlags Flags; float t;
  int GB = GRID_DIM * BLOCK_DIM;
  loopDeadEnds(x); // DEBUG: remove this
  auto xt = transposeWithDegree(x); print(xt); printf(" (transposeWithDegree)\n");
  auto xn = transposeForNvgraph(x); print(xn); printf(" (transposeForNvgraph)\n");
  auto cs = components(x, xt); auto ds = joinUntilSize(cs, GB);
  auto xe = crossEdges(xt, ds);     print(xe); printf(" (crossEdges)\n");
  auto xf = edgeDifference(xt, xe); print(xf); printf(" (crossEdgesFree)\n");
  auto de = deadEnds(x); auto id = inIdenticals(x, xt); auto ch = chains(x, xt);
  printf("components: %zu largeComponents: %zu\n", cs.size(), ds.size());
  printf("deadEnds: %zu inIdenticals: %zu chains: %zu\n", de.size(), id.size(), ch.size());
  auto r1 = pageRankNvgraph(t, xn);
  printf("[%09.3f ms; %03dR] [%.4e] pageRankNvgraph\n", t, 0, absError(r1, r1)); if (all) println(r1);
  for (int o=0; o<1; o++) // DEBUG: o<256
    runPageRankCuda(x, xt, xe, xf, cs, id, ch, all, Flags(o), r1);
  for (int o=0; o<256; o++)
    runPageRankSteppedCuda(x, xt, xe, xf, cs, id, ch, all, Flags(o), r1);
}


template <class G, class C>
auto runPageRankDynamicNvgraph(G& x, bool all, C& r1, int b, PageRankUpdateFlags f, PageRankUpdateMode m) {
  // typedef PageRankUpdateFlags Flags;
  // typedef PageRankUpdateMode  Mode;
  // auto a  = copy(x); float t;
  // int deg = int(x.size()/x.order());
  // for (int i=0; i<b; i++) { switch(m) {
  //   case Mode::RANDOM: removeRandomEdge(a); break;
  //   case Mode::DEGREE: removeRandomEdgeByDegree(a); break;
  //   case Mode::RANK:   removeRandomEdgeByRank(a, r1); break;
  // } }
  // print(a);  printf(" (batch: %d; flags: %s; mode: %s)\n", b, stringify(f).c_str(), stringify(m).c_str());
  // auto an = transposeForNvgraph(a);
  // print(an); printf(" (batch: %d; flags: %s; mode: %s; transposeForNvgraph)\n", b, stringify(f).c_str(), stringify(m).c_str());
  // auto r2 = pageRankNvgraph(t, an);
  // printf("[%09.3f ms; %03dR] [%.4e] pageRankNvgraph\n", t, 0, absError(r2, r2)); if (all) println(r2);
  // auto r3 = pageRankNvgraph(t, an, &r1);
  // printf("[%09.3f ms; %03dR] [%.4e] pageRankNvgraphDynamic\n", t, 0, absError(r2, r3)); if (all) println(r3);
}


template <class G>
void runPageRankDynamic(G& x, bool all) {
  typedef PageRankUpdateFlags Flags;
  typedef PageRankUpdateMode  Mode;
  int BATCH_BEGIN  = 10, BATCH_END = int(0.1*x.size());
  int BATCH_REPEAT = 10; float t;
  auto xn = transposeForNvgraph(x); print(xn); printf(" (transposeForNvgraph)\n");
  auto r1 = pageRankNvgraph(t, xn);
  printf("[%09.3f ms; %03dR] [%.4e] pageRankNvgraph\n", t, 0, absError(r1, r1)); if (all) println(r1);
  for (int b=BATCH_BEGIN; b<BATCH_END; b*=10) {
    for (int r=0; r<BATCH_REPEAT; r++) {
      for (int f=0; f<1; f++) {
        for (int m=0; m<3; m++)
          // runPageRankDynamicNvgraph(x, all, r1, b, Flags(f), static_cast<Mode>(m));
      }
    }
  }
}


int main(int argc, char **argv) {
  char *file = argv[1];
  bool all = argc > 2;
  setupAll(); testAll();
  printf("Loading graph %s ...\n", file);
  auto x = readMtx(file); println(x);
  runPageRankDynamic(x, all);
  printf("\n");
  return 0;
}
