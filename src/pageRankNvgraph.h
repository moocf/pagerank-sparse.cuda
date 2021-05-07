#pragma once
#include <vector>
#include "_nvgraph.h"
#include "vertices.h"
#include "edges.h"

using std::vector;




template <class G>
auto pageRankNvgraph(float& t, G& x, vector<float> *iranks=nullptr, float p=0.85f, float E=1e-6f) {
  nvgraphHandle_t     h;
  nvgraphGraphDescr_t g;
  struct nvgraphCSCTopology32I_st csc;
  vector<cudaDataType_t> vtype {CUDA_R_32F, CUDA_R_32F};
  vector<cudaDataType_t> etype {CUDA_R_32F};
  vector<float> ranks(x.order());
  if (iranks) ranks = *iranks;
  auto ks    = vertices(x);
  auto vfrom = sourceOffsets(x);
  auto efrom = destinationIndices(x);
  auto vdata = vertexData(x);
  auto edata = edgeData(x);

  TRY_NVGRAPH( nvgraphCreate(&h) );
  TRY_NVGRAPH( nvgraphCreateGraphDescr(h, &g) );

  csc.nvertices = x.order();
  csc.nedges    = x.size();
  csc.destination_offsets = vfrom.data();
  csc.source_indices      = efrom.data();
  TRY_NVGRAPH( nvgraphSetGraphStructure(h, g, &csc, NVGRAPH_CSC_32) );

  TRY_NVGRAPH( nvgraphAllocateVertexData(h, g, vtype.size(), vtype.data()) );
  TRY_NVGRAPH( nvgraphAllocateEdgeData  (h, g, etype.size(), etype.data()) );
  TRY_NVGRAPH( nvgraphSetVertexData(h, g, vdata.data(), 0) );
  TRY_NVGRAPH( nvgraphSetVertexData(h, g, ranks.data(), 1) );
  TRY_NVGRAPH( nvgraphSetEdgeData  (h, g, edata.data(), 0) );

  t = measureDuration([&]() { TRY_NVGRAPH( nvgraphPagerank(h, g, 0, &p, 0, !!iranks, 1, E, 0) ); });
  TRY_NVGRAPH( nvgraphGetVertexData(h, g, ranks.data(), 1) );

  TRY_NVGRAPH( nvgraphDestroyGraphDescr(h, g) );
  TRY_NVGRAPH( nvgraphDestroy(h) );
  return vertexContainer(x, ranks, ks);
}
