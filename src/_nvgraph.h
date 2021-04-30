#pragma once
#include <stdio.h>
#include <nvgraph.h>




#ifndef TRY_NVGRAPH
void try_nvgraph(nvgraphStatus_t err, const char* exp, const char* func, int line, const char* file) {
  if (err == NVGRAPH_STATUS_SUCCESS) return;
  fprintf(stderr,
      "ERROR: nvGraph-%d\n"
      "  in expression %s\n"
      "  at %s:%d in %s\n",
      err,
      exp,
      func, line, file);
  exit(err);
}

// Prints an error message and exits, if nvGraph expression fails.
// TRY_NVGRAPH( nvgraphCreate(&h) );
#define TRY_NVGRAPH(exp) try_nvgraph(exp, #exp, __func__, __LINE__, __FILE__)
#endif
