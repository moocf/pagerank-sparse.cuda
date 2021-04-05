#pragma once
#include <string>
#include "pageRank.h"

using std::string;




// Prints page rank flags.
auto stringify(PageRankFlags& x) {
  string a = "{";
  a += x.splitComponents?  'S':' ';
  a += x.orderComponents?  'T':' ';
  a += x.removeIdenticals? 'I':' ';
  a += x.removeChains?     'C':' ';
  a += x.skipConverged?    'D':' ';
  a += "}";
  return a;
}
