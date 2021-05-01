#pragma once
#include <string>
#include "write.h"
#include "pageRank.h"

using std::string;




const char* stringify(PageRankMode x) {
  typedef PageRankMode Mode;
  switch (x) {
    default:
    case Mode::BLOCK:    return "{B}";
    case Mode::THREAD:   return "{T}";
    case Mode::SWITCHED: return "{S}";
  }
}


auto stringify(PageRankFlags& x) {
  string a = "{";
  a += x.splitComponents?  'S':' ';
  a += x.largeComponents?  'L':' ';
  a += x.orderVertices?    'O':' ';
  a += x.orderComponents?  'T':' ';
  a += x.removeIdenticals? 'I':' ';
  a += x.removeChains?     'C':' ';
  a += x.skipConverged?    'D':' ';
  a += "}";
  return a;
}
