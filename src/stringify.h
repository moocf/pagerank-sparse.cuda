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
  a += x.orderComponents?  'T':' ';
  a += x.orderVertices?    'O':' ';
  a += x.crossPropagate?   'P':' ';
  a += x.removeIdenticals? 'I':' ';
  a += x.removeChains?     'C':' ';
  a += x.skipConverged?    'D':' ';
  a += "}";
  return a;
}




string stringify(PageRankUpdateMode x) {
  typedef PageRankUpdateMode Mode;
  switch (x) {
    default:
    case Mode::RANDOM: return "{?}";
    case Mode::DEGREE: return "{D}";
    case Mode::RANK:   return "{R}";
  }
}


string stringify(PageRankUpdateFlags& x) {
  string a = "{";
  a += x.addVertices?    'V':' ';
  a += x.removeVertices? 'v':' ';
  a += x.addEdges?       'E':' ';
  a += x.removeEdges?    'e':' ';
  a += "}";
  return a;
}
