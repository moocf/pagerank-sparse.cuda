#pragma once
#include "pageRank.h"




bool isSilly(PageRankFlags x) {
  return !x.splitComponents && (x.largeComponents || x.orderComponents);
}
