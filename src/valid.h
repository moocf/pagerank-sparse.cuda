#pragma once
#include "pageRank.h"




bool isValid(PageRankFlags x) {
  return !(x.largeComponents || x.orderComponents) || x.splitComponents;
}


bool isValidStepped(PageRankFlags x) {
  return x.splitComponents && x.largeComponents && x.orderComponents;
}
