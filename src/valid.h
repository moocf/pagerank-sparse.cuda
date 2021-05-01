#pragma once
#include "pageRank.h"




bool isValid(PageRankFlags x) {
  if ((x.largeComponents || x.orderComponents) && !x.splitComponents) return false;
  if (x.crossPropagate) return false;
  return true;
}


bool isValidSwitched(PageRankFlags x) {
  if (x.splitComponents && !x.largeComponents) return false;
  return true;
}


bool isValidStepped(PageRankFlags x) {
  if (!(x.splitComponents && x.largeComponents && x.orderComponents)) return false;
  return true;
}
