#pragma once
#include "src/main.h"

using namespace std;

DiGraph<> MIN2C;
DiGraph<> MIN2D;
DiGraph<> MIN4C;
DiGraph<> MIN5C;
DiGraph<> MINNV;




void setupGraphMin2c() {
  auto& x = MIN2C;
  x.addEdge(1, 2);
  x.addEdge(2, 4);
  x.addEdge(4, 3);
  x.addEdge(3, 1);
  x.addEdge(2, 5);
  x.addEdge(4, 5);
  x.addEdge(4, 7);
  x.addEdge(5, 6);
  x.addEdge(6, 8);
  x.addEdge(8, 7);
  x.addEdge(7, 5);
  x.addEdge(5, 8);
}

void setupGraphMin2d() {
  auto& x = MIN2D;
  x.addEdge(1, 2);
  x.addEdge(1, 4);
  x.addEdge(1, 5);
  x.addEdge(2, 3);
  x.addEdge(4, 5);
  x.addEdge(5, 3);
}

void setupGraphMin4c() {
  auto& x = MIN4C;
  x.addEdge(1, 2);
  x.addEdge(2, 3);
  x.addEdge(1, 3);
  x.addEdge(1, 4);
  x.addEdge(3, 4);
  x.addEdge(4, 5);
  x.addEdge(5, 1);
  x.addEdge(6, 7);
  x.addEdge(7, 8);
  x.addEdge(8, 6);
  x.addEdge(8, 9);
  x.addEdge(9, 6);
  x.addEdge(10, 11);
  x.addEdge(11, 12);
  x.addEdge(11, 13);
  x.addEdge(12, 13);
  x.addEdge(13, 10);
  x.addEdge(21, 14);
  x.addEdge(14, 15);
  x.addEdge(15, 16);
  x.addEdge(15, 20);
  x.addEdge(16, 17);
  x.addEdge(16, 20);
  x.addEdge(17, 18);
  x.addEdge(18, 19);
  x.addEdge(19, 17);
  x.addEdge(19, 20);
  x.addEdge(20, 21);
  x.addEdge(3, 6);
  x.addEdge(3, 9);
  x.addEdge(4, 21);
  x.addEdge(4, 10);
  x.addEdge(5, 13);
  x.addEdge(11, 21);
  x.addEdge(12, 20);
}

void setupGraphMin5c() {
  auto& x = MIN5C;
  x.addEdge(1, 2);
  x.addEdge(1, 3);
  x.addEdge(3, 4);
  x.addEdge(2, 4);
  x.addEdge(4, 1);
  x.addEdge(2, 5);
  x.addEdge(5, 6);
  x.addEdge(6, 7);
  x.addEdge(7, 5);
  x.addEdge(4, 8);
  x.addEdge(6, 9);
  x.addEdge(8, 9);
  x.addEdge(9, 11);
  x.addEdge(11, 8);
  x.addEdge(11, 10);
  x.addEdge(10, 8);
  x.addEdge(7, 12);
  x.addEdge(12, 13);
  x.addEdge(13, 15);
  x.addEdge(15, 14);
  x.addEdge(13, 14);
  x.addEdge(12, 14);
  x.addEdge(14, 12);
  x.addEdge(9, 17);
  x.addEdge(11, 16);
  x.addEdge(17, 19);
  x.addEdge(19, 16);
  x.addEdge(16, 17);
  x.addEdge(16, 18);
  x.addEdge(18, 19);
}

void setupGraphMinNv() {
  auto& x = MINNV;
  x.addEdge(1, 2);
  x.addEdge(1, 3);
  x.addEdge(3, 1);
  x.addEdge(3, 2);
  x.addEdge(3, 5);
  x.addEdge(4, 5);
  x.addEdge(4, 6);
  x.addEdge(5, 4);
  x.addEdge(5, 6);
  x.addEdge(6, 4);
}


void setupGraphs() {
  setupGraphMin2c();
  setupGraphMin2d();
  setupGraphMin4c();
  setupGraphMin5c();
  setupGraphMinNv();
}


void setupAll() {
  setupGraphs();
}
