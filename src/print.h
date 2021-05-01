#include <iostream>
#include "write.h"

using std::cout;




// PRINT
// -----

template <class T>
void print(T *x, int N) { write(cout, x, N); }

template <class T>
void print(vector<T>& x) { write(cout, x); }

template <class K, class T>
void print(ostream& a, unordered_map<K, T>& x) { write(cout, x); }


template <class T>
void print(T *x, int R, int C) { write(cout, x, R, C); }

template <class T>
void print(vector<vector<T>>& x) { write(cout, x); }


template <class G>
void print(G& x, bool all=false) { write(cout, x, all); }




// PRINTLN
// -------

template <class T>
void println(T *x, int N) { print(x, N); cout << "\n"; }

template <class T>
void println(vector<T>& x) { print(x); cout << "\n"; }

template <class K, class T>
void println(unordered_map<K, T>& x) { print(x); cout << "\n"; }


template <class T>
void println(T *x, int R, int C) { print(x, R, C); cout << "\n"; }

template <class T>
void println(vector<vector<T>>& x) { print(x); cout << "\n"; }


template <class G>
void println(G& x, bool all=false) { print(x, all); cout << "\n"; }
