#pragma once




#ifndef UINT
typedef unsigned int uint;
#define UINT uint
#endif

#ifndef UINT8
typedef unsigned char uint8;
#define UINT8 uint8
#endif

#ifndef NONE
struct None {
  friend bool operator==(None a, None b) noexcept { return true; }

  template <class T>
  friend bool operator==(None a, const T& b) noexcept { return false; }

  template <class T>
  friend bool operator==(const T& a, None b) noexcept { return false; }
};
#define NONE None
#endif

#ifndef IDENTITY
struct Identity {
  template <class T>
  constexpr T operator()(T x) const noexcept { return x; }
};
#define IDENTITY Identity()
#endif




#ifndef GET2D
// Gets value at given row, column of 2D array
#define GET2D(x, r, c, C) (x)[(C)*(r) + (c)]
#endif




#ifndef ITERATOR_DEREF
#define ITERATOR_DEREF(It, se, be, ae) \
  reference operator*() { return se; } \
  reference operator[](difference_type i) { return be; } \
  pointer operator->() { return ae; }
#endif

#ifndef ITERATOR_NEXT
#define ITERATOR_NEXTP(It, ie)  \
  It& operator++() { ie; return *this; }  \
  It operator++(int) { It a = *this; ++(*this); return a; }
#define ITERATOR_NEXTN(It, de) \
  It& operator--() { de; return *this; }  \
  It operator--(int) { It a = *this; --(*this); return a; }
#define ITERATOR_NEXT(It, ie, de) \
  ITERATOR_NEXTP(It, ie) \
  ITERATOR_NEXTN(It, de)
#endif

#ifndef ITERATOR_ADVANCE
#define ITERATOR_ADVANCEP(It, i, fe) \
  It& operator+=(difference_type i) { fe; return *this; }
#define ITERATOR_ADVANCEN(It, i, be) \
  It& operator-=(difference_type i) { be; return *this; }
#define ITERATOR_ADVANCE(It, i, fe, be) \
  ITERATOR_ADVANCEP(It, i, fe) \
  ITERATOR_ADVANCEN(It, i, be)
#endif

#ifndef ITERATOR_ARITHMETICP
#define ITERATOR_ARITHMETICP(It, a, b, ...)  \
  friend It operator+(const It& a, difference_type b) { return It(__VA_ARGS__); } \
  friend It operator+(difference_type b, const It& a) { return It(__VA_ARGS__); }
#endif

#ifndef ITERATOR_ARITHMETICN
#define ITERATOR_ARITHMETICN(It, a, b, ...) \
  friend It operator-(const It& a, difference_type b) { return It(__VA_ARGS__); } \
  friend It operator-(difference_type b, const It& a) { return It(__VA_ARGS__); }
#endif

#ifndef ITERATOR_COMPARISION
#define ITERATOR_COMPARISION(It, a, b, ae, be)  \
  friend bool operator==(const It& a, const It& b) { return ae == be; } \
  friend bool operator!=(const It& a, const It& b) { return ae != be; } \
  friend bool operator>=(const It& a, const It& b) { return ae >= be; } \
  friend bool operator<=(const It& a, const It& b) { return ae <= be; } \
  friend bool operator>(const It& a, const It& b) { return ae > be; } \
  friend bool operator<(const It& a, const It& b) { return ae < be; }
#endif

#ifndef ITERABLE_SIZE
#define ITERABLE_SIZE(se) \
  size_t size() { return se; } \
  bool empty() { return size() == 0; }
#endif




// Gets nth argument.
#define ARG_GET0(V, ...) V
#define ARG_GET1(_0, V, ...) V
#define ARG_GET2(_0, _1, V, ...) V
#define ARG_GET3(_0, _1, _2, V, ...) V
#define ARG_GET4(_0, _1, _2, _3, V, ...) V
#define ARG_GET5(_0, _1, _2, _3, _4, V, ...) V
#define ARG_GET6(_0, _1, _2, _3, _4, _5, V, ...) V
#define ARG_GET7(_0, _1, _2, _3, _4, _5, _6, V, ...) V
#define ARG_GET8(_0, _1, _2, _3, _4, _5, _6, _7, V, ...) V
#define ARG_GET9(_0, _1, _2, _3, _4, _5, _6, _7, _8, V, ...) V
#define ARG_GET10(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, V, ...) V
#define ARG_GET11(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, V, ...) V
#define ARG_GET12(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, V, ...) V
#define ARG_GET13(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, V, ...) V
#define ARG_GET14(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, V, ...) V
#define ARG_GET15(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, V, ...) V
#define ARG_GET16(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, V, ...) V




// Calls a function each argument.
#define ARG_CALL0(F,  ...) {}
#define ARG_CALL1(F, _0, ...) { F(_0); }
#define ARG_CALL2(F, _0, _1, ...) { F(_0); F(_1); }
#define ARG_CALL3(F, _0, _1, _2, ...) { F(_0); F(_1); F(_2); }
#define ARG_CALL4(F, _0, _1, _2, _3, ...) { F(_0); F(_1); F(_2); F(_3); }
#define ARG_CALL5(F, _0, _1, _2, _3, _4, ...) { F(_0); F(_1); F(_2); F(_3); F(_4); }
#define ARG_CALL6(F, _0, _1, _2, _3, _4, _5, ...) { F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); }
#define ARG_CALL7(F, _0, _1, _2, _3, _4, _5, _6, ...) {  F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); }
#define ARG_CALL8(F, _0, _1, _2, _3, _4, _5, _6, _7, ...) { F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); }
#define ARG_CALL9(F, _0, _1, _2, _3, _4, _5, _6, _7, _8, ...) { F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); }
#define ARG_CALL10(F, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, ...) { F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); }
#define ARG_CALL11(F, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, ...) { F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); }
#define ARG_CALL12(F, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, ...) { F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); }
#define ARG_CALL13(F, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, ...) { F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); }
#define ARG_CALL14(F, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, ...) { F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); }
#define ARG_CALL15(F, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, ...) { F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); F(_0); }
#define ARG_CALL(F, ...) ARG_GET0(ARG_GET16(0, ##__VA_ARGS__, ARG_CALL15, ARG_CALL14, ARG_CALL13, ARG_CALL12, ARG_CALL11, ARG_CALL10, ARG_CALL9, ARG_CALL8, ARG_CALL7, ARG_CALL6, ARG_CALL5, ARG_CALL4, ARG_CALL3, ARG_CALL2, ARG_CALL1, ARG_CALL0)(F, ##__VA_ARGS__))
