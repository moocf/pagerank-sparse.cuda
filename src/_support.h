#pragma once




#ifndef UINT
typedef unsigned int uint;
#define UINT uint
#endif

#ifndef UINT8
typedef unsigned char uint8;
#define UINT8 uint8
#endif




#ifndef GET2D
// Gets value at given row, column of 2D array
#define GET2D(x, r, c, C) (x)[(C)*(r) + (c)]
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
