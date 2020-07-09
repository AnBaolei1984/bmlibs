#ifndef BMCOMPILER_OP_CODE_H
#define BMCOMPILER_OP_CODE_H

#define REDUCE_MEAN 0
#define REDUCE_SUM  1
#define REDUCE_MAX  2
#define REDUCE_MIN  3
#define REDUCE_PROD 4
#define REDUCE_ALL 5
#define REDUCE_ANY 6

#define BINARY_ADD 0
#define BINARY_SUB 1
#define BINARY_MUL 2
#define BINARY_DIV 3
#define BINARY_MAX 4
#define BINARY_MIN 10000

#define BINARY_GT 10001
#define BINARY_GE 10002
#define BINARY_LT 10003
#define BINARY_LE 10004
#define BINARY_EQ 10005
#define BINARY_NE 10006
#define BINARY_SQUARED_DIFF 10007
#define BINARY_FLOOR_MOD 10008 //only for shape tensor
#define BINARY_FLOOR_DIV 10009 //only for shape tensor

#define ACTIVE_TANH     0
#define ACTIVE_SIGMOID  1
#define ACTIVE_RELU     2
#define ACTIVE_EXP      3
#define ACTIVE_ELU      4
#define ACTIVE_SQRT     5
#define ACTIVE_SQUARE   6
#define ACTIVE_RSQRT    7
#define ACTIVE_ABSVAL   8
#define ACTIVE_LN       9
#define ACTIVE_ROUND    10  //only for shape tensor
#define ACTIVE_CEIL     11  //only for shape tensor
#define ACTIVE_FLOOR    12  //only for shape tensor
#define ACTIVE_SIN      13
#define ACTIVE_COS      14
#define ACTIVE_IS_FINITE 15
#define ACTIVE_MISH     16

#endif
