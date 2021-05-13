#ifndef BMCOMPILER_OP_CODE_H
#define BMCOMPILER_OP_CODE_H

namespace bmcompiler {

typedef enum {
  REDUCE_MEAN = 0,
  REDUCE_SUM = 1,
  REDUCE_MAX = 2,
  REDUCE_MIN = 3,
  REDUCE_PROD = 4,
  REDUCE_ALL = 5,
  REDUCE_ANY = 6
} BmReduceType;

typedef enum {
  BINARY_ADD          = 0,
  BINARY_SUB          = 1,
  BINARY_MUL          = 2,
  BINARY_DIV          = 3,
  BINARY_MAX          = 4,
  BINARY_MIN          = 10000,
  BINARY_GT           = 10001,
  BINARY_GE           = 10002,
  BINARY_LT           = 10003,
  BINARY_LE           = 10004,
  BINARY_EQ           = 10005,
  BINARY_NE           = 10006,
  BINARY_SQUARED_DIFF = 10007,
  BINARY_FLOOR_MOD    = 10008,
  BINARY_FLOOR_DIV    = 10009
} BmBinaryType;

typedef enum {
  ACTIVE_TANH      = 0,
  ACTIVE_SIGMOID   = 1,
  ACTIVE_RELU      = 2,
  ACTIVE_EXP       = 3,
  ACTIVE_ELU       = 4,
  ACTIVE_SQRT      = 5,
  ACTIVE_SQUARE    = 6,
  ACTIVE_RSQRT     = 7,
  ACTIVE_ABSVAL    = 8,
  ACTIVE_LN        = 9,
  ACTIVE_ROUND     = 10,
  ACTIVE_CEIL      = 11,
  ACTIVE_FLOOR     = 12,
  ACTIVE_SIN       = 13,
  ACTIVE_COS       = 14,
  ACTIVE_IS_FINITE = 15,
  ACTIVE_MISH      = 16,
  ACTIVE_SWISH     = 17,
  ACTIVE_HSWISH    = 18,
  ACTIVE_SILU      = 19
} BmActiveType;

typedef enum {
  STRIDE_ADD = 0,
  STRIDE_SUB = 1,
  STRIDE_MUL = 2,
  STRIDE_DIV = 3,
  STRIDE_MAX = 4,
  STRIDE_CPY = 5,
} BmStrideCalculateType;

} // namespace bmcompiler

#endif
