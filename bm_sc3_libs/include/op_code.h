#ifndef OP_CODE_H_
#define OP_CODE_H_

typedef enum tensor_arith_op {
    TENSOR_ADD,
    TENSOR_SUB,
    TENSOR_MUL,
    TENSOR_DIV,
    TENSOR_MAX,
    TENSOR_CPY
} TENSOR_ARITH_OP;

typedef enum align_tensor_op {
    ALIGN_TENSOR_ADD,
    ALIGN_TENSOR_SUB,
    ALIGN_TENSOR_MUL,
    ALIGN_TENSOR_DIV,
    TENSOR_INVALID
} ALIGN_TENSOR_OP;

typedef enum linear_op {
    LINEAR_MAC,
    LINEAR_ADD_SQR,
    LINEAR_SUB_SQR
} LINEAR_OP;

typedef enum sfu_op {
    SFU_XN,
    SFU_EX,
    SFU_LNX,
    SFU_RSQ,
    SFU_INVALID
} SFU_OP;

typedef enum pad_op {
    CONSTANT,
    REFLECT,
    SYMMETRIC,
    EDGE
} PAD_OP;

typedef struct tensor_4d_t {
    int n;
    int c;
    int h;
    int w;
}bm_tensor_4d_t;


#define TENSOR_ADD 0
#define TENSOR_SUB 1
#define TENSOR_MUL 2
//Note the div should be implmented by KAMAKE algorithm
#define TENSOR_DIV 3
#define TENSOR_MAX 4
#define TENSOR_CPY 5
#define TENSOR_MAC 6

#define TENSOR_N_DIM 0
#define TENSOR_C_DIM 1
#define TENSOR_H_DIM 2
#define TENSOR_W_DIM 3

#define SHARE_REG_MESSAGE_WP            0
#define SHARE_REG_MESSAGE_RP            1
#define SHARE_REG_MESSAGE_IRQSTATUS     2
#define SHARE_REG_CDMA_IRQSTATUS    3 
#define SHARE_REG_MSGIRQ_NUM_LO     4
#define SHARE_REG_MSGIRQ_NUM_HI     5
#define SHARE_REG_API_PROCESS_TIME  6
#define SHARE_REG_MISC_INFO         7
#define SHARE_REG_FW_STATUS         9
#define SHARE_REG_ATTRIBUTE         10

#define SHAREMEM_MSG_FIXED_OFFSET  (8192)
#define SHAREMEM_SIZE_BIT  10
#define SHAREMEM_MASK      ((1<<SHAREMEM_SIZE_BIT) - 1)
#define SHARE_REG_CNT      16

#define IRQ_STATUS_CDMA_INT             0x1111
#define IRQ_STATUS_MSG_DONE_INT         0x2222

#define REDUCE_MEAN 0
#define REDUCE_SUM  1
#define REDUCE_MAX  2
#define REDUCE_MIN  3
#define REDUCE_PROD 4

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
#define BINARY_FLOOR_MOD 10008
#define BINARY_FLOOR_DIV 10009

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
#define ACTIVE_ROUND    10  //only for float shape tensor
#define ACTIVE_CEIL     11  //only for float shape tensor
#define ACTIVE_FLOOR    12  //only for float shape tensor

// Channel shift macro(left,right,circle left,circle right)
#define CH_SHIFT_L      0
#define CH_SHIFT_R      1
#define CH_SHIFT_CL     2
#define CH_SHIFT_CR     3

#endif /* OP_CODE_H_ */
