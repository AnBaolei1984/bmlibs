#ifndef __BMCOMPILER_DEFS_H__
#define __BMCOMPILER_DEFS_H__

#define ARCH_BM1682 "BM1682"
#define ARCH_BM1684 "BM1684"

typedef unsigned long long u64;

typedef enum {
  DTYPE_FP32 = 0,
  DTYPE_FP16 = 1,
  DTYPE_INT8 = 2,
  DTYPE_UINT8 = 3,
  DTYPE_INT16 = 4,
  DTYPE_UINT16 = 5,
  DTYPE_INT32 = 6,
  DTYPE_UINT32 = 7,
  DTYPE_UNKNOWN = -1,
} bm_data_type_t;
typedef bm_data_type_t DATA_TYPE_T;

typedef enum {
    LAYER_FORMAT_NCHW = 0,
    LAYER_FORMAT_NHWC = 1
} bm_layer_format_t;

typedef enum {
    NORMAL_TENSOR = 0,
    CONST_TENSOR = 1,
    SHAPE_TENSOR = 2,
    FLOW_TENSOR = 3,
    OTHER_TENSOR = -1
} bm_tensor_type_t;

typedef enum {
    DUMP_NOTHING = 0,
    DUMP_INOUT = 1,
    DUMP_REF = 2
} bm_net_dump_t;

typedef struct {
    int dynamic;                 // dynamic compile
    int compare;                 // compare after compile
    int optimize_level;          // optimize level
    int enable_profile;          // generate profile information
    int enable_multistage;       // compile with multistage
    int enable_winograd;         // use winograd conv
    int use_bmlang;              // show bmlang info
    int no_same_tensor_name;     // this can allow to add loop in graph
    bm_net_dump_t dump_level;    // dump nothing/inout/ref
    u64 ta_global_size;          // the max size of tensorarray global memory
} bm_compiler_config_t;

typedef struct {
    bm_tensor_type_t ttype;
    bm_data_type_t dtype;
    const int *shape;
    int dims;
} bm_user_tensor_t;

typedef struct {
    int id;
    int type;
    int input_num;
    char** input_names;
    int output_num;
    char** output_names;
} bm_subnet_info_t;

typedef struct {
  int size;
  float* data;
} BmFloatArray;

typedef struct {
  int size;
  int* data;
} BmIntArray;

typedef struct {
  int scale_num;
  BmFloatArray* scale;
  int zero_point_num;
  BmIntArray* zero_point;
} BmQuantizeInfo;

#endif
