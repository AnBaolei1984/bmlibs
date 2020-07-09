#ifndef BMCOMPILER_IF_UNIFIED_DATATYPE_H_
#define BMCOMPILER_IF_UNIFIED_DATATYPE_H_
#include "bmcompiler_defs.h"
#include <vector>
#ifdef __cplusplus
extern "C" {
#endif

using std::vector;

typedef struct DataInfo{
    char*    name;
    int*     shape;
    int      dim;
    int      is_coeff;
    void*    data;
    int      scale;
    int      rshift_num;
    DATA_TYPE_T data_type;
} DataInfo;

typedef struct BmConvParam {
    char*  layer_name;
    int    kh;
    int    kw;
    int    pad_h_t;
    int    pad_h_b;
    int    pad_w_l;
    int    pad_w_r;
    int    stride_h;
    int    stride_w;
    int    dh;
    int    dw;
    bool   have_bias;
    int    groups;
    bool   use_winograd;
    float  scale;
    int    mode;
    BmQuantizeInfo quantize_info;
} BmConvParam;

typedef struct BmFcParam {
  int has_bias;
  int weight_col_is_in_neruon_num;
  int mode; //0: LowPrecision, 1: HighPrecisionSymmtric, 2: HighPrecisionAsymmetric
  BmQuantizeInfo quantize_info;
} BmFcParam;

typedef struct BmShapeFetchParam {
  int begin_axis;
  int end_axis;
  int step;
} BmShapeFetchParam;


void add_select_layer_unified(
    void*   p_bmcpl,
    const   DataInfo cond_data,
    const   DataInfo then_data,
    const   DataInfo else_data,
    const   DataInfo out_data
);

void add_broadcast_binary_layer_unified(
    void*   p_bmcpl,
    const   DataInfo a_data,
    const   DataInfo b_data,
    const   DataInfo out_data,
    int     binary_op  /* BINARY_ADD, BINARY_MUL, BINARY_MAX, BINARY_SUB, BINARY_MIN, BINARY_DIV */
    );

void add_eltwise_binary_layer_unified(
    void*   p_bmcpl,
    const   DataInfo a_data,
    const   DataInfo b_data,
    const   DataInfo out_data,
    int     binary_op  /* BINARY_ADD, BINARY_MUL, BINARY_MAX, BINARY_SUB, BINARY_MIN, BINARY_DIV */
    );

void add_const_binary_layer_unified(
    void*   p_bmcpl,
    const   DataInfo a_data,
    const   DataInfo b_data,
    const   DataInfo out_data,
    int     inversed,    /* 0: A op B, 1: B op A, for BINARY_SUB usually */
    int     binary_op  /* BINARY_ADD, BINARY_MUL, BINARY_MAX, BINARY_SUB, BINARY_MIN, BINARY_DIV */
    );

void add_binary_layer_unified(
    void*   p_bmcpl,
    const   DataInfo A_data,
    const   DataInfo B_data,
    const   DataInfo out_data,
    int     binary_op  /* BINARY_ADD, BINARY_MUL, BINARY_MAX, BINARY_SUB, BINARY_MIN, BINARY_DIV */
    );

void add_active_layer_unified(
    void*   p_bmcpl,
    const   DataInfo a_data,
    const   DataInfo out_data,
    int     active_type_id
);

void add_relu_layer_unified(
    void*   p_bmcpl,
    const   DataInfo a_data,
    const   DataInfo out_data,
    float 	negative_slope,
    float 	upper_limit
    );


void add_stride_slice_layer_unified(
    void*   p_bmcpl,
    const   DataInfo inp_data,
    const   DataInfo out_data,
    int     begin_mask,
    int     end_mask,
    const   int* begin_index,
    const   int* end_index,
    const   int* stride
    );

void add_conv_layer_unified(
    void*   p_bmcpl,
    const DataInfo in_data,
    const DataInfo out_data,
    const DataInfo weight_data,
    const DataInfo bias_data,
    BmConvParam param
);

void add_upsample_layer_unified(
    void*   p_bmcpl,
    const   DataInfo inp_data,
    const   DataInfo out_data,
    int     scale
    );

void add_arg_layer_unified(
    void*   p_bmcpl,
    const   DataInfo inp_data,
    const   DataInfo out_data,
    int     axis,
    int     method
);

void add_deconv_layer_unified(
    void*   p_bmcpl,
    const DataInfo in_data,
    const DataInfo out_data,
    const DataInfo weight_data,
    const DataInfo bias_data,
    BmConvParam param
);

void add_permute_layer_unified(
    void*   p_bmcpl,
    const DataInfo a_data,
    const DataInfo out_data,
    int*    permute_order
);

void add_transpose_layer_unified(
  void*          p_bmcpl,
  const DataInfo a_data,
  const DataInfo out_data,
  const int*     order
);

void add_tile_layer_unified(
  void*          p_bmcpl,
  const DataInfo a_data,
  const DataInfo out_data,
  int            coeff_is_fixed,
  const char     *coeff_name,
  const int *    tile_coeff
);

void add_concat_layer_unified(
    void*  p_bmcpl,
    const  DataInfo* in_data,
    const  DataInfo out_data,
    int    input_num,
    int    axis);

void add_reduce_full_layer_unified(
    void*   p_bmcpl,
    const   DataInfo inp_data,
    const   DataInfo out_data,
    int     need_keep_dim,
    int     axis_num,
    const   int* axis_list,
    int     method);

void add_reshape_layer_unified(
    void*   p_bmcpl,
    const   DataInfo inp_data,
    const   DataInfo out_data,
    const   int* raw_param);

void add_channel_shift_layer_unified(
    void*   	p_bmcpl,
    const DataInfo a_data,
    const DataInfo out_data,
    int     	shift_dir,
    int     	shift_num
);


void add_interleave_layer_unified(
    void*   p_bmcpl,
    const   vector<DataInfo>& inp_data,
    const   DataInfo out_data,
    int     axis,
    int     step);

void add_fc_layer_unified(
        void * p_bmcpl,
        const DataInfo A_data,
        const DataInfo B_data,
        const DataInfo out_data,
        const DataInfo bias_data,
        const BmFcParam& fc_param,
        int axis = 1 
        );

void add_split_layer_unified(
    void* p_bmcpl,
    const DataInfo in_data,
    const DataInfo* out_data,
    int out_num,
    int shape_dim,
    int axis,
    const int* split_size,
    int split_num
);

void add_pad_layer_unified(
    void* p_bmcpl,
    const DataInfo in_data,
    const DataInfo out_data,
    const int* padding,
    float pad_value,
    int pad_mode
);
void add_topk_layer_unified(
    void*  p_bmcpl,
    const  DataInfo in_data,
    const  DataInfo value_data,
    const  DataInfo index_data,
    int    k,
    int    dim);


void add_stridecal_layer_unified(
    void*   p_bmcpl,
    int     inp_num,
    const   DataInfo A_data,
    const   DataInfo B_data,
    const   DataInfo R_data,
    int     op,
    const   int* stride,
    const   int* offset,
    int     result_add,
    const   int* B_axis_is_1,
    int     A_is_const,
    int     B_is_const,
    float   A_const_val,
    float   B_const_val);
void add_bitwise_layer_unified(
    void*   p_bmcpl,
    const   DataInfo A_data,
    const   DataInfo B_data,
    const   DataInfo out_data,
    int     binary_op
);
void add_const_bitwise_layer_unified(
    void*   p_bmcpl,
    const   DataInfo a_data,
    const   DataInfo b_data,
    const   DataInfo out_data,
    int     binary_op
);

void add_elementwise_shift_layer_unified(
    void*             p_bmcompiler,
    const DataInfo    in_data,
    const DataInfo    shift_data,
    const DataInfo    out_data,
    int               shiftType,
    int               shift_num,
    int               shift_mode
);

void add_squeeze_layer_unified(
    void*   p_bmcpl,
    const   DataInfo in_data,
    const   DataInfo out_data,
    const int* squeeze_axis,
    const int  axis_num);

void add_pooling_layer_unified(
    void*   p_bmcpl,
    const   DataInfo inp_data,
    const   vector<DataInfo>& out_data,
    int     kh,
    int     kw,
    int     up_pad_h,
    int     down_pad_h,
    int     left_pad_w,
    int     right_pad_w,
    int     stride_h,
    int     stride_w,
    int     is_avg_pooling,
    int     avg_pooling_mode,
    int     is_global_pooling,
    int     out_ceil_mode,
    const   char* coeff_mask
    );

void add_masked_select_layer_unified(
    void*   p_bmcpl,
    const   DataInfo& in_data,
    const   DataInfo& mask_data,
    const   DataInfo& out_data,
    bool    bcast_from_begin = false /*e.g. [2,1,3] and [2,3], if true, => [2,3,3]; if false, => [2,2,3] */
);

void add_sort_layer_unified(
    void* p_bmcpl,
    const DataInfo in_data,
    const DataInfo index_data,
    const DataInfo out_data,
    int   dim = -1,
    bool  stable = false,
    bool  descending = false,
    bool  is_argsort = false
);

void add_argsort_layer_unified(
    void* p_bmcpl,
    const DataInfo in_data,
    const DataInfo index_data,
    int   dim = -1,
    bool  stable = false,
    bool  descending = false
);

void add_index_select_layer_unified(
    void* p_bmcpl,
    const DataInfo in_data,
    const DataInfo out_data,
    const DataInfo index_data,
    int   dim
);

void add_nms_layer_unified(
    void* p_bmcpl,
    const DataInfo box_data,
    const DataInfo out_data,
    const DataInfo score_data,
    float threshold,
    bool  stable
);

void add_lut_layer_unified(
    void* p_bmcpl,
    const DataInfo in_data,
    const DataInfo table_data,
    const DataInfo out_data);

void add_const_data_layer_unified(
    void* p_bmcpl,
    const DataInfo in_data);

void add_shape_fetch_layer_unified(
    void* p_bmcpl,
    const DataInfo in_data,
    const DataInfo out_data,
    const BmShapeFetchParam &param);

void add_where_layer_unified(
    void* p_bmcpl,
    const DataInfo in_data,
    const DataInfo out_data);

void add_expand_dims_layer_unified(
    void* p_bmcpl,
    const DataInfo in_data,
    const DataInfo out_data,
    int axis, int ndims);

void add_space_to_batch_layer_unified(
  void*          p_bmcpl,
  const DataInfo in_data,
  const DataInfo out_data,
  const int*  block_sizes,
  const int*  pad_sizes);

void add_batch_to_space_layer_unified(
  void*          p_bmcpl,
  const DataInfo in_data,
  const DataInfo out_data,
  const int*  block_sizes,
  const int*  crop_sizes);

#ifdef __cplusplus
}
#endif

#endif
