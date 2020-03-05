#ifndef BMCOMPILER_IF_FIX8B_H_
#define BMCOMPILER_IF_FIX8B_H_

#ifdef __cplusplus
extern "C" {
#endif

void add_reorg_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int     stride,
    int     reverse,
    int     data_sign  //0: unsigned, 1: signed
  );

void add_priorbox_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    void*   output_data,
    int     in_sign,
    int     out_sign
  );

void add_permute_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int*    permute_order,
    int     data_sign
  );

void add_normalize_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    char*   layer_name,
    int     across_spatial,
    int     channel_shared,
    void*   scale_data,
    float   eps,
    int     in_sign,
    int     out_sign,
    int     coeff_sign
  );

void add_flatten_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int     data_sign
  );

void add_reshape_layer_fix8b(
    void*              p_bmcpl,
    int*               input_shape,
    int                input_shape_dim,
    char*              input_name,
    int*               output_shape,
    int                output_shape_dim,
    char*              output_name,
    const int*         raw_param,
    int                data_sign
  );

void add_active_layer_fix8b_gdma(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int     in_sign,
    int     out_sign,
    float   input_scale,
    float   output_scale,
    int     active_type_id
  );

void add_active_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int     in_sign,
    int     out_sign,
    float   input_scale,
    float   output_scale,
    int     active_type_id
  );

void add_conv_layer_fix8b(
    void*       p_bmcpl,
    const int*  input_shape,
    const int   input_shape_dim,
    const char* input_name,
    const int*  output_shape,
    int         output_shape_dim,
    const char* output_name,
    const char* layer_name,
    const void* weight,
    const void* bias,
    int     kh,
    int     kw,
    int     groups,
    int     pad_h_up,
    int     pad_h_down,
    int     pad_w_left,
    int     pad_w_right,
    int     stride_h,
    int     stride_w,
    int     dh,
    int     dw,
    int     have_bias,
    int     in_sign,
    int     out_sign,
    int     weight_sign,
    int     bias_sign,
    int     rshift_num,
    bool    use_winograd
 );

void add_conv_layer_fix8b_v2(
    void*       p_bmcpl,
    const int*  input_shape,
    const int   input_shape_dim,
    const char* input_name,
    const int*  output_shape,
    int         output_shape_dim,
    const char* output_name,
    const char* weight_name,
    const void* weight,
    const char* bias_name,
    const void* bias,
    int     kh,
    int     kw,
    int     groups,
    int     pad_h_up,
    int     pad_h_down,
    int     pad_w_left,
    int     pad_w_right,
    int     stride_h,
    int     stride_w,
    int     dh,
    int     dw,
    int     have_bias,
    int     in_dtype,
    int     out_dtype,
    int     weight_dtype,
    int     bias_dtype,
    int     rshift_num,
    bool    use_winograd
 );

void add_deconv_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    char*   layer_name,
    void*   weight,
    void*   bias,
    int     kh,
    int     kw,
    int     groups,
    int     pad_h_up,
    int     pad_h_down,
    int     pad_w_left,
    int     pad_w_right,
    int     stride_h,
    int     stride_w,
    int     dh,
    int     dw,
    int     have_bias,
    int     in_sign,
    int     out_sign,
    int     weight_sign,
    int     bias_sign,
    int     rshift_num
  );

void add_crop_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    char*   shape_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int*    offsets,
    int     data_sign
  );

void add_psroipooling_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    input_shape1,
    int     input_shape_dim1,
    char*   input_name1,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int     output_dim,
    int     group_size,
    float   spatial_scale_,
    int     roi_nums,
    int     in_sign,
    int     out_sign
 );

void add_roipooling_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    input_shape1,
    int     input_shape_dim1,
    char*   input_name1,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int     pooled_h,
    int     pooled_w,
    float   spatial_scale_,
    int     roi_nums,
    int     data_sign
);

void add_rpnproposal_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    input_shape1,
    int     input_shape_dim1,
    char*   input_name1,
    int*    input_shape2,
    int     input_shape_dim2,
    char*   input_name2,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int     feat_stride_,
    int     base_size_,
    int     min_size_,
    int     pre_nms_topN_,
    int     post_nms_topN_,
    float   nms_thresh_,
    float   score_thresh_,
    int     in_sign,
    float   scale_val
  );

void add_pooling_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int     kh,
    int     kw,
    int     pad_h,
    int     pad_h_after,
    int     pad_w,
    int     pad_w_after,
    int     stride_h,
    int     stride_w,
    int     is_avg_pooling,
    int     avg_pooling_mode,
    int     in_sign,
    int     out_sign,
    int     is_global_pooling,
    int     out_ceil_mode
  );

void add_dropout_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int     data_sign
  );

void add_upsample_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int     size,
    int     in_sign,
    int     out_sign
  );

void add_fc_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    char*   layer_name,
    int     num_input_neuron,
    int     num_output_neuron,
    void*   weight,
    void*   bias,
    int     have_bias,
    int     weight_col_is_in_neruon_num,
    int     in_sign,
    int     out_sign,
    int     weight_sign,
    int     bias_sign,
    int     rshift_num
  );

void add_batchnorm_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    char*   layer_name,
    void*   alpha,
    void*   beta,  //y = alpha * x + beta
    int     in_sign,
    int     out_sign,
    int     alpha_sign,
    int     beta_sign,
    int*    rshift_num
  );

void add_scale_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    char*   layer_name,
    int     in_sign,
    int     out_sign,
    void*   scale_factor,
    void*   bias_factor,
    void*   rshift_num
  );

void add_eltwise_layer_fix8b(
    void*   p_bmcpl,
    int     input_num,
    int**   input_shape,
    int*    input_shape_dim,
    char**  input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int     op_code,
    int*    coefficients,
    int*    in_sign,
    int     out_sign,
    int*    rshift_num
  );

void add_interp_layer_fix8b(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    int   	  pad_bag,
    int   	  pad_end,
    int       sign,
    unsigned char *coeff[],
    int   	  platform_sp //0:for caffeinterface 1:for tensorflow interface
  );

void add_concat_layer_fix8b(
    void*   p_bmcpl,
    int     input_num,
    int**   input_shape,
    int*    input_shape_dim,
    char**  input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int     concat_axis,
    int*    in_sign,
    int     out_sign,
    int*    rshift_num,
    int*    scale_num
  );

void add_multiregion_layer_fix8b(
    void*   p_bmcpl,
    int     input_num,
    int**   input_shape,
    int*    input_shape_dim,
    char**  input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int     classes,
    int     coords,
    int     nums,
    int*    Activate_parm,
    int     in_sign,
    int     out_sign
  );

void add_lrn_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    float   alpha_,
    int     size_,
    float   beta_,
    float   k_,
    int     in_sign,
    int     out_sign,
    float   scale_in,
    float   scale_out
  );

void add_prelu_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    char*   layer_name,
    int     channel_shared,
    void*   slope_,
    int     in_sign,
    int     out_sign,
    int     slope_sign,
    int     rshift_num
  );

void add_relu_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int     negative_slope,
    int     upper_limit,
    int     rshift_num,
    int     in_sign,
    int     out_sign
  );

void add_split_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int     output_num,
    int**   output_shape,
    int*    output_shape_dim,
    char**  output_name,
    int     data_sign
  );

void add_softmax_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int     inner_num,
    int     outer_num,
    int     softmax_dim,
    int     in_sign,
    float   scale_val
  );
void add_shufflechannel_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int     group_,
    int     data_sign
  );
void add_pooling_tf_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int     kh,
    int     kw,
    int     up_pad_h,
    int     down_pad_h,
    int     left_pad_w,
    int     right_pad_w,
    int     stride_h,
    int     stride_w,
    int     is_avg_pooling,
    int     in_sign,
    int     out_sign
  );
void add_biasadd_layer_fix8b(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    const char*   layer_name,
    int           num_output_neuron,
    const void*   bias,
    int           in_sign,
    int           out_sign,
    int           bias_sign,
    int           scale,
    int           rshift
    );

void add_stride_slice_layer_fix8b(
  void*   p_bmcpl,
    char*   input_name,
    int*    input_shape,
    int     input_shape_dim,
    char*   output_name,
    int*    begin_index,
    int*    end_index,
    int*    strides,
    int     index_size,
    int     begin_mask,
    int     end_mask,
    int     shrink_axis_mask,
    int     new_axis_mask,
    int     ellipsis_mask,
    int     in_sign,
    int     out_sign
  );

void add_pad_layer_fix8b(
    void*       p_bmcpl,
    const int*  input_shape,
    int         input_shape_dim,
    const char* input_name,
    const int*  output_shape,
    int         output_shape_dim,
    const char* output_name,
    const int*  paddings_,
    int         pad_dim,
    float       pad_val,
    int         pad_mode,
    int         in_sign,
    int         out_sign
  );

void add_eltwise_binary_layer_fix8b(
    void*         p_bmcpl,
    const char*   input_A_name,
    int           A_is_coeff,
    const float*  A_data,
    const char*   input_B_name,
    int           B_is_coeff,
    const float*  B_data,
    const int*    input_shape,
    int           input_dim,
    const char*   output_name,
    int           binary_op,  /* BINARY_ADD, BINARY_MUL, BINARY_MAX, BINARY_SUB, BINARY_MIN, BINARY_DIV */
    int*          in_sign,
    int           out_sign,
    int*          scale,
    int*          rshift_num
  );

void add_const_binary_layer_fix8b(
    void*         p_bmcpl,
    const char*   input_A_name,
    const int*    input_A_shape,
    int           input_A_dim,
    float         B_value,
    const char*   output_name,
    int           binary_op,  /* BINARY_ADD, BINARY_MUL, BINARY_MAX, BINARY_SUB, BINARY_MIN, BINARY_DIV */
    int           inversed,    /* 0: A op B, 1: B op A, for BINARY_SUB usually */
    int*          in_sign,
    int           out_sign,
    int*          scale,
    int*          rshift_num
  );

void add_binary_layer_fix8b(
    void*   p_bmcpl,
    const char*   input_A_name,
    const int*    input_A_shape,
    int           input_A_dim,
    int           A_is_coeff,
    const float*  A_data,
    const char*   input_B_name,
    const int*    input_B_shape,
    int           input_B_dim,
    int           B_is_coeff,
    const float*  B_data,
    const char*   output_name,
    int           binary_op,  /* BINARY_ADD, BINARY_MUL, BINARY_MAX, BINARY_SUB, BINARY_MIN, BINARY_DIV */
    int*          in_sign,
    int           out_sign,
    int*          scale,
    int*          rshift_num);

void add_broadcast_binary_layer_fix8b(
    void*   p_bmcpl,
    const char*   input_A_name,
    const int*    input_A_shape,
    int           input_A_dim,
    int           A_is_coeff,
    const float*  A_data,
    const char*   input_B_name,
    const int*    input_B_shape,
    int           input_B_dim,
    int           B_is_coeff,
    const float*  B_data,
    const char*   output_name,
    int           binary_op,  /* BINARY_ADD, BINARY_MUL, BINARY_MAX, BINARY_SUB, BINARY_MIN, BINARY_DIV */
    int*          in_sign,
    int           out_sign,
    int*          scale,
    int*          rshift_num);

void add_tile_layer_fix8b(
    void *p_bmcpl,
    const char *input_name,
    const int *input_shape,
    int input_dim,
    int input_is_coeff,
    const float *input_data,
    int coeff_is_fixed,
    const char *coeff_name,
    const int *tile_coeff,
    const char *output_name,
    int     in_sign,
    int     out_sign);

void add_space_to_batch_layer_fix8b(
  void*      p_bmcpl,
  const char* input_name,
  const int*  input_shape,
  int         input_dim,
  int         block_is_dynamic,
  const char* block_name,  //if dynamic, must be an existed shape tensor name
  const int*  block_sizes, //if not dynamic, must be valid--{hblock,wblock}
  int         pad_is_dynamic,
  const char* pad_name,    //if dynamic, must be an existed shape tensor name
  const int*  pad_sizes,   //if not dynamic, must be valid--{ht_pad, hb_pad, wl_pad, wr_pad}
  const char* output_name,
  int         in_sign,
  int         out_sign);

void add_batch_to_space_layer_fix8b(
  void*      p_bmcpl,
  const char* input_name,
  const int*  input_shape,
  int         input_dim,
  int         block_is_dynamic,
  const char* block_name,  //if dynamic, must be an existed shape tensor name
  const int*  block_sizes, //if not dynamic, must be valid--{hblock,wblock}
  int         crop_is_dynamic,
  const char* crop_name,    //if dynamic, must be an existed shape tensor name
  const int*  crop_sizes,   //if not dynamic, must be valid--{ht_crop, hb_crop, wl_crop, wr_crop}
  const char* output_name,
  int         in_sign,
  int         out_sign);

void add_reduce_full_layer_fix8b(
    void*       p_bmcpl,
    const char* input_name,
    const char* output_name,
    const int*  input_shape,
    int         input_dims,
    const int*  axis_list,
    int         axis_num,
    int         reduce_method,
    int         need_keep_dims,
    int     in_sign,
    int     out_sign,
    float input_scale,
    float output_scale
);

void add_transpose_layer_fix8b(
    void*         p_bmcpl,
    const char*   input_name,
    const char*   output_name,
    const int*    input_shape,
    const int*    order,
    int           dims,
    int     in_sign,
    int     out_sign
);

void add_shape_ref_layer_fix8b(
    void*   p_bmcpl,
    const char*   input_name,
    const int*    input_shape,
    int     input_dim,
    const char*   shape_name,
    int     data_sign
  );
void add_shape_assign_layer_fix8b(
    void*   p_bmcpl,
    const char*   input_name,
    int*    input_shape,
    int     input_dim,
    const char*   shape_name,
    const char*   output_name,
    int     data_sign
  );
void add_squeeze_layer_fix8b(
    void*   p_bmcpl,
    char*   input_name, //must exist already
    int*    input_shape,
    int     input_dim,
    int*    axis_list,
    int     axis_num, //0 means removal of all '1' dims
    char*   output_name,
    int    data_sign
  );
void add_expand_ndims_layer_fix8b(
    void*         p_bmcpl,
    const char*   input_name, //must exist already
    const int*    input_shape,
    int           input_dim,
    const float*  input_data, //input data is not Null means it is coeff
    int           axis,
    int           ndims,
    const char*   output_name,
    int           data_sign
  );
void add_expand_dims_layer_fix8b(
    void*         p_bmcpl,
    const char*   input_name, //must exist already
    const int*    input_shape,
    int           input_dim,
    int           axis,
    const char*   output_name,
    int           data_sign
  );

void add_cpu_layer_fix8b(
    void*              p_bmcpl,
    int                input_num,
    const int* const*  input_shape,      /* input1 shape, input2 shape */
    const int*         input_shape_dim,  /* input1 dim, input2 dim... */
    const char* const* input_name,       /* input1 name, input2 name...*/
    int                output_num,
    const int* const*  output_shape,
    const int*         output_shape_dim,
    const char* const* output_name,
    const int*   input_dtype,      /* input1 data type, input2 data type...
                                    * 0:FP32, 1:FP16, 2:INT8, 3:UINT8, 4:INT16,
                                    * 5:UINT16, 6:INT32, 7:UINT32), same as
                                    * DATA_TYPE_T defined in bmcompiler_common.h
                                    */
    const int*   output_dtype,
    const float* input_scales,     /* input1 scale value, input2 scale value.. */
    const float* output_scales,
    int          op_type,
    const void*  layer_param,      /* pass through to cpu.so */
    int          param_size
  );

void add_user_cpu_layer_fix8b(
    void*              p_bmcpl,
    int                input_num,
    const int* const*  input_shape,      /* input1 shape, input2 shape */
    const int*         input_shape_dim,  /* input1 dim, input2 dim... */
    const char* const* input_name,       /* input1 name, input2 name...*/
    int                output_num,
    const int* const*  output_shape,
    const int*         output_shape_dim,
    const char* const* output_name,
    const int*   input_dtype,      /* input1 data type, input2 data type...
                                    * 0:FP32, 1:FP16, 2:INT8, 3:UINT8, 4:INT16,
                                    * 5:UINT16, 6:INT32, 7:UINT32), same as
                                    * DATA_TYPE_T defined in bmcompiler_common.h
                                    */
    const int*   output_dtype,
    const float* input_scales,     /* input1 scale value, input2 scale value.. */
    const float* output_scales,
    const void*  layer_param,      /* pass through to cpu.so */
    int          param_size
  );

void add_priorbox_cpu_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    void*   output_data,
    float*   min_sizes,
    int     real_min_size,
    float*   max_sizes,
    int     real_max_size,
    float*   aspect_ratios,
    int     real_spect_size,
    float*   variance,
    int     real_variance_size,
    int     num_priors,
    int     img_w,
    int     img_h,
    float   step_w,
    float   step_h,
    float   offset,
    float   thTop,
    int     bottom_0_width,
    int     bottom_0_height,
    int     bottom_1_width,
    int     bottom_1_height,
    int     dim,
    bool    has_dim,
    bool    flip,
    bool    clip,
    int     in_sign,
    int     out_sign
);

void add_select_layer_fix8b(
    void*         p_bmcpl,
    const char*   cond_name,
    const char*   s0_name,
    const int     s0_is_const,
    const int     s0_value_fix8b,
    const char*   s1_name,
    const int     s1_is_const,
    const int     s1_value_fix8b,
    const char*   output_name,
    const int     in_sign,
    const int     s0_sign,
    const int     s1_sign,
    const int     out_sign,
    int           scalea,
    int           nshifta,
    int           scaleb,
    int           nshiftb,
    const int*    shape,
    const int     dims
);

void add_arg_layer_fix8b(
    void*       p_bmcpl,
    const int*  input_shape,
    int         input_shape_dim,
    const char* input_name,
    const int*  output_shape,
    int         output_shape_dim,
    const char* output_name,
    int         in_sign,
    int         axis,
    int         method
);

#define SSD_DETECT_OUT_MAX_NUM 200
void add_ssd_detect_out_layer_fix8b(
    void*              p_bmcpl,
    const int* const*  input_shape,
    const int*         input_shape_dim,
    const char* const* input_name,
    const int*         input_dtype,
    const float*       input_scale,
    const int*         output_shape,
    int                output_shape_dim,
    const char*        output_name,
    const int          output_dtype,
    const float        output_scale,
    int                num_classes,
    bool               share_location,
    int                background_label_id,
    int                code_type,
    bool               variance_encoded_in_target,
    int                keep_top_k,
    float              confidence_threshold,
    float              nms_threshold,
    float              eta,
    int                top_k
    );

#define YOLOV3_DETECT_OUT_MAX_NUM 200
void add_yolov3_detect_out_layer_fix8b(
    void*              p_bmcpl,
    int                input_num,
    const int* const*  input_shape,
    const int*         input_shape_dim,
    const char* const* input_name,
    const int*         input_dtype,
    const float*       input_scale,
    int*               output_shape,
    int                output_shape_dim,
    const char*        output_name,
    const int          output_dtype,
    const float        output_scale,
    int                num_classes,
    int                num_boxes,
    int                mask_group_size,
    int                keep_top_k,
    float              confidence_threshold,
    float              nms_threshold,
    float*             bias,
    float*             anchor_scale,
    float*             mask);

void add_tf_split_layer_fix8b(
     void*              p_bmcpl,
     const int*         input_shape,
     int                input_shape_dim,
     const char*        input_name,
     int                output_num,
     const int* const*  output_shape,
     const int*         output_shape_dim,
     const char* const* output_name,
     int                shape_dim,
     int                axis,
     const int*         split_size,
     int                split_num,
     int                in_sign,
     int                out_sign);

#ifdef __cplusplus
}
#endif

#endif
