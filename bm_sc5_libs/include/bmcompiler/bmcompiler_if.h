#ifndef BMCOMPILER_IF_H_
#define BMCOMPILER_IF_H_

/*!
* \file bmcompiler_if.h
* \brief interface of bmcompiler
*/

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief create bmcompiler pointer
 * \return bmcompiler pointer
 */
void* create_bmcompiler(const char* chip_name);
void* create_bmcompiler_dir(const char* chip_name, const char* dir);
/**
 * \brief finish bmcompile
 *
 *   Store context and destroy p_bmcpl.
 *   Note: Must use it at last
 * \param p_bmcpl void* pointer returned from create_bmcompiler()
 */
void finish_bmcompiler(void* p_bmcpl);

/**
 * \brief finish bmcompile and get bmodel data in memory
 *
 * @param [out] bmodel_data  The bmodel data in memory. Declare it as void* bmodel_data = NULL,
 *                           use it as &bmodel_data. It will be malloc memory internal.
 *                           User need to free(bmodel_data) if do not use it.
 * @param [out] size         The size of bmodel data in memory, unit: byte
 *
 */
void finish_bmcompiler_data(void* p_bmcpl, void** bmodel_data, unsigned int* size);

/**
* \brief start the static compilation of bmcompiler
* \param p_bmcpl      The pointer that had been created
* \param net_name     The name of the neuron network
*/
void __bmcompile(void* p_bmcpl, char* net_name);
/**
* \brief start the dynamic compilation of bmcompiler
* \param p_bmcpl      The pointer that had been created
* \param net_name     The name of the neuron network
*/
void __bmcompile_ir(void* p_bmcpl, char* net_name);
/**
* \brief start the static compilation of bmcompiler with selecting optimization level
* \param p_bmcpl      The pointer that had been created
* \param net_name     The name of the neuron network
* \param opt_level    The optimization level
*/
void __bmcompile_opt(void* p_bmcpl, char* net_name, int opt_level);
/**
* \brief start the dynamic compilation of bmcompiler with selecting optimization level
* \param p_bmcpl      The pointer that had been created
* \param net_name     The name of the neuron network
* \param opt_level    The optimization level
*/
void __bmcompile_ir_opt(void* p_bmcpl, char* net_name, int opt_level);

/**
* \brief start the dynamic compilation of bmcompiler with selecting optimization level, but not use multistage strategy
* \param p_bmcpl      The pointer that had been created
* \param net_name     The name of the neuron network
* \param opt_level    The optimization level
*/
void __bmcompile_ir_opt_no_multistage(void* p_bmcpl, char* net_name, int opt_level);

/**
 * \brief save u_model interface
 * \param p_bmcpl     void* pointer returned from create_bmcompiler()
 * \param net_name    file name of u_model
 */
void _bmcompiler_save_umodel(void* p_bmcpl, char* net_name);

void _bmcompiler_save_umodel_with_check(void* p_bmcpl, char* net_name,
        char* input_name[], float** input_data, int input_num,
        char* refer_name[], float** refer_data, int refer_num);


/**
 * \brief delete bmcompiler point
 *
 *   Destroy p_bmcpl.
 * \param p_bmcpl void* pointer returned from create_bmcompiler()
 */
void delete_bmcompiler(void* p_bmcpl);

/**
* \brief start the dynamic compilation of bmcompiler with selecting optimization level
* \param p_bmcpl      The pointer that had been created
* \param input_name
* \param input_data
* \param input_num
* \param refer_name
* \param refer_data
* \param refer_num
* \param net_name     The name of the neuron network
*/
void __compile_with_result_check_py(void* p_bmcpl, char* input_name[], float** input_data, int input_num,
        char* refer_name[], float** refer_data, int refer_num, char* net_name); //wxc 20181128 for bmnetm

void compile_with_result_check(void* p_bmcpl, char** input_name, float** input_data, int input_num,
        char** refer_name, float** refer_data, int refer_num, char* net_name);
/**
* \brief set the optimization level before start bmcompiler
* \param opt_level    The optimization level
*/
void __bmcompile_set_net_config(int opt_level);

/**
* \brief set the max allowed mismatch for fixpoint compiling
* \param fix_delta
*/
void __bmcompile_set_fixpoint_cmp_margin(int fixpoint_cmp_margin);

/**
 * \brief set winograd flag
 * \param winograd_flag    0: does not use winograd, 1: use winograd without coeff optimization, 2: use winograd with coeff optimization
 */
void __bmcompiler_set_winograd(int winograd_flag);

void add_layer_name(void* p_bmcpl, int layer_id, const char * layer_name);
int get_max_layer_id(void* p_bmcompiler_net);
void set_bmcompiler_profile(void * p_bmcpl, bool enable);

/**
 * \brief set layer bmlang flag
 * \param flag   true: bmlang info will be added to layer type when showing graph
 */
void set_layer_bmlang_flag(bool flag);

/**
 * \brief get layer bmlang flag
 */
bool get_layer_bmlang_flag();

/**
 * \brief set cpu layer use_max_as_th flag
 * \param p_bmcpl           void* pointer returned from create_bmcompiler()
 * \param use_max_as_th     true: using max value as threshold in calibration per tensor.
 * \param num               tensor number to be set use_max_as_th.
 */
void set_cpu_layer_use_max_as_th(void *p_bmcpl, bool use_max_as_th[], int num);

/**
 * \brief The common description for all add layer api
 *
 *   The following is about layer adding.
 *   After using create_bmcompiler to create p_bmcpl, we need to add layer to p_bmcpl.
 *   This following is the common used param.
 * \param p_bmcpl           void* pointer returned from create_bmcompiler()
 * \param input_num         the number of layer input tensors
 * \param input_shape       the tensor shape of each input. One dimension array if input_num == 1, two dimension array if input_num > 1
 * \param input_shape_dim   the dimension of input shape, ex. The shape dimension of (n, c, h, w) is 4
 * \param input_name        the name of input tensors
 * \param output_num        the number of layer output tensors
 * \param output_shape_dim  the dimension of output shape
 * \param output_name       the name of output tensors
 * \param layer_name        the name of layer
 */

/**
 * \brief add convolution layer
 * \param  weight            the float data of convolution kernel
 * \param  bias              the float data of convolution bias
 * \param  kh                kernel height
 * \param  kw                kernel width
 * \param  groups            the number of convolution groups
 * \param  pad_h             height padding
 * \param  pad_w             width padding
 * \param  stride_h          height stride of kernel
 * \param  stride_w          width stride of kernel
 * \param  dh                dilation in height dimension
 * \param  dw                dilation in width dimension
 * \param  have_bias         whether the convolution use bias or not, 1 use, 0 not use
 */
void add_conv_layer(
    void* 	      p_bmcpl,
    const int*    input_shape,
    int   	      input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	      output_shape_dim,
    const char*   output_name,
    const char*   layer_name,
    const float*  weight,
    const float*  bias,
    int   	  kh,
    int   	  kw,
    int   	  groups,
    int   	  pad_h_up,
    int   	  pad_h_down,
    int   	  pad_w_left,
    int   	  pad_w_right,
    int   	  stride_h,
    int   	  stride_w,
    int   	  dh,
    int   	  dw,
    int   	  have_bias
);


void add_conv_layer_v2(
    void* 	      p_bmcpl,
    const int*    input_shape,
    int   	      input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	      output_shape_dim,
    const char*   output_name,
    const char*   weight_name,
    const float*  weight,
    const char*   bias_name,
    const float*  bias,
    int   	  kh,
    int   	  kw,
    int   	  groups,
    int   	  pad_h_up,
    int   	  pad_h_down,
    int   	  pad_w_left,
    int   	  pad_w_right,
    int   	  stride_h,
    int   	  stride_w,
    int   	  dh,
    int   	  dw,
    int   	  have_bias
);

/**
 * \brief add deconvolution layer
 * \param  weight            the float data of convolution kernel
 * \param  bias              the float data of convolution bias
 * \param  kh                kernel height
 * \param  kw                kernel width
 * \param  groups            the number of convolution groups
 * \param  pad_h             height padding
 * \param  pad_w             width padding
 * \param  stride_h          height stride of kernel
 * \param  stride_w          width stride of kernel
 * \param  dh                dilation in height dimension
 * \param  dw                dilation in width dimension
 * \param  have_bias         whether the convolution use bias or not, 1 use, 0 not use
 */
void add_deconv_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    const char*   layer_name,
    const float*  weight,
    const float*  bias,
    int   	  kh,
    int   	  kw,
    int   	  groups,
    int   	  pad_h_up,
    int   	  pad_h_down,
    int   	  pad_w_left,
    int   	  pad_w_right,
    int   	  stride_h,
    int   	  stride_w,
    int   	  dh,
    int   	  dw,
    int   	  have_bias
  );
void add_crop_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const char*   shape_name,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    const int*    offsets
  );
/**
 * \brief add pooling layer
 * \param  kh                kernel height
 * \param  kw                kernel width
 * \param  pad_h             height padding
 * \param  pad_w             width padding
 * \param  stride_h          height stride of kernel
 * \param  stride_w          widtth stride of kernel
 * \param  is_avg_pooling    average pooling or not. 1 is, 0 not
 * \parame avg_pooling_mode  0 -- all position is divided by a constant(kh * kw)
 *                           1 -- Pad value is elimated in average computation.
 * \param  out_ceil_mode:    used for top tensor shape calculation
 *                           this parameter is needed for dynamic compiling.
 *                           0 -- floor mode
 *                           1 -- ceil mode
 *                           2 -- caffe mode(default)
 */
void add_pooling_layer(
    void*   		p_bmcpl,
    const int*    	input_shape,
    int     		input_shape_dim,
    const char*   	input_name,
    int     		output_number,
    const int* const*   output_shape,
    const int*    	output_shape_dim,
    const char* const*  output_name,
    int     		kh,
    int     		kw,
    int     		up_pad_h,
    int     		down_pad_h,
    int     		left_pad_w,
    int     		right_pad_w,
    int     		stride_h,
    int     		stride_w,
    int     		is_avg_pooling,
    int     		avg_pooling_mode,
    int     		is_global_pooling,
    int     		out_ceil_mode,
    const char*   	layer_name,
    const float*  	coeff_mask
  );
/**
 * \brief add interp layer
 * \param       platform_sp:
 *                           0 -- caffe bilinear
 *                           1 -- tensorflow bilinear
 *                           2 -- caffe nearest
 *                           3 -- tensorflow nearest
 */

void add_interp_layer(
   void* 	 p_bmcpl,
   const int*    input_shape,
   int   	 input_shape_dim,
   const char*   input_name,
   const int*    output_shape,
   int    	 output_shape_dim,
   const char*   output_name,
   int    	 pad_bag,
   int    	 pad_end,
   int    	 platform_sp
 );

void add_interp_layer_v2(
   void*   p_bmcpl,
   const int*    input_shape,
   int     input_shape_dim,
   const char*   input_name,
   int     shape_is_fixed,
   // if shape is fixed, shape data should be given
   const int*    output_shape,
   int     output_shape_dim,
   // if shape is not fixed, its name should be given
   // which must be a shape tensor that be added already
   const char*   output_shape_name,
   const char*   output_name,
   int     pad_bag,
   int     pad_end,
   int     platform_sp
 );

void add_rpnproposal_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    input_shape1,
    int   	  input_shape_dim1,
    const char*   input_name1,
    const int*    input_shape2,
    int   	  input_shape_dim2,
    const char*   input_name2,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    int   	  feat_stride_,
    int   	  base_size_,
    int   	  min_size_,
    int   	  pre_nms_topN_,
    int   	  post_nms_topN_,
    float   	  nms_thresh_,
    float 	  score_thresh_
  );
void add_psroipooling_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    input_shape1,
    int   	  input_shape_dim1,
    const char*   input_name1,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    int   	  output_dim,
    int   	  group_size,
    float 	  spatial_scale_,
    int   	  roi_nums
  );

void add_roipooling_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    input_shape1,
    int   	  input_shape_dim1,
    const char*   input_name1,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    int   	  pooled_h,
    int   	  pooled_w,
    float 	  spatial_scale_,
    int   	  roi_nums
    );

void add_adaptivepooling_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    int   	  pooled_h,
    int   	  pooled_w
    );

void add_shufflechannel_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    int   	  group_
    );

void add_dropout_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name
  );
void add_upsample_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    int   	  size
  );
/**
 * \brief add full connect layer
 * \param num_input_neuron              The number of input neuron.
 * \param num_output_neuron             The number of output neuron.
 * \param weight                        The float data of fc weight.
 * \param bias                          The float data of fc bias.
 * \param have_bias                     1: have, 0: not have
 * \param weight_col_is_in_neruon_num   Whether the weight collum number is the input neuron number or not. 1: yes, 0: no
 */
void add_fc_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    const char*   layer_name,
    int   	  num_input_neuron,
    int   	  num_output_neuron,
    const float*  weight,
    const float*  bias,
    int   	  have_bias,
    int   	  weight_col_is_in_neruon_num
  );

void add_fc_weight_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    const char*   layer_name,
    const int*    weight_shape,
    int   	  weight_shape_dim,
    const char*   weight_name,
    int           num_input_neuron,
    const float*  bias,
    int   	  have_bias,
    int   	  weight_col_is_in_neruon_num
  );

void add_lrn_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    float 	  alpha_,
    int   	  size_,
    float 	  beta_,
    float 	  k_
  );

void add_batchnorm_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    const char*   layer_name,
    const float*  mean_param,
    const float*  variance_param,
    float         scale_val,
    float         epsilon,
    int           NormMethod,
    int           is_var_need_calc = 1
  );

void add_scale_layer(
    void*               p_bmcpl,
    int                 input_num,
    const int* const*   input_shape,
    const int*          input_shape_dim,
    const char* const*  input_name,
    const int*          output_shape,
    int                 output_shape_dim,
    const char*         output_name,
    const char*         layer_name,
    const float*        scale_factor,
    const float*        bias_factor,
    int                 num_axes,
    int                 axis_,
    int                 have_bias
  );

/**
 * \brief
 *   add element-wise layer
 * \param
 *   op_code, 0 PRODUCT, 1 SUM, 2 MAX
 */
void add_eltwise_layer(
    void* 		p_bmcpl,
    int   		input_num,
    const int* const*   input_shape,
    const int*    	input_shape_dim,
    const char* const*  input_name,
    const int*    	output_shape,
    int     		output_shape_dim,
    const char*   	output_name,
    int     		op_code,
    const float*  	coefficients
  );

/**
 * \brief
 *   add concat layer
 * \param
 *    concat_axis, 0 batch size, 1 channels, 2 height, 3 width
 */
void add_concat_layer(
    void*   		p_bmcpl,
    int     		input_num,
    const int* const*   input_shape,
    const int*    	input_shape_dim,
    const char* const*  input_name,
    const int*    	output_shape,
    int     		output_shape_dim,
    const char*   	output_name,
    int     		concat_axis
  );
void add_prelu_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    const char*   layer_name,
    int   	  channel_shared,
    const float*  slope_
  );
void add_multiregion_layer(
    void*   		p_bmcpl,
    int     		input_num,
    const int* const*   input_shape,
    const int*    	input_shape_dim,
    const char* const*  input_name,
    const int*    	output_shape,
    int     		output_shape_dim,
    const char*   	output_name,
    int     		classes,
    int     		coords,
    int     		nums,
    const int*    	Activate_parm
  );

void add_reorg_layer(
    void*   		p_bmcpl,
    const int*		input_shape,
    int     		input_shape_dim,
    const char*		input_name,
    const int*  	output_shape,
    int     		output_shape_dim,
    const char*   	output_name,
    int     		stride,
    int     		reverse
  );

void add_priorbox_layer(
    void*   		p_bmcpl,
    const int*    	input_shape,
    int     		input_shape_dim,
    const char*   	input_name,
    const int*    	output_shape,
    int     		output_shape_dim,
    const char*   	output_name,
    const float*  	output_data,
    /* for save umodel */
    int     		min_len,
    const 		float*  min_size,
    int     		max_len,
    const float*  	max_size,
    int     		ratio_len,
    const float*  	aspect_ratio,
    int     		flip,
    int     		clip,
    int     		var_len,
    const float*  	variance,
    int     		img_h,
    int     		img_w,
    float   		step_h,
    float   		step_w,
    float   		offset
  );

void add_permute_layer(
    void*   		p_bmcpl,
    const int*    	input_shape,
    int     		input_shape_dim,
    const char*   	input_name,
    const int*    	output_shape,
    int     		output_shape_dim,
    const char*   	output_name,
    const int*    	permute_order
  );

void add_reverse_layer(
    void*   		p_bmcpl,
    const int*    	input_shape,
    int     		input_shape_dim,
    const char*   	input_name,
    const int*    	output_shape,
    int     		output_shape_dim,
    const char*   	output_name,
    int     		axis
  );

void add_normalize_layer(
    void*   		p_bmcpl,
    const int*    	input_shape,
    int     		input_shape_dim,
    const char*   	input_name,
    const int*    	output_shape,
    int     		output_shape_dim,
    const char*   	output_name,
    const char*   	layer_name,
    int     		across_spatial,
    int     		channel_shared,
    const float*  	scale_data,
    float   		eps
  );
//[DEPRECATED]: try to use add_reshape_layer_v2
void add_flatten_layer(
    void*   		p_bmcpl,
    const int*    	input_shape,
    int     		input_shape_dim,
    const char*   	input_name,
    const int*    	output_shape,
    int     		output_shape_dim,
    const char*   	output_name,
    const int*         	raw_param
  );

//flatten shape[begin_dim, end_dim)
void add_flatten_layer_v2(
    void*       p_bmcpl,
    const char* input_name,
    const int*  input_shape,
    int         input_dim,
    const char* output_name,
    int         begin_dim,
    int         end_dim
  );

//[DEPRECATED]: try to use add_reshape_layer_v2 instead
void add_reshape_layer(
    void*              	p_bmcpl,
    const int*       	input_shape,
    int                	input_shape_dim,
    const char*		input_name,
    const int*         	output_shape,
    int                	output_shape_dim,
    const char*        	output_name,
    const int*         	raw_param
  );

/**
 * each shape value in the new_shape can be the following cases (according MXNet)
 * a. positive value,
 * b. 0(means using corresponding bottom shape),
 * c. -1(means auto calulated, can only contain one),
 * d. -2(copy all/remainder of the input dimensions to the output shape)
 * e. -3(use the product of two consecutive dimensions of the input shape as the output dimension.)
 * f. -4(split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).)
 */

void add_reshape_layer_v2(
    void*              p_bmcpl,
    const char*        input_name,
    const int*         input_shape,
    int                input_dim,
    const char*        output_name,
    const int*         new_shape,
    int                new_dims
  );

void add_biasadd_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    const char*   layer_name,
    int           num_output_neuron,
    const float*  bias,
    bool          bIsCoefInt8);

/**
 * \param active_type_id 0: Tanh, 1: Sigmoid
 */
void add_active_layer(
    void*   		p_bmcpl,
    const int*    	input_shape,
    int     		input_shape_dim,
    const char*   	input_name,
    const int*    	output_shape,
    int     		output_shape_dim,
    const char*   	output_name,
    int     		active_type_id
  );

void add_cpu_layer(
    void*   		p_bmcpl,
    int     		input_num,
    const int* const*   input_shape,
    const int*    	input_shape_dim,
    const char* const*  input_name,
    int     		output_num,
    const int* const*   output_shape,
    const int*    	output_shape_dim,
    const char* const*  output_name,
    int     		op_type,
    const void*   	layer_param,
    int     		param_size
    );

void add_user_cpu_layer(
    void*   		p_bmcpl,
    int     		input_num,
    const int* const*   input_shape,
    const int*    	input_shape_dim,
    const char* const*  input_name,
    int     		output_num,
    const int* const*   output_shape,
    const int*    	output_shape_dim,
    const char* const*  output_name,
    const void*   	layer_param,
    int     		param_size
    );

void add_relu_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    float 	  negative_slope,
    float 	  upper_limit
  );

void add_softmax_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    int   	  inner_num,
    int   	  outer_num,
    int   	  softmax_dim
  );

void add_split_layer(
    void*   		p_bmcpl,
    const int*    	input_shape,
    int     		input_shape_dim,
    const char*   	input_name,
    int     		output_num,
    const int* const*   output_shape,
    const int*    	output_shape_dim,
    const char* const*  output_name
  );

void add_lstm_layer(
    void*   		p_bmcpl,
    int     		input_num,
    const int* const*   input_shape,
    const int*    	input_shape_dim,
    const char* const*  input_name,
    int     		output_num,
    const int* const*   output_shape,
    const int*    	output_shape_dim,
    const char* const*  output_name,
    const char*   	layer_name,
    int     		batch_num,
    int     		time_num,
    int     		input_dim,
    int     		output_dim,
    int     		user_define_cont,
    int     		with_x_static,
    int     		expose_hidden,
    const float*  	x_weight,
    const float*  	x_bias,
    const float*  	h_weight,
    const float*  	x_static_weight
  );

void add_pad_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    const int*    paddings_,
    int   	  pad_dim,
    float 	  pad_val,
    int   	  pad_mode
  );

void add_arg_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    int   	  axis,
    int   	  method
  );

void add_pooling_tf_layer(
    void* 	  p_bmcpl,
    const int*    input_shape,
    int   	  input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	  output_shape_dim,
    const char*   output_name,
    int   	  kh,
    int   	  kw,
    int   	  up_pad_h,
    int   	  down_pad_h,
    int   	  left_pad_w,
    int   	  right_pad_w,
    int   	  stride_h,
    int   	  stride_w,
    int   	  is_avg_pooling);

void add_transpose_layer(
    void*         p_bmcpl,
    const char*   input_name,
    const char*   output_name,
    const int*    shape,
    const int*    permute_order,
    int           dims
  );

void add_transpose_layer_v2(
    void*         p_bmcpl,
    const char*   input_name,
    const int*    input_shape,
    int           input_dims,
    int           input_dtype,
    const char*   output_name,
    const char*   order_name,  // order_name can be NULL if permute_order provided
    const int*    permute_order  //permute_order can be NULL if order_name provided
  );

void add_coeff_layer(
    void 	  *p_bmcpl,
    const char 	  *name,
    const int 	  *shape,
    const void 	  *data,
    int 	  dims
  );

void add_const_data_layer(
        void*		p_bmcpl,
        const char*	input_name,
        const char*	output_name,
        const int*	shape,
        const void*	data,
        int 		dims
        );

void add_select_layer(
        void*  p_bmcpl,
        const char* cond_name,
        const char* s0_name,
        const int   s0_is_const,
        const float s0_value,
        const char* s1_name,
        const int   s1_is_const,
        const float s1_value,
        const char* output_name,
        const int*  shape,
        const int   dims
        );

void add_where_layer(
        void*  p_bmcpl,
        const char* cond_name,
        const char* output_name,
        const int*  in_shape,
        const int*  out_shape,
        const int   in_dims,
        const int   out_dims
        );

//support inputs with different dims
void add_binary_layer_v2(
        void*         p_bmcpl,
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
        int           binary_op  /* BINARY_ADD, BINARY_MUL, BINARY_MAX, BINARY_SUB, BINARY_MIN, BINARY_DIV */
        );

void add_binary_layer(
        void*         p_bmcpl,
        const char*   input_A_name,
        const int*    input_A_shape,
        int           A_is_coeff,
        const float*  A_data,
        const char*   input_B_name,
        const int*    input_B_shape,
        int           B_is_coeff,
        const float*  B_data,
        int           input_dim,
        const char*   output_name,
        int           binary_op  /* BINARY_ADD, BINARY_MUL, BINARY_MAX, BINARY_SUB, BINARY_MIN, BINARY_DIV */
        );

void add_broadcast_binary_layer_v2(
        void*         p_bmcpl,
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
        int           binary_op/* BINARY_ADD, BINARY_MUL, BINARY_MAX, BINARY_SUB, BINARY_MIN, BINARY_DIV */
        );

void add_broadcast_binary_layer(
        void*         p_bmcpl,
        const char*   input_A_name,
        const int*    input_A_shape,
        int           A_is_coeff,
        const float*  A_data,
        const char*   input_B_name,
        const int*    input_B_shape,
        int           B_is_coeff,
        const float*  B_data,
        int           input_dim,
        const char*   output_name,
        int           binary_op  /* BINARY_ADD, BINARY_MUL, BINARY_MAX, BINARY_SUB, BINARY_MIN, BINARY_DIV */
        );

void add_const_binary_layer(
        void*         p_bmcpl,
        const char*   input_A_name,
        const int*    input_A_shape,
        int           input_A_dim,
        float         B_value,
        const char*   output_name,
        int           binary_op,  /* BINARY_ADD, BINARY_MUL, BINARY_MAX, BINARY_SUB, BINARY_MIN, BINARY_DIV */
        int           inversed    /* 0: A op B, 1: B op A, for BINARY_SUB usually */
        );

void add_eltwise_binary_layer(
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
        int           binary_op  /* BINARY_ADD, BINARY_MUL, BINARY_MAX, BINARY_SUB, BINARY_MIN, BINARY_DIV */
        );

void add_tile_layer_v2(
        void*  p_bmcpl,
        const char*  input_name,
        const int*   input_shape,
        int    input_dim,
        int    input_is_coeff, //when is coeff, input_data must be valid
        const float* input_data,
        int    coeff_is_fixed,
        const char*  coeff_name, //when is not fixed, name must be a valid shape tensor name
        const int*   tile_coeff, //when is not fixed, tile_coeff is not used
        const char*  output_name
        );

void add_expand_layer(void 	*p_bmcpl,
                      const char *input_name,
                      const int *input_shape,
                      int 	input_dim,
                      int 	input_is_coeff,
                      const float *input_data,
                      int 	output_shape_is_fixed,
                      const char *output_shape_name,
                      const int *output_shape,
                      int 	output_dim,
                      const char *output_name);

void add_repeat_layer(
        void*  		p_bmcpl,
        const char*  	input_name,
        const int*   	input_shape,
        int    		is_coeff,
        const float* 	data,
        const int*   	tile_coeff,
        int    		input_dim,
        const char*  	output_name
        );

void add_reduce_layer(
        void*	    p_bmcpl,
        const int*  input_shape,
        int         input_shape_dim,
        const char* input_name,
        const int*  output_shape,
        int         output_shape_dim,
        const char* output_name,
        int         reduce_method // (reduce_method >> 16) = 0 : reduce nhw, (reduce_method >> 16) = 1: reduce w
        );

//refer to tensorflow: tf.strided_slice(...)
void add_stride_slice_layer_v2(
        void*   p_bmcpl,
        const char*   input_name,
        const int*    input_shape,
        int     input_shape_dim,
        const char*   output_name,
        const int*    begin_index,
        const int*    end_index,
        const int*    strides,
        int     index_size,
        int     begin_mask,
        int     end_mask,
        int     shrink_axis_mask,
        int     new_axis_mask,  // not implemented, use 0
        int     ellipsis_mask   // not implemented, use 0
        );

void add_stride_slice_layer(
        void*   p_bmcpl,
        const int*    input_shape,
        int     input_shape_dim,
        const char*   input_name,
        const int*    output_shape,
        int     output_shape_dim,
        const char*   output_name,
        int     shape_size,
        int     begin_mask,
        int     end_mask,
        const int*    begin_index,
        const int*    end_index,
        const int*    stride
    );

void add_slice_like_layer(
    void*   		p_bmcpl,
    const int* const*   input_shape,
    const int*    	input_shape_dim,
    const char* const*  input_name,
    const int*    	output_shape,
    int     		output_shape_dim,
    const char*   	output_name,
    const int*    	axis,
    int     		axis_num
    );

void add_upsamplemask_layer(
    void*   		p_bmcpl,
    int     		input_num,
    const int* const*   input_shape,
    const int*    	input_shape_dim,
    const char* const*  input_name,
    const int*    	output_shape,
    int     		output_shape_dim,
    const char*   	output_name,
    const char*   	layer_name,
    const float*  	mask_coeff
  );

void add_pad_layer(
    void*   	p_bmcpl,
    const int*  input_shape,
    int     	input_shape_dim,
    const char* input_name,
    const int*  output_shape,
    int     	output_shape_dim,
    const char* output_name,
    const int*  paddings_,
    int     	pad_dim,
    float   	pad_val,
    int     	pad_mode
    );

void add_arg_layer(
    void*   	p_bmcpl,
    const int*  input_shape,
    int     	input_shape_dim,
    const char* input_name,
    const int*  output_shape,
    int     	output_shape_dim,
    const char* output_name,
    int     	axis,
    int     	method
  );

//when split_num=1, split_size[0] as the real split_num
//and use same split_size=input_shape[axis]/split_size[0]
void add_tf_split_layer(
     void*  		p_bmcpl,
     const int*    	input_shape,
     int     		input_shape_dim,
     const char*   	input_name,
     int     		output_num,
     const int* const*  output_shape,
     const int*    	output_shape_dim,
     const char* const* output_name,
     int     		shape_dim,
     int     		axis,
     const int*    	split_size,
     int     		split_num
  );

void add_topk_layer(
    void*  		p_bmcpl,
    const int*    	input_shape,
    int     		input_shape_dim,
    const char*   	input_name,
    int     		k,
    int     		dim,
    const int* const*   output_shape,
    const int*    	output_shape_dim,
    const char* const*  output_name
        );

void add_output_layer(
    void*              p_bmcpl,
    const int*         io_shape,
    int                io_shape_dim,
    const char*        input_name,
    const char*        output_name
  );

void add_output_layer_v2(
    void*              p_bmcpl,
    const char*        name
  );

void add_shape_slice_layer(
        void*       p_bmcpl,
        const char* shape_name,
        int         begin,
        int         end,
        int         step,
        const char* slice_name
);

void add_shape_slice_layer_v2(
        void*       p_bmcpl,
        const char* shape_name,
        const int*  begin,
        const int*  end,
        const int*  step,
        int         num,
        int         begin_mask,
        int         end_mask,
        int         shrink_mask,
        const char* slice_name
);
void add_shape_slice_layer_v3(
        void*       p_bmcpl,
        const char* shape_name,
        const int*  begin,
        const int*  end,
        const int*  step,
        int         num,
        int         begin_mask,
        int         end_mask,
        int         shrink_mask,
        int         new_axis_mask,
        int         ellipsis_mask,
        const char* slice_name
);

void add_shape_pack_layer(
        void*               p_bmcpl,
        const char* const*  shape_names,
        int                 shape_num,
        const char*         pack_name
);

void add_shape_pack_layer_v2(
        void*               p_bmcpl,
        const char* const*  shape_names,
        int                 shape_num,
        int                 axis,
        const char*         pack_name
);

void add_shape_const_layer(
        void*       p_bmcpl,
        const int*  shape_data,
        int         shape_dims,
        const char* shape_name
);

// enhanced add_shape_const_layer
void add_shape_const_layer_v2(
        void*       p_bmcpl,
        const int*  data,    //shape's data
        const int*  shape,   //shape's shape
        int         dims,    //shape's shape's dims
        const char* name     //shape's name
        );

void add_shape_op_layer(
        void*         p_bmcpl,
        const char*   in0_name,
        const char*   in1_name,
        int           binary_op,
        const char*   out_name
);

void add_shape_ref_layer(
        void*   p_bmcpl,
        const char*   input_name,
        const int*    input_shape,
        int           input_dim,
        const char*   shape_name
);

void add_shape_addn_layer(
       void*   p_bmcpl,
       const char* const* input_names,
       const int input_num,
       const char* output_name
);

void add_rank_layer(
        void*         p_bmcpl,
        const char*   input_name,
        const int*    input_shape,
        int           input_dim,
        const char*   output_name
);

void add_squeeze_layer(
        void*         p_bmcpl,
        const char*   input_name, //must exist already
        const int*    input_shape,
        int           input_dim,
        const int*    axis_list,
        int           axis_num, //0 means removal of all '1' dims
        const char*   output_name
);

void add_expand_ndims_layer(
        void*         p_bmcpl,
        const char*   input_name, //must exist already
        const int*    input_shape,
        int           input_dim,
        const float*  input_data, //input data is not Null means it is coeff
        int           axis,
        int           ndims,
        const char*   output_name);

void add_expand_dims_layer(
        void*         p_bmcpl,
        const char*   input_name, //must exist already
        const int*    input_shape,
        int           input_dim,
        int           axis,
        const char*   output_name
);

void add_shape_assign_layer(
        void*         p_bmcpl,
        const char*   input_name,
        const int*          input_shape,
        int           input_dim,
        const char*   shape_name,
        const char*   output_name
);

void add_shape_reorder_layer(
        void*       p_bmcpl,
        const char* shape_name,
        const int*  shape_order,
        int         order_num,  //must be the same length of shape_dims
        const char* output_name
);

void add_ref_crop_layer(
        void*         p_bmcpl,
        const char*   input_name,
        const int*    input_shape,
        int           input_dim,
        const char*   crop_name,
        const char*   output_name
);
void add_ref_pad_layer(
        void*       p_bmcpl,
        const char* input_name,
        const int*  input_shape,
        int         input_dim,
        const char* pad_name,
        int         pad_mode,
        float       pad_value,
        const char* output_name
);

void add_conv_weight_layer(
        void*       p_bmcpl,
        const int*  input_shape,
        int         input_shape_dim,
        const char* input_name,
        const int*  output_shape,
        int         output_shape_dim,
        const char* output_name,
        const char* layer_name,
        const int*  weight_shape,
        int         weight_shape_dim,
        const char* weight_name,
        const float* bias,
        int         groups,
        int         pad_h_up,
        int         pad_h_down,
        int         pad_w_left,
        int         pad_w_right,
        int         stride_h,
        int         stride_w,
        int         dh,
        int         dw,
        int         have_bias
);

void add_reduce_full_layer(
        void*       p_bmcpl,
        const char* input_name,
        const char* output_name,
        const int*  input_shape,
        int         input_dims,
        const int*  axis_list,
        int         axis_num,
        int         reduce_method,
        int         need_keep_dims
);

void add_space_to_batch_layer(
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
        const char* output_name
);

void add_batch_to_space_layer(
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
        const char* output_name
);

void add_identity_layer(
    void*       p_bmcpl,
    const char* input_name,
    const int*  input_shape,
    int         input_dims,
    const char* output_name
);

void add_embedding_layer(
        void *p_bmcpl,
        const char *coeff_name,
        int coeff_len,
        const float *coeff_data,
        const char *shape_name,
        int padding_idx,
        const char *output_name);

void add_cumsum_layer(
    void*  	p_bmcpl,
    const int*  input_shape,
    int     	input_shape_dim,
    const char* input_name,
    int     	dim,
    const int*  output_shape,
    int     	output_shape_dim,
    const char* output_name
    );

void add_stride_calculate_layer(
    void*   		p_bmcpl,
    int     		input_num,
    const int* const*   input_shape,
    const int*    	input_shape_dim,
    const char* const*  input_name,
    const int*    	output_shape,
    int     		output_shape_dim,
    const char*   	output_name,
    int     		op_code,
    int     		result_add,
    int     		A_is_const,
    int     		B_is_const,
    float   		A_const_val,
    float   		B_const_val,
    const int*    	offset,
    const int*    	stride,
    const int*    	B_axis_is_1
  );

void add_channel_shift_layer(
    void*   	p_bmcpl,
    const int*  input_shape,
    int     	input_shape_dim,
    const char* input_name,
    const char* output_name,
    int     	shift_dir,
    int     	shift_num
  );

void add_number_like_layer(
    void*   p_bmcpl,
    const char* input_name,
    const int* input_shape,
    const int input_dims,
    const char* output_name,
    float filled_value
);

void add_number_like_layer_v2(
    void*   p_bmcpl,
    const char* input_name,
    const int*  input_shape,
    const int input_dims,
    const char* output_name,
    void *filled_value,
    int dtype
);

void add_constant_fill_layer(
    void*   p_bmcpl,
    const char* shape_name, //must exist in graph already
    const char* output_name,
    const void *filled_value,
    int type_len);

void add_constant_fill_layer_v2(
    void*   p_bmcpl,
    const char* shape_name, //must exist in graph already
    const char* output_name,
    const void* filled_value,
    int dtype
    );

// Dtype(0:f32, 1:f16, 2:i8, 3:u8, 4:i16, 5:u16, 6:i32, 7:u32)
// is same as DATA_TYPE_T defined in bmcompiler_common.h
void add_dtype_convert_layer(
    void*   	p_bmcpl,
    const int*  input_shape,
    int     	input_shape_dim,
    const char* input_name,
    const char* output_name,
    int     	src_dtype,
    int     	dst_dtype
  );

void add_batch_matmul_layer(
    void*   p_bmcpl,

    const char*   input0_name,
    const int*    input0_shape,
    int           input0_shape_dim,
    int           input0_is_const,
    const float*  input0_data,

    const char*   input1_name,
    const int*    input1_shape,
    int           input1_shape_dim,
    int           input1_is_const,
    const float*  input1_data,

    const char*   output_name
);

// Interleave input by parameter axis and step
// Example: input  shape(3,4,6,9), axis is 3, step is 3
//          output shape(3,4,6,18).
//          000000..., 111111... => 000111000111...
void add_interleave_layer(
    void*   		p_bmcpl,
    int     		input_num,
    const int* const*   input_shape,
    const int*    	input_shape_dim,
    const char* const*  input_name,
    const int*    	output_shape,
    int     		output_shape_dim,
    const char*   	output_name,
    int     		axis,
    int     		step
  );

void add_shape_range_layer(
    void*       p_bmcpl,
    const char* begin_name,
    const char* delta_name,
    const char* end_name,
    const char* out_name
);

void add_shape_tile_layer(
    void*       p_bmcpl,
    const char* input_name,
    const int*  tile_coeff,
    int         tile_len,
    const char* output_name
);
void add_shape_tile_layer_v2(
    void*       p_bmcpl,
    const char* input_name,
    const char* tile_name,
    const char* output_name
);

void add_shape_reverse_layer(
    void*       p_bmcpl,
    const char* input_name,
    int         axis,
    const char* output_name
);

void add_shape_expand_ndims_layer(
    void*       p_bmcpl,
    const char* input_name,
    int         axis,
    int         expand_num,
    const char* output_name
);

void add_shape_cast_layer(
  void *p_bmcpl,
  const char* input_name,
  const char* output_name,
  int dst_dtype
);

void add_shape_reshape_layer(
  void *p_bmcpl,
  const char* input_name,
  const char* new_shape_name,
  const char* output_name
);

void add_shape_reduce_layer(
  void*       p_bmcpl,
  const char* input_name,
  const int*  axis_list,
  int         axis_num,
  int         keep_dims,  //true or false
  int         reduce_method,
  const char* output_name
);

void set_net_inout_layer_scale(
  void*       p_bmcpl,
  const char* net_name,
  const char* layer_name,
  const char* tensor_name,
  float scale
);

void set_tensor_layer_name(
  void*       p_bmcpl,
  const char* layer_name,
  const char* tensor_name
);

void add_priorbox_cpu_layer(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
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
    char*   output_name,
    float*  output_data
);

void add_conv_fix8b_to_fp32_layer(
  void*       p_bmcpl,
  const int*    input_shape,
  int         input_shape_dim,
  const char*   input_name,
  const int*    output_shape,
  int         output_shape_dim,
  const char*   output_name,
  const char*   layer_name,
  const float*  weight,
  const float*  bias,
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
  float   input_scale,
  float   output_scale
);

void add_yolo_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    int           n,
    int           classes,
    int           coords,
    int           background,
    int           softmax
  );


void add_yolo_fix8b_to_fp32_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int     input_dtype,
    const float   input_scale,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    int           n,
    int           classes,
    int           coords,
    int           background,
    int           softmax);


#define SSD_DETECT_OUT_MAX_NUM 200
void add_ssd_detect_out_layer(
    void*        p_bmcpl,
    const int*   input0_shape,
    int          input0_shape_dim,
    const char*  input0_name,
    const int*   input1_shape,
    int          input1_shape_dim,
    const char*  input1_name,
    const int*   input2_shape,
    int          input2_shape_dim,
    const char*  input2_name,
    const int*   output_shape,
    int          output_shape_dim,
    const char*  output_name,
    int          num_classes,
    bool         share_location,
    int          background_label_id,
    int          code_type,
    bool         variance_encoded_in_target,
    int          keep_top_k,
    float        confidence_threshold,
    float        nms_threshold,
    float        eta,
    int          top_k
    );

void add_shape_active_layer(
    void* pbmcpl,
    const char* input_name,
    const char* output_name,
    int active_op
    );

void add_cpu_layer_v2(
    void*               p_bmcpl,
    int                 input_num,
    const char* const*  input_names,         /* input1 name, input2 name...*/
    const int* const*   input_shapes,        /* input1 shape, input2 shape */
    const int*          input_dims,    /* input1 dim, input2 dim,... */
    const int*          input_dtypes,        /* bm_data_type_t: DTYPE_FP32 ... */
    int                 output_num,
    const char* const*  output_names,
    const int* const*   output_shapes,
    const int*          output_dims,
    const int*          output_dtypes,      /* bm_data_type_t: DTYPE_FP32 ... */
    int                 op_type,
    const void*         layer_param,        /* bmnetc --> cpu.so, not parse in compiler */
    int                 param_size
    );

void add_user_cpu_layer_v2(
    void*               p_bmcpl,
    int                 input_num,
    const char* const*  input_names,         /* input1 name, input2 name...*/
    const int* const*   input_shapes,        /* input1 shape, input2 shape */
    const int*          input_dims,    /* input1 dim, input2 dim,... */
    const int*          input_dtypes,        /* bm_data_type_t: DTYPE_FP32 ... */
    int                 output_num,
    const char* const*  output_names,
    const int* const*   output_shapes,
    const int*          output_dims,
    const int*          output_dtypes,      /* bm_data_type_t: DTYPE_FP32 ... */
    const void*         layer_param,        /* bmnetc --> cpu.so, not parse in compiler */
    int                 param_size
    );

#define YOLOV3_DETECT_OUT_MAX_NUM 200
void add_yolov3_detect_out_layer(
    void*               p_bmcpl,
    int                 input_num,
    const int* const*   input_shapes,
    const int*          input_shape_dims,
    const char* const*  input_names,
    int*                output_shape,
    const int           output_shape_dim,
    const char*         output_name,
    int                 num_classes,
    int                 num_boxes,
    int                 mask_group_size,
    int                 keep_top_k,
    float               confidence_threshold,
    float               nms_threshold,
    float*              bias,
    float*              anchor_scale,
    float*              mask);

void add_lut_layer(
    void* 	      p_bmcpl,
    const int*    input_shape,
    int   	      input_shape_dim,
    const char*   input_name,
    const char*   output_name,
    const int     table_size,
    const float*  table
  );

/**
 * \brief add convolution 3d layer
 * \param  weight            the float data of convolution kernel
 * \param  bias              the float data of convolution bias
 * \param  kt                kernel time
 * \param  kh                kernel height
 * \param  kw                kernel width
 * \param  groups            the number of convolution groups
 * \param  pad_t             time padding
 * \param  pad_t_after       time padding after
 * \param  pad_h             height padding
 * \param  pad_h_after       height padding after
 * \param  pad_w             width padding
 * \param  pad_w_after       width padding after
 * \param  stride_t          time stride of kernel
 * \param  stride_h          height stride of kernel
 * \param  stride_w          width stride of kernel
 * \param  dt                dilation in time dimension
 * \param  dh                dilation in height dimension
 * \param  dw                dilation in width dimension
 * \param  have_bias         whether the convolution use bias or not, 1 use, 0 not use
 */
void add_conv3d_layer(
    void* 	  p_bmcpl,
    const int    *input_shape,
    int   	  input_shape_dim,
    const char   *input_name,
    const int    *output_shape,
    int   	  output_shape_dim,
    const char   *output_name,
    const char   *layer_name,
    const float  *weight,
    const float  *bias,
    int           kt,
    int   	  kh,
    int   	  kw,
    int   	  groups,
    int           pad_t,
    int           pad_t_after,
    int   	  pad_h,
    int   	  pad_h_after,
    int   	  pad_w,
    int   	  pad_w_after,
    int           stride_t,
    int   	  stride_h,
    int   	  stride_w,
    int           dt,
    int   	  dh,
    int   	  dw,
    int   	  have_bias
);


#ifdef __cplusplus
}
#endif

#endif
