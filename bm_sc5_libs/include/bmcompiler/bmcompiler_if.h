#ifndef BMCOMPILER_IF_H_
#define BMCOMPILER_IF_H_

/*!
* \file bmcompiler_if.h
* \brief interface of bmcompiler
*/

#ifdef __cplusplus
extern "C" {
#endif

namespace bmcompiler {

/**
 * \brief Set the optimization level.
 *
 * \param opt_level - The optimization level, can be either 1 or 2.
 */
void __bmcompile_set_net_config(int opt_level);

/**
* \brief Set the max allowed mismatch (aka delta) for fixpoint comparison
*
* \param fixpoint_cmp_margin - The delta for fixpoint comparison.
*
* \note
* Normally, the delta for fixpoint should be set to 0; In some rare cases, the delta can be
* increase to 1, considering that in case the calculations is implemented internally
* using FP32, there are may be some different calculation order or rounding behaviors.
*/
void __bmcompile_set_fixpoint_cmp_margin(int fixpoint_cmp_margin);

/**
 * \brief Set winograd flag
 *
 * \param winograd_flag - 0: do not use winograd; 1: use winograd without coeff optimization, 2: use winograd with coeff optimization;
 */
void __bmcompiler_set_winograd(int winograd_flag);

/**
 * \brief Create a bmcompiler handle
 *
 * \param chip_name - The name of the target chip, e.g., "BM1684"
 *
 * \return - An opaque handle to a bmcompiler instance
 */
void* create_bmcompiler(const char* chip_name);

/**
 * \brief Create a bmcompiler handle, and also create the output directory
 *
 * \param chip_name - The name of the target chip, e.g., "BM1684"
 * \param dir - The output directory; if not exist, it will be created.
 *
 * \return - An opaque handle to a bmcompiler instance
 */
void* create_bmcompiler_dir(const char* chip_name, const char* dir);

/**
 * \brief Finish bmcompiler: save results to disk, and destroy the bmcompiler instance and its associated resources.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 *
 * \note It must be called at the end of the process.
 */
void finish_bmcompiler(void* p_bmcpl);

/**
 * \brief Finish bmcompiler: pass result back in memory, and destroy the bmcompiler instance and its associated resources.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param[out] bmodel_data - The pointer of pointer to the bmodel data buffer in memory. The caller is expected
 *   to declare it as `void* bmodel_data = NULL`, and pass it in as `&bmodel_data`. The memory is allocated
 *   by bmcompiler, and the caller is expected to free the buffer after use, by `free(bmodel_data)`.
 * \param[out] size - The size of the bmodel data buffer, in bytes.
 */
void finish_bmcompiler_data(void* p_bmcpl, void** bmodel_data, unsigned int* size);

/**
 * \brief Delete bmcompiler instance, without saving result to disk or memory.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 */
void delete_bmcompiler(void* p_bmcpl);

/**
 * \brief Start the static compilation of bmcompiler, with default optimization level (1).
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param net_name - The name of the neural network model
 */
void __bmcompile(void* p_bmcpl, char* net_name);

/**
 * \brief Start the dynamic compilation of bmcompiler, with default optimization level (1).
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param net_name - The name of the neural network model
 */
void __bmcompile_ir(void* p_bmcpl, char* net_name);

/**
 * \brief Start the static compilation of bmcompiler with specified optimization level
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param net_name - The name of the neural network model
 * \param opt_level - The optimization level, either 1 or 2.
 */
void __bmcompile_opt(void* p_bmcpl, char* net_name, int opt_level);

/**
 * \brief Start the dynamic compilation of bmcompiler with specified optimization level
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param net_name - The name of the neural network model
 * \param opt_level - The optimization level, either 1 or 2.
 */
void __bmcompile_ir_opt(void* p_bmcpl, char* net_name, int opt_level);

/**
 * \brief Start the dynamic compilation of bmcompiler with specified optimization level, but not use multistage strategy
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param net_name - The name of the neural network model
 * \param opt_level - The optimization level, either 1 or 2.
 */
void __bmcompile_ir_opt_no_multistage(void* p_bmcpl, char* net_name, int opt_level);

/**
 * \brief Save umodel to disk
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param net_name - The file name for the output umodel
 */
void _bmcompiler_save_umodel(void* p_bmcpl, char* net_name);

/**
 * \brief Save umodel to disk, and compare layer's outputs with reference tensors
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param net_name - The name of the neural network model
 * \param input_name - Names of all input tensors of the network
 * \param input_data - Data of all input tensors of the network, corresponding to the `input_name` parameter
 * \param input_num -  Number of input tensors of the network
 * \param refer_name - Names of the reference tensors to be compared
 * \param refer_data - Data of The reference tensors, corresponding to the `refer_name` parameter
 * \param refer_num - Number of reference tensors
 */
void _bmcompiler_save_umodel_with_check(
    void* p_bmcpl,
    char* net_name,
    char* input_name[],
    float** input_data,
    int input_num,
    char* refer_name[],
    float** refer_data,
    int refer_num);

/**
 * \brief Compile the network model with comparing outputs with reference tensors
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_name - Names of all input tensors of the network
 * \param input_data - Data of all input tensors of the network, corresponding to the `input_name` parameter
 * \param input_num -  Number of input tensors of the network
 * \param refer_name - Names of the reference tensors to be compared
 * \param refer_data - Data of The reference tensors, corresponding to the `refer_name` parameter
 * \param refer_num - Number of reference tensors
 * \param net_name - The name of the neural network model
 */
void compile_with_result_check(
    void* p_bmcpl,
    char** input_name,
    float** input_data,
    int input_num,
    char** refer_name,
    float** refer_data,
    int refer_num,
    char* net_name);

/**
 * \brief This is just another name of the API `compile_with_result_check()`.
 *
 * \note
 * This API was created for bmnetm python interface only (wxc 20181128 for bmnetm),
 * it's kept here only for backward compatibility.
 */
void __compile_with_result_check_py(
    void* p_bmcpl,
    char* input_name[],
    float** input_data,
    int input_num,
    char* refer_name[],
    float** refer_data,
    int refer_num,
    char* net_name);

/**
 * \brief Assign a name to a layer; If the layer already has a name, the old name will be replace by the new name.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param layer_id - The unique id of the layer
 * \param layer_name - The name to be assigned to the layer
 */
void add_layer_name(void* p_bmcpl, int layer_id, const char * layer_name);

/**
 * \brief Get the maximum layer id of the current network model
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 */
int get_max_layer_id(void* p_bmcpl);

/**
 * \brief Enable or disable the profiling functionality, which is disabled by default.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param enable - The flag to enable (true) or disable (false) profiling.
 */
void set_bmcompiler_profile(void* p_bmcpl, bool enable);

/**
 * \brief Enable or disable output whitelist mode. If enabled, outputs only in
 *        whitelist (from OUTPUT_LAYER) will be kept. This is disabled by default.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param enable - The flag to enable (true) or disable (false) output whitelist.
 */
void set_enable_output_whitelist(void* p_bmcpl, bool enable);

/**
 * \brief Enable or disable showing bmlang type in network graph. This is disabled by default.
 *        When enabled, bmlang info will be added to layer type when showing graph.
 *
 * \param flag - The flag to enable (true) or disable (false) it.
 */
void set_layer_bmlang_flag(bool flag);

/**
 * \brief Get the layer bmlang flag
 *
 * \return `ture` if layer bmlang flag is enabled; `false` otherwise.
 */
bool get_layer_bmlang_flag();

/**
 * \brief `use_max_as_th` is a flag telling whether or not use max value as threshold in fp32-->int8 calibration.
 *        This API is to set cpu layer's use_max_as_th flags
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param use_max_as_th - The flags for each out tensor.
 * \param num - Number flags (for each output tensor).
 */
void set_cpu_layer_use_max_as_th(void *p_bmcpl, bool use_max_as_th[], int num);


/**
 * ######################################################################
 * \brief The following are API for add layers to bmcompiler. After creating
 * a bmcompiler handle `p_bmcpl`, one needs to add each layer to bmcompiler
 * via add_xxx_layer() APIs. Below are common parameters used in those APIs:
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_num - The number of layer's input tensors
 * \param input_shape - The tensor shape of each input. It's a one dimension array when input_num==1, and two dimension array when input_num > 1
 * \param input_shape_dim - The dimension of input shape, e.g., the shape dimension of (n, c, h, w) is 4
 * \param input_name - The name(s) of input tensor(s)
 * \param output_num - The number of layer's output tensors
 * \param output_shape_dim - The dimension(s) of output shape(s)
 * \param output_name - The name(s) of output tensor(s)
 * \param layer_name - The name of the layer
 * ######################################################################
 */


/**
 * \brief Add Convolution layer
 *
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
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    const char*   layer_name,
    const float*  weight,
    const float*  bias,
    int           kh,
    int           kw,
    int           groups,
    int           pad_h_up,
    int           pad_h_down,
    int           pad_w_left,
    int           pad_w_right,
    int           stride_h,
    int           stride_w,
    int           dh,
    int           dw,
    int           have_bias);

/**
 * \brief Add Convolution layer v2, that the weight_name and bias_name can be set by user
 *
 * \param weight_name - The name of the weight tensor
 * \param bias_name - The name of the bias tensor
 */
void add_conv_layer_v2(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    const char*   weight_name,
    const float*  weight,
    const char*   bias_name,
    const float*  bias,
    int           kh,
    int           kw,
    int           groups,
    int           pad_h_up,
    int           pad_h_down,
    int           pad_w_left,
    int           pad_w_right,
    int           stride_h,
    int           stride_w,
    int           dh,
    int           dw,
    int           have_bias);

/**
 * \brief Add Deconvolution layer
 *
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
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    const char*   layer_name,
    const float*  weight,
    const float*  bias,
    int           kh,
    int           kw,
    int           groups,
    int           pad_h_up,
    int           pad_h_down,
    int           pad_w_left,
    int           pad_w_right,
    int           stride_h,
    int           stride_w,
    int           dh,
    int           dw,
    int           have_bias);

/**
 * \brief Add Crop layer
 *
 * \param shape_name: crop shape as the tensor with shape_name. if shape_name == NULL, use output_shape
 * \param offsets: crop start offsets, which contain 4 elements: {n_offset, c_offset, h_offset, w_offset}
 */
void add_crop_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const char*   shape_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    const int*    offsets);

/**
 * \brief Add Crop layer v2, adding a mask parameter to indicate which dimension should be cropped.
 *
 * \param crop_mask: bitN=1 means Nth-dim will not be cropped
 */
void add_crop_layer_v2(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const char*   shape_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    const int*    offsets,
    unsigned int  crop_mask);

/**
 * \brief add Pooling layer
 *
 * \param  kh                kernel height
 * \param  kw                kernel width
 * \param  up_pad_h          top padding in height dimension
 * \param  down_pad_h        bottom padding in height dimension
 * \param  left_pad_w        left padding in width dimension
 * \param  right_pad_w       right padding in width dimension
 * \param  stride_h          height stride of kernel
 * \param  stride_w          widtth stride of kernel
 * \param  is_avg_pooling    average pooling or not. 1 is, 0 not
 * \param  avg_pooling_mode  0 -- all position is divided by a constant(kh * kw)
 *                           1 -- Pad value is elimated in average computation.
 * \param  is_global_pooling If global_pooling then it will pool over the size of the bottom by doing
 *                           kernel_h = bottom->height and kernel_w = bottom->width
 * \param  out_ceil_mode:    used for top tensor shape calculation
 *                           this parameter is needed for dynamic compiling.
 *                           0 -- floor mode
 *                           1 -- ceil mode
 *                           2 -- caffe mode (default)
 * \param coeff_mask         For BM1682: coefficients for mask output, when output_numer=2;
 *                           For BM1684: just pass a NULL pointer.
 */
void add_pooling_layer(
    void*              p_bmcpl,
    const int*         input_shape,
    int                input_shape_dim,
    const char*        input_name,
    int                output_number,
    const int* const*  output_shape,
    const int*         output_shape_dim,
    const char* const* output_name,
    int                kh,
    int                kw,
    int                up_pad_h,
    int                down_pad_h,
    int                left_pad_w,
    int                right_pad_w,
    int                stride_h,
    int                stride_w,
    int                is_avg_pooling,
    int                avg_pooling_mode,
    int                is_global_pooling,
    int                out_ceil_mode,
    const char*        layer_name,
    const float*       coeff_mask);

/**
 * \brief Add Pooling3D layer, with added parameter `kt` and `front_pad_t/back_pad_t`,
 *        as compared to API add_pooling_layer().
 *
 * \param  kt                kernel time
 * \param  front_pad_t       time padding
 * \param  back_pad_t        time padding
 */
void add_pooling3d_layer(
    void*              p_bmcpl,
    const int*         input_shape,
    int                input_shape_dim,
    const char*        input_name,
    int                output_number,
    const int* const*  output_shape,
    const int*         output_shape_dim,
    const char* const* output_name,
    int                kt,
    int                kh,
    int                kw,
    int                front_pad_t,
    int                back_pad_t,
    int                up_pad_h,
    int                down_pad_h,
    int                left_pad_w,
    int                right_pad_w,
    int                stride_t,
    int                stride_h,
    int                stride_w,
    int                is_avg_pooling,
    int                avg_pooling_mode,
    int                is_global_pooling,
    int                out_ceil_mode,
    const char*        layer_name,
    const float*       coeff_mask);

/**
 * \brief Add Interp layer
 *
 * \param pad_beg - padding amount at begining, for both H and W
 * \param pad_end - padding amount at end, for both H and W
 * \param platform_sp - interpolation type:
 *                           0 -- caffe bilinear
 *                           1 -- tensorflow bilinear
 *                           2 -- caffe nearest
 *                           3 -- tensorflow nearest
 *                           4 -- pytorch bilinear
 *                           5 -- pytorch nearest
 */
void add_interp_layer(
   void*       p_bmcpl,
   const int*  input_shape,
   int         input_shape_dim,
   const char* input_name,
   const int*  output_shape,
   int         output_shape_dim,
   const char* output_name,
   int         pad_beg,
   int         pad_end,
   int         platform_sp);

/**
 * \brief Add Interpolation layer v2
 *
 * \param shape_is_fixed - A boolean flag (0 for false, 1 for true). If true, output_shape
 *        and output_shape_dim should be provided; If false, output_shape_name should be provided,
          and it must be a shape tensor's name that been added already
 * \param pad_beg - Same as add_interp_layer()
 * \param pad_end - Same as add_interp_layer()
 * \param align_corners
 * \param half_pixel_centers
 * \param platform_sp - Same as add_interp_layer()
 */
void add_interp_layer_v2(
   void*         p_bmcpl,
   const int*    input_shape,
   int           input_shape_dim,
   const char*   input_name,
   int           shape_is_fixed,
   const int*    output_shape,
   int           output_shape_dim,
   const char*   output_shape_name,
   const char*   output_name,
   int           pad_beg,
   int           pad_end,
   int           align_corners,
   int           half_pixel_centers,
   int           platform_sp);

/**
 * \brief add_rpnproposal_layer
 *
 * \param input_shape       The 1st input is `rpn_cls_prob_reshape`. The shape of the 1st input
 * \param input_shape_dim   The rank of the 1st input
 * \param input_name        The name of the 1st input
 * \param input_shape1      The 2nd input is `rpn_bbox_pred`. The shape of the 2nd input
 * \param input_shape_dim1  The rank of the 2nd input
 * \param input_name1       The name of the 2nd input
 * \param input_shape2      The 3rd input is `im_info`. The shape of the 3rd input
 * \param input_shape_dim2  The rank of the 3rd input
 * \param input_name2       The name of the 3rd input
 * \param output_shape      The name of the output (`rois`)
 * \param output_shape_dim  The shape of the output
 * \param output_name       The name of the output
 * \param feat_stride_      Each pixel in the feature map (of the backbone network for RPN) represents an area of size [feat_stride_, feat_stride_] from the standard input image.
 * \param base_size_        The side length (in pixels within the original input image) of the base square anchorbox.
 * \param min_size_         The minimum side length (in pixels within the original input image) of any anchorbox.
 * \param pre_nms_topN_     Sort anchorboxes according to their foreground scores, and keep `pre_nms_topN_` of them, before NMS.
 * \param post_nms_topN_    After NMS, sort anchorboxes again, and keep `post_nms_topN_` of them.
 * \param nms_thresh_       The IoU threshold used in NMS for anchorboxes.
 * \param score_thresh_     The foreground score threshold for filtering anchorboxes, i.e., anchroboxes with foreground scores lower than the threshold will be discarded.
 */
void add_rpnproposal_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    input_shape1,
    int           input_shape_dim1,
    const char*   input_name1,
    const int*    input_shape2,
    int           input_shape_dim2,
    const char*   input_name2,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    int           feat_stride_,
    int           base_size_,
    int           min_size_,
    int           pre_nms_topN_,
    int           post_nms_topN_,
    float         nms_thresh_,
    float         score_thresh_);

/**
 * \brief add_psroipooling_layer
 *
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param input_shape1
 * \param input_shape_dim1
 * \param input_name1
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param output_dim
 * \param group_size
 * \param spatial_scale_
 * \param roi_nums
 */
void add_psroipooling_layer(
    void*       p_bmcpl,
    const int*  input_shape,
    int         input_shape_dim,
    const char* input_name,
    const int*  input_shape1,
    int         input_shape_dim1,
    const char* input_name1,
    const int*  output_shape,
    int         output_shape_dim,
    const char* output_name,
    int         output_dim,
    int         group_size,
    float       spatial_scale_,
    int         roi_nums
  );

/**
 * \brief add_roipooling_layer
 *
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param input_shape1
 * \param input_shape_dim1
 * \param input_name1
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param pooled_h
 * \param pooled_w
 * \param spatial_scale_
 * \param roi_nums
 */
void add_roipooling_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    input_shape1,
    int           input_shape_dim1,
    const char*   input_name1,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    int           pooled_h,
    int           pooled_w,
    float         spatial_scale_,
    int           roi_nums
    );

/**
 * \brief add_adaptivepooling_layer
 *
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param pooled_h
 * \param pooled_w
 */
void add_adaptivepooling_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    int           pooled_h,
    int           pooled_w
    );

/**
 * \brief add_shufflechannel_layer
 *
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param group_
 */
void add_shufflechannel_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    int           group_
    );

/**
 * \brief add_dropout_layer: just for connecting the graph, it will be remove during optimizing phase
 *
 */
void add_dropout_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name
  );

/**
 * \brief add_upsample_layer
 *
 * \param size: upsample size
 */
void add_upsample_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    int           size);

/**
 * \brief add full connect layer: output = input x weight + bias, where
 * - input shape is [M, K], M is batch_size, K is num_input_neuron;
 * - weight shape is [K, N], N is num_output_neuron;
 * - bias shape is [1, N];
 * - output shape is [M, N];
 *
 * \param num_input_neuron              K, the number of input neurons of the layer, after lumping according to the layer parameter \ref axis.
 * \param num_output_neuron             N, the number of output neurons of the layer.
 * \param weight                        The float data of fc weight.
 * \param bias                          The float data of fc bias.
 * \param have_bias                     1: have, 0: not have
 * \param weight_col_is_in_neruon_num   If 0, weight shape is [K, N] ; Otherwise, weight shape is [N, K] (Caffe default).
 */
void add_fc_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    const char*   layer_name,
    int           num_input_neuron,
    int           num_output_neuron,
    const float*  weight,
    const float*  bias,
    int           have_bias,
    int           weight_col_is_in_neruon_num
  );

/**
 * \brief add_fc_weight_layer
 *
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param layer_name
 * \param weight_shape
 * \param weight_shape_dim
 * \param weight_name
 * \param num_input_neuron
 * \param bias
 * \param have_bias
 * \param weight_col_is_in_neruon_num
 */
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

/**
 * \brief add_lrn_layer
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param alpha_
 * \param size_
 * \param beta_
 * \param k_
 */
void add_lrn_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    float         alpha_,
    int           size_,
    float         beta_,
    float         k_
  );

/**
 * \brief add_batchnorm_layer
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param layer_name
 * \param mean_param
 * \param variance_param
 * \param scale_val
 * \param epsilon
 * \param NormMethod
 * \param is_var_need_calc
 */
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

/**
 * \brief add_scale_layer: input * scale + bias
 *           scale can be const data or tensor. if scale is const data, scale_factor must be valid to provide data,
 *             else scale factor is provided through input_num, input_name, input_shape, input_shape_dim
 *           if have_bias, bias can be const data or tensor. if bias is const data, bias_factor must be valid to provide data,
 *             else bias factor is provided through input_num, input_name, input_shape, input_shape_dim
 * \param input_num
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param layer_name
 * \param scale_factor
 * \param bias_factor
 * \param num_axes
 * \param axis_
 * \param have_bias
 */
void add_scale_layer(
    void*              p_bmcpl,
    int                input_num,
    const int* const*  input_shape,
    const int*         input_shape_dim,
    const char* const* input_name,
    const int*         output_shape,
    int                output_shape_dim,
    const char*        output_name,
    const char*        layer_name,
    const float*       scale_factor,
    const float*       bias_factor,
    int                num_axes,
    int                axis_,
    int                have_bias
  );

/**
 * \brief
 *   add element-wise layer
 * \param
 *   op_code, 0 PRODUCT: in0*in1*...
 *            1 SUM: c0*in0 + c1*in1 + ...
 *            2 MAX: max(in0, in1, ...)
 * \param coefficients: provide c0, c1, ...
 */
void add_eltwise_layer(
    void*               p_bmcpl,
    int                 input_num,
    const int* const*   input_shape,
    const int*          input_shape_dim,
    const char* const*  input_name,
    const int*          output_shape,
    int                 output_shape_dim,
    const char*         output_name,
    int                 op_code,
    const float*        coefficients
  );

/**
 * \brief
 *   add concat layer
 * \param
 *    concat_axis, 0 batch size, 1 channels, 2 height, 3 width
 */
void add_concat_layer(
    void*               p_bmcpl,
    int                 input_num,
    const int* const*   input_shape,
    const int*          input_shape_dim,
    const char* const*  input_name,
    const int*          output_shape,
    int                 output_shape_dim,
    const char*         output_name,
    int                 concat_axis
  );

/**
 * \brief add_prelu_layer: do f(x) = x if x>=0 else x*slope, x shape is (n,c,h,w)
 * \param channel_shared: if channel_shared == 1, length of slope_ is 1, else  length of slope_ is c
 * \param slope_: provide slope data
 */

void add_prelu_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    const char*   layer_name,
    int           channel_shared,
    const float*  slope_
  );

/**
 * \brief add_multiregion_layer
 * \param p_bmcpl
 * \param input_num
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param classes
 * \param coords
 * \param nums
 * \param Activate_parm
 */
void add_multiregion_layer(
    void*               p_bmcpl,
    int                 input_num,
    const int* const*   input_shape,
    const int*          input_shape_dim,
    const char* const*  input_name,
    const int*          output_shape,
    int                 output_shape_dim,
    const char*         output_name,
    int                 classes,
    int                 coords,
    int                 nums,
    const int*          Activate_parm
  );

/**
 * \brief add_reorg_layer
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param stride
 * \param reverse
 */
void add_reorg_layer(
    void*               p_bmcpl,
    const int*            input_shape,
    int                 input_shape_dim,
    const char*            input_name,
    const int*        output_shape,
    int                 output_shape_dim,
    const char*         output_name,
    int                 stride,
    int                 reverse
  );

/**
 * \brief add_priorbox_layer
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param output_data
 * \param min_len
 * \param min_size
 * \param max_len
 * \param max_size
 * \param ratio_len
 * \param aspect_ratio
 * \param flip
 * \param clip
 * \param var_len
 * \param variance
 * \param img_h
 * \param img_w
 * \param step_h
 * \param step_w
 * \param offset
 */
void add_priorbox_layer(
    void*               p_bmcpl,
    const int*          input_shape,
    int                 input_shape_dim,
    const char*         input_name,
    const int*          output_shape,
    int                 output_shape_dim,
    const char*         output_name,
    const float*        output_data,
    /* for save umodel */
    int                 min_len,
    const             float*  min_size,
    int                 max_len,
    const float*        max_size,
    int                 ratio_len,
    const float*        aspect_ratio,
    int                 flip,
    int                 clip,
    int                 var_len,
    const float*        variance,
    int                 img_h,
    int                 img_w,
    float               step_h,
    float               step_w,
    float               offset
  );

/**
 * \brief add_permute_layer: transpose the input tensor as permute_order
 * \param permute_order: provide transpose order, length of order must be input_shape_dim
 */
void add_permute_layer(
    void*               p_bmcpl,
    const int*          input_shape,
    int                 input_shape_dim,
    const char*         input_name,
    const int*          output_shape,
    int                 output_shape_dim,
    const char*         output_name,
    const int*          permute_order
  );

/**
 * \brief add_reverse_layer: reverse the data on axis
 * \param axis
 */
void add_reverse_layer(
    void*               p_bmcpl,
    const int*          input_shape,
    int                 input_shape_dim,
    const char*         input_name,
    const int*          output_shape,
    int                 output_shape_dim,
    const char*         output_name,
    int                 axis
  );

/**
 * \brief add_normalize_layer
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param layer_name
 * \param across_spatial
 * \param channel_shared
 * \param scale_data
 * \param eps
 */
void add_normalize_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    const char*   layer_name,
    int           across_spatial,
    int           channel_shared,
    const float*  scale_data,
    float         eps
  );

/**
 * \brief add_flatten_layer: [DEPRECATED] use add_reshape_layer_v2 instead
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param raw_param
 */
void add_flatten_layer(
    void*        p_bmcpl,
    const int*   input_shape,
    int          input_shape_dim,
    const char*  input_name,
    const int*   output_shape,
    int          output_shape_dim,
    const char*  output_name,
    const int*   raw_param
  );

/**
 * \brief add_flatten_layer_v2: flatten shape[begin_dim, end_dim)
 * \param p_bmcpl
 * \param input_name
 * \param input_shape
 * \param input_dim
 * \param output_name
 * \param begin_dim
 * \param end_dim
 */
void add_flatten_layer_v2(
    void*       p_bmcpl,
    const char* input_name,
    const int*  input_shape,
    int         input_dim,
    const char* output_name,
    int         begin_dim,
    int         end_dim
  );

/**
 * \brief add_reshape_layer: try to use add_reshape_layer_v2 instead
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param raw_param
 */
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
 */
/**
 * \brief add_reshape_layer_v2
 * \param new_shape
 * \param new_dims
 *  each shape value in the new_shape can be the following cases (according MXNet)
 *  a. positive value,
 *  b. 0(means using corresponding bottom shape),
 *  c. -1(means auto calulated, can only contain one),
 *  d. -2(copy all/remainder of the input dimensions to the output shape)
 *  e. -3(use the product of two consecutive dimensions of the input shape as the output dimension.)
 *  f. -4(split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).)
 */
void add_reshape_layer_v2(
    void*        p_bmcpl,
    const char*  input_name,
    const int*   input_shape,
    int          input_dim,
    const char*  output_name,
    const int*   new_shape,
    int          new_dims
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
 * \param active_type_id: value from BmActiveType in bmcompiler_op_code.h
 *            ACTIVE_TANH      = 0,
 *            ACTIVE_SIGMOID   = 1,
 *            ACTIVE_RELU      = 2,
 *            ACTIVE_EXP       = 3,
 *            ACTIVE_ELU       = 4,
 *            ACTIVE_SQRT      = 5,
 *            ACTIVE_SQUARE    = 6,
 *            ACTIVE_RSQRT     = 7,
 *            ACTIVE_ABSVAL    = 8,
 *            ACTIVE_LN        = 9,
 *            ACTIVE_ROUND     = 10,
 *            ACTIVE_CEIL      = 11,
 *            ACTIVE_FLOOR     = 12,
 *            ACTIVE_SIN       = 13,
 *            ACTIVE_COS       = 14,
 *            ACTIVE_IS_FINITE = 15,
 *            ACTIVE_MISH      = 16,
 *            ACTIVE_SWISH     = 17,  // only for fix8b currently
 *            ACTIVE_HSWISH    = 18   // only for fix8b currently
 */
void add_active_layer(
    void*        p_bmcpl,
    const int*   input_shape,
    int          input_shape_dim,
    const char*  input_name,
    const int*   output_shape,
    int          output_shape_dim,
    const char*  output_name,
    int          active_type_id
  );

/**
 * \brief add_cpu_layer
 * \param op_type: value from bmcpu_common.h
 * \param layer_param
 * \param param_size
 */
void add_cpu_layer(
    void*               p_bmcpl,
    int                 input_num,
    const int* const*   input_shape,
    const int*          input_shape_dim,
    const char* const*  input_name,
    int                 output_num,
    const int* const*   output_shape,
    const int*          output_shape_dim,
    const char* const*  output_name,
    int                 op_type,
    const void*         layer_param,
    int                 param_size
    );

/**
 * \brief add_user_cpu_layer
 * \param p_bmcpl
 * \param input_num
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_num
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param layer_param
 * \param param_size
 */
void add_user_cpu_layer(
    void*               p_bmcpl,
    int                 input_num,
    const int* const*   input_shape,
    const int*          input_shape_dim,
    const char* const*  input_name,
    int                 output_num,
    const int* const*   output_shape,
    const int*          output_shape_dim,
    const char* const*  output_name,
    const void*         layer_param,
    int                 param_size
    );

/**
 * \brief add_relu_layer: f(x)= min(x, upper_limit) if x>=0 else x*negative_slope
 * \param negative_slope
 * \param upper_limit: upper_limit<0 means no upper_limit
 */
void add_relu_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    float         negative_slope,
    float         upper_limit
  );

/**
 * \brief add_softmax_layer: do softmax(x) on axis
 * \param inner_num: product(input_dim[axis+1]...)
 * \param outer_num: product(input_dims[0],...,input_dims[axis-1])
 * \param softmax_dim: input_dim[axis]
 */
void add_softmax_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    int           inner_num,
    int           outer_num,
    int           softmax_dim
  );

/**
 * \brief add_log_softmax_layer: do log(softmax(x)) on axis
 * \param inner_num: product(input_dim[axis+1]...)
 * \param outer_num: product(input_dims[0],...,input_dims[axis-1])
 * \param softmax_dim: input_dim[axis]
 */
void add_log_softmax_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    int           inner_num,
    int           outer_num,
    int           softmax_dim
  );

/**
 * \brief add_split_layer: just for connecting the graph, from caffe
 */
void add_split_layer(
    void*               p_bmcpl,
    const int*          input_shape,
    int                 input_shape_dim,
    const char*         input_name,
    int                 output_num,
    const int* const*   output_shape,
    const int*          output_shape_dim,
    const char* const*  output_name
  );

/**
 * \brief add_lstm_layer
 * \param p_bmcpl
 * \param input_num
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_num
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param layer_name
 * \param batch_num
 * \param time_num
 * \param input_dim
 * \param output_dim
 * \param user_define_cont
 * \param with_x_static
 * \param expose_hidden
 * \param x_weight
 * \param x_bias
 * \param h_weight
 * \param x_static_weight
 */
void add_lstm_layer(
    void*               p_bmcpl,
    int                 input_num,
    const int* const*   input_shape,
    const int*          input_shape_dim,
    const char* const*  input_name,
    int                 output_num,
    const int* const*   output_shape,
    const int*          output_shape_dim,
    const char* const*  output_name,
    const char*         layer_name,
    int                 batch_num,
    int                 time_num,
    int                 input_dim,
    int                 output_dim,
    int                 user_define_cont,
    int                 with_x_static,
    int                 expose_hidden,
    const float*        x_weight,
    const float*        x_bias,
    const float*        h_weight,
    const float*        x_static_weight
  );

/**
 * \brief add_pad_layer: only support input_shape_dim<=4
 * \param paddings_: data of multiple {pad_front, pad_back}, ...
 * \param pad_dim: must be 2*input_shape_dim
 * \param pad_val: constant pad value
 * \param pad_mode: 0: constant pad using pad_val
 *                  1: reflect mode
 *                    [[1,2,3],[4,5,6]] pad(1,1,2,2) to
 *                    [[6, 5, 4, 5, 6, 5, 4],
 *                     [3, 2, 1, 2, 3, 2, 1],
 *                     [6, 5, 4, 5, 6, 5, 4],
 *                     [3, 2, 1, 2, 3, 2, 1]]
 *                  2: symmetric mode
 *                    [[1,2,3],[4,5,6]] pad(1,1,2,2) to
 *                    [[2, 1, 1, 2, 3, 3, 2],
 *                     [2, 1, 1, 2, 3, 3, 2],
 *                     [5, 4, 4, 5, 6, 6, 5],
 *                     [5, 4, 4, 5, 6, 6, 5]]
 */
void add_pad_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    const int*    paddings_,
    int           pad_dim,
    float         pad_val,
    int           pad_mode
  );

/**
 * \brief add_arg_layer
 * \param p_bmcpl
 * \param axis: do argmin or argmax on axis
 * \param method: 0: argmax, 1: argmin
 */
void add_arg_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    int           axis,
    int           method
  );

/**
 * \brief add_pooling_tf_layer
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param kh
 * \param kw
 * \param up_pad_h
 * \param down_pad_h
 * \param left_pad_w
 * \param right_pad_w
 * \param stride_h
 * \param stride_w
 * \param is_avg_pooling
 */
void add_pooling_tf_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int           output_shape_dim,
    const char*   output_name,
    int           kh,
    int           kw,
    int           up_pad_h,
    int           down_pad_h,
    int           left_pad_w,
    int           right_pad_w,
    int           stride_h,
    int           stride_w,
    int           is_avg_pooling);

/**
 * \brief add_transpose_layer: do arbitray fixed transpose
 * \param shape: input shape
 * \param permute_order: length is dims
 * \param dims: input dims
 */
void add_transpose_layer(
    void*         p_bmcpl,
    const char*   input_name,
    const char*   output_name,
    const int*    shape,
    const int*    permute_order,
    int           dims
  );

/**
 * \brief add_transpose_layer_v2
 * \param shape: input shape
 * \param permute_order: length is dims
 * \param dims: input dims
 */
/**
 * \brief add_transpose_layer_v2: do arbitray transpose
 * \param order_name: use the tensor with order_name as transpose order,
 *            order_name can be NULL if permute_order provided
 * \param permute_order: permute_order can be NULL if order_name provided
 */
void add_transpose_layer_v2(
    void*         p_bmcpl,
    const char*   input_name,
    const int*    input_shape,
    int           input_dims,
    int           input_dtype,
    const char*   output_name,
    const char*   order_name,
    const int*    permute_order
  );

/**
 * \brief add_coeff_layer: insert a coeff to graph
 * \param p_bmcpl
 * \param name
 * \param shape
 * \param data
 * \param dims
 */
void add_coeff_layer(
    void*        p_bmcpl,
    const char*  name,
    const int*   shape,
    const void*  data,
    int          dims
  );

/**
 * \brief add_const_data_layer: const_in -> out
 * \param p_bmcpl
 * \param input_name
 * \param output_name
 * \param shape
 * \param data
 * \param dims
 */
void add_const_data_layer(
        void*        p_bmcpl,
        const char*  input_name,
        const char*  output_name,
        const int*   shape,
        const void*  data,
        int          dims
        );

/**
 * \brief add_select_layer: do cond? s0: s1
 * \param p_bmcpl
 * \param cond_name
 * \param s0_name: if s0_is_scalar=0, s0_name must be provided
 * \param s0_is_scalar: s0 can be tensor or const scalar
 * \param s0_scalar: if s0_is_scalar=1, s0_scalar must be provided
 * \param s1_name: if s1_is_scalar=0, s1_name must be provided
 * \param s1_is_scalar: s1 can be tensor or const scalar
 * \param s1_scalar: if s1_is_scalar=1, s1_scalar must be provided
 * \param output_name
 * \param shape
 * \param dims
 */
void add_select_layer(
        void*  p_bmcpl,
        const char* cond_name,
        const char* s0_name,
        const int   s0_is_scalar,
        const float s0_scalar,
        const char* s1_name,
        const int   s1_is_scalar,
        const float s1_scalar,
        const char* output_name,
        const int*  shape,
        const int   dims
        );

/**
 * \brief add_where_layer
 * \param p_bmcpl
 * \param cond_name
 * \param output_name
 * \param in_shape
 * \param out_shape
 * \param in_dims
 * \param out_dims
 */
void add_where_layer(
        void*       p_bmcpl,
        const char* cond_name,
        const char* output_name,
        const int*  in_shape,
        const int*  out_shape,
        const int   in_dims,
        const int   out_dims
        );

/**
 * \brief add_binary_layer_v2: do `a op b`, such as a+b, a-b, a*b...
 *      support inputs with different dims
 * \param p_bmcpl
 * \param input_A_name
 * \param input_A_shape
 * \param input_A_dim
 * \param A_is_coeff
 * \param A_data
 * \param input_B_name
 * \param input_B_shape
 * \param input_B_dim
 * \param B_is_coeff
 * \param B_data
 * \param output_name
 * \param binary_op: value from BmBinaryType in bmcompiler_op_code.h
 *    BINARY_ADD, BINARY_MUL, BINARY_MAX, BINARY_SUB, BINARY_MIN, BINARY_DIV, ...
 */
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
        int           binary_op
        );

/**
 * \brief add_binary_layer_v2: do `a op b`, such as a+b, a-b, a*b...
 *      inputs must have same shape dim
 * \param p_bmcpl
 * \param input_A_name
 * \param input_A_shape
 * \param A_is_coeff
 * \param A_data
 * \param input_B_name
 * \param input_B_shape
 * \param B_is_coeff
 * \param B_data
 * \param input_dim
 * \param output_name
 * \param binary_op
 */
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

/**
 * \brief add_broadcast_binary_layer_v2: use add_binary_layer_v2 instead
 */
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
        int           binary_op /* BINARY_ADD, BINARY_MUL, BINARY_MAX, BINARY_SUB, BINARY_MIN, BINARY_DIV */
        );

/**
 * \brief add_broadcast_binary_layer: use add_binary_layer_v2 instead
 * \param p_bmcpl
 * \param input_A_name
 * \param input_A_shape
 * \param A_is_coeff
 * \param A_data
 * \param input_B_name
 * \param input_B_shape
 * \param B_is_coeff
 * \param B_data
 * \param input_dim
 * \param output_name
 * \param binary_op
 */
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

/**
 * \brief add_const_binary_layer: do `A op c` or `c op A`, such as A+2, 3-A, ...
 * \param binary_op: value from BmBinaryType in bmcompiler_op_code.h
 *    BINARY_ADD, BINARY_MUL, BINARY_MAX, BINARY_SUB, BINARY_MIN, BINARY_DIV, ...
 * \param inversed: if inversed==1 do `B op A`, else do `A op B`
 */
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

/**
 * \brief add_eltwise_binary_layer: use add_binary_layer_v2 instead
 */
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

/**
 * \brief add_tile_layer_v2
 * \param p_bmcpl
 * \param input_name
 * \param input_shape
 * \param input_dim
 * \param input_is_coeff: input_is_coeff=1 means input is const, and input_data must be valid
 * \param input_data
 * \param coeff_is_fixed: coeff_is_fixed=1 means tile param is const, and tile_coeff must be valid
 * \param coeff_name: use tensor with coeff_name as tile param
 * \param tile_coeff: constant tile param
 * \param output_name
 * \param type: 0: tile mode [1,2,3] tile(2) to [1,2,3,1,2,3]
 *              1: repeat mode [1,2,3] repeat(2) to [1,1,2,2,3,3]
 */
void add_tile_layer_v2(
        void*        p_bmcpl,
        const char*  input_name,
        const int*   input_shape,
        int          input_dim,
        int          input_is_coeff,
        const float* input_data,
        int          coeff_is_fixed,
        const char*  coeff_name, //when is not fixed, name must be a valid shape tensor name
        const int*   tile_coeff,
        const char*  output_name,
        int type /* 0: Tile; 1: Repeat */
        );

/**
 * \brief add_expand_layer: similar to tile mode, just set final output_shape insteal tile param
 * \param p_bmcpl
 * \param input_name
 * \param input_shape
 * \param input_dim
 * \param input_is_coeff
 * \param input_data
 * \param output_shape_is_fixed
 * \param output_shape_name
 * \param output_shape
 * \param output_dim
 * \param output_name
 */
void add_expand_layer(
        void*         p_bmcpl,
        const char*   input_name,
        const int*    input_shape,
        int           input_dim,
        int           input_is_coeff,
        const float*  input_data,
        int           output_shape_is_fixed,
        const char*   output_shape_name,
        const int*    output_shape,
        int           output_dim,
        const char*   output_name);

/**
 * \brief add_repeat_layer: add_tile_layer repeat mode
 * \param p_bmcpl
 * \param input_name
 * \param input_shape
 * \param is_coeff
 * \param data
 * \param tile_coeff
 * \param input_dim
 * \param output_name
 */
void add_repeat_layer(
        void*         p_bmcpl,
        const char*   input_name,
        const int*    input_shape,
        int           is_coeff,
        const float*  data,
        const int*    tile_coeff,
        int           input_dim,
        const char*   output_name
        );

/**
 * \brief add_reduce_layer: 4D tensor reduce, special cases for add_reduce_full_layer
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param reduce_method: (reduce_method & 0xFFFF): values from BmReduceType in bmcompiler_op_code.h
 *                       (reduce_method >> 16) = 0 : reduce nhw, (reduce_method >> 16) = 1: reduce w
 *
 */
void add_reduce_layer(
        void*       p_bmcpl,
        const int*  input_shape,
        int         input_shape_dim,
        const char* input_name,
        const int*  output_shape,
        int         output_shape_dim,
        const char* output_name,
        int         reduce_method
        );

/**
 * \brief add_stride_slice_layer_v2: a litte complicated, please refer to tensorflow: tf.strided_slice(...)
 * \param p_bmcpl
 * \param input_name
 * \param input_shape
 * \param input_shape_dim
 * \param output_name
 * \param begin_index: slice begin index for each dim
 * \param end_index: slice end index for each dim
 * \param strides: slice stride for each dim, all stride values must be positive
 * \param index_size: index_size must be equal to input_shape_dim
 * \param begin_mask: i-th bit is 1, means begin_index[i] = 0
 * \param end_mask: i-th bit is 1, means end_index[i] = input_shape[i]
 * \param shrink_axis_mask: i-th bit is 1, means squeezing i-th dim
 * \param new_axis_mask: i-th bit is 1, means expand i-th dim
 * \param ellipsis_mask: i-th bit is 1, means not slice i-th dim
 */
void add_stride_slice_layer_v2(
        void*   p_bmcpl,
        const char*   input_name,
        const int*    input_shape,
        int     input_shape_dim,
        const char*   output_name,
        const int*    begin_index,
        const int*    end_index,
        const int*    strides,
        int           index_size,
        int           begin_mask,
        int           end_mask,
        int           shrink_axis_mask,
        int           new_axis_mask,
        int           ellipsis_mask
        );

/**
 * \brief add_stride_slice_layer
 * \param p_bmcpl
 * \param shape_size: index_size must be equal to input_shape_dim
 * \param begin_mask: i-th bit is 1, means begin_index[i] = 0
 * \param end_mask: i-th bit is 1, means end_index[i] = input_shape[i]
 * \param begin_index: slice begin index for each dim
 * \param end_index: slice end index for each dim
 * \param stride: slice stride for each dim, all stride values must be positive
 */
void add_stride_slice_layer(
        void*         p_bmcpl,
        const int*    input_shape,
        int           input_shape_dim,
        const char*   input_name,
        const int*    output_shape,
        int           output_shape_dim,
        const char*   output_name,
        int           shape_size,
        int           begin_mask,
        int           end_mask,
        const int*    begin_index,
        const int*    end_index,
        const int*    stride
    );

/**
 * \brief add_slice_like_layer: slice INPUT0 as INPUT1's shape on some axes
 * \param p_bmcpl
 * \param axis: slice axis list
 * \param axis_num: length of slice axis list
 */
void add_slice_like_layer(
    void*               p_bmcpl,
    const int* const*   input_shape,
    const int*          input_shape_dim,
    const char* const*  input_name,
    const int*          output_shape,
    int                 output_shape_dim,
    const char*         output_name,
    const int*          axis,
    int                 axis_num
    );

/**
 * \brief add_upsamplemask_layer
 * \param p_bmcpl
 * \param input_num
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param layer_name
 * \param mask_coeff
 */
void add_upsamplemask_layer(
    void*               p_bmcpl,
    int                 input_num,
    const int* const*   input_shape,
    const int*          input_shape_dim,
    const char* const*  input_name,
    const int*          output_shape,
    int                 output_shape_dim,
    const char*         output_name,
    const char*         layer_name,
    const float*        mask_coeff
  );


/**
 * \brief add_tf_split_layer: split a tensor on axis into multiple parts
 *  Note: when split_num=1, split_size[0] as the real split_num
 *             and use same split_size=input_shape[axis]/split_size[0]
 * \param p_bmcpl
 * \param shape_dim: same as input_shape_dim
 * \param axis: do split on axis dim
 * \param split_size
 * \param split_num
 */
void add_tf_split_layer(
     void*              p_bmcpl,
     const int*          input_shape,
     int                 input_shape_dim,
     const char*         input_name,
     int                 output_num,
     const int* const*   output_shape,
     const int*          output_shape_dim,
     const char* const*  output_name,
     int                 shape_dim,
     int                 axis,
     const int*          split_size,
     int                 split_num
  );

/**
 * \brief add_topk_layer
 * \param k
 * \param dim: do topk on this dim
 */
void add_topk_layer(
    void*               p_bmcpl,
    const int*          input_shape,
    int                 input_shape_dim,
    const char*         input_name,
    int                 k,
    int                 dim,
    const int* const*   output_shape,
    const int*          output_shape_dim,
    const char* const*  output_name
  );

/**
 * \brief add_output_layer: use add_output_layer_v2 instead
 * \param p_bmcpl
 * \param io_shape
 * \param io_shape_dim
 * \param input_name: a name that is already in the graph as final output
 * \param output_name: new name for output
 */
void add_output_layer(
    void*              p_bmcpl,
    const int*         io_shape,
    int                io_shape_dim,
    const char*        input_name,
    const char*        output_name
  );

/**
 * \brief add_output_layer_v2: mark tensor as graph final output
 * \param p_bmcpl
 * \param p_bmcpl
 * \param name
 */
void add_output_layer_v2(
    void*              p_bmcpl,
    const char*        name
  );

/**
 * \brief add_shape_slice_layer: slice for 1-dim tensor, output=input[begin:end:step]
 * \param shape_name: input_name
 * \param begin
 * \param end
 * \param step
 * \param slice_name: output name
 */
void add_shape_slice_layer(
        void*       p_bmcpl,
        const char* shape_name,
        int         begin,
        int         end,
        int         step,
        const char* slice_name
);

/**
 * \brief add_shape_slice_layer_v2: same as add_stride_slice_layer_v2, use it instead
 */
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

/**
 * \brief add_shape_slice_layer_v3: same as add_stride_slice_layer_v2, use it instead
 */
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

/**
 * \brief add_shape_pack_layer: concat multiple 1D tensors into one
 * \param shape_names: input tensor names
 * \param shape_num: input tensor number
 * \param pack_name: output name
 */
void add_shape_pack_layer(
        void*               p_bmcpl,
        const char* const*  shape_names,
        int                 shape_num,
        const char*         pack_name
);

/**
 * \brief add_shape_pack_layer_v2: same as add_concat_layer
 * \param p_bmcpl
 * \param shape_names: input names
 * \param shape_num: input numbers
 * \param axis: concat axis
 * \param pack_name: output name
 */
void add_shape_pack_layer_v2(
        void*               p_bmcpl,
        const char* const*  shape_names,
        int                 shape_num,
        int                 axis,
        const char*         pack_name
);

/**
 * \brief add_shape_const_layer: add 1D const shape tensor
 * \param p_bmcpl
 * \param shape_data
 * \param shape_length: input length
 * \param shape_name
 */
void add_shape_const_layer(
        void*       p_bmcpl,
        const int*  shape_data,
        int         shape_length,
        const char* shape_name
);

/**
 * \brief add_shape_const_layer_v2: enhanced add_shape_const_layer
 * \param p_bmcpl
 * \param data
 * \param shape
 * \param dims
 * \param name
 */
void add_shape_const_layer_v2(
        void*       p_bmcpl,
        const int*  data,    //shape's data
        const int*  shape,   //shape's shape
        int         dims,    //shape's shape's dims
        const char* name     //shape's name
        );

/**
 * \brief add_shape_op_layer: do `A op B` for shape tensor
 * \param p_bmcpl
 * \param in0_name
 * \param in1_name
 * \param binary_op: value from BmBinaryType in bmcompiler_op_code.h
 * \param out_name
 */
void add_shape_op_layer(
        void*         p_bmcpl,
        const char*   in0_name,
        const char*   in1_name,
        int           binary_op,
        const char*   out_name
);

/**
 * \brief add_shape_ref_layer: use input's shape as output tensor
 * \param p_bmcpl
 * \param input_name
 * \param input_shape
 * \param input_dim
 * \param shape_name: output names
 */
void add_shape_ref_layer(
        void*   p_bmcpl,
        const char*   input_name,
        const int*    input_shape,
        int           input_dim,
        const char*   shape_name
);

/**
 * \brief add_shape_addn_layer: do sum(in0, in1, ...)
 * \param p_bmcpl
 * \param input_names
 * \param input_num
 * \param output_name
 */
void add_shape_addn_layer(
       void*   p_bmcpl,
       const char* const* input_names,
       const int input_num,
       const char* output_name
);

/**
 * \brief add_rank_layer: use rank of input as output tensor
 * \param p_bmcpl
 * \param input_name
 * \param input_shape
 * \param input_dim
 * \param output_name
 */
void add_rank_layer(
        void*         p_bmcpl,
        const char*   input_name,
        const int*    input_shape,
        int           input_dim,
        const char*   output_name
);

/**
 * \brief add_squeeze_layer: squeeze input tensor dims
 * \param axis_list
 * \param axis_num: if axis_num=0, means axis_list=[0,1,...input_dim-1]
 * \param output_name
 */
void add_squeeze_layer(
        void*         p_bmcpl,
        const char*   input_name, //must exist already
        const int*    input_shape,
        int           input_dim,
        const int*    axis_list,
        int           axis_num, //0 means removal of all '1' dims
        const char*   output_name
);

/**
 * \brief add_expand_ndims_layer: expand input tensor shape N dims before axis
 *     shape=[2], N=2, axis=0 => [1,1,2]
 *     shape=[2], N=2, axis=1 => [2,1,1]
 *     shape=[2], N=2, axis=-1 => [2,1,1]
 * \param axis: when less than 0, real axis is input_dim + axis + 1
 * \param ndims: N
 */
void add_expand_ndims_layer(
        void*         p_bmcpl,
        const char*   input_name, //must exist already
        const int*    input_shape,
        int           input_dim,
        const float*  input_data, //input data is not Null means it is coeff
        int           axis,
        int           ndims,
        const char*   output_name);

/**
 * \brief add_expand_dims_layer:  expand input tensor shape 1 dims before axis
 * \param axis: when less than 0, real axis is input_dim + axis + 1
 */
void add_expand_dims_layer(
        void*         p_bmcpl,
        const char*   input_name, //must exist already
        const int*    input_shape,
        int           input_dim,
        int           axis,
        const char*   output_name
);

/**
 * \brief add_shape_assign_layer: dynamic reshape, assign a shape tensor as input tensor' shape
 * \param input_name
 * \param input_shape
 * \param input_dim
 * \param shape_name: shape tensor's name
 * \param output_name
 */
void add_shape_assign_layer(
        void*         p_bmcpl,
        const char*   input_name,
        const int*    input_shape,
        int           input_dim,
        const char*   shape_name,
        const char*   output_name
);

/**
 * \brief add_shape_reorder_layer: reorder 1D shape tensor's values according to shape_order
 *          shape tensor's value is [2,3,4,5], shape_order is [1,3,2,0] then output tensor value is [3,5,4,2]
 * \param p_bmcpl
 * \param shape_name
 * \param shape_order
 * \param order_num: length of shape order, must same as shape tenosr's length
 * \param output_name
 */
void add_shape_reorder_layer(
        void*       p_bmcpl,
        const char* shape_name,
        const int*  shape_order,
        int         order_num,  //must be the same length of shape_dims
        const char* output_name
);

/**
 * \brief add_ref_crop_layer: use tensor as crop param,
 *           crop param tensor must have shape [4,2]
 * \param crop_name: crop param tensor's name
 */
void add_ref_crop_layer(
        void*         p_bmcpl,
        const char*   input_name,
        const int*    input_shape,
        int           input_dim,
        const char*   crop_name,
        const char*   output_name
);

/**
 * \brief add_ref_pad_layer: use tensor as pad param,
 *           pad param tensor must have shape [4,2]
 * \param pad_name: pad param tensor's name
 * \param pad_value: constant pad value
 * \param pad_mode: 0: constant pad using pad_val
 *                  1: reflect mode
 *                    [[1,2,3],[4,5,6]] pad(1,1,2,2) to
 *                    [[6, 5, 4, 5, 6, 5, 4],
 *                     [3, 2, 1, 2, 3, 2, 1],
 *                     [6, 5, 4, 5, 6, 5, 4],
 *                     [3, 2, 1, 2, 3, 2, 1]]
 *                  2: symmetric mode
 *                    [[1,2,3],[4,5,6]] pad(1,1,2,2) to
 *                    [[2, 1, 1, 2, 3, 3, 2],
 *                     [2, 1, 1, 2, 3, 3, 2],
 *                     [5, 4, 4, 5, 6, 6, 5],
 *                     [5, 4, 4, 5, 6, 6, 5]]
 */
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

/**
 * \brief add_conv_weight_layer: supports conv weights is not const
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param layer_name
 * \param weight_shape
 * \param weight_shape_dim
 * \param weight_name
 * \param bias
 * \param groups
 * \param pad_h_up
 * \param pad_h_down
 * \param pad_w_left
 * \param pad_w_right
 * \param stride_h
 * \param stride_w
 * \param dh
 * \param dw
 * \param have_bias
 */
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

/**
 * \brief add_reduce_full_layer: arbitary reduce
 * \param p_bmcpl
 * \param axis_list: reduce axis list
 * \param axis_num: axis list num, if axis_num=0, reduce all axes
 * \param reduce_method: values from BmReduceType in bmcompiler_op_code.h
 * \param need_keep_dims: if =0, squeeze the reduced dims
 */
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

/**
 * \brief add_space_to_batch_layer: just for 4D tensor, refer tf.sapce_to_batch
 * \param p_bmcpl
 * \param input_name
 * \param input_shape
 * \param input_dim
 * \param block_is_dynamic
 * \param block_name: if dynamic, must be an existed shape tensor name
 * \param block_sizes: if not dynamic, must be valid--{hblock,wblock}
 * \param pad_is_dynamic
 * \param pad_name: if dynamic, must be an existed shape tensor name
 * \param pad_sizes: if not dynamic, must be valid--{ht_pad, hb_pad, wl_pad, wr_pad}
 * \param output_name
 */
void add_space_to_batch_layer(
        void*      p_bmcpl,
        const char* input_name,
        const int*  input_shape,
        int         input_dim,
        int         block_is_dynamic,
        const char* block_name,
        const int*  block_sizes,
        int         pad_is_dynamic,
        const char* pad_name,
        const int*  pad_sizes,
        const char* output_name
);

/**
 * \brief add_batch_to_space_layer: just from 4D tensor, refer to tf.batch_to_space
 * \param p_bmcpl
 * \param input_name
 * \param input_shape
 * \param input_dim
 * \param block_is_dynamic
 * \param block_name: if dynamic, must be an existed shape tensor name
 * \param block_sizes: if not dynamic, must be valid--{hblock,wblock}
 * \param crop_is_dynamic
 * \param crop_name: if dynamic, must be an existed shape tensor name
 * \param crop_sizes: if not dynamic, must be valid--{ht_crop, hb_crop, wl_crop, wr_crop}
 * \param output_name
 */
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

/**
 * \brief add_identity_layer: just for connecting graph
 * \param p_bmcpl
 * \param input_name
 * \param input_shape
 * \param input_dims
 * \param output_name
 */
void add_identity_layer(
    void*       p_bmcpl,
    const char* input_name,
    const int*  input_shape,
    int         input_dims,
    const char* output_name
);

/**
 * \brief add_embedding_layer
 * \param p_bmcpl
 * \param coeff_name
 * \param coeff_len
 * \param coeff_data
 * \param shape_name
 * \param padding_idx
 * \param output_name
 */
void add_embedding_layer(
        void*        p_bmcpl,
        const char*  coeff_name,
        int          coeff_len,
        const float* coeff_data,
        const char*  shape_name,
        int          padding_idx,
        const char*  output_name
        );

/**
 * \brief add_cumsum_layer: accumulated sum on one dim
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param dim
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 */
void add_cumsum_layer(
    void*       p_bmcpl,
    const int*  input_shape,
    int         input_shape_dim,
    const char* input_name,
    int         dim,
    const int*  output_shape,
    int         output_shape_dim,
    const char* output_name
    );

/**
 * \brief add_stride_calculate_layer
 * \param p_bmcpl
 * \param input_num
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param op_code
 * \param result_add
 * \param A_is_const
 * \param B_is_const
 * \param A_const_val
 * \param B_const_val
 * \param offset
 * \param stride
 * \param B_axis_is_1
 */
void add_stride_calculate_layer(
    void*               p_bmcpl,
    int                 input_num,
    const int* const*   input_shape,
    const int*          input_shape_dim,
    const char* const*  input_name,
    const int*          output_shape,
    int                 output_shape_dim,
    const char*         output_name,
    int                 op_code,
    int                 result_add,
    int                 A_is_const,
    int                 B_is_const,
    float               A_const_val,
    float               B_const_val,
    const int*          offset,
    const int*          stride,
    const int*          B_axis_is_1
  );

/**
 * \brief add_channel_shift_layer
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_name
 * \param shift_dir
 * \param shift_num
 */
void add_channel_shift_layer(
    void*       p_bmcpl,
    const int*  input_shape,
    int         input_shape_dim,
    const char* input_name,
    const char* output_name,
    int         shift_dir,
    int         shift_num
  );

/**
 * \brief add_number_like_layer: output a tensor with input's shape and values are filled with filled_value
 *           filled_value=1.0, means ones_like
 *           filled_value=0.0, means zeros_like
 * \param p_bmcpl
 * \param input_name
 * \param input_shape
 * \param input_dims
 * \param output_name
 * \param filled_value
 */
void add_number_like_layer(
    void*        p_bmcpl,
    const char*  input_name,
    const int*   input_shape,
    const int    input_dims,
    const char*  output_name,
    float        filled_value
);

/**
 * \brief add_number_like_layer_v2: output a tensor with input's shape and assigned dtype
 *               its values are filled with filled_value
 * \param p_bmcpl
 * \param input_name
 * \param input_shape
 * \param input_dims
 * \param output_name
 * \param filled_value
 * \param dtype
 */
void add_number_like_layer_v2(
    void*       p_bmcpl,
    const char* input_name,
    const int*  input_shape,
    const int   input_dims,
    const char* output_name,
    void*       filled_value,
    int         dtype
);

/**
 * \brief add_constant_fill_layer: output a tensor with shape from input tensor's values
 *               its values are filled with filled_value
 * \param p_bmcpl
 * \param shape_name: input tensor's name
 * \param output_name
 * \param filled_value
 * \param type_len
 */
void add_constant_fill_layer(
    void*       p_bmcpl,
    const char* shape_name, //must exist in graph already
    const char* output_name,
    const void* filled_value,
    int         type_len);

/**
 * \brief add_constant_fill_layer_v2: output a tensor with shape from input tensor's values and assigned dtype
 *               its values are filled with filled_value
 * \param p_bmcpl
 * \param shape_name
 * \param output_name
 * \param filled_value
 * \param dtype
 */
void add_constant_fill_layer_v2(
    void*       p_bmcpl,
    const char* shape_name, //must exist in graph already
    const char* output_name,
    const void* filled_value,
    int         dtype
    );

/**
 * \brief add_dtype_convert_layer: convert tensor's data type
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_name
 * \param src_dtype
 *        Dtype(0:f32, 1:f16, 2:i8, 3:u8, 4:i16, 5:u16, 6:i32, 7:u32)
 * \param dst_dtype
 *        Dtype(0:f32, 1:f16, 2:i8, 3:u8, 4:i16, 5:u16, 6:i32, 7:u32)
 */
void add_dtype_convert_layer(
    void*       p_bmcpl,
    const int*  input_shape,
    int         input_shape_dim,
    const char* input_name,
    const char* output_name,
    int         src_dtype,
    int         dst_dtype
  );

/**
 * \brief add_batch_matmul_layer
 * \param p_bmcpl
 * \param input0_name
 * \param input0_shape
 * \param input0_shape_dim
 * \param input0_is_const
 * \param input0_data
 * \param input1_name
 * \param input1_shape
 * \param input1_shape_dim
 * \param input1_is_const
 * \param input1_data
 * \param output_name
 */
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

/**
 * \brief add_interleave_layer
 * Interleave input by parameter axis and step
 * Example: input  shape(3,4,6,9), axis is 3, step is 3
 *          output shape(3,4,6,18).
 *          000000..., 111111... => 000111000111...
 * \param p_bmcpl
 * \param input_num
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param axis
 * \param step
 */
void add_interleave_layer(
    void*               p_bmcpl,
    int                 input_num,
    const int* const*   input_shape,
    const int*          input_shape_dim,
    const char* const*  input_name,
    const int*          output_shape,
    int                 output_shape_dim,
    const char*         output_name,
    int                 axis,
    int                 step
  );

/**
 * \brief add_shape_range_layer: use begin_tensor, delta_tensor, end_tensor's values to generate a range (begin:end:delta) output
 * \param p_bmcpl
 * \param begin_name: begin tensor's name
 * \param delta_name: delta tensor's name
 * \param end_name:   end tensor's name
 * \param out_name
 */
void add_shape_range_layer(
    void*       p_bmcpl,
    const char* begin_name,
    const char* delta_name,
    const char* end_name,
    const char* out_name
);

/**
 * \brief add_shape_tile_layer: use add_tile_layer_v2 instead
 */
void add_shape_tile_layer(
    void*       p_bmcpl,
    const char* input_name,
    const int*  tile_coeff,
    int         tile_len,
    const char* output_name
);

/**
 * \brief add_shape_tile_layer_v2: use add_tile_layer_v2 instead
 */
void add_shape_tile_layer_v2(
    void*       p_bmcpl,
    const char* input_name,
    const char* tile_name,
    const char* output_name
);

/**
 * \brief add_shape_reverse_layer: use add_reverse_layer instead
 */
void add_shape_reverse_layer(
    void*       p_bmcpl,
    const char* input_name,
    int         axis,
    const char* output_name
);

/**
 * \brief add_shape_expand_ndims_layer: use add_expand_ndims_layer instead
 */
void add_shape_expand_ndims_layer(
    void*       p_bmcpl,
    const char* input_name,
    int         axis,
    int         expand_num,
    const char* output_name
);

/**
 * \brief add_shape_cast_layer: use add_dtype_convert_layer instead
 */
void add_shape_cast_layer(
  void *p_bmcpl,
  const char* input_name,
  const char* output_name,
  int dst_dtype
);

/**
 * \brief add_shape_reshape_layer: use add_shape_assign_layer instead
 */
void add_shape_reshape_layer(
  void *p_bmcpl,
  const char* input_name,
  const char* new_shape_name,
  const char* output_name
);

/**
 * \brief add_shape_reduce_layer: use add_reduce_full_layer instead
 */
void add_shape_reduce_layer(
  void*       p_bmcpl,
  const char* input_name,
  const int*  axis_list,
  int         axis_num,
  int         keep_dims,  //true or false
  int         reduce_method,
  const char* output_name
);

/**
 * \brief set_net_inout_layer_scale
 * \param p_bmcpl
 * \param net_name
 * \param layer_name
 * \param tensor_name
 * \param scale
 */
void set_net_inout_layer_scale(
  void*       p_bmcpl,
  const char* net_name,
  const char* layer_name,
  const char* tensor_name,
  float scale
);

/**
 * \brief set_tensor_layer_name
 * \param p_bmcpl
 * \param layer_name
 * \param tensor_name
 */
void set_tensor_layer_name(
  void*       p_bmcpl,
  const char* layer_name,
  const char* tensor_name
);


/**
 * \brief add_priorbox_cpu_layer
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param min_sizes
 * \param real_min_size
 * \param max_sizes
 * \param real_max_size
 * \param aspect_ratios
 * \param real_spect_size
 * \param variance
 * \param real_variance_size
 * \param num_priors
 * \param img_w
 * \param img_h
 * \param step_w
 * \param step_h
 * \param offset
 * \param thTop
 * \param bottom_0_width
 * \param bottom_0_height
 * \param bottom_1_width
 * \param bottom_1_height
 * \param dim
 * \param has_dim
 * \param flip
 * \param clip
 * \param output_name
 * \param output_data
 */
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

/**
 * \brief add_conv_fix8b_to_fp32_layer
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param layer_name
 * \param weight
 * \param bias
 * \param kh
 * \param kw
 * \param groups
 * \param pad_h_up
 * \param pad_h_down
 * \param pad_w_left
 * \param pad_w_right
 * \param stride_h
 * \param stride_w
 * \param dh
 * \param dw
 * \param have_bias
 * \param in_sign
 * \param out_sign
 * \param input_scale
 * \param output_scale
 */
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

/**
 * \brief add_yolo_layer
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param n
 * \param classes
 * \param coords
 * \param background
 * \param softmax
 */
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


/**
 * \brief add_yolo_fix8b_to_fp32_layer
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param input_dtype
 * \param input_scale
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param n
 * \param classes
 * \param coords
 * \param background
 * \param softmax
 */
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
/**
 * \brief add_ssd_detect_out_layer
 * \param p_bmcpl
 * \param input0_shape
 * \param input0_shape_dim
 * \param input0_name
 * \param input1_shape
 * \param input1_shape_dim
 * \param input1_name
 * \param input2_shape
 * \param input2_shape_dim
 * \param input2_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param num_classes
 * \param share_location
 * \param background_label_id
 * \param code_type
 * \param variance_encoded_in_target
 * \param keep_top_k
 * \param confidence_threshold
 * \param nms_threshold
 * \param eta
 * \param top_k
 */
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

/**
 * \brief add_shape_active_layer: use add_active_layer instead
 */
void add_shape_active_layer(
    void* pbmcpl,
    const char* input_name,
    const char* output_name,
    int active_op
    );

/**
 * \brief add_cpu_layer_v2: enhanced add_cpu_layer, it can set input/output dtype
 * \param p_bmcpl
 * \param input_num
 * \param input_names
 * \param input_shapes
 * \param input_dims
 * \param input_dtypes
 * \param output_num
 * \param output_names
 * \param output_shapes
 * \param output_dims
 * \param output_dtypes
 * \param op_type
 * \param layer_param
 * \param param_size
 */
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

/**
 * \brief add_user_cpu_layer_v2: enhanced add_user_cpu_layer, it can set input/output dtype
 * \param p_bmcpl
 * \param input_num
 * \param input_names
 * \param input_shapes
 * \param input_dims
 * \param input_dtypes
 * \param output_num
 * \param output_names
 * \param output_shapes
 * \param output_dims
 * \param output_dtypes
 * \param layer_param
 * \param param_size
 */
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
/**
 * \brief add_yolov3_detect_out_layer
 * \param p_bmcpl
 * \param input_num
 * \param input_shapes
 * \param input_shape_dims
 * \param input_names
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param num_classes
 * \param num_boxes
 * \param mask_group_size
 * \param keep_top_k
 * \param confidence_threshold
 * \param nms_threshold
 * \param bias
 * \param anchor_scale
 * \param mask
 */
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

/**
 * \brief add_lut_layer: lookup table layer
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_name
 * \param table_size
 * \param table
 */
void add_lut_layer(
    void*         p_bmcpl,
    const int*    input_shape,
    int           input_shape_dim,
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
    void*         p_bmcpl,
    const int    *input_shape,
    int           input_shape_dim,
    const char   *input_name,
    const int    *output_shape,
    int           output_shape_dim,
    const char   *output_name,
    const char   *layer_name,
    const float  *weight,
    const float  *bias,
    int           kt,
    int           kh,
    int           kw,
    int           groups,
    int           pad_t,
    int           pad_t_after,
    int           pad_h,
    int           pad_h_after,
    int           pad_w,
    int           pad_w_after,
    int           stride_t,
    int           stride_h,
    int           stride_w,
    int           dt,
    int           dh,
    int           dw,
    int           have_bias
);

/**
 * \brief add_unfold_layer
 * \param p_bmcpl
 * \param input_shape
 * \param input_shape_dim
 * \param input_name
 * \param output_shape
 * \param output_shape_dim
 * \param output_name
 * \param axis
 * \param size
 * \param step
 */
void add_unfold_layer(
    void*         p_bmcpl,
    const int    *input_shape,
    int           input_shape_dim,
    const char   *input_name,
    const int    *output_shape,
    int           output_shape_dim,
    const char   *output_name,
    int           axis,
    int           size,
    int           step);

/**
 * \brief add_matrix_band_part_layer
 * \param p_bmcpl
 * \param input_shape
 * \param input_dim
 * \param input_name
 * \param output_shape
 * \param output_dim
 * \param output_name
 * \param lower
 * \param upper
 */
void add_matrix_band_part_layer(
        void *p_bmcpl,
        const int *input_shape,
        int input_dim,
        const char *input_name,
        const int *output_shape,
        int output_dim,
        const char *output_name,
        int lower,
        int upper);

void add_gru_layer(
        void *p_bmcpl,
        const int *input_shapes_0,
        int input_dim_0,
        const char *input_name_0,
        const int *input_shapes_1,
        int input_dim_1,
        const char *input_name_1,
        const int *output_shape_0,
        int output_dim_0,
        const char *output_name_0,
        const int *output_shape_1,
        int output_dim_1,
        const char *output_name_1,
        const float *weights,
        const float *bias,
        bool bidirection,
        bool batch_first,
        const char *layer_name);

void add_pytorch_lstm_layer(
        void *p_bmcpl,
        const int *input_shapes_0,
        int input_dim_0,
        const char *input_name_0,
        const int *input_shapes_1,
        int input_dim_1,
        const char *input_name_1,
        const int *input_shapes_2,
        int input_dim_2,
        const char *input_name_2,
        const int *output_shape_0,
        int output_dim_0,
        const char *output_name_0,
        const int *output_shape_1,
        int output_dim_1,
        const char *output_name_1,
        const int *output_shape_2,
        int output_dim_2,
        const char *output_name_2,
        const float *weights,
        const float *bias,
        bool bidirection,
        bool batch_first,
        const char *layer_name);

} // namespace bmcompiler

#ifdef __cplusplus
}
#endif

#endif
