#ifndef BMCOMPILER_IF_FIX8B_H_
#define BMCOMPILER_IF_FIX8B_H_

#ifdef __cplusplus
extern "C" {
#endif

namespace bmcompiler {

/**
 * \brief Add a Reorg layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param stride - stride parameter of the Reorg operation.
 * \param reverse - Reverse the Reorg operation if not 0.
 * \param data_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 */
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
    int     data_sign
  );

/**
 * \brief Add a PriorBox layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param output_data - The pointer to reference output data.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 *
 * \note
 * For accuracy, PriorBox layer usually requires fp32 arithmetics; Please consider using the fp32 API \ref add_priorbox_cpu_layer() instead.
 */
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

/**
 * \brief Add a Priorbox layer to BMCompiler.
 *
 * \param input_shape - The shape of the input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param output_data - The pointer to reference output data.
 * \param min_sizes - The pointer to the array of min_size parameter of the layer.
 * \param real_min_size - The size of the min_sizes array.
 * \param max_sizes - The pointer to the array of max_size parameter of the layer.
 * \param real_max_size - The size of the max_sizes array.
 * \param aspect_ratios - The pointer to the array of aspect_ratio parameter of the layer.
 * \param real_aspect_ratios - The size of the aspect_ratios array.
 * \param variance - The pointer to the array of variance parameter of the layer.
 * \param real_variance_size - The size of the variance array.
 * \param num_priors - The number of priorboxes for each pixel in the feature map.
 * \param img_w - The width (in pixel) of the network's input image.
 * \param img_h - The height (in pixel) of the network's input image.
 * \param step_w - The horizontal distance (in pixel) in the network's input image between priorboxes corresponding to two adjacent feature map pixels.
 * \param step_h - The vertical distance (in pixel) in the network's input image between priorboxes corresponding to two adjacent feature map pixels.
 * \param offset - The offset (in pixel) for calculating centers of priorboxes.
 * \param thTop - The threshold (of quantization) for the output tensor. The default value is 1.
 * \param bottom_0_width - The width of the feature map.
 * \param bottom_0_height - The height of the feature map.
 * \param bottom_1_width - The width of the network's input image.
 * \param bottom_1_height - The height of the network's input image.
 * \param dim - The size of the last axis of the output tensor, i.e., the output tensor shape is [n, 2, dim], where n is the batch size.
 * \param has_dim - Wheter or not the dim argument above is valid.
 * \param flip - Whether or not flip each aspect ratio (0 for False, 1 for True).
 * \param clip - Whether or not clip prior such that it's within [0,1].
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 *
 * \note
 * Again, for accuracy, PriorBox layer usually requires fp32 arithmetics; Please consider using the fp32 API \ref add_priorbox_cpu_layer() instead.
 */
void add_priorbox_cpu_layer_fix8b(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    void*   output_data,
    float*  min_sizes,
    int     real_min_size,
    float*  max_sizes,
    int     real_max_size,
    float*  aspect_ratios,
    int     real_spect_size,
    float*  variance,
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

/**
 * \brief Add a Permute layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param permute_order - The new orders of the axes of data. Notice it should be within the same range as the input data, and it starts from 0.
 * \param data_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 */
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

/**
 * \brief Add a Normalize layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param layer_name - The name of the layer in the network model.
 * \param across_spatial - Whether or not across spatial (0 for False, 1 for True).
 * \param channel_shared - Whether or not share scale parameters across channels (0 for False, 1 for True).
 * \param scale_data - The pointer to the scale parameter tensor's data.
 * \param eps - Epsilon for not dividing by zero while normalizing variance.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 * \param coeff_sign - The sign of the scale parameter tensor (0 for unsigned, 1 for signed).
 */
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

/**
 * \brief Add a Flatten layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param data_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 */
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

/**
 * \brief Add a Reshape layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor. Notice that this parameter is now ignored, use \ref raw_param instead (below).
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param raw_param - The shape of the output tensor.
 * \param data_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 *
 * \note
 * This API is deprecated, use \ref add_reshape_layer_fix8b_v2() when possible.
 */
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

/**
 * \brief Add a Reshape layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_name - The name of the output tensor.
 * \param new_shape - The shape of the output tensor.
 * \param new_dim - The rank of the output tensor.
 * \param data_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 *
 * \note
 * This API replaces the deprecated one: \ref add_reshape_layer_fix8b().
 */
void add_reshape_layer_fix8b_v2(
    void*        p_bmcpl,
    const char*  input_name,
    const int*   input_shape,
    int          input_dim,
    const char*  output_name,
    const int*   new_shape,
    int          new_dim,
    int          data_sign
  );

/**
 * \brief Add a Active layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 * \param input_scale - The scale parameter for the input tensor.
 * \param output_scale - The scale parameter for the output tensor.
 * \param active_type_id - The type of Active method. For the supported Active types, check \ref BmActiveType.
 *
 * \note
 * This API is a duplication of another one \ref add_active_layer_fix8b(). The only difference is tha name of the APIs.
 */
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

/**
 * \brief Add a Active layer to BMCompiler.
 *
 * \note
 * This API is a duplication of another one \ref add_active_layer_fix8b_gdma(). The only difference is tha name of the APIs.
 */
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

/**
 * \brief Add a Convolution layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param layer_name - The name of the layer in the network model.
 * \param weight - The pointer to the weight data.
 * \param bias - The pointer to the bias data. Note that bias data type is fix16b.
 * \param kh - The height of the kernel.
 * \param kw - The width of the kernel.
 * \param groups - The number of groups for group convolution.
 * \param pad_xx_yy - The padding paramters in four directions.
 * \param stride_h - The stride parameter in height (vertical) direction.
 * \param stride_w - The stride parameter in width (horizontal) direction.
 * \param dh - The dilation parameter in height (vertical) direction.
 * \param dw - The dilation parameter in width (horizontal) direction.
 * \param have_bias - Whether or not have bias (1 for True, 0 for False).
 * \param in_sign - The sign of the input tensor (0 for unsigned fix8b, 1 for signed fix8b).
 * \param out_sign - The sign of the output tensor (0 for unsigned fix8b, 1 for signed fix8b).
 * \param weight_sign - The sign of the weight tensor (0 for unsigned fix8b, 1 for signed fix8b).
 * \param bias_sign - The sign of the bias tensor (0 for unsigned fix16b, 1 for signed fix16b).
 * \param rshift_num - The right-shifting size (in bits) of the layer.
 * \param use_winograd - Whether or not use winograd algorithm.
 */
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

/**
 * \brief Add a Convolution layer to BMCompiler.
 *
 * \note
 * The main difference between this API and \ref add_conv_layer_fix8b() is that,\n
 * besides fix8b data types, this API also support fix16b data types. so the sign parameters\n
 * are changed to data type parameters.\n
 * The following fix8b and fix16b data types are supported:\n
 *   signed fix8b:    DTYPE_INT8;\n
 *   unsigned fix8b:  DTYPE_UINT8;\n
 *   signed fix16b:   DTYPE_INT16;\n
 *   unsigned fix16b: DTYPE_UINT16;\n
 * For all data type definition, please check \ref DATA_TYPE_T or \ref bm_data_type_t.\n
 */
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

void add_conv3d_layer_fix8b(
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
    int           have_bias,
    // fix8b-specific parameters:
    int           in_sign,
    int           out_sign,
    int           weight_sign,
    int           bias_sign,
    int           rshift_num
 );

/**
 * \brief Add a Deconvolution layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param layer_name - The name of the layer in the network model.
 * \param weight - The pointer to the weight data.
 * \param bias - The pointer to the bias data. Note that bias data type is fix16b.
 * \param kh - The height of the kernel.
 * \param kw - The width of the kernel.
 * \param groups - The number of groups for group convolution.
 * \param pad_xx_yy - The padding paramters in four directions.
 * \param stride_h - The stride parameter in height (vertical) direction.
 * \param stride_w - The stride parameter in width (horizontal) direction.
 * \param dh - The dilation parameter in height (vertical) direction.
 * \param dw - The dilation parameter in width (horizontal) direction.
 * \param have_bias - Whether or not have bias (1 for True, 0 for False).
 * \param in_sign - The sign of the input tensor (0 for unsigned fix8b, 1 for signed fix8b).
 * \param out_sign - The sign of the output tensor (0 for unsigned fix8b, 1 for signed fix8b).
 * \param weight_sign - The sign of the weight tensor (0 for unsigned fix8b, 1 for signed fix8b).
 * \param bias_sign - The sign of the bias tensor (0 for unsigned fix16b, 1 for signed fix16b).
 * \param rshift_num - The right-shifting size (in bits) of the layer.
 */
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

/**
 * \brief Add a Crop layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param offsets - The offsets parameter of the operation.
 * \param data_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 */
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

/**
 * \brief Add a Crop layer to BMCompiler, version 2, adding a mask parameter.
 *
 * \param crop_mask - if bit0=1, it means that 0-dim will not be cropped; The same logic applies for other dimensions.
 */
void add_crop_layer_fix8b_v2(
    void*   p_bmcpl,
    int*    input_shape,
    int     input_shape_dim,
    char*   input_name,
    char*   shape_name,
    int*    output_shape,
    int     output_shape_dim,
    char*   output_name,
    int*    offsets,
    int     data_sign,
    unsigned int crop_mask);

/**
 * \brief Add a PSROIPooling layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the 1st input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the 1st input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the 1st input tensor.
 * \param input_shape1 - The shape of the 2nd input tensor.
 * \param input_shape_dim1 - The rank of the 2nd input tensor.
 * \param input_name1 - The name of the 2nd input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param output_dim - The output_dim parameter of the layer.
 * \param group_size - the group_size parameter of the layer.
 * \param spatial_scale - The spatial_scale parameter of the layer.
 * \param roi_nums - The roi_nums parameter of the layer.
 * \param in_sign - The sign of the 1st input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 */
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

/**
 * \brief Add a ROIPooling layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the 1st input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the 1st input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the 1st input tensor.
 * \param input_shape1 - The shape of the 2nd input tensor.
 * \param input_shape_dim1 - The rank of the 2nd input tensor.
 * \param input_name1 - The name of the 2nd input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param pooled_h - The pooled_h parameter of the layer.
 * \param pooled_w - The pooled_w parameter of the layer.
 * \param spatial_scale - The spatial_scale parameter of the layer.
 * \param roi_nums - The roi_nums parameter of the layer.
 * \param data_sign - The sign of the 1st intput tensor (0 for unsigned, 1 for signed).
 */
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

/**
 * \brief Add a RPN layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the 1st input tensor.
 * \param input_shape_dim - The rank of the input tensor.
 * \param input_name - The name of the 1st input tensor.
 * \param input_shape1 - The shape of the 2nd input tensor.
 * \param input_shape_dim1 - The rank of the 2nd input tensor.
 * \param input_name1 - The name of the 2nd input tensor.
 * \param input_shape2 - The shape of the 3rd input tensor.
 * \param input_shape_dim2 - The rank of the 3rd input tensor.
 * \param input_name2 - The name of the 3rd input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param feat_stride_ - Each pixel in the feature map (of the backbone network for RPN) represents an area of size [feat_stride_, feat_stride_] from the standard input image.
 * \param base_size_ - The side length (in pixels within the original input image) of the base square anchorbox.
 * \param min_size_ - The minimum side length (in pixels within the original input image) of any anchorbox.
 * \param pre_nms_topN_ - Sort anchorboxes according to their foreground scores, and keep `pre_nms_topN_` of them, before NMS.
 * \param post_nms_topN_ - After NMS, sort anchorboxes again, and keep `post_nms_topN_` of them.
 * \param nms_thresh_ - The IoU threshold used in NMS for anchorboxes.
 * \param score_thresh_ - The foreground score threshold for filtering anchorboxes, i.e., anchroboxes with foreground scores lower than the threshold will be discarded.
 * \param in_sign - The sign of the 2nd intput tensor (0 for unsigned, 1 for signed).
 * \param scale_val - The scale value for converting the 2nd input tensor from fix8b to FP32.
 *
 * \note
 * This layer has 3 inputs (only the 2nd input is fix8b, the other two inputs are FP32) and 1 output (of FP32):
 * 1: rpn_cls_prob: FP32
 * 2: rpn_bbox_pred: fix8b. To convert it to FP32, each element will be multiplied by `scale_val` parameter.
 * 3: im_info: FP32
 *
 * The processing of the layer is as below:
 * 1. Generate all anchorboxes, assuming that the number of generated anchorboxes match the 1st and 2nd input tensors.
 * 2. Update anchorbox positions using the 2nd input tensor (it should be already converted to FP32).
 * 3. Filter anchorboxes:
 *    - remove anchorboxes whose size are too small;
 *    - clip anchorboxes which are outside the input image;
 *    - remove anchorboxes whose foreground score is less than `score_thresh_`
 * 4. Sort anchorboxes according to their foreground scores, then keep the 1st `pre_nms_topn` anchorboxes
 * 5. NMS the anchorboxes using `nms_thresh` parameter
 * 6. Output the 1st `post_nms_topn` anchroboxes
 */
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

/**
 * \brief Add a Pooling layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param kh - The height of the kernel.
 * \param kw - The width of the kernel.
 * \param pad_xx_yy - The padding paramters in four directions.
 * \param stride_h - The stride parameter in height (vertical) direction.
 * \param stride_w - The stride parameter in width (horizontal) direction.
 * \param is_avg_pooling - Whether or not use average pooling method (1 for True, 0 for False).
 * \param avg_pooling_mode - Whether or not use average pooling mode (1 for True, 0 for False).
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 * \param is_global_pooling - Whether or not use global pooling method (1 for True, 0 for False).
 * \param out_ceil_mode - the round mode parameter of the layer:\n
 *   0: CEIL;\n
 *   1: FLOOR;\n
 *   2: CFDFT(Caffe Default);\n
 */
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

void add_pooling3d_layer_fix8b(
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
    const float*       coeff_mask,
    // fix8b-specific parameters:
    int                in_sign,
    int                out_sign
 );

/**
 * \brief Add a PoolingTF (TensorFlow Pooling) layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param kh - The height of the kernel.
 * \param kw - The width of the kernel.
 * \param xx_pad_yy - The padding paramters in four directions.
 * \param stride_h - The stride parameter in height (vertical) direction.
 * \param stride_w - The stride parameter in width (horizontal) direction.
 * \param is_avg_pooling - Whether or not use average pooling method (1 for True, 0 for False).
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 */
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

/**
 * \brief Add a Dropout layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param data_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 */
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

/**
 * \brief Add an Upsample (aka UpsampleCopy) layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor, which is an array of integers.
 * \param input_shape_dim - The size of the input shape array, aka the rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param size - The scale parameter of the layer.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 */
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

/**
 * \brief Add an UpsampleMask layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_num - The number of input tensors, which must be equal to 2.
 * \param input_shape[2] - The shapes of the two input tensors.
 * \param input_shape_dim[2] - The ranks of the two input tensors.
 * \param input_name[2] - The names of the two input tensors.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param layer_name - The name of the layer in the network model.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 *
 * \note
 * The 1st input is the tensor to be upsampled, and the 2nd input is\n
 * the "mask" which contains indices in the output tensor. if an index\n
 * equals to -1, then the corresponding output tensor elements is 0.\n
 * Some constraints:\n
 * - these two input tensors should have exactly the same shape;\n
 * - the 1st input tensor data type is fix8b;\n
 * - the 2nd input tensor data type is int32;\n
 */
void add_upsamplemask_layer_fix8b(
    void*   		p_bmcpl,
    int     		input_num,
    const int* const   input_shape[2],
    const int    	input_shape_dim[2],
    const char* const  input_name[2],
    const int*    	output_shape,
    int     		output_shape_dim,
    const char*   	output_name,
    const char*   	layer_name,
    int             in_sign,
    int             out_sign
  );

/**
 * \brief Add a FC (aka InnerProduct) layer to BMCompiler: output = input x weight + bias, where
 * - input shape is [M, K], M is batch_size, K is num_input_neuron;
 * - weight shape is [K, N], N is num_output_neuron;
 * - bias shape is [1, N];
 * - output shape is [M, N];
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor.
 * \param input_shape_dim - The rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param layer_name - The name of the layer in the network model.
 * \param num_input_neuron - K, the number of input neurons of the layer, after lumping according to the layer parameter \ref axis.
 * \param num_output_neuron - N, the number of output neurons of the layer.
 * \param weight - The weight data pointer of the layer.
 * \param bias - The bias data pointer of the layer. Note that bias data type is fix16b.
 * \param have_bias - Whether or not the layer uses bias (0 for False, 1 for True).
 * \param weight_col_is_in_neuron_num - If 0, weight shape is [K, N]; Otherwise, weight shape is [N, K] (i.e., Caffe default).
 * \param in_sign - The sign of the input tensor (0 for unsigned fix8b, 1 for signed fix8b).
 * \param out_sign - The sign of the output tensor (0 for unsigned fix8b, 1 for signed fix8b).
 * \param weight_sign - The sign of the weight tensor (0 for unsigned fix8b, 1 for signed fix8b).
 * \param bias_sign - The sign of the bias tensor (0 for unsigned fix16b, 1 for signed fix16b).
 * \param rshift_num - The right-shifting size (in bits) of the layer.
 */
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

/**
 * \brief Add a BatchNorm layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor.
 * \param input_shape_dim - The rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param layer_name - The name of the layer in the network model.
 * \param alpha - The data pointer to alpha (scale).
 * \param beta - The data pointer to beta (bias). y = alpha * x_norm + beta. Note that beta (bias) data type is fix16b.
 * \param in_sign - The sign of the input tensor (0 for unsigned fix8b, 1 for signed fix8b).
 * \param out_sign - The sign of the output tensor (0 for unsigned fix8b, 1 for signed fix8b).
 * \param alpha_sign - The sign of alpha (0 for unsigned fix8b, 1 for signed fix8b).
 * \param beta_sign - The sign of beta (0 for unsigned fix16b, 1 for signed fix16b).
 * \param rshift_num - The right-shifting size (in bits) of the layer.
 *
 * \note
 * The BatchNorm layer is finally converted to `y=(ax+b)>>shift` after quantization, which can be
 * treated as Scale operation . So for fix8b BatchNorm layers, it's recommended to call scale
 * layer API, i.e., add_scale_layer_fix8b() or add_scale_layer_fix8b_v2(), instead of this one.
 */
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
    void*   beta,
    int     in_sign,
    int     out_sign,
    int     alpha_sign,
    int     beta_sign,
    int*    rshift_num
  );

/**
 * \brief Add a Scale layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor.
 * \param input_shape_dim - The rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param layer_name - The name of the layer in the network model.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 * \param scale_factor - The pointer to the scale factor.
 * \param bias_factor - The pointer to the bias factor.
 * \param rshift_num - The right-shifting size (in bits) of the layer.
 *
 * \note
 * This API has the following two assumptions:\n
 * 1. The `scale` is a constant tensor, taken from scale_factor parameter;\n
 * 2. The `axis` always takes the default value 1 (i.e., start from channel);\n
 * If either of the two assumptions is not valid, please use \ref add_scale_layer_fix8b_v2() instead.\n
 */
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

/**
 * \brief Add a Scale layer to BMCompiler (API version 2).
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_num - The number of input tensors, can be either 1 or 2.
 * \param input_shape - The shapes of the input tensors.
 * \param input_shape_dim - The ranks of the input tensors.
 * \param input_name - The names of the input tensors.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param layer_name - The name of the layer in the network model.
 * \param scale_factor - The pointer to the scale factor. NULL if input_num=2.
 * \param bias_factor - The pointer to the bias factor. NULL if the layer parameter bais_term is False.
 * \param num_axes - The rank of scale factor tensor, if input_num=2; It's ignored, if input_num=1.
 * \param axis - The first axis of the 1st input tensor along which to apply the scale factor tensor.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 * \param rshift_num - The right-shifting size (in bits) of the layer.
 *
 * \note
 * 1. input_num can be 1 or 2;\n
 * 2. The 1st input (the tensor to be scaled) is mandatory, and the 2nd input (the scale factor tensor) is optional.\n
      If the 2nd input tensor is not present, use scale_factor parameter as the content of the constant scale factor\n
      tensor.
 */
void add_scale_layer_fix8b_v2(
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
    int                 axis,
    int                 in_sign,
    int                 out_sign,
    void*               rshift_num);

/**
 * \brief Add an Eltwise (Element-wise Operation) layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_num - The number of input tensors.
 * \param input_shape - The shapes of the input tensors.
 * \param input_shape_dim - The ranks of the input tensors.
 * \param input_name - The names of the input tensors.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param op_code - The operation code of the layer (0: Product; 1: Sum; 2: Max).
 * \param in_scale - the scale factor of all input tensors.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 * \param rshift_num - The right-shifting sizes (in bits) of the input tensors.
 */
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
    int*    in_scale,
    int*    in_sign,
    int     out_sign,
    int*    rshift_num
  );

/**
 * \brief Add an Interp layer to BMCompiler
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor.
 * \param input_shape_dim - The rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param pad_beg - The padding at begin of input.
 * \param pad_end - The padding at end of input.
 * \param method - Interpolation method (0: Caffe Bilinear; 1: TensorFlow Bilinear; 2: Caffe Nearest; 3: TensorFlow Nearest; 4: PyTorch Bilinear; 5: Pytorch Nearest).
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 */
void add_interp_layer_fix8b(
    void* 	      p_bmcpl,
    const int*    input_shape,
    int   	      input_shape_dim,
    const char*   input_name,
    const int*    output_shape,
    int   	      output_shape_dim,
    const char*   output_name,
    int   	      pad_beg,
    int   	      pad_end,
    int           method,
    int           in_sign
   );

/**
 * \brief Add a Concat layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_num - The number of input tensors.
 * \param input_shape - The shapes of the input tensors.
 * \param input_shape_dim - The ranks of the input tensors.
 * \param input_name - The names of the input tensors.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param concat_axis - The index of the axis along which all input tensors to be concatenated.
 * \param in_sign - The signs of the input tensors (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 * \param rshift_num - The right-shifting sizes (in bits) of the input tensors.
 * \param scale_num - The scale factors of the input tensors.
 */
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

/**
 * \brief Add a MultiRegion layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_num - The number of input tensors.
 * \param input_shape - The shapes of the input tensors.
 * \param input_shape_dim - The ranks of the input tensors.
 * \param input_name - The names of the input tensors.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param classes - The classes parameter of the layer.
 * \param coords - The coordinates parameter of the layer.
 * \param nums - The nums parameter of the layer.
 * \param Active_parm - The activate_parm parameter of the layer. Exactly 4 values should be provided.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
*/
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

/**
 * \brief Add a LRN (Local Response Normalization) layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor.
 * \param input_shape_dim - The rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param alpha_ - The alpha parameter of the layer.
 * \param size_ - The local_size parameter of the layer.
 * \param beta_ - The beta parameter of the layer.
 * \param k_ - The k parameter of the layer.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 * \param scale_in - The scale_in parameter of the layer.
 * \param scale_out - The scale_out parameter of the layer.
*/
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

/**
 * \brief Add a PReLU (Parametric ReLU) layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor.
 * \param input_shape_dim - The rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param layer_name - The name of the layer in the network model.
 * \param channel_shared - Whether or not slope parameters are shared across channels (0 for False, 1 for True).
 * \param slope_ - The slope parameter tensor of the layer.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 * \param slope_sign - The sign of the slope parameter tensor (0 for unsigned, 1 for signed).
 * \param rshift_num - The right-shifting size (in bits) of the layer.
 */
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

/**
 * \brief Add a ReLU layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor.
 * \param input_shape_dim - The rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param negative_slope - The negative_slope parameter of the layer.
 * \param upper_limit - The upper_limit parameter of the layer.
 * \param rshift_num - The right-shifting size (in bits) of the layer.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 */
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

/**
 * \brief Add a (Caffe's) Split layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor.
 * \param input_shape_dim - The rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_num - The number of output tensors.
 * \param output_shape - The shapes of the output tensors.
 * \param output_shape_dim - The ranks of the output tensors.
 * \param output_name - The names of the output tensors.
 * \param data_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 *
 * \note
 * This layer just clones the input tensor into multiple copies, with different names (from Caffe).\n
 * The functionality is different from that of \ref add_tf_split_layer_fix8b().\n
 */
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

/**
 * \brief Add a (TensorFlow's) Split layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor.
 * \param input_shape_dim - The rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_num - The number of output tensors.
 * \param output_shape - The shapes of the output tensors.
 * \param output_shape_dim - The ranks of the output tensors.
 * \param output_name - The names of the output tensors.
 * \param shape_dim - The rank of the input tensor (duplicated with \ref input_shape_dim).
 * \param axis - The axis of the input tensor along which to split.
 * \param split_size - If \ref split_num=1, it points to an integer specifying the size on \ref axis for each sub tensors.\n
 * It implies that the sub tensors are evenly split from the input tensor, so it's required that the input_shape[axis] is divisible by this integer value;\n
 * If \ref split_num>1, it points to an an array of integers specifying the sizes (on axis) for each output tensor.\n
 * It implies that the sub tensors may have different dimension on \ref axis, but it's required that the sum of the\n
 * interger array equals to input_shape[axis].
 * \param split_num - The number of elements in the \ref split_size array.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 *
 * \note
 * This layer split an input tensor into a list of sub tensors. According to parameter \ref split_size\n
 * and \ref split_num, the output sub tensors may or may not have the same shape.\n
 * In either case, it's required that the number of sub tensors equals to \ref output_num parameter.
 */
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
     int                out_sign
  );

/**
 * \brief Add a Softmax layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor.
 * \param input_shape_dim - The rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param inner_num - The number of elements of the inner tensor.
 * \param outer_num - The number of elements of the outer tensor.\n
 * If softmax_axis is the index of the input tensor's axis along which to perform softmax, then:\n
 * - inner_num = input_tensor->count(softmax_axis + 1);\n
 * - outer_num = input_tensor->count(0, softmax_axis);\n
 * where count() computes the volume of a slice (the product of dimensions among a range of axes).
 * \param softmax_dim - The dimension of the axis along which to perform the softmax.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param scale_val - The scale_in parameter (of quantization) of the layer.
 */
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

/**
 * \brief Add a ShuffleChannel layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor.
 * \param input_shape_dim - The rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param group_ - The number of group.
 * \param data_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 */
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

/**
 * \brief Add a Bias layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor.
 * \param input_shape_dim - The rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param layer_name - The name of the layer in the network model.
 * \param num_output_neuron - Must be the number of channels (input_shape[1]) of the input tensor.
 * \param bias - The pointer to the bias tensor's data.
 * \param in_sign - The sign of the input tensor (0 for unsigned fix8b, 1 for signed fix8b).
 * \param out_sign - The sign of the output tensor (0 for unsigned fix8b, 1 for signed fix8b).
 * \param bias_sign - The sign of the bias tensor (0 for unsigned fix16b, 1 for signed fix16b).
 * \param scale - The scale factor of the layer.
 * \param rshift - The right-shifting size (in bits) of the layer.
 */
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

/**
 * \brief Add a StridedSlice (of TensorFlow) layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_name - The name of the input tensor.
 * \param input_shape - The shape of the input tensor.
 * \param input_shape_dim - The rank of the input tensor.
 * \param output_name - The name of the output tensor.
 * \param begin_index - The pointer to the array of begin indices in the input tensor's axes.
 * \param end_index - The pointer to the array of end indices.
 * \param strides - The pointer to the array of strides.
 * \param index_size - The size of the begin_index array (begin_index, end_index, and strides are of the same array size).
 * \param begin_mask - The begin mask parameter of the layer.
 * \param end_mask - The end mask parameter of the layer.
 * \param shrink_axis_mask - The shrink axis mask parameter of the layer.
 * \param new_axis_mask - The new axis mask parameter of the layer.
 * \param ellipsis_mask - The ellipsys mask parameter of the layer.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 *
 * \note
 * All layer parameters follow the rule specified by the strided_slice op of Tensorflow, please refer to it for details.
 */
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

/**
 * \brief Add a pad layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor.
 * \param input_shape_dim - The rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param paddings_ - The pointer to the array containing the padding configuration\n
 * for all dimensions, once per spatial dimension. Each dimension has padding configuration,\n
 * which contains two values:\n
 * 1. padding_low: Padding amount on the low-end (next to the index 0);\n
 * 2. padding_high: Padding amount on the high-end (next to the highest index);\n
 * For example, if the input tensor rank is 4, then the array size of paddings_ is 8.
 * \param pad_dim - The number of axes to pad, usually equals to the rank of the input tensor.
 * \param pad_val - The value for the padded elements.
 * \param pad_mode - The padding method:\n
 * 0: const, value;\n
 * 1: symmtric(mirror): 3,2,1,0,|0,1,2,3,4;\n
 * 2: reflect: 4,3,2,1,|0,1,2,3,4;\n
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 */
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

/**
 * \brief Add a ConstBinary layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_A_name - The name of the 1st input tensor.
 * \param input_A_shape - The shape of the 1st input tensor.
 * \param input_A_dim - The rank of the 1st input tensor.
 * \param B_value - The value of the 2nd input, which is a constant scalar.
 * \param output_name - The name of the output tensor.
 * \param binary_op - The supported constant binary operations (check \ref BmBinaryType).
 * \param inversed - Whether or not input A and input B reverse order (0: A op B; 1: B op A).
 * \param in_sign - The signs of the two input tensors (0 for unsigned, 1 for signed); For the 2nd scalar input, the sign value is 1.
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 * \param scale - The scale factor of the input tensors. For the 2nd scalar input, the scale value is 1.
 * \param rshift_num - The right-shifting sizes (in bits) of the input tensors. For the 2nd scalar input tensor, the rshift_num value is 0.
 */
void add_const_binary_layer_fix8b(
    void*         p_bmcpl,
    const char*   input_A_name,
    const int*    input_A_shape,
    int           input_A_dim,
    float         B_value,
    const char*   output_name,
    int           binary_op,
    int           inversed,
    int*          in_sign,
    int           out_sign,
    int*          scale,
    int*          rshift_num
  );

/**
 * \brief Add an EltwiseBinary (Element-wise Binary Operation) layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_A_name - The name of the 1st input tensor.
 * \param A_is_coeff - Whether or not the 1st input tensor is a constant tensor.
 * \param A_data - The pointer to the 1st input tensor's data.
 * \param input_B_name - The name of the 2nd input tensor.
 * \param B_is_coeff - Whether or not the 2nd input tensor is a constant tensor.
 * \param B_data - The pointer to the 2nd input tensor's data.
 * \param input_shape - The shape of the two input tensors.
 * \param input_dim - The rank of the two input tensors.
 * \param output_name - The name of the output tensor.
 * \param binary_op - The operation code of the layer (check \ref BmBinaryType).
 * \param in_sign - The sign of the input tensor (0 for unsigned fix8b, 1 for signed fix8b, 2 for signed fix16b).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 * \param scale - The scale factors of the input tensors.
 * \param rshift_num - The right-shifting sizes (in bits) of the input tensors.
 */
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
    int           binary_op,
    int*          in_sign,
    int           out_sign,
    int*          scale,
    int*          rshift_num
  );

/**
 * \brief Add an EltwiseBinary Extended layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_A_name - The name of the 1st input tensor.
 * \param input_A_shape - The shape of the 1st input tensor.
 * \param input_A_dim - The rank of the 1st input tensor.
 * \param input_B_hi8_name - The name of the high 8-bit part of the 2nd input tensor. It's assumed that the 2nd input tensor is a fix16b tensor.
 * \param output_name - The name of the output tensor.
 * \param binary_op - The operation code of the layer (BINARY_MUL).
 * \param in_sign[2] - The signs of the two input tensors (0 for unsigned, 1 for signed).
 * \param input_scale[2] - The scale of the two input tensors.
 * \param input_rshift_num[2] - The right-shifting sizes (in bits) of the two input tensors.
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 *
 * \note
 * The EXtended version supports different input data widths, ie., a fix8b input A, and a fix16b const input B (coeff).\n
 * In this API:\n
 * - input A is fix8b;\n
 * - input B is fix16b (const/coeff), which has been physically split into two parts: hi8 and lo8;\n
 * - both hi8 and lo8 const tensors should already added to BMCompiler before this call (so their data pointer is not needed);\n
 * - the sign info of both inputs is passed in input_sign[2].\n
 * - two inputs and the output have the same shape;\n
 */
void add_eltwise_binary_layer_fix8b_ex(
    void*          p_bmcpl,
    const char*    input_A_name,
    const int*     input_A_shape,
    int            input_A_dim,
    const char*    input_B_hi8_name,
    const char*    input_B_lo8_name,
    const char*    output_name,
    int            binary_op,
    int            input_sign[2],
    int            input_scale[2],
    int            input_rshift_num[2],
    int            out_sign
  );

/**
 * \brief Add a Binary layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_A_name - The name of the 1st input tensor.
 * \param input_A_shape - The shape of the 1st input tensor.
 * \param input_A_dim - The rank of the 1st input tensor.
 * \param A_is_coeff - Whether or not the 1st input tensor is a constant tensor.
 * \param A_data - The pointer to the 1st input tensor's data.
 * \param input_B_name - The name of the 2nd input tensor.
 * \param input_B_shape - The shape of the 2nd input tensor.
 * \param input_B_dim - The rank of the 2nd input tensor.
 * \param B_is_coeff - Whether or not the 2nd input tensor is a constant tensor.
 * \param B_data - The pointer to the 2nd input tensor's data.
 * \param output_name - The name of the output tensor.
 * \param binary_op - The operation code of the layer (check \ref BmBinaryType).
 * \param in_sign - The sign of the input tensor (0 for unsigned fix8b, 1 for signed fix8b, 2 for signed fix16b).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 * \param scale - The scale factors of the input tensors.
 * \param rshift_num - The right-shifting sizes (in bits) of the input tensors.
 */
void add_broadcast_binary_layer_fix8b(
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
    int           binary_op,
    int*          in_sign,
    int           out_sign,
    int*          scale,
    int*          rshift_num
  );

/**
 * \brief Add a Binary layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_A_name - The name of the 1st input tensor.
 * \param input_A_shape - The shape of the 1st input tensor.
 * \param input_A_dim - The rank of the 1st input tensor.
 * \param A_is_coeff - Whether or not the 1st input tensor is a constant tensor.
 * \param A_data - The pointer to the 1st input tensor's data.
 * \param input_B_name - The name of the 2nd input tensor.
 * \param input_B_shape - The shape of the 2nd input tensor.
 * \param input_B_dim - The rank of the 2nd input tensor.
 * \param B_is_coeff - Whether or not the 2nd input tensor is a constant tensor.
 * \param B_data - The pointer to the 2nd input tensor's data.
 * \param output_name - The name of the output tensor.
 * \param binary_op - The operation code of the layer (check \ref BmBinaryType).
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 * \param scale - The scale factors for the input tensors.
 * \param rshift_num - The right-shifting sizes (in bits) of the input tensors.
 *
 * \note
 * This is wrapper of \ref add_eltwise_binary_layer_fix8b() and \ref add_broadcast_binary_layer_fix8b():
 * - If input A and input B are of the same shape, this API calls \ref add_eltwise_binary_layer_fix8b();\n
 * - Otherwise, this API calls \ref add_broadcast_binary_layer_fix8b().
  */
void add_binary_layer_fix8b(
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
    int           binary_op,
    int*          in_sign,
    int           out_sign,
    int*          scale,
    int*          rshift_num
  );

/**
 * \brief Add a Tile layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_name - The name of the input tensor.
 * \param input_shape - The shape of the input tensor.
 * \param input_dim - The rank of the input tensor.
 * \param input_is_const - Whether or not the input tensor is a constant tensor.
 * \param input_data - The pointer to the input tensor data, if the input tensor is a constant tensor; otherwise, it should be NULL;
 * \param param_is_const - Whether or not the Tile op parameters (an array of integers) are constant.
 * \param shape_tensor_name - The name of the shape tensor, whose content is the tile op parameters. If param_is_const=1, this should be NULL;
 * \param tile_param - The pointer to the constant Tile op parameters. If param_is_const=0, this should be NULL.
 * \param output_name - The name of the output tensor.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 * \param type - The type of Tile operation (0: TensorFlow's Tile; 1: MXNET's Repeat).
 *
 * \note
 * Tile layer requires two inputs:\n
 * 1. A input tensor, which can be a normal (dynamic) tensor or a constant tensor. when the input tensor\n
 *    is a constant tensor, its content comes from `input_data`; otherwise, its content is tracked\n
 *    by `input_name`.\n
 * 2. The tile parameters, which is an array of integers. The array can be either constant (in this case,\n
 *    the values come from tile_param[]), or from a shape tensor (in this case, the values come from a SHAPE\n
 *    tensor whose name is `shape_tensor_name`).\n
 *
 * Other constraints:\n
 * - The input tensor and tile parameters cannot be both constant;\n
 * - The number of integers in the parameter array must match (i.e., not less than) the rank of the input tensor;\n
 * - The type parameter can be either 0 (TF's Tile) or 1 (MXNET's Repeat).
 */
void add_tile_layer_fix8b(
    void        *p_bmcpl,
    /* the following are info about tile op's input tensor */
    const char  *input_name,
    const int   *input_shape,
    int         input_dim,
    int         input_is_const,
    const float *input_data,
    /* the following are info about tile op's parameters */
    int         param_is_const,
    const char  *shape_tensor_name,
    const int   *tile_param,
    /* other info */
    const char  *output_name,
    int         in_sign,
    int         out_sign,
    int         type
  );

/**
 * \brief Add a SpaceToBatch layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_name - The name of the input tensor.
 * \param input_shape - The shape of the input tensor.
 * \param input_dim - The rank of the input tensor.
 * \param block_is_dynamic - The block_is_dynamic parameter of the layer.
 * \param block_name - The name of a shape tensor, if block_is_dynamic=1.
 * \param block_sizes - The {hblock, wblock} parameters of the layer, if block_is_dynamic=0.
 * \param pad_is_dynamic - The pad_is_dynamic parameter of the layer.
 * \param pad_name - The name of a shape tensor, if pad_is_dynamic=1.
 * \param pad_sizes - The {ht_pad, hb_pad, wl_pad, wr_pad} parameters of the layer, if pad_is_dynamic=0.
 * \param output_name - The name of the output tensor.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 */
void add_space_to_batch_layer_fix8b(
  void*       p_bmcpl,
  const char* input_name,
  const int*  input_shape,
  int         input_dim,
  int         block_is_dynamic,
  const char* block_name,
  const int*  block_sizes,
  int         pad_is_dynamic,
  const char* pad_name,
  const int*  pad_sizes,
  const char* output_name,
  int         in_sign,
  int         out_sign
  );

/**
 * \brief Add a BatchToSpace layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_name - The name of the input tensor.
 * \param input_shape - The shape of the input tensor.
 * \param input_dim - The rank of the input tensor.
 * \param block_is_dynamic - The block_is_dynamic parameter of the layer.
 * \param block_name - The name of a shape tensor, if block_is_dynamic=1.
 * \param block_sizes - The {hblock, wblock} parameters of the layer, if block_is_dynamic=0.
 * \param crop_is_dynamic - The crop_is_dynamic parameter of the layer.
 * \param crop_name - The name of a shape tensor, if crop_is_dynamic=1.
 * \param crop_sizes - The {ht_crop, hb_crop, wl_crop, wr_crop} parameters of the layer, if crop_is_dynamic=0.
 * \param output_name - The name of the output tensor.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 */
void add_batch_to_space_layer_fix8b(
  void*       p_bmcpl,
  const char* input_name,
  const int*  input_shape,
  int         input_dim,
  int         block_is_dynamic,
  const char* block_name,
  const int*  block_sizes,
  int         crop_is_dynamic,
  const char* crop_name,
  const int*  crop_sizes,
  const char* output_name,
  int         in_sign,
  int         out_sign
  );

/**
 * \brief Add a ReduceFull layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_name - The name of the input tensor.
 * \param output_name - The name of the output tensor.
 * \param input_shape - The shape of the input tensor.
 * \param input_dim - The rank of the input tensor.
 * \param axis_list - The array of axes indices to reduce to a scalar.
 * \param axis_num - The size of axis_list.
 * \param reduce_method - The reduce operation (check \ref BmReduceType).
 * \param need_keep_dims - The need_keep_dims parameter of the layer.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 * \param input_scale - The scale factor (of quantization) for the input tensor.
 * \param output_scale - The scale factor (of quantization) for the output tensor.
 */
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
    int         in_sign,
    int         out_sign,
    float       input_scale,
    float       output_scale
  );

/**
 * \brief Add a Transpose layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_name - The name of the input tensor.
 * \param output_name - The name of the output tensor.
 * \param input_shape - The shape of the input tensor.
 * \param order - The new order of the axes indices.
 * \param dim - The rank of the input tensor.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 */
void add_transpose_layer_fix8b(
    void*         p_bmcpl,
    const char*   input_name,
    const char*   output_name,
    const int*    input_shape,
    const int*    order,
    int           dims,
    int           in_sign,
    int           out_sign
  );

/**
 * \brief Add a ShapeRef layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_name - The name of the input tensor.
 * \param input_shape - The shape of the input tensor.
 * \param input_dim - The rank of the input tensor.
 * \param shape_name - The name of the output shape tensor.
 * \param data_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 */
void add_shape_ref_layer_fix8b(
    void*         p_bmcpl,
    const char*   input_name,
    const int*    input_shape,
    int           input_dim,
    const char*   shape_name,
    int           data_sign
  );

/**
 * \brief Add a ShapeAssign layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_name - The name of the 1st input tensor.
 * \param input_shape - The shape of the 1st input tensor.
 * \param input_dim - The rank of the 1st input tensor.
 * \param shape_name - The name of the 2nd input tensor, which is a shape tensor.
 * \param output_name - The name of the output tensor.
 * \param data_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 */
void add_shape_assign_layer_fix8b(
    void*         p_bmcpl,
    const char*   input_name,
    int*          input_shape,
    int           input_dim,
    const char*   shape_name,
    const char*   output_name,
    int           data_sign
  );

/**
 * \brief Add a Squeeze layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_name - The name of the input tensor, which must exist already in BMCompiler.
 * \param input_shape - The shape of the input tensor.
 * \param input_dim - The rank of the input tensor.
 * \param axis_list - The array of integers contains axes indices parameter of the layer.
 * \param axis_num - The size of the axes indices array (value 0 means removal of all '1' dims).
 * \param output_name - The name of the output tensor.
 * \param data_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 */
void add_squeeze_layer_fix8b(
    void*   p_bmcpl,
    char*   input_name,
    int*    input_shape,
    int     input_dim,
    int*    axis_list,
    int     axis_num,
    char*   output_name,
    int     data_sign
  );

/**
 * \brief Add an ExpandNDims layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_name - The name of the input tensor.
 * \param input_shape - The shape of the input tensor.
 * \param input_dim - The rank of the input tensor.
 * \param input_data - The pointer to the input tensor data. If not NULL, it means that the input tensor is a constant tensor.
 * \param axis - The axis parameter of the layer.
 * \param ndims - The ndims parameter of the layer.
 * \param output_name - The name of the output tensor.
 * \param data_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 */
void add_expand_ndims_layer_fix8b(
    void*         p_bmcpl,
    const char*   input_name,
    const int*    input_shape,
    int           input_dim,
    const float*  input_data,
    int           axis,
    int           ndims,
    const char*   output_name,
    int           data_sign
  );

/**
 * \brief Add an ExpandDims layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_name - The name of the input tensor.
 * \param input_shape - The shape of the input tensor.
 * \param input_dim - The rank of the input tensor.
 * \param axis - The axis parameter of the layer.
 * \param output_name - The name of the output tensor.
 * \param data_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 */
void add_expand_dims_layer_fix8b(
    void*         p_bmcpl,
    const char*   input_name,
    const int*    input_shape,
    int           input_dim,
    int           axis,
    const char*   output_name,
    int           data_sign
  );

/**
 * \brief Add a CPU layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_num - The number of input tensors.
 * \param input_shape - The shapes of all input tensors.
 * \param input_shape_dim - The ranks of all input tensors.
 * \param input_name - The names of all input tensors.
 * \param output_num - The number of output tensors.
 * \param output_shape - The shapes of all output tensors.
 * \param output_shape_dim - The ranks of all output tensors.
 * \param output_name - The names of all output tensors.
 * \param input_dtype - The data type (of DATA_TYPE_T) of all input tensors.
 * \param output_dtype - The data type (of DATA_TYPE_T) of all output tensors.
 * \param input_scales - The scale factor (of quantization) of all input tensors.
 * \param output_scales - The scale factor (of quantization) of all output tensors.
 * \param op_type - The CPU operation type (check \ref CPU_LAYER_TYPE_T).
 * \param layer_param - The pointer to the CPU layer parameters (check \ref cpu_xxx_param_t).
 * \param param_size - The size (in bytes) of the CPU layer parameters.
 */
void add_cpu_layer_fix8b(
    void*              p_bmcpl,
    int                input_num,
    const int* const*  input_shape,
    const int*         input_shape_dim,
    const char* const* input_name,
    int                output_num,
    const int* const*  output_shape,
    const int*         output_shape_dim,
    const char* const* output_name,
    const int*         input_dtype,
    const int*         output_dtype,
    const float*       input_scales,
    const float*       output_scales,
    int                op_type,
    const void*        layer_param,
    int                param_size
  );

/**
 * \brief Add an UserCPU layer to BMCompiler.
 *
 * \note
 * This API is a wrapper of \ref add_cpu_layer_fix8b(). The only difference is that, the \ref op_type parameter\n
 * in \ref add_cpu_layer_fix8b() is omitted in this API, since the \ref op_type parateter is set to constant value\n
 * \ref CPU_USER_DEFINED.\n
 */
void add_user_cpu_layer_fix8b(
    void*              p_bmcpl,
    int                input_num,
    const int* const*  input_shape,
    const int*         input_shape_dim,
    const char* const* input_name,
    int                output_num,
    const int* const*  output_shape,
    const int*         output_shape_dim,
    const char* const* output_name,
    const int*         input_dtype,
    const int*         output_dtype,
    const float*       input_scales,
    const float*       output_scales,
    const void*        layer_param,
    int                param_size
  );


/**
 * \brief Add a Select layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param cond_name - The name of the condition tensor.
 * \param s0_name - The name of the 1st selection condidate tensor.
 * \param s0_is_scalar - Whether or not the 1st candidate tensor is a constant scalar.
 * \param s0_scalar_fix8b - The scalar value for s0, if s0_is_scalar=1.
 * \param s1_name - The name of the 2nd selection condidate tensor.
 * \param s1_is_scalar - Whether or not the 2nd candidate tensor is a constant scalar.
 * \param s1_scalar_fix8b - The scalar value for s1, if s1_is_scalar=1.
 * \param output_name - The name of the output tensor.
 * \param in_sign - The sign of the condition tensor (0 for unsigned, 1 for signed).
 * \param s0_sign - The sign of the 1st candiate tensor (0 for unsigned, 1 for signed).
 * \param s1_sign - The sign of the 2nd candiate tensor (0 for unsigned, 1 for signed).
 * \param out_sign - The sign of the output tensor (0 for unsigned, 1 for signed).
 * \param scalea - The scale factor for the 1st candiate tensor. If s0_is_const=1, this value should be 1.
 * \param nshifta - The right-shifting size (in bits) for the 1st candiate tensor. If s0_is_const=1, this value should be 1.
 * \param scaleb - The scale factor of the 2nd candiate tensor. If s1_is_const=1, this value should be 1.
 * \param nshiftb - The right-shifting size (in bits) for the 2nd candiate tensor. If s0_is_const=1, this value should be 1.
 * \param shape - The shape of the condition tensor.
 * \param dims - The rank of the condition tensor.
 */
void add_select_layer_fix8b(
    void*         p_bmcpl,
    const char*   cond_name,
    const char*   s0_name,
    const int     s0_is_scalar,
    const int     s0_scalar_fix8b,
    const char*   s1_name,
    const int     s1_is_scalar,
    const int     s1_scalar_fix8b,
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

/**
 * \brief Add an Arg layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor.
 * \param input_shape_dim - The rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param in_sign - The sign of the input tensor (0 for unsigned, 1 for signed).
 * \param axis - The axis of the input tensor along which to find the maximize or minimize element.
 * \param method - The Arg method (0 for argmax, 1 for argmin).
 */
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

/**
 * \brief Add a DetectionOutput (SSD) layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shapes of 3 input tensors.
 * \param input_shape_dim - The ranks of 3 input tensors.
 * \param input_name - The names of 3 input tensors.
 * \param input_dtype - The data type of 3 input tensors. Normally the 1st input tensor is fix8b (either signed or unsigned), and the 2nd/3rd input tensors are of DTYPE_FP32 type.
 * \param input_scale - The scale factor (of quantization) of 3 input tensors.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param output_dtype - The data type of the output tensor. Normally it's of DTYPE_FP32 type.
 * \param output_scale - The scale factor of the output tensor.
 * \param num_classes - The number of classes to be predicted.
 * \param share_location - Whether or not the bounding boxes are shared among different classes (0 for False, 1 for True).
 * \param background_label_id - The label id for the background. If there is not background class, set it as -1.
 * \param code_type - The type of coding method for bbox (check \ref PriorBoxParameter_CodeType).
 * \param variance_encoded_in_target - Whether or not the variance is encoded in target.
 * \param keep_top_k - The number of total bboxes to be kept per image after NMS step. -1 means keeping all bboxes after NMS step.
 * \param condifence_threshold - The threshold of confidence that only consider detections whose confidences are larger than the threshold.
 * \param nms_threshold - The NMS threshold.
 * \param eta - The eta parameter for NMS.
 * \param top_k - The top_k parameter for NMS.
 */
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

/**
 * \brief Add a Yolov3DetectionOutput layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_num - The number of input tensors.
 * \param input_shape - The shapes of all input tensors.
 * \param input_shape_dim - The ranks of all input tensors.
 * \param input_name - The names of all input tensors.
 * \param input_dtype - The data type of all input tensors.
 * \param input_scale - The scale factor (of quantization) of all input tensors.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param output_dtype - The data type of the output tensor. Normally it's of DTYPE_FP32 type.
 * \param output_scale - The scale factor of the output tensor.
 * \param num_classes - The number of classes to be predicted.
 * \param num_boxes - The number of boxes.
 * \param mask_group_size - The group size of masks.
 * \param keep_top_k - The number of total bboxes to be kept per image after NMS step. -1 means keeping all bboxes after NMS step.
 * \param condifence_threshold - The threshold of confidence that only consider detections whose confidences are larger than the threshold.
 * \param nms_threshold - The NMS threshold.
 * \param bias - The pointer to bias data.
 * \param anchor_scale - The pointer to anchors' scale data.
 * \param mask - The pointer to the mask data.
 */
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
    float*             mask
  );

/**
 * \brief Add an ArithmeticShift layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shape of the input tensor.
 * \param input_shape_dim - The rank of the input tensor.
 * \param input_name - The name of the input tensor.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of the output tensor.
 * \param output_name - The name of the output tensor.
 * \param shift_per_channel - The shift num for each channel.
 * \param shiftType - The shift type (check \ref ElementwiseShiftType).
 * \param shift_num - The shift num.
 * \param shift_mode - The shift mode:\n
 * 0: unified shift of all elements in the input tensor use \ref shift_num;\n
 * 1: perchannel shift, the shift num for each channel comes from \ref shift_per_channel;\n
 * 2: shift num is stored in another tensor, which has the same shape of the input tensor. Not applicable here.
 * \param in_type - The data type of the input tensor (check \ref DATA_TYPE_T).
 * \param out_type - The data type of the output tensor (check \ref DATA_TYPE_T).
 *
 * \note
 * When shift_mode=0, both input tensor and output tensor are of fix8b data type;\n
 * When shift_mode=1, the input tensor is of fix16b data type, and the output tensor is of fix8b data type;\n
 */
void add_arith_shift_layer_fix8b(
    void*              p_bmcpl,
    const int*         input_shape,
    int                input_shape_dim,
    const char*        input_name,
    const int*         output_shape,
    int                output_shape_dim,
    const char*        output_name,
    const void *       shift_per_channel,
    int                shiftType,
    int                shift_num,
    int                shift_mode,
    int                shift_is_const,
    int                in_type,
    int                out_type
  );

/**
 * \brief Add a SliceLike layer to BMCompiler.
 *
 * \param p_bmcpl - Handle to a BMCompiler instance, which is created by \ref create_bmcompiler() or \ref create_bmcompiler_dir().
 * \param input_shape - The shapes of the two input tensors.
 * \param input_shape_dim - The ranks of the two input tensor.
 * \param input_name - The names of the two input tensors.
 * \param output_shape - The shape of the output tensor.
 * \param output_shape_dim - The rank of all output tensor.
 * \param output_name - The name of all output tensor.
 * \param axis - The pointer to an integer array specifying which axes of the 1st input tensor will be sliced according to the 2nd input tensor.
 * \param axis_num - The size of the integer array pointed by \ref axis parameter.
 * \param input_sign - The input signs of the two input tensors. The data type (including the sign) of the output is the same as that of the 1st input tensor;
 */
void add_slice_like_layer_fix8b(
    void*   	        p_bmcpl,
    const int* const* 	input_shape,
    const int*          input_shape_dim,
    const char* const*  input_name,
    const int*    	    output_shape,
    int     		    output_shape_dim,
    const char*   	    output_name,
    const int*    	    axis,
    int     		    axis_num,
    const int*          input_sign
  );

} // namespace bmcompiler

#ifdef __cplusplus
}
#endif

#endif
