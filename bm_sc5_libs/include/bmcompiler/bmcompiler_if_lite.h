#ifndef BMCOMPILER_IF_LITE_H
#define BMCOMPILER_IF_LITE_H
#include "bmcompiler_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace bmcompiler {

/**
 * \brief bm_set_input_param: use to set layer input tensor's shape, data_type, tensor_type info
 *    Supposed Usage:
 *        1. Use bm_xxx functions to connect the graph according tensor's name first
 *        2. Use bm_set_input_param to set the info of input tensors of the whole graph
 *        3. Call bmcompiler compile_xxx interface to compile the whole graph,
 *                the missing shape info of the inner layers will be inferred layer-by-layer
 * \param compiler
 * \param input_name
 * \param input_info
 */
void bm_set_input_param(void* compiler, const char* input_name, const bm_user_tensor_t* input_info);

/**
 * \brief bm_set_input_param: use to set layer input tensor's shape, data_type, tensor_type info
 *    Supposed Usage:
 *        1. Use bm_xxx functions to connect the graph according tensor's name first
 *        2. Use bm_set_input_param to set the info of input tensors of the whole graph
 *        3. Call bmcompiler compile_xxx interface to compile the whole graph,
 *                the missing shape info of the inner layers will be inferred layer-by-layer
 * \param compiler
 * \param input_name
 * \param input_info
 * \param data: preset data for tensor shape inference
 */
void bm_set_input_param_and_data(void *handle, const char *input_name, const bm_user_tensor_t *input_info, const void* data);

/**
 * \brief bm_get_tensor_info set input data, shape, dtype...
 * \param compiler
 * \param name can be inner name or input name
 * \param info
 */
void bm_get_tensor_info(void* compiler, const char* name, bm_user_tensor_t *info);

/**
 * \brief bm_add_const_tensor: insert a const tensor into graph
 * \param compiler
 * \param name
 * \param shape
 * \param dims
 * \param dtype
 * \param data
 */
void bm_add_const_tensor(
    void*          compiler,
    const char*    name,
    const int*     shape,
    int            dims,
    bm_data_type_t dtype,
    const void*    data
    );

/**
 * \brief bm_disable_output forbid the tensor as output tensor
 * \param compiler
 * \param output_name
 */
void bm_disable_output(void *compiler, const char* output_name);

/**
 * \brief bm_add_tensor_array: declare a tensor_array in graph
 * \param compiler
 * \param size_name: tensorarray size tensor's name
 * \param handle_name: tensorarray's handle tensor's name
 * \param flowout_name: tensorarray's flow output's name
 * \param dtype: tensorarray's data type
 * \param elem_shape: shape of elements in tensorarray
 * \param elem_dims: dims of elements in tensorarray
 * \param clear_after_read: clear the element after reading from tensorarray
 * \param dynamic_size: tensorarray has dynamic size
 * \param elem_identical: every element in tensorarray has idenitical shape
 */
void bm_add_tensor_array(
    void*       compiler,
    const char* size_name,    //in
    const char* handle_name,  //out
    const char* flowout_name, //out
    DATA_TYPE_T dtype,
    const int*  elem_shape,
    int         elem_dims,
    int         clear_after_read,
    int         dynamic_size,
    int         elem_identical
     );
/**
 * \brief bm_add_ta_size_layer: get tensorarray size as output tensor
 * \param handle_name: tensorarray's handle tensor's name
 * \param flowin_name: flow tensor's name which is from other tensorarray operations
 * \param output_name: size tensor's name
 */
void bm_add_ta_size_layer(
    void*       compiler,
    const char* handle_name,
    const char* flowin_name,
    const char* output_name
    );

/**
 * \brief bm_add_ta_read_layer: read from tensorarray
 * \param handle_name: tensorarray's handle tensor's name
 * \param index_name: name of index tensor whose value is the read position
 * \param flowin_name: flow tensor's name which is from other tensorarray operations
 * \param output_name: normal tensor
 */
void bm_add_ta_read_layer(
    void*       compiler,
    const char* handle_name,
    const char* index_name,
    const char* flowin_name,
    const char* output_name
    );

/**
 * \brief bm_add_ta_write_layer
 * \param handle_name: tensorarray's handle tensor's name
 * \param index_name: name of index tensor whose value is the write position
 * \param value_name: normal tensor to write into tensorarray
 * \param flowin_name: flow tensor's name which is from other tensorarray operations
 * \param flowout_name: output flow tensor's name
 */
void bm_add_ta_write_layer(
    void*       compiler,
    const char* handle_name,
    const char* index_name,
    const char* value_name,
    const char* flowin_name,
    const char* flowout_name
    );

/**
 * \brief bm_add_ta_gather_layer: read from multiple positions, expand first dim and concat them together on expanded dim
 * \param handle_name: tensorarray's handle tensor's name
 * \param indice_name: name of indice tensor whose values are read positions
 * \param flowin_name: flow tensor's name which is from other tensorarray operations
 * \param output_name: name of output normal tensor
 */
void bm_add_ta_gather_layer(
    void*       compiler,
    const char* handle_name,
    const char* indice_name,
    const char* flowin_name,
    const char* output_name
    );

/**
 * \brief bm_add_ta_concat_layer: read all elements in tensorarray, and concat them together on the first dim
 * \param handle_name: tensorarray's handle tensor's name
 * \param flowin_name: flow tensor's name which is from other tensorarray operations
 * \param output_name: name of output normal tensor
 */
void bm_add_ta_concat_layer(
    void*       compiler,
    const char* handle_name,
    const char* flowin_name,
    const char* output_name
    );

/**
 * \brief bm_add_ta_scatter_layer: split on the first dim and squeeze, then write i-th tensor into tensorarray[indice[i]]
 * \param handle_name: tensorarray's handle tensor's name
 * \param indice_name: name of indice tensor whose values are writing positions
 * \param value_name: name of tensor to be written into tensorarray
 * \param flowin_name: flow tensor's name which is from other tensorarray operations
 * \param flowout_name: name of output flow tensor
 */
void bm_add_ta_scatter_layer(
    void*       compiler,
    const char* handle_name,
    const char* indice_name,
    const char* value_name,
    const char* flowin_name,
    const char* flowout_name
    );

/**
 * \brief bm_add_ta_split_layer: split the value tensor according lengths tensor's value on the first dim
 *                               then write them into tensorarray
 * \param handle_name: tensorarray's handle tensor's name
 * \param value_name: name of the written tensor
 * \param lengths_name: name of the lengths tensor
 * \param flowin_name: flow tensor's name which is from other tensorarray operations
 * \param flowout_name: name of output flow tensor
 */
void bm_add_ta_split_layer(
    void *compiler,
    const char* handle_name,
    const char* value_name,
    const char* lengths_name,
    const char* flowin_name,
    const char* flowout_name
    );

/**
 * \brief bm_add_switch_layer: run graph conditional,
 *           if cond tensor's value is true, then input is forwarded to the true tensor and run the true branch
 *           if cond tensor's value is false, then input is forwarded to the false tensor and run the false branch
 * \param compiler
 * \param cond_name
 * \param input_name
 * \param false_name
 * \param true_name
 */
void bm_add_switch_layer(
    void*       compiler,
    const char* cond_name,
    const char* input_name,
    const char* false_name,
    const char* true_name
    );

/**
 * \brief bm_add_merge_layer: the newest calculated input tensor will be forwarded to output
 * \param compiler
 * \param input_num: number of input tensors
 * \param input_names: names of input tensors
 * \param output_name: name of output tensors
 */
void bm_add_merge_layer(
    void*              compiler,
    int                input_num,
    const char* const* input_names,
    const char*        output_name
    );

/**
 * \brief bm_add_host2device_layer: move ARM data to DDR, should not use this outside
 */
void bm_add_host2device_layer(
    void*       compiler,
    const char* input_name,
    const char* output_name
    );

/**
 * \brief bm_add_host2device_layer: move DDR data to ARM, should not to use this outside
 */
void bm_add_device2host_layer(
    void*       compiler,
    const char* input_name,
    const char* output_name
    );

/**
 * \brief bm_add_identity_layer: just for connecting the graph
 */
void bm_add_identity_layer(
    void *compiler,
    const char* input_name,
    const char* output_name
    );

/**
 * \brief bm_add_slice_layer: dynamic slice layer
 * \param input_name: name of input tensor
 * \param index_name: name of index tensor whose values are slice start positions for each dim
 * \param size_name: name of index tensor whose values are slice sizes for each dim
 * \param output_name
 */
void bm_add_slice_layer(
    void *        compiler,
    const char*   input_name,
    const char*   index_name,
    const char*   size_name,
    const char*   output_name
  );

/**
 * \brief bm_add_slice_layer_v2: dynamic slice layer
 * \param input_name: name of input tensor
 * \param index_name: name of index tensor whose values are slice start positions for each dim
 * \param size_name: name of index tensor whose values are slice sizes for each dim
 * \param output_name
 * \param slice_mask: bitN=1 means N-th dim will not be sliced
 */
void bm_add_slice_layer_v2(
    void *        compiler,
    const char*   input_name,
    const char*   index_name,
    const char*   size_name,
    const char*   output_name,
    unsigned int  slice_mask
  );

/**
 * \brief bm_add_stride_slice_layer:
 * \param compiler
 * \param input_name: name of input tensor
 * \param begin_index_name: name of begin index tensor whose values are slice begin indice
 * \param end_index_name: name of end index tensor whose values are slice end indice
 * \param strides_name: name of stride tensor whose values are slice strides
 * \param output_name: output tensor name
 * \param begin_mask: i-th bit is 1, means begin_index[i] = 0
 * \param end_mask: i-th bit is 1, means end_index[i] = input_shape[i]
 * \param shrink_axis_mask: i-th bit is 1, means squeezing i-th dim
 * \param new_axis_mask: i-th bit is 1, means expand i-th dim
 * \param ellipsis_mask: i-th bit is 1, means not slice i-th dim
 */
void bm_add_stride_slice_layer(
    void *        compiler,
    const char*   input_name,
    const char*   begin_index_name,
    const char*   end_index_name,
    const char*   strides_name,
    const char*   output_name,
    int           begin_mask,
    int           end_mask,
    int           ellipsis_mask,
    int           new_axis_mask,
    int           shrink_axis_mask
  );

/**
 * \brief bm_add_topk_layer: dynamic topk
 * \param compiler
 * \param input_name
 * \param k_name: name of tensor whose value is k
 * \param output_name
 * \param axis
 */
void bm_add_topk_layer(
    void*         compiler,
    const char*   input_name,
    const char*   k_name,
    const char*   values_name,
    const char*   indices_name,
    int           axis,
    int           descending
    );

/**
 * \brief bm_add_pad_layer: dynamic pad
 * \param compiler
 * \param input name
 * \param paddings name: name of padding param tensor whose values are padding sizes
 * \param output_name
 * \param pad value: same as add_pad_layer
 * \param pad mode: 0-constant, 1-reflect, 2-symmetric, same as add_pad_layer
 */
void bm_add_pad_layer(
    void*         compiler,
    const char*   input_name,
    const char*   paddings_name,
    const char*   output_name,
    float         pad_val,
    int           pad_mode
    );

/**
 * \brief bm_coeff_to_neuron: convert a coeff tensor to a neuron tensor, should not use outside
 */
void bm_coeff_to_neuron(
        void*       compiler,
        const char* input_name,
        const char* output_name
        );

/**
 * \brief bm_add_depth_to_space_v2: just for 4D tensor, refer to tf.depth_to_space and tf.space_to_depth
 * \param block_sizes: {h_block_size, w_block_size}
 * \param in_is_nchw: 0: NHWC, 1: NCHW
 * \param out_is_nchw: 0: NHWC, 1: NCHW
 * \param is_inversed: 0: depth2space, 1: space2depth
 */
void bm_add_depth_to_space_v2(
        void*       compiler,
        const char* input_name,
        const char* output_name,
        const int*  block_sizes,
        int         in_is_nchw,
        int         out_is_nchw,
        int         is_inversed
        );

/**
 * \brief bm_add_depth_to_space: just for 4D tensor, refer to tf.depth_to_space and tf.space_to_depth
 * \param compiler
 * \param input_name
 * \param output_name
 * \param block_sizes: {h_block_size, w_block_size}
 * \param is_nchw: input and output Format 0: NHWC, 1: NCHW
 * \param is_inversed: 0: depth2space, 1: space2depth
 */
void bm_add_depth_to_space(
        void*       compiler,
        const char* input_name,
        const char* output_name,
        const int*  block_sizes,
        int         is_nchw,
        int         is_inversed
        );

/**
 * \brief bm_add_broadcast_like_layer: broadcast lhs tensor as rhs tensor's shape (from MxNet)
 * \param compiler
 * \param lhs_name: name of first input
 * \param rhs_name: name of second input
 * \param out_name: name of output tensor
 * \param lhs_axis: Axes to perform broadcast on in the first input array
 * \param rhs_axis: Axes to copy from the second input array
 * \param axes_num: length of lhs_axis or rhs_axis. axes_num=0 means lhs's shape will broadcast to rhs'shape
 */
void bm_add_broadcast_like_layer(
        void*       compiler,
        const char* lhs_name,
        const char* rhs_name,
        const char* out_name,
        const int*  lhs_axis,
        const int*  rhs_axis,
        int         axes_num
        );

/**
 * \brief bm_add_box_decode_layer (from MxNet)
 *      Decode bounding boxes training target with normalized center offsets.
 *      Input bounding boxes are using corner type: x_{min}, y_{min}, x_{max}, y_{max} or center type: `x, y, width, height.) array
 * \param handle
 * \param input_name: name of (B, N, 4) predicted bbox offset
 * \param anchors_name: name of (1, N, 4) encoded in corner or center
 * \param output_name
 * \param stds: length = 4, value to be divided from the 1st,2st,3st,4st encoded values
 * \param clip: if larger than 0, bounding box target will be clipped to this value
 * \param anchor_format: 0-(xmin, ymin, xmax, ymax), 1-(xcenter, ycenter, width, height)
 */
void bm_add_box_decode_layer(
        void*        compiler,
        const char*  input_name,
        const char*  anchors_name,
        const char*  output_name,
        const float* stds,
        float        clip,
        int          anchor_format
        );

/**
 * \brief bm_add_split_tf_layer: same as add_tf_split_layer in bmcompiler_if.h
 */
void bm_add_split_tf_layer(
        void* compiler,
        const char* in_name,
        const char* const* out_names,
        int out_num,
        const int* split_sizes,
        int split_num,
        int split_axis
        );

/**
 * \brief bm_add_concat_layer: same as add_concat_layer in bmcompiler_if.h
 */
void bm_add_concat_layer(
        void* compiler,
        const char* const* in_names,
        int in_num,
        const char* out_name,
        int concat_axis
        );

/**
 * \brief bm_add_active_layer: same as add_active_layer in bmcompiler_if.h
 */
void bm_add_active_layer(
        void* compiler,
        const char* in_name,
        const char* out_name,
        int active_op
        );

/**
 * \brief bm_add_const_binary_layer: same as add_const_binary_layer in bmcompiler_if.h
 */
void bm_add_const_binary_layer(
        void* compiler,
        const char* in_name,
        const char* out_name,
        float const_value,
        int inversed,
        int binary_op
        );

/**
 * \brief bm_add_binary_layer: same as add_binary_layer_v2 in bmcompiler_if.h
 */
void bm_add_binary_layer(
        void* compiler,
        const char* in0_name,
        const char* in1_name,
        const char* out_name,
        int binary_op
        );

/**
 * \brief bm_add_mish_activation: use add_active_layer with op_code=ACTIVE_MISH instead
 */
void bm_add_mish_activation(
        void* compiler,
        const char* in_name,
        const char* out_name
        );

/**
 * \brief bm_add_shape_ref_layer: same as add_shape_ref_layer in bmcompiler_if.h
 */
void bm_add_shape_ref_layer(
        void* compiler,
        const char* in_name,
        const char* out_name
        );

/**
 * \brief bm_add_shape_assign_layer: same as add_shape_assign_layer in bmcompiler_if.h
 */
void bm_add_shape_assign_layer(
        void* compiler,
        const char* in_name,
        const char* shape_name,
        const char* out_name
    );

/**
 * \brief bm_add_tile_layer: same as add_tile_layer in bmcompiler_if.h
 */
void bm_add_tile_layer(
    void* compiler,
    const char* in_name,
    const char* coeff_name,
    const char* out_name
    );


/**
 * \brief bm_add_eltwise_select_layer: same as add_select_layer in bmcompiler_if.h
 *          Note: shape(cond)=shape(then)=shape(else)
 */
void bm_add_eltwise_select_layer(
        void* compiler,
        const char* cond_name,
        const char* then_name,
        const char* else_name,
        const char* out_name
    );

/**
 * \brief bm_add_batch_select_layer: same as add_select_layer in bmcompiler_if.h
            Note: shape(cond)!=shape(then), shape(then)=shape(else)
 */
void bm_add_batch_select_layer(
        void* compiler,
        const char* cond_name,
        const char* then_name,
        const char* else_name,
        const char* out_name
    );

/**
 * \brief bm_add_enter_layer: Enter op from tensorflow
 */
void bm_add_enter_layer(
        void* compiler,
        const char* input_name,
        const char* output_name,
        bool is_const
    );

} // namespace bmcompiler


#ifdef __cplusplus
}
#endif

#endif // BMCOMPILER_IF_LITE_H
