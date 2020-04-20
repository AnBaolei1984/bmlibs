#ifndef BMCOMPILER_IF_LITE_H
#define BMCOMPILER_IF_LITE_H
#include "bmcompiler_defs.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief bm_set_compiler_config
 * @param compiler handle
 * @param config compilation params
 */
void bm_set_compiler_config(void* compiler, const bm_compiler_config_t * config);

/**
 * @brief bm_get_compiler_config
 * @param compiler handle
 * @param config [out] current compilation param
 */
void bm_get_compiler_config(void* compiler, bm_compiler_config_t * config);

/**
 * @brief bm_set_input_param
 * @param compiler
 * @param input_name can be subnet input name
 * @param input_info
 */
void bm_set_input_param(void* compiler, const char* input_name, const bm_user_tensor_t* input_info);

/**
 * @brief bm_get_tensor_info set input data, shape, dtype...
 * @param compiler
 * @param name can be inner name or input name
 * @param info
 */
void bm_get_tensor_info(void* compiler, const char* name, bm_user_tensor_t *info);

/**
 * @brief bm_add_ref_data for compilation compare
 * @param compiler
 * @param ref_name
 * @param ref_data must keep valid during compiling
 */
void bm_add_ref_data(void* compiler, const char* ref_name, const void* ref_data);

void bm_compile(void* compiler, const char *net_name);


/**
 * @brief bm_add_const_tensor
 * @param compiler
 * @param name
 * @param shape
 * @param dims
 * @param dtype
 * @param data
 */

void bm_add_const_tensor(
    void* compiler,
    const char* name,
    const int *shape,
    int dims,
    bm_data_type_t dtype,
    const void* data
    );

/**
 * @brief bm_disable_output forbid the tensor as output tensor
 * @param handle
 * @param output_name
 */
void bm_disable_output(void *handle, const char* output_name);

/**
 * @brief bm_add_tensor_array
 * @param compiler
 * @param size_name
 * @param handle_name
 * @param flowout_name
 * @param dtype
 * @param elem_shape
 * @param elem_dims
 * @param clear_after_read
 * @param dynamic_size
 * @param elem_identical
 */
void bm_add_tensor_array(
    void *compiler,
    const char* size_name,    //in
    const char* handle_name,  //out
    const char* flowout_name, //out
    DATA_TYPE_T dtype,
    const int* elem_shape,
    int elem_dims,
    int clear_after_read,
    int dynamic_size,
    int elem_identical
     );
/**
 * @brief bm_add_ta_size_layer
 * @param handle_name
 * @param flowin_name
 * @param output_name
 */
void bm_add_ta_size_layer(
    void *compiler,
    const char* handle_name,
    const char* flowin_name,
    const char* output_name
    );

/**
 * @brief bm_add_ta_read_layer
 * @param handle_name
 * @param index_name
 * @param flowin_name
 * @param output_name
 */
void bm_add_ta_read_layer(
    void *compiler,
    const char* handle_name,
    const char* index_name,
    const char* flowin_name,
    const char* output_name
    );

/**
 * @brief bm_add_ta_write_layer
 * @param handle_name
 * @param index_name
 * @param value_name
 * @param flowin_name
 * @param flowout_name
 */
void bm_add_ta_write_layer(
    void *compiler,
    const char* handle_name,
    const char* index_name,
    const char* value_name,
    const char* flowin_name,
    const char* flowout_name
    );
/**
 * @brief bm_add_ta_gather_layer
 * @param handle_name
 * @param indices_name
 * @param flowin_name
 * @param output_name
 */
void bm_add_ta_gather_layer(
    void *compiler,
    const char* handle_name,
    const char* indices_name,
    const char* flowin_name,
    const char* output_name
    );

/**
 * @brief bm_add_ta_concat_layer
 * @param handle_name
 * @param flowin_name
 * @param output_name
 */
void bm_add_ta_concat_layer(
    void *compiler,
    const char* handle_name,
    const char* flowin_name,
    const char* output_name
    );

/**
 * @brief bm_add_ta_scatter_layer
 * @param handle_name
 * @param indices_name
 * @param value_name
 * @param flowin_name
 * @param flowout_name
 */
void bm_add_ta_scatter_layer(
    void *compiler,
    const char* handle_name,
    const char* indices_name,
    const char* value_name,
    const char* flowin_name,
    const char* flowout_name
    );

/**
 * @brief bm_add_ta_split_layer
 * @param handle_name
 * @param value_name
 * @param lengths_name
 * @param flowin_name
 * @param flowout_name
 */
void bm_add_ta_split_layer(
    void *compiler,
    const char* handle_name,
    const char* value_name,
    const char* lengts_name,
    const char* flowin_name,
    const char* flowout_name
    );

/**
 * @brief bm_add_switch_layer
 * @param compiler
 * @param cond_name
 * @param input_name
 * @param false_name
 * @param true_name
 */
void bm_add_switch_layer(
    void*       compiler,
    const char* cond_name,
    const char* input_name,
    const char* false_name,
    const char* true_name
    );

/**
 * @brief bm_add_merge_layer
 * @param compiler
 * @param input_num
 * @param input_names
 * @param output_name
 */
void bm_add_merge_layer(
    void* compiler,
    int   input_num,
    const char* const* input_names,
    const char* output_name
    );

void bm_add_host2device_layer(
    void*       compiler,
    const char* input_name,
    const char* output_name
    );

void bm_add_device2host_layer(
    void*       compiler,
    const char* input_name,
    const char* output_name
    );

/**
 * @brief bm_add_identity_layer
 * @param compiler
 * @param input_name
 * @param output_name
 */
void bm_add_identity_layer(
    void *compiler,
    const char* input_name,
    const char* output_name
    );

/**
 * @brief bm_add_slice_layer
 * @param compiler
 * @param input_name
 * @param index_name
 * @param size_name
 * @param output_name
 */
void bm_add_slice_layer(
    void *        compiler,
    const char*   input_name,
    const char*   index_name,
    const char*   size_name,
    const char*   output_name
  );

/**
 * @brief bm_add_stride_slice_layer
 * @param compiler
 * @param input_name
 * @param begin_index_name
 * @param end_index_name
 * @param strides_name
 * @param output_name
 * @param begin_mask
 * @param end_mask
 * @param ellipsis_mask
   @param new_axis_mask
   @param shrink_mask
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
 * @brief bm_add_topk_layer
 * @param compiler
 * @param input_name
 * @param k_name
 * @param output_name
 * @param axis
 */
void bm_add_topk_layer(
    void*         handle,
    const char*   input_name,
    const char*   k_name,
    const char*   values_name,
    const char*   indices_name,
    int           axis
    );

/**
 * @brief bm_add_pad_layer
 * @param compiler
 * @param input name
 * @param paddings name
 * @param output_name
 * @param pad value
 * @param pad mode
 */
void bm_add_pad_layer(
    void*         handle,
    const char*   input_name,
    const char*   paddings_name,
    const char*   output_name,
    float         pad_val,
    int           pad_mode
    );

void bm_set_last_layer_format(
        void* handle,
        bm_layer_format_t format);

void bm_coeff_to_neuron(
        void* handle,
        const char* input_name,
        const char* output_name
        );
#ifdef __cplusplus
}
#endif

#endif // BMCOMPILER_IF_LITE_H
