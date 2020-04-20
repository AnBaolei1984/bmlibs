#ifndef _CPU_OP_H_
#define _CPU_OP_H_

#include <vector>
#include <string>
using std::vector;
using std::string;

#if defined (__cplusplus)
extern "C" {
#endif

/**
 * @name   bmcpu_init
 * @brief  initialize bmcpu library
 *
 * @retval  bmcpu handler
 */
void* bmcpu_init();

/**
 * @name   bmcpu_uninit
 * @brief  deinitialize bmcpu library
 *
 * @param   [in]    bmcpu_handle  The pointer of cpu handler.
 */
void bmcpu_uninit(void* bmcpu_handle);

/**
 * @name    bmcpu_process
 * @brief   Call cpu process
 *
 * The interface will call the process the corresponding cpu layer
 *
 * @param   [in]    bmcpu_handle   The pointer of cpu handler.
 * @param   [in]    op_type        The type of the cpu op that is defined in CPU_LAYER_TYPE.
 * @param   [in]    param          The pointer of the cpu op parameter.
 * @param   [in]    input_tensors  The data pointer of each inpyyut tensor.
 * @param   [in]    input_shapes   The shape of each input tensor.
 * @param   [in]    output_tensors The data pointer of each output tensor.
 * @param   [in]    output_shapes  The shape of each output tensor.
 *
 * @retval  0      success
 * @retval  other  fail
 */
int  bmcpu_process(void* bmcpu_handle, int op_type, void *param,
                   const vector<float *>& input_tensors,
                   const vector<vector<int>>& input_shapes,
                   const vector<float *>& output_tensors,
                   vector<vector<int>>& output_shapes
                   );

int  bmcpu_user_process(void* bmcpu_handle, void *param,
                   const vector<float *>& input_tensors,
                   const vector<vector<int>>& input_shapes,
                   const vector<float *>& output_tensors,
                   vector<vector<int>>& output_shapes
                   );

/**
 * @name    bmcpu_reshape
 * @brief   output reshape with given input shape
 *
 * The interface will call change output shape with given input shape
 *
 * @param   [in]    bmcpu_handle   The pointer of cpu handler.
 * @param   [in]    op_type        The type of the cpu op that is defined in CPU_LAYER_TYPE.
 * @param   [in]    param          The pointer of the cpu op parameter.
 * @param   [in]    input_shapes   The shape of each input tensor.
 * @param   [in]    output_shapes  The shape of each output tensor.
 *
 * @retval  0      success
 * @retval  other  fail
 */
int  bmcpu_reshape(void* bmcpu_handle, int op_type, void *param,
                   const vector<vector<int>>& input_shapes,
                   vector<vector<int>>& output_shapes
                   );

int  bmcpu_user_reshape(void* bmcpu_handle, void *param,
                        const vector<vector<int>>& input_shapes,
                        vector<vector<int>>& output_shapes
                        );

#if defined (__cplusplus)
}
#endif


#endif /* _CPU_OP_H_ */
