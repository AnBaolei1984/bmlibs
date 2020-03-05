#ifndef _USER_CPU_OP_H_
#define _USER_CPU_OP_H_

#include <vector>
using std::vector;

#if defined (__cplusplus)
extern "C" {
#endif

/**
 * @name   bmcpu_init
 * @brief  initialize bmcpu library
 *
 * @retval  bmcpu user handler
 */
void* user_cpu_init();

/**
 * @name   bmcpu_uninit
 * @brief  deinitialize bmcpu library
 *
 * @param   [in]    bmcpu_user_handle  The pointer of cpu handler.
 */
void user_cpu_uninit(void *bmcpu_user_handle);

/**
 * @name    user_cpu_process
 * @brief   Call user cpu process
 *
 * The interface will call the process the corresponding cpu layer
 *
 * @param   [in]    bmcpu_user_handle  The pointer of cpu handler.
 * @param   [in]    param          The pointer of the cpu op
 *          parameter.
 * @param   [in]    input_tensors  The data pointer of each inpyyut tensor.
 * @param   [in]    input_shapes   The shape of each input tensor.
 * @param   [in]    output_tensors The data pointer of each output tensor.
 * @param   [in]    output_shapes  The shape of each output tensor.
 *
 * @retval  0      success
 * @retval  other  fail
 */
int  user_cpu_process(void* bmcpu_user_handle, void *param,
                   const vector<float *>& input_tensors,
                   const vector<vector<int>>& input_shapes,
                   const vector<float *>& output_tensors,
                   vector<vector<int>>& output_shapes
                   );

/**
 * @name    user_cpu_reshape
 * @brief   output reshape with given input shape
 *
 * The interface will call change output shape with given input shape
 *
 * @param   [in]    bmcpu_user_handle   The pointer of cpu handler.
 * @param   [in]    param               The pointer of the cpu op parameter.
 * @param   [in]    input_shapes        The shape of each input tensor.
 * @param   [in]    output_shapes       The shape of each output tensor.
 *
 * @retval  0      success
 * @retval  other  fail
 */
int  user_cpu_reshape(void* bmcpu_handle, void *param,
                      const vector<vector<int>>& input_shapes,
                      vector<vector<int>>& output_shapes
                      );

#if defined (__cplusplus)
}
#endif


#endif /* _USER_CPU_OP_H_ */
