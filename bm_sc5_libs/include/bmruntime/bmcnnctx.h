/*****************************************************************************
 *
 *    Copyright (c) 2016-2026 by Bitmain Technologies Inc. All rights reserved.
 *
 *    The material in this file is confidential and contains trade secrets
 *    of Bitmain Technologies Inc. This is proprietary information owned by
 *    Bitmain Technologies Inc. No part of this work may be disclosed,
 *    reproduced, copied, transmitted, or used in any way for any purpose,
 *    without the express written permission of Bitmain Technologies Inc.
 *
 *****************************************************************************/

#ifndef __BM_CNN_CONTEXT_H__
#define __BM_CNN_CONTEXT_H__

#include <string>

namespace bmcnn {

typedef void *bmcnn_ctx_t;
/**
 * \brief Create context of BMCNN.
 *
 * \param ctx_dir - Directory of context files generated by NET_COMPILER
 *
 * \note
 * The context will be created in the device of ID 0.\n
 *
 * \return
 * NULL - Creating failed.\n
 * non-NULL - The handle of the context (creating succeeded).\n
 */
//bmcnn_ctx_t bmcnn_ctx_create(const std::string &ctx_dir);
/**
 * \brief Destroy context of BMCNN
 *
 * \param handle - Handle of the context to be destroyed
 */
void bmcnn_ctx_destroy(bmcnn_ctx_t handle);
/**
 * \brief Create context of BMCNN in specific devide.
 *
 * \param ctx_dir - Directory of context files generated by NET_COMPILER
 * \param devid - ID of device where the context will be placed.
 *
 * \note
 * Call \ref bm_dev_getcount to get total number of devices, e.g. N is returned,
 * valid devid should be in range of 0 ~ (N-1).\n
 *
 * \return
 * NULL - Creating failed that might be caused by incorrect parameter.\n
 * non-NULL - The handle of the context (creating succeeded).\n
 */
bmcnn_ctx_t bmcnn_ctx_create_by_devid(const std::string &ctx_dir, int devid, const std::string &chipname = "BM1682");
/**
 * \brief Append context of BMCNN.
 *
 * \param ctx_dir - Directory of context files generated by NET_COMPILER.
 * \param bmrt    - The created handle of context.
 *
 * \return
 * false - Appending failed.\n
 * true  - Appending succeeded.\n
 */
bool bmcnn_ctx_append(const std::string &ctx_dir, void *bmrt);
} /* namespace bmcnn */

#endif /* __BM_CNN_CONTEXT_H__ */
