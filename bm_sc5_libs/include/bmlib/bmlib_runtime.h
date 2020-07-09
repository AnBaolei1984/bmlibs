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

/**************************************************************************
 * bmlib_runtime defines interfaces that operate TPU devices.
 * The functions can be divided into serveral categories.
 * 1) device handle creation and destroy
 * 2) memory help functions
 * 3) global memory allocation and free
 * 4) data transfer between host and device
 * 5) data transfer within device memory
 * 6) api send and synchronization
 * 7) global memory map and coherence
 * 8) trace and profile
 * 9) power management
 * 10) miscellaneous functions
 *************************************************************************/

#ifndef BMLIB_RUNTIME_H_
#define BMLIB_RUNTIME_H_
#include <stdbool.h>
#include <stddef.h>
#include <stdarg.h>

#if !defined(__x86_64__) && !defined(__aarch64__)
#error "BM needs 64 - bit to compile"
#endif

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum {
  MODULE_CDMA = 0,
  MODULE_GDMA = 1,
  MODULE_TPU = 2,
  MODULE_SMMU = 3,
  MODULE_SRAM = 4,
  MODULE_END = 5
} MODULE_ID;

#define BM_MEM_ADDR_NULL (0xfffffffff)

#ifndef BM_MEM_DESC_T_
#define BM_MEM_DESC_T_
/* BM function return code definitions */
typedef enum {
  BM_SUCCESS = 0,
  BM_ERR_DEVNOTREADY = 1, /* Device not ready yet */
  BM_ERR_FAILURE = 2,     /* General failure */
  BM_ERR_TIMEOUT = 3,     /* Timeout */
  BM_ERR_PARAM = 4,       /* Parameters invalid */
  BM_ERR_NOMEM = 5,       /* Not enough memory */
  BM_ERR_DATA = 6,        /* Data error */
  BM_ERR_BUSY = 7,        /* Busy */
  BM_ERR_NOFEATURE = 8,   /* Not supported yet */
  BM_NOT_SUPPORTED = 9
} bm_status_t;

/* BM memory type definitions */
typedef enum {
  BM_MEM_TYPE_DEVICE = 0,
  BM_MEM_TYPE_HOST = 1,
  BM_MEM_TYPE_SYSTEM = 2,
  BM_MEM_TYPE_INT8_DEVICE = 3,
  BM_MEM_TYPE_INVALID = 4
} bm_mem_type_t;

typedef enum {
  PERF_MONITOR_GDMA = 0,
  PERF_MONITOR_TPU = 1
} PERF_MONITOR_ID;

/*
* bm performace monitor
*/
struct bm_perf_monitor {
  long long buffer_start_addr; /*buffer address to store perf data*/
  int buffer_size; /*buffer size*/
  PERF_MONITOR_ID monitor_id; /*PERF_MONITOR_GDMA or PERF_MONITOR_TPU*/
};

typedef union {
  struct {
    bm_mem_type_t mem_type : 3;
    unsigned int reserved : 29;
  } u;
  unsigned int rawflags;
} bm_mem_flags_t;

/* BM memory descriptor definition*/
typedef struct bm_mem_desc {
  union {
    struct {
      unsigned long device_addr;
      unsigned int reserved;
      int dmabuf_fd;
    } device;

    struct {
      void *system_addr;
      unsigned int reserved0;
      int reserved1;
    } system;
  } u;

  bm_mem_flags_t flags;
  unsigned int size;
} bm_mem_desc_t;

typedef struct bm_mem_desc bm_device_mem_t;
typedef struct bm_mem_desc bm_system_mem_t;
#endif

struct bm_context;
typedef struct bm_context *bm_handle_t;

#ifndef USING_CMODEL
#define BM_CHECK_RET(call)                                                    \
  do {                                                                        \
    bm_status_t ret = call;                                                   \
    if (ret != BM_SUCCESS) {                                                  \
      printf("BM_CHECK_RET fail %s: %s: %d\n", __FILE__, __func__, __LINE__); \
      return ret;                                                             \
    }                                                                         \
  } while (0)
#else
#define BM_CHECK_RET(call)                     \
  do {                                         \
    bm_status_t ret = call;                    \
    if (ret != BM_SUCCESS) {                   \
      printf("BM_CHECK_RET failed %d\n", ret); \
      ASSERT(0);                               \
      exit(-ret);                              \
    }                                          \
  } while (0)
#endif

/*******************handle releated functions *********************************/
/**
 * @name    bm_dev_getcount
 * @brief   To get the number of sophon devices in system.
 *          If N is got, valid devid is [0, N-1]
 * @ingroup bmlib_runtime
 *
 * @param [out] count  The result number of sophon devices
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_dev_getcount(int *count);

/**
 * @name    bm_dev_query
 * @brief   To query if a device is present
 * @ingroup bmlib_runtime
 *
 * @param [in] devid  The id of the device to query
 * @retval  BM_SUCCESS Device is present
 *          Other code Devcie is not present
 */
bm_status_t bm_dev_query(int devid);

/**
 * @name    bm_dev_request
 * @brief   To create a handle for the given device
 * @ingroup bmlib_runtime
 *
 * @param [out] handle  The created handle
 * @param [in]  devid   Specify on which device to create handle
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_dev_request(bm_handle_t *handle, int devid);

/**
 * @name    bm_dev_free
 * @brief   To free a handle
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The handle to free
 */
void bm_dev_free(bm_handle_t handle);

/*******************memory help functions ************************************/
/**
 * @name    bm_mem_get_type
 * @brief   To get a memory descriptor's type
 * @ingroup bmlib_runtime
 *
 * @param [in]  mem  The memory descriptor queried
 * @retval  BM_MEM_TYPE_DEVICE  Device global memory
 * @retval  BM_MEM_TYPE_SYSTEM  Host user memory
 */
bm_mem_type_t bm_mem_get_type(struct bm_mem_desc mem);

/**
 * @name    bm_mem_get_device_addr
 * @brief   To get a device memory descriptor's address
 * @ingroup bmlib_runtime
 *
 * @param [in]  mem  The device memory descriptor queried
 * @retval  unsigned long long  The device memory address
 */
unsigned long long bm_mem_get_device_addr(struct bm_mem_desc mem);

/**
 * @name    bm_mem_set_device_addr
 * @brief   To set a device memory descriptor's address
 * @ingroup bmlib_runtime
 *
 * @param [in]  pmem   The device memory descriptor pointer
 * @param ]in]  addr  The new device address of the device memory
 */
void bm_mem_set_device_addr(struct bm_mem_desc* pmem, unsigned long long addr);

/**
 * @name    bm_mem_get_device_size
 * @brief   To get a device memory descriptor's size
 * @ingroup bmlib_runtime
 *
 * @param [in]  mem      The device memory descriptor queried
 * @retval unsigned int  The device memory's size in bytes
 */
unsigned int bm_mem_get_device_size(struct bm_mem_desc mem);

/**
 * @name    bm_mem_set_device_size
 * @brief   To set a device memory descriptor's size
 * @ingroup bmlib_runtime
 *
 * @param [out]  pmem  The device memory descriptor pointer
 * @param [in]  size  The new device memory size (in bytes) of the device memory
 */
void bm_mem_set_device_size(struct bm_mem_desc* pmem, unsigned int size);

/**
 * @name    bm_set_device_mem
 * @brief   To fill in a device memory descriptor with size and address
 * @ingroup bmlib_runtime
 *
 * @param [in] pmem  The device memory descriptor pointer
 * @param [in]  size  The device memory descriptor's size
 * @param [in]  addr  The device memory descriptor's address
 */
void bm_set_device_mem(bm_device_mem_t* pmem, unsigned int size,
                       unsigned long long addr);

/**
 * @name    bm_mem_from_device
 * @brief   To create a device memory descriptor from address and size
 * @ingroup bmlib_runtime
 *
 * @param [in] device_addr The device memory address
 * @param [in] len         The device memory size
 * @retval bm_device_mem_t The device memory descriptor created
 */
bm_device_mem_t bm_mem_from_device(unsigned long long device_addr,
                                   unsigned int len);

/**
 * @name    bm_mem_get_system_addr
 * @brief   To get a system memory descriptor's address
 * @ingroup bmlib_runtime
 *
 * @param [in] mem  The system memory descriptor
 * @retval void *   The system memory descriptor's address
 */
void *bm_mem_get_system_addr(struct bm_mem_desc mem);

/**
 * @name    bm_mem_set_system_addr
 * @brief   To set a system memory descriptor's address
 * @ingroup bmlib_runtime
 *
 * @param [in]  pmem  The system memory descriptor pointer
 * @param [in]   addr The system memory address
 */
void bm_mem_set_system_addr(struct bm_mem_desc* pmem, void *addr);

/**
 * @name    bm_mem_from_system
 * @brief   To create a system memory descriptor with the given system address
 * @ingroup bmlib_runtime
 *
 * @param [in]  system_addr  The system address in the descriptor
 * @retval  bm_system_mem_t  The system memory descriptor created
 */
bm_system_mem_t bm_mem_from_system(void *system_addr);

/*******************memory alloc and free functions ***************************/
/**
 * @name    bm_mem_null
 * @brief   Return an illegal device memory descriptor
 * @ingroup bmlib_runtime
 *
 * @retval  bm_device_mem_t  An invalid device memory descriptor
 */
bm_device_mem_t bm_mem_null(void);
#define BM_MEM_NULL (bm_mem_null())

/**
 * @name    bm_malloc_neuron_device
 * @brief   To malloc device memory according to a tensor shape
 *          (each neuron is 32 bits)
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result devcie memory descriptor
 * @param [in]  n, c, h, w  The shape of the input tensor
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_malloc_neuron_device(bm_handle_t handle, bm_device_mem_t *pmem,
                                    int n, int c, int h, int w);

/**
 * @name    bm_malloc_device_dword
 * @brief   To malloc device memory in size of dword (32 bits)
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result device memory descriptor
 * @param [in]   count  The number of dwords(32bits) to allocate
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_malloc_device_dword(bm_handle_t handle, bm_device_mem_t *pmem,
                                   int count);

/**
 * @name    bm_malloc_device_byte
 * @brief   To malloc device memory in size of byte
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result device memory descriptor
 * @param [in]   size   The number of bytes to allocate
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_malloc_device_byte(bm_handle_t handle, bm_device_mem_t *pmem,
                                  unsigned int size);

/**
 * @name    bm_malloc_device_byte_heap
 * @brief   To malloc device memory in size of byte within the specified heap
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result device memory descriptor
 * @param [in]  heap_id The heap where to allocate  0/1
 * @param [in]   size   The number of bytes to allocate
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_malloc_device_byte_heap(bm_handle_t handle, bm_device_mem_t *pmem,
                                  int heap_id, unsigned int size);

/**
 * @name    bm_free_device
 * @brief   To free device memory
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  mem     The device memory descriptor to free
 */
void bm_free_device(bm_handle_t handle, bm_device_mem_t mem);

/**
 * @name    bm_gmem_arm_reserved_request
 * @brief   To obtain the address of global memory reserved for arm926
 * @param [in]  handle  The device handle
 *
 * @retval unsigned long long  The absolute address of gmem reserved for arm926
 */
unsigned long long bm_gmem_arm_reserved_request(bm_handle_t handle);

/**
 * @name    bm_gmem_arm_reserved_release
 * @brief   To release the global memory reserved for arm926
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 */
void bm_gmem_arm_reserved_release(bm_handle_t handle);

/*******************memory copy functions *************************************/
/**
 * @name    bm_memcpy_s2d
 * @brief   To copy data from system memory to device memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in] dst     The destination memory (device memory descriptor )
 * @param [in] src     The source memory (system memory, a void* pointer)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_memcpy_s2d(bm_handle_t handle, bm_device_mem_t dst, void *src);

/**
 * @name    bm_memcpy_s2d_partial_offset
 * @brief   To copy specified bytes of data from system memory to device memory
 *          with an offset in device memory address.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (device memory descriptor)
 * @param [in]  src    The source memory (system memory, a void* pointer)
 * @param [in] size    The size of data to copy (in bytes)
 * @param [in] offset  The offset of the device memory address
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_memcpy_s2d_partial_offset(bm_handle_t handle,
                                         bm_device_mem_t dst, void *src,
                                         unsigned int size,
                                         unsigned int offset);

/**
 * @name    bm_memcpy_s2d_partial
 * @brief   To copy specified bytes of data from system memory to device memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (device memory descriptor)
 * @param [in]  src    The source memory (system memory, a void* pointer)
 * @param [in] size    The size of data to copy (in bytes)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_memcpy_s2d_partial(bm_handle_t handle, bm_device_mem_t dst,
                                  void *src, unsigned int size);

/**
 * @name    bm_memcpy_d2s
 * @brief   To copy data from device memory to system memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (system memory, a void* pointer)
 * @param [in]  src    The source memory (device memory descriptor)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_memcpy_d2s(bm_handle_t handle, void *dst, bm_device_mem_t src);

/**
 * @name    bm_memcpy_d2s_partial_offset
 * @brief   To copy specified bytes of data from device memory to system memory
 *          with an offset in device memory address.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (system memory, a void* pointer)
 * @param [in]  src    The source memory (device memory descriptor)
 * @param [in] size    The size of data to copy (in bytes)
 * @param [in] offset  The offset of the device memory address
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_memcpy_d2s_partial_offset(bm_handle_t handle, void *dst,
                                         bm_device_mem_t src, unsigned int size,
                                         unsigned int offset);

/**
 * @name    bm_memcpy_d2s_partial
 * @brief   To copy specified bytes of data from device memory to system memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (system memory, a void* pointer)
 * @param [in]  src    The source memory (device memory descriptor)
 * @param [in] size    The size of data to copy (in bytes)
 *
 * @retval  BM_SUCCESS  Data transfer succeeds.
 *          Other code  Data transfer fails.
 */
bm_status_t bm_memcpy_d2s_partial(bm_handle_t handle, void *dst,
                                  bm_device_mem_t src, unsigned int size);

/**
 * @name    bm_memcpy_d2d
 * @brief   To copy specified dwords of data from one piece of device memory
 *          to another piece of device memory within one device. Both source
 *          and destination offsets can be specified.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle     The device handle
 * @param [in]  dst       The destination device memory
 * @param [in] dst_offset The offset of destination device memory address
 * @param [in]  src       The source device memory
 * @param [in] src_offset The offset of source device memory address
 * @param [in]  len       Length of data to copy (in DWORD 4 bytes)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_memcpy_d2d(bm_handle_t handle, bm_device_mem_t dst,
                          int dst_offset, bm_device_mem_t src, int src_offset,
                          int len);

/**
 * @name    bm_memcpy_d2d_byte
 * @brief   To copy specified bytes of data from one piece of device memory
 *          to another piece of device memory within one device. Both source
 *          and destination offsets can be specified.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle     The device handle
 * @param [in]  dst       The destination device memory
 * @param [in] dst_offset The offset of destination device memory address (in bytes)
 * @param [in]  src       The source device memory
 * @param [in] src_offset The offset of source device memory address (in bytes)
 * @param [in]  size      Size of data to copy (in bytes)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_memcpy_d2d_byte(bm_handle_t handle, bm_device_mem_t dst,
                               size_t dst_offset, bm_device_mem_t src,
                               size_t src_offset, size_t size);

/**
 * @name    bm_memcpy_d2d_stride
 * @brief   To copy specified data from one piece of device memory
 *          to another piece of device memory within one device. Both source
 *          and destination offsets can be specified.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle      The device handle
 * @param [in] dst         The destination device memory
 * @param [in] dst_stride  The data stride of destination data
 * @param [in] src         The source device memory
 * @param [in] src_stride  The data stride of source data
 * @param [in] count       Count of data to copy
 * @param [in] format_size Data format byte size, such as sizeof(uint8_t), sizeof(float), etc.
 *                         format_size only support 1/2/4.
 *
 * dst_stride MUST be 1, EXCEPT: dst_stride == 4 && src_stride == 1 && format_size ==1
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_memcpy_d2d_stride(bm_handle_t     handle,
                                 bm_device_mem_t dst,
                                 int             dst_stride,
                                 bm_device_mem_t src,
                                 int             src_stride,
                                 int             count,
                                 int             format_size);

/**
 * @name    bm_memcpy_c2c
 * @brief   To copy data from one chip to another chip.
 *          (Used in multi-chip card scenario)
 * @ingroup bmlib_runtime
 *
 * @param [in] src_handle The source device handle
 * @param [in] dst_handle The destination device handle
 * @param [in] src        The source device memory descriptor
 * @param [in] dst        The destination device memory descriptor
 * @param [in] force_dst_cdma If use the CDMA engine of the destination device
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_memcpy_c2c(bm_handle_t src_handle, bm_handle_t dst_handle,
                          bm_device_mem_t src, bm_device_mem_t dst,
                          bool force_dst_cdma);

/**
 * @name    bm_memset_device
 * @brief   To fill in specified device memory with the given value
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   value  The value used to fill. (int type)
 * @param [in]  mem     The device memory which will be filled in
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_memset_device(bm_handle_t handle, const int value,
                             bm_device_mem_t mem);

/**
 * @name    bm_mem_convert_system_to_device_neuron
 * @brief   To malloc a piece of device memory according to the shape of
 *          neuron(in DWORD 4 bytes); copy neuron from system memory to
 *          device memory if need_copy is true.
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  dev_mem The device memory descriptor
 * @param [in]  sys_mem The system memory descriptor
 * @param [in]  need_copy If copy from system to device is needed
 * @param [in]  n,c,h,w  Neuron shape size
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_mem_convert_system_to_device_neuron(bm_handle_t handle,
                                                   struct bm_mem_desc *dev_mem,
                                                   struct bm_mem_desc sys_mem,
                                                   bool need_copy, int n, int c,
                                                   int h, int w);

/**
 * @name    bm_mem_convert_system_to_device_neuron_byte
 * @brief   To malloc a piece of device memory according to the shape of
 *          neuron(in bytes); copy neuron from system memory to
 *          device memory if need_copy is true.
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  dev_mem The device memory descriptor
 * @param [in]  sys_mem The system memory descriptor
 * @param [in]  need_copy If copy from system to device is needed
 * @param [in]  n,c,h,w  Neuron shape size
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_mem_convert_system_to_device_neuron_byte(
    bm_handle_t handle, struct bm_mem_desc *dev_mem, struct bm_mem_desc sys_mem,
    bool need_copy, int n, int c, int h, int w);

/**
 * @name    bm_mem_convert_system_to_device_coeff
 * @brief   To malloc a piece of device memory according to the size of
 *          coefficient (in DWORD 4 bytes); copy coefficient from system
 *          memory to device memory if need_copy is true.
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  dev_mem The device memory descriptor
 * @param [in]  sys_mem The system memory descriptor
 * @param [in]  need_copy If copy from system to device is needed
 * @param [in]  coeff_count Coefficient size
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_mem_convert_system_to_device_coeff(bm_handle_t handle,
                                                  struct bm_mem_desc *dev_mem,
                                                  struct bm_mem_desc sys_mem,
                                                  bool need_copy,
                                                  int coeff_count);
/**
 * @name    bm_mem_convert_system_to_device_coeff_byte
 * @brief   To malloc a piece of device memory according to the size of
 *          coefficient (in bytes); copy coefficient from system
 *          memory to device memory if need_copy is true.
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  dev_mem The device memory descriptor
 * @param [in]  sys_mem The system memory descriptor
 * @param [in]  need_copy If copy from system to device is needed
 * @param [in]  coeff_count Coefficient size
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_mem_convert_system_to_device_coeff_byte(
    bm_handle_t handle, struct bm_mem_desc *dev_mem, struct bm_mem_desc sys_mem,
    bool need_copy, int coeff_count);

/*******************memory map functions *************************************/
/**
 * @name    bm_mem_mmap_device_mem
 * @brief   To map a piece of device memory to user space with cache enabled.
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  dev_mem The device memory to map
 * @param [out] vmem    The virtual address of the mapped device memory
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_mem_mmap_device_mem(bm_handle_t handle, bm_device_mem_t *dmem,
                                   unsigned long long *vmem);

/**
 * @name    bm_mem_invalidate_device_mem
 * @brief   To invalidate a piece of mapped device memory to maintain
 *          cache coherence
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   dmem   The device memory to invalidate
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_mem_invalidate_device_mem(bm_handle_t handle,
                                         bm_device_mem_t *dmem);

/**
 * @name    bm_mem_invalidate_partial_device_mem
 * @brief   To invalidate part of mapped device memory to maintain
 *          cache coherence
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   dmem   The device memory to invalidate
 * @param [in]  offset  The offset of device memory address
 * @param [in]  len     The length of memory to invalidate in bytes
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_mem_invalidate_partial_device_mem(bm_handle_t handle,
                                                 bm_device_mem_t *dmem,
                                                 unsigned int offset,
                                                 unsigned int len);
/**
 * @name    bm_mem_flush_device_mem
 * @brief   To flush a piece of mapped device memory to maintain
 *          cache coherence
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   dmem   The device memory to flush
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_mem_flush_device_mem(bm_handle_t handle, bm_device_mem_t *dmem);

/**
 * @name    bm_mem_flush_partial_device_mem
 * @brief   To flush part of mapped device memory to maintain
 *          cache coherence
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   dmem   The device memory to flush
 * @param [in]  offset  The offset of device memory address
 * @param [in]  len     The length of memory to flush in bytes
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_mem_flush_partial_device_mem(bm_handle_t handle,
                                            bm_device_mem_t *dmem,
                                            unsigned int offset,
                                            unsigned int len);
/**
 * @name    bm_mem_unmap_device_mem
 * @brief   To unmap a piece of mapped device memory
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   vmem   The virtual address of the mapped device memory
 * @param [in]  size    The size of unmapped memory
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_mem_unmap_device_mem(bm_handle_t handle, void *vmem, int size);

/*******************api(kernel) functions *************************************/
/**
 * @name    bm_flush
 * @brief   To synchronize APIs of the current thread. The thread will block
 *          until all the outstanding APIs of the current thread are finished.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 */
void bm_flush(bm_handle_t handle);

/**
 * @name    bm_device_sync
 * @brief   To synchronize APIs of the device. The thread will block
 *          until all the outstanding APIs of the device are finished.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle   The device handle
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_device_sync(bm_handle_t handle);

/**
 * @name    bm_handle_sync
 * @brief   To synchronize APIs of the handle. The thread will block
 *          until all the outstanding APIs of the handle are finished.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle   The device handle
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_handle_sync(bm_handle_t handle);

/**
 * @name    bm_thread_sync
 * @brief   To synchronize APIs of the current thread. The thread will block
 *          until all the outstanding APIs of the current thread are finished.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @retval  BM_SUCCESS Succeeds.
 *          Other code Fails.
 */
bm_status_t bm_thread_sync(bm_handle_t handle);

/*******************trace and profile releated functions **********************/
typedef struct bm_profile {
  unsigned long cdma_in_time;
  unsigned long cdma_in_counter;
  unsigned long cdma_out_time;
  unsigned long cdma_out_counter;
  unsigned long tpu_process_time;
  unsigned long sent_api_counter;
  unsigned long completed_api_counter;
} bm_profile_t;
/**
 * @name    bm_get_profile
 * @brief   To get the profile data at the moment
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out] profile The result profile data
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_get_profile(bm_handle_t handle, bm_profile_t *profile);

struct bm_trace_item_data {
  int trace_type; /*0---cdma; 1---api*/
  unsigned long sent_time;
  unsigned long start_time;
  unsigned long end_time;
  int api_id;
  int cdma_dir;
};

/**
 * @name    bm_trace_enable
 * @brief   To enable trace for the current thread.
 * @ingroup bmlib_runtime
 *
 * @parama [in] handle  The device handle
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_trace_enable(bm_handle_t handle);

/**
 * @name    bm_trace_disable
 * @brief   To disable trace for the current thread.
 * @ingroup bmlib_runtime
 *
 * @parama [in] handle  The device handle
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_trace_disable(bm_handle_t handle);

/**
 * @name    bm_traceitem_number
 * @brief   To get the number of traced items of the current thread
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out] number  The number of traced items
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_traceitem_number(bm_handle_t handle, long *number);

/**
 * @name    bm_trace_dump
 * @brief   To get one traced item data
 *          (the oldest recorded item of the current thread).
 *          Once fetched, this item data is no longer recorded in kernel.
 * @ingroup  bmlib_runtime
 *
 * @param [in]  handle     The device handle
 * @param [out] trace_data The traced data
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_trace_dump(bm_handle_t handle, struct bm_trace_item_data *trace_data);

/**
 * @name    bm_trace_dump_all
 * @brief   To get all the traced item data of the current thread.
 *          Once fetched, the item data is no longer recorded in kernel.
 *          (bm_traceitem_number should be called first to determine how many
 *          items are available. Buffer should be allocated to store the
 *          traced item data).
 * @ingroup  bmlib_runtime
 *
 * @param [in]  handle     The device handle
 * @param [out] trace_data The traced data
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_trace_dump_all(bm_handle_t handle,
                              struct bm_trace_item_data *trace_data);

/**
 * @name    bm_get_last_api_process_time_us
 * @brief   This function is abandoned.
 */
bm_status_t bm_get_last_api_process_time_us(bm_handle_t handle,
                                            unsigned long *time_us);

/*******************tpu clock and module reset releated functions *************/
/**
 * @name    bm_set_module_reset
 * @brief   To reset TPU module. (Only valid in PCIE mode)
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  module  The ID of module to reset
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_set_module_reset(bm_handle_t handle, MODULE_ID module);

/**
 * @name    bm_set_clk_tpu_freq
 * @brief   To set the clock frequency of TPU (only valid in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   freq   The TPU target frequency
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_set_clk_tpu_freq(bm_handle_t handle, int freq);

/**
 * @name    bm_get_clk_tpu_freq
 * @brief   To get the clock frequency of TPU
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  freq   The current TPU frequency
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_get_clk_tpu_freq(bm_handle_t handle, int *freq);

/*******************misc functions ********************************************/
struct bm_misc_info {
  int pcie_soc_mode;  /*0---pcie; 1---soc*/
  int ddr_ecc_enable; /*0---disable; 1---enable*/
  long long ddr0a_size;
  long long ddr0b_size;
  long long ddr1_size;
  long long ddr2_size;
  unsigned int chipid;
#define BM1682_CHIPID_BIT_MASK (0X1 << 0)
#define BM1684_CHIPID_BIT_MASK (0X1 << 1)
  unsigned long chipid_bit_mask;
  unsigned int driver_version;
  int domain_bdf;
  int board_version; /*hardware board version [23:16]-mcu sw version, [15:8]-board type, [7:0]-hw version*/
};

/**
 * @name    bm_get_misc_info
 * @brief   To get miscellaneous information of the device
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle     The device handle
 * @param [out] pmisc_info The fetched misc info
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_get_misc_info(bm_handle_t handle, struct bm_misc_info *pmisc_info);

/**
 * @name    bm_get_chipid
 * @brief   To get the chipid of the device. (0x1682 / 0x1684 / 0x168?)
 * @ingroup bmlib_runtime
 *
 * @param [in] handle    The device handle
 * @param [out] p_chipid The chip id of the device
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_get_chipid(bm_handle_t handle, unsigned int *p_chipid);

#define BMLIB_LOG_QUIET    -8
#define BMLIB_LOG_PANIC     0
#define BMLIB_LOG_FATAL     8
#define BMLIB_LOG_ERROR    16
#define BMLIB_LOG_WARNING  24
#define BMLIB_LOG_INFO     32
#define BMLIB_LOG_VERBOSE  40
#define BMLIB_LOG_DEBUG    48
#define BMLIB_LOG_TRACE    56

/**
 * @name    bmlib_log_get_level
 * @brief   To get the bmlib log level
 * @ingroup bmlib_log
 *
 * @param void
 * @retval  The level of bmlib log level
 */
int  bmlib_log_get_level(void);

/**
 * @name    bmlib_log_set_level
 * @brief   To set the bmlib log level
 * @ingroup bmlib_log
 *
 * @param [in] level    The level of bmlib log level
 * @retval  void
 */
void bmlib_log_set_level(int level);

/**
 * @name    bmlib_log_set_callback
 * @brief   To set callback to get bmlib log
 * @ingroup bmlib_log
 *
 * @param [in]  callback     The callback function to get bmlib log
 * @retval  void
 */
void bmlib_log_set_callback(void (*callback)(const char*, int, const char*, va_list args));

/**
 * @name    bm_set_debug_mode
 * @brief   To set the debug mode for firmware log for tpu
 * @ingroup bmlib_log
 *
 * @param [in]  handle  The device handle
 * @param [in]  mode    The debug mode of fw log, 0/1 for disable/enable log
 * @retval  void
 */
void bm_set_debug_mode(bm_handle_t handle, int mode);

/**
 * @name    bmlib_api_dbg_callback
 * @brief   To set debug callback to get firmware log
 * @ingroup bmlib_log
 *
 * @param [in]  bmlib_api_dbg_callback  callback to get firmware log
 * @retval  void
 */
typedef void (*bmlib_api_dbg_callback)(int, int, int, const char*);
// api, result, duratioin, log, third int for api duration for future
void bmlib_set_api_dbg_callback(bmlib_api_dbg_callback callback);

/**
 * @name    bm_start_cpu
 * @brief   Start cpu in pcie mode
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  boot_file       Fip file
 * @param [in]  core_file       Itb file
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_start_cpu(bm_handle_t handle, char *boot_file, char *core_file);

/**
 * @name    bm_open_process
 * @brief   Open a process to do some work
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  flags           Process flags
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  >= 0 process handle
 *          < 0  Other code Fails.
 */
int bm_open_process(bm_handle_t handle, unsigned int flags, int timeout);

/**
 * @name    bm_load_library
 * @brief   Load a share library(so) to specific process
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  process_handle  Process handle
 * @param [in]  library_file    Library file path
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_load_library(bm_handle_t handle, int process_handle, char *library_file, int timeout);

/**
 * @name    bm_exec_function
 * @brief   Execute specific function in specific process
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  process_handle  Process handle
 * @param [in]  function_name   Function name
 * @param [in]  function_param  Function parameters
 * @param [in]  param_size      Parameters size in bytes
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  0   success.
 *          >0  code fails from bmlib
 *          <0  code fails from function
 */
int bm_exec_function(bm_handle_t handle,
                     int process_handle,
                     char *function_name,
                     void *function_param,
                     unsigned int param_size,
                     int timeout);

/**
 * @name    bm_exec_function_async
 * @brief   Execute specific function in specific process asynchronous
 *          user should use bm_query_exec_function_result to query result
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  process_handle  Process handle
 * @param [in]  function_name   Function name
 * @param [in]  function_param  Function param
 * @param [in]  param_size      Param size in bytes
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_exec_function_async(bm_handle_t handle,
                                   int process_handle,
                                   char *function_name,
                                   void *function_param,
                                   unsigned int param_size,
                                   unsigned long long *api_handle);

/**
 * @name    bm_query_exec_function_result
 * @brief   Query result from function called by bm_exec_function
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  api_handle      Api handle return by bm_exec_function_async
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  0   success.
 *          >0  code fails from bmlib
 *          <0  code fails from function
 */
int bm_query_exec_function_result(bm_handle_t handle, unsigned long long api_handle, int timeout);

/**
 * @name    bm_map_phys_addr
 * @brief   Map physical address in specific process
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  process_handle  Process handle
 * @param [in]  phys_addr       Physical address
 * @param [in]  size            Map size in bytes
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  >0  virtual address
 *          0   fails
 */
void *bm_map_phys_addr(bm_handle_t handle, int process_handle, void *phys_addr, unsigned int size, int timeout);

/**
 * @name    bm_close_process
 * @brief   Close process
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  process_handle  Process handle
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_close_process(bm_handle_t handle, int process_handle, int timeout);

/**
 * @name    bm_enable_perf_monitor
 * @brief   enable perf monitor to get gdma and tpu performance data
 * @ingroup bmlib_perf
 *
 * @param [in]  handle         The device handle
 * @param [in]  perf_monitor   The monitor to perf
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_enable_perf_monitor(bm_handle_t handle, bm_perf_monitor *perf_monitor);

/**
 * @name    bm_disable_perf_monitor
 * @brief   disable perf monitor to get gdma and tpu performance data
 * @ingroup bmlib_perf
 *
 * @param [in]  handle         The device handle
 * @param [in]  perf_monitor   The monitor to perf
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_disable_perf_monitor(bm_handle_t handle, bm_perf_monitor *perf_monitor);

/**
 * @name    bm_set_log
 * @brief   Set log options
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  log_level       0: DEBUG  1:INFO 2:WARN 3:ERROR 4:FATAL
 * @param [in]  log_to_console  1: YES  0: No
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_set_log(bm_handle_t handle, unsigned int log_level,  unsigned int log_to_console, int timeout);

/**
 * @name    bm_get_log
 * @brief   Get log file
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  process_handle  Process handle
 * @param [in]  log_file        save log as file
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_get_log(bm_handle_t handle, int process_handle, char *log_file, int timeout);

/**
 * @name    bm_sync_cpu_time
 * @brief   Sync device cpu time with host
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_sync_cpu_time(bm_handle_t handle);

/*******************trace and profile releated functions **********************/
typedef struct bm_dev_stat {
  int mem_total;
  int mem_used;
  int tpu_util;
} bm_dev_stat_t;

/**
 * @name    bm_get_stat
 * @brief   To get the stat data at the moment
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out] profile The result stat data
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_get_stat(bm_handle_t handle, bm_dev_stat_t *stat);

#if defined(__cplusplus)
}
#endif

#endif /* BM_RUNTIME_H_ */
