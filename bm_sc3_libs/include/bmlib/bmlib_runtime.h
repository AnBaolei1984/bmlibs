#ifndef BMLIB_RUNTIME_H_
#define BMLIB_RUNTIME_H_
#include <stdbool.h>
#include <stddef.h>
#include <stdarg.h>

#if !defined(__x86_64__) && !defined(__aarch64__)
#error "BM needs 64-bit to compile"
#endif

#if defined (__cplusplus)
extern "C" {
#endif

#define BMLIB_DBG_LEVEL 1
extern int PLUS_LEVEL;

#define ASSERT_S(_cond)                           \
  do {                                            \
    if (!(_cond)) {                              \
      if((PRINT_LEVEL+PLUS_LEVEL)<BMLIB_DBG_LEVEL){\
        printf("ASSERT_S %s: %s: %d: %s\n",          \
          __FILE__, __func__, __LINE__, #_cond); \
        print_trace();                             \
      }\
      return (bm_status_t)(-1);                               \
    }                                            \
  } while(0)

#define BM_MEM_ADDR_NULL     (0xfffffffff)

#ifndef BM_MEM_DESC_T_
#define BM_MEM_DESC_T_
typedef enum {
  BM_SUCCESS                 = 0,
  BM_ERR_DEVNOTREADY         = 1,   /* Device not ready yet */
  BM_ERR_FAILURE             = 2,   /* General failure */
  BM_ERR_TIMEOUT             = 3,   /* Timeout */
  BM_ERR_PARAM               = 4,   /* Parameters invalid */
  BM_ERR_NOMEM               = 5,   /* Not enough memory */
  BM_ERR_DATA                = 6,   /* Data error */
  BM_ERR_BUSY                = 7,   /* Busy */
  BM_ERR_NOFEATURE           = 8,    /* Not supported yet */
  BM_NOT_SUPPORTED           = 9,
  BM_ERR_VERSION             = 10   /* version mis-match between bmlib and driver */
} bm_status_t;

typedef enum {
  BM_MEM_TYPE_DEVICE  = 0,
  BM_MEM_TYPE_HOST    = 1,
  BM_MEM_TYPE_SYSTEM  = 2,
  BM_MEM_TYPE_INT8_DEVICE  = 3,
  BM_MEM_TYPE_INVALID = 4
} bm_mem_type_t;

typedef enum {
  PERF_MONITOR_GDMA = 0,
  PERF_MONITOR_TPU = 1
} PERF_MONITOR_ID;

/*
* bm performace monitor
*/
typedef struct {
  long long buffer_start_addr; /*buffer address to store perf data*/
  int buffer_size; /*buffer size*/
  PERF_MONITOR_ID monitor_id; /*PERF_MONITOR_GDMA or PERF_MONITOR_TPU*/
}bm_perf_monitor;

typedef union {
	struct {
		bm_mem_type_t        mem_type : 3;
		unsigned int         reserved : 29;
	} u;
	unsigned int           rawflags;
} bm_mem_flags_t;

typedef struct bm_mem_desc {
	union {
		struct {
			unsigned long         device_addr;
			unsigned int         reserved;
			int         dmabuf_fd;
		} device;

		struct {
			void *      system_addr;
			unsigned int reserved0;
			int         reserved1;
		} system;
	} u;

	bm_mem_flags_t         flags;
	unsigned int                    size;
} bm_mem_desc_t;

typedef struct bm_mem_desc   bm_device_mem_t;
typedef struct bm_mem_desc   bm_system_mem_t;
#endif

struct bm_context;
typedef struct bm_context *  bm_handle_t;

#define BM_CHECK_RET(call)                         \
    do {                                        \
      bm_status_t ret = call;                \
	  if ( ret != BM_SUCCESS ) {             \
	if(PRINT_LEVEL+PLUS_LEVEL<BMLIB_DBG_LEVEL){\
        printf("BM_CHECK_RET failed %d\n", ret);   \
 	}\
        ASSERT_S(0);                              \
      }                                         \
    } while(0)

/*
 * control
 */
void bm_flush(
    bm_handle_t      handle);

bm_status_t bm_flush_s(
    bm_handle_t      handle);

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
 * @brief   @brief   This function is abandoned.
 */
bm_status_t bm_get_profile(bm_handle_t handle, bm_profile_t *profile);

/*
 * brief malloc device memory according to a tensor shape(each neuron is 32 bits)
*/

bm_status_t bm_malloc_neuron_device(
    bm_handle_t      handle,
    bm_device_mem_t *pmem,
    int              n,
    int              c,
    int              h,
    int              w);

/*
 * brief malloc device memory in size of dword(32 bits)
*/

bm_status_t bm_malloc_device_dword(
    bm_handle_t      handle,
    bm_device_mem_t *pmem,
    int              count);
/*
 * brief malloc device memory in size of byte
*/

bm_status_t bm_malloc_device_byte(
    bm_handle_t      handle,
    bm_device_mem_t *pmem,
    unsigned int     size);

void bm_free_device(
    bm_handle_t      handle,
    bm_device_mem_t  mem);

bm_status_t bm_free_device_s(
    bm_handle_t      handle,
    bm_device_mem_t  mem);

/*
 * Memory copy and set
 */

bm_status_t bm_memcpy_s2d(
    bm_handle_t      handle,
    bm_device_mem_t  dst,
    void            *src);

bm_status_t bm_memcpy_s2d_partial_offset(
    bm_handle_t handle,
    bm_device_mem_t dst,
    void           *src,
    unsigned int size,
    unsigned int offset);

bm_status_t bm_memcpy_s2d_partial(
    bm_handle_t handle,
    bm_device_mem_t dst,
    void           *src,
    unsigned int size);

bm_status_t bm_memcpy_d2s(
    bm_handle_t      handle,
    void            *dst,
    bm_device_mem_t  src);

bm_status_t bm_memcpy_d2s_partial_offset(
    bm_handle_t handle,
    void           *dst,
    bm_device_mem_t src,
    unsigned int size,
    unsigned int offset);

bm_status_t bm_memcpy_d2s_partial(
    bm_handle_t handle,
    void           *dst,
    bm_device_mem_t src,
    unsigned int size);

bm_status_t bm_memcpy_d2d(
    bm_handle_t     handle,
    bm_device_mem_t dst,
    int             dst_offset,
    bm_device_mem_t src,
    int             src_offset,
    int             len);

// 1682 not support
bm_status_t bm_memcpy_d2d_byte(bm_handle_t handle,
                               bm_device_mem_t dst,
                               size_t dst_offset,
                               bm_device_mem_t src,
                               size_t src_offset,
                               size_t size);

// 1682 not support
bm_status_t bm_memcpy_d2d_stride(bm_handle_t handle,
                                 bm_device_mem_t dst,
                                 int dst_stride,
                                 bm_device_mem_t src,
                                 int src_stride,
                                 int count,
                                 int format_size);

bm_status_t bm_memset_device(
    bm_handle_t      handle,
    const int        value,
    bm_device_mem_t  mem);

bm_device_mem_t bm_mem_from_system(void* system_addr);

bm_device_mem_t bm_mem_from_device(
    unsigned long long  device_addr,
    int                 len);

/*
*brief malloc one device memory with the shape of (N,C,H,W), copy the sys_mem to
device mem if need_copy is true
*/

bm_status_t bm_mem_convert_system_to_device_neuron(
    bm_handle_t          handle,
    struct bm_mem_desc  *dev_mem,
    struct bm_mem_desc   sys_mem,
    bool                 need_copy,
    int                  n,
    int                  c,
    int                  h,
    int                  w);

/*
*brief malloc one device memory with the size of coeff_count, copy the sys_mem to
device mem if need_copy is true
*/
bm_status_t bm_mem_convert_system_to_device_coeff(
    bm_handle_t          handle,
    struct bm_mem_desc  *dev_mem,
    struct bm_mem_desc   sys_mem,
    bool                 need_copy,
    int                  coeff_count);

/*
 * memory info get and set
 */
unsigned long long bm_mem_get_device_addr(struct bm_mem_desc mem);
void               bm_mem_set_device_addr(struct bm_mem_desc *pmem, unsigned long long addr);
unsigned int       bm_mem_get_device_size(struct bm_mem_desc mem);
void               bm_mem_set_device_size(struct bm_mem_desc *pmem, unsigned int size);
void bm_set_device_mem(bm_device_mem_t *pmem, size_t size, unsigned long long addr);
void * bm_mem_get_system_addr(struct bm_mem_desc mem);
void bm_mem_set_system_addr(struct bm_mem_desc *pmem, void* addr);
bm_mem_type_t      bm_mem_get_type(struct bm_mem_desc mem);
/*
 *  following functions are for SOC mode only
 */
bm_status_t bm_mem_mmap_device_mem(
    bm_handle_t      handle,
    bm_device_mem_t *dmem,
    unsigned long long *vmem);

bm_status_t bm_mem_invalidate_device_mem(
    bm_handle_t      handle,
    bm_device_mem_t *dmem);

bm_status_t bm_mem_invalidate_partial_device_mem(
    bm_handle_t      handle,
    bm_device_mem_t *dmem,
    unsigned int     offset,
    unsigned int     len);

bm_status_t bm_mem_flush_device_mem(
    bm_handle_t      handle,
    bm_device_mem_t *dmem);

bm_status_t bm_mem_flush_partial_device_mem(
    bm_handle_t      handle,
    bm_device_mem_t *dmem,
    unsigned int     offset,
    unsigned int     len);

bm_status_t bm_mem_unmap_device_mem(
    bm_handle_t      handle,
    void *           vmem,
    int              size);

/*
 *  end functions for SOC mode only
 */


unsigned long long bm_gmem_arm_reserved_request(bm_handle_t handle);
void bm_gmem_arm_reserved_release(bm_handle_t handle);

/*
 * Helper functions
 */

/**
* \brief Get the number of nodechip (Constant 1 in bm1682)
* \return
* \ref NO
*/
int bm_get_nodechip_num(
    bm_handle_t      handle);

/**
* \brief Get the number of nodechip (Constant 64 in bm1682)
* \return
* \ref NO
*/
int bm_get_npu_num(
    bm_handle_t      handle);
int bm_get_eu_num( bm_handle_t handle);
/**
* \brief Get the number of nodechip (Constant 64 in bm1682)
* \return
* \ref NO
*/
bm_device_mem_t bm_mem_null(void);
#define BM_MEM_NULL  (bm_mem_null())

bm_status_t bm_dev_getcount(int* count);
bm_status_t bm_dev_query(int devid);
bm_status_t bm_dev_request(bm_handle_t *handle, int devid);
void bm_dev_free(bm_handle_t handle);

void bm_enable_iommu(bm_handle_t handle);
void bm_disable_iommu(bm_handle_t handle);

typedef struct bm_fw_desc {
	unsigned int *itcm_fw;
	int itcmfw_size;
	unsigned int *ddr_fw;
	int ddrfw_size;
} bm_fw_desc, *pbm_fw_desc;
bm_status_t bm_update_firmware(bm_handle_t handle, pbm_fw_desc pfw);

/**
* \brief Initialize bmkernel running context.
*
* \param handle - pointer to per thread bmlib context.
*
* \return
* \ref BM_SUCCESS, \ref BM_ERR_FAILURE
*/
bm_status_t bmlib_kernel_init(bm_handle_t *handle);
/**
* \brief Deinitialize bmkernel running context.
*
* \param handle - per thread bmlib context.
*
*/
void bmlib_kernel_deinit(bm_handle_t handle);
/**
* \brief Launch firmware to device.
*
* \param handle - per thread bmlib context.
* \param firmware - Path of firmware file.
*
* \return
* \ref BM_SUCCESS, \ref BM_ERR_FAILURE
*/
bm_status_t bmlib_kernel_launch(bm_handle_t handle, const char *firmware);
/**
* \brief Send arguments to device.
*
* \param handle - per thread bmlib context.
* \param args - Pointer to arguments.
* \param size - Size of arguments in bytes.
*
* \note
* If args is set NULL, size will be taken as 0.
*
* \return
* \ref BM_SUCCESS, \ref BM_ERR_FAILURE
*/
bm_status_t bmlib_kernel_send_args(bm_handle_t handle, const void *args, unsigned long size);

/**
* profile data
*/
bm_status_t bm_get_last_api_process_time_us(bm_handle_t handle, unsigned long* time_us);

bm_status_t bm_get_chipid(bm_handle_t handle, unsigned int* p_chipid);

#define BMLIB_LOG_QUIET    -8
#define BMLIB_LOG_PANIC     0
#define BMLIB_LOG_FATAL     8
#define BMLIB_LOG_ERROR    16
#define BMLIB_LOG_WARNING  24
#define BMLIB_LOG_INFO     32
#define BMLIB_LOG_VERBOSE  40
#define BMLIB_LOG_DEBUG    48
#define BMLIB_LOG_TRACE    56

void bm_set_debug_mode(bm_handle_t handle, int mode);
void bmlib_log_set_level(int);
int  bmlib_log_get_level(void);
void bmlib_log_set_callback(void (*callback)(int, const char*, va_list args));

typedef void (*bmlib_api_dbg_callback)(int, int, int, const char*); // api, result, duratioin, log, third int for api duration for future
void bmlib_set_api_dbg_callback(bmlib_api_dbg_callback callback);

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

#if defined (__cplusplus)
}
#endif

#endif /* BM_RUNTIME_H_ */
