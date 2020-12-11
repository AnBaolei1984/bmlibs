#ifndef BMCV_API_EXT_H
#define BMCV_API_EXT_H
#include "bmlib_runtime.h"

#if defined(__cplusplus)
extern "C" {
#endif

/*
 * bmcv api with the new interface.
 */

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
} __attribute__((packed)) face_rect_t;

#define MIN_PROPOSAL_NUM (1)
#define MAX_PROPOSAL_NUM (8192)
typedef struct nms_proposal {
    int          size;
    int          capacity;
    face_rect_t  face_rect[MAX_PROPOSAL_NUM];
    face_rect_t *begin;
    face_rect_t *end;
} __attribute__((packed)) nms_proposal_t;

#define MAX_RECT_NUM (8 * 1024)
typedef struct {
    face_rect_t  face_rect[MAX_RECT_NUM];
    int          size;
    int          capacity;
    face_rect_t *begin;
    face_rect_t *end;
} __attribute__((packed)) m_proposal_t;

typedef enum {
    LINEAR_WEIGHTING = 0,
    GAUSSIAN_WEIGHTING,
    MAX_WEIGHTING_TYPE
} weighting_method_e;

typedef enum bmcv_heap_id_ {
    BMCV_HEAP0_ID = 0,
    BMCV_HEAP1_ID = 1,
    BMCV_HEAP_ANY
} bmcv_heap_id;

// BMCV_IMAGE_FOR_IN and BMCV_IMAGE_FOR_OUT may be deprecated in future version.
// We recommend not use this.
#define BMCV_IMAGE_FOR_IN BMCV_HEAP1_ID
#define BMCV_IMAGE_FOR_OUT BMCV_HEAP0_ID

typedef enum bm_image_data_format_ext_ {
    DATA_TYPE_EXT_FLOAT32,
    DATA_TYPE_EXT_1N_BYTE,
    DATA_TYPE_EXT_4N_BYTE,
    DATA_TYPE_EXT_1N_BYTE_SIGNED,
    DATA_TYPE_EXT_4N_BYTE_SIGNED
} bm_image_data_format_ext;

typedef enum bm_image_format_ext_ {
    FORMAT_YUV420P,
    FORMAT_YUV422P,
    FORMAT_YUV444P,
    FORMAT_NV12,
    FORMAT_NV21,
    FORMAT_NV16,
    FORMAT_NV61,
    FORMAT_NV24,
    FORMAT_RGB_PLANAR,
    FORMAT_BGR_PLANAR,
    FORMAT_RGB_PACKED,
    FORMAT_BGR_PACKED,
    FORMAT_RGBP_SEPARATE,
    FORMAT_BGRP_SEPARATE,
    FORMAT_GRAY,
    FORMAT_COMPRESSED,
} bm_image_format_ext;

typedef enum bmcv_resize_algorithm_ {
    BMCV_INTER_NEAREST = 0,
    BMCV_INTER_LINEAR
} bmcv_resize_algorithm;

typedef enum bm_cv_nms_alg_ {
    HARD_NMS = 0,
    SOFT_NMS,
    ADAPTIVE_NMS,
    SSD_NMS,
    MAX_NMS_TYPE
} bm_cv_nms_alg_e;

struct bm_image_private;

struct bm_image {
    int                      width;
    int                      height;
    bm_image_format_ext      image_format;
    bm_image_data_format_ext data_type;
    bm_image_private *       image_private = NULL;
};

typedef struct bmcv_rect {
    int start_x;
    int start_y;
    int crop_w;
    int crop_h;
} bmcv_rect_t;

typedef struct bmcv_copy_to_atrr_s {
    int           start_x;
    int           start_y;
    unsigned char padding_r;
    unsigned char padding_g;
    unsigned char padding_b;
    int           if_padding;
} bmcv_copy_to_atrr_t;

typedef struct bmcv_padding_atrr_s {
    unsigned int  dst_crop_stx;
    unsigned int  dst_crop_sty;
    unsigned int  dst_crop_w;
    unsigned int  dst_crop_h;
    unsigned char padding_r;
    unsigned char padding_g;
    unsigned char padding_b;
    int           if_memset;
} bmcv_padding_atrr_t;

typedef struct bm_image_format_info {
    int                      plane_nb;
    bm_device_mem_t          plane_data[8];
    int                      stride[8];
    int                      width;
    int                      height;
    bm_image_format_ext      image_format;
    bm_image_data_format_ext data_type;
    bool                     default_stride;
} bm_image_format_info_t;

typedef struct {
    int csc_coe00;
    int csc_coe01;
    int csc_coe02;
    int csc_add0;
    int csc_coe10;
    int csc_coe11;
    int csc_coe12;
    int csc_add1;
    int csc_coe20;
    int csc_coe21;
    int csc_coe22;
    int csc_add2;
} __attribute__((packed)) csc_matrix_t;

typedef enum csc_type {
    CSC_YCbCr2RGB_BT601 = 0,
    CSC_YPbPr2RGB_BT601,
    CSC_RGB2YCbCr_BT601,
    CSC_YCbCr2RGB_BT709,
    CSC_RGB2YCbCr_BT709,
    CSC_RGB2YPbPr_BT601,
    CSC_YPbPr2RGB_BT709,
    CSC_RGB2YPbPr_BT709,
    CSC_USER_DEFINED_MATRIX = 1000,
    CSC_MAX_ENUM
} csc_type_t;

const char *bm_get_bmcv_version();

/** bm_image_create
 * @brief Create and fill bm_image structure
 * @param [in] handle                     The bm handle which return by
 * bm_dev_request.
 * @param [in] img_h                      The height or rows of the creating
 * image.
 * @param [in] img_w                     The width or cols of the creating
 * image.
 * @param [in] image_format      The image_format of the creating image,
 *  please choose one from bm_image_format_ext enum.
 * @param [in] data_type               The data_type of the creating image,
 *  be caution that not all combinations between image_format and data_type
 *  are supported.
 * @param [in] stride                        the stride array for planes, each
 * number in array means corresponding plane pitch stride in bytes. The plane
 * size is determinated by image_format. If this array is null, we may use
 * default value.
 *  @param [out] image                   The filled bm_image structure.
 *  For example, we need create a 480x480 NV12 format image, we know that NV12
 * format has 2 planes, we need pitch stride is 256 aligned(just for example) so
 * the pitch stride for the first plane is 512, so as the same for the second
 * plane.
 * The call may as following
 * bm_image res;
 *  int stride[] = {512, 512};
 *  bm_image_create(handle, 480, 480, FORMAT_NV12, DATA_TYPE_EXT_1N_BYTE, &res,
 * stride); If bm_image_create return BM_SUCCESS, res is created successfully.
 */
bm_status_t bm_image_create(bm_handle_t              handle,
                            int                      img_h,
                            int                      img_w,
                            bm_image_format_ext      image_format,
                            bm_image_data_format_ext data_type,
                            bm_image *               image,
                            int *                    stride = NULL);

/** bm_image_destroy
 * @brief Destroy bm_image and free the corresponding system memory and device
 * memory.
 * @param [in] image                     The bm_image structure ready to
 * destroy. If bm_image_destroy return BM_SUCCESS, image is destroy successfully
 * and the corresponding system memory and device memory are freed.
 */
bm_status_t bm_image_destroy(bm_image image);

/** bm_image_get_handle
 * @brief return the device handle, this handle is exactly the first parameter
 * when bm_image_create called.
 * @param [in] image                                   The bm_image structure
 *  @param [return] bm_handle_t          The device handle where bm_image bind
 * to. If image is not created by bm_image_create, this function would return
 * NULL.
 */
bm_handle_t bm_image_get_handle(bm_image *image);

/** bm_image_write_to_bmp
 * @brief dump this bm_image to .bmp file.
 * @param [in] image                 The bm_image structure you would like to
 * dump
 *  @param [in] filename           path and filename for the creating bmp file,
 * it's better end with ".bmp" If bm_image_write_to_bmp return BM_SUCCESS, a
 * .bmp file is create in the path filename point to.
 */
bm_status_t bm_image_write_to_bmp(bm_image image, const char *filename);

bm_status_t bm_image_copy_host_to_device(bm_image image, void *buffers[]);
bm_status_t bm_image_copy_device_to_host(bm_image image, void *buffers[]);

bm_status_t bm_image_attach(bm_image image, bm_device_mem_t *device_memory);
bm_status_t bm_image_detach(bm_image);
bool        bm_image_is_attached(bm_image);
int         bm_image_get_plane_num(bm_image);
bm_status_t bm_image_get_stride(bm_image image, int *stride);
bm_status_t bm_image_get_format_info(bm_image *            image,
                                     bm_image_format_info *info);

bm_status_t bm_image_alloc_dev_mem(bm_image image, int heap_id = BMCV_HEAP_ANY);
bm_status_t bm_image_alloc_dev_mem_heap_mask(bm_image image, int heap_mask);
bm_status_t bm_image_get_byte_size(bm_image image, int *size);
bm_status_t bm_image_get_device_mem(bm_image image, bm_device_mem_t *mem);

bm_status_t bm_image_alloc_contiguous_mem(int       image_num,
                                          bm_image *images,
                                          int       heap_id = BMCV_HEAP_ANY);
bm_status_t bm_image_alloc_contiguous_mem_heap_mask(int       image_num,
                                                    bm_image *images,
                                                    int       heap_mask);
bm_status_t bm_image_free_contiguous_mem(int image_num, bm_image *images);
bm_status_t bm_image_attach_contiguous_mem(int             image_num,
                                           bm_image *      images,
                                           bm_device_mem_t dmem);
bm_status_t bm_image_dettach_contiguous_mem(int image_num, bm_image *images);

bm_status_t bm_image_get_contiguous_device_mem(int              image_num,
                                               bm_image *       images,
                                               bm_device_mem_t *mem);

bm_status_t bmcv_image_yuv2bgr_ext(bm_handle_t handle,
                                   int         image_num,
                                   bm_image *  input,
                                   bm_image *  output);

bm_status_t bmcv_image_storage_convert(bm_handle_t handle,
                                       int         image_num,
                                       bm_image *  input,
                                       bm_image *  output);

bm_status_t bmcv_image_storage_convert_with_csctype(bm_handle_t handle,
                                                    int         image_num,
                                                    bm_image *  input,
                                                    bm_image *  output,
                                                    csc_type_t  csc_type);

bm_status_t bmcv_image_copy_to(bm_handle_t         handle,
                               bmcv_copy_to_atrr_t copy_to_attr,
                               bm_image            input,
                               bm_image            output);

bm_status_t bmcv_image_crop(bm_handle_t         handle,
                            int                 crop_num,
                            bmcv_rect_t *       rects,
                            bm_image            input,
                            bm_image *          output);

bm_status_t bmcv_image_split(bm_handle_t         handle,
                             bm_image            input,
                             bm_image *          output);

typedef struct bmcv_affine_matrix_s {
    float m[6];
} bmcv_affine_matrix;

typedef struct bmcv_affine_image_matrix_s {
    bmcv_affine_matrix *matrix;
    int                 matrix_num;
} bmcv_affine_image_matrix;

typedef struct bmcv_perspective_matrix_s {
    float m[9];
} bmcv_perspective_matrix;

typedef struct bmcv_perspective_image_matrix_s {
    bmcv_perspective_matrix *matrix;
    int                      matrix_num;
} bmcv_perspective_image_matrix;

typedef struct bmcv_perspective_coordinate_s {
    int x[4];
    int y[4];
} bmcv_perspective_coordinate;

typedef struct bmcv_perspective_image_coordinate_s {
    bmcv_perspective_coordinate *coordinate;
    int                         coordinate_num;
} bmcv_perspective_image_coordinate;

typedef struct bmcv_resize_s {
    int start_x;
    int start_y;
    int in_width;
    int in_height;
    int out_width;
    int out_height;
} bmcv_resize_t;

typedef struct bmcv_resize_image_s {
    bmcv_resize_t *resize_img_attr;
    int            roi_num;
    unsigned char  stretch_fit;
    unsigned char  padding_b;
    unsigned char  padding_g;
    unsigned char  padding_r;
    unsigned int   interpolation;
} bmcv_resize_image;

typedef struct bmcv_convert_to_attr_s {
    float alpha_0;
    float beta_0;
    float alpha_1;
    float beta_1;
    float alpha_2;
    float beta_2;
} bmcv_convert_to_attr;
/**
 * @brief Do warp affine operation with the transform matrix.
 *        For 1N mode, only support 4 images.
 *        For 4N mode, only support 1 images.
 * @param [in] handle        The bm handle which return by bm_dev_request.
 * @param [in] image_num    The really input image number, should be less than
 *or equal to 4.
 * @param [in] matrix        The input transform matrix and matrix number for
 *each image.
 * @param [in] input        The input bm image, could be 1N or 4N.
 *                for each image. And do any operation if matrix[n] is nullptr.
 * @param [out]            The output image, could be 1N or 4N.
 *                If setting to 1N, the output image number should have summary
 *of matrix_num[n]. If setting to 4N, the output image number should have
 *summary of ROUNDUP(matrix_num[n], 4)/4
 */
bm_status_t bmcv_image_warp_affine(
        bm_handle_t              handle,
        int                      image_num,
        bmcv_affine_image_matrix matrix[4],
        bm_image *               input,
        bm_image *               output,
        int                      use_bilinear = 0);

bm_status_t bmcv_image_warp_perspective(
        bm_handle_t                   handle,
        int                           image_num,
        bmcv_perspective_image_matrix matrix[4],
        bm_image *                    input,
        bm_image *                    output,
        int                           use_bilinear = 0);

bm_status_t bmcv_image_warp_perspective_with_coordinate(
        bm_handle_t                       handle,
        int                               image_num,
        bmcv_perspective_image_coordinate coordinate[4],
        bm_image *                        input,
        bm_image *                        output,
        int                               use_bilinear = 0);

bm_status_t bmcv_image_resize(
        bm_handle_t          handle,
        int                  input_num,
        bmcv_resize_image    resize_attr[],
        bm_image *           input,
        bm_image *           output);

bm_status_t bmcv_image_convert_to(
        bm_handle_t          handle,
        int                  input_num,
        bmcv_convert_to_attr convert_to_attr,
        bm_image *           input,
        bm_image *           output);

bm_status_t bmcv_width_align(
        bm_handle_t handle,
        bm_image    input,
        bm_image    output);

bm_status_t bmcv_image_jpeg_enc(
        bm_handle_t handle,
        int         image_num,
        bm_image *  src,
        void **     p_jpeg_data,
        size_t *    out_size,
        int         quality_factor = 85);

bm_status_t bmcv_image_jpeg_dec(
        bm_handle_t handle,
        void **     p_jpeg_data,
        size_t *    in_size,
        int         image_num,
        bm_image *  dst);

bm_status_t bmcv_nms(
        bm_handle_t     handle,
        bm_device_mem_t input_proposal_addr,
        int             proposal_size,
        float           nms_threshold,
        bm_device_mem_t output_proposal_addr);

bm_status_t bmcv_nms_ext(
        bm_handle_t     handle,
        bm_device_mem_t input_proposal_addr,
        int             proposal_size,
        float           nms_threshold,
        bm_device_mem_t output_proposal_addr,
        int             topk            = 1,
        float           score_threshold = 0.0f,
        int             nms_alg         = HARD_NMS,
        float           sigma = 1.0,
        int             weighting_method = 0,
        float         * densities = NULL,
        float           eta = 0.0f);

bm_status_t bmcv_image_draw_rectangle(
        bm_handle_t   handle,
        bm_image      image,
        int           rect_num,
        bmcv_rect_t * rect,
        int           line_width,
        unsigned char r,
        unsigned char g,
        unsigned char b);

bm_status_t bmcv_image_fill_rectangle(
        bm_handle_t   handle,
        bm_image      image,
        int           rect_num,
        bmcv_rect_t * rect,
        unsigned char r,
        unsigned char g,
        unsigned char b);

bm_status_t bmcv_sort(
        bm_handle_t     handle,
        bm_device_mem_t src_index_addr,
        bm_device_mem_t src_data_addr,
        int             data_cnt,
        bm_device_mem_t dst_index_addr,
        bm_device_mem_t dst_data_addr,
        int             sort_cnt,
        int             order,
        bool            index_enable,
        bool            auto_index);

bm_status_t bmcv_feature_match_normalized(
        bm_handle_t     handle,
        bm_device_mem_t input_data_global_addr,
        bm_device_mem_t db_data_global_addr,
        bm_device_mem_t db_feature_global_addr,
        bm_device_mem_t output_similarity_global_addr,
        bm_device_mem_t output_index_global_addr,
        int             batch_size,
        int             feature_size,
        int             db_size);

bm_status_t bmcv_feature_match(
        bm_handle_t     handle,
        bm_device_mem_t input_data_global_addr,
        bm_device_mem_t db_data_global_addr,
        bm_device_mem_t output_sorted_similarity_global_addr,
        bm_device_mem_t output_sorted_index_global_addr,
        int             batch_size,
        int             feature_size,
        int             db_size,
        int             sort_cnt   = 1,
        int             rshiftbits = 0);

bm_status_t bmcv_base64_enc(
        bm_handle_t     handle,
        bm_device_mem_t src,
        bm_device_mem_t dst,
        unsigned long   len[2]);

bm_status_t bmcv_base64_dec(
        bm_handle_t     handle,
        bm_device_mem_t src,
        bm_device_mem_t dst,
        unsigned long   len[2]);

bm_status_t bmcv_debug_savedata(bm_image image, const char *name);

bm_status_t bmcv_image_transpose(bm_handle_t handle,
                                 bm_image input,
                                 bm_image output);

bm_status_t bmcv_matmul(
        bm_handle_t      handle,
        int              M,
        int              N,
        int              K,
        bm_device_mem_t  A,
        bm_device_mem_t  B,
        bm_device_mem_t  C,
        int              A_sign,  // 1: signed 0: unsigned
        int              B_sign,
        int              rshift_bit,
        bool             is_C_16bit,  // else 8bit
        bool             is_B_trans);

#ifndef USING_CMODEL
bm_status_t bmcv_image_vpp_basic(bm_handle_t           handle,
                                 int                   in_img_num,
                                 bm_image*             input,
                                 bm_image*             output,
                                 int*                  crop_num_vec = NULL,
                                 bmcv_rect_t*          crop_rect = NULL,
                                 bmcv_padding_atrr_t*  padding_attr = NULL,
                                 bmcv_resize_algorithm algorithm = BMCV_INTER_LINEAR,
                                 csc_type_t            csc_type = CSC_MAX_ENUM,
                                 csc_matrix_t*         matrix = NULL);

bm_status_t bmcv_image_vpp_convert(
    bm_handle_t           handle,
    int                   output_num,
    bm_image              input,
    bm_image *            output,
    bmcv_rect_t *         crop_rect = NULL,
    bmcv_resize_algorithm algorithm = BMCV_INTER_LINEAR);

bm_status_t bmcv_image_vpp_csc_matrix_convert(bm_handle_t           handle,
                                              int                   output_num,
                                              bm_image              input,
                                              bm_image *            output,
                                              csc_type_t            csc,
                                              csc_matrix_t *        matrix = nullptr,
                                              bmcv_resize_algorithm algorithm = BMCV_INTER_LINEAR,
                                              bmcv_rect_t *         crop_rect = NULL);

bm_status_t bmcv_image_vpp_convert_padding(
    bm_handle_t           handle,
    int                   output_num,
    bm_image              input,
    bm_image *            output,
    bmcv_padding_atrr_t *        padding_attr,
    bmcv_rect_t *         crop_rect = NULL,
    bmcv_resize_algorithm algorithm = BMCV_INTER_LINEAR);

bm_status_t bmcv_image_vpp_stitch(
    bm_handle_t           handle,
    int                   input_num,
    bm_image*              input,
    bm_image            output,
    bmcv_rect_t*         dst_crop_rect,
    bmcv_rect_t*         src_crop_rect = NULL,
    bmcv_resize_algorithm algorithm = BMCV_INTER_LINEAR);
#endif

/**
 * Legacy functions
 */

typedef bmcv_affine_image_matrix bmcv_warp_image_matrix;
typedef bmcv_affine_matrix bmcv_warp_matrix;

bm_status_t bmcv_image_warp(
        bm_handle_t            handle,
        int                    image_num,
        bmcv_warp_image_matrix matrix[4],
        bm_image *             input,
        bm_image *output) __attribute__((deprecated));

bm_status_t bm_image_dev_mem_alloc(bm_image image,
                                   int heap_id = BMCV_HEAP_ANY) __attribute__((deprecated));


#if defined(__cplusplus)
}
#endif

#endif /* BMCV_API_EXT_H */
