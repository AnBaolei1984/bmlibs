#ifndef _CPU_COMMON_H_
#define _CPU_COMMON_H_

/*
 * NOTE:
 * To guarantee the compatibility with all previous bmodel
 * We must follow the rules:
 *   1. Positions and numbers in CPU_LAYER_TYPE_T cannot be modified!
 *   2. New element for CPU_LAYER_TYPE_T must be added to the last!
 *   3. After release, element in each cpu layer param cannot be deleted!
 *   4. New elements for each cpu layer param must be added to the last!
 *      And cpu layer coding must considers compatibility!
 */

typedef enum {
    CPU_SSD_DETECTION_OUTPUT      = 0,  /* CAFFE  */
    CPU_ANAKIN_DETECT_OUTPUT      = 1,  /* ANAKIN */
    CPU_RPN                       = 2,
    CPU_USER_DEFINED              = 3,  /* USER DEFINED LAYER */
    CPU_ROI_POOLING               = 4,  /* ROI Pooling Layer */
    CPU_ROIALIGN                  = 5,  /* from MXNet */
    CPU_BOXNMS                    = 6,  /* from MXNet */
    CPU_YOLO                      = 7,  /* YOLO LAYER */
    CPU_CROP_AND_RESIZE           = 8,  /* CROP AND RESIZE LAYER */
    CPU_GATHER                    = 9,  /* GATHER LAYER */
    CPU_NON_MAX_SUPPRESSION       = 10, /* NON MAX SUPPRESSION LAYER */
    CPU_ARGSORT                   = 11, /* ARGSORT FROM MXNET */
    CPU_GATHERND                  = 12, /* GATHER_ND LAYER*/
    CPU_YOLOV3_DETECTION_OUTPUT   = 13, /* YOLO V3 DETECT OUT */
    CPU_WHERE                     = 14, /* WHERE LAYER */
    CPU_ADAPTIVE_AVERAGE_POOL     = 15, /* ADAPTIVE AVERAGE POOLING */
    CPU_ADAPTIVE_MAX_POOL         = 16, /* ADAPTIVE MAX POOLING */
    CPU_TOPK                      = 17, /* TOPK */
    CPU_RESIZE_INTERPOLATION      = 18, /* CPU RESIZE INTERPOLATION */
    CPU_GATHERND_TF               = 19, /* CPU GATHER_ND TENSORFLOW LAYER */
    CPU_SORT_PER_DIM              = 20, /* CPU SORT_PER_DIM LAYER */
    CPU_WHERE_SQUEEZE_GATHER      = 21, /* CPU WHERE_SQUEEZE_GATHER LAYER */
    CPU_MASKED_SELECT             = 22, /* CPU MASKED_SELECT LAYER */
    CPU_UNARY                     = 23, /* CPU UNARY LAYER */
    CPU_EMBEDDING                 = 24, /* CPU EMBEDDING */
    CPU_TOPK_MX                   = 25, /* TOPK from MXNET*/
    CPU_INDEX_PUT                 = 26, /* CPU INDEX PUT */
    CPU_SCATTER_ND                = 27, /* CPU SCATTER ND */
    CPU_RANDOM_UNIFORM            = 28, /* CPU RANDOM UNIFORM */
    CPU_GATHER_PT                 = 29, /* CPU GATHER FOR PYTORCH */
    CPU_BINARY                    = 30, /* CPU BINARY: MOD, DIV, ... */
    CPU_TENSORFLOW_NMS_V5         = 31, /* CPU TENSORFLOW NMS V5 */
    CPU_GENERATE_PROPOSALS        = 32, /* CPU GENERATE PROPOSALS */
    CPU_BBOX_TRANSFORM            = 33, /* CPU BBOX TRANSFORM */
    CPU_BOX_WITH_NMS_LIMIT        = 34, /* CPU BOX WITH NMS LIMIT */
    CPU_COLLECT_RPN_PROPOSALS     = 35, /* CPU COLLECT RPN PROPOSALS */
    CPU_DISTRIBUTE_FPN_PROPOSALS  = 36, /* CPU DISTRIBUTE FPN PROPOSALS */
    CPU_DISTRIBUTE_FPN_PROPOSALS_ROI_ALIGN_CONCAT = 37,
    CPU_PYTORCH_ROI_ALIGN         = 38, /* CPU PYTORCH ROI ALIGN */
    CPU_AFFINE_GRID_GENERATOR     = 39, /* CPU AFFINE GRID GENERATOR */
    CPU_GRID_SAMPLER              = 40, /* CPU GRID SAMPLER */
    CPU_AFFINE_GRID_SAMPLER       = 41, /* CPU AFFINE GRID SAMPLER */
    CPU_RANDOM_UNIFORM_INT        = 42, /* CPU RANDOM UNIFORM INT */
    CPU_TOPK_ASCENDING            = 43, /* CPU TOPK BY ASCENDING ORDER */
    CPU_LAYER_NUM,
    CPU_LAYER_UNKNOW = CPU_LAYER_NUM,
    CPU_DEBUG                     = 88888, /* CPU DEBUG by dump tensor*/
} CPU_LAYER_TYPE_T;

//must be the same as bmcompiler
typedef enum {
    CPU_DTYPE_FP32 = 0,
    CPU_DTYPE_FP16 = 1,
    CPU_DTYPE_INT8 = 2,
    CPU_DTYPE_UINT8 = 3,
    CPU_DTYPE_INT16 = 4,
    CPU_DTYPE_UINT16 = 5,
    CPU_DTYPE_INT32 = 6,
    CPU_DTYPE_UINT32 = 7,
    CPU_DTYPE_UNKNOWN = -1,
} CPU_DATA_TYPE_T;

typedef struct cpu_exp_param {
    float inner_scale_;
    float outer_scale_;
} cpu_exp_param_t;

typedef struct cpu_relu_param {
    float negative_slope_;
} cpu_relu_param_t;

typedef struct cpu_ssd_detect_out_param {
    int num_classes_;
    bool share_location_;

    int background_label_id_;

    float nms_threshold_;
    int top_k_;

    //CodeType code_type_;
    int code_type_;
    int keep_top_k_;
    float confidence_threshold_;

    //int num_;
    int num_priors_;
    int num_loc_classes_;
    bool variance_encoded_in_target_;
    float eta_;
    float objectness_score_;
} cpu_ssd_detect_out_param_t;

typedef struct cpu_rpnproposal_param {
    int feat_stride_;
    int min_size_;
    int pre_nms_topN_;
    int post_nms_topN_;
    float nms_thresh_;
    float score_thresh_;
    int base_size_;
    int scales_num_;
    int ratios_num_;
    int anchor_scales_[5];
    float ratios_[5];
} cpu_rpnproposal_param_t;

typedef struct cpu_roi_pooling_param {
    int pooled_height_;
    int pooled_width_;
    float spatial_scale_;
} cpu_roi_pooling_param_t;

//must be the same as bmnetm by python
typedef struct {
    int pooled_height;
    int pooled_width;
    float spatial_scale;
    int sampling_ratio;
    int position_sensitive;
} cpu_roi_align_param_t;

typedef enum {
    BOX_FORMAT_CORNER=0,
    BOX_FORMAT_CENTER=1
} box_nms_format_t;

typedef struct {
    float overlap_thresh; //Overlapping(IoU) threshold to suppress object with smaller sclore
    float valid_thresh;   //Filter input boxes to those whose scores greater than valid_thresh
    int topk;             //Apply nms to topk boxes with descending scores, -1 to no restriction
    int coord_start;      //Start index of the consecutive 4 coordinates
    int score_index;      //Index of the scores/confidence of boxes
    int id_index;         //Optional, index of the class categories, -1 to disable
    int background_id;    //Optional, id of the background class which will be ignored in nms
    int force_suppress;   //Optional, if set false and id_index is provided, nms will only apply to boxes belongs to the same category
    int in_format;        //0-corner|1-center, default 0: corner means boxes are encoded as [xmin, ymin, xmax, ymax]
    int out_format;       //0-corner|1-center, default 0: center means boxes are encoded as [x, y, width, height]
} cpu_box_nms_param_t;

typedef struct tag_cpu_yolo_param {
    int classes;
    int num;
    tag_cpu_yolo_param() {
        num = 3;
    }
} cpu_yolo_param_t;

typedef enum {
    METHOD_BILINEAR         = 0,    /* bilinear */
    METHOD_NEAREST          = 1,     /* nearest */
    METHOD_BILINEAR_PYTORCH = 2,
    METHOD_NEAREST_PYTORCH  = 3
} RESIZE_METHOD_T;

typedef struct cpu_crop_and_resize {
    RESIZE_METHOD_T method;
    float extrapolation_value;
    int crop_h;
    int crop_w;
} cpu_crop_and_resize_t;

typedef struct cpu_gather {
    int axis;
} cpu_gather_t;

typedef struct cpu_where_squeeze_gather {
    int axes[8];
} cpu_where_squeeze_gather_t;

typedef struct cpu_nms {
    float iou_threshold;
    float score_threshold;
    int max_output_size;
} cpu_nms_t;

typedef struct cpu_argsort_param {
    int axis;
    bool is_ascend;
} cpu_argsort_param_t;

typedef struct cpu_yolov3_detect_out_param {
    int num_inputs_;
    int num_classes_;
    int num_boxes_;

    float confidence_threshold_;
    float nms_threshold_;

    int mask_group_size_;

    float biases_[18];
    float anchors_scale_[3];
    float mask_[9];
} cpu_yolov3_detect_out_param_t;

typedef struct cpu_topk_param {
    cpu_topk_param() {
        k      = -1;
        axis   = -1;
        sorted = true;
        descending = true;
    }
    int  k;
    int  axis;
    bool sorted;
    bool descending;
} cpu_topk_param_t;

#define MX_TOPK_RET_INDICES 0
#define MX_TOPK_RET_VALUE   1
#define MX_TOPK_RET_BOTH    2
#define MX_TOPK_RET_MASK    3
typedef struct {
    int k = 1;
    int axis = -1;
    int ret_type = MX_TOPK_RET_INDICES;
    int is_ascend = 0;
    int dtype = 0; //0: DTYPE_FP32, 6: DTYPE_INT32, 7: DTYPE_UINT32;
} cpu_topk_mx_param_t;

typedef enum {
    BMCPU_NCHW = 0,
    BMCPU_NHWC = 1
} DataFormat;
typedef struct cpu_resize_interpolation_param {
    int align_corners;
    int half_pixel_centers;
    RESIZE_METHOD_T intepolation_method;
    DataFormat ifmt;
    DataFormat ofmt;
    int oh;
    int ow;
} cpu_resize_interpolation_param_t;

typedef struct cpu_sort_per_dim_param {
    int dim;
    bool is_argsort;
    bool stable;
    bool descending;
} cpu_sort_per_dim_param_t;

typedef struct cpu_masked_select_param {
    bool bcast_from_begin;
} cpu_masked_select_param_t;

typedef enum {
    OP_SIN        = 0,    /* sin */
    OP_COS        = 1,    /* cos */
    OP_ISFINITE   = 2,    /* isfinite */
    OP_CEIL       = 3,
    OP_FLOOR      = 4,
    OP_ROUND      = 5,
} UNARY_OP_CODE_T;

typedef struct cpu_unary_param {
    UNARY_OP_CODE_T unary_op;
} cpu_unary_param_t;

typedef enum {
    OP_UNKNOWN   = -1,
    OP_MOD       = 0, // a%b
    OP_DIV       = 1,  // a/b
} BINARY_OP_CODE_T;

typedef struct {
    BINARY_OP_CODE_T op;
    CPU_DATA_TYPE_T dtype;
} cpu_binary_param_t;

typedef struct cpu_embedding_param {
    int* padding_idx;
} cpu_embedding_param_t;

typedef struct cpu_gathernd {
    int indice_is_int = 0;
} cpu_gathernd_t;

typedef struct cpu_index_put_param {
    int mode;
} cpu_index_put_param_t;

typedef struct cpu_debug_param {
    int tensor_id;
    int tensor_dtype;
    int from_layer_id;
    int from_layer_type;
} cpu_debug_param_t;

typedef struct cpu_scatter_nd_param {
    int dim;
    int shape[8];
} cpu_scatter_nd_param_t;

typedef struct cpu_random_uniform_param {
    int dim;
    int shape[8];
    float lower;
    float upper;
    long long seed;
} cpu_random_uniform_param_t;

typedef struct cpu_random_uniform_int_param {
    int dim;
    int shape[8];
    long long seed;
} cpu_random_uniform_int_param_t;

typedef struct cpu_generate_proposals_param {
    float spatial_scale;
    int rpn_pre_nms_topN;
    int rpn_post_nms_topN;
    float rpn_nms_thresh;
    float rpn_min_size;
    bool angle_bound_on;
    int angle_bound_lo;
    int angle_bound_hi;
    float clip_angle_thresh;
    bool legacy_plus_one;
} cpu_generate_proposals_param_t;

typedef struct cpu_bbox_transform_param {
    float weights[4];
    bool apply_scale;
    bool rotated;
    bool angle_bound_on;
    int angle_bound_lo;
    int angle_bound_hi;
    float clip_angle_thresh;
    bool legacy_plus_one;
} cpu_bbox_transform_param_t;

typedef struct cpu_box_with_nms_limit_param {
    float score_thresh;
    float nms;
    int detections_per_im;
    bool soft_nms_enabled;
    int soft_nms_method;
    float soft_nms_sigma;
    float soft_nms_min_score_thres;
    bool rotated;
    bool cls_agnostic_bbox_reg;
    bool input_boxes_include_bg_cls;
    bool output_classes_include_bg_cls;
    bool legacy_plus_one;
} cpu_box_with_nms_limit_param_t;

typedef struct cpu_collect_rpn_proposals_param {
    int rpn_max_level;
    int rpn_min_level;
    int rpn_post_nms_topN;
} cpu_collect_rpn_proposals_param_t;

typedef struct cpu_distribute_fpn_proposals_param {
    int roi_canonical_scale;
    int roi_canonical_level;
    int roi_max_level;
    int roi_min_level;
    bool legacy_plus_one;
} cpu_distribute_fpn_proposals_param_t;

typedef struct cpu_pytorch_roi_align_param {
    int pooled_height;
    int pooled_width;
    float spatial_scale;
    int sampling_ratio;
    bool align;
} cpu_pytorch_roi_align_param_t;

typedef struct cpu_distribute_fpn_proposals_roi_align_concat_param {
    cpu_distribute_fpn_proposals_param_t dfp;
    cpu_pytorch_roi_align_param_t ra[4];
} cpu_distribute_fpn_proposals_roi_align_concat_param_t;

typedef struct cpu_tensorflow_nms_v5_param {
    float iou_threshold;
    float score_threshold;
    float soft_nms_sigma;
    bool pad_to_max_output_size;
    int max_output_size;
} cpu_tensorflow_nms_v5_param_t;

typedef struct cpu_affine_grid_generator_param {
    bool align_corners;
    int N;
    int H;
    int W;
} cpu_affine_grid_generator_param_t;

typedef struct cpu_grid_sampler_param {
    int mode;
    int padding_mode;
    bool align_corners;
} cpu_grid_sampler_param_t;

typedef struct cpu_affine_grid_sampler_param {
    cpu_affine_grid_generator_param_t generator;
    cpu_grid_sampler_param_t sampler;
} cpu_affine_grid_sampler_param_t;

//} /* namespace bmcpu */
#endif /* _CPU_COMMON_H_ */
