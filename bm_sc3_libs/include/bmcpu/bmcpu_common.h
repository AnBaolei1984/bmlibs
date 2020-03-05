#ifndef _CPU_COMMON_H_
#define _CPU_COMMON_H_

/*
 * NOTE:
 * To guarantee the compatibility with all previous bmodel
 * We must follow the rules:
 *   1. Positions and numbers in CPU_LAYER_TYPE_T cannot be modified!
 *   2. New element for CPU_LAYER_TYPE_T must be added to the last!
 *   3. Each cpu layer param must be defined carefully and cannot
 *      be revised after release.
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
    CPU_LAYER_NUM,
    CPU_LAYER_UNKNOW = CPU_LAYER_NUM
} CPU_LAYER_TYPE_T;

typedef struct cpu_exp_param
{
    float inner_scale_;
    float outer_scale_;
} cpu_exp_param_t;

typedef struct cpu_relu_param
{
    float negative_slope_;
} cpu_relu_param_t;

typedef struct cpu_ssd_detect_out_param
{
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

typedef struct cpu_rpnproposal_param
{
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

typedef struct cpu_roi_pooling_param
{
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

typedef enum{
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

typedef struct tag_cpu_yolo_param
{
    int classes;
    int num;
    tag_cpu_yolo_param() {
      num = 3;
    }
} cpu_yolo_param_t;

typedef enum {
    METHOD_BILINEAR     = 0,    /* bilinear */
    METHOD_NEAREST      = 1     /* nearest */
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
} cpu_nms_t;

typedef struct cpu_argsort_param {
    int axis;
    bool is_ascend;
}cpu_argsort_param_t;

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
    }
    int  k;
    int  axis;
    bool sorted;
}cpu_topk_param_t;

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

typedef struct cpu_resize_interpolation_param {
    int align_corners;
    int half_pixel_centers;
    RESIZE_METHOD_T intepolation_method;      /* 1->caffe, 0->tensorflow */
} cpu_resize_interpolation_param_t;

typedef struct cpu_sort_per_dim_param
{
    int dim;
    bool is_argsort;
    bool stable;
    bool descending;
} cpu_sort_per_dim_param_t;

typedef struct cpu_masked_select_param
{
    bool bcast_from_begin;
} cpu_masked_select_param_t;

typedef enum {
    OP_SIN        = 0,    /* sin */
    OP_COS        = 1,    /* cos */
    OP_ISFINITE   = 2,    /* isfinite */
} UNARY_OP_CODE_T;
typedef struct cpu_unary_param
{
    UNARY_OP_CODE_T unary_op;
} cpu_unary_param_t;

typedef struct cpu_embedding_param
{
    int* padding_idx;
} cpu_embedding_param_t;

typedef struct cpu_gathernd {
    int indice_is_int = 0;
} cpu_gathernd_t;

//} /* namespace bmcpu */
#endif /* _CPU_COMMON_H_ */
