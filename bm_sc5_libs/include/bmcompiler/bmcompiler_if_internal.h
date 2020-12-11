#ifndef BMCOMPILER_IF_INTERNAL_H_
#define BMCOMPILER_IF_INTERNAL_H_

#ifdef __cplusplus
extern "C" {
#endif

//internal bmcompiler interface for C++
void __compile_with_result_check(void* p_bmcpl, void* input_data, void* layer_top_references, char* net_name);
void __compile_ir_with_result_check(void* p_bmcpl, void* input_data, void* layer_top_references, char* net_name);

void __compile_with_result_check_opt(void* p_bmcpl, void* input_data, void* layer_top_references, char* net_name, int opt_level);
void __compile_ir_with_result_check_opt(void* p_bmcpl, void* input_data, void* layer_top_references, char* net_name, int opt_level);

//internal bmcompiler interface for C
//void compile_with_result_check(void* p_bmcpl, char** input_name, float** input_data, int input_num, //wxc 20181128 move to bmcompiler_if.h
//        char** refer_name, float** refer_data, int refer_num, char* net_name);
void set_opt_level(int opt_level);
void compile_ir_with_result_check(void* p_bmcpl, char** input_name, float** input_data, int input_num,
        char** refer_name, float** refer_data, int refer_num, char* net_name);

#ifdef __cplusplus
}
#endif

#endif
