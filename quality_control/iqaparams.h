////
//   Created by SEU-BME-chl
//   图像质量评估所用的参数
//   所有评估类共用，严禁擅自修改内容部分！
//   如有修改需求，请先找成大冤种确认！！！
////
#pragma once
#include "typemappings.h"



//////////////////////////////////////// ROI提取使用的参数 ////////////////////////////////////////
struct RoIExtractParams
{
    // 先验知识涉及参数
    std::vector<std::string> viewnames;                            // 切面类型名称

    std::string mask_dir = "../../extern/roi_extraction/";         // ROI Mask图像文件夹
    std::string mask_name = "original_mask";                       // ROI Mask图像文件名
};
//////////////////////////////////////////////////////////////////////////////////////////////////




/////////////////////////////////////// 视图分类使用的参数 ///////////////////////////////////////
struct ViewClassificationParams
{
    // 先验知识涉及参数
    std::vector<std::string> viewnames;                            // 切面类型名称

    // 模型预测时部分参数
    std::string device;                                            // 推理所用设备（实际环境默认GPU0推理）
    bool use_gpu = true;                                           // 是否使用GPU进行推理
    std::string model_dir = "../../extern/view_classification/";   // 切面分类模型文件夹
    std::string model_name;                                        // 切面分类模型文件名
    const int frame_num = 64;                        // 模型输入图像序列长度（由于TensorRT特性，更改后模型并不会变化）
    const int imgSize = 256;                         // 原始输入图像尺寸（后续会Crop为输入视频）
    const int inputSize = 224;                       // 模型输入图像尺寸（由于TensorRT特性，更改后模型并不会变化）
    int frameSampleRate = 1;                         // 对原始帧的采样率（SlowFast）
    int inferenceTime = 9;                           // 对单个视频默认的推理次数
    //std::vector<float> mMean = { 114.7748f, 107.7354f, 99.475f };  // 归一化均值
    // frame = (frame - np.array([[[128.0, 128.0, 128.0]]]))/128.0
    std::vector<float> mMean = { 128.0f, 128.0f, 128.0f };  // 归一化均值

};
//////////////////////////////////////////////////////////////////////////////////////////////////




/////////////////////////////////////// 结构检测使用的参数 ///////////////////////////////////////
struct StructureDetectionParams
{
    // 先验知识涉及参数
    std::string viewname;                                          // 切面类型名称
    std::vector<std::string> structures;                           // 相应切面所包含的解剖结构
    std::vector<std::string> classnames;                           // 目标检测的解剖结构表

    // 模型预测时部分参数
    std::string device;                                            // 推理所用设备（实际环境默认GPU0推理）
    bool use_gpu = true;                                           // 是否使用GPU进行推理
    std::string model_dir = "../../extern/structure_detection/";   // 切面分类模型文件夹
    std::string model_name;                                        // 结构结构检测模型文件名
    std::vector<float> score_threshold;                            // 各解剖结构的score阈值（后续处理为string:float pair）
    std::vector<int> frame_threshold;                              // 检测到解剖结构的帧数阈值（后续处理为string:int pair）
    const int inputSize = 300;                                     // 模型输入图像尺寸（由于TensorRT特性，更改后模型并不会变化）
    std::vector<float> mMean = { 114.7748f, 107.7354f, 99.475f };  // 归一化均值
};
//////////////////////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////// 增益分类使用的参数 ///////////////////////////////////////
////////////// 20230507更新信息：由于弃用局部直方图+随机森林的增益分类方法，重写GainClassification，该部分弃用
//struct GainParams
//{
//    // 先验知识涉及参数
//    std::string viewname;                                          // 切面类型名称
//    s_s_map viewname_mapping;                                      // 切面类型名称符号与切面类型的字符映射
//    std::vector<std::string> view_structures;                      // 相应切面所包含的解剖结构
//    std::vector<std::string> classnames;                           // 目标检测的解剖结构表
//
//    // 模型预测时部分参数
//    std::vector<std::string> gain_structures;                      // 图像增益判断用到的解剖结构（使用局部直方图）
//    std::string models_dir = "../../extern/gain_classification/";  // 图像增益分类子模型根目录（实际路径models_dir+viewname）
//    std::vector<float> gain_threshold;                             // ML模型预测划分阈值（0：欠佳）（后续处理为string:float pair）
//    std::vector<float> frame_threshold;                            // 图像增益欠佳帧数百分比阈值（后续处理为string:float pair）
//};
////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////// 增益分类使用的参数 ///////////////////////////////////////
////////////// 20230507更新信息：由于启用DL增益分类方法，重写GainClassification，该部分启用
////////////// 20231110更新信息：由于合并dark/bright模型，重写GainClassification，该部分弃用
//struct GainParams
//{
//    // 先验知识涉及参数
//    std::string viewname;                                          // 切面类型名称
//    s_s_map viewname_mapping;                                      // 切面类型名称符号与切面类型的字符映射
//    float frame_mean_th = 40.0f;                                   // 视频均值阈值
//    float brightframe_rate_th = 0.6f;                              // 较亮视频帧占比
//
//    // 模型预测时部分参数
//    std::string models_dir = "../../extern/gain_classification/";  // 图像增益分类子模型根目录（实际路径models_dir+viewname+后缀）
//    std::vector<float> gain_threshold;                             // 模型预测划分阈值（0：欠佳）（dark/bright模型）
//    std::vector<float> frame_threshold = { 0.495f, 0.495f };       // 图像增益欠佳帧数百分比阈值（dark/bright模型）
//    const int inputSize = 112;                                      // 模型输入图像尺寸（由于TensorRT特性，更改后模型并不会变化）
//    std::vector<float> mMean = { 0.485f, 0.456f, 0.406f };         // 归一化均值
//    std::vector<float> mStd = { 0.229f, 0.224f, 0.225f };          // 归一化标准差
//};
////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////// 增益分类使用的参数 ///////////////////////////////////////
//////////// 20231110更新信息：由于合并dark/bright模型，重写GainClassification，该部分启用
struct GainParams
{
    // 先验知识涉及参数
    std::string viewname;                                          // 切面类型名称
    s_s_map viewname_mapping;                                      // 切面类型名称符号与切面类型的字符映射

    // 模型预测时部分参数
    std::string models_dir = "../../extern/gain_classification/";  // 图像增益分类子模型根目录（实际路径models_dir+viewname+后缀）
    float gain_threshold;                                          // 模型预测划分阈值（0：欠佳）
    float frame_threshold;                                         // 图像增益欠佳帧数百分比阈值
    const int inputSize = 112;                                     // 模型输入图像尺寸（由于TensorRT特性，更改后仅输入尺寸改变，模型并不会变化）
    std::vector<float> mMean = { 0.485f, 0.456f, 0.406f };         // 归一化均值
    std::vector<float> mStd = { 0.229f, 0.224f, 0.225f };          // 归一化标准差
};
//////////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////// 关键帧检测使用的参数 //////////////////////////////////////
struct KeyframeDetectionParams
{
    // 先验知识涉及参数
    std::string viewname;                                          // 切面类型名称

    // 模型预测时部分参数
    std::string device;                                            // 推理所用设备（实际环境默认GPU0推理）
    bool use_gpu = true;                                           // 是否使用GPU进行推理
    std::string model_dir = "../../extern/keyframe_detection/";    // 关键帧检测模型文件夹
    std::string model_name;                                        // 关键帧检测模型文件名
    std::vector<float> mMean = { 114.7748f, 107.7354f, 99.475f };  // 归一化均值
    const int stride = 1;                                          // 预测滑动步长
    const int frame_num = 16;                                      // 模型输入图像序列长度（由于TensorRT特性，更改后模型并不会变化）
    const int inputSize = 334;                                     // 模型输入图像尺寸（由于TensorRT特性，更改后模型并不会变化）
    const int inputSize_plax = 300;                                // PLAX切面模型输入图像尺寸
};
//////////////////////////////////////////////////////////////////////////////////////////////////




///////////////////////////////////// 心轴角度计算使用的参数 /////////////////////////////////////
// 心轴角度计算使用的参数
struct AngleCalculationParams
{
    // 先验知识涉及参数
    std::string viewname;                                          // 切面类型名称
    float angle_threshold;                                         // 心轴角度定量阈值（定量分析时，使用具体角度值，如A4C）
    float class_threshold;                                         // 心轴角度定性阈值（定性分析时，使用回归分数，如PLAX）
    std::vector<std::string> structures;                           // 相应切面所包含的解剖结构
    std::vector<std::string> classnames;                           // 目标检测的解剖结构表
    s_i_map structure_idx_mapping;                                 // 各解剖结构名称对应的目标检测结果索引
    std::vector<std::string> ac_structures;                        // 心轴角度计算用到的解剖结构

    // 模型预测时部分参数
    std::string device;                                            // 推理所用设备（实际环境默认GPU0推理）
    bool use_gpu = true;                                           // 是否使用GPU进行推理
    std::string model_dir = "../../extern/angle_calculation/";     // 心轴角度计算模型文件夹
    std::string model_name;                                        // 分割/分类 模型文件名
    //const int inputSize = 256;                       // 模型输入图像尺寸（由于TensorRT特性，更改后模型并不会变化）
    ////// update 2023.12.28
    int inputSize = 256;                       // 由于各切面Unet模型的输入尺寸不一致，在模型训练前临时用一下
    std::vector<float> mMean = { 114.7748f, 107.7354f, 99.475f };  // 归一化均值
};
//////////////////////////////////////////////////////////////////////////////////////////////////




//////////////////////////////////////// 质量评估整体参数 ////////////////////////////////////////
struct EchoQualityAssessmentParams
{
    // 先验知识涉及参数
    std::string viewname;                                          // 切面类型名称
    std::vector<std::string> structures;                           // 相应切面所包含的解剖结构
    std::string scaling_structure = "HEART";                       // 切面缩放比例基准解剖结构
    std::vector<float> scaling_range;                              // 相应切面缩放比例的合理范围

    // 检测类参数
    StructureDetectionParams sdetection_params;                    // 关键帧检测相关参数
    GainParams gain_params;                                        // 图像增益判断相关参数
    KeyframeDetectionParams kfdetection_params;                    // 关键帧检测相关参数
    AngleCalculationParams ac_params;                              // 心轴角度相关参数
    std::vector<float> mMean = { 114.7748f, 107.7354f, 99.475f };  // 归一化均值
};
//////////////////////////////////////////////////////////////////////////////////////////////////
