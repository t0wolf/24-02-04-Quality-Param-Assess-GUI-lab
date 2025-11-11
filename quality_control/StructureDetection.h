#pragma once
#include "cplusfuncs.h"
#include "opencvfuncs.h"
#include "typemappings.h"
#include "iqaparams.h"
#include "structureinfer.h"



class StructureDetection
{
public:
    StructureDetection(StructureDetectionParams params);

    virtual ~StructureDetection();

    void predict(std::vector<cv::Mat> frames,
        s_frames& structure_frames,
        f_structures& frame_structures); // 预测整个视频

private:

    void load_model();                               // 载入目标检测模型

    StructureInfer* model;                           // 目标检测模型

    // 先验知识涉及参数
    std::string viewname;                            // 切面类型名称
    std::vector<std::string> structures;             // 相应切面所包含的解剖结构
    std::vector<std::string> classnames;             // 目标检测的解剖结构表
    i_s_map idx_structure_mapping;                   // 目标检测结果索引对应的各解剖结构名称

    // 模型预测时部分参数
    std::string model_name;                          // 目标检测模型文件名
    std::string model_path;                          // 目标检测模型路径
    s_f_map score_threshold;                         // 各解剖结构的score阈值
    s_i_map frame_threshold;                         // 检测到解剖结构的帧数阈值
    cv::Size inputSize;                              // 模型输入图像尺寸（由于TensorRT特性，更改后模型并不会变化）
    std::vector<float> mMean;                        // 归一化均值
};

