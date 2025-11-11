/*
   Created by SEU-BME-LBMD-chl, SEU-BME-LBMD-zzy, SEU-BME-LBMD-scj
*/
#pragma once
#include "cplusfuncs.h"
#include "opencvfuncs.h"
#include "typemappings.h"
#include "iqaparams.h"
#include "keyframeinfer.h"



class KeyframeDetection
{
public:
    KeyframeDetection(KeyframeDetectionParams params);

    virtual ~KeyframeDetection();

    std::vector<int> predict(std::vector<cv::Mat> frames); // 预测整个视频

private:

    void load_model();                               // 载入关键帧检测模型

    KeyframeInfer* model;                            // 关键帧检测模型

    // 先验知识涉及参数
    std::string viewname;                            // 切面类型名称

    // 模型预测时部分参数
    std::string model_path;                          // 关键帧检测模型路径
    std::string model_dir;                           // 关键帧检测模型文件夹
    std::string model_name;                          // 关键帧检测模型文件名
    std::vector<float> mMean;                        // 归一化均值
    cv::Size inputSize;                              // 输入图像尺寸（由于TensorRT特性，更改后模型并不会变化）
    int frame_num = 16;                              // 模型输入图像序列长度（由于TensorRT特性，更改后模型并不会变化）
    int stride = 1;                                  // 预测滑动步长
};

