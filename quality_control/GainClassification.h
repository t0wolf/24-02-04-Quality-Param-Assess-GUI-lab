/*
   Created by SEU-BME-LBMD-chl, SEU-BME-LBMD-zzy, SEU-BME-LBMD-scj
*/
#pragma once
//#include "opencv2/ml/ml.hpp"
#include "cplusfuncs.h"
#include "opencvfuncs.h"
#include "typemappings.h"
#include "iqaparams.h"
#include "gaininfer.h"



class GainClassification
{
public:
    GainClassification(GainParams gainparams);

    virtual ~GainClassification();

    virtual float predict(
        std::vector<cv::Mat> frames);

private:

    void load_model();                                 // 载入增益分类模型

    
    /////////////////////////// 2023.11.10：因dark/bright模型被弃用，产生的改动
    //int decide_model(
    //    std::vector<cv::Mat> frames, 
    //    cv::Mat roi_mask);     // 决定使用dark还是bright模型

    // GainInfer* model_d;                                // 增益分类模型-较暗视频（dark）
    // GainInfer* model_b;                                // 增益分类模型-较亮视频（bright）

    // float mMeanThreshold = 40.0f;                      // 视频均值阈值（划分属于dark还是bright帧）
    // float mBrightRateThreshold = 0.6f;                 // 较亮视频帧占比
    // std::string modelname_s[2] = {"_dark","_bright"};  // 图像增益分类子模型文件名后缀（dark:neg，bright:pos）
    ////////////////////////////////////////////////////////////////////////////////////////////////////////


    GainInfer* model;                                  //全部视频
    // 先验知识涉及参数
    std::string viewname;                              // 切面类型名称
    s_s_map viewname_mapping;                          // 切面类型名称符号与切面类型的字符映射
    std::vector<std::string> name_and_mapname[2] = {
        {"A4C","a4c","PSAX_GV","PSAXGV"},
        {"a4c","a4c","psaxgv","psaxgv"}
    };                                                 // 切面类型名称符号别称


    // 模型预测时部分参数
    std::vector <cv::Mat> video_data;                  // opencv bgr格式视频数据
    std::string models_dir;                            // 图像增益分类子模型路径
    cv::Size inputSize;                                // 输入图像尺寸（由于TensorRT特性，更改后模型并不会变化）
    float gain_threshold;                              // 模型预测划分阈值（0：欠佳）
    float frame_threshold;                             // 图像增益欠佳帧数百分比阈值
    std::vector<float> mMean = { 0.485f, 0.456f, 0.406f }; // 归一化均值
    std::vector<float> mStd = { 0.229f, 0.224f, 0.225f };  // 归一化标准差
};