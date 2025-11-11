#pragma once
#include "cplusfuncs.h"
#include "opencvfuncs.h"
#include "quality_utils.h"
#include "StructureDetection.h"
#include "GainClassification.h"
#include "KeyframeDetection.h"
#include "AngleCalculationPLAX.h"



class EchoQualityAssessmentPLAX
{
public:

    EchoQualityAssessmentPLAX(EchoQualityAssessmentParams params);

    virtual ~EchoQualityAssessmentPLAX();

    virtual s_f_map predict(std::vector<cv::Mat> frames,std::vector<float> radius, singleResult& results);

    bool is_standard = false;                                   // 是否为标准切面
    float total_score_th = 8.0f;                                // 图像各子项目总分标准切面判定阈值
    float total_score = 0.0f;                                   // 图像各子项目总分
    s_f_map score_weighted;                                     // 各子项目各项权重
    float structure_score;                                      // 解剖结构总分
    s_score_dict structure_scores;                              // 解剖结构项各类型结构得分（必须得分项合并为1部分+可选得分项2部分）
    float scaling_score = 0.0f;                                 // 缩放比例得分
    float gain_score = 0.0f;                                    // 图像增益得分

protected:

    void load_structure_model();                                // 解剖结构检测模型载入
    void load_gain_model();                                     // 图像增益模型载入



    // 解剖结构得分
    float structure_judgment(s_frames structure_frames,
        int frame_nums,
        std::vector<float>& grades);

    // 判断视频图像缩放比例项的得分，需要获取心脏的检测框参数，以及ROI参数
    float scaling_judgment(s_frames structure_frames,
        f_structures frame_structures,
        std::vector<float> radius);

    //// 图像增益得分
    //////////////// 20230507更新信息：由于弃用局部直方图+随机森林的增益分类方法，重写GainClassification，该部分弃用
    //float gain_judgment(s_frames structure_frames,
    //    f_structures frame_structures,
    //    std::vector<cv::Mat> frames,
    //    cv::Mat roi_mask);
    ////////////// 20230507更新信息：由于启用DL增益分类方法，重写GainClassification，该部分启用
    float gain_judgment(
        std::vector<cv::Mat> frames);



    StructureDetection* structure_detection;                    // 结构检测类
    GainClassification* gain_classification;                    // 图像增益类
    KeyframeDetection* keyframe_detection;                      // 关键帧检测类
    AngleCalculationPLAX* angleJudgement_plax;                  // 心轴角度计算类

    // 先验知识涉及参数
    std::string viewname = "PLAX";                              // 切面类型名称
    std::vector<std::string> classnames;                        // 目标检测的解剖结构表
    //std::vector<std::string> structures = {"AV",
    //     "HEART","IVS","LA","LVLVOT","LVPW", "MV","RV"};                  // 相应切面所包含的解剖结构（建议标注时"HEART"放第一个）
    // 2023.03.19，根据新一轮训练时，各结构的排序进行的更新操作
    std::vector<std::string> structures = {
     "HEART","AV","IVS","LA","LVLVOT","MV","RV","LVPW" };                  // 相应切面所包含的解剖结构（建议标注时"HEART"放第一个）
    s_i_map structure_idx_mapping;                              // 各解剖结构名称对应的目标检测结果索引

    // 解剖结构
    EchoQualityAssessmentParams* mParams;
    s_score_structure structure_score_structures;               // 解剖结构项各类型结构
    s_score_weight structure_score_weight;                      // 解剖结构项各项权重

    // 缩放比例
    std::string scaling_structure = "HEART";                    // 切面缩放比例基准解剖结构
    std::vector<float> scaling_range = { 2.0f / 3.0f, 4.0f / 5.0f };  // 相应切面缩放比例的合理范围

    std::vector<int> frame_threshold;
    std::vector<float> score_threshold;

    std::vector<int> keyframes;                                 // 关键帧
    s_frames structure_frames;                                  // 整个视频，每一类解剖结构，被检测到的帧
    f_structures frame_structures;                              // 整个视频，每一帧，检测到的解剖结构
    std::vector<float> mMean = { 114.7748f, 107.7354f, 99.475f }; // 归一化均值
};
