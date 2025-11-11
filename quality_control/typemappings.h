////
//   Created by SEU-BME-chl
//   图像质量评估所用的数据结构
//   所有评估类共用，严禁擅自修改内容部分！
//  （finalResult结构体，各切面grades的解释性注释除外）
//   如有修改需求，请先找成大冤种确认！！！
////
#pragma once
#include <cstdlib>
//#include "opencv.hpp"
#include "opencv2/opencv.hpp"

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;

//////////////////////////////////////// lib和dll库文件最终返回结构体 ////////////////////////////////////////
// 单个视频段预测结果
struct singleResult {
    std::string label;                        // 预测的切面类型结果
    float totalGrade;                         // 总体得分情况，因部分切面子评分项存在0.5分，故最终使用float类型
    float totalGrade_th;                      // 标准切面分数阈值
    std::vector<float> grades;                // 各项具体得分
};

// 视频中不同视频段可能对应了多个切面，因此以视频段为单位描述结果
struct finalResult {
    std::vector<std::string> dicomInfos;      // 服务器需要的DICOM文件相关信息
    std::vector<singleResult> singleResults;  // 各视频段相应预测结果
};

//// A4C切面各项具体得分grades中包含的分数依次为：
// 1）增益；
// 2）缩放比例；
// 3）心轴角度；
// 常规必须项结构：4）左房、左室、右房、右室、房间隔及室间隔；
// 非常规项结构：5）二尖瓣前后叶显示清晰； 6）三尖瓣前瓣及隔瓣/后瓣显示清晰。

//// PSAXGV切面各项具体得分grades中包含的分数依次为：
// 1）增益；
// 2）缩放比例；
// 常规必须项结构：3）左房、右室流出道、主动脉短轴；
// 常规可选项结构：4）主动脉瓣；5）主肺动脉； 6）右房。

//// PSAXPM切面各项具体得分grades中包含的分数依次为：
// 1）增益；
// 2）缩放比例；
// 常规必须项结构：3）右室、室间隔、乳头肌、左室下后壁；
// 常规可选项结构：4）左室前壁；5）左室侧壁；
// 非常规项结构：6）两组乳头肌显示清晰完整。

//////////////////////////////////////////////////////////////////////////////////////////////////////////////







//////////////////////////////////////// 解剖结构项得分计算所用结构体 ////////////////////////////////////////
// 常规必须项：集合内元素均需被检测到，缺一不可的解剖结构
// 常规可选项：检测到即得分的解剖结构
// 非常规项：  需要单独处理的解剖结构，如A4C“二尖瓣显示清晰”除各帧检测结果，还需要结合心动周期

// structure_score_dict 三类解剖结构的初步得分
struct s_score_dict
{
    float routine_must = 0.0;                          // 常规必须项分数
    std::vector<float> routine_optional;               // 常规可选项分数
    std::vector<float> unroutine_optional;             // 非常规项分数
};

// structure_score_weight 三类解剖结构的分数权重
struct s_score_weight
{
    float routine_must;                                // 常规必须项权重
    std::vector<float> routine_optional;               // 常规可选项权重
    std::vector<float> unroutine_optional;             // 非常规可选项权重
};

// structure_score_structure 三类解剖结构的计算得分时，分别涉及的解剖结构
struct s_score_structure
{
    std::vector<std::string> routine_must;             // 常规必须项结构
    std::vector<std::string> routine_optional;         // 常规可选项结构
    std::vector<std::string> unroutine_optional;       // 非常规项结构
    std::vector<float> unroutine_optional_params;      // 非常规项参数（后续可能会进一步修改）
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////




//////////////////////////////////////////// DICOM解析所用结构体 ////////////////////////////////////////////
struct dicom_res
{
    int re_flag = 0;                     // 解析DICOM文件是否成功的flag
    std::vector<std::string> dicomInfos; // 服务器需要的DICOM文件相关信息
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////// 切面类型分类所用结构体 ////////////////////////////////////////////
// 某类型切面所对应的各视频段起始帧索引
typedef std::map <int, std::vector<int>> clip_fidxs;

// int到std::vector<int>的pair
typedef std::pair<int, std::vector<int>> int2ints;
// string到std::vector<int>的pair
typedef std::pair<std::string, std::vector<int>> str2ints;
// 根据可能性依次排序后的切面类型（int型类别索引）
typedef std::vector<int2ints> view_fidxs;
// 根据可能性依次排序后的切面类型（字符型类别索引）
typedef std::vector<str2ints> viewname_fidxs;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////// 目标检测所用结构体 ////////////////////////////////////////////
// 目标检测所用检测框类型
struct object_box
{
    cv::Rect rect;   // 检测框
    int label;       // 所属类别
    float prob;      // 置信度
};

// 根据检测框进行后处理所需数据结构（单帧的检测框vector）
struct frame_object_boxes
{
    std::vector<cv::Rect> rects;
    std::vector<int> labels;
    std::vector<float> probs;
};

// 整个视频，每一类解剖结构，被检测到的帧
typedef std::map<std::string, std::vector<int>> s_frames;  //structure_frames

// 整个视频，每一帧，检测到的解剖结构
typedef std::map<int, frame_object_boxes> f_structures;  //frame_structures
/////////////////////////////////////////////////////////////////////////////////////////////////////////////




////////////////////////////////////////// std::map近似python dict //////////////////////////////////////////
// 参数传递使用dict/list，但由于需要基于v140平台工具集，基于C++17语法的std::variant、std::visit无法使用，
// 因此折中使用 C++11 std::map结合枚举的方式实现
//  s:std::string;    i:int;    f:float;    d:double;    m:cv::Mat

// 类似python dict，字符与其他数据类型的pair
typedef std::map <std::string, std::string> s_s_map;
typedef std::map <std::string, int> s_i_map;
typedef std::map <std::string, float> s_f_map;
typedef std::map <std::string, double> s_d_map;
typedef std::map <std::string, cv::Mat> s_m_map;

// 类似python dict，int数字与其他数据类型的pair
typedef std::map <int, std::string> i_s_map;
typedef std::map <int, int> i_i_map;
typedef std::map <int, float> i_f_map;
typedef std::map <int, double> i_d_map;
typedef std::map <int, cv::Mat> i_m_map;
