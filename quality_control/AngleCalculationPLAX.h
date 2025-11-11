
#include "cplusfuncs.h"
#include "opencvfuncs.h"
#include "typemappings.h"
#include "iqaparams.h"
#include "angleinfer_plax.h"


/// <summary>
/// 以分类直接判断角度是否正确，定性计算角度为代表的技术路线
/// </summary>
class AngleCalculationPLAX
{
public:
    AngleCalculationPLAX(AngleCalculationParams params);

    virtual ~AngleCalculationPLAX();

    virtual float predict(std::vector<cv::Mat> frames,
        std::vector<int> keyframes,
        f_structures frame_structures); // 预测整个视频

private:

    void load_model();                               // 载入关键帧检测模型

    AngleInferPLAX* model;                           // 关键帧检测模型

    // 先验知识涉及参数
    std::string viewname;                            // 切面类型名称
    float class_threshold = 0.5f;                    // 分类阈值
    std::vector<std::string> structures;             // 相应切面所包含的解剖结构
    std::vector<std::string> classnames;             // 目标检测的解剖结构表
    s_i_map structure_idx_mapping;                   // 各解剖结构名称对应的目标检测结果索引
    std::string lvlvot_name = "LVLVOT";

    // 模型预测时部分参数
    std::string model_path;                          // 关键帧检测模型路径
    std::string model_dir;                           // 关键帧检测模型文件夹
    std::string model_name;                          // 关键帧检测模型文件名
    std::vector<float> mMean;                        // 归一化均值
    cv::Size inputSize;                              // 输入图像尺寸（由于TensorRT特性，更改后模型并不会变化）
    cv::Size mInputSize;
    cv::Size mOutputSize;
};