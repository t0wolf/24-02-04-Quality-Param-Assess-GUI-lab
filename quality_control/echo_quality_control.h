#ifndef ECHOQUALITYCONTROL_H
#define ECHOQUALITYCONTROL_H
#include <QWidget>
// #include <QMap>
// #include <QString>
#include "EchoQuality.h"
// #include "EchoQualityAssessmentPSAXGV.h"

class EchoQualityControl
{
public:
    EchoQualityControl();
    ~EchoQualityControl();

    int qualityAssess(std::vector<cv::Mat>& vVideoClips, std::vector<int>& keyframeIdxes, std::string& viewName, float fRadius, s_f_map& videoResult);

    inline bool isKeyframeEnable(std::string viewName)
    {
        return m_keyframeMap[viewName];
    }

    inline bool isQualityControlled(singleResult& videoResult)
    {
        return videoResult.totalGrade >= videoResult.totalGrade_th;
    }

private:
    int cropSingleEchoCycle(std::vector<cv::Mat>& vVideoClips, std::vector<cv::Mat>& sampledVideoClips, std::vector<int>& keyframeIdxes);

private:
    std::map<std::string, bool> m_keyframeMap {
        {"A2C", true},
        {"A3C", true},
        {"A4C", true},
        {"A5C", true},
        {"PLAX", true},
        {"PSAXGV", false},
        {"PSAXPM", false},
        {"PSAXMV", false},
        {"PSAXA", false},
    };

};

//class EchoQualityControl
//{
//public:
//    EchoQualityControl();  // 构造函数
//    ~EchoQualityControl(); // 析构函数
//
//    int doFrameInfer(cv::Mat& src, std::string& viewName);  // 我把拉流来的一帧图像传给你，并且给你切面的名称供你调用对应切面的接口
//
//    bool getQualityControlResult(std::string& viewName);  // 返回给定切面的质控结果
//
//private:
//    int initialize();  // 初始化各切面的TensorRT Engine
//
//    std::map<std::string, s_f_map> m_eachViewQualityScores;  // 每个切面的每项质控分数
//};

#endif // ECHOQUALITYCONTROL_H
