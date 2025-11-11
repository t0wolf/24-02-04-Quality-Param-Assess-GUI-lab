#pragma once
#include <iostream>
#include <QString>
#include <QMap>
//#include <filesystem>
#include "segment_infer_base.h"
#include "ivs_and_pw_assess.h"
#include "aao_segment_inferer.h"
#include "aao_assess.h"
#include "aad_assess.h"
#include "asd_and_sjd_access.h"
#include "avad_assess.h"
#include "avad_keypoints_inferer.h"
#include "laad_assess.h"
#include "general_utils.h"
#include "multi_struct_inferer.h"
#include "ed_es_line_assess.h"
#include "ed_es_line_aorta_assess.h"
#include "QtLogger.h"
#include "config_parse.h"
#include "PLAXPoseClsInferer.h"
#include "PLAXAorticDetect.h"

class PLAXParamsAssess
{
public:
    PLAXParamsAssess();

	PLAXParamsAssess(ConfigParse* config);

    ~PLAXParamsAssess();

    int getStrucParamsRst(std::vector<cv::Mat>& video, std::vector<int>& keyframeIdx, QMap<QString, QVector<float>>& values, QMap<QString, QImage>& resultPic);

private:
	int getSampledVector(const std::vector<cv::Mat>& inputVector, std::vector<cv::Mat>& outputVector, size_t length);

	int getKeyframes(std::vector<int>& keyframeIdx, std::vector<cv::Mat>& inputVideo, std::vector<cv::Mat>& outputVideo, int length);

	int parseKeyframes(std::vector<int>& keyframeIdx, std::vector<cv::Mat>& inputVideo, std::vector<cv::Mat>& outputVideo, int length);

	cv::Mat concatMultiImages(std::vector<cv::Mat>& vecImages);

	int adjustAADandAoD(int nPLAXPosCls, std::map<std::string, std::vector<float>>& mapValues, std::map<std::string, cv::Mat>& mapPremiums);

	int aorticQC(cv::Mat& src, std::map<std::string, std::vector<float>>& mapValues, std::map<std::string, cv::Mat>& mapPremiums);

	cv::Mat m_inferFrame;
	int m_signal = 0;

	PLAXPoseClsInferer* m_pPLAXPosClsInferer;

	IVSAndPWAssess* m_pIvsandpwAssesser;

	ASDAndSJDAssess* m_pAsdAndsjdAssesser;

	AADAssess* m_pAadAssesser;

	LAADAssess* m_pLaadAssesser;

	MultiStructInferer* m_multiStructInferer;

	PLAXAorticDetect* m_pPLAXAorticQCDetect;

public:
	MultiLineAssess* m_pMultiLineAssesser;
	MultiLineAssess_Aorta* m_pMultiLineAortaAssesser;
	AVADAssess* m_pAvadAssesser;
	//AVADAssess* m_avadKeyptInferer;
};
