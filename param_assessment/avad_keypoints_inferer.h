#pragma once

#include "segment_infer_base.h"
#include "assess_utils.h"
#include "image_process.h"

class AVADKeypointsInferer :public SegmentInferBase
{
public:
	AVADKeypointsInferer(std::string& sEnginePath);

	~AVADKeypointsInferer();

	int setScaleInfo(float& scaleLength, float& scale);

	int doInference(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics, float m_scale);

	int preprocess(cv::Mat& src, float* blob);

	int blobFromImage(cv::Mat& src, float* blob);

private:
	int doSingleInfer(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics, float m_scale);

	int postProcess(float* output, cv::Mat src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics, float m_scale);
	int parseFinalResults(std::map<std::string, std::vector<float>>& values, std::map<std::string, std::vector<float>>& realValues, std::map<std::string, cv::Mat>& resultPics);
	float m_scaleLength;
	float m_scale;
	std::map<std::string, std::pair<float, float>> m_referRange = {
	{ "AoD", {1.5f, 6.5f} }
	};
	cv::Mat m_invertTrans;
};
