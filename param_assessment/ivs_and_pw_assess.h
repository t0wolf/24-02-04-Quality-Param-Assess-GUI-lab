#pragma once
#include <QMap>
#include <QString>
#include "segment_infer_base.h"
#include "image_process.h"
#include "assess_utils.h"

using namespace ImageProcess;

//IVS Interventricular Septum 室间隔 
class IVSAndPWAssess
{
public:
	IVSAndPWAssess();

	int doSegInference(cv::Mat& src, cv::Mat& mask);

	int doSegInference(cv::Mat& src, std::vector<cv::Mat>& vMasks);

	int ivsAndPwAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);
	
private:

	int postProcess(cv::Mat& src, cv::Mat& mask);

	int postProcess(cv::Mat& src, std::vector<cv::Mat>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	float getPWStructureThickness(std::vector<std::vector<cv::Point>>& interPoints, std::vector<cv::Point>& resultLine);

	float getIVSStructureThickness(std::vector<std::vector<cv::Point>>& interPoints, std::vector<cv::Point>& resultLine);

	float getLVIDStructureThickness(std::vector<std::vector<cv::Point>>& ivsInterPoints, std::vector<std::vector<cv::Point>>& pwInterPoints, std::vector<cv::Point>& resultLine);

	inline std::vector<cv::Point> getLVIDPoint(std::vector<cv::Point>& ivsPoint, std::vector<cv::Point>& pwPoint)
	{
		cv::Point ivsBottomPoint = ivsPoint[0].y > ivsPoint[1].y ? ivsPoint[0] : ivsPoint[1];
		cv::Point pwUpPoint = pwPoint[0].y < pwPoint[1].y ? pwPoint[0] : pwPoint[1];

		std::vector<cv::Point> vLVIDPoint{ ivsBottomPoint, pwUpPoint };
		return vLVIDPoint;
	}

private:
	std::string m_segEnginePath;

	SegmentInferBase m_segInferer;

};

