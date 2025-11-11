#pragma once
#include "ao_vti_segment_inferer.h"
#include "image_process.h"
#include "assess_utils.h"

// The flow velocity of the aortic valve
class FVAVAssess
{
public:
	FVAVAssess(std::string& sEngineFilePath);

	~FVAVAssess();

	int doSegInference(cv::Mat& src, std::vector<cv::Mat>& vMasks);

	int fvavAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

private:
	int postProcess(cv::Mat& src, std::vector<cv::Mat>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

private:
	std::string m_segEnginePath;

	AoVTISegmentInferer* m_aoVTISegInferer;

};
