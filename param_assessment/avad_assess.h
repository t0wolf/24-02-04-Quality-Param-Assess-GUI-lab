#pragma once
#include "avad_keypoints_inferer.h"

// Aortic valve annulus diameter 主动脉瓣环直径
class AVADAssess
{
public:
	AVADAssess(std::string& sEnginePath);
	
	~AVADAssess();

	int doKeyptInference(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int avadAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int setScaleInfo(float& scaleLength, float& scale);

private:
	std::string m_keyptEnginePath;
	float m_scaleLength;
	float m_scale;
	AVADKeypointsInferer* m_avadKeyptInferer;
};