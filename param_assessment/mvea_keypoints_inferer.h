#pragma once

#include "avad_keypoints_inferer.h"

class MVEAKeypointsInferer : public AVADKeypointsInferer
{
public:
	MVEAKeypointsInferer(std::string& sEnginePath);

	~MVEAKeypointsInferer();

	int doInference(cv::Mat& src, std::vector<cv::Point>& eaPoints);

	int preprocess(cv::Mat& src, float* blob);

	//int blobFromImage(cv::Mat& src, float* blob);

private:
	int doSingleInfer(cv::Mat& src, std::vector<cv::Point>& eaPoints);

	int postProcess(float* output, cv::Mat& src, std::vector<cv::Point>& eaPoints);

	cv::Mat m_invertTrans;

};

