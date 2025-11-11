#pragma once

#include "param_roi_detection.h"
#include "tdi_ivs_keypoints_inferer.h"

class TDIIVSAssess
{
public:
	TDIIVSAssess(std::string& sObjdetectEnginePath, std::string& sKeyptEnginePath);

	~TDIIVSAssess();

	int doInference(cv::Mat& src, std::vector<Object>& objects, std::vector<std::vector<cv::Point>>& eaPointsList);

	int tdiivsAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int cropObjects(cv::Mat& src, std::vector<Object>& objects, std::vector<cv::Mat>& croppedImgs);

	int postProcess(cv::Mat& src, std::vector<Object>& objects, std::vector<std::vector<cv::Point>>& eaPointsList,
		std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

private:
	std::string m_objdetectEnginePath;

	std::string m_keyptEnginePath;

	ParamROIDetection* m_pTdiivsObjdetectInferer;

	TDIIVSKeypointsInferer* m_pTdiivsKeyptInferer;
};