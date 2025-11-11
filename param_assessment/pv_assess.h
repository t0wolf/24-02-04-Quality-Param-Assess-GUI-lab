#pragma once

#include "param_roi_detection.h"
#include "pv_keypoints_inferer.h"
#include "mvea_assess.h"

class PVAssess
{
public:
	PVAssess(std::string& sObjdetectEnginePath, std::string& sKeyptEnginePath);

	~PVAssess();

	int doInference(cv::Mat& src, std::vector<Object>& objects, std::vector<std::vector<cv::Point>>& eaPointsList);

	int pvAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int cropObjects(cv::Mat& src, std::vector<Object>& objects, std::vector<cv::Mat>& croppedImgs);

	int postProcess(cv::Mat& src, std::vector<Object>& objects, std::vector<std::vector<cv::Point>>& eaPointsList,
		std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

private:
	std::string m_objdetectEnginePath;

	std::string m_keyptEnginePath;

	ParamROIDetection* m_pPvObjdetectInferer;

	PVKeypointsInferer* m_pPvKeyptInferer;
};