#pragma once
#include "param_roi_detection.h"
#include "mvea_keypoints_inferer.h"
#include "type_define.h"
#include "assess_utils.h"

// mitral valve blood flow velocity E、A
class MVEAAssess	
{
public:
	MVEAAssess(std::string& sObjdetectEnginePath, std::string& sKeyptEnginePath);

	~MVEAAssess();

	int doInference(cv::Mat& src, std::vector<Object>& objects, std::vector<std::vector<cv::Point>>& eaPointsList);

	int mveaAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int cropObjects(cv::Mat& src, std::vector<Object>& objects, std::vector<cv::Mat>& croppedImgs);

	int postProcess(cv::Mat& src, std::vector<Object>& objects, std::vector<std::vector<cv::Point>>& eaPointsList,
					std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int generateFinalResult(std::vector<int>& originValues, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

private:
	std::string m_objdetectEnginePath;

	std::string m_keyptEnginePath;

	ParamROIDetection* m_pMveaObjdetectInferer;

	MVEAKeypointsInferer* m_pMveaKeyptInferer;
};