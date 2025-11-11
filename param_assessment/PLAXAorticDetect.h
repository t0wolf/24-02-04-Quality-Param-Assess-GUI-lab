#pragma once
#include "quality_control/roi_detection.h"


class PLAXAorticDetect : public ROIDetection
{
public:
	PLAXAorticDetect(std::string& strEnginePath);

	std::vector<Object> doInference(cv::Mat& src);
};

