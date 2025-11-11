#pragma once
#include "cplusfuncs.h"
#include "opencvfuncs.h"
#include "general_utils.h"

#include "EchoQualityAssessmentA2C.h"
#include "EchoQualityAssessmentA3C.h"
#include "EchoQualityAssessmentA4C.h"
#include "EchoQualityAssessmentA5C.h"
#include "EchoQualityAssessmentPLAX.h"
#include "EchoQualityAssessmentPSAXGV.h"
#include "EchoQualityAssessmentPSAXMV.h"
#include "EchoQualityAssessmentPSAXPM.h"
#include "EchoQualityAssessmentPSAXA.h"
#include "roi_detection.h"


// finalResult qualityAssessVideo(std::string videoPath);

s_f_map qualityAssessVideo(std::string viewname, std::vector<cv::Mat> croppedVideo, std::vector<int> keyframes, float fRadius);

s_f_map doQualityAssessVideo(std::string viewname, std::vector<cv::Mat> croppedVideo, std::vector<int> keyframes, float fRadius);

//class QualityAssessment
//{
//public:
//	QualityAssessment();
//
//	~QualityAssessment();
//
//    finalResult predict(std::string videoPath);
//
//private:
//
//    utils* preprocess;
//    RoIExtract* roi_extraction;
//    ViewClassification* view_classification;
//    EchoQualityAssessmentA4C* echo_a4c;
//    EchoQualityAssessmentPSAXGV* echo_psaxgv;
//
//    ViewClassificationParams view_params;
//    EchoQualityAssessmentParams params_a4c;
//    EchoQualityAssessmentParams params_psax;
//};

