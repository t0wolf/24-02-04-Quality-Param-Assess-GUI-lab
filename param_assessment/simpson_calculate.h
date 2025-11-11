#pragma once
#include "lvv_assess.h"
#include "assess_utils.h"

struct VolumeInfo
{
    std::vector<cv::Point> vecLongAxisPoints;
    float fLongAxisLength = 0.0f;
    std::vector<std::vector<cv::Point>> vecShortAxisLength;
    float fVolume = 0.0f;
    std::string strViewName;
    std::vector<cv::Point> lvMaskContour;
    float fPixPerUnit = -10000.0f;
};


namespace SimpsonCalculation
{
	int doSimsponCalc(cv::Mat& a2cImge, cv::Mat& a2cMask, cv::Mat& a4cImage, cv::Mat& a4cMask, 
        std::map<std::string, float>& values, std::map<std::string, cv::Mat>& resultPics);

    int doSimsponCalc(cv::Mat& currImage, cv::Mat& currMask, VolumeInfo& histVolumeInfo, VolumeInfo& currVolumeInfo,
        std::map<std::string, float>& values, std::map<std::string, cv::Mat>& resultPics, std::string& strPlaneMode, std::string& strViewName);

	//int postProcess(std::vector<cv::Mat>& video, std::vector<cv::Mat>& vMasks,
	//	std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

    int uniPlaneSimpsonCalc(cv::Mat& image, cv::Mat& mask, VolumeInfo& currVolumeInfo, std::map<std::string, float>& values, std::map<std::string, cv::Mat>& resultPics);

    float biPlaneSimsponCalc(cv::Mat& a2cImge, cv::Mat& a2cMask, cv::Mat& a4cImage,
        cv::Mat& a4cMask, cv::Mat& visA2CImage, cv::Mat& visA4CImage);

    int drawSimpsonLinesOnImage(cv::Mat& demoImage, VolumeInfo& currVolumeInfo);

    float biPlaneSimsponCalc(VolumeInfo& histVolumeInfo, VolumeInfo& currVolumeInfo);

    std::vector<float> computeSimpson(cv::Mat& a2cImge, cv::Mat& a2cMask, cv::Mat& a4cImage,
        cv::Mat& a4cMask, cv::Mat& visA2CImage, cv::Mat& visA4CImage);

    int calcLineNormal(std::pair<cv::Point, cv::Point>& vecLinePoints, std::pair<float, float>& vecNormal);

    int findLongAndShortAxis(cv::Mat& src, cv::Mat& mask, std::vector<cv::Point>& midLinePoints, 
        std::vector<std::vector<cv::Point>>& lvInterPointsList, cv::Mat& visImage);

    void findMaskContours(const cv::Mat& lvvMask, std::vector<std::vector<cv::Point>>& maskContours);

    cv::RotatedRect getMinAreaRect(const std::vector<cv::Point>& maskContour);

    std::vector<cv::Point2f> getRectPoints(const cv::RotatedRect& maskContourRect);

    std::vector<cv::Point> getMidlineEndpoints(const std::vector<cv::Point2f>& rectPoints, const std::vector<std::vector<cv::Point>>& maskContours);

    std::vector<cv::Point> getMidlinePoints(const std::vector<cv::Point>& midLineEndPoints);

    std::vector<cv::Point> getMidlinePoints(const std::vector<cv::Point>& midLineEndPoints, std::vector<cv::Point>& contour);

    float pixelDistanceToUnit(float fPixDist, float fPixToUnit);

    int getShortAxisLines(cv::Mat& src, std::vector<std::vector<cv::Point>>& contour, std::vector<std::pair<cv::Point, cv::Point>>& vEndPoints,
        std::pair<cv::Point, cv::Point>& pairLongAxisLine, float fLongAxisLength, std::vector<std::vector<cv::Point>>& interPoints, int nSegments);

    float calculateLVVolume(std::vector<std::vector<cv::Point>>& lvInterPointsList, float length, float fPixToUnit);

    float calculateBiplaneLVVolume(std::vector<std::vector<cv::Point>>& shortAxisLinesA2C, float longAxisA2C, float f2CPixToUnit,
        std::vector<std::vector<cv::Point>>& shortAxisLinesA4C, float longAxisA4C, float f4CPixToUnit);

    float biplaneEllipseArea(std::vector<cv::Point>& dsaA2C, float f2CPixToUnit, std::vector<cv::Point>& dsaA4C, float f4CPixToUnit);
    
    cv::Mat blendImages(const cv::Mat& src, const cv::Mat& drawLine);

    cv::Mat visualizeImage(cv::Mat& src, std::vector<cv::Point>& maskContour,
        std::vector<cv::Point>& longAxisPoints, std::vector<std::vector<cv::Point>>& shortAxisLines);
};

