#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <numeric>
#include "QtLogger.h"


namespace ParamsAssessUtils
{
    std::vector<float> linspace(float start, float end, int num);

    int calcLinesDistance(std::vector<std::vector<cv::Point>>& vPoints, std::vector<float>& vDists);

    inline float calcLineDist(std::vector<cv::Point>& vPoints)
    {
        if (vPoints.empty())
            return 0.0f;

        cv::Point2f p1 = vPoints[0];
        cv::Point2f p2 = vPoints[1];

        cv::Point2f v = p2 - p1;
        float fDist = sqrt(v.x * v.x + v.y * v.y);
        return fDist;
    }


    inline float calcLineDist(std::pair<cv::Point, cv::Point>& pairPoints)
    {
        cv::Point2f p1 = pairPoints.first;
        cv::Point2f p2 = pairPoints.second;

        cv::Point2f v = p2 - p1;
        float fDist = sqrt(v.x * v.x + v.y * v.y);
        return fDist;
    }

    inline float calcLineDist(cv::Point2f& point1, cv::Point2f& point2)
    {
        cv::Point2f p1 = point1;
        cv::Point2f p2 = point2;

        cv::Point2f v = p2 - p1;
        float fDist = sqrt(v.x * v.x + v.y * v.y);
        return fDist;
    }

    int removeAbnormalValues(std::vector<float>& vValues);

    int removeAbnormalInterPoints(std::vector<std::vector<cv::Point>>& vInterPoints);

    std::vector<double> findLocalMaximum(std::vector<double> Vec);

    std::vector<std::pair<double, size_t>> findLocalMaximumPair(std::vector<double> Vec);

    std::vector<double> findLocalMinimum(std::vector<double> Vec);

    std::vector<size_t> findLocalMinimumIdx(std::vector<std::pair<double, size_t>> Vec);

    std::vector<float> gradientOneDimension(std::vector<float> vValues);

    std::vector<float> gradientOneDimension(std::vector<double> vValues);

    std::vector<size_t> findIndices(const std::vector<float>& array, float threshold);

    float sign(float value);

    std::vector<int> findPeaks(std::vector<int> x, std::vector<double> plateauSize = {}, std::vector<double> height = {}, std::vector<double> threshold = {}, int distance = 0, std::vector<double> prominence = {}, int wlen = -1, std::vector<double> width = {}, double relHeight = 0.5);

    struct Peak
    {
        int index;
        double height;
    };

    struct PeakIdx
    {
        int index;
        int height;
    };

    //std::vector<Peak> findPeaks(const std::vector<double>& x, double height = 0.0, double threshold = 0.0, int distance = 1, double prominence = 0.0, double width = 0.0, double rel_height = 0.5, int plateau_size = 0);

    std::vector<PeakIdx> findPeakIdxs(const std::vector<int>& x, double height = 0.0, double threshold = 0.0, int distance = 1, double prominence = 0.0, double width = 0.0, double rel_height = 0.5, int plateau_size = 0);

    double calculateStdDev(const std::vector<int>& data, double mean);

    double calculateMean(const std::vector<int>& data);

    std::vector<int> filterOutliers(const std::vector<int>& data, double threshold, int& outlierCounter);

    int findMedium(std::vector<int>& inputData);

    int generateFinalResult(const std::string valueName,
        std::vector<int>& originValues,
        std::map<std::string, std::vector<float>>& values,
        cv::Mat& showingPics,
        std::map<std::string, cv::Mat>& resultPics);

    int parseKeyframes(std::vector<int>& keyframeIdx, std::vector<cv::Mat>& inputVideo, std::vector<cv::Mat>& outputVideo, int length);

}
