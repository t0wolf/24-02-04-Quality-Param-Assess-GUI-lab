#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "assess_utils.h"
#include "general_utils.h"
#define NOMINMAX

namespace ImageProcess
{
	int findMaxAreaConnected(cv::Mat& src, cv::Mat& dst);

	std::vector<cv::Point> findMaxContour(std::vector<std::vector<cv::Point>>& contours);

	float findMaxContourArea(std::vector<std::vector<cv::Point>>& contours);  // 求最大轮廓面积

	int filterSmallContour(std::vector<std::vector<cv::Point>>& contours);  // 滤除小的连通域

	int filterSmallComp(int numComp, cv::Mat& src, cv::Mat& labeledImage, cv::Mat& stats);

	int getSkeletonizeMask(cv::Mat& mask, cv::Mat& skeleton);

	int skeletonize(cv::Mat& src, cv::Mat& skeleton);

	int skeletonizeIter(cv::Mat& src, int iter);

	int distTransformFilter(cv::Mat& src);

	std::vector<cv::Point> removeConcavePart(std::vector<cv::Point>& skelContours);

	int getSkelPoints(cv::Mat& src, std::vector<cv::Point>& vPoints, int value);

	int getSkelPoints(cv::Mat& src, std::vector<cv::Point>& vPoints, bool value);

	int windowAverageCurve(std::vector<cv::Point>& points, std::vector<cv::Point2f>& smoothedPoints, int windowSize);  // 平滑点

	int windowAverageScale(std::vector<double>& scale, std::vector<double>& smoothedScale, int windowSize);  // 平滑标量

	int getLVMidLine(std::vector<cv::Point>& ivsPoints, std::vector<cv::Point>& pwPoints, std::vector<cv::Point>& midLine);  // 获取中轴线

	int getLVMidLine(std::vector<cv::Point2f>& ivsPoints, std::vector<cv::Point2f>& pwPoints, std::vector<cv::Point>& midLine);

	inline void pointSortByX(std::vector<cv::Point>& vPoints)  // 将点依照x从大到小的方向排列
	{
		std::sort(vPoints.begin(), vPoints.end(), [](const cv::Point& p1, const cv::Point& p2) {
			return p1.x < p2.x;
			});
	}

	int calcNormals(std::vector<cv::Point>& midLine, std::vector<std::pair<float, float>>& vNormals);  // 计算中轴线的法向量

	int calcNormals(std::vector<cv::Point2f>& midLine, std::vector<std::pair<float, float>>& vNormals);

	int extendNormalsOnImage(cv::Mat& src, std::vector<cv::Point>& skelPoints, std::vector<std::pair<float, float>>& vNormals,  // 计算中轴线法向量延长线端点值
		std::vector<std::pair<cv::Point, cv::Point>>& vEndPoints, int length = 400);

	int extendNormalsOnImage(cv::Mat& src, std::vector<cv::Point2f>& skelPoints, std::vector<std::pair<float, float>>& vNormals,
		std::vector<std::pair<cv::Point, cv::Point>>& vEndPoints, int length = 400);

	int extendNormal(cv::Mat& src, cv::Point& point, std::pair<float, float>& pairNormal, std::pair<cv::Point, cv::Point>& pairEndPoint, int length = 400);

	inline cv::Point clampPoint(cv::Point& point, int maxWidth, int maxHeight) {  // 限制法向量方向延申的点坐标不超过图像本身大小
		int x = (std::max)(0, (std::min)(point.x, maxWidth - 1));
		int y = (std::max)(0, (std::min)(point.y, maxHeight - 1));
		return cv::Point(x, y);
	}

	inline void pointSortByY(std::vector<cv::Point>& vPoints)  // 将点依照y从小到大的方向排列
	{
		std::sort(vPoints.begin(), vPoints.end(), [](const cv::Point& p1, const cv::Point& p2) {
			return p1.y < p2.y;
			});
	}

	int getLineRectIntersectionPoint(cv::Mat& src, cv::Rect& boundBox, std::pair<cv::Point, cv::Point>& endPoint, std::vector<cv::Point>& interPoints);

	std::vector<cv::Point> iterateLine(std::pair<cv::Point, cv::Point>& lineEdgePoints, int nStride);

	bool calculateIntersection(float x0, float y0, float dx, float dy, float& t, int imgWidth, int imgHeight);

	int getContourIntersectionPoint(cv::Mat& src, std::vector<std::vector<cv::Point>>& contour, std::vector<std::pair<cv::Point, cv::Point>>& vEndPoints,
		std::vector<std::vector<cv::Point>>& interPoints);  // 找两个端点对应的直线与轮廓的交点 但是此处contour是多个轮廓的集合

	int removeDuplicatePoints(std::vector<cv::Point>& srcPoints);

	int removeDuplicatePoints(std::vector<cv::Point>& srcPoints, float distThresh);

	std::vector<cv::Point> mapPointsBackToOriginal(const cv::Mat& originalImage, const std::vector<cv::Point>& scaledPoints);

	cv::Point calculateLineIntersection(const cv::Point& line1Pt1, const cv::Point& line1Pt2,
		const cv::Point& line2Pt1, const cv::Point& line2Pt2);

	inline bool compareRectPoints(const cv::Point2f& a, const cv::Point2f& b) {
		return a.x < b.x || (a.x == b.x && a.y < b.y);
	}

	void orderRotatedRectPoints(cv::Point2f pts[4], cv::Point2f dst[4]);

	std::pair<cv::Mat, cv::Mat> affineTransform(cv::Mat src, int fixedSize[2]);  // 仿射变换

	std::pair<cv::Mat, cv::Mat> affineTransform(cv::Mat src, int fixedSize[2], float scale);  // 仿射变换

	std::vector<float> adjustImage(int targetBox[4], int fixedSize[2]);

	std::vector<cv::Point> affinePoints(std::vector<cv::Point2f> originPoints, cv::Mat trans);

	int ivsAndPWPostProcess(cv::Mat& src, 
		std::vector<cv::Mat>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	float getPWStructureThickness(std::vector<std::vector<cv::Point>>& interPoints, std::vector<cv::Point>& resultLine);

	float getIVSThickness(std::vector<std::vector<cv::Point>>& interPoints, std::vector<cv::Point>& resultLine);
	
	std::vector<cv::Point> getLVIDPoint(std::vector<cv::Point>& ivsPoint, std::vector<cv::Point>& pwPoint);

	float getLVIDStructureThickness(std::vector<std::vector<cv::Point>>& ivsInterPoints, std::vector<std::vector<cv::Point>>& pwInterPoints, std::vector<cv::Point>& resultLine);

	int ladPostProcess(cv::Mat& src, cv::Mat& laMask, cv::Mat& avMask, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int asdAndSJDAPostProcess(cv::Mat& src, cv::Mat& asdAndsjdMask, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int aadPostProcess(cv::Mat& src, cv::Mat& aadMask, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	std::pair<int, float> getAADDist(std::vector<std::vector<cv::Point>>& vInterPoints);

	int polyfit(const std::vector<cv::Point>& points, int degree, std::vector<double>& coefficients);  // 依据给出的点计算多项式拟合的曲线

	inline std::vector<double> polyderivative(const std::vector<double>& coeffs) {	// 计算多项式的导数系数
		std::vector<double> derivative;
		for (size_t i = 1; i < coeffs.size(); ++i) {
			derivative.push_back(coeffs[i] * i);
		}
		return derivative;
	}

	inline double evaluatePolynomial(const std::vector<double>& coeffs, double x) {	// 计算多项式函数值
		double y = 0.0;
		for (size_t i = 0; i < coeffs.size(); ++i) {
			y += coeffs[i] * std::pow(x, i);
		}
		return y;
	}

	int findClosestPointOnLine(const cv::Point& lineStart, const cv::Point& lineEnd, const std::vector<cv::Point>& points, cv::Point& closestPoint);

	int getPerpendicularLineEndpoints(const std::vector<double>& coeffs, const cv::Point& pt_target, std::vector<cv::Point>& endPoints, double length);

	bool lineIntersection(const cv::Point& p1, const cv::Point& p2, const cv::Point& p3, const cv::Point& p4, cv::Point& intersection);

	int findLineContourIntersections(const std::vector<cv::Point>& line, const std::vector<std::vector<cv::Point>>& maskContours, std::vector<cv::Point>& intersections);

	bool isColorJudge(cv::Mat& src);

	// 调整对比度的函数
	QImage adjustContrast(const QImage& image, double alpha, int beta);

	int judgePointInBoundingRect(cv::Point& point, cv::Rect& rect);
};

