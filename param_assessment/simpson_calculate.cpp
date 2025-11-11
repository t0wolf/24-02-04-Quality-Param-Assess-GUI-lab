#include "simpson_calculate.h"


int SimpsonCalculation::doSimsponCalc(cv::Mat& a2cImage, cv::Mat& a2cMask, cv::Mat& a4cImage, cv::Mat& a4cMask,
	std::map<std::string, float>& values, std::map<std::string, cv::Mat>& resultPics)
{
	cv::Mat a2cVisImage, a4cVisImage;
	cv::resize(a2cMask, a2cMask, a2cImage.size());
	cv::resize(a4cMask, a4cMask, a4cImage.size());
	float volumeValue = biPlaneSimsponCalc(a2cImage, a2cMask, a4cImage, a4cMask, a2cVisImage, a4cVisImage);

	values.insert({ "LV Volume",  volumeValue });
	resultPics.insert({ "A2C", a2cVisImage });
	resultPics.insert({ "A4C", a4cVisImage });

	if (volumeValue < 0.0f)
		return 0;

	return 1;
}

int SimpsonCalculation::doSimsponCalc(cv::Mat& currImage, cv::Mat& currMask, VolumeInfo& histVolumeInfo, VolumeInfo& currVolumeInfo, std::map<std::string, float>& values,
	std::map<std::string, cv::Mat>& resultPics, std::string& strPlaneMode, std::string& strViewName)
{
	currVolumeInfo.strViewName = strViewName;
	int ret = uniPlaneSimpsonCalc(currImage, currMask, currVolumeInfo, values, resultPics);

	if (!ret)
		return 0;

	if (strPlaneMode == "SP")
	{
		values.insert({ "BP LV Volume",  values["LV Volume"]});
		return 1;
	}

	else
	{
		float fBPVolume = biPlaneSimsponCalc(histVolumeInfo, currVolumeInfo);
		if (fBPVolume <= 0.0f)
			return 0;

		//cv::Mat currDemoImage = visualizeImage(currImage, currVolumeInfo.lvMaskContour, currVolumeInfo.vecLongAxisPoints, currVolumeInfo.vecShortAxisLength);
		values.insert({ "BP LV Volume",  fBPVolume });
		//resultPics.insert({ "BP", a2cVisImage });
		return 1;
	}


	//if (strViewName == "(BP)")
	//{
	//	ret = doSimsponCalc(a2cImage, a2cMask, a4cImage, a4cMask, values, resultPics);
	//}

	//else if (strViewName == "(A2C)" || strViewName == "(A4C)")
	//{
	//	ret = uniPlaneSimpsonCalc(a2cImage, a2cMask, values, resultPics);
	//}
	//else
	//{
	//	return 0;
	//}
	return ret;
}

int SimpsonCalculation::uniPlaneSimpsonCalc(cv::Mat& image, cv::Mat& mask, VolumeInfo& currVolumeInfo, std::map<std::string, float>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<cv::Point> longAxisPoints;
	std::vector<std::vector<cv::Point>> shortAxisLines, lvMaskContours;
	cv::Mat blendImage;
	std::string strCurrViewName = currVolumeInfo.strViewName;

	cv::resize(mask, mask, image.size());
	findLongAndShortAxis(image, mask, longAxisPoints, shortAxisLines, blendImage);
	findMaskContours(mask, lvMaskContours);

	float length = ParamsAssessUtils::calcLineDist(longAxisPoints);
	if (longAxisPoints.empty() || shortAxisLines.empty())
		return 0;

	float lvVolume = calculateLVVolume(shortAxisLines, length, currVolumeInfo.fPixPerUnit);

	if (lvVolume < 0.0f)
		return 0;

	//cv::imshow("blendImage", image);
	//cv::waitKey(0);
	//blendImage = blendImages(image, blendImage);

	currVolumeInfo.vecLongAxisPoints = longAxisPoints;
	currVolumeInfo.fLongAxisLength = length;
	currVolumeInfo.vecShortAxisLength = shortAxisLines;
	currVolumeInfo.fVolume = lvVolume;
	currVolumeInfo.lvMaskContour = lvMaskContours[0];

	values.insert({ "LV Volume",  lvVolume });
	resultPics.insert({ strCurrViewName, blendImage });

	return 1;
}

float SimpsonCalculation::biPlaneSimsponCalc(cv::Mat& a2cImage, cv::Mat& a2cMask, cv::Mat& a4cImage, cv::Mat& a4cMask,
	cv::Mat& visA2CImage, cv::Mat& visA4CImage)
{
	std::vector<cv::Point> a2cLongAxisPoints, a4cLongAxisPoints;
	std::vector<std::vector<cv::Point>> a2cDsa, a4cDsa;

	findLongAndShortAxis(a2cImage, a2cMask, a2cLongAxisPoints, a2cDsa, visA2CImage);
	findLongAndShortAxis(a4cImage, a4cMask, a4cLongAxisPoints, a4cDsa, visA4CImage);

	if (a2cLongAxisPoints.empty() || a4cLongAxisPoints.empty() || a2cDsa.empty() || a4cDsa.empty())
		return -10000.0f;

	float a2cLongAxisLength = ParamsAssessUtils::calcLineDist(a2cLongAxisPoints);
	float a4cLongAxisLength = ParamsAssessUtils::calcLineDist(a4cLongAxisPoints);
	float volume = calculateBiplaneLVVolume(a2cDsa, a2cLongAxisLength, -10000.0f, a4cDsa, a4cLongAxisLength, -10000.0f);

	return volume;
}

int SimpsonCalculation::drawSimpsonLinesOnImage(cv::Mat& demoImage, VolumeInfo& currVolumeInfo)
{
	return 0;
}

float SimpsonCalculation::biPlaneSimsponCalc(VolumeInfo& histVolumeInfo, VolumeInfo& currVolumeInfo)
{
	std::vector<std::vector<cv::Point>> histDsa = histVolumeInfo.vecShortAxisLength;
	std::vector<std::vector<cv::Point>> currDsa = currVolumeInfo.vecShortAxisLength;
	float histLsa = histVolumeInfo.fLongAxisLength;
	float currLsa = currVolumeInfo.fLongAxisLength;

	float fBPVolume = calculateBiplaneLVVolume(histDsa, histLsa, histVolumeInfo.fPixPerUnit, currDsa, currLsa, currVolumeInfo.fPixPerUnit);
	if (fBPVolume < 0)
		return -10000.0f;

	return fBPVolume;
}

std::vector<float> SimpsonCalculation::computeSimpson(cv::Mat& a2cImge, cv::Mat& a2cMask, cv::Mat& a4cImage, cv::Mat& a4cMask, cv::Mat& visA2CImage, cv::Mat& visA4CImage)
{
	return std::vector<float>();
}

int SimpsonCalculation::calcLineNormal(std::pair<cv::Point, cv::Point>& vecLinePoints, std::pair<float, float>& vecNormal)
{
	cv::Point topPoint = vecLinePoints.first;
	cv::Point bottomPoint = vecLinePoints.second;
	float x = static_cast<float>(bottomPoint.x) - static_cast<float>(topPoint.x);
	float y = static_cast<float>(bottomPoint.y) - static_cast<float>(topPoint.y);
	float norm = sqrt(x * x + y * y);

	if (norm != 0.0f)
		vecNormal = std::make_pair<float, float>(x / norm, y / norm);
	else
		vecNormal = std::make_pair<float, float>(0.0f, 0.0f);
	return 1;
}

int SimpsonCalculation::findLongAndShortAxis(cv::Mat& src, cv::Mat& lvvMask, std::vector<cv::Point>& longAxisPoints,
	std::vector<std::vector<cv::Point>>& shortAxisLines, cv::Mat& visImage)
{
	cv::Mat debugImage = lvvMask.clone();
	cv::cvtColor(debugImage, debugImage, cv::COLOR_GRAY2BGR);

	std::vector<std::vector<cv::Point>> maskContours;
	findMaskContours(lvvMask, maskContours);

	if (maskContours.empty())
	{
		QtLogger::instance().logMessage("[I] Simpson Calculate - Mask contour empty.");
		return 0;
	}

	std::vector<cv::Point> maskContour = maskContours[0];
	cv::RotatedRect maskContourRect = getMinAreaRect(maskContour);
	std::vector<cv::Point2f> rectPoints = getRectPoints(maskContourRect);

	longAxisPoints = getMidlineEndpoints(rectPoints, maskContours);
	cv::drawContours(debugImage, maskContours, -1, cv::Scalar(0, 255, 0), 2);
	for (int i = 0; i < 4; i++) {
		cv::line(debugImage, rectPoints[i], rectPoints[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
	}
	//for (auto& point : longAxisPoints)
	//{
	//	cv::circle(debugImage, point, 5, cv::Scalar(0, 255, 0), 2);
	//}

	if (longAxisPoints.empty())
	{
		QtLogger::instance().logMessage("[I] Simpson Calculate - Long axis points empty.");
		return 0;
	}

	std::vector<cv::Point> midLinePoints = getMidlinePoints(longAxisPoints, maskContour);
	if (midLinePoints.empty())
	{
		QtLogger::instance().logMessage("[I] Simpson Calculate - Long axis line points empty.");
		return 0;
	}

	for (auto& point : midLinePoints)
	{
		cv::circle(debugImage, point, 5, cv::Scalar(255, 0, 0), 1);
	}

	cv::line(debugImage, midLinePoints[0], midLinePoints[1], cv::Scalar(255, 0, 0), 2);
	//cv::imshow("longAxisPoint", debugImage);
	//cv::waitKey(0);

	float fLongAxisLength = ParamsAssessUtils::calcLineDist(midLinePoints);

	std::pair<float, float> pairNormal;
	std::pair<cv::Point, cv::Point> pairLinePoints = std::make_pair(midLinePoints[0], midLinePoints.back());
	calcLineNormal(pairLinePoints, pairNormal);
	std::vector<std::pair<float, float>> vNormals = std::vector<std::pair<float, float>>{ pairNormal };

	std::vector<std::pair<cv::Point, cv::Point>> endPointList;
	std::vector<std::vector<cv::Point>> lvInterPointsList;

	cv::Mat drawLine = lvvMask.clone();
	cv::cvtColor(lvvMask, drawLine, cv::COLOR_GRAY2BGR);

	getShortAxisLines(debugImage, maskContours, endPointList, pairLinePoints, fLongAxisLength, shortAxisLines, 20);
	if (shortAxisLines.empty())
	{
		QtLogger::instance().logMessage("[I] Simpson Calculate - Short axis line points empty.");
		return 0;
	}
	visImage = visualizeImage(src, maskContour, longAxisPoints, shortAxisLines);

	//cv::imshow("vis_image", visImage);
	//cv::waitKey(0);

	return 1;
}

void SimpsonCalculation::findMaskContours(const cv::Mat& lvvMask, std::vector<std::vector<cv::Point>>& maskContours)
{
	cv::Mat preprocessMask;
	lvvMask.convertTo(preprocessMask, CV_8U);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::morphologyEx(preprocessMask, preprocessMask, cv::MORPH_OPEN, kernel);

	cv::findContours(lvvMask, maskContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	ImageProcess::filterSmallContour(maskContours);
}

cv::RotatedRect SimpsonCalculation::getMinAreaRect(const std::vector<cv::Point>& maskContour)
{
	return cv::minAreaRect(maskContour);
}

std::vector<cv::Point2f> SimpsonCalculation::getRectPoints(const cv::RotatedRect& maskContourRect)
{
	cv::Point2f rectPoints[4];
	maskContourRect.points(rectPoints);
	std::sort(rectPoints, rectPoints + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
		return a.y < b.y;
		});
	return std::vector<cv::Point2f>(rectPoints, rectPoints + 4);
}

std::vector<cv::Point> SimpsonCalculation::getMidlineEndpoints(const std::vector<cv::Point2f>& rectPoints, const std::vector<std::vector<cv::Point>>& maskContours)
{
	std::vector<cv::Point> midLineEndPoints;
	std::vector<std::pair<cv::Point, cv::Point>> vEndPoints = { {rectPoints[0], rectPoints[1]} };

	//for (const auto& pointsPair : vEndPoints) 
	//{
	//	std::vector<cv::Point> tempPoints;
	//	cv::LineIterator lineIt(pointsPair.first, pointsPair.second, 4, true);

	//	for (int i = 0; i < lineIt.count; ++i, ++lineIt) {
	//		cv::Point currPoint = lineIt.pos();
	//		if (cv::pointPolygonTest(maskContours[0], cv::Point2f((float)currPoint.x, (float)currPoint.y), false) > 0) {
	//			tempPoints.push_back(currPoint);
	//		}
	//	}

	//	if (tempPoints.empty()) {
	//		return {};
	//	}
	//	cv::Point selectedPoint = *tempPoints.begin();
	//	midLineEndPoints.push_back(selectedPoint);
	//}

	for (const auto& pointsPair : vEndPoints) {
		std::vector<cv::Point> tempPoints;
		cv::LineIterator lineIt(pointsPair.first, pointsPair.second, 8, true);

		for (int i = 0; i < lineIt.count; ++i, ++lineIt) {
			cv::Point currPoint = lineIt.pos();
			if (cv::pointPolygonTest(maskContours[0], currPoint, false) == 0) {
				tempPoints.push_back(currPoint);
			}
		}

		if (tempPoints.empty()) {
			return {};
		}

		cv::Point selectedPoint = tempPoints[tempPoints.size() / 2];
		midLineEndPoints.push_back(selectedPoint);
	}

	cv::Point lowerMidPoint((rectPoints[2].x + rectPoints[3].x) / 2, (rectPoints[2].y + rectPoints[3].y) / 2);
	midLineEndPoints.push_back(lowerMidPoint);
	return midLineEndPoints;
}

std::vector<cv::Point> SimpsonCalculation::getMidlinePoints(const std::vector<cv::Point>& midLineEndPoints)
{
	std::vector<cv::Point> midLinePoints;
	cv::LineIterator midLineIt(midLineEndPoints[0], midLineEndPoints[1], 4, true);
	for (int i = 0; i < midLineIt.count; ++i, ++midLineIt) {
		midLinePoints.push_back(midLineIt.pos());
	}
	return midLinePoints;
}

std::vector<cv::Point> SimpsonCalculation::getMidlinePoints(const std::vector<cv::Point>& midLineEndPoints, std::vector<cv::Point>& contour)
{
	if (midLineEndPoints.empty())
		return std::vector<cv::Point>();

	std::vector<cv::Point> linePointsInContour; // 存储最终符合条件的点
	std::vector<cv::Point> finalMidlinePoints;
	cv::Point start = midLineEndPoints[0];
	cv::Point end = midLineEndPoints.back();

	// 计算直线的增量
	int dx = end.x - start.x;
	int dy = end.y - start.y;

	// 计算直线的长度
	int steps = std::max(std::abs(dx), std::abs(dy)); // 直线像素点的数量
	float xStep = dx / static_cast<float>(steps);     // 单位步长（x方向）
	float yStep = dy / static_cast<float>(steps);     // 单位步长（y方向）

	// 初始化起点
	float x = static_cast<float>(start.x);
	float y = static_cast<float>(start.y);

	// 遍历直线上的每个点
	for (int i = 0; i <= steps; ++i) {
		// 获取当前点
		cv::Point currentPoint(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));

		// 判断当前点是否在轮廓内
		double result = cv::pointPolygonTest(contour, cv::Point2f(static_cast<float>(currentPoint.x), static_cast<float>(currentPoint.y)), false);

		// 如果点在轮廓内或在轮廓边界上（result >= 0），将其存储
		if (result >= 0) {
			linePointsInContour.push_back(currentPoint);
		}

		// 更新点的坐标
		x += xStep;
		y += yStep;
	}

	if (linePointsInContour.empty())
		return linePointsInContour;

	finalMidlinePoints = std::vector<cv::Point>{ linePointsInContour[0], linePointsInContour.back() };
	return finalMidlinePoints;
}

float SimpsonCalculation::pixelDistanceToUnit(float fPixDist, float fPixToUnit)
{
	if (fPixToUnit == 0.0f || fPixToUnit == -10000.0f)
		return fPixDist;
	
	float fUnitDist = fPixDist / fPixToUnit;
	return fUnitDist;
}

int SimpsonCalculation::getShortAxisLines(cv::Mat& src, std::vector<std::vector<cv::Point>>& contour,
	std::vector<std::pair<cv::Point, cv::Point>>& vEndPoints,
	std::pair<cv::Point, cv::Point>& pairLongAxisLine, float fLongAxisLength,
	std::vector<std::vector<cv::Point>>& interPoints, int nSegments)
{
	cv::Mat debugImage = src.clone();
	//cv::cvtColor(debugImage, debugImage, cv::COLOR_GRAY2BGR);
	int nShortAxis = vEndPoints.size();
	float stride = static_cast<float>(nShortAxis) / static_cast<float>(nSegments);

	float fSegmentHeight = fLongAxisLength / static_cast<float>(nSegments);

	std::pair<float, float> pairLongAxisNormal;
	calcLineNormal(pairLongAxisLine, pairLongAxisNormal);
	std::pair<float, float> pairShortAxisNormal = { -1.0 * pairLongAxisNormal.second, pairLongAxisNormal.first };

	std::vector<std::pair<cv::Point, cv::Point>> vecSelectedLines;

	for (int i = 0; i < nSegments + 1; ++i)
	{
		cv::Point currSegmentPoint;
		std::pair<cv::Point, cv::Point> pairExtendOrthLine;

		currSegmentPoint.x = pairLongAxisLine.first.x + static_cast<float>(i) * fSegmentHeight * pairLongAxisNormal.first;
		currSegmentPoint.y = pairLongAxisLine.first.y + static_cast<float>(i) * fSegmentHeight * pairLongAxisNormal.second;
		ImageProcess::extendNormal(src, currSegmentPoint, pairShortAxisNormal, pairExtendOrthLine);

		cv::circle(debugImage, currSegmentPoint, 5, cv::Scalar(0, 0, 255), 2);
		cv::line(debugImage, pairExtendOrthLine.first, pairExtendOrthLine.second, cv::Scalar(255, 255, 255), 1);
		vecSelectedLines.push_back(pairExtendOrthLine);
	}

	//cv::imshow("SegPoint", debugImage);
	//cv::waitKey(0);

	//for (int i = 0; i < nSegments; ++i)
	//{
	//	int currSelectedIdx = static_cast<int>(round(static_cast<float>(i) * stride));
	//	vSelectedPoints.push_back(vEndPoints[currSelectedIdx]);
	//}

	ImageProcess::getContourIntersectionPoint(src, contour, vecSelectedLines, interPoints);
	return 1;
}

float SimpsonCalculation::calculateLVVolume(std::vector<std::vector<cv::Point>>& lvInterPointsList, float length, float fPixToUnit)
{
	float lvVolume = 0.0f;
	int num = lvInterPointsList.size();
	float fUnitLength = pixelDistanceToUnit(length, fPixToUnit);

	for (auto& points : lvInterPointsList) {
		float lvInterDia = pixelDistanceToUnit(ParamsAssessUtils::calcLineDist(points), fPixToUnit);
		lvVolume += std::pow(lvInterDia, 2);
	}
	lvVolume = lvVolume * fUnitLength * Pi / 4 / num;
	return lvVolume;
}

float SimpsonCalculation::calculateBiplaneLVVolume(std::vector<std::vector<cv::Point>>& shortAxisLinesA2C, float longAxisA2C, float f2CPixToUnit,
	std::vector<std::vector<cv::Point>>& shortAxisLinesA4C, float longAxisA4C, float f4CPixToUnit)
{
	float lvVolume = 0.0f;
	int num = shortAxisLinesA2C.size();
	if (shortAxisLinesA2C.size() != shortAxisLinesA4C.size())
	{
		return 0.0f;
		QtLogger::instance().logMessage("[I] LVEF: Short Axis Size NOT equal.");
	}

	for (int i = 0; i < num; ++i)
	{
		auto a2cShortAxisLine = shortAxisLinesA2C[i];
		auto a4cShortAxisLine = shortAxisLinesA4C[i];

		float area = biplaneEllipseArea(a2cShortAxisLine, f2CPixToUnit, a4cShortAxisLine, f4CPixToUnit);
		lvVolume += area;
	}

	float fUnitA2CLongAxis = pixelDistanceToUnit(longAxisA2C, f2CPixToUnit);
	float fUnitA4CLongAxis = pixelDistanceToUnit(longAxisA4C, f4CPixToUnit);

	float fCurrLongAxisLength = std::max(fUnitA2CLongAxis, fUnitA4CLongAxis);
	lvVolume = lvVolume * fCurrLongAxisLength / float(num) * Pi / 4.0f;
	return lvVolume;
}

float SimpsonCalculation::biplaneEllipseArea(std::vector<cv::Point>& dsaA2C, float f2CPixToUnit, std::vector<cv::Point>& dsaA4C, float f4CPixToUnit)
{
	float dsaLengthA2C = ParamsAssessUtils::calcLineDist(dsaA2C);
	float dsaLengthA4C = ParamsAssessUtils::calcLineDist(dsaA4C);

	return pixelDistanceToUnit(dsaLengthA2C, f2CPixToUnit) * pixelDistanceToUnit(dsaLengthA4C, f4CPixToUnit);
}

cv::Mat SimpsonCalculation::blendImages(const cv::Mat& src, const cv::Mat& drawLine)
{
	cv::Mat blendImg;
	cv::Mat coloredDrawLine;
	cv::cvtColor(drawLine, coloredDrawLine, cv::COLOR_GRAY2BGR);
	coloredDrawLine.setTo(cv::Scalar(0, 0, 255), drawLine > 0);
	cv::resize(coloredDrawLine, coloredDrawLine, cv::Size(src.cols, src.rows));
	cv::addWeighted(src, 0.7, coloredDrawLine, 0.3, 0.0, blendImg);
	return blendImg;
}

cv::Mat SimpsonCalculation::visualizeImage(cv::Mat& src, std::vector<cv::Point>& maskContour, std::vector<cv::Point>& longAxisPoints,
	std::vector<std::vector<cv::Point>>& shortAxisLines)
{
	cv::Mat dst = src.clone();
	cv::drawContours(dst, std::vector<std::vector<cv::Point>>{maskContour}, 0, cv::Scalar(0, 255, 0), 3);
	cv::line(dst, longAxisPoints[0], longAxisPoints[1], cv::Scalar(255, 0, 0));

	for (auto& point : shortAxisLines)
		cv::line(dst, point[0], point[1], cv::Scalar(0, 0, 255), 3);

	return dst;
}
