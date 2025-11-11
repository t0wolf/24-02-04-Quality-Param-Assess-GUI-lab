#include "ivs_and_pw_assess.h"



IVSAndPWAssess::IVSAndPWAssess()
    : m_segEnginePath("D:/Resources/20240221/param_assess_models/IVS_PW_1219.engine")
	, m_segInferer(m_segEnginePath)
{

}

int IVSAndPWAssess::doSegInference(cv::Mat& src, cv::Mat& mask)
{
	m_segInferer.doInference(src, mask);

	return 1;
}

int IVSAndPWAssess::doSegInference(cv::Mat& src, std::vector<cv::Mat>& vMasks)
{
	m_segInferer.doInference(src, vMasks);

	return 1;
}

int IVSAndPWAssess::ivsAndPwAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)  // 先进行图像分割推理，得到分割结果后再进行后处理。
{
	std::vector<cv::Mat> vMask;
	doSegInference(src, vMask);
	auto start = std::chrono::high_resolution_clock::now();  // 使用std::chrono库获取当前时间点，作为代码执行开始的时间。
	postProcess(src, vMask, values, resultPics);
	auto end = std::chrono::high_resolution_clock::now();  // 使用std::chrono库获取当前时间点，作为代码执行结束的时间。
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);  // 输出代码执行的时间，以毫秒为单位。

	// 输出耗时
	//std::cout << "Time taken by code: " << duration.count() << " milliseconds." << std::endl;
	return 1;
}

int IVSAndPWAssess::postProcess(cv::Mat& src, cv::Mat& mask)
{
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::Mat skeleton;
	cv::Mat displayContours = cv::Mat::zeros(mask.size(), CV_32SC1);
	mask.convertTo(mask, CV_8UC1);
	cv::Mat binaryMask;

	//cv::threshold(mask, binaryMask, 0.0, 255.0, cv::THRESH_BINARY);
	//cv::findContours(binaryMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	getSkeletonizeMask(mask, skeleton);
	std::vector<std::vector<cv::Point>> skelContours;
	//std::vector<cv::Point> temp;

	//cv::drawContours(contourImage, skelContours, 1, cv::Scalar(0, 255, 0), 2);
	//cv::drawContours(contourImage, std::vector<std::vector<cv::Point>>{convexHull}, 0, cv::Scalar(0, 0, 255), 2);

	std::vector<cv::Point> lvMidLine;
	//getLVMidLine(skelContours, lvMidLine);

	//for (auto& point : lvMidLine)
	//	cv::circle(skeleton, point, 3, 255, 3);

	//cv::imshow("test", skeleton);
	//cv::waitKey(0);

	return 1;
}

int IVSAndPWAssess::postProcess(cv::Mat& src, std::vector<cv::Mat>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::Mat ivsSkeleton, pwSkeleton;
	cv::Mat ivsMask, pwMask;

	ivsMask = vMasks[0];
	pwMask = vMasks[1];

	ImageProcess::findMaxAreaConnected(ivsMask, ivsMask);
	ImageProcess::findMaxAreaConnected(pwMask, pwMask);

	cv::resize(ivsMask, ivsMask, src.size(), 0.0, 0.0, cv::INTER_NEAREST);
	cv::resize(pwMask, pwMask, src.size(), 0.0, 0.0, cv::INTER_NEAREST);

	cv::Mat displayContours = cv::Mat::zeros(ivsMask.size(), CV_32SC1);
	ivsMask.convertTo(ivsMask, CV_8UC1);
	pwMask.convertTo(pwMask, CV_8UC1);

	cv::Mat binaryMask;
	//cv::imshow("PW Mask", pwMask);
	//cv::imshow("IVS Mask", ivsMask);
	//cv::waitKey(1000);

	getSkeletonizeMask(ivsMask, ivsSkeleton);
	getSkeletonizeMask(pwMask, pwSkeleton);

	std::vector<cv::Point> ivsPoints, pwPoints;

	// for test
	//cv::imshow("PW skeleton", pwSkeleton);
	//cv::imshow("IVS skeleton", ivsSkeleton);
	//cv::waitKey(0);

	getSkelPoints(ivsSkeleton, ivsPoints, 255);
	getSkelPoints(pwSkeleton, pwPoints, 255);

	if (pwPoints.size() == 0 || ivsPoints.size() == 0) 
	{
		return 0;
	}

	removeDuplicatePoints(ivsPoints);
	removeDuplicatePoints(pwPoints);
	              
	pointSortByX(ivsPoints);
	pointSortByX(pwPoints);

	std::vector<cv::Point> midLine;
	std::vector<cv::Point2f> smoothedMidLine, smoothedIVSLine, smoothedPWLine;
	windowAverageCurve(ivsPoints, smoothedIVSLine, 5);
	windowAverageCurve(pwPoints, smoothedPWLine, 5);
	getLVMidLine(smoothedIVSLine, smoothedPWLine, midLine);
	windowAverageCurve(midLine, smoothedMidLine, 9);

	cv::Mat drawImage = src.clone();

	// for test
	//for (auto& point : ivsPoints)
	//	cv::circle(drawImage, point, 2, cv::Scalar(0, 0, 255), 2);
	//cv::imshow("midline", drawImage);
	//cv::waitKey(0);


	std::vector<std::pair<float, float>> vNormals;
	calcNormals(smoothedMidLine, vNormals);

	std::vector<std::pair<cv::Point, cv::Point>> endPointLists;
	extendNormalsOnImage(drawImage, smoothedMidLine, vNormals, endPointLists);

	//for (auto endPoint : endPointLists)
	//{
	//	cv::line(drawImage, endPoint.first, endPoint.second, cv::Scalar(0, 0, 255), 1);
	//}
	//cv::imshow("midline", drawImage);
	//cv::waitKey(0);

	std::vector<std::vector<cv::Point>> ivsContours, pwContours;
	std::vector<std::vector<cv::Point>> ivsInterPoints, pwInterPoints;
	cv::findContours(ivsMask, ivsContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	cv::findContours(pwMask, pwContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	filterSmallContour(ivsContours);
	filterSmallContour(pwContours);

	// for test
	//cv::drawContours(drawImage, ivsContours, -1, cv::Scalar(0, 0, 255), 2);
	//for (auto& point : smoothedIVSLine)
	//	cv::circle(drawImage, point, 2, cv::Scalar(0, 0, 255), 2);
	//cv::imshow("midline", drawImage);
	//cv::waitKey(0);

	getContourIntersectionPoint(src, ivsContours, endPointLists, ivsInterPoints);
	getContourIntersectionPoint(src, pwContours, endPointLists, pwInterPoints);
	if (ivsInterPoints.size() == 0 || pwInterPoints.size() == 0) 
	{
		return 0;
	}

	// for test
	//cv::drawContours(drawImage, ivsContours, -1, cv::Scalar(0, 0, 255), 2);
	//cv::drawContours(drawImage, pwContours, -1, cv::Scalar(0, 255, 0), 2);
	//for (auto& point : smoothedMidLine)
	//	cv::circle(drawImage, point, 2, cv::Scalar(255, 0, 0), 2);
	//for (auto& point : smoothedIVSLine)
	//	cv::circle(drawImage, point, 2, cv::Scalar(0, 0, 255), 2);
	//for (auto& point : smoothedPWLine)
	//	cv::circle(drawImage, point, 2, cv::Scalar(0, 0, 255), 2);
	//for (auto endPoint : endPointLists)
	//{
	//	cv::line(drawImage, endPoint.first, endPoint.second, cv::Scalar(0, 0, 255), 1);
	//}
	//cv::imshow("midline", drawImage);
	//cv::waitKey(0);

	std::vector<cv::Point> ivsLine, pwLine, lvidLine;
	float fIVSDist = getIVSStructureThickness(ivsInterPoints, ivsLine);
	float fPWDist = getPWStructureThickness(pwInterPoints, pwLine);
	float fLVIDDist = getLVIDStructureThickness(ivsInterPoints, pwInterPoints, lvidLine);

	//cv::drawContours(drawImage, ivsContours, -1, cv::Scalar(0, 0, 255), 2);
	//cv::drawContours(drawImage, pwContours, -1, cv::Scalar(0, 255, 0), 2);
	//for (auto& point : smoothedMidLine)
	//	cv::circle(drawImage, point, 2, cv::Scalar(255, 0, 0), 2);
	//for (auto& point : smoothedIVSLine)
	//	cv::circle(drawImage, point, 2, cv::Scalar(0, 0, 255), 2);
	//for (auto& point : smoothedPWLine)
	//	cv::circle(drawImage, point, 2, cv::Scalar(0, 0, 255), 2);
	
	cv::Mat drawImageIsv = src.clone();
	cv::Mat drawImagePw = src.clone();
	cv::Mat drawImageLvid = src.clone();

	cv::line(drawImageIsv, ivsLine[0], ivsLine[1], cv::Scalar(0, 255, 0), 2);  // 绿色
    float ivsDst = ParamsAssessUtils::calcLineDist(ivsLine);

	cv::line(drawImagePw, pwLine[0], pwLine[1], cv::Scalar(0, 0, 255), 2);
    float pwDst = ParamsAssessUtils::calcLineDist(pwLine);

	cv::line(drawImageLvid, lvidLine[0], lvidLine[1], cv::Scalar(255, 0, 0), 2);
    float lvidDst = ParamsAssessUtils::calcLineDist(lvidLine);

	values.insert({ "LVDd", std::vector<float>{lvidDst} });
	values.insert({ "IVSTd", std::vector<float>{ivsDst} });
	values.insert({ "LVPWTd", std::vector<float>{pwDst }});
	resultPics.insert({ "LVDd", drawImageLvid });
	resultPics.insert({ "IVSTd", drawImageIsv });
	resultPics.insert({ "LVPWTd", drawImagePw });
	//values["LVPWd"] = pwDst;

	//cv::imshow("midline", drawImageLvid);
	//cv::imshow("midline1", drawImageIsv);
	//cv::imshow("midline2", drawImagePw);
	//cv::waitKey(0);
	return 0;
}


float IVSAndPWAssess::getPWStructureThickness(std::vector<std::vector<cv::Point>>& interPoints, std::vector<cv::Point>& resultLine)
{
	std::vector<float> vDists;
    ParamsAssessUtils::calcLinesDistance(interPoints, vDists);

	int segmentIdx2 = static_cast<int>(interPoints.size() * 2.0f / 3.0f);

	resultLine = interPoints[segmentIdx2];
	return vDists[segmentIdx2];
}

float IVSAndPWAssess::getIVSStructureThickness(std::vector<std::vector<cv::Point>>& interPoints, std::vector<cv::Point>& resultLine)
{
	std::vector<float> vDists;
    ParamsAssessUtils::calcLinesDistance(interPoints, vDists);

	//int segment1Range = static_cast<int>(interPoints.size() / 3.0f);
	//auto distMaxElem = std::max_element(vDists.begin(), vDists.begin() + segment1Range);
	//int maxIdx = std::distance(vDists.begin(), distMaxElem);
	//float maxDist = *distMaxElem;

	int segmentIdx2 = static_cast<int>(interPoints.size() * 0.5f);
	if (segmentIdx2 == 0) 
	{
		segmentIdx2 += 1;
	}
	auto seg2MaxElem = std::max_element(vDists.begin() + segmentIdx2 - 1, vDists.end());
	int seg2MaxIdx = std::distance(vDists.begin(), seg2MaxElem);
	std::vector<cv::Point> closestDistPoints = interPoints[seg2MaxIdx];
	float closestDist = vDists[seg2MaxIdx];
	resultLine = closestDistPoints;
	return closestDist;

	//if (maxDist - closestDist > 0.3f * maxDist)
	//{
	//	resultLine = closestDistPoints;
	//	return closestDist;
	//}
	//else
	//{
	//	resultLine = interPoints[maxIdx];
	//	return maxDist;
	//}
}

float IVSAndPWAssess::getLVIDStructureThickness(std::vector<std::vector<cv::Point>>& ivsInterPoints, std::vector<std::vector<cv::Point>>& pwInterPoints, std::vector<cv::Point>& resultLine)
{
	int numPointSample = std::min(pwInterPoints.size(), ivsInterPoints.size());
	std::vector<std::vector<cv::Point>> vLVIDPoints;
	std::vector<float> vDists;
	for (int i = 0; i < numPointSample; i++)
	{
		std::vector<cv::Point> currIVSInterPoint = ivsInterPoints[i];
		std::vector<cv::Point> currPWInterPoint = pwInterPoints[i];
		std::vector<cv::Point> currLVIDPoint = getLVIDPoint(currIVSInterPoint, currPWInterPoint);
		vLVIDPoints.push_back(currLVIDPoint);
	}
    ParamsAssessUtils::calcLinesDistance(vLVIDPoints, vDists);

	int startIdx = static_cast<int>(vLVIDPoints.size() / 3.0f);
	int endIdx = static_cast<int>(vLVIDPoints.size() * 7.0f / 10.0f);
	auto maxDistElem = std::max_element(vDists.begin() + startIdx, vDists.begin() + endIdx);
	int maxIdx = std::distance(vDists.begin(), maxDistElem);

	resultLine = vLVIDPoints[maxIdx];

	return *maxDistElem;
}

