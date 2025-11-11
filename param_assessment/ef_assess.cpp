#include "ef_assess.h"

EFAssess::EFAssess(std::string& sSegEngineFilePath, std::string& sVideoEngineFilePath)
	:m_segEnginePath(sSegEngineFilePath)
	,m_videoEnginePath(sVideoEngineFilePath)
{
	m_segInferer = new EFSegmentInferer(m_segEnginePath);
	m_videoInferer = new EFVideoInferer(m_videoEnginePath);
}

EFAssess::~EFAssess()
{
	delete m_segInferer;
	delete m_videoInferer;
}

int EFAssess::doEFInference(std::vector<cv::Mat>& video, std::vector<int>& framePixels, std::vector<cv::Mat>& frameMasks, std::vector<float>& predScores)
{
	m_segInferer->doInference(video, framePixels, frameMasks);
	m_videoInferer->doInference(video, predScores);
	return 0;
}

int EFAssess::efAssessment(std::vector<cv::Mat>& video, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<int> framePixels;
	std::vector<cv::Mat> frameMasks;
	std::vector<float> predScores;
	doEFInference(video, framePixels, frameMasks, predScores);
	postProcess(video, framePixels, frameMasks, predScores, values, resultPics);

	return 1;
}

int EFAssess::postProcess(std::vector<cv::Mat>& video, std::vector<int>& framePixels, std::vector<cv::Mat> frameMasks, std::vector<float>& predScores, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	// 射血分数
	std::vector<int> framePixelsSorted(framePixels);
	std::sort(framePixelsSorted.begin(), framePixelsSorted.end());

	int trim_min = framePixelsSorted[static_cast<int>(pow(framePixelsSorted.size(), 0.05))];
	int trim_max = framePixelsSorted[static_cast<int>(pow(framePixelsSorted.size(), 0.95))];
	int trim_range = trim_max - trim_min;

	std::vector<int> edvPeakIndex = ParamsAssessUtils::findPeaks(framePixels, {}, {}, {}, 20, { 0.50 * trim_range, std::numeric_limits<double>::infinity() }, -1, {}, 0.5);  // 舒张期
	for (auto& size : framePixels) 
	{
		size = -size;
	}
	std::vector<int> esvPeakIndex = ParamsAssessUtils::findPeaks(framePixels, {}, {}, {}, 20, { 0.50 * trim_range, std::numeric_limits<double>::infinity() }, -1, {}, 0.5);  // 收缩期

	float score = 0.0f;
	int num = 0;

	for (auto& fScore : predScores)
	{
		score += fScore;
	}
	score /= static_cast<float>(predScores.size());
	//for (auto& peakIdx : esvPeakIndex) 
	//{
	//	if (0 <= peakIdx - 32 && peakIdx - 32 <= predScores.size()) 
	//	{
	//		score += predScores[peakIdx - 32];
	//		num++;
	//	}
	//}
	//if (num == 0) 
	//{
	//	std::cout << "该关键帧索引不在评分帧数范围内" << std::endl;
	//}
	// assessValues.push_back(score / num);
	std::vector<float> efValue = { score };
	values.insert({ "EF", efValue });

	// 计算左室容积
	/*std::vector<cv::Mat> edvMasks, esvMasks, edvBlended, esvBlended;

	for (auto& edvIdx : edvPeakIndex) 
	{
		cv::Mat blendImg, bgrMask;
		cv::cvtColor(frameMasks[edvIdx], bgrMask, cv::COLOR_GRAY2BGR);
		bgrMask.setTo(cv::Scalar(0, 0, 255), bgrMask > 0);
		cv::addWeighted(video[edvIdx], 0.7, bgrMask, 0.3, 0.0, blendImg);
		edvMasks.push_back(frameMasks[edvIdx]);
		edvBlended.push_back(blendImg);
		//cv::namedWindow("test", 0);
		//cv::resizeWindow("test", 500, 500);
		//cv::imshow("test", blendImg);
		//cv::waitKey(0);
	}
	for (auto& esvIdx : esvPeakIndex) 
	{	
		if (framePixels[esvIdx] != 0) 
		{
			cv::Mat blendImg, bgrMask;
			cv::cvtColor(frameMasks[esvIdx], bgrMask, cv::COLOR_GRAY2BGR);
			bgrMask.setTo(cv::Scalar(0, 0, 255), bgrMask > 0);
			cv::addWeighted(video[esvIdx], 0.7, bgrMask, 0.3, 0.0, blendImg);
			esvMasks.push_back(frameMasks[esvIdx]);
			esvBlended.push_back(blendImg);
			//cv::namedWindow("test", 0);
			//cv::resizeWindow("test", 500, 500);
			//cv::imshow("test", frameMasks[esvIdx]);
			//cv::waitKey(0);
		}
	}
	float edvVolume = 0.0f;
	float esvVolume = 0.0f;
	if (edvMasks.size() != 0) 
	{
		calcLvVolume(edvMasks, edvVolume);
		resultPics.insert({ "EDV", edvBlended[0] });
		values.insert({ "EDV", edvVolume });
	}
	else 
	{
		std::cout << "舒张期左室像素为0" << std::endl;
	}
	if (esvMasks.size() != 0) 
	{
		calcLvVolume(esvMasks, esvVolume);
		resultPics.insert({ "ESV", esvBlended[0] });
		values.insert({ "ESV", esvVolume });
	}
	else 
	{
		std::cout << "收缩期左室像素为0" << std::endl;
	}*/
	
	return 0;
}

int EFAssess::calcLvVolume(std::vector<cv::Mat>& lvMasks, float& averageVolume)
{
	std::vector<float> lvVolumes;
	for (auto& lvMask : lvMasks) 
	{
		std::vector<std::vector<cv::Point>> maskContours;
		cv::findContours(lvMask, maskContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		ImageProcess::filterSmallContour(maskContours);
		std::vector<cv::Point> maskContour = maskContours[0];
		cv::RotatedRect maskContourRect = cv::minAreaRect(maskContour);

		//cv::Rect contourBoundRect = cv::boundingRect(maskContour);

		cv::Point2f rectPoints[4];
		maskContourRect.points(rectPoints);
		//cv::Point2f orderedPoints[4];
		//ImageProcess::orderRotatedRectPoints(rectPoints, orderedPoints);  // 找到最小外界矩形的四个顶点

		std::sort(rectPoints, rectPoints + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
			return a.y < b.y;
		});  // 按照纵坐标从小到大排序

		// for test
		cv::Mat drawLine = lvMask.clone();
		cv::cvtColor(lvMask, drawLine, cv::COLOR_GRAY2BGR);

		//for (int i = 0; i < 4; i++) 
		//{
		//	cv::line(drawLine, orderedPoints[i], orderedPoints[(i + 1) % 4], cv::Scalar(0, 255, 0), 1);
		//}
		//cv::rectangle(drawLine, contourBoundRect, cv::Scalar(0, 0, 255), 1);

		//cv::namedWindow("test", 0);
		//cv::resizeWindow("test", 500, 500);
		//cv::imshow("test", drawLine);
		//cv::waitKey(0);


		// 找外界矩形上段和下段与LV轮廓的交点的中点
		std::vector<cv::Point> midLineEndPoints, midLinePoints;
		std::vector<std::pair<cv::Point, cv::Point>> vEndPoints = { {rectPoints[0], rectPoints[1]} /*, {rectPoints[2], rectPoints[3]}*/ };


 		for (int i = 0; i < vEndPoints.size(); i++)   // 上点用轮廓与mask的交点
		{
			std::pair<cv::Point, cv::Point> pointsPair = vEndPoints[i];
			int pixelNum = 1;

			//if (i == 1) 
			//{
			//	pixelNum = 10;
			//}
			//int maxSize = 0;

			std::vector<cv::Point> tempPoints, maxPoints;

			//for (int j = 0; j < pixelNum; j++)   // 当遍历到下段的时候，将下端水平上移，使得其与轮廓交点数最多
			//{	
				//pointsPair.first.y -= j;
				//pointsPair.second.y -= j;

			cv::LineIterator lineIt(pointsPair.first, pointsPair.second, 8, true);

			for (int i = 0; i < lineIt.count; ++i, ++lineIt)
			{
				cv::Point currPoint = lineIt.pos();
				if (cv::pointPolygonTest(maskContour, currPoint, false) == 0)
				{
					tempPoints.push_back(currPoint);
					//cv::circle(drawLine, currPoint, 1, cv::Scalar(255, 0, 0), -1);
				}
			}
				//if (tempPoints.size() > maxSize) 
				//{
				//	maxSize = tempPoints.size();
				//	maxPoints.clear();
				//	maxPoints = tempPoints;
				//}
			//}

			cv::Point selectedPoint = tempPoints[tempPoints.size() / 2];
			midLineEndPoints.push_back(selectedPoint);

			tempPoints.clear();
			
			//cv::circle(drawLine, selectedPoint, 1, cv::Scalar(0, 255, 255), -1);
			//cv::namedWindow("test", 0);
			//cv::resizeWindow("test", 500, 500);
			//cv::imshow("test", drawLine);
			//cv::waitKey(0);

		}
		cv::Point lowerMidPoint((rectPoints[2].x + rectPoints[3].x) / 2, (rectPoints[2].y + rectPoints[3].y) / 2);  // 下轮廓的点用矩形下段中点
		midLineEndPoints.push_back(lowerMidPoint);

		// 遍历中线上的点，获取法向量，及其与轮廓交点;计算L（腔室高度）和ai（腔室半径）的值，这两个值都需要比例尺
		float length = ParamsAssessUtils::calcLineDist(midLineEndPoints);

		cv::LineIterator midLineIt(midLineEndPoints[0], midLineEndPoints[1], 4, true);
		for (int i = 0; i < midLineIt.count; ++i, ++midLineIt)
		{
			cv::Point currPoint = midLineIt.pos();
			midLinePoints.push_back(currPoint);

		}
		std::vector<std::pair<float, float>> vNormals;
		ImageProcess::calcNormals(midLinePoints, vNormals);

		std::vector<std::pair<cv::Point, cv::Point>> endPointList;
		std::vector<std::vector<cv::Point>> lvInterPointsList;

		ImageProcess::extendNormalsOnImage(drawLine, midLinePoints, vNormals, endPointList);
		ImageProcess::getContourIntersectionPoint(drawLine, maskContours, endPointList, lvInterPointsList);  // 得到各个内径的两个端点

		float lvVolume = 0.0f;  // 利用辛普森法计算左室容积（无心尖二腔切面，以心尖四腔内径代替）
		int num = lvInterPointsList.size();
		for (auto& points : lvInterPointsList) 
		{
			cv::line(drawLine, points[0], points[1], cv::Scalar(0, 0, 255), 1);
			float lvInterDia = ParamsAssessUtils::calcLineDist(points);
			lvVolume += pow(lvInterDia, 2) * (length / num);
		}

		lvVolume *= Pi / 4;
		lvVolumes.push_back(lvVolume);
		
		//cv::circle(drawLine, lowerMidPoint, 1, cv::Scalar(0, 255, 255), -1);
		//cv::namedWindow("test", 0);
		//cv::resizeWindow("test", 500, 500);
		//cv::imshow("test", drawLine);
		//cv::waitKey(0);
	}

	averageVolume = std::accumulate(lvVolumes.begin(), lvVolumes.end(), 0) / lvVolumes.size();

	return 1;
}


