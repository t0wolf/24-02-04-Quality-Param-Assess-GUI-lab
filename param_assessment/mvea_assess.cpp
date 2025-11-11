#include "mvea_assess.h"

MVEAAssess::MVEAAssess(std::string& sObjdetectEnginePath, std::string& sKeyptEnginePath)
	:m_objdetectEnginePath(sObjdetectEnginePath)
	,m_keyptEnginePath(sKeyptEnginePath)
{
	m_pMveaObjdetectInferer = new ParamROIDetection(m_objdetectEnginePath);
	m_pMveaKeyptInferer = new MVEAKeypointsInferer(m_keyptEnginePath);
}

MVEAAssess::~MVEAAssess()
{
	delete m_pMveaObjdetectInferer;
	delete m_pMveaKeyptInferer;
}

int MVEAAssess::doInference(cv::Mat& src, std::vector<Object>& objects, std::vector<std::vector<cv::Point>>& eaPointsList)
{
	std::vector<cv::Mat> croppedImgs;
	objects = m_pMveaObjdetectInferer->doInference(src);
	cropObjects(src, objects, croppedImgs);
	for (auto& croppedImg : croppedImgs) 
	{
		//cv::imshow("crop", croppedImg);
		//cv::waitKey(0);

		std::vector<cv::Point> eaPoints;
		m_pMveaKeyptInferer->doInference(croppedImg, eaPoints);
		eaPointsList.push_back(eaPoints);
	}

	return 0;
}

int MVEAAssess::mveaAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<Object> objects;
	std::vector<std::vector<cv::Point>> eaPointsList;
	doInference(src, objects, eaPointsList);
	if (objects.size() < 2)
	{
		return 0;
	}
	postProcess(src, objects, eaPointsList, values, resultPics);

	return 1;
}



int MVEAAssess::cropObjects(cv::Mat& src, std::vector<Object>& objects, std::vector<cv::Mat>& croppedImgs)
{
	cv::Mat cropImg = src.clone();
	for (auto& object : objects) 
	{
		cv::Mat croppedImg = cropImg(object.rect);
		croppedImgs.push_back(croppedImg);
	}
	return 1;
}

int MVEAAssess::postProcess(cv::Mat& src, std::vector<Object>& objects, std::vector<std::vector<cv::Point>>& eaPointsList,
							std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	int yBaseline = 0;  // 检测框的下边沿y均值
	int yCoordE = 0;  // E点的y坐标均值
	int yCoordA = 0;  // A点的y坐标均值
	int APoints_false = 0;  // 与E点重合的A点的数量
	std::vector<int> vCoordE, vCoordA;
	std::vector<std::vector<cv::Point>> originEAPoints(eaPointsList);
	for (int i = 0; i < objects.size(); i++) // 计算检测到的点在原图上的坐标
	{
		int currBaseline = objects[i].rect.y + objects[i].rect.height;
		yBaseline += currBaseline;

		double DistanceEA = std::pow(originEAPoints[i][1].x - originEAPoints[i][0].x, 2);
		DistanceEA += std::pow(originEAPoints[i][1].y - originEAPoints[i][0].y, 2);
		DistanceEA = std::sqrt(DistanceEA);

		originEAPoints[i][0].x += objects[i].rect.x;  // 0为E
		originEAPoints[i][0].y += objects[i].rect.y;
		yCoordE += originEAPoints[i][0].y;
		vCoordE.push_back(std::abs(originEAPoints[i][0].y - currBaseline));

		if (DistanceEA > 10)
		{
			originEAPoints[i][1].x += objects[i].rect.x;  // 1为A
			originEAPoints[i][1].y += objects[i].rect.y;
			yCoordA += originEAPoints[i][1].y;
			vCoordA.push_back(std::abs(originEAPoints[i][1].y - currBaseline));
		}
		else
		{
			originEAPoints[i][1].x = -1;
			originEAPoints[i][1].y = -1;
			APoints_false += 1;
		}
	}
	yBaseline /= objects.size();
	yCoordE /= objects.size();
	if (objects.size() - APoints_false > 0)
		yCoordA /= objects.size() - APoints_false;

	////buffer_250429
	//int yBaseline = 0;  // 检测框的下边沿y均值
	//int yCoordE = 0;  // E点的y坐标均值
	//int yCoordA = 0;  // A点的y坐标均值
	//std::vector<int> vCoordE, vCoordA;
	//std::vector<std::vector<cv::Point>> originEAPoints(eaPointsList);
	//for (int i = 0; i < objects.size(); i++) // 计算检测到的点在原图上的坐标
	//{
	//	int currBaseline = objects[i].rect.y + objects[i].rect.height;
	//	yBaseline += currBaseline;

	//	originEAPoints[i][0].x += objects[i].rect.x;  // 0为E
	//	originEAPoints[i][0].y += objects[i].rect.y;
	//	yCoordE += originEAPoints[i][0].y;
	//	vCoordE.push_back(std::abs(originEAPoints[i][0].y - currBaseline));

	//	originEAPoints[i][1].x += objects[i].rect.x;  // 1为A
	//	originEAPoints[i][1].y += objects[i].rect.y;
	//	yCoordA += originEAPoints[i][1].y;
	//	vCoordA.push_back(std::abs(originEAPoints[i][1].y - currBaseline));
	//}
	//yBaseline /= objects.size();
	//yCoordE /= objects.size();
	//yCoordA /= objects.size();
	////buffer_250429_end

	cv::Mat drawPoints = src.clone();
	//cv::rectangle(drawPoints, cv::Rect(100, 100, 100, 100), cv::Scalar(255, 0, 0), 2);

	int counter = 0;
	for (auto& points : originEAPoints) 
	{
		Object currObj = objects[counter];

		cv::rectangle(drawPoints, currObj.rect, cv::Scalar(0, 255, 0), 2);

		// E点
		cv::circle(drawPoints, points[0], 4, cv::Scalar(0, 255, 0), -1);
		cv::putText(drawPoints, "E", cv::Point(points[0].x, points[0].y - 20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);
		// A点
		if (points[1].x > 0 && points[1].y > 0)
		{
			cv::circle(drawPoints, points[1], 4, cv::Scalar(0, 0, 255), -1);
			cv::putText(drawPoints, "A", cv::Point(points[1].x, points[1].y - 20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);
		}
		//cv::circle(drawPoints, points[1], 4, cv::Scalar(0, 0, 255), -1);
		//cv::putText(drawPoints, "A", cv::Point(points[1].x, points[1].y - 20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);

		++counter;
	}

	ParamsAssessUtils::generateFinalResult("E", vCoordE, values, drawPoints, resultPics);
	ParamsAssessUtils::generateFinalResult("A", vCoordA, values, drawPoints, resultPics);

	return 1;
}


