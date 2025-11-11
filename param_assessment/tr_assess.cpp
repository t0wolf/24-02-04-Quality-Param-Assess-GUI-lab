#include "tr_assess.h"

TRAssess::TRAssess(std::string& sObjdetectEnginePath, std::string& sKeyptEnginePath)
	:PVAssess(sObjdetectEnginePath, sKeyptEnginePath)
{

}

int TRAssess::trAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
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

int TRAssess::postProcess(cv::Mat& src, std::vector<Object>& objects, std::vector<std::vector<cv::Point>>& eaPointsList,
	std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	int yBaseline = 0;  // 检测框的上边沿y均值
	int yCoordTr = 0;  // pv点的y坐标均值
	std::vector<int> vCoordsTr;
	//int yCoordA = 0;  // A点的y坐标均值
	std::vector<std::vector<cv::Point>> originEAPoints(eaPointsList);
	for (int i = 0; i < objects.size(); i++) // 计算检测到的点在原图上的坐标
	{
		yBaseline += objects[i].rect.y;

		originEAPoints[i][0].x += objects[i].rect.x;  // 0为pv
		originEAPoints[i][0].y += objects[i].rect.y;
		yCoordTr += originEAPoints[i][0].y;
		vCoordsTr.push_back(std::abs(originEAPoints[i][0].y - objects[i].rect.y));

		//originEAPoints[i][1].x += objects[i].rect.x;  // 1为A
		//originEAPoints[i][1].y += objects[i].rect.y;
		//yCoordA += originEAPoints[i][1].y;
	}
	yBaseline /= objects.size();
	yCoordTr /= objects.size();
	//yCoordA /= objects.size();

	cv::Mat drawPoints = src.clone();
	for (auto& points : originEAPoints)
	{
		// E点
		cv::circle(drawPoints, points[0], 4, cv::Scalar(0, 255, 0), -1);
		//cv::putText(drawPoints, "E", cv::Point(points[0].x, points[0].y - 20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);
		// A点
		//cv::circle(drawPoints, points[1], 4, cv::Scalar(0, 0, 255), -1);
		//cv::putText(drawPoints, "A", cv::Point(points[1].x, points[1].y - 20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);
	}
	//cv::imshow("test", drawPoints);
	//cv::waitKey(0);

	//float scale = 1.0f;
	//std::vector<float> mvTR = { (yCoordTr - yBaseline) * scale };
	////float mvA = (yBaseline - yCoordA) * scale;

	//values.insert({ "TR", mvTR });
	////values.insert({ "MVA", mvA });
	//resultPics.insert({ "TR", drawPoints });

	ParamsAssessUtils::generateFinalResult("TR", vCoordsTr, values, drawPoints, resultPics);

	return 1;
}

