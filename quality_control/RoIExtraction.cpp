//
// Created by Ziyon Zeng && Hanlin Cheng on 2022/7/20.
//
#include "RoIExtraction.h"
#include "quality_utils.h"
#include <QDebug>


RoIExtract::RoIExtract()
{
	;
}

RoIExtract::~RoIExtract()
{
	std::vector<cv::Mat>().swap(this->m_framesGray);
	this->m_maskTemp.release();
	this->m_maskAll.release();
}


float findMax(std::vector<float> vec) {
    // float max = -999;
    // for (auto v : vec) {
    // 	if (max < v) max = v;
    // }
    auto it = std::max_element(vec.begin(), vec.end());
    return *it;
}

float CalcMHWScore(std::vector<float> hWScores)
{
    float median = 0.0f;
    sort(hWScores.begin(), hWScores.end());
    median = hWScores[int(hWScores.size() / 2)];
	return median;
}

void RoIExtract::setInput(std::vector<cv::Mat> frames_gray, const int totalCount) {
    std::vector<cv::Mat> meanFramesGray;
	int cols, rows;
	this->m_framesGray = frames_gray;
	cols = this->m_framesGray[0].cols;
	rows = this->m_framesGray[0].rows;

    m_maskTemp = this->m_framesGray[0].clone();
    for (auto& imgR : this->m_framesGray)
    {
        cv::Mat frame;
        imgR.convertTo(frame, CV_32FC3);
        float mean = cv::mean(frame).val[0];
        mean += 0.0001f;
        mean = (std::min)(mean, 32.0f);
        frame = frame / mean;
        meanFramesGray.push_back(frame);

        // double minVal, maxVal;
        // cv::minMaxLoc(frame, &minVal, &maxVal);
        // double median = (minVal + maxVal) / 2.0;
        // maxVal = (std::min)((std::max)(maxVal, 0.0), 8.0);
        // median = (std::min)((std::max)(median, 0.0), 4.0);
        // float finalVal = (maxVal + median) / 2.0;
        // finalVal = (std::min)(finalVal, 2.0f);
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::vector<float> pixelValues;
            for (auto it = meanFramesGray.begin(); it != meanFramesGray.end(); it += 4)
            {
                pixelValues.push_back(it->at<float>(i, j));
            }
            // for (auto& img : meanFramesGray)
            // {
            //     float tempValue = img.at<float>(i, j);
            //     pixelValues.push_back(tempValue);
            // }
            float max = findMax(pixelValues);
            float median = CalcMHWScore(pixelValues);
            if (max < 0) {
                max = 0;
            }
            else if (max > 8) {
                max = 8;
            }
            if (median < 0) {
                median = 0;
            }
            else if (median > 4) {
                median = 4;
            }
            float tempp = (max + median) / 2;
            if (tempp > 2) {
                tempp = 2;
            }
            m_maskTemp.at<uchar>(i, j) = tempp / 2 * 255;
        }
    }
	
}


//void RoIExtract::extractRuler() {
//
//}

void RoIExtract::drawMask(cv::Mat &realmask, cv::Point crosspt, std::vector<float> &radius, std::vector<cv::Vec4i> lines) {
	realmask = this->m_framesGray[0].clone();
	realmask.setTo(0);
	for (auto line : lines) {
		cv::Vec4i templine = utils::_extend_line(line, realmask.cols, realmask.rows);
		cv::line(realmask, cv::Point(templine[0], templine[1]), cv::Point(templine[2], templine[3]), 255, 1);
	}
	for (int i = 0; i < radius.size(); i++) {
		int temp_r = int(radius[i] + 0.5);
		cv::circle(realmask, crosspt, temp_r, 255, 1);
	}
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(realmask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	realmask.setTo(0);
	cv::drawContours(realmask, contours, -1, 255, cv::FILLED);
	cv::Mat kernel_o = utils::createKernel(2);
	cv::morphologyEx(realmask, realmask, cv::MORPH_OPEN, kernel_o);
}

std::vector<float> RoIExtract::findPeakvalue(std::vector<float> num, int count){
	std::vector<int> sign;
	for (int i = 1; i < count; i++)
	{
		/*相邻值做差：
		 *小于0，赋-1
		 *大于0，赋1
		 *等于0，赋0
		 */
		float diff = num[i] - num[i-1];
		if (diff > 0)
		{
			sign.push_back(1);
		}
		else if (diff < 0)
		{
			sign.push_back(-1);
		}
		else
		{
			sign.push_back(0);
		}
	}
	//再对sign相邻位做差  
	//保存极大值和极小值的位置  
	std::vector<float> indMax;
	for (int j = 1; j < sign.size(); j++)
	{
		int diff = sign[j] - sign[j - 1];
		if (diff < 0)
		{
			indMax.push_back(num[j]);
		}
	}
	return indMax;
}

void RoIExtract::houghPostProcessing(int faultflag, cv::Point &crosspt, std::vector<cv::Vec2f> lines, std::vector<float> &radius) {
	//extractRuler();
	std::cout << "霍夫变换未检测到线" << std::endl;
}

int MaxFreq_indexa(std::vector<int> a, int &n)
{
	std::map<int, int> mp;
	int maxfreqNuma = 0;
	n = 0;

	for (int i = 0; i < a.size(); i++)
		if (++mp[a[i]] >= n)
		{
			maxfreqNuma = a[i];
			n = mp[a[i]];
		}
			
	//n = mp[maxfreqNuma];
	return maxfreqNuma;
}

void RoIExtract::countMask(float gap, cv::Mat &realmask, cv::Point &crosspt, std::vector<float> &radius, bool firsttime) {
	auto morph_kernel = utils::createKernel(2);
	cv::Mat tempmaskall = firsttime ? m_maskAll.clone(): realmask.clone();

	//开后闭
	cv::morphologyEx(tempmaskall, tempmaskall, cv::MORPH_OPEN, morph_kernel);
	cv::morphologyEx(tempmaskall, tempmaskall, cv::MORPH_CLOSE, morph_kernel);

	/*-------------------- 计算扇形/环形径线参数（所得参数异常时直接返回0） --------------------*/
	int rows = tempmaskall.rows;
	int cols = tempmaskall.cols;

	//计算边缘
	cv::Mat edges = tempmaskall.clone();
	cv::Canny(tempmaskall, edges, 32, 64, 5);

	//cv::imshow("edges", tempmaskall);
	//cv::waitKey(0);


	//霍夫变换，得到扇形区域两个径线所在直线
	double radian[4] = { 25.0f, 80.0f, 100.0f, 155.0f };
	for (int i = 0; i < 4; i++) {
		radian[i] = radian[i] / 180.0 * CV_PI;
	}

	// Hough transform
	std::vector<cv::Vec2f> lines;
	cv::HoughLines(edges, lines, 0.5, CV_PI / 360.0f, 64, 0.0, 0.0, radian[0], radian[3]);

	if (lines.empty()) {
		int defalutflag = 0;
		houghPostProcessing(defalutflag, crosspt, lines, radius);

		// 因为houghPostProcessing实际上没有后处理，因此先return
		return;
	}

	// 根据直线斜率筛除部分检测到的直线
	std::vector<std::vector<std::vector<float>>> templines;
	std::vector<std::vector<float>> v2;
	std::vector<float> v1;
	v2.resize(2, v1);
	templines.resize(2, v2);
	for (auto line : lines) {
		float rho = line[0];
		float theta = line[1];
		if (theta > radian[1] && theta < radian[2]) {
			continue;
		}
		int index = int(theta >= radian[2]);
		templines[index][0].push_back(rho);
		templines[index][1].push_back(theta);
	}
	int losslinenum = 0;
    std::vector<cv::Vec2f> finalline(2, cv::Vec2f::all(0.0f));
	for (int i = 0; i < finalline.size(); i++) {
		if (templines[i].size() > 0) {
			v2 = templines[i];
			// 求原始line集合均值
			double mean[2];
			double stdev[2];
			for (int j = 0; j < 2; j ++) {
				double sum = std::accumulate(std::begin(v2[j]), std::end(v2[j]), 0.0);
				mean[j] = sum / v2[j].size(); //均值 
				double accum = 0.0;
				std::for_each(std::begin(v2[j]), std::end(v2[j]), [&](const double d) {
					accum += (d - mean[j])*(d - mean[j]);
				});
				if (v2[j].size() == 1) {
					stdev[j] = 0;
					continue;
				}
				stdev[j] = sqrt(accum / (v2[j].size()));//方差  
			}
			if (stdev[0] < 1.25) {
				stdev[0] = 1.25;
			}
			if (stdev[1] < 0.025) {
				stdev[1] = 0.025;
			}
			float finalrho = 0;
			float finaltheta = 0;
			int num = 0;
			for (auto line : lines) {
				float a = stdev[0] - abs(line[0] - mean[0]) + 0.00001;
				float b = stdev[1] - abs(line[1] - mean[1]) + 0.00001;
				if (a >= 0.0f && b >= 0.0f) {
					finalrho = finalrho + line[0];
					finaltheta = finaltheta + line[1];
					num++;
				}
 			}
            finalrho = finalrho / (static_cast<float>(num) + 0.00001f);
            finaltheta = finaltheta / (static_cast<float>(num) + 0.00001f);
			finalline[i][0] = finalrho;
			finalline[i][1] = finaltheta;
		}
	}

	cv::Mat showv = this->m_framesGray[0].clone();
	std::vector<cv::Vec4i> outlines;

	outlines = utils::rho_theta_2_x_y(finalline, showv.cols);
    if (outlines.size() < 2) {
		return;
	}

    crosspt = utils::get_line_cross_point(outlines[0], outlines[1]);
	// 根据轮廓上各个点到两径线交点的距离，确定扇形/环形半径
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(m_maskAll, contours, 0, 1);
    if (contours.empty())
    {
        return;
    }


	std::vector<cv::Point> largecontour;
	largecontour = utils::findMaxArea(contours);

	std::vector<float> dis;
	float temp;
	for (int k = 0; k < int(largecontour.size()); k++) {
		temp = sqrt(pow(largecontour[k].x - crosspt.x, 2) + pow(largecontour[k].y - crosspt.y, 2));
		dis.push_back(temp);
	}
	std::vector<float> disafterfind;
	disafterfind = findPeakvalue(dis, dis.size());

	std::vector<float> sortdis = disafterfind;
	std::sort(sortdis.begin(), sortdis.end());
	float thred = (sortdis[0] + sortdis.back()) / 2;
	int thred_max = showv.rows < showv.cols ? showv.rows : showv.cols;
	std::vector<std::vector<float>> finalradius(2);
	for (auto tdis : disafterfind) {
		if (tdis > thred_max)
		{
			continue;
		}
		if (tdis > thred) {
			finalradius[0].push_back(tdis);
		}
		else {
			finalradius[1].push_back(tdis);
		}
	}

	if (finalradius[1].size() > 5) {
		std::vector<int> testttttt;
		int nn = finalradius[1].size();
		for (auto m : finalradius[1]) {
			testttttt.push_back(int(int(m * 10 + 0.5) / 10 + 0.5));
		}
		int abcdef = MaxFreq_indexa(testttttt, nn);
		/*float tempradius = std::accumulate(std::begin(finalradius[1]), std::end(finalradius[1]), 0.0) / finalradius[1].size();
		for (auto m : finalradius[1]) {
			if (abs(m - tempradius) < 5) {
				tempra.push_back(m);
			}
		}
		tempradius = std::accumulate(std::begin(tempra), std::end(tempra), 0.0) / tempra.size();*/
		if (nn > 6) {
			radius.push_back(abcdef);
		}
		//tempra.clear();
	}
	if (finalradius[0].size() > 1) {
		std::vector<float> tempra;
		float tempradius = std::accumulate(std::begin(finalradius[0]), std::end(finalradius[0]), 0.0) / finalradius[0].size();
		for (auto m : finalradius[0]) {
			if (abs(m - tempradius) < 10) {
				tempra.push_back(m);
			}
		}
		//tempradius = std::accumulate(std::begin(tempra), std::end(tempra), 0.0) / tempra.size();
		//radius.push_back(tempradius);
		////////////////////////////////////// update on 2022.09.29
		if (tempra.size() > 0)
		{
			tempradius = std::accumulate(std::begin(tempra), std::end(tempra), 0.0) / tempra.size();
			radius.push_back(tempradius);
		}
	}
	if (!radius.empty() && realmask.empty() &&
		(crosspt.x != 0 && crosspt.y != 0) && !outlines.empty()) {
		drawMask(realmask, crosspt, radius, outlines);
	}
	else
	{
		//cv::Mat tempmask = cv::imread("../../extern/roi_extraction/original_mask.jpg", 0).clone();
        cv::Mat tempmask = cv::imread("D:/resources/20240221/quality_control_models/roi_extraction/original_mask.jpg", 0).clone();
		cv::resize(tempmask.clone(), tempmask, cv::Size(showv.cols, showv.rows));
		realmask = tempmask.clone();
		//realmask = cv::imread("../../extern/roi_extraction/original_mask.jpg",0).clone();
	}
}

void RoIExtract::getCropPoints(cv::Mat realmask, cv::Vec4i &points) {
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Point> maxcontours;
	cv::findContours(realmask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	if (!contours.empty())
	{
		if (contours.size() > 1) {
			maxcontours = utils::findMaxArea(contours);
		}
		else {
			maxcontours = contours[0];
		}
	}
	else
	{
		return;
	}
	
	std::vector<int> x;
	std::vector<int> y;
	for (auto point : maxcontours) {
		x.push_back(point.x);
		y.push_back(point.y);
	}
	sort(x.begin(), x.end());
	sort(y.begin(), y.end());
	points[0] = x[0];
	points[1] = y[0];
	points[2] = x[x.size() - 1];
	points[3] = y[y.size() - 1];
}


void RoIExtract::preprocessVideo(std::vector<cv::Mat> frames_gray, const int totalCount, cv::Mat &realmask, std::vector<float> &radius, cv::Rect &croprect) {
	setInput(frames_gray, totalCount);
	std::vector<std::vector<cv::Point>> contours;
	cv::Mat kO, kC, kE;
	std::vector<int> kernels = {1, 2, 5};
    //cv::imshow("mask_temp_cpp", mask_temp);
	//cv::waitKey(30);
	utils::getImageRoI(m_maskTemp, m_maskAll, contours, 32, kO, kC, kE, kernels);
	//cv::imshow("mask_all_cpp", maskall);
	//cv::waitKey(30);

	if (contours.empty())
		return;

	cv::Point crosspt;
	//std::vector<float> radius;
	countMask(2.0, realmask, crosspt, radius, true);
	//cv::imshow("realmask_cpp", realmask);
	//cv::waitKey(30);
	//cv::destroyAllWindows();
	cv::Vec4i croppoints;
	RoIExtract::getCropPoints(realmask, croppoints);
	if (radius.empty())
	{
		countMask(2.0, realmask, crosspt, radius, false);
		//RoIExtract::getCropPoints(realmask, croppoints);
	}
	croprect = cv::Rect(croppoints[0], croppoints[1], (croppoints[2]-croppoints[0]), (croppoints[3]-croppoints[1]));
}

//int main() {
//	RoIExtract a;
//	cv::Mat realmask;
//	cv::Rect croprect;
//	a.proprecessVideo("../b.avi", realmask, croprect);
//	return 0;
//}
