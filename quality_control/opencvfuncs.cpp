/*
   Created by SEU-BME-LBMD-chl, SEU-BME-LBMD-zzy, SEU-BME-LBMD-scj
*/
#include "opencvfuncs.h"


//序列图像的矩形区域裁剪
//输入：Mat图像vector，矩形区域（起始点x,y；裁剪范围）
//输出：与输入图像类型一致的Mat图像vector
std::vector<cv::Mat> getFramesRect(std::vector<cv::Mat> img_r, cv::Rect rect)
{
    std::vector<cv::Mat> img_w;
    cv::Mat image_cut;
    for (int i = 0; i < img_r.size(); i++)
    {
        image_cut = cv::Mat(img_r[i], rect);
        img_w.push_back(image_cut.clone());
    }
    image_cut.release();

    return img_w;
}


//序列图像的缩放
//输入：Mat图像vector，矩形区域（起始点x,y；裁剪范围）
//输出：与输入图像类型一致的Mat图像vector
std::vector<cv::Mat> getFramesResize(std::vector<cv::Mat> img_r, int new_size)
{
    std::vector<cv::Mat> img_w;
    cv::Mat image_temp;
    for (int i = 0; i < img_r.size(); i++)
    {
        cv::resize(img_r[i].clone(), image_temp, cv::Size(new_size, new_size));
        img_w.push_back(image_temp.clone());
    }
    image_temp.release();

    return img_w;
}


//序列图像的缩放
//输入：Mat图像vector，矩形区域（起始点x,y；裁剪范围）
//输出：与输入图像类型一致的Mat图像vector
std::vector<cv::Mat> getFramesResizeWH(std::vector<cv::Mat> img_r, int new_size)
{
    std::vector<cv::Mat> img_w;
    cv::Mat image_temp;

    // 计算缩放比例
    int frame_width = img_r[0].cols;
    int frame_height = img_r[0].rows;
    int resize_height;
    int resize_width;
    if (frame_height < frame_width)
    {
        resize_height = new_size;
        float resize_ratio_width = float(resize_height) / float(frame_height);
        resize_width = int(resize_ratio_width * frame_width);
    }
    else
    {
        resize_width = new_size;
        float resize_ratio_height = float(resize_width) / float(frame_width);
        resize_height = int(resize_ratio_height * frame_height);
    }

    //
    for (int i = 0; i < img_r.size(); i++)
    {
        cv::resize(img_r[i].clone(), image_temp, cv::Size(resize_width, resize_height));
        img_w.push_back(image_temp.clone());
    }
    image_temp.release();

    return img_w;
}



std::vector<int> genSampleIdxs(std::vector<cv::Mat> img_r, int frameSampleRate)
{
    std::vector<int> idxs;
    if (frameSampleRate <= 1)
    {
        for (int i = 0; i < img_r.size(); i++)
        {
            idxs.push_back(i);
        }
    }
    else
    {
        int choosed_idx = rand() % (frameSampleRate - 1 - 0) + 0;
        for (int i = choosed_idx; i < img_r.size(); i += frameSampleRate)
        {
            idxs.push_back(i);
        }
    }

    return idxs;
}


std::vector<cv::Mat> getFramesSampled(std::vector<cv::Mat> img_r, std::vector<int> sampleIdxs)
{

    std::vector<cv::Mat> img_w;
    for (int i = 0; i < sampleIdxs.size(); i ++)
    {
        img_w.push_back(img_r[i].clone());
    }
    
    return img_w;
}



//序列图像的缩放
//输入：Mat图像vector，矩形区域（起始点x,y；裁剪范围）
//输出：与输入图像类型一致的Mat图像vector
std::vector<cv::Mat> getFramesMaskedData(std::vector<cv::Mat> img_r, cv::Mat img_mask)
{
    std::vector<cv::Mat> img_w;
    cv::Mat image_temp;
    cv::Mat image_read;
    // 创建与序列图像同样数据格式的mask
    cv::Mat image_mask = img_mask.clone();
    image_mask.convertTo(image_mask, img_r[0].type());

    for (int i = 0; i < img_r.size(); i++)
    {
        image_read = img_r[i].clone();
        image_read.copyTo(image_temp, image_mask);
        img_w.push_back(image_temp.clone());
    }
    image_temp.release();
    image_read.release();

    return img_w;
}



std::vector<cv::Mat> getFramesBGR2RGB(std::vector<cv::Mat> img_r)
{
    std::vector<cv::Mat> img_w;
    cv::Mat image_temp;
    for (int i = 0; i < img_r.size(); i++)
    {
        cv::cvtColor(img_r[i].clone(), image_temp, cv::COLOR_BGR2RGB);
        img_w.push_back(image_temp.clone());
    }
    image_temp.release();

    return img_w;
}



//计算图像直方图
HistogramMat::HistogramMat()
{
	this->histSize[0] = 256;
	this->hranges[0] = 0.0;
	this->hranges[1] = 256.0;
	this->ranges[0] = hranges;
	this->channels[0] = 0;
}

cv::Mat HistogramMat::getHistogram(const cv::Mat& image, const cv::Mat mask)
{
	cv::Mat hist;
	cv::Mat hist_t;

	//存在部分mask与image尺寸不一致的情况
	cv::Rect rect(0, 0, image.cols, image.rows);
	cv::Mat image_cut = cv::Mat(mask, rect);
	cv::Mat cropped_mask = image_cut.clone();

	cv::calcHist(&image,
		1,//仅为一个图像的直方图
		channels,//使用的通道
		cropped_mask,//掩码
		hist,//作为结果的直方图
		1,//一维的直方图
		histSize,//箱子数量
		ranges//像素值的范围
	);

	// 256*1转换为1*256
	hist_t = hist.t();
	return hist_t;
}
