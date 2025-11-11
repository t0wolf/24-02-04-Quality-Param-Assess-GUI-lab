//
// Created by 成汉林 && 曾子炀 && 单淳劼.
//
#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
//#include "itkGDCMImageIO.h"
//#include "itkImageFileReader.h"
//#include "itkGDCMSeriesFileNames.h"
//#include "itkRGBPixel.h"
#include "typemappings.h"


//// ITK处理DICOM文件相关
//typedef itk::RGBPixel<unsigned char> PixelType;          //像素数据类型（鼓楼医院超声所用）
//typedef itk::Image<PixelType, 3> ImageType;              // 3维图像序列
//typedef itk::ImageFileReader<ImageType> ReaderType;
//typedef itk::GDCMImageIO ImageIOType;
//typedef itk::GDCMSeriesFileNames NamesGeneratorType;
//typedef itk::MetaDataDictionary DictionaryType;          // DICOM info
//typedef itk::MetaDataObject<std::string> MetaDataStringType;
//typedef itk::MetaDataObject<int> MetaDataIntType;


class utils {
public:
    static int getImageRoI(cv::Mat &img_r,
                           cv::Mat &imgC,
                           std::vector<std::vector<cv::Point>> &vContours,
                           int pixelThresh,
                           cv::Mat &kO,
                           cv::Mat &kC,
                           cv::Mat &kE,
                           std::vector<int> &kSizes);

    static std::vector<int> getImageRange(cv::Mat &img_r,
                                          bool isMask,
                                          int pixelThresh,
                                          std::vector<int> &kSizes,
                                          int dataMin = 0,
                                          int huRange = 255,
                                          int huOffset = 0);

    double getVideodata(std::string videoPath,
                      std::vector<cv::Mat>& frames,
                      std::vector<cv::Mat>& frames_gray,
                      int& totalCount);

    static dicom_res readDicomFile(std::string filepath, std::vector<cv::Mat>& frames, std::vector<cv::Mat>& frames_gray, int& totalCount);
    //static int _gettagidx(itk::GDCMImageIO::Pointer gdcmIO, std::string colortag, DictionaryType::Iterator& tag_idx);
    //static int _findTagVal(itk::GDCMImageIO::Pointer gdcmIO, std::string tagTag, std::string& tag_val);
    
    static cv::Mat createKernel(int radius);
	static std::vector <cv::Vec4i> rho_theta_2_x_y(std::vector<cv::Vec2f> lines_in, int img_size);
	static cv::Point get_line_cross_point(cv::Vec4i line1, cv::Vec4i line2);
	static std::vector<cv::Point> findMaxArea(std::vector<std::vector<cv::Point>> contours);
	static cv::Vec4i _extend_line(cv::Vec4i line, int cols, int rows, int basepix = 8, int flag = 0);
    static char* ParseDcmFile(std::string csFileName);
    // static std::string GetElementValue(char* pHeader, WORD wG, WORD wE);

//private:

	//// ITK处理DICOM文件相关
	//typedef unsigned char PixelType;                         //像素数据类型（鼓楼医院超声所用）
	//typedef itk::Image<PixelType, 3> ImageType;              // 3维图像序列
	//typedef itk::ImageSeriesReader<ImageType> ReaderType;
	//typedef itk::GDCMImageIO ImageIOType;
	//typedef itk::GDCMSeriesFileNames NamesGeneratorType;
    //typedef itk::MetaDataDictionary DictionaryType;          // DICOM info
    //typedef itk::MetaDataObject<std::string> MetaDataStringType;
};

