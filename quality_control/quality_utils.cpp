//
// Created by 成汉林 && 曾子炀 && 单淳劼.
//

#include "quality_utils.h"
#include <sys/stat.h>



int
utils::getImageRoI(cv::Mat &img_r, cv::Mat &imgC, std::vector<std::vector<cv::Point>> &vContours, int pixelThresh,
                   cv::Mat &kO, cv::Mat &kC, cv::Mat &kE, std::vector<int> &kSizes) {
    int maxPixel = 255;
    // `s` stands for `size`
    int sO = 0;
    int sC = 0;
    int sE = 0;

    if (kO.empty() || kC.empty() || kE.empty()) {
        sO = kSizes[0];
        sE = kSizes[kSizes.size() - 1];
        if (kSizes.size() == 3)
            sC = kSizes[1];
        else
            sC = (sO + sE) / 3;
    }

    // building kernels if kO, kC, kE are empty.
    if (kO.empty())
        kO = createKernel(sO);
    if (kC.empty())
        kC = createKernel(sC);
    if (kE.empty())
        kE = createKernel(sE);

    cv::Mat imgCopy = img_r.clone();
//    assert(imgCopy.type() == CV_8U && "Image type should be uint8.");

    // threshold segmentation
    cv::Mat imgW1;
    cv::threshold(imgCopy, imgW1, pixelThresh, maxPixel, cv::THRESH_BINARY);//阈值分割，低于起始值

    // open and close op
    cv::Mat openResult;
    cv::Mat close1;
    cv::morphologyEx(imgW1, openResult, cv::MORPH_OPEN, kO);
    cv::morphologyEx(openResult, close1, cv::MORPH_CLOSE, kO);//开闭运算

    cv::Mat close2;
    cv::morphologyEx(close1, openResult, cv::MORPH_OPEN, kC);
    cv::morphologyEx(openResult, close2, cv::MORPH_CLOSE, kC);

    // find contours
    std::vector<std::vector<cv::Point>> contours1;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(close2, contours1, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);//查找二值图像中的轮廓

    // draw contours, padding with 255.
    cv::Mat imgTemp2 = cv::Mat::zeros(img_r.size(), CV_8UC1);
    cv::drawContours(imgTemp2, contours1, -1, cv::Scalar(maxPixel, maxPixel, maxPixel), cv::FILLED);

    cv::Mat open3;
    cv::morphologyEx(imgTemp2, open3, cv::MORPH_OPEN, kE);

    std::vector<std::vector<cv::Point>> contours2;
    std::vector<cv::Vec4i> hierarchy2;
    cv::findContours(open3, contours2, hierarchy2, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // for maximum area of contours.
    std::vector<double> vArea;
    for (auto &contour : contours2) {
        vArea.push_back(cv::contourArea(contour));
    }

    // if the body is bigger than the RoI, img_w3 in Python version.
    imgC = cv::Mat::zeros(img_r.size(), CV_8UC1);
    if (vArea.size() > 1) {
        int maxIdx = std::max_element(vArea.begin(), vArea.end()) - vArea.begin();
        int counter = 0;
        for (auto &contour : contours2) {
            cv::RotatedRect rect = cv::minAreaRect(contour);
            float aspectRatio = rect.size.height / rect.size.width;
            bool flag = true;

            if (aspectRatio < 0.35f || aspectRatio > 1.0f / 0.35f) {//找最小的矩形框，看他横纵比对劲不，不对劲就换更大的
                if (counter != maxIdx)
                    flag = false;
            }

            if (flag)
                vContours.push_back(contour);

            ++counter;
        }
        cv::drawContours(imgC, vContours, -1, (maxPixel, maxPixel, maxPixel), cv::FILLED);
    } else if (cv::sum(imgW1)[0] > imgW1.rows * imgW1.cols * maxPixel / 64)
        imgC = cv::Mat::ones(img_r.size(), CV_8UC1) * maxPixel;

    //else
    //    vContours = contours2;

    //cv::cvtColor(imgC, imgC, cv::COLOR_GRAY2RGB);
    //cv::drawContours(imgC, vContours, -1, (maxPixel, maxPixel, maxPixel), cv::FILLED);
    //cv::drawContours(imgC, vContours, -1, cv::Scalar(0, 255, 0), cv::FILLED);
    //cv::imshow("imgC_cpp", imgC);
    //cv::waitKey(0);
    //cv::destroyAllWindows();
    return 1;
}

cv::Mat utils::createKernel(int radius) {
    int kernelD = 2 * radius + 1;  // the diameter of kernel
    cv::Mat kernel = cv::Mat::ones(cv::Size(kernelD, kernelD), CV_8UC1);
    kernel = kernel * 255;

    for (int i = 0; i < kernel.rows; i++) {
        for (int j = 0; j < kernel.cols; j++) {
            if (std::pow((i - radius), 2) + std::pow((j - radius), 2) > std::pow(radius, 2))
                kernel.at<uint8_t>(i, j) = 0;
        }
    }

    return kernel;
}

std::vector<int>
utils::getImageRange(cv::Mat &img_r, bool isMask, int pixelThresh, std::vector<int> &kSizes, int dataMin, int huRange,
                     int huOffset) {
    int maxPixel = 255;
    int numPart = 5;

    int pixelD = static_cast<int>(maxPixel / numPart);
    int pixelU = maxPixel - static_cast<int>(maxPixel / numPart);

    cv::Mat imgData = img_r.clone();

    // make sure `imgData` is uint8
    if (imgData.type() == CV_16UC1) {
        cv::Mat fImgData;
        imgData.convertTo(fImgData, CV_32F);
        if (dataMin < 0)
            dataMin = huOffset;
        else
            dataMin = 0;

        cv::Mat tempData = (fImgData - static_cast<float>(dataMin)) / static_cast<float>(huRange) * 255.0f;
        tempData.convertTo(imgData, CV_8UC1);
    }

    cv::Mat imgTemp = imgData.clone();

    // set small and big value to 0
    for (int i = 0; i < imgTemp.rows; i++) {
        for (int j = 0; j < imgTemp.cols; j++) {
            auto value = imgTemp.at<uchar>(i, j);
            if (value < pixelD || value > pixelU)
                imgTemp.at<uchar>(i, j) = 0;
        }
    }

    if (static_cast<int>((cv::sum(imgTemp)[0]) < 256 * maxPixel) && !isMask) {
        std::vector<std::vector<cv::Point>> contours;
        cv::Mat kO;
        cv::Mat kC;
        cv::Mat kE;
        getImageRoI(imgData, imgData, contours, pixelThresh, kO, kC, kE, kSizes);
    }

    int rows = imgData.rows;
    int cols = imgData.cols;

    int firstRow = 0;
    int lastRow = rows - 1;
    int firstCol = 0;
    int lastCol = cols - 1;

    std::vector<int> rowFlag;
    std::vector<int> colFlag;

    for (int i = 0; i < imgData.rows; i++) {
        cv::Mat singleRow = imgData.row(i);
        if (cv::sum(singleRow)[0] > 8.0)
            rowFlag.push_back(i);
    }

    if (rowFlag.size() >= 2) {
        auto minMax = std::minmax_element(rowFlag.begin(), rowFlag.end());
        firstRow = *minMax.first;
        lastRow = *minMax.second;
    }

    for (int j = 0; j < imgData.cols; j++) {
        cv::Mat singleCol = imgData.col(j);
        if (cv::sum(singleCol)[0] > 8.0)
            colFlag.push_back(j);
    }

    if (colFlag.size() >= 2) {
        auto minMax = std::minmax_element(colFlag.begin(), colFlag.end());
        firstCol = *minMax.first;
        lastCol = *minMax.second;
    }

    std::vector<int> result = {firstRow, lastRow, firstCol, lastCol};

    return result;
}



// 利用opencv解析视频，并处理为std::vector<cv::Mat> imgdata
double
utils::getVideodata(std::string videoPath, std::vector<cv::Mat> &frames, std::vector<cv::Mat> &frames_gray, int &totalCount)
{
    cv::VideoCapture capture;
    capture.open(videoPath);
    if (!capture.isOpened())
    {
        std::cerr << "[E] Cannot open this video.\n";
    }
    double frame_rate = static_cast<double>(capture.get(cv::CAP_PROP_FPS));
    cv::Mat frame;
    int counter = 0;
    int frame_count = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));

    while (counter < frame_count) {
        if (!capture.isOpened())
        {
            std::cout << "[I] Single video ends there.\n";
            break;
        }

        ++counter;
        int ret = capture.read(frame);
        if (!ret)
        {
            counter -= 1;
            frame_count = counter;
            break;
        }
        frames.push_back(frame.clone());
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        frames_gray.push_back(frame.clone());
    }

    capture.release();
    totalCount = frame_count;

    return frame_rate;
}


//// 利用ITK解析DICOM文件，并处理为std::vector<cv::Mat> imgdata
//dicom_res
//utils::readDicomFile(std::string filepath, std::vector<cv::Mat>& frames, std::vector<cv::Mat>& frames_gray, int& totalCount)
//{
//    // 判断文件是否存在，不存在则直接返回
//    dicom_res dicom_info;
//    struct _stat buf;
//    int filestat = _stat(filepath.c_str(), &buf);
//    if (filestat != 0)
//    {
//        dicom_info.re_flag = EXIT_FAILURE;
//        return dicom_info;
//    }
//
//    // ITK在读取超大DICOM文件时也会抛出异常(如1538帧、436MB的文件)，因此大尺寸文件单独处理
//    // (0028,0008) NumberOfFrames
//    long filesize_th = 129 * 1024 * 1024;
//    char* pHeader = ParseDcmFile(filepath);
//    if (pHeader == NULL)
//    {
//        dicom_info.re_flag = EXIT_FAILURE;
//        return dicom_info;
//    }
//    std::string	numFramesStr = GetElementValue(pHeader, 0x0028, 0x0008);
//    int numFrames = atoi(numFramesStr.c_str());
//    if ((numFrames > 1024) || (buf.st_size > filesize_th))
//    {
//        dicom_info.re_flag = EXIT_FAILURE;
//        return dicom_info;
//    }
//
//
//    // 利用ITK解析DICOM文件
//    try
//    {  
//        // 用于最后获取数据的ImageFileReader
//        itk::ImageFileReader<itk::Image<PixelType, 3>>::Pointer reader = itk::ImageFileReader<itk::Image<PixelType, 3>>::New();
//        // 用于最后获取数据的GDCMImageIO
//        itk::GDCMImageIO::Pointer gdcmIO = itk::GDCMImageIO::New();
//
//        reader->SetImageIO(gdcmIO);
//        reader->SetFileName(filepath);
//
//
//        /// 判断能否正常读取，不能则直接返回
//        try
//        {
//            reader->Update();
//            reader->GetMetaDataDictionary();
//            gdcmIO->GetMetaDataDictionary();
//        }
//        catch (itk::ExceptionObject& ex)
//        {
//            //// ITK高版本中使用了智能指针，因此不再需要手动释放
//            //gdcmIO->Delete();
//            //reader->Delete();
//            dicom_info.re_flag = EXIT_FAILURE;
//            return dicom_info;
//        }
//
//
//        /// 判断成像模态是否为超声
//        // 512这个数值一定一定不要轻易修改！ITK GetModality()默认返回char数组长度512，改了值会造成数组越界！
//        const size_t modality_str_len = 512;
//        char modality[modality_str_len];
//        try
//        {
//            gdcmIO->GetModality(modality);
//        }
//        catch (itk::ExceptionObject& ex)
//        {
//            dicom_info.re_flag = EXIT_FAILURE;
//            return dicom_info;
//        }
//
//        if (strcmp(modality, "US") != 0)
//        {
//            dicom_info.re_flag = EXIT_FAILURE;
//            return dicom_info;
//        }
//
//
//        /// 判断帧数是否大于等于2（数据维度是否维度为3）
//        int tagframes_val = 0; // 使用ITK的GetNumberOfDimensions 可能失败，且未能捕获异常，因此...
//        std::string framestag = "0028|0008";
//        std::string tagframes_val_str;
//        if (_findTagVal(gdcmIO, framestag, tagframes_val_str) == EXIT_FAILURE)
//        {
//            dicom_info.re_flag = EXIT_FAILURE;
//            return dicom_info;
//        }
//
//        tagframes_val = atoi(tagframes_val_str.c_str());
//        if (tagframes_val < 2)
//        {
//            dicom_info.re_flag = EXIT_FAILURE;
//            return dicom_info;
//        }
//
//
//        // 通过TAG"0028|0014"判断是B超还是D超（鼓楼医院dicom文件特性）
//        // 若无TAG则直接不进行处理
//        std::string tagcolor_val;
//        std::string colortag = "0028|0014";
//        if (_findTagVal(gdcmIO, colortag, tagcolor_val) == EXIT_FAILURE)
//        {
//            dicom_info.re_flag = EXIT_FAILURE;
//            return dicom_info;
//        }
//
//        if (strcmp(tagcolor_val.c_str(), "1") == 0)
//        {
//            dicom_info.re_flag = EXIT_FAILURE;
//            return dicom_info;
//        }
//        
//
//        // 判断是否为DICOM RGB格式数据
//        itk::ImageIOBase::SizeType imgpixnum;
//        itk::ImageIOBase::SizeType imgbytenum;
//        itk::CommonEnums::IOPixel pixelType;
//        try
//        {
//            imgpixnum = gdcmIO->GetImageSizeInPixels();
//            imgbytenum = gdcmIO->GetImageSizeInBytes();
//            pixelType = gdcmIO->GetPixelType();
//        }
//        catch (itk::ExceptionObject& ex)
//        {
//            dicom_info.re_flag = EXIT_FAILURE;
//            return dicom_info;
//        }
//
//        if (pixelType != itk::CommonEnums::IOPixel::RGB)
//        {
//            dicom_info.re_flag = EXIT_FAILURE;
//            return dicom_info;
//        }
//        
//
//
//        // 通过TAG"0008|0018"获取InstanceUID
//        // 若无TAG则直接不进行处理
//        std::string taguid_val;
//        std::string uidtag = "0008|0018";
//        if (_findTagVal(gdcmIO, uidtag, taguid_val) == EXIT_FAILURE)
//        {
//            dicom_info.re_flag = EXIT_FAILURE;
//            return dicom_info;
//        }
//
//
//        //// 通过TAG"0020|000D"获取StInstanceUID
//        //// 若无TAG则直接不进行处理
//        //std::string tagstuid_val;
//        //std::string stuidtag = "0020|000D";
//        ////if (_findTagVal(gdcmIO, stuidtag, tagstuid_val) == EXIT_FAILURE)
//        ////{
//        ////    dicom_info.re_flag = EXIT_FAILURE;
//        ////    return dicom_info;
//        ////}
//        //try
//        //{
//        //    tagstuid_val = static_cast<std::string>(gdcmIO->GetStudyInstanceUID());
//        //}
//        //catch (itk::ExceptionObject& ex)
//        //{
//        //    dicom_info.re_flag = EXIT_FAILURE;
//        //    return dicom_info;
//        //}
//
//        
//        // 处理DICOM RGB格式数据
//        PixelType* ptr = reader->GetOutput()->GetBufferPointer();                                  // 获取数据缓存区指针
//        ImageType::SizeType img_size = reader->GetOutput()->GetLargestPossibleRegion().GetSize();  // 获取图像尺寸
//        long int frame_num = img_size[2];
//        long int rows = img_size[1];
//        long int cols = img_size[0];
//        long int frame_pixs = rows * cols;
//        // 小于4帧默认为非视频图像
//        if (frame_num<4)
//        {
//            dicom_info.re_flag = EXIT_FAILURE;
//            return dicom_info;
//        }
//
//        int frame_count = 0;
//        cv::Mat img_bgr = cv::Mat(cv::Size(cols, rows), CV_8UC3, cv::Scalar(0, 0, 0));
//        cv::Mat img_gray = cv::Mat(cv::Size(cols, rows), CV_8U, cv::Scalar(0));
//        for (int k = 0; k < frame_num; k++) {
//            for (long int j = 0; j < rows; j++) {
//                for (long int i = 0; i < cols; i++) {
//                    // ITK RGB图像需要利用 itk::RGBPixel 实现RGB图像像素色彩成分的访问
//                    PixelType ptr_val = ptr[k*frame_pixs + j*cols + i];
//                    img_bgr.at<cv::Vec3b>(j, i)[0] = static_cast<uchar>(ptr_val.GetBlue());  // blue
//                    img_bgr.at<cv::Vec3b>(j, i)[1] = static_cast<uchar>(ptr_val.GetGreen()); // green
//                    img_bgr.at<cv::Vec3b>(j, i)[2] = static_cast<uchar>(ptr_val.GetRed());   // red
//                }
//            }
//
//            // 检查数组是否为空
//            if (img_bgr.empty()) {
//                break;
//            }
//
//            // 记得不要取消.clone()，push_back是浅拷贝
//            cv::cvtColor(img_bgr.clone(), img_gray, cv::COLOR_BGR2GRAY);
//            frames.push_back(img_bgr.clone());         
//            frames_gray.push_back(img_gray.clone());
//            frame_count++;
//        }
//        totalCount = frame_count;
//
//        //// ITK高版本中使用了智能指针，因此不再需要手动释放
//        //gdcmIO->Delete();
//        //reader->Delete();
//
//        // 一切正常情况下返回EXIT_SUCCESS
//        dicom_info.re_flag = EXIT_SUCCESS;
//        dicom_info.dicomInfos.push_back(taguid_val);
//        //dicom_info.dicomInfos.push_back(tagstuid_val);
//        return dicom_info;
//
//    }
//    catch (itk::ExceptionObject& e)
//    {
//        dicom_info.re_flag = EXIT_FAILURE;
//        return dicom_info;
//    }
//}



//int 
//utils::_gettagidx(itk::GDCMImageIO::Pointer gdcmIO, std::string colortag, DictionaryType::Iterator& tag_idx)
//{
//    DictionaryType dictionary;
//    try
//    {
//        dictionary = gdcmIO->GetMetaDataDictionary();
//    }
//    catch (itk::ExceptionObject& ex)
//    {
//        return EXIT_FAILURE;
//    }
//    try
//    {
//        tag_idx = dictionary.Find(colortag);
//    }
//    catch (std::exception& err)
//    {
//        return EXIT_FAILURE;
//    }
//    if (tag_idx == dictionary.End())
//    {
//        return EXIT_FAILURE;
//    }
//
//    // 一切正常情况下返回EXIT_SUCCESS
//    return EXIT_SUCCESS;
//}
//
//
//int 
//utils::_findTagVal(itk::GDCMImageIO::Pointer gdcmIO, std::string tagTag, std::string& tag_val)
//{
//    DictionaryType dictionary;
//    MetaDataStringType::Pointer tag_point; // 注意ITK中数字类型的值同样以字符形式存放
//    DictionaryType::Iterator tag_idx;
//    if (_gettagidx(gdcmIO, tagTag, tag_idx) == EXIT_FAILURE)
//    {
//        return EXIT_FAILURE;
//    }
//
//    try
//    {
//        tag_point = dynamic_cast<MetaDataStringType*>(tag_idx->second.GetPointer());
//        if (tag_point != nullptr)
//        {
//            try
//            {
//                tag_val = static_cast<std::string>(tag_point->GetMetaDataObjectValue());
//            }
//            catch (itk::ExceptionObject& ex)
//            {
//                return EXIT_FAILURE;
//            }
//        }
//        else
//        {
//            return EXIT_FAILURE;
//        }
//    }
//    catch (itk::ExceptionObject& ex)
//    {
//        return EXIT_FAILURE;
//    }
//
//    return EXIT_SUCCESS;
//}


// [line_left, line_right](line_left=[rho, theta]) 转换为直角坐标系[[x1,y1,x2,y2],[x3,y3,x4,y4]]
std::vector <cv::Vec4i> utils::rho_theta_2_x_y(std::vector<cv::Vec2f> lines_in, int img_size)
{
	std::vector <cv::Vec4i> lines_out;
    cv::Vec4i line_points;
	for (int line_idx = 0; line_idx < lines_in.size(); line_idx++)
	{
        if (lines_in[line_idx][1] > 0)
        {
            float rho = lines_in[line_idx][0];
            float theta = lines_in[line_idx][1];
            float a = cos(theta);
            float b = sin(theta);
            float x0 = a * rho;
            float y0 = b * rho;
            int x1 = int(x0 + float(img_size) * (-b));
            int y1 = int(y0 + float(img_size) * (a));
            int x2 = int(x0 - float(img_size) * (-b));
            int y2 = int(y0 - float(img_size) * (a));
            line_points[0] = x1;
            line_points[1] = y1;
            line_points[2] = x2;
            line_points[3] = y2;
            lines_out.push_back(line_points);
        }

	}

	return lines_out;
}


//////////////////////////////////////////////// 帮助性函数 ////////////////////////////////////////////////

/////////////////////////////////////
//// 参考张奎老师DICOM文件处理，考虑仅使用std进行的修改
//char* utils::ParseDcmFile(std::string csFileName)
//{
//    FILE* pFile = NULL;
//    errno_t err = fopen_s(&pFile, csFileName.c_str(), "rb");
//    if (err != 0)
//    {
//        return NULL;
//    }
//
//    char newbuffer[8192] = { 0 };
//    char* byData = newbuffer;
//    if (pFile != NULL)
//    {
//        int nSize = fread(byData, 1, 8192, pFile);
//        fclose(pFile);
//        if (nSize < 8192)
//        {
//            return NULL;
//        }
//
//        return byData;
//    }
//
//    return NULL;
//}

//std::string utils::GetElementValue(char* pHeader, WORD wG, WORD wE)
//{
//    for (int i = 128; i < 8192; i += 2)
//    {
//        if (pHeader[i] == char(wG) && pHeader[i + 1] == char(wG >> 8) &&
//            pHeader[i + 2] == char(wE) && pHeader[i + 3] == char(wE >> 8))
//        {
//            int nVaLen = pHeader[i + 7] * 255 + pHeader[i + 6];
//            char	byXfer[128] = { 0 };
//            memcpy(byXfer, pHeader + i + 8, nVaLen);
//
//            std::string csXfer;
//            csXfer = byXfer;
//
//            return csXfer;
//        }
//    }
//
//    return "";
//}
////////////////////////////////////////////////////


// 求两直线交点-sub
cv::Vec3f _calc_abc_from_line_2d(cv::Vec4i line)
{
	cv::Vec3f a_b_c;
	float x0 = line[0];
	float y0 = line[1];
	float x1 = line[2];
	float y1 = line[3];
	a_b_c[0] = y0 - y1;
	a_b_c[1] = x1 - x0;
	a_b_c[2] = x0 * y1 - x1 * y0;

	return a_b_c;
}

// 求两直线交点
cv::Point utils::get_line_cross_point(cv::Vec4i line1, cv::Vec4i line2)
{
	cv::Point point;
	cv::Vec3f a_b_c1 = _calc_abc_from_line_2d(line1);
	cv::Vec3f a_b_c2 = _calc_abc_from_line_2d(line2);
	int a0 = a_b_c1[0];
	int b0 = a_b_c1[1];
	int c0 = a_b_c1[2];
	int a1 = a_b_c2[0];
	int b1 = a_b_c2[1];
	int c1 = a_b_c2[2];
	float D = a0 * b1 - a1 * b0;
	if (D == 0)
		return point;
	float x = (b0 * c1 - b1 * c0) / D;
	float y = (a1 * c0 - a0 * c1) / D;
	point.x = x;
	point.y = y;

	return point;
}

// 求最大连通域的contour
std::vector<cv::Point> utils::findMaxArea(std::vector<std::vector<cv::Point>> contours) {
	int temp[2];
	temp[0] = static_cast<int>(cv::contourArea(contours[0]));
    temp[1] = 0;
	for (int i = 1; i < contours.size(); i++) {
        int currContourArea = static_cast<int>(cv::contourArea(contours[i]));
		if (currContourArea > temp[0]) {
			temp[0] = currContourArea;
			temp[1] = i;
		}
	}
	return contours[temp[1]];
}

// 延长直线
cv::Vec4i utils::_extend_line(cv::Vec4i line, int cols, int rows, int basepix, int flag)
{
	cv::Vec4i line_out;
	float x1 = float(line[0]);
	float y1 = float(line[1]);
	float x2 = float(line[2]);
	float y2 = float(line[3]);
	int x = cols - 1 - basepix;
	int y = rows - 1 - basepix;
	if (flag == 1)
	{
		if (abs(y1 - y2) < 0.00001)
		{
			line_out[0] = basepix;
			line_out[1] = y1;
			line_out[2] = x;
			line_out[3] = y2;
		}
		else
		{
			float k = float(y2 - y1) / float(x2 - x1);
			float b = float(x1 * y2 - x2 * y1) / float(x1 - x2);
			line_out[0] = basepix;
			line_out[1] = int(b);
			line_out[2] = x;
			line_out[3] = int(k * line_out[2] + b);
		}
	}
	else
	{
		if (abs(y1 - y2) < 0.00001)
		{
			line_out[0] = x1;
			line_out[1] = basepix;
			line_out[2] = x2;
			line_out[3] = y;
		}
		else
		{
			float k = float(y2 - y1) / float(x2 - x1);
			float b = float(x1 * y2 - x2 * y1) / float(x1 - x2);
			line_out[1] = basepix;
			line_out[0] = int(-1 * b / k);
			line_out[3] = y;
			line_out[2] = int((line_out[3] - b) / k);
		}
	}

	return line_out;
}
