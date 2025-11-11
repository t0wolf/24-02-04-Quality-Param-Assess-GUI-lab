#include "EchoQuality.h"

float getAvgHeight(std::vector<Object>& objects)
{
    if (objects.empty())
        return 0.0f;

    float fAvgHeight = 0.0f;
    int counter = 0;

    for (auto& object : objects)
    {
        fAvgHeight += object.rect.height;
        ++counter;
    }

    return fAvgHeight / static_cast<float>(counter);
}

std::vector<cv::Mat> getVideoFramesCropped(std::vector<cv::Mat>& vClips, cv::Rect& cropRect)
{
    std::vector<cv::Mat> vCroppedClips;

    for (auto& frame : vClips)
    {
        cv::Mat croppedFrame = frame(cropRect);
        vCroppedClips.push_back(croppedFrame);
    }
    return vCroppedClips;
}


s_f_map qualityAssessVideo(std::string viewname, std::vector<cv::Mat> croppedVideo, std::vector<int> keyframes, float fRadius)
{
    finalResult results;
    std::vector<cv::Mat> frames;
    std::vector<cv::Mat> frames_gray;//用于表示任意维度的稠密数组，OpenCV使用它来存储和传递图像
    frames = croppedVideo;
    std::vector<cv::Mat> frames_processed = croppedVideo;
    // ---------------------------------- ROI提取与模型构建 ---------------------------------- //
    // std::vector<float> radius;
    cv::Mat roi_mask, frames_roimask;
    std::vector<float> radiuses({ fRadius });
    s_f_map  result_map;

    singleResult result;
    result.label = viewname;//标签
    if (result.label == "A4C") {
        EchoQualityAssessmentParams params_a4c;
        EchoQualityAssessmentA4C echo_a4c(params_a4c);
        result_map = echo_a4c.predict(frames_processed, radiuses, result);
    }
    else if (result.label == "PSAXGV") {
        EchoQualityAssessmentParams params_psaxgv;
        EchoQualityAssessmentPSAXGV echo_psaxgv(params_psaxgv);

        result_map = echo_psaxgv.predict(frames_processed, radiuses, result);
    }
    // result.label = "PSAXGV";//自行魔改，原版需要先进行切面分类的预测，切面分类不在我这，我这直接赋值
    else if (result.label == "PSAXMV") {

        EchoQualityAssessmentParams params_psaxgv;
        EchoQualityAssessmentPSAXMV echo_psaxmv(params_psaxgv);

        result_map = echo_psaxmv.predict(frames_processed, radiuses, result);
    }
    else if (result.label == "PSAXPM")
    {
        EchoQualityAssessmentParams params_psaxpm;
        EchoQualityAssessmentPSAXPM echo_psaxpm(params_psaxpm);

        result_map = echo_psaxpm.predict(frames_processed, radiuses, result);
    }
    else if (result.label == "PSAXA")
    {
        EchoQualityAssessmentParams params_psaxa;
        EchoQualityAssessmentPSAXA echo_psaxa(params_psaxa);

        result_map = echo_psaxa.predict(frames_processed, radiuses, result);
    }
    else if (result.label == "PLAX")
    {
        EchoQualityAssessmentParams params_plax;
        EchoQualityAssessmentPLAX echo_plax(params_plax);

        result_map = echo_plax.predict(frames_processed, radiuses, result);
    }
    else if (result.label == "A2C")
    {
        EchoQualityAssessmentParams params_a2c;
        EchoQualityAssessmentA2C echo_a2c(params_a2c);
        result_map = echo_a2c.predict(frames_processed, radiuses, result);
    }
    else if (result.label == "A3C")
    {
        EchoQualityAssessmentParams params_a3c;
        EchoQualityAssessmentA3C echo_a3C(params_a3c);
        result_map = echo_a3C.predict(frames_processed, radiuses, result);
    }
    else if (result.label == "A5C")
    {
        EchoQualityAssessmentParams params_a5c;
        EchoQualityAssessmentA5C echo_a5c(params_a5c);
        result_map = echo_a5c.predict(frames_processed, radiuses, result);
    }

    return result_map;
}

s_f_map doQualityAssessVideo(std::string viewname, std::vector<cv::Mat> croppedVideo, std::vector<int> keyframes, float fRadius)
{
    std::vector<cv::Mat> vShortVideoClip;
    if (croppedVideo.empty())
        return s_f_map();

    if (croppedVideo.size() >= 3)
    {
        int stride = croppedVideo.size() / 3;
        for (int i = 0; i < croppedVideo.size(); i += stride)
            vShortVideoClip.push_back(croppedVideo[i]);
    }
    else
        vShortVideoClip = croppedVideo;

    // short clip quality assessment
    s_f_map shortQualityScores = qualityAssessVideo(viewname, vShortVideoClip, keyframes, fRadius);
    float fStructScore = shortQualityScores["structure_score"];
    if (fStructScore >= 4.0f)
    {
        s_f_map longQualityScores = qualityAssessVideo(viewname, croppedVideo, keyframes, fRadius);
        return longQualityScores;
    }
    
    return shortQualityScores;
}

std::vector<cv::Mat> video_mat(std::string videoPath) {

    finalResult results;
    std::vector<cv::Mat> frames;
    std::vector<cv::Mat> frames_gray;//用于表示任意维度的稠密数组，OpenCV使用它来存储和传递图像

    cv::VideoCapture cap;//从摄像机读取视频
    cap.open(videoPath);

    if (!cap.isOpened())
    {
        std::cout << "[E] Cannot open this video: " << videoPath << std::endl;
       // return results;
    }

    while (cap.isOpened())
    {
        cv::Mat frame;
        cv::Mat frameGray;
        int ret = cap.read(frame);//获取视频中的每一帧图像
        if (ret)
        {
            cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
            frames.push_back(frame);
            frames_gray.push_back(frameGray);//        cv::InputArray src, // 输入序列 cv::OutputArray dst, // 输出序列
               // int code, // 颜色映射码
               // int dstCn = 0 // 输出的通道数 (0='automatic')
        }
        else
        {
            break;
        }
    }
    return frames;
}

// int main()
// {
//     //std::string databaseFile = "../../samples_pred.xlsx";
//  //   std::string databaseFile = "D:/Drum_Echocardiography/Echocardiography_Codes/AIserverOffline/preds/samples_pred.xlsx";
//     //std::string databaseFile = "D:/Drum_Echocardiography/Echocardiography_Codes/AIserverOffline/preds/samples_pred_A3CA5C_extended.xlsx";
//  //   std::string databaseFile = "D:/Drum_Echocardiography/Echocardiography_Codes/AIserverOffline/preds/samples_pred_A4C.xlsx";
//     std::vector<std::string> vResultFileLists;
//     std::vector<finalResult> vResults;
//     std::string file_path = "D:/Drum_Echocardiography/Echocardiography_Datas/Quality_Assessment_AutoML/Datasets/PSAXMV/videos_original/USm.1.2.840.113619.2.391.3036.1667315942.289.1.512_SePACS.dcm.avi ";
//     //std::string file_path = "D:/Drum_Echocardiography/Echocardiography_Datas/Quality_Assessment_AutoML/Datasets/PSAXGV/videos/USm.1.2.840.113619.2.239.10202.1661932593.0.2424.512_SePACS.dcm.avi ";
//    // finalResult results = QualityAssessmentVideo(file_path);
    
//     //外部接口调用
//     std::string viewname;//切面类型
//     std::vector<cv::Mat> video;//视频
//     video= video_mat(file_path);
//     std::vector<int>keyframes;//关键帧
//     viewname = "PSAXMV";
//     keyframes.push_back(5);
//     finalResult results = QualityAssessmentVideo_port(viewname, video, keyframes);
// }
