#ifndef VIDEOSTREAMTHREAD_H
#define VIDEOSTREAMTHREAD_H
#include <QWidget>
#include <QThread>
#include <opencv2/opencv.hpp>
#include "config_parse.h"
#include "process_threads/ROIScaleDetThread.h"
#include "models_inference_thread.h"
#include "process_threads/InfoExtractThread.h"
#include "param_assessment/image_process.h"
#include <QMessageBox>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#include <libavdevice/avdevice.h>
#include <libavutil/time.h>
#include <libavfilter/avfilter.h>
//#include <libavfilter/avfiltergraph.h>
#include <libavfilter/buffersrc.h>
#include <libavfilter/buffersink.h>
}


class VideoStreamThread : public QThread
{
    Q_OBJECT

public:
    //VideoStreamThread(QObject* parent = nullptr,
    //    ROIScaleDetThread* roiScaleDetThread = nullptr);

    VideoStreamThread(QObject* parent = nullptr) : QThread(parent) {};

    VideoStreamThread(QObject* parent = nullptr, 
        ROIScaleDetThread* roiScaleDetThread = nullptr, 
        ModelsInferenceThread* modelsInferenceThread = nullptr, 
        InfoExtractThread* infoExtractThread = nullptr,
        ConfigParse* config = nullptr);

    ~VideoStreamThread()
    {
        exitThread();
        cleanupFFmpeg();
    }

    void run() override;
    void exitThread();

    inline int setVideoFilePath(std::string strCurrLocalFilePath)
    {
        m_localVideoFilePath = strCurrLocalFilePath;
        return 1;
    }

    inline bool getVideoCapStatus()
    {
        return m_running;
    }

private:
    int inputImageToThreads(cv::Mat& image);

    int initFFmpegCtx();

    int deInitFFmpegCtx();

    int initFilters();

    int processFrame();

    bool isMotionDetected(cv::Mat& currFrame);

    void cleanupFFmpeg() 
    {
        if (m_pFrame) {
            av_frame_free(&m_pFrame);
        }
        if (m_pCodecCtx) {
            avcodec_free_context(&m_pCodecCtx);
        }
        if (m_pFormatCtx) {
            avformat_close_input(&m_pFormatCtx);
        }
    }

private:
    cv::VideoCapture m_cap;
    AVFormatContext* m_pFormatCtx;
    AVCodecContext* m_pCodecCtx;
    AVFrame* m_pFrame;
    AVPacket* m_packet;
    SwsContext* m_sws_ctx;

    AVFilterGraph* filterGraph = nullptr;
    AVFilterContext* buffersrc_ctx = nullptr;
    AVFilterContext* buffersink_ctx = nullptr;
    AVFilterContext* eq_ctx = nullptr;

    int m_ffmpegVideoStreamIdx;

    std::string m_camOrder, m_localVideoFilePath;
    std::string m_frameFPS;
    std::string m_frameWidth, m_captureWidth;
    std::string m_frameHeight, m_captureHeight;
    std::string m_contrast, m_bright;

    int m_delay = 1;

    cv::Mat m_prevFrame;
    float m_fMotionThreashold = 100.0;

    QMutex m_mutex;

    ROIScaleDetThread* m_roiScaleDetThread;

    ModelsInferenceThread* m_modelsInferThread;

    InfoExtractThread* m_infoExtractThread;

    bool stopFlag = false;

    bool m_running = false;
    QString m_deviceName = "video=VC-007PRO";

    int m_counter = 0;

signals:
    void frameAvailable(const QImage &image);

    void sigInfoFrameAvailable(const QImage& image);

    void cvFrameAvailable(const cv::Mat &image);

};

#endif // VIDEOSTREAMTHREAD_H
