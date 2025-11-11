#include "video_stream_thread.h"
#include <QDebug>

void VideoStreamThread::exitThread()
{
    this->requestInterruption();
    this->quit();
    this->wait();

    deInitFFmpegCtx();
}

int VideoStreamThread::inputImageToThreads(cv::Mat& image)
{
    m_roiScaleDetThread->inputVideoFrame(image);
    m_modelsInferThread->inputVideoFrame(image);
    m_infoExtractThread->inputVideoFrame(image);
    return 1;
}

int VideoStreamThread::initFFmpegCtx()
{
    m_pFrame = av_frame_alloc();
    if (!m_pFrame)
    {
        qDebug() << "Could not allocate frame.";
        return 0;
    }

    m_packet = av_packet_alloc();

    QString videoSize = QString("%1x%2").arg(QString::fromStdString(m_captureWidth)).arg(QString::fromStdString(m_captureHeight));
    //QString videoSize = QString("1920x1080");

    AVDictionary* options = nullptr;
    av_dict_set(&options, "video_size", videoSize.toStdString().c_str(), 0); // 设置视频分辨率
    av_dict_set(&options, "framerate", m_frameFPS.c_str(), 0); // 设置帧率
    av_dict_set(&options, "verbose", "true", 0);
    av_dict_set(&options, "input_format", "mjpeg", 0);

    //std::string videoFilePath = "D:/OBS Recording/2024-05-22 16-39-47.mkv"; // 设置视频文件路径
    //int ret = avformat_open_input(&m_pFormatCtx, videoFilePath.c_str(), NULL, &options);

    const AVInputFormat* inputFormat = av_find_input_format("dshow");

    int ret = -1;
    QFileInfo fileInfo(QString::fromStdString(m_camOrder));
    if (fileInfo.exists() && (fileInfo.isFile() || fileInfo.isDir()))
    {
        ret = avformat_open_input(&m_pFormatCtx, m_camOrder.c_str(), nullptr, nullptr);
        m_delay = 30;
    }
    else
    {
        std::string deviceName;
        if (m_camOrder == "desktop")
        {
            inputFormat = av_find_input_format("gdigrab");
            deviceName = "desktop";
        }
        else
            deviceName = QString("video=%1").arg(QString::fromStdString(m_camOrder)).toStdString();

        ret = avformat_open_input(&m_pFormatCtx, deviceName.c_str(), inputFormat, &options);
    }

    if (ret != 0) {
        char errbuf[128];
        av_strerror(ret, errbuf, sizeof(errbuf));
        qDebug() << "Couldn't open video stream:" << errbuf;
        return 0;
    }

    if (avformat_find_stream_info(m_pFormatCtx, NULL) < 0) {
        qDebug() << "Couldn't find stream information.";
        return 0;
    }

    for (unsigned int i = 0; i < m_pFormatCtx->nb_streams; i++) {
        if (m_pFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            m_ffmpegVideoStreamIdx = i;
            break;
        }
    }

    if (m_ffmpegVideoStreamIdx == -1) {
        qDebug() << "Didn't find a video stream.";
        return 0;
    }

    AVCodecParameters* pCodecParams = m_pFormatCtx->streams[m_ffmpegVideoStreamIdx]->codecpar;
    const AVCodec* pCodec = avcodec_find_decoder(pCodecParams->codec_id);
    if (pCodec == NULL) {
        qDebug() << "Unsupported codec!";
        return 0;
    }

    m_pCodecCtx = avcodec_alloc_context3(pCodec);
    if (avcodec_parameters_to_context(m_pCodecCtx, pCodecParams) < 0) {
        qDebug() << "Couldn't copy codec context.";
        return 0;
    }

    if (avcodec_open2(m_pCodecCtx, pCodec, NULL) < 0) {
        qDebug() << "Couldn't open codec.";
        return 0;
    }

    m_sws_ctx = sws_getContext(m_pCodecCtx->width, m_pCodecCtx->height, m_pCodecCtx->pix_fmt,
        m_pCodecCtx->width, m_pCodecCtx->height, AV_PIX_FMT_RGB24,
        SWS_BILINEAR, NULL, NULL, NULL);

    //if (initFilters() < 0) {
    //    return -1;
    //}

    return 1;
}

int VideoStreamThread::deInitFFmpegCtx()
{
    if (m_packet) {
        av_packet_free(&m_packet);
        m_packet = nullptr;
    }

    if (m_pFrame) {
        av_frame_free(&m_pFrame);
        m_pFrame = nullptr;
    }

    if (m_pCodecCtx) {
        avcodec_free_context(&m_pCodecCtx);
        m_pCodecCtx = nullptr;
    }

    if (m_pFormatCtx) {
        avformat_close_input(&m_pFormatCtx);  // 同时释放 fmt_ctx
        m_pFormatCtx = nullptr;
    }

    // 反初始化网络模块
    //avformat_network_deinit();
    return 1;
}

int VideoStreamThread::initFilters()
{
    filterGraph = avfilter_graph_alloc();
    if (!filterGraph) {
        QtLogger::instance().logMessage(QString("Could not allocate filter graph."));
        return -1;
    }

    // Create buffer source filter
    const AVFilter* buffersrc = avfilter_get_by_name("buffer");
    const AVFilter* buffersink = avfilter_get_by_name("buffersink");
    const AVFilter* eq = avfilter_get_by_name("eq");

    char args[512];
    snprintf(args, sizeof(args), "video_size=%dx%d:pix_fmt=%d:time_base=1/25:pixel_aspect=0",
        m_pCodecCtx->width, m_pCodecCtx->height, m_pCodecCtx->pix_fmt);
    //snprintf(args, sizeof(args),
    //    "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
    //    m_pCodecCtx->width, m_pCodecCtx->height, m_pCodecCtx->pix_fmt,
    //    m_pCodecCtx->time_base.num, m_pCodecCtx->time_base.den,
    //    m_pCodecCtx->sample_aspect_ratio.num, m_pCodecCtx->sample_aspect_ratio.den);

    // Create input filter
    int ret = avfilter_graph_create_filter(&buffersrc_ctx, buffersrc, "in", args, nullptr, filterGraph);
    if (ret < 0) {
        QtLogger::instance().logMessage(QString("Could not create buffer source filter."));
        return ret;
    }

    // Create output filter
    ret = avfilter_graph_create_filter(&buffersink_ctx, buffersink, "out", nullptr, nullptr, filterGraph);
    if (ret < 0) {
        QtLogger::instance().logMessage(QString("Could not create buffer sink filter."));
        return ret;
    }

    // Create eq filter for contrast adjustment
    // 确保对比度值在有效范围内
    float contrast = std::stof(m_contrast);
    if (contrast < 0.0f || contrast > 10.0f) {
        QtLogger::instance().logMessage("Contrast value is out of range.");
        return -1;
    }
    snprintf(args, sizeof(args), "brightness=0.1:contrast=%.2f", contrast);
    ret = avfilter_graph_create_filter(&eq_ctx, eq, "eq", args, nullptr, filterGraph);
    if (ret < 0) {
        QtLogger::instance().logMessage(QString("Could not create eq filter."));
        return ret;
    }

    // Link filters: buffersrc -> eq -> buffersink
    //avfilter_link(buffersrc_ctx, 0, eq_ctx, 0);
    //avfilter_link(eq_ctx, 0, buffersink_ctx, 0);
    ret = avfilter_link(buffersrc_ctx, 0, eq_ctx, 0);
    if (ret < 0) {
        QtLogger::instance().logMessage("Could not link buffersrc to eq filter.");
        return ret;
    }

    ret = avfilter_link(eq_ctx, 0, buffersink_ctx, 0);
    if (ret < 0) {
        QtLogger::instance().logMessage("Could not link eq to buffersink filter.");
        return ret;
    }

    // Configure the filter graph
    ret = avfilter_graph_config(filterGraph, NULL);
    if (ret < 0) {
        QtLogger::instance().logMessage(QString("Could not configure filter graph."));
        return ret;
    }

    return 0;
}

int VideoStreamThread::processFrame()
{
    // Send the packet to the decoder
    int ret = avcodec_send_packet(m_pCodecCtx, m_packet);
    if (ret < 0) {
        QtLogger::instance().logMessage(QString("Failed to send packet, error code: %1").arg(ret));
        return -1;
    }

    // Receive the decoded frame
    ret = avcodec_receive_frame(m_pCodecCtx, m_pFrame);
    if (ret < 0) {
        QtLogger::instance().logMessage(QString("Failed to receive frame, error code:").arg(ret));
        return -1;
    }

    // Check if the decoded frame is valid
    if (m_pFrame->width <= 0 || m_pFrame->height <= 0) {
        QtLogger::instance().logMessage(QString("Decoded frame is invalid (empty)."));
        return -1;
    }

    //// 创建 QImage 并发送信号
    int width = m_pCodecCtx->width;
    int height = m_pCodecCtx->height;

    QImage img(width, height, QImage::Format_RGB888);

    uint8_t* dest[4] = { img.bits(), nullptr, nullptr, nullptr };
    int dest_linesize[4] = { img.bytesPerLine(), 0, 0, 0 };
    sws_scale(m_sws_ctx, m_pFrame->data, m_pFrame->linesize, 0, height, dest, dest_linesize);
    //sws_scale(m_sws_ctx, filteredFrame->data, filteredFrame->linesize, 0, height, dest, dest_linesize);

    // Resize the QImage to the desired dimensions
    int currHeight = std::stoi(m_frameHeight);
    int currWidth = std::stoi(m_frameWidth);
    float fContrastAlpha = std::stof(m_contrast);
    int nBrightBeta = std::stoi(m_bright);

    img = img.scaled(currWidth, currHeight);
    //img = ImageProcess::adjustContrast(img, fContrastAlpha, nBrightBeta);
    // Emit the processed frame
    QImage qInferImage = img.copy();
    cv::Mat currFrame = GeneralUtils::qImage2cvMat(qInferImage);
    if (fContrastAlpha != 1.0 && nBrightBeta != 0)
        currFrame.convertTo(currFrame, CV_8UC3, fContrastAlpha, nBrightBeta);

    if (!currFrame.empty()) {
        inputImageToThreads(currFrame);
    }

    emit frameAvailable(img);
    //cv::imshow("test", currMat);
    //cv::waitKey(0);

    // Sleep for a specified delay
    QThread::msleep(m_delay);
    return 1;
}

bool VideoStreamThread::isMotionDetected(cv::Mat& currFrame)
{
    if (m_prevFrame.empty())
    {
        m_prevFrame = currFrame.clone();
        return false;
    }
    else
    {
        if (currFrame.size() != m_prevFrame.size())
            return false;
        else
        {
            cv::Mat diff;
            cv::absdiff(currFrame, m_prevFrame, diff);
            if (diff.channels() > 1)
            {
                cv::cvtColor(diff, diff, cv::COLOR_BGR2GRAY);
            }

            cv::threshold(diff, diff, 25, 255, cv::THRESH_BINARY);
            int nonZeroCount = cv::countNonZero(diff);

            return nonZeroCount > m_fMotionThreashold;
        }
    }
    return false;
}

VideoStreamThread::VideoStreamThread(QObject* parent, 
    ROIScaleDetThread* roiScaleDetThread, 
    ModelsInferenceThread* modelsInferenceThread, 
    InfoExtractThread* infoExtractThread,
    ConfigParse* config)
    : m_roiScaleDetThread(roiScaleDetThread)
    , m_modelsInferThread(modelsInferenceThread)
    , m_infoExtractThread(infoExtractThread)
    , m_pFormatCtx(nullptr)
    , m_pCodecCtx(nullptr)
    , m_pFrame(nullptr)
    , m_packet(nullptr)
    , m_sws_ctx(nullptr)
    , m_ffmpegVideoStreamIdx(-100)
{
    avdevice_register_all();
    avformat_network_init();

    std::string camOrder = "";
    std::string frameWidth = "";
    std::string frameHeight = "";
    std::string frameFPS = "";
    std::string contrast = "";
    std::string bright = "";

    bool ret = config->getSpecifiedNode("OFFLINE_VIDEO_ID", camOrder);
    if (ret)
        m_camOrder = camOrder;
    else
        m_camOrder = "0";

    ret = config->getSpecifiedNode("FRAME_WIDTH", frameWidth);
    if (ret)
        m_frameWidth = frameWidth;
    else
        m_frameWidth = "1080";

    ret = config->getSpecifiedNode("FRAME_HEIGHT", frameHeight);
    if (ret)
        m_frameHeight = frameHeight;
    else
        m_frameHeight = "1920";

    ret = config->getSpecifiedNode("CAPTURE_WIDTH", frameWidth);
    if (ret)
        m_captureWidth = frameWidth;
    else
        m_captureWidth = "1920";

    ret = config->getSpecifiedNode("CAPTURE_HEIGHT", frameHeight);
    if (ret)
        m_captureHeight = frameHeight;
    else
        m_captureHeight = "1080";

    ret = config->getSpecifiedNode("FRAME_FPS", frameFPS);
    if (ret)
        m_frameFPS = frameFPS;
    else
        m_frameFPS = "30";

    ret = config->getSpecifiedNode("FRAME_CONTRAST", contrast);
    if (ret)
        m_contrast = contrast;
    else
        m_contrast = "1.0";

    ret = config->getSpecifiedNode("FRAME_BRIGHT", bright);
    if (ret)
        m_bright = bright;
    else
        m_bright = "0";

}

void VideoStreamThread::run()
{   
    m_running = true;

    if (!m_localVideoFilePath.empty())
    {
        cv::VideoCapture cap(m_localVideoFilePath);
        while (!isInterruptionRequested())
        {
            cv::Mat currFrame;
            bool ret = cap.read(currFrame);
            if (!ret || currFrame.empty())
                break;
            
            QImage img = GeneralUtils::matToQImage(currFrame);
            float fContrastAlpha = std::stof(m_contrast);
            int nBrightBeta = std::stoi(m_bright);

            QImage qInferImage = img.copy();
            currFrame = GeneralUtils::qImage2cvMat(qInferImage);
            if (fContrastAlpha != 1.0 && nBrightBeta != 0)
                currFrame.convertTo(currFrame, CV_8UC3, fContrastAlpha, nBrightBeta);

            if (!currFrame.empty()) {
                inputImageToThreads(currFrame);
            }

            emit frameAvailable(img);
            //cv::imshow("test", currMat);
            //cv::waitKey(0);

            // Sleep for a specified delay
            QThread::msleep(30);
            
        }
    }

    else
    {
        try
        {
            initFFmpegCtx();

            while (!isInterruptionRequested())
            {
                int ret = av_read_frame(m_pFormatCtx, m_packet);
                if (ret < 0) {
                    qDebug() << "Failed to read frame, error code:" << ret;

                    // 检查是否是流的结束或其他严重错误
                    if (ret == AVERROR_EOF || ret == AVERROR(EIO)) {
                        //QMessageBox::warning(nullptr, "Stream Error",
                        //    "End of stream or IO error occurred.");
                        break; // 在流结束或 IO 错误时中止循环
                    }
                    else {
                        continue; // 对于其他错误，尝试继续读取下一帧
                    }
                }

                if (m_packet->stream_index == m_ffmpegVideoStreamIdx)
                {
                    processFrame();
                }
                av_packet_unref(m_packet);
            }
        }
        catch (const std::exception& e)
        {
            QMessageBox::warning(nullptr, "Exception",
                QString("An unexpected error occurred: %1").arg(e.what()));
        }
    }

    m_running = false;
    //int videoID = -1;
    //try
    //{
    //    videoID = std::stoi(m_camOrder);
    //}
    //catch (const std::invalid_argument& e)
    //{
    //    qDebug() << "[I] Video debug mode.";
    //}

    //if (videoID >= 0)
    //    m_cap.open(videoID);
    //else
    //    m_cap.open(m_camOrder);

    //if (!m_frameWidth.empty())
    //{
    //    int frameWidth = 0;
    //    try
    //    {
    //        frameWidth = std::stoi(m_frameWidth);
    //        if (frameWidth > 0)
    //            m_cap.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<float>(frameWidth));
    //    }
    //    catch (std::invalid_argument& e) {
    //        qDebug() << "Width not specified";
    //    }
    //}

    //if (!m_frameHeight.empty())
    //{
    //    int frameHeight = 0;
    //    try
    //    {
    //        frameHeight = std::stoi(m_frameHeight);
    //        if (frameHeight > 0)
    //            m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<float>(frameHeight));
    //    }
    //    catch (std::invalid_argument& e) {
    //        qDebug() << "Height not specified";
    //    }
    //}

    //if (!m_frameFPS.empty())
    //{
    //    int frameFPS = 0;
    //    try
    //    {
    //        frameFPS = std::stoi(m_frameFPS);
    //        if (frameFPS > 0)
    //            m_cap.set(cv::CAP_PROP_FPS, static_cast<float>(frameFPS));
    //    }
    //    catch (std::invalid_argument& e) {
    //        qDebug() << "FPS not specified";
    //    }
    //}

    //if (!m_cap.isOpened())
    //{
    //    qDebug() << "[E] Failed to open video capture";
    //    return;
    //}

    //while (!isInterruptionRequested())
    //{
    //    cv::Mat frame;

    //    int ret = m_cap.read(frame);
    //    if (ret)
    //    {
    //        //bool isMotionFlag = isMotionDetected(frame);
    //        //qDebug() << "[I] isMotion: " << isMotionFlag;
    //        inputImageToThreads(frame);

    //        QImage image(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
    //        image = image.rgbSwapped();  // 颜色通道顺序转换

    //        emit frameAvailable(image);  // 发送图像帧信号
    //        ++m_counter;
    //        if (m_counter > 1024)
    //            m_counter = 0;

    //        //if (m_counter % 15 == 0)
    //        //    emit sigInfoFrameAvailable(image);

    //        m_prevFrame = frame.clone();
    //        QThread::msleep(1);
    //    }
    //    else
    //    {
    //        QThread::msleep(1);
    //    }
    //}

    //std::cout << "[I] Video thread stopped.\n";
    //m_cap.release();
}
