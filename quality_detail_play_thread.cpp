#include "quality_detail_play_thread.h"

// QualityDetailPlayThread::QualityDetailPlayThread() {}

void QualityDetailPlayThread::exitThread()
{
    this->requestInterruption();
    this->quit();
    this->wait();
}

void QualityDetailPlayThread::run()
{
    if (!m_videoBuffer.empty())
    {
        for (auto& image : m_videoBuffer)
        {
            if (!isInterruptionRequested())
            {
                QImage qImage = GeneralUtils::matToQImage(image);
                emit frameAvailable(qImage);
                QThread::msleep(30);
            }
            else
            {
                break;
            }
        }

         exitThread();
    }
}
