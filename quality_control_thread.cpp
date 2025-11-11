#include "quality_control_thread.h"

QualityControlThread::QualityControlThread(QObject *parent)
{

}

void QualityControlThread::run()
{
    while (!isInterruptionRequested())
    {
        singleResult currQualityResult;
        std::string currViewName = m_currViewName.toStdString();

        m_qualityControler->qualityAssess(m_currVideoClips, m_currKeyframeIdxes, currViewName, currQualityResult);
        QVariant qVar;
        qVar.setValue(currQualityResult);
        emit sigQualityScoresAvailable(qVar);
    }
}
