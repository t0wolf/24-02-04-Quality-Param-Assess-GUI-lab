#include "progress_super_thread.h"

ProgressSuperThread::ProgressSuperThread(QObject *parent)
{

}

// void ProgressSuperThread::run()
// {
// 	while (true)
// 	{
// 		// update progress
// 		if (m_bIsStatusUpdate)
// 		{
// 			QMutexLocker locker(&m_mutex);
// 			emit uiProgressUpdateAvailable(m_updateName);
			
// 			m_updateName.clear();
// 			m_bIsStatusUpdate = false;
// 		}
		
// 	}
// }

void ProgressSuperThread::setParamList(rapidjson::Document& docParamEvents)
{
    if (docParamEvents.IsObject())
    {
        for (auto it = docParamEvents.MemberBegin(); it != docParamEvents.MemberEnd(); it++)
        {
            m_paramProgressMap[QString::fromUtf8(it->name.GetString())] = false;
        }
    }
}

void ProgressSuperThread::setProgressMapUpdate(const QString name)
{
    QMutexLocker locker(&m_mutex);
    if (m_mParamAssessProgress.contains(name))
    {
        m_mParamAssessProgress[name] = true;
    }
    else
    {
        m_mQualityControlProgress[name] = true;
    }
    m_updateName = name;
    m_bIsQualityUpdate = true;
}

void ProgressSuperThread::run()
{
    while (!isInterruptionRequested())
    {
        if (m_bIsQualityUpdate)
        {
            QMutexLocker locker(&m_mutex);
            emit viewNameImageAvailable(m_currentViewName, m_currentImage);
            // for recording latest view.
            m_prevViewName = m_currentViewName;
            m_prevImage = m_currentImage;

            m_bIsQualityUpdate = false;
            // m_currentImage = QImage();
            // m_currentViewName = "";
        }

        if (m_bIsParamAssessUpdate)
        {
            QMutexLocker locker(&m_mutex);
            if (m_currentViewName == QString("PLAX"))
                emit sigStructParamValuePremiumsAvailable(m_structParamValues, m_currentParamPremiums);
            else
                emit sigParamValuePremiumsAvailable(m_currentViewName, m_currentParamValues, m_currentParamPremiums);
            // emit paramValuesAvailable(m_currentViewName, m_currentParamValues);
            // for recording latest view.
            m_prevViewName = m_currentViewName;
            m_prevImage = m_currentImage;
            m_prevParamValues = m_currentParamValues;

            m_bIsParamAssessUpdate = false;
            m_currentParamValues = QVariant();
            m_currentParamPremiums = QVariant();
            m_currentViewName = "";
        }

        // if (!m_bIsParamAssessUpdate || !m_bIsQualityUpdate)
        // {
        //     if (!m_currentViewName.isEmpty() && !m_currentImage.isNull())
        //     {
        //         emit viewNameImageAvailable(m_currentViewName, m_currentImage);
        //     }
        //     // else
        //     // {
        //     //     emit viewNameImageAvailable(m_prevViewName, m_prevImage);
        //     // }

        // }
        QThread::sleep(1);
    }
}

void ProgressSuperThread::setCurrentViewNameImage(const QString viewName, QImage image)
{
    QMutexLocker locker(&m_mutex);
    m_bIsQualityUpdate = true;
    m_currentImage = image;
    m_currentViewName = viewName;
}

void ProgressSuperThread::setCurrentViewNameVideo(const QString viewName, QVariant qVar)
{
    QMutexLocker locker(&m_mutex);
    m_bIsQualityUpdate = true;
    m_currentQualityScores = qVar;
    m_currentViewName = viewName;
}

void ProgressSuperThread::setCurrentViewName(const QString viewName)
{
    QMutexLocker locker(&m_mutex);
    m_bIsQualityUpdate = true;
    m_currentViewName = viewName;
}

void ProgressSuperThread::setCurrentParam(const QString viewName, QVariant qVar)
{
    QMutexLocker locker(&m_mutex);
    m_bIsParamAssessUpdate = true;
    m_currentViewName = viewName;
    m_currentParamValues = qVar;
}

void ProgressSuperThread::setCurrentParamValuePics(QString viewName, QVariant qValues, QVariant qPremium)
{
    QMutexLocker locker(&m_mutex);
    m_bIsParamAssessUpdate = true;
    m_currentViewName = viewName;
    m_currentParamValues = qValues;
    m_currentParamPremiums = qPremium;
}

void ProgressSuperThread::setStructParamValuePics(QVariant qValues, QVariant qPremium)
{
    QMutexLocker locker(&m_mutex);
    m_bIsParamAssessUpdate = true;
    m_currentViewName = QString("PLAX");
    m_structParamValues = qValues;
    m_currentParamPremiums = qPremium;
}
