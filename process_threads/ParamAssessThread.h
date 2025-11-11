#pragma once

#include <QThread>
#include <QMutex>
#include <QMap>
#include <opencv2/opencv.hpp>

#include "param_assesser.h"
#include "type_registering.h"
#include "type_define.h"
#include "config_parse.h"


class ParamAssessThread  : public QThread
{
	Q_OBJECT

public:
	ParamAssessThread(QObject *parent = nullptr, ConfigParse* config = nullptr);
    ~ParamAssessThread()
    {
        exitThread();
    }

    void exitThread()
    {
        this->requestInterruption();
        this->quit();
        this->wait();
    }

    void run() override;

    int inputParamAssess(QString& viewName, QVariant& videoClip, QVariant& keyframeIdxes, QVariant& modeInfo);

    void performParamAssess(QString viewName, QVariant videoClip, QVariant keyframeIdxes, QVariant modeInfo);

    inline bool isParamEnable(QString viewName)
    {
        std::string tempViewName = viewName.toStdString();
        return m_paramAssesser->isParamEnable(tempViewName);
    }

    inline void clearLVEFDataCache()
    {
        m_paramAssesser->clearLVEFDataCache();
    }

public slots:
    void setParamAssessInput(QString viewName, QVariant videoBuffer, QVariant keyframeIdxes, QVariant modeInfo);

    void setPatientName(QString patientName);

    void setScaleInfo(QVariant qScaleInfo);

signals:
    void sigParamsResult(QString viewName, QVariant result, QVariant premiums);

private:
    ScaleInfo m_currScaleInfo;

    ParamAssesser* m_paramAssesser;

    QMutex m_mutex;

	QString m_currViewName;

	QVector<cv::Mat> m_vVideoClip;

	QVector<int> m_keyframeIdx;

    ModeInfo m_currModeInfo;

    bool m_bIsVideoClipsUpdate = false;

	int m_frameCounter = 0;

};
