#ifndef PROGRESSSUPERTHREAD_H
#define PROGRESSSUPERTHREAD_H

#include <QWidget>
#include <QThread>
#include <qmutex.h>
#include <QMap>
#include <QVariant>
#include <QString>
#include <rapidjson/document.h>
#include "type_registering.h"
#include "quality_control/typemappings.h"

//#include "quality_control/quality_control.h"

class ProgressSuperThread : public QThread
{
    Q_OBJECT

public:
    ProgressSuperThread(QObject *parent = nullptr);

    void run() override;

    void exitThread()
    {
        this->requestInterruption();
        this->quit();
        this->wait();
    }

    void setParamList(rapidjson::Document& docParamEvents);

    inline bool getViewProgress(QString sViewName)
    {
        return m_viewProgressMap[sViewName];
    }

    inline bool getParamProgress(QString sParamEventName)
    {
        return m_paramProgressMap[sParamEventName];
    }

public slots:
    void setProgressMapUpdate(const QString name);

    // void setViewNameImageUpdate(const QString viewName, QImage image);

    void setCurrentViewName(const QString viewName);

    void setCurrentViewNameImage(const QString viewName, QImage qImage);

    void setCurrentViewNameVideo(const QString viewName, QVariant qVar);

    void setCurrentParam(const QString viewName, QVariant qVar);

    void setCurrentParamValuePics(QString viewName, QVariant qValues, QVariant qPremium);

    void setStructParamValuePics(QVariant qValues, QVariant qPremium);

    inline bool isKeyframeEnable(std::string viewName)
    {
        return m_keyframeMap[viewName];
    }

    inline bool isQualityControlled(singleResult& videoResult)
    {
        return videoResult.totalGrade >= videoResult.totalGrade_th;
    }

    void setParamComplete(QString paramEventName)
    {
        m_paramProgressMap[paramEventName] = true;
    }

    void setParamUncomplete(QString paramEventName)
    {
        m_paramProgressMap[paramEventName] = false;
    }

    void setViewQualityComplete(QString viewName)
    {
        m_viewProgressMap[viewName] = true;
    }

    void setViewQualityUncomplete(QString viewName)
    {
        m_viewProgressMap[viewName] = false;
    }

    bool getViewQualityStatus(QString viewName)
    {
        return m_viewProgressMap[viewName];
    }

signals:
    void uiProgressUpdateAvailable(const QString name);

    void viewNameImageAvailable(const QString viewName, QImage qImage);

    void paramValuesAvailable(const QString viewName, QVariant qVar);

    void sigParamValuePremiumsAvailable(QString viewName, QVariant qValues, QVariant qPremium);

    void sigStructParamValuePremiumsAvailable(QVariant qValues, QVariant qPremium);

private:
    QMap<QString, bool> m_mQualityControlProgress;
    QMap<QString, bool> m_mParamAssessProgress;
    bool m_bIsQualityControl = false;
    bool m_bIsParamAssessUpdate = false;
    bool m_bIsQualityUpdate = false;

    QString m_updateName;

    QMutex m_mutex;

    QMap<QString, bool> m_viewProgressMap = {
                                             {"A2C", false},
                                             {"A3C", false},
                                             {"A4C", false},
                                             {"A5C", false},
                                             {"PLAX", false},
                                             {"PSAXA", false},
                                             {"PSAXGV", false},
                                             {"PSAXPM", false},
                                             {"PSAXMV", false},
                                            };

    QMap<QString, bool> m_paramProgressMap;

    std::map<std::string, bool> m_keyframeMap {
                                              {"A2C", true},
                                              {"A3C", true},
                                              {"A4C", true},
                                              {"A5C", true},
                                              {"PLAX", true},
                                              {"PSAXGV", false},
                                              {"PSAXPM", false},
                                              {"PSAXMV", false},
                                              {"PSAXA", false},
                                              };

    QString m_prevViewName;
    QImage m_prevImage;
    QVariant m_prevParamValues;

    QString m_currentViewName;
    QImage m_currentImage;

    QVariant m_currentQualityScores;
    QVariant m_currentParamValues;
    QVariant m_structParamValues;
    QVariant m_currentParamPremiums;

};

#endif // PROGRESSSUPERTHREAD_H
