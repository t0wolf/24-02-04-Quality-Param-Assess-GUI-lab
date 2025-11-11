#ifndef PARAM_DISPLAY_WIDGET_H
#define PARAM_DISPLAY_WIDGET_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <QWidget>
#include <QMap>
#include <QTimer>
#include <QListWidgetItem>
#include <QScroller>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>
#include <QTextStream>

#include "progress_super_thread.h"
#include "rapidjson/document.h"

#include "param_display_item_widget.h"
#include "ParamBrowserWidget.h"
#include "ParamSaver.h"
#include "ParamTransmitter.h"

namespace Ui {
class ParamDisplayWidget;
}

class ParamDisplayWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ParamDisplayWidget(QWidget *parent = nullptr, 
        ProgressSuperThread *progressThread = nullptr, 
        ParamBrowserWidget* paramBrowserWidget = nullptr,
        ParamTransmitter* paramTransmitter = nullptr);
    ~ParamDisplayWidget();

    int paramLabelsReInit();

    void saveParamToCsv(const QString& rootPath, const QString& patientName);

    void saveParamJsonToFile(const QString& rootPath, const QString& patientName);

    void clearParamValues()
    {
        m_qmTotalParamValues.clear();
    }

    void initParamSaver(QString patientName);

public slots:
    void setParamValues(const QString viewName, QVariant qVar);

    void setParamValuesPics(QString viewName, QVariant qValues, QVariant qPremiums);

    void setStructParamValuesPics(QVariant qValues, QVariant qPremiums);

    void setParamValueDeleted(QString paramName);

    void onItemPressed(QListWidgetItem *item)
    {
        item->setBackground(Qt::yellow);
    }

    void onItemReleased(QListWidgetItem *item)
    {
        QTimer::singleShot(100, this, [item](){
            item->setBackground(Qt::white);
        });
    }

    //void paramBrowserClicked();

private:
    int parseJSONFile(std::string jsonPath);

    int initialize();

    int initParamEvents();

    void handleFuncParam(QString& strParamName);

    void changeFuncParamWidgetName(QString strCurrParamName, QString strChangedParamName);

    QString getCurrentDateFileName() {
        QDateTime currentDateTime = QDateTime::currentDateTime();
        return currentDateTime.toString("yyyyMMdd_HHmmss");
    }

    QString getCurrentDateFolderName() {
        QDateTime currentDateTime = QDateTime::currentDateTime();
        return currentDateTime.toString("yyyyMMdd");
    }

    ParamDisplayItemWidget* findSpecificParamWidget(QString paramName);

    int addParamToSaver(ParamData& paramData);

    int addParamToTransmitter(QMap<QString, QString>& paramEventValue);

    int addParamToTransmitter(QString& strViewName, QMap<QString, QString>& paramEventValue, QMap<QString, QImage>& paramEventPremium);

private:
    Ui::ParamDisplayWidget *ui;

    ParamBrowserWidget* m_paramBrowserWidget;

    ProgressSuperThread *m_progressThread;

    rapidjson::Document m_paramEvents;

    QMap<QString, QVector<QString>> m_qmTotalParamValues;

    QMap<QString, QVector<QString>> m_structParamEvents, m_spectrumParamEvents, m_funcParamEvents;

    ParamSaver* m_paramSaver;

    ParamTransmitter* m_paramTransmitter;
};

#endif // PARAM_DISPLAY_WIDGET_H
