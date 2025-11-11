#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#pragma execution_character_set("utf-8")

#include <QMainWindow>
#include "display_stream_widget.h"
// #include "quality_control_widget.h"
#include "param_assess_widget.h"
// #include "view_progress_widget.h"
#include "progress_super_thread.h"
#include "quality_display_widget.h"
// #include "quality_detail_widget.h"
#include "param_display_widget.h"
#include "ParamBrowserWidget.h"
#include "config_parse.h"

#include "DarkStyle.h"
#include "framelesswindow/framelesswindow.h"
#include "framelesswindow/windowdragger.h"
//#include "process_threads/HttpJSONReceiver.h"
#include "process_threads/HttpServerThread.h"
#include "ParamTransmitter.h"
#include "QtLogger.h"
#include "settingsdialog.h"
#include "aboutdialog.h"
#include "PatientManagerWindow.h"

#include <QTextCodec>
#include <QString>
#include <QDateTime>
#include <QRegularExpression>
#include <QCoreApplication>
#include <QtGlobal>
#include <QFile>
#include <QTextStream>


QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE


void logException(const QString& message);

void customMessageHandler(QtMsgType type, const QMessageLogContext& context, const QString& msg);


class SafeApplication : public QApplication {
public:
    SafeApplication(int& argc, char** argv) : QApplication(argc, argv) {}

    bool notify(QObject* receiver, QEvent* event) override {
        try {
            return QApplication::notify(receiver, event);
        }
        catch (const std::exception& e) {
            // 处理标准异常
            logException(e.what());
        }
        catch (...) {
            // 处理不明确的异常
            logException("未知异常");
        }
        return false;
    }
};


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

signals:
    void sigModelInferClear();

    void sigPatientName(QString patientName);

public slots:
    void setPatientInfo(QString patientName, QString patientID);

    void setParamSaverInitByCheckID(const QString& checkID);

    void setParamTransmitterInitByCheckID(const QString& checkID);

    void on_clearAllButton_clicked();

    void on_browserParamButton_clicked();

    void onActionSettingsTriggered();

    void onAboutSettingsTriggered();

    void onPatientListButtonClicked();

private slots:
    void onChangePassword();

private:
    bool loadCredentials(const QString& filePath);
    bool saveCredentials(const QString& filePath, const QString& username, const QString& password);

    void setupMenu();

    QString md5Hash(const QString& input)
    {
        return QString(QCryptographicHash::hash(input.toUtf8(), QCryptographicHash::Md5).toHex());
    }

    void setQualityParamWidgetClear();

    void setupPatientManager();

    QString removePunctuationAndSpaces(QString input) 
    {
        // 使用正则表达式匹配标点符号和空格
        QRegularExpression re("[\\p{P}\\p{S}\\s]");
        return input.remove(re);
    }


private:
    QString m_username;
    QString m_password;

    ConfigParse* m_configFile;
    Ui::MainWindow *ui;
    DisplayStreamWidget *m_displayStreamWidget;
    ProgressSuperThread* m_progressSuperThread;
    QualityDisplayWidget* m_qualityDisplayWidget;
    ParamBrowserWidget* m_paramBrowserWidget;
    // QualityDetailWidget* m_qualityDetailWidget;
    ParamDisplayWidget* m_paramDisplayWidget;
    HttpServer* m_httpJSONReceiveThread;
    ParamTransmitter* m_paramTransmitter;

    PatientManagerWindow* m_patientManagerWindow; // 病人管理窗口

    SettingsDialog* m_settingsDialog;
    AboutDialog* m_aboutDialog;

    QString m_currCheckID;
    QString m_currPatientName, m_currPatientID;
};
#endif // MAINWINDOW_H
