#ifndef SETTINGSDIALOG_H
#define SETTINGSDIALOG_H

#include <QDialog>
#include <QFile>
#include <QTextStream>
#include <QTableWidgetItem>
#include <QDebug>
#include <QVector>
#include "yaml-cpp/yaml.h"
#include "config_parse.h"

namespace Ui {
class SettingsDialog;
}

class SettingsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SettingsDialog(QWidget *parent = nullptr, ConfigParse* config = nullptr);
    ~SettingsDialog();

    void initconfigsWidget();

private slots:
    void on_okButton_clicked();

    void on_cancleButton_clicked();

private:
    Ui::SettingsDialog *ui;

private:
    QStringList m_originalLines;                      // 每行原始文本
    QMap<QString, int> m_keyToLineIndex;              // key → 行号
    QMap<QString, QString> m_originalValuesMap;       // key → 初始值（用于检测是否被用户改动）
    QVector<QString> m_chineseMap = {"OFFLINE_VIDEO_ID", "FRAME_WIDTH", "FRAME_HEIGHT", };
    QVector<QString> m_englishMap = {"视频源ID", "帧宽", "帧高"};
    ConfigParse* m_config;
};

#endif // SETTINGSDIALOG_H
