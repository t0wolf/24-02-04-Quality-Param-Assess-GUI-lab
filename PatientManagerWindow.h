#pragma once
#include <QDialog>
#include <QListWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QPushButton>
#include <QDir>
#include <QTextStream>
#include <QLineEdit>
#include <QInputDialog>
#include <QMessageBox>
//#include "PatientManager.h"

struct Patient {
    QString name;  // 病人姓名
    QString id;    // 病人 ID
    QMap<QString, QVector<double>> measurements; // 存储各种测量参数
};

class PatientManagerWindow : public QDialog {
    Q_OBJECT

public:
    explicit PatientManagerWindow(QWidget* parent = nullptr);
    void addPatient(const QString& name, const QString& id); // 添加新病人
    void updatePatientMeasurement(const QString& parameterName, float value); // 更新病人测量参数
    void switchPatient(const QString& patientId); // 切换病人

public slots:
    void addNewPatient();
    void deletePatient();
    void searchPatient();

private:
    QListWidget* m_patientListWidget; // 病人列表
    QLabel* infoLabel;               // 显示病人信息的标签
    QVBoxLayout* layout;             // 布局
    QDir m_patientHistDir;             // 病人历史数据目录
    Patient m_currentPatient;           // 当前病人
    QLineEdit* searchInput;

    bool m_bIsPatientSelected;           // 当前是否选择了病人

    void setupUI();
    void applyDarkStyle();
    void saveCurrentPatientData();     // 保存当前病人数据
    void displayPatientInfo();          // 显示当前病人详细信息

private slots:
    void on_patientList_itemClicked(QListWidgetItem* item); // 病人列表项被点击
};

