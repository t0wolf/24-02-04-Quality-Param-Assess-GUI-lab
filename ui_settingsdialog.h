/********************************************************************************
** Form generated from reading UI file 'settingsdialog.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SETTINGSDIALOG_H
#define UI_SETTINGSDIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTableWidget>

QT_BEGIN_NAMESPACE

class Ui_SettingsDialog
{
public:
    QPushButton *okButton;
    QPushButton *cancleButton;
    QTableWidget *configsWidget;

    void setupUi(QDialog *SettingsDialog)
    {
        if (SettingsDialog->objectName().isEmpty())
            SettingsDialog->setObjectName(QString::fromUtf8("SettingsDialog"));
        SettingsDialog->resize(400, 300);
        okButton = new QPushButton(SettingsDialog);
        okButton->setObjectName(QString::fromUtf8("okButton"));
        okButton->setGeometry(QRect(210, 270, 80, 23));
        cancleButton = new QPushButton(SettingsDialog);
        cancleButton->setObjectName(QString::fromUtf8("cancleButton"));
        cancleButton->setGeometry(QRect(310, 270, 80, 23));
        configsWidget = new QTableWidget(SettingsDialog);
        if (configsWidget->columnCount() < 2)
            configsWidget->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        configsWidget->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        configsWidget->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        if (configsWidget->rowCount() < 1)
            configsWidget->setRowCount(1);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        configsWidget->setVerticalHeaderItem(0, __qtablewidgetitem2);
        configsWidget->setObjectName(QString::fromUtf8("configsWidget"));
        configsWidget->setGeometry(QRect(0, 20, 400, 200));
        configsWidget->horizontalHeader()->setVisible(false);
        configsWidget->horizontalHeader()->setHighlightSections(true);
        configsWidget->verticalHeader()->setVisible(false);
        configsWidget->verticalHeader()->setHighlightSections(true);

        retranslateUi(SettingsDialog);

        QMetaObject::connectSlotsByName(SettingsDialog);
    } // setupUi

    void retranslateUi(QDialog *SettingsDialog)
    {
        SettingsDialog->setWindowTitle(QCoreApplication::translate("SettingsDialog", "\350\256\276\347\275\256", nullptr));
        okButton->setText(QCoreApplication::translate("SettingsDialog", "\344\277\235\345\255\230", nullptr));
        cancleButton->setText(QCoreApplication::translate("SettingsDialog", "\345\217\226\346\266\210", nullptr));
        QTableWidgetItem *___qtablewidgetitem = configsWidget->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QCoreApplication::translate("SettingsDialog", "New Column", nullptr));
        QTableWidgetItem *___qtablewidgetitem1 = configsWidget->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QCoreApplication::translate("SettingsDialog", "New Column", nullptr));
        QTableWidgetItem *___qtablewidgetitem2 = configsWidget->verticalHeaderItem(0);
        ___qtablewidgetitem2->setText(QCoreApplication::translate("SettingsDialog", "New Row", nullptr));
    } // retranslateUi

};

namespace Ui {
    class SettingsDialog: public Ui_SettingsDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SETTINGSDIALOG_H
