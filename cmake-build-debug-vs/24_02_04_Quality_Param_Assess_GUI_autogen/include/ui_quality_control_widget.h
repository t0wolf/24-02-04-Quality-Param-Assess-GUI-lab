/********************************************************************************
** Form generated from reading UI file 'quality_control_widget.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QUALITY_CONTROL_WIDGET_H
#define UI_QUALITY_CONTROL_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_QualityControlWidget
{
public:
    QGridLayout *gridLayout;
    QFrame *frame;
    QGridLayout *gridLayout_2;
    QTableWidget *qualityRulesTable;
    QProgressBar *progressBar;
    QLabel *infoLabel;

    void setupUi(QWidget *QualityControlWidget)
    {
        if (QualityControlWidget->objectName().isEmpty())
            QualityControlWidget->setObjectName(QString::fromUtf8("QualityControlWidget"));
        QualityControlWidget->resize(268, 399);
        gridLayout = new QGridLayout(QualityControlWidget);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        frame = new QFrame(QualityControlWidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout_2 = new QGridLayout(frame);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        qualityRulesTable = new QTableWidget(frame);
        if (qualityRulesTable->columnCount() < 2)
            qualityRulesTable->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        qualityRulesTable->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        qualityRulesTable->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        if (qualityRulesTable->rowCount() < 10)
            qualityRulesTable->setRowCount(10);
        qualityRulesTable->setObjectName(QString::fromUtf8("qualityRulesTable"));
        qualityRulesTable->setMinimumSize(QSize(30, 0));
        qualityRulesTable->setRowCount(10);
        qualityRulesTable->setColumnCount(2);

        gridLayout_2->addWidget(qualityRulesTable, 1, 0, 1, 1);

        progressBar = new QProgressBar(frame);
        progressBar->setObjectName(QString::fromUtf8("progressBar"));
        progressBar->setValue(24);
        progressBar->setOrientation(Qt::Vertical);

        gridLayout_2->addWidget(progressBar, 1, 1, 1, 1);

        infoLabel = new QLabel(frame);
        infoLabel->setObjectName(QString::fromUtf8("infoLabel"));
        QFont font;
        font.setPointSize(15);
        infoLabel->setFont(font);
        infoLabel->setAlignment(Qt::AlignCenter);

        gridLayout_2->addWidget(infoLabel, 0, 0, 1, 1);


        gridLayout->addWidget(frame, 0, 1, 1, 1);


        retranslateUi(QualityControlWidget);

        QMetaObject::connectSlotsByName(QualityControlWidget);
    } // setupUi

    void retranslateUi(QWidget *QualityControlWidget)
    {
        QualityControlWidget->setWindowTitle(QCoreApplication::translate("QualityControlWidget", "Form", nullptr));
        QTableWidgetItem *___qtablewidgetitem = qualityRulesTable->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QCoreApplication::translate("QualityControlWidget", "\350\264\250\346\216\247\350\247\204\345\210\231", nullptr));
        QTableWidgetItem *___qtablewidgetitem1 = qualityRulesTable->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QCoreApplication::translate("QualityControlWidget", "\345\276\227\345\210\206", nullptr));
        infoLabel->setText(QCoreApplication::translate("QualityControlWidget", "\347\255\211\345\276\205\346\243\200\346\237\245", nullptr));
    } // retranslateUi

};

namespace Ui {
    class QualityControlWidget: public Ui_QualityControlWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QUALITY_CONTROL_WIDGET_H
