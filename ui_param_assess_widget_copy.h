/********************************************************************************
** Form generated from reading UI file 'param_assess_widget_copy.ui'
**
** Created by: Qt User Interface Compiler version 5.12.11
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PARAM_ASSESS_WIDGET_COPY_H
#define UI_PARAM_ASSESS_WIDGET_COPY_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ParamAssessWidget
{
public:
    QGridLayout *gridLayout_2;
    QFrame *frame;
    QGridLayout *gridLayout;
    QLabel *label;
    QLabel *label_3;
    QTableWidget *structureParamTable;
    QLabel *label_2;
    QTableWidget *spectrumParamTable;
    QTableWidget *functionParamTable;

    void setupUi(QWidget *ParamAssessWidget)
    {
        if (ParamAssessWidget->objectName().isEmpty())
            ParamAssessWidget->setObjectName(QString::fromUtf8("ParamAssessWidget"));
        ParamAssessWidget->resize(277, 761);
        gridLayout_2 = new QGridLayout(ParamAssessWidget);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        frame = new QFrame(ParamAssessWidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout = new QGridLayout(frame);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(frame);
        label->setObjectName(QString::fromUtf8("label"));
        QFont font;
        font.setPointSize(18);
        font.setBold(false);
        label->setFont(font);
        label->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(label, 0, 0, 1, 1);

        label_3 = new QLabel(frame);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setFont(font);
        label_3->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(label_3, 4, 0, 1, 1);

        structureParamTable = new QTableWidget(frame);
        if (structureParamTable->columnCount() < 2)
            structureParamTable->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        structureParamTable->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        structureParamTable->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        if (structureParamTable->rowCount() < 8)
            structureParamTable->setRowCount(8);
        structureParamTable->setObjectName(QString::fromUtf8("structureParamTable"));

        gridLayout->addWidget(structureParamTable, 1, 0, 1, 1);

        label_2 = new QLabel(frame);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setFont(font);
        label_2->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(label_2, 2, 0, 1, 1);

        spectrumParamTable = new QTableWidget(frame);
        if (spectrumParamTable->columnCount() < 4)
            spectrumParamTable->setColumnCount(4);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        spectrumParamTable->setHorizontalHeaderItem(0, __qtablewidgetitem2);
        QTableWidgetItem *__qtablewidgetitem3 = new QTableWidgetItem();
        spectrumParamTable->setHorizontalHeaderItem(1, __qtablewidgetitem3);
        if (spectrumParamTable->rowCount() < 5)
            spectrumParamTable->setRowCount(5);
        spectrumParamTable->setObjectName(QString::fromUtf8("spectrumParamTable"));
        spectrumParamTable->setRowCount(5);
        spectrumParamTable->setColumnCount(4);

        gridLayout->addWidget(spectrumParamTable, 3, 0, 1, 1);

        functionParamTable = new QTableWidget(frame);
        if (functionParamTable->columnCount() < 4)
            functionParamTable->setColumnCount(4);
        QTableWidgetItem *__qtablewidgetitem4 = new QTableWidgetItem();
        functionParamTable->setHorizontalHeaderItem(0, __qtablewidgetitem4);
        QTableWidgetItem *__qtablewidgetitem5 = new QTableWidgetItem();
        functionParamTable->setHorizontalHeaderItem(1, __qtablewidgetitem5);
        if (functionParamTable->rowCount() < 5)
            functionParamTable->setRowCount(5);
        functionParamTable->setObjectName(QString::fromUtf8("functionParamTable"));
        functionParamTable->setRowCount(5);
        functionParamTable->setColumnCount(4);

        gridLayout->addWidget(functionParamTable, 5, 0, 1, 1);


        gridLayout_2->addWidget(frame, 0, 0, 1, 1);


        retranslateUi(ParamAssessWidget);

        QMetaObject::connectSlotsByName(ParamAssessWidget);
    } // setupUi

    void retranslateUi(QWidget *ParamAssessWidget)
    {
        ParamAssessWidget->setWindowTitle(QApplication::translate("ParamAssessWidget", "Form", nullptr));
        label->setText(QApplication::translate("ParamAssessWidget", "\347\273\223\346\236\204\345\217\202\346\225\260", nullptr));
        label_3->setText(QApplication::translate("ParamAssessWidget", "\345\212\237\350\203\275\345\217\202\346\225\260", nullptr));
        QTableWidgetItem *___qtablewidgetitem = structureParamTable->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QApplication::translate("ParamAssessWidget", "\345\217\202\346\225\260\351\241\271", nullptr));
        QTableWidgetItem *___qtablewidgetitem1 = structureParamTable->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QApplication::translate("ParamAssessWidget", "\345\217\202\346\225\260\345\200\274", nullptr));
        label_2->setText(QApplication::translate("ParamAssessWidget", "\351\242\221\350\260\261\345\217\202\346\225\260", nullptr));
        QTableWidgetItem *___qtablewidgetitem2 = spectrumParamTable->horizontalHeaderItem(0);
        ___qtablewidgetitem2->setText(QApplication::translate("ParamAssessWidget", "\345\217\202\346\225\260\351\241\271", nullptr));
        QTableWidgetItem *___qtablewidgetitem3 = spectrumParamTable->horizontalHeaderItem(1);
        ___qtablewidgetitem3->setText(QApplication::translate("ParamAssessWidget", "\345\217\202\346\225\260\345\200\274", nullptr));
        QTableWidgetItem *___qtablewidgetitem4 = functionParamTable->horizontalHeaderItem(0);
        ___qtablewidgetitem4->setText(QApplication::translate("ParamAssessWidget", "\345\217\202\346\225\260\351\241\271", nullptr));
        QTableWidgetItem *___qtablewidgetitem5 = functionParamTable->horizontalHeaderItem(1);
        ___qtablewidgetitem5->setText(QApplication::translate("ParamAssessWidget", "\345\217\202\346\225\260\345\200\274", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ParamAssessWidget: public Ui_ParamAssessWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PARAM_ASSESS_WIDGET_COPY_H
