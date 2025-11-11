/********************************************************************************
** Form generated from reading UI file 'param_assess_widget.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PARAM_ASSESS_WIDGET_H
#define UI_PARAM_ASSESS_WIDGET_H

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
    QTableWidget *paramTable;

    void setupUi(QWidget *ParamAssessWidget)
    {
        if (ParamAssessWidget->objectName().isEmpty())
            ParamAssessWidget->setObjectName(QString::fromUtf8("ParamAssessWidget"));
        ParamAssessWidget->resize(1027, 68);
        gridLayout_2 = new QGridLayout(ParamAssessWidget);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        frame = new QFrame(ParamAssessWidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setMinimumSize(QSize(0, 0));
        frame->setMaximumSize(QSize(16777215, 20));
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

        paramTable = new QTableWidget(frame);
        if (paramTable->columnCount() < 2)
            paramTable->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        paramTable->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        paramTable->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        if (paramTable->rowCount() < 10)
            paramTable->setRowCount(10);
        paramTable->setObjectName(QString::fromUtf8("paramTable"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(paramTable->sizePolicy().hasHeightForWidth());
        paramTable->setSizePolicy(sizePolicy);
        paramTable->setRowCount(10);

        gridLayout->addWidget(paramTable, 1, 0, 1, 1);


        gridLayout_2->addWidget(frame, 0, 0, 1, 1);


        retranslateUi(ParamAssessWidget);

        QMetaObject::connectSlotsByName(ParamAssessWidget);
    } // setupUi

    void retranslateUi(QWidget *ParamAssessWidget)
    {
        ParamAssessWidget->setWindowTitle(QCoreApplication::translate("ParamAssessWidget", "Form", nullptr));
        label->setText(QCoreApplication::translate("ParamAssessWidget", "\345\217\202\346\225\260\346\265\213\351\207\217", nullptr));
        QTableWidgetItem *___qtablewidgetitem = paramTable->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QCoreApplication::translate("ParamAssessWidget", "\345\217\202\346\225\260\351\241\271", nullptr));
        QTableWidgetItem *___qtablewidgetitem1 = paramTable->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QCoreApplication::translate("ParamAssessWidget", "\345\217\202\346\225\260\345\200\274", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ParamAssessWidget: public Ui_ParamAssessWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PARAM_ASSESS_WIDGET_H
