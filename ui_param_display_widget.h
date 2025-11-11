/********************************************************************************
** Form generated from reading UI file 'param_display_widget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.11
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PARAM_DISPLAY_WIDGET_H
#define UI_PARAM_DISPLAY_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ParamDisplayWidget
{
public:
    QGridLayout *gridLayout_2;
    QFrame *frame;
    QGridLayout *gridLayout;
    QListWidget *paramListWidget;
    QListWidget *structureParamListWidget;

    void setupUi(QWidget *ParamDisplayWidget)
    {
        if (ParamDisplayWidget->objectName().isEmpty())
            ParamDisplayWidget->setObjectName(QString::fromUtf8("ParamDisplayWidget"));
        ParamDisplayWidget->resize(369, 553);
        gridLayout_2 = new QGridLayout(ParamDisplayWidget);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        frame = new QFrame(ParamDisplayWidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(frame->sizePolicy().hasHeightForWidth());
        frame->setSizePolicy(sizePolicy);
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout = new QGridLayout(frame);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        paramListWidget = new QListWidget(frame);
        paramListWidget->setObjectName(QString::fromUtf8("paramListWidget"));
        paramListWidget->setStyleSheet(QString::fromUtf8("QScrollBar:vertical { width: 20px; }"));

        gridLayout->addWidget(paramListWidget, 0, 2, 1, 1);

        structureParamListWidget = new QListWidget(frame);
        structureParamListWidget->setObjectName(QString::fromUtf8("structureParamListWidget"));
        structureParamListWidget->setStyleSheet(QString::fromUtf8("QScrollBar:vertical { width: 20px; }"));

        gridLayout->addWidget(structureParamListWidget, 0, 1, 1, 1);


        gridLayout_2->addWidget(frame, 0, 0, 1, 1);


        retranslateUi(ParamDisplayWidget);

        QMetaObject::connectSlotsByName(ParamDisplayWidget);
    } // setupUi

    void retranslateUi(QWidget *ParamDisplayWidget)
    {
        ParamDisplayWidget->setWindowTitle(QApplication::translate("ParamDisplayWidget", "Form", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ParamDisplayWidget: public Ui_ParamDisplayWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PARAM_DISPLAY_WIDGET_H
