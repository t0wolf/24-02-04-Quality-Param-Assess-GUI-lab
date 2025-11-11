/********************************************************************************
** Form generated from reading UI file 'param_display_item_widget.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PARAM_DISPLAY_ITEM_WIDGET_H
#define UI_PARAM_DISPLAY_ITEM_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ParamDisplayItemWidget
{
public:
    QGridLayout *gridLayout;
    QFrame *frame;
    QHBoxLayout *horizontalLayout;
    QLabel *paramEventLabel;
    QFrame *line;
    QLabel *paramValueLabel;

    void setupUi(QWidget *ParamDisplayItemWidget)
    {
        if (ParamDisplayItemWidget->objectName().isEmpty())
            ParamDisplayItemWidget->setObjectName(QString::fromUtf8("ParamDisplayItemWidget"));
        ParamDisplayItemWidget->resize(303, 95);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(ParamDisplayItemWidget->sizePolicy().hasHeightForWidth());
        ParamDisplayItemWidget->setSizePolicy(sizePolicy);
        ParamDisplayItemWidget->setStyleSheet(QString::fromUtf8("QFrame {\n"
"    background-color: #333; /* \346\267\261\347\201\260/\351\273\221\350\211\262\350\203\214\346\231\257 */\n"
"    border: 1px solid #555; /* \347\225\245\344\272\256\347\232\204\347\201\260\350\211\262\350\276\271\346\241\206 */\n"
"    border-radius: 5px; /* \345\234\206\350\247\222 */\n"
"}\n"
"\n"
"QLabel {\n"
"    color: #ddd; /* \346\265\205\347\201\260\350\211\262\346\226\207\346\234\254\357\274\214\346\230\276\350\221\227\344\272\216\346\267\261\350\211\262\350\203\214\346\231\257\344\271\213\344\270\212 */\n"
"    font: bold 14px; /* \347\262\227\344\275\223\357\274\214\351\200\202\344\270\255\347\232\204\345\255\227\344\275\223\345\244\247\345\260\217 */\n"
"    padding: 4px; /* \345\206\205\350\276\271\350\267\235 */\n"
"    margin: 0 2px; /* \344\270\244\344\270\252\346\240\207\347\255\276\344\271\213\351\227\264\347\225\245\345\276\256\351\227\264\351\232\224 */\n"
"}\n"
""));
        gridLayout = new QGridLayout(ParamDisplayItemWidget);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        frame = new QFrame(ParamDisplayItemWidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        horizontalLayout = new QHBoxLayout(frame);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(1, -1, -1, -1);
        paramEventLabel = new QLabel(frame);
        paramEventLabel->setObjectName(QString::fromUtf8("paramEventLabel"));
        QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(paramEventLabel->sizePolicy().hasHeightForWidth());
        paramEventLabel->setSizePolicy(sizePolicy1);
        paramEventLabel->setMinimumSize(QSize(50, 0));
        paramEventLabel->setMaximumSize(QSize(50, 16777215));
        QFont font;
        font.setBold(true);
        font.setItalic(false);
        font.setWeight(75);
        paramEventLabel->setFont(font);
        paramEventLabel->setScaledContents(true);

        horizontalLayout->addWidget(paramEventLabel);

        line = new QFrame(frame);
        line->setObjectName(QString::fromUtf8("line"));
        line->setFrameShape(QFrame::VLine);
        line->setFrameShadow(QFrame::Sunken);

        horizontalLayout->addWidget(line);

        paramValueLabel = new QLabel(frame);
        paramValueLabel->setObjectName(QString::fromUtf8("paramValueLabel"));
        sizePolicy.setHeightForWidth(paramValueLabel->sizePolicy().hasHeightForWidth());
        paramValueLabel->setSizePolicy(sizePolicy);
        paramValueLabel->setMinimumSize(QSize(125, 0));
        paramValueLabel->setMaximumSize(QSize(200, 16777215));

        horizontalLayout->addWidget(paramValueLabel);


        gridLayout->addWidget(frame, 0, 0, 1, 1);


        retranslateUi(ParamDisplayItemWidget);

        QMetaObject::connectSlotsByName(ParamDisplayItemWidget);
    } // setupUi

    void retranslateUi(QWidget *ParamDisplayItemWidget)
    {
        ParamDisplayItemWidget->setWindowTitle(QCoreApplication::translate("ParamDisplayItemWidget", "Form", nullptr));
        paramEventLabel->setText(QString());
        paramValueLabel->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class ParamDisplayItemWidget: public Ui_ParamDisplayItemWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PARAM_DISPLAY_ITEM_WIDGET_H
