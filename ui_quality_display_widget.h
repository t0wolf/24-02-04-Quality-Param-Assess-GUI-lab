/********************************************************************************
** Form generated from reading UI file 'quality_display_widget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.11
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QUALITY_DISPLAY_WIDGET_H
#define UI_QUALITY_DISPLAY_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QWidget>
#include "clickable_label.h"

QT_BEGIN_NAMESPACE

class Ui_QualityDisplayWidget
{
public:
    QGridLayout *gridLayout_2;
    QFrame *frame;
    QGridLayout *gridLayout_3;
    QGridLayout *gridLayout;
    ClickableLabel *plaxLabel;
    ClickableLabel *a2cLabel;
    ClickableLabel *psaxmvLabel;
    ClickableLabel *psaxgvLabel;
    ClickableLabel *a4cLabel;
    ClickableLabel *a3cLabel;
    ClickableLabel *a5cLabel;
    ClickableLabel *psaxpmLabel;
    ClickableLabel *psaxaLabel;

    void setupUi(QWidget *QualityDisplayWidget)
    {
        if (QualityDisplayWidget->objectName().isEmpty())
            QualityDisplayWidget->setObjectName(QString::fromUtf8("QualityDisplayWidget"));
        QualityDisplayWidget->resize(500, 500);
        QSizePolicy sizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(QualityDisplayWidget->sizePolicy().hasHeightForWidth());
        QualityDisplayWidget->setSizePolicy(sizePolicy);
        QualityDisplayWidget->setMinimumSize(QSize(0, 350));
        QualityDisplayWidget->setMaximumSize(QSize(500, 500));
        gridLayout_2 = new QGridLayout(QualityDisplayWidget);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        frame = new QFrame(QualityDisplayWidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setMaximumSize(QSize(500, 500));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout_3 = new QGridLayout(frame);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        plaxLabel = new ClickableLabel(frame);
        plaxLabel->setObjectName(QString::fromUtf8("plaxLabel"));
        QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(plaxLabel->sizePolicy().hasHeightForWidth());
        plaxLabel->setSizePolicy(sizePolicy1);
        plaxLabel->setMaximumSize(QSize(200, 200));
        QFont font;
        font.setPointSize(12);
        font.setBold(true);
        font.setWeight(75);
        plaxLabel->setFont(font);
        plaxLabel->setAutoFillBackground(true);
        plaxLabel->setStyleSheet(QString::fromUtf8(""));
        plaxLabel->setScaledContents(true);
        plaxLabel->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        gridLayout->addWidget(plaxLabel, 1, 1, 1, 1);

        a2cLabel = new ClickableLabel(frame);
        a2cLabel->setObjectName(QString::fromUtf8("a2cLabel"));
        a2cLabel->setMaximumSize(QSize(200, 200));
        a2cLabel->setFont(font);
        a2cLabel->setAutoFillBackground(false);
        a2cLabel->setStyleSheet(QString::fromUtf8("border: 2px solid white; /* \350\276\271\346\241\206\345\256\275\345\272\246\344\270\2722px\357\274\214\351\242\234\350\211\262\344\270\272\351\273\221\350\211\262 */\n"
"border-radius: 4px; /* \350\276\271\346\241\206\345\234\206\350\247\222\344\270\2724px */"));
        a2cLabel->setScaledContents(true);
        a2cLabel->setAlignment(Qt::AlignHCenter|Qt::AlignTop);

        gridLayout->addWidget(a2cLabel, 0, 0, 1, 1);

        psaxmvLabel = new ClickableLabel(frame);
        psaxmvLabel->setObjectName(QString::fromUtf8("psaxmvLabel"));
        psaxmvLabel->setMaximumSize(QSize(200, 200));
        psaxmvLabel->setFont(font);
        psaxmvLabel->setAutoFillBackground(false);
        psaxmvLabel->setStyleSheet(QString::fromUtf8("border: 2px solid white; /* \350\276\271\346\241\206\345\256\275\345\272\246\344\270\2722px\357\274\214\351\242\234\350\211\262\344\270\272\351\273\221\350\211\262 */\n"
"border-radius: 4px; /* \350\276\271\346\241\206\345\234\206\350\247\222\344\270\2724px */"));
        psaxmvLabel->setScaledContents(true);

        gridLayout->addWidget(psaxmvLabel, 2, 1, 1, 1);

        psaxgvLabel = new ClickableLabel(frame);
        psaxgvLabel->setObjectName(QString::fromUtf8("psaxgvLabel"));
        psaxgvLabel->setMaximumSize(QSize(200, 200));
        psaxgvLabel->setFont(font);
        psaxgvLabel->setAutoFillBackground(false);
        psaxgvLabel->setStyleSheet(QString::fromUtf8("border: 2px solid white; /* \350\276\271\346\241\206\345\256\275\345\272\246\344\270\2722px\357\274\214\351\242\234\350\211\262\344\270\272\351\273\221\350\211\262 */\n"
"border-radius: 4px; /* \350\276\271\346\241\206\345\234\206\350\247\222\344\270\2724px */"));
        psaxgvLabel->setScaledContents(true);
        psaxgvLabel->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignTop);

        gridLayout->addWidget(psaxgvLabel, 1, 2, 1, 1);

        a4cLabel = new ClickableLabel(frame);
        a4cLabel->setObjectName(QString::fromUtf8("a4cLabel"));
        a4cLabel->setMaximumSize(QSize(200, 200));
        a4cLabel->setFont(font);
        a4cLabel->setAutoFillBackground(false);
        a4cLabel->setStyleSheet(QString::fromUtf8("border: 2px solid white; /* \350\276\271\346\241\206\345\256\275\345\272\246\344\270\2722px\357\274\214\351\242\234\350\211\262\344\270\272\351\273\221\350\211\262 */\n"
"border-radius: 4px; /* \350\276\271\346\241\206\345\234\206\350\247\222\344\270\2724px */"));
        a4cLabel->setScaledContents(true);

        gridLayout->addWidget(a4cLabel, 0, 2, 1, 1);

        a3cLabel = new ClickableLabel(frame);
        a3cLabel->setObjectName(QString::fromUtf8("a3cLabel"));
        a3cLabel->setMaximumSize(QSize(200, 200));
        a3cLabel->setFont(font);
        a3cLabel->setAutoFillBackground(false);
        a3cLabel->setStyleSheet(QString::fromUtf8("border: 2px solid white; /* \350\276\271\346\241\206\345\256\275\345\272\246\344\270\2722px\357\274\214\351\242\234\350\211\262\344\270\272\351\273\221\350\211\262 */\n"
"border-radius: 4px; /* \350\276\271\346\241\206\345\234\206\350\247\222\344\270\2724px */"));
        a3cLabel->setScaledContents(true);

        gridLayout->addWidget(a3cLabel, 0, 1, 1, 1);

        a5cLabel = new ClickableLabel(frame);
        a5cLabel->setObjectName(QString::fromUtf8("a5cLabel"));
        a5cLabel->setMaximumSize(QSize(200, 200));
        a5cLabel->setFont(font);
        a5cLabel->setAutoFillBackground(false);
        a5cLabel->setStyleSheet(QString::fromUtf8("border: 2px solid white; /* \350\276\271\346\241\206\345\256\275\345\272\246\344\270\2722px\357\274\214\351\242\234\350\211\262\344\270\272\351\273\221\350\211\262 */\n"
"border-radius: 4px; /* \350\276\271\346\241\206\345\234\206\350\247\222\344\270\2724px */"));
        a5cLabel->setScaledContents(true);

        gridLayout->addWidget(a5cLabel, 1, 0, 1, 1);

        psaxpmLabel = new ClickableLabel(frame);
        psaxpmLabel->setObjectName(QString::fromUtf8("psaxpmLabel"));
        psaxpmLabel->setMaximumSize(QSize(200, 200));
        psaxpmLabel->setFont(font);
        psaxpmLabel->setAutoFillBackground(false);
        psaxpmLabel->setStyleSheet(QString::fromUtf8("border: 2px solid white; /* \350\276\271\346\241\206\345\256\275\345\272\246\344\270\2722px\357\274\214\351\242\234\350\211\262\344\270\272\351\273\221\350\211\262 */\n"
"border-radius: 4px; /* \350\276\271\346\241\206\345\234\206\350\247\222\344\270\2724px */"));
        psaxpmLabel->setScaledContents(true);

        gridLayout->addWidget(psaxpmLabel, 2, 0, 1, 1);

        psaxaLabel = new ClickableLabel(frame);
        psaxaLabel->setObjectName(QString::fromUtf8("psaxaLabel"));
        psaxaLabel->setMaximumSize(QSize(200, 200));
        psaxaLabel->setFont(font);
        psaxaLabel->setAutoFillBackground(false);
        psaxaLabel->setStyleSheet(QString::fromUtf8("border: 2px solid white; /* \350\276\271\346\241\206\345\256\275\345\272\246\344\270\2722px\357\274\214\351\242\234\350\211\262\344\270\272\351\273\221\350\211\262 */\n"
"border-radius: 4px; /* \350\276\271\346\241\206\345\234\206\350\247\222\344\270\2724px */"));
        psaxaLabel->setScaledContents(true);

        gridLayout->addWidget(psaxaLabel, 2, 2, 1, 1);


        gridLayout_3->addLayout(gridLayout, 0, 0, 1, 1);


        gridLayout_2->addWidget(frame, 0, 0, 1, 1);


        retranslateUi(QualityDisplayWidget);

        QMetaObject::connectSlotsByName(QualityDisplayWidget);
    } // setupUi

    void retranslateUi(QWidget *QualityDisplayWidget)
    {
        QualityDisplayWidget->setWindowTitle(QApplication::translate("QualityDisplayWidget", "Form", nullptr));
        plaxLabel->setText(QString());
        a2cLabel->setText(QString());
        psaxmvLabel->setText(QString());
        psaxgvLabel->setText(QString());
        a4cLabel->setText(QString());
        a3cLabel->setText(QString());
        a5cLabel->setText(QString());
        psaxpmLabel->setText(QString());
        psaxaLabel->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class QualityDisplayWidget: public Ui_QualityDisplayWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QUALITY_DISPLAY_WIDGET_H
