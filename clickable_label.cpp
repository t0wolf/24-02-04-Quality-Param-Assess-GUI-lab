#include "clickable_label.h"
#include "general_utils.h"

ClickableLabel::ClickableLabel(QWidget* parent, Qt::WindowFlags f)
    : QLabel(parent)
    , m_qualityDetailWidget(new QualityDetailWidget())
    , m_detailPlayThread(new QualityDetailPlayThread())
{
    connect(this, &ClickableLabel::mouseClicked, this, &ClickableLabel::slotDisplayQualityDetail);
    connect(m_detailPlayThread, &QualityDetailPlayThread::frameAvailable, m_qualityDetailWidget, &QualityDetailWidget::setSavedVideoFrame);
}

ClickableLabel::~ClickableLabel() {}

void ClickableLabel::mousePressEvent(QMouseEvent* event)
{
    emit mouseClicked();
}

// Slots
void ClickableLabel::slotDisplayQualityDetail()
{
    m_qualityDetailWidget->show();
    QPoint globalPos = this->mapToGlobal(QPoint(0, 0));
    QRect screenGeometry = QApplication::primaryScreen()->geometry();

    int xPosition = globalPos.x() + this->width(); // 在CustomWidget的右侧
    int yPosition = globalPos.y(); // 与CustomWidget的顶部对齐

    // 如果弹出窗口的右边界超出屏幕，尝试将其移至CustomWidget的左侧显示
    if (xPosition + m_qualityDetailWidget->width() > screenGeometry.right()) {
        xPosition = globalPos.x() - m_qualityDetailWidget->width(); // 移至CustomWidget的左侧
    }

    // 如果弹出窗口的下边界超出屏幕，调整其在CustomWidget的上方或下方显示以适应屏幕
    if (yPosition + m_qualityDetailWidget->height() > screenGeometry.bottom()) {
        yPosition = screenGeometry.bottom() - m_qualityDetailWidget->height(); // 向上调整至屏幕边缘
    }

    m_qualityDetailWidget->move(xPosition, yPosition); // 移动PopupWindow到计算后的位置

    if (!m_videoClipBuffer.empty())
    {
        m_detailPlayThread->setVideoBuffer(m_videoClipBuffer);
        m_detailPlayThread->start();
    }
}

void ClickableLabel::setViewName(const QString viewName)
{
    m_viewName = viewName;
    m_qualityDetailWidget->setViewName(viewName);
}

void ClickableLabel::setPlayThreadExit()
{
    m_detailPlayThread->exitThread();
}

void ClickableLabel::setInitialize()
{
    this->clear();
    this->m_videoClipBuffer.clear();

    QLabel *textLabel = nullptr;
    textLabel = this->findChild<QLabel*>();

    if (textLabel)
    {
        delete textLabel;
        textLabel = nullptr;
    }

    textLabel = new QLabel(this); // 将文本QLabel作为图像QLabel的子控件
    textLabel->raise();
    textLabel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

    textLabel->setStyleSheet(
        "QLabel {"
        "   border: none;" // 去除边框
        "   font-family: Arial;" // font
        "   font-weight: bold;" // 字体加粗
        "   font-size: 12px;" // 字体大小
        "}"
        );
    textLabel->resize(this->width(), this->height());
    textLabel->setText(QString("<h1 style='color: #808080;'>%1</h1>").arg(m_viewName));
    textLabel->setAlignment(Qt::AlignLeft | Qt::AlignTop); // 文本顶部居中对齐
    textLabel->show();
}

void ClickableLabel::setQualityScores(QVariant qVResults)
{
    float fCurrViewQualityScore = m_qualityDetailWidget->setQualityScores(qVResults);

}
