#include "param_display_item_widget.h"
#include "ui_param_display_item_widget.h"

ParamDisplayItemWidget::ParamDisplayItemWidget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::ParamDisplayItemWidget)
    , m_paramPreWidget(new ParamPremiumWidget())
{
    ui->setupUi(this);
}

ParamDisplayItemWidget::~ParamDisplayItemWidget()
{
    delete ui;
}

void ParamDisplayItemWidget::initializeParamEvents()
{
    setParamValueDeleted();
}

void ParamDisplayItemWidget::setName(QString paramName)
{
    m_paramName = paramName;
    ui->paramEventLabel->setText(paramName);
    m_paramPreWidget->setCurrentParamEvent(paramName);
}

int ParamDisplayItemWidget::setParamValue(float fValue)
{
    m_currentValue = fValue;
    //ui->paramValueLabel->setText(QString::number(fValue));
    ui->paramValueLabel->setText(QString("<span style='color: #00FF00;'>%1</span>").arg(QString::number(fValue)));
    return 1;
}

int ParamDisplayItemWidget::setParamValue(QString sValue)
{
    //m_currentValue = fValue;
    //ui->paramValueLabel->setText(QString::number(fValue));
    ui->paramValueLabel->setText(QString("<span style='color: #00FF00;'>%1</span>").arg(sValue));
    return 1;
}

int ParamDisplayItemWidget::setParamPics(QImage& qFrame)
{
    m_preImage = qFrame.copy();
    m_paramPreWidget->setCurrentParamsPics(m_preImage);
    return 1;
}

void ParamDisplayItemWidget::setParamValueDeleted()
{
    ui->paramValueLabel->clear();
}

void ParamDisplayItemWidget::mousePressEvent(QMouseEvent *event) {
    m_pressPos = event->pos();
    m_wasDragged = false;
    // 调用基类的鼠标点击事件处理函数
    // QWidget::mousePressEvent(event);
}

void ParamDisplayItemWidget::mouseReleaseEvent(QMouseEvent *event) {
    if (!m_wasDragged)
    {
        // 触发点击的逻辑，比如显示某个界面
        m_paramPreWidget->show();
        m_paramPreWidget->showCurrentParamsPics();
        m_paramPreWidget->setCurrentWindowTitle(m_paramName);
    }
}

void ParamDisplayItemWidget::mouseMoveEvent(QMouseEvent *event) {
    // 如果鼠标移动的距离超过一定阈值，则认为是拖动
    if ((event->pos() - m_pressPos).manhattanLength() > QApplication::startDragDistance()) {
        m_wasDragged = true; // 标记为拖动
    }
}
