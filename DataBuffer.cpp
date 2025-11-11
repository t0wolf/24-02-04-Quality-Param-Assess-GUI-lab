#include "DataBuffer.h"

DataBuffer::DataBuffer()
	: m_isRoIDataUpdate(false)
{
	m_roiScaleInfoBuffer.clear();
}

DataBuffer::~DataBuffer()
{}

void DataBuffer::setData(RoIScaleInfo& roiScaleInfo)
{
	QMutexLocker locker(&m_dataBufferMutex);
	m_roiScaleInfoBuffer = roiScaleInfo;
	m_isRoIDataUpdate = true;
}

void DataBuffer::getData(RoIScaleInfo& roiScaleInfo)
{
	QMutexLocker locker(&m_dataBufferMutex);
	roiScaleInfo = m_roiScaleInfoBuffer;
	m_isRoIDataUpdate = false;
}
