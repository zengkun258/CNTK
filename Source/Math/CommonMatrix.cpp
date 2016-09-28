//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CommonMatrix.cpp
//
#include "stdafx.h"
#include "CommonMatrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

std::unordered_map<DEVICEID_TYPE, std::unique_ptr<BufferManagement>> BufferManagement::m_instances;

template <>
std::multimap<size_t, float*>& BufferManagement::BufferContainer<float>() { return m_bufferFloatContainer; }
template <>
std::multimap<size_t, double*>& BufferManagement::BufferContainer<double>() { return m_bufferDoubleContainer; }
template <>
std::multimap<size_t, char*>& BufferManagement::BufferContainer<char>() { return m_bufferCharContainer; }
template <>
std::multimap<size_t, short*>& BufferManagement::BufferContainer<short>() { return m_bufferShortContainer; }
template <>
std::multimap<size_t, int*>& BufferManagement::BufferContainer<int>() { return m_bufferIntContainer; }

}}}
