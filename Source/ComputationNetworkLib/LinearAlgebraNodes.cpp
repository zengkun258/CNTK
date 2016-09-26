//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "LinearAlgebraNodes.h"

using namespace Microsoft::MSR::CNTK;

template <class ElemType>
AccumulatorNode<ElemType>::AccumulatorNode(DEVICEID_TYPE deviceId, const wstring& name)
    : Base(deviceId, name), m_numSamples(0)
{
}

template <class ElemType>
AccumulatorNode<ElemType>::AccumulatorNode(const Microsoft::MSR::ScriptableObjects::IConfigRecordPtr configp)
    : AccumulatorNode(configp->Get(L"deviceId"), L"<placeholder>")
{
    AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
}

template <class ElemType>
void AccumulatorNode<ElemType>::BackpropToNonLooping(size_t /*inputIndex*/)
{
    LogicError("%ls operation is used for forward only.", OperationName().c_str());
}

template <class ElemType>
void AccumulatorNode<ElemType>::OnEpochStart()
{
    Reset();
}

template <class ElemType>
void AccumulatorNode<ElemType>::ForwardPropNonLooping()
{
    FrameRange fr(Input(0)->GetMBLayout());
    // Set gaps to zero, since we are reducing in time.
    Input(0)->MaskMissingValueColumnsToZero(fr);

    size_t numNewSamples = Input(0)->GetMBLayout()->GetActualNumSamples();
    size_t totalNumSamples = m_numSamples + numNewSamples;
    if (totalNumSamples == 0)
        totalNumSamples = 1;
    ElemType alpha = (ElemType)1.0f / totalNumSamples;
    ElemType beta = (ElemType)m_numSamples / totalNumSamples;

    size_t rank = DetermineElementwiseTensorRank();
    auto input = Input(0)->ValueTensorFor(rank, fr);
    auto accumulator = DataTensorFor(m_accumulator, rank, FrameRange());

    // accumulator = beta * accumulator + alpha * input.
    accumulator.DoCopyOf(beta, input, alpha);

    // Value gets resized in UpdateFunctionValuesSize that is called in BeforeForwardProp. Resize fills matrix with NaN
    // values, so m_value matrix cannot be used as persistent storage between ForwardProp calls.
    Value().SetValue(*m_accumulator);

    m_numSamples += numNewSamples;
}

template <class ElemType>
void AccumulatorNode<ElemType>::CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName,
                                       const CopyNodeFlags flags) const
{
    Base::CopyTo(nodeP, newName, flags);
    if (flags & CopyNodeFlags::copyNodeValue)
    {
        auto node = nodeP->As<AccumulatorNode<ElemType>>();
        node->m_numSamples = m_numSamples;
        node->m_accumulator->SetValue(*m_accumulator);
    }
}

template <class ElemType>
void AccumulatorNode<ElemType>::Validate(bool isFinalValidationPass)
{
    Base::Validate(isFinalValidationPass);
    SetDims(Input(0)->GetSampleLayout(), HasMBLayout());
}

template <class ElemType>
void AccumulatorNode<ElemType>::RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
{
    Base::RequestMatricesBeforeForwardProp(matrixPool);
    RequestMatrixFromPool(m_accumulator, matrixPool);
    const size_t sampleSize = GetSampleLayout().GetNumElements();
    m_accumulator->Resize(sampleSize, 1);
    Reset();
}

template <class ElemType>
void AccumulatorNode<ElemType>::Reset()
{
    m_accumulator->SetValue(0);
    m_numSamples = 0;
}

template class AccumulatorNode<float>;
template class AccumulatorNode<double>;