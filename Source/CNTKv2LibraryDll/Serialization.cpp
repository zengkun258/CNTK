//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include <istream>
#include <ostream>
#include <string>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 4800 4267 4610 4512 4100 4510)
#include "CNTK.pb.h"
#pragma warning(pop)

namespace CNTK
{

    proto::DictionaryValue* CreateProto(const DictionaryValue& src);
    proto::Dictionary* CreateProto(const Dictionary& src);
    proto::Vector* CreateProto(const std::vector<DictionaryValue>& src);
    proto::NDArrayView* CreateProto(const NDArrayView& src);
    proto::Axis* CreateProto(const Axis& src);
    proto::NDShape* CreateProto(const NDShape& src);

    DictionaryValue* CreateFromProto(const proto::DictionaryValue& src);
    Dictionary* CreateFromProto(const proto::Dictionary& src);
    std::vector<DictionaryValue>* CreateFromProto(const proto::Vector& src);
    NDArrayView* CreateFromProto(const proto::NDArrayView& src);
    Axis* CreateFromProto(const proto::Axis& src);
    NDShape* CreateFromProto(const proto::NDShape& src);

    // TODO: safe as bytes instead?
    std::string ToString(const std::wstring& wstring)
    {
        return std::string(wstring.begin(), wstring.end());
    }

    std::wstring ToWString(const std::string& string)
    {
        return std::wstring(string.begin(), string.end());
    }

    proto::NDArrayView::DataType ToProtoType(DataType type)
    {
        if (!proto::NDArrayView::DataType_IsValid((int)type))
        {
            InvalidArgument("NDArrayView::DataType is invalid.");
        }
        return proto::NDArrayView_DataType(type);
        switch (type)
        {
        case DataType::Float:
            return proto::NDArrayView::Float;
        case DataType::Double:
            return proto::NDArrayView::Double;
        case DataType::Unknown:
            return proto::NDArrayView::Unknown;
        default:
            NOT_IMPLEMENTED
        }
    }

    DataType FromProtoType(proto::NDArrayView::DataType type)
    {
        switch (type)
        {
        case proto::NDArrayView::Float:
            return DataType::Float;
        case proto::NDArrayView::Double:
            return DataType::Double;
        case proto::NDArrayView::Unknown:
            return DataType::Unknown;
        default:
            NOT_IMPLEMENTED
        }
    }

    proto::NDArrayView::StorageFormat ToProtoType(StorageFormat type)
    {
        if (!proto::NDArrayView::StorageFormat_IsValid((int)type))
        {
            InvalidArgument("NDArrayView::StorageFormat is invalid.");
        }
        switch (type)
        {
        case StorageFormat::Dense:
            return proto::NDArrayView::Dense;
        case StorageFormat::SparseCSC:
            return proto::NDArrayView::SparseCSC;
        case StorageFormat::SparseBlockCol:
            return proto::NDArrayView::SparseBlockCol;
        default:
            NOT_IMPLEMENTED
        }
    }

    proto::DictionaryValue::Type ToProtoType(DictionaryValue::Type type)
    {
        if (!proto::DictionaryValue::Type_IsValid((int)type))
        {
            InvalidArgument("DictionaryValue::Type is invalid.");
        }
        switch (type)
        {
        case DictionaryValue::Type::None:
            return proto::DictionaryValue::None;
        case DictionaryValue::Type::Bool:
            return proto::DictionaryValue::Bool;
        case DictionaryValue::Type::Int:
            return proto::DictionaryValue::Int;
        case DictionaryValue::Type::SizeT:
            return proto::DictionaryValue::SizeT;
        case DictionaryValue::Type::Float:
            return proto::DictionaryValue::Float;
        case DictionaryValue::Type::Double:
            return proto::DictionaryValue::Double;
        case DictionaryValue::Type::NDShape:
            return proto::DictionaryValue::NDShape;
        case DictionaryValue::Type::Axis:
            return proto::DictionaryValue::Axis;
        case DictionaryValue::Type::Vector:
            return proto::DictionaryValue::Vector;
        case DictionaryValue::Type::Dictionary:
            return proto::DictionaryValue::Dictionary;
        case DictionaryValue::Type::NDArrayView:
            return proto::DictionaryValue::NDArrayView;
        default:
            NOT_IMPLEMENTED
        }
    }
    // TODO: use arenas for message allocations
    proto::NDShape* CreateProto(const NDShape& src)
    {
        proto::NDShape* dst = new proto::NDShape();
        auto size = src.Rank();
        dst->mutable_shape_dim()->Reserve(size);
        for (auto i = 0; i < size; i++)
        {
            dst->add_shape_dim(src[i]);
        }
        return dst;
    }

    NDShape* CreateFromProto(const proto::NDShape& src)
    {
        auto size = src.shape_dim_size();
        NDShape* dst = new NDShape(size);
        for (auto i = 0; i < size; i++)
        {
            dst[i] = src.shape_dim[i];
        }
        return dst;
    }

    proto::Axis* CreateProto(const Axis& src)
    {
        proto::Axis* dst = new proto::Axis();
        dst->set_static_axis_idx(src.StaticAxisIndex(false));
        dst->set_name(ToString(src.Name()));
        dst->set_is_ordered_dynamic_axis(!src.IsStaticAxis() && src.IsOrdered());
        return dst;
    }

    Axis* CreateFromProto(const proto::Axis& src)
    {
        if (Axis(src.static_axis_idx()).IsStaticAxis())
        {
             return new Axis(src.static_axis_idx());
        }
        else
        {
            return new Axis(ToWString(src.name()), src.is_ordered_dynamic_axis());
        }
    }

    template <typename T>
    void CopyData(const NDArrayView& src, ::google::protobuf::RepeatedField<T>* dst)
    {
        auto size = src.Shape().TotalSize();
        dst->Reserve(size);
        T* buffer = src.DataBuffer<T>();
        for (auto i = 0; i < size; ++i)
        {
            dst->Add(buffer[i]);
        }
    }

    proto::NDArrayView* CreatProto(const NDArrayView& src)
    {
        proto::NDArrayView* dst = new proto::NDArrayView();
        dst->set_data_type(ToProtoType(src.GetDataType()));
        dst->set_allocated_shape(CreateProto(us.Shape()));
        dst->set_is_read_only(src.IsReadOnly());
        dst->set_storage_format(ToProtoType(src.GetStorageFormat()));
        if (src.GetDataType() == DataType::Float) 
        {
            CopyData<float>(src, dst->mutable_float_values()->mutable_value());
        } 
        else if (src.GetDataType() == DataType::Double) 
        {
            CopyData<double>(src, dst->mutable_double_values()->mutable_value());
        }
    }

    NDArrayView* CreatFromProto(const proto::NDArrayView& src)
    {
         NDArrayView* viewPtr = new NDArrayView(dtype, shape, DeviceDescriptor::CPUDevice());
        NDArrayView* dst = new NDArrayView();
        dst->set_data_type(ToProtoType(src.GetDataType()));
        dst->set_allocated_shape(CreateProto(us.Shape()));
        dst->set_is_read_only(src.IsReadOnly());
        dst->set_storage_format(ToProtoType(src.GetStorageFormat()));
        if (src.GetDataType() == DataType::Float) 
        {
            CopyData<float>(src, dst->mutable_float_values()->mutable_value());
        } 
        else if (src.GetDataType() == DataType::Double) 
        {
            CopyData<double>(src, dst->mutable_double_values()->mutable_value());
        }
    }

    void Copy(const std::vector<DictionaryValue>& src, proto::Vector& dst)
    {
        for (const auto& value : src)
        {
            Copy(value, *dst.add_value());
        }
    }

    void Copy(const Dictionary& src, proto::Dictionary& dst)
    {
        for (const auto& kv : src)
        {
            Copy(kv.second, dst.mutable_data()->operator[ToString(kv.first)]);
        }
    }

    void Copy(const DictionaryValue& src, proto::DictionaryValue& dst)
    {
        auto valueType = src.ValueType();
        dst.set_value_type(ToProtoType(valueType));
        switch (valueType)
        {
        case DictionaryValue::Type::None:
            return;
        case DictionaryValue::Type::Bool:
            dst.set_bool_value(src.Value<bool>());
        case DictionaryValue::Type::Int:
            dst.set_int_value(src.Value<int>());
        case DictionaryValue::Type::SizeT:
            dst.set_size_t_value(src.Value<size_t>());
        case DictionaryValue::Type::Float:
            dst.set_float_value(src.Value<float>());
        case DictionaryValue::Type::Double:
            dst.set_double_value(src.Value<double>());
        case DictionaryValue::Type::NDShape:
            Copy(src.Value<NDShape>(), *dst.mutable_nd_shape_value());
        case DictionaryValue::Type::Axis:
            Copy(src.Value<Axis>(), *dst.mutable_axis_value());
        case DictionaryValue::Type::Vector:
            Copy(src.Value<std::vector<DictionaryValue>>(), *dst.mutable_vector_value());
        case DictionaryValue::Type::Dictionary:
            Copy(src.Value<Dictionary>(), *dst.mutable_dictionary_value());
        case DictionaryValue::Type::NDArrayView:
            Copy(src.Value<Dictionary>(), *dst.mutable_dictionary_value());
        default:
            NOT_IMPLEMENTED
        }
    }



    template <typename T>
    void Read(BinaryIStreamWrapper& stream, NDArrayView& view)
    {
        assert(view.Device().Type() == DeviceKind::CPU);
        
        auto numElements = view.Shape().TotalSize();
        T* buffer = view.WritableDataBuffer<T>();
        for (auto i = 0; i < numElements; ++i)
        {
            stream >> buffer[i];
        }
    }

    std::istream& operator>>(std::istream& stdStream, DictionaryValue& us)
    {
        BinaryIStreamWrapper stream(stdStream);
        size_t version;
        stream >> version;
        
        unsigned int type;
        stream >> type;
        us.m_valueType = static_cast<DictionaryValue::Type>(type);

        switch (us.ValueType())
        {
        case DictionaryValue::Type::Bool:
            stream >> us.m_data.m_boolean;
            break;
        case DictionaryValue::Type::Int:
            stream >> us.m_data.m_int;
            break;
        case DictionaryValue::Type::SizeT:
            stream >> us.m_data.m_sizeT;
            break;
        case DictionaryValue::Type::Float:
            stream >> us.m_data.m_float;
            break;
        case DictionaryValue::Type::Double:
            stream >> us.m_data.m_double;
            break;
        case DictionaryValue::Type::String:
        {
            std::wstring* strPtr = new std::wstring();
            stream >> *strPtr;
            us.m_data.m_ptr = strPtr;
            break;
        }
        case DictionaryValue::Type::NDShape:
        {
            size_t size;
            stream >> size;
            NDShape* shapePtr = new NDShape(size);
            for (auto i = 0; i < size; i++)
            {
                stream >> shapePtr->operator[](i);
            }
            us.m_data.m_ptr = shapePtr;
            break;
        }
        case DictionaryValue::Type::Axis:
        {
            int staticAxisIdx;
            stream >> staticAxisIdx;

            std::wstring axisName;
            stream >> axisName;

            bool isOrderedDynamicAxis;
            stream >> isOrderedDynamicAxis;

            Axis* axisPtr = nullptr;
            if (Axis(staticAxisIdx).IsStaticAxis())
                axisPtr = new Axis(staticAxisIdx);
            else
                axisPtr = new Axis(axisName, isOrderedDynamicAxis);

            us.m_data.m_ptr = axisPtr;
            break;
        }
        case DictionaryValue::Type::Vector:
        {   
            size_t size;
            stream >> size;
            std::vector<DictionaryValue>* vectorPtr = new std::vector<DictionaryValue>(size);
            for (auto i = 0; i < size; i++)
            {
                stream >> vectorPtr->at(i);
            }
            us.m_data.m_ptr = vectorPtr;
            break;
        }
        case DictionaryValue::Type::Dictionary:
        {
            Dictionary* dictPtr = new Dictionary();
            stream >> *dictPtr;
            us.m_data.m_ptr = dictPtr;
            break;
        }
        case DictionaryValue::Type::NDArrayView:
        {
            unsigned int type;
            stream >> type;
            DataType dtype = static_cast<DataType>(type);

            size_t size;
            stream >> size;
            NDShape shape(size);
            for (auto i = 0; i < size; i++)
            {
                stream >> shape[i];
            }

            NDArrayView* viewPtr = new NDArrayView(dtype, shape, DeviceDescriptor::CPUDevice());
            switch (dtype)
            {
            case DataType::Float:
                Read<float>(stream, *viewPtr);
                break;
            case DataType::Double:
                Read<double>(stream, *viewPtr);
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(dtype));
            }

            us.m_data.m_ptr = viewPtr;
            break;
        }
        default:
            NOT_IMPLEMENTED;
        }
        return stream;
    }

    std::ostream& operator<<(std::ostream& stdStream, const DictionaryValue& us)
    {
        BinaryOStreamWrapper stream(stdStream);

        stream << us.version;

        stream << static_cast<unsigned int>(us.ValueType());

        switch (us.ValueType())
        {
        case DictionaryValue::Type::Bool:
            stream << us.m_data.m_boolean;
            break;
        case DictionaryValue::Type::Int:
            stream << us.m_data.m_int;
            break;
        case DictionaryValue::Type::SizeT:
            stream << us.m_data.m_sizeT;
            break;
        case DictionaryValue::Type::Float:
            stream << us.m_data.m_float;
            break;
        case DictionaryValue::Type::Double:
            stream << us.m_data.m_double;
            break;
        case DictionaryValue::Type::String:
        {
            std::wstring* stringPtr = reinterpret_cast<std::wstring*>(us.m_data.m_ptr);
            stream << *stringPtr;
            break;
        }
        case DictionaryValue::Type::NDShape:
        {
            NDShape* shapePtr = reinterpret_cast<NDShape*>(us.m_data.m_ptr);
            stream << *shapePtr;
            break;
        }
        case DictionaryValue::Type::Axis:
        {
            Axis* axisPtr = reinterpret_cast<Axis*>(us.m_data.m_ptr);
            stream << *axisPtr;
            break;
        }
        case DictionaryValue::Type::Vector:
        {
            std::vector<DictionaryValue>* vectorPtr =
                reinterpret_cast<std::vector<DictionaryValue>*>(us.m_data.m_ptr);
            auto size = vectorPtr->size();
            stream << size;
            for (auto i = 0; i < size; i++)
            {
                stream << vectorPtr->at(i);
            }
            break;
        }
        case DictionaryValue::Type::Dictionary:
        {
            Dictionary* dictPtr = reinterpret_cast<Dictionary*>(us.m_data.m_ptr);
            stream << *dictPtr;
            break;
        }
        case DictionaryValue::Type::NDArrayView:
        {
            NDArrayView* viewPtr = reinterpret_cast<NDArrayView*>(us.m_data.m_ptr);
            stream << static_cast<unsigned int>(viewPtr->GetDataType());
            stream << viewPtr->Shape();
            switch (viewPtr->GetDataType())
            {
            case DataType::Float:
                Write<float>(stream, *viewPtr);
                break;
            case DataType::Double:
                Write<double>(stream, *viewPtr);
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(viewPtr->GetDataType()));
            }
            break;
        }
        default:
            NOT_IMPLEMENTED;
        }
        return stream;
    }

    std::ostream& operator<<(std::ostream& stdStream, const Dictionary& us)
    {
        BinaryOStreamWrapper stream(stdStream);
        stream << us.version;
        stream << us.m_dictionaryData->size();
        for (auto& kv : *(us.m_dictionaryData))
        {
            stream << kv.first;
            stream << kv.second;
        }
        return stream;
    }

    std::istream& operator>>(std::istream& stdStream, Dictionary& us)
    {
        BinaryIStreamWrapper stream(stdStream);
        size_t version;
        stream >> version;
        size_t size;
        stream >> size;
        us.m_dictionaryData->reserve(size);
        for (auto i = 0; i < size; i++)
        {
            std::wstring key;
            stream >> key;
            stream >> us[key];
        }
        return stream;
    }
}