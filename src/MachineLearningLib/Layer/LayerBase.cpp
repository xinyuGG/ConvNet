#include "LayerBase.h"
#include <iostream>
namespace FengML
{
    static LayerBase dummyLayer;
    static typename Tensor1::DataType dummy1d;
    static typename Tensor3::DataType dummy3d;
    LayerBase& LayerBase::Add(std::shared_ptr<LayerBase> _nextLayer)
	//调用这个函数的是this 上一层函数
    {
        nextLayer = _nextLayer;
        nextLayer->previousLayer = this;
		nextLayer->Reshape(nextLayer->previousLayer);
        nextLayer->Initialize();
        return *nextLayer;
    }

	void LayerBase::Reshape(LayerBase* prelayer)
	{

	}
	void LayerBase::Forward(Blob<float>& indata, Blob<float>& outputdata)
	{

	}

	void LayerBase::Getlayerattri()
	{

	}

	void LayerBase::Initialize(int for_bach)
	{
		//如果FORWARD_ONLY = 0
		//利用初始化函数初始权重和偏置

		//否则加载外部权重数据
	}

	void LayerBase::Setpath(const std::string& weightpath, const std::string& biaspath)
	{

	}

/*    size_t LayerBase::Dim1()
    {
        return 0;
    }
		template<class T>
	void ConvLayer<T>::Setpath(std::string& weightpath, std::string& biaspath)
	{
		_weightpath = weightpath;
		_biaspath = biaspath;
	}

    Dim3Type LayerBase::Dim3()
    {
        Dim3Type dim = { 0, 0, 0 };
        return dim;
    }

    typename Tensor1::DataType& LayerBase::GetData1D()
    {
        return dummy1d;
    }

    typename Tensor1::DataType& LayerBase::GetGradient1D()
    {
        return dummy1d;
    }

    typename Tensor3::DataType& LayerBase::GetData3D()
    {
        return dummy3d;
    }

    typename Tensor3::DataType& LayerBase::GetGradient3D()
    {
        return dummy3d;
    }*/


    LayerBase* LayerBase::Previous()
    {
        return previousLayer;
    }

/*    LayerBase& LayerBase::Flatten()
    {
        return dummyLayer;
    }
*/
    std::shared_ptr<LayerBase> LayerBase::Next()
    {
        return nextLayer;
    }

/*    void LayerBase::PrintDim()
    {
        std::cout << "dummpy layer, no dimension" << std::endl;
    }*/
}
