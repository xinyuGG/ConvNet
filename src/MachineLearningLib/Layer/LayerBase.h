#pragma once
#include <memory>
#include <tuple>
#include "..\Common\Vector.h"
#include "..\Common\Matrix.h"
#include "..\Common\Blob.h"
#include "..\Common\Tensor.h"

#include <time.h>
namespace FengML
{
    class Tensor1
    {
    public:
        typedef size_t DT;
        typedef Vector<float> DataType;
    };

    class Tensor3
    {
    public:
        typedef std::tuple<size_t, size_t, size_t> DT;
        typedef std::vector<Matrix<float>> DataType;
    };

    typedef typename Tensor3::DT Dim3Type;

    class LayerBase
    {
    public:
        LayerBase() : previousLayer(nullptr), nextLayer(nullptr){}
//传入数据是一个Blob 传出是一个指针？
        LayerBase* Previous();
        std::shared_ptr<LayerBase> Next();
		virtual void Reshape(LayerBase* prelayer);

		virtual void Forward(Blob<float>& indata, Blob<float>& outputdata);
		virtual void Getlayerattri();
		virtual void Initialize(int FORWARD_ONLY = 0);//初始化weight 和 bias
		virtual void Setpath(const std::string& weightpath, const std::string& biaspath);
		virtual void Setcomputeplatform(bool gpu_only)
		{
			_GPU_ONLY = gpu_only;
		}
		LayerBase& Add(std::shared_ptr<LayerBase> _nextLayer); //负责把所有的层连在一起 初步设置每一层的数据/不依赖于上一层输出的数据
//		virtual void Reshape(LayerBase* prelayer);//根据初始化数据和上一层数据 设置本层的全部参数
//		virtual void Initialize();//根据参数执行权重初始化任务
/*        virtual size_t Dim1();
        virtual Dim3Type Dim3();
        virtual void PrintDim();*/


    protected:
		bool _GPU_ONLY;
        LayerBase* previousLayer;
        std::shared_ptr<LayerBase> nextLayer;
    };
}
