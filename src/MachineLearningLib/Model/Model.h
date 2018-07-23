#pragma once
#include <string>
#include <memory>
#include <iostream>
#include "..\Data\DataSet.h"
#include "..\Config\Configuration.h"
#include "..\Layer\LayerBase.h"
#include "..\Common\Blob.h"
namespace FengML
{
    class Model
    {
    public:
		Model() = default;
        Model(const Configuration& _config) : m_config(_config) {}
       // void virtual Fit(const DataSet& trainingSet, const DataSet& validateSet);
        float Test( DataSet& dataSet, float& loss);
        virtual size_t Eval(const Vector<float>& data) = 0;
        virtual bool Load(const std::string& filePath) = 0;
        virtual bool Save() = 0;
        virtual void Update() = 0;
        virtual float Loss(const OneHotVector& y);
        virtual void ClearGradient() = 0;
        virtual void ComputeGradient(const Vector<float>& x, const OneHotVector& y) = 0;
		void Forward(std::shared_ptr<LayerBase> layer);
		//void Backward(LayerBase* layer);
    protected:
        const Configuration& m_config;
        Vector<float> y_hat;
		LayerBase* _lastLayer;
		std::shared_ptr<LayerBase> _firstLayer;
    };
}

//首先建立model  add 输入大小 每一层的大小都要知道
//然后初始化model
//然后读取数据
