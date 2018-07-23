#pragma once
#include "Model.h"
#include "..\Common\OneHotVector.h"
#include "..\data/DataSet.h"
#include <string>
#include "..\Layer\LayerBase.h"
#include <iostream>
#include "src\MachineLearningLib\Layer\ConvLayer.h"
#include "src\MachineLearningLib\Layer\PoolingLayer.h"
#include "src\MachineLearningLib\Layer\InputLayer.h"
#include "src\MachineLearningLib\Layer\FulllyConnectedLayer.h"
#define absErr 1e-9
#define relErr 1e-5

#define W_conv1 "F:\\Desktop\\matrixMulCUBLAS\\data\\W_conv1.bin"
#define W_conv2 "F:\\Desktop\\matrixMulCUBLAS\\data\\W_conv2.bin"
#define W_fc1 "F:\\Desktop\\matrixMulCUBLAS\\data\\W_fc1.bin"
#define W_fc2 "F:\\Desktop\\matrixMulCUBLAS\\data\\W_fc2.bin"

#define b_conv1 "F:\\Desktop\\matrixMulCUBLAS\\data\\b_conv1.bin"
#define b_conv2 "F:\\Desktop\\matrixMulCUBLAS\\data\\b_conv2.bin"
#define b_fc1 "F:\\Desktop\\matrixMulCUBLAS\\data\\b_fc1.bin"
#define b_fc2 "F:\\Desktop\\matrixMulCUBLAS\\data\\b_fc2.bin"

namespace FengML
{
    class CNNModel
    {
    public:
		CNNModel() = default;
        CNNModel(int batchsize, bool GPU_ONLY) ;
        CNNModel(const Configuration& config);
		~CNNModel();

		void Setlayer();//设置模型各层参数

		void Loadlables( DataSet& dataset, OneHotVector* modellables, int poch);
		void Loaddata( DataSet& dataset, Blob<float>* modeldata, int poch);
		void Forward(DataSet& trainingSet,int totalpoch);
		void Layerinit(int forward_only = 0);
		void Initialize(int forward_only = 0);
		template<class T>
		int& Compare(T* arr_A, T* arr_B, int length);
		bool Isequal(float a, float b, float absError, float relError);
		void Gettime();
	/*	size_t Eval(const Vector<float>& data) override;
		void ClearGradient() override;
		bool Load(const std::string& filePath) override;
		bool Save() override;
		void ComputeGradient(const Vector<float>& x, const OneHotVector& y) override;
		void Update() override;
		void Fit(const DataSet& trainingSet, const DataSet& validateSet);

		void Setup();
	*/
    private:
        Matrix<float> W;
        Vector<float> b;
        Matrix<float> dW;
        Vector<float> db;
        Vector<float> y_diff;
//
		int _layernum;
		std::vector<LayerBase*> _layerlist;
		int _batchsize;
		Blob<float>*  _pInputdata; //可以用来初始化Inputlayer 然后初始化Convlayer
		//也可以用作加载输入数据
		OneHotVector* _pGoal;
		bool _GPU_ONLY;
    };

}
