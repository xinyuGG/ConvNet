#pragma once
#include "LayerBase.h"
#include "..\Common\Matrix.h"
#include "..\Common\Blob.h"
#include "..\Common\Vector.h"
#include <iostream>
namespace FengML
{
		template<class T>
		class Blob;

		template<class T>
		class Vector;

		template<class T>
		class Matrix;

		template<class T>
		class Math;

	template<typename T>
	class FullyConnectedLayer : public LayerBase
	{
		template<class U>
		friend class Blob;
	public:
		FullyConnectedLayer(int outputLength): _output_channel(outputLength), _pKernel(nullptr), _pBias(nullptr)
		{
			_kernelnum = _output_channel;
			_time = 0;
		}

		~FullyConnectedLayer();
		int getWidth() const
		{
			return _output_w;
		}
		void ComputeConv_4d(Blob<T>& indata, Blob<T>& kernel, Blob<T>& outputdata);
		void ComputeBias(Blob<T>& data2bias, Blob<T>& Bias);
		void Setpath(const std::string& weightpath, const std::string& biaspath);
		bool Load();
		void Tfreshape();
		int getHeight() const
		{
			return _output_h;
		}

		int getChannel() const
		{
			return _output_channel;
		}

		int getBatch() const
		{
			return _batchsize;
		}

		void Getlayerattri();
		void Forward(Blob<float>& indata, Blob<float>& outputdata);
		clock_t getTime()
		{
			return _time;
		}
		virtual void Initialize(int FORWARD_ONLY = 0);//
		void Reshape(LayerBase* prelayer);
		void Setcomputeplatform(bool gpu_only)
		{
			_GPU_ONLY = gpu_only;
		}
	protected:
		/*假设最后一个卷积层的输出为7×7×512，连接此卷积层的全连接层为1×1×4096。
		如果将这个全连接层转化为卷积层：
		1.共有4096组滤波器
		2.每组滤波器含有512个卷积核
		3.每个卷积核的大小为7×7
		4.则输出为1×1×4096
		由于每个滤波核的大小和上一层的feature map大小一样，保证了转换后的卷积层的运算结果和全连接层是一样的*/
		//上一层的所有参数
		int _width ;
		int _height ;
		int _channel;
		int _batchsize;
		//本层卷积核的参数
		int _kernel_h;
		int _kernel_w;//卷积核height width应和输入图像的一致，卷积核的channel和输入图像的一致
		int _kernelnum;//卷积核数量就是输出维度 比如例子中的4096

		const int _output_h = 1;
		const int _output_w = 1;
		int _output_channel;

		int _GPU_ONLY;
		clock_t _time;
		std::string _weightpath;
		std::string _biaspath;
		Blob<T>* _pKernel;
		Blob<T>* _pBias;
	};

	template<typename T>
	void FullyConnectedLayer<T>::Reshape(LayerBase* prelayer_base)
	{
		if(ConvLayer<float>* prelayer = dynamic_cast<ConvLayer<float>*>(prelayer_base))
		{
			_width = (*prelayer).getWidth(); // 这里 步长不一定为1 padding 也要重新算
			_height = (*prelayer).getHeight();
			_channel = (*prelayer).getChannel();
			_batchsize = (*prelayer).getBatch();
		}
		else if(PoolingLayer<float>* prelayer = dynamic_cast<PoolingLayer<float>*>(prelayer_base))
		{
			_width = (*prelayer).getWidth(); // 这里 步长不一定为1 padding 也要重新算
			_height = (*prelayer).getHeight();
			_channel = (*prelayer).getChannel();
			_batchsize = (*prelayer).getBatch();
		}
		else if(FullyConnectedLayer<float>* prelayer = dynamic_cast<FullyConnectedLayer<float>*>(prelayer_base))
		{
			_width = (*prelayer).getWidth(); // 这里 步长不一定为1 padding 也要重新算
			_height = (*prelayer).getHeight();
			_channel = (*prelayer).getChannel();
			_batchsize = (*prelayer).getBatch();
		}
		_kernel_h = _width;
		_kernel_w = _height;
		_pKernel = new Blob<T>(_kernelnum , _channel , _kernel_h ,_kernel_w);
		_pBias = new Blob<T>(1,_output_channel,1,1);
	}

	template<typename T>
	FullyConnectedLayer<T>::~FullyConnectedLayer()
	{
		if (_pKernel != nullptr)//说明new了东西
		{
			delete _pKernel;
			_pKernel = nullptr;
		}

		if (_pBias != nullptr)
		{
			delete _pBias;
			_pBias = nullptr;
		}
	}

	template<typename T>
	void FullyConnectedLayer<T>::Initialize(int for_back)
	{
		if (!for_back)
		{
			//初始化权重矩阵
			for (int index = 0; index <_kernelnum * _channel * _kernel_h * _kernel_w; index++)
				(*_pKernel)[index] = (float)index;
			//初始化偏置
			for (int index = 0; index <_output_channel; index++)
				(*_pBias)[index] = (float)index;
		}
		else
		{
			if (Load())
				std::cout << " load from file successfully";
			Tfreshape();
		}
		Transblob(_pKernel->getData(), 1, _kernelnum, _channel * _kernel_h * _kernel_w);
	}

	template<typename T>
	void FullyConnectedLayer<T>::Forward(Blob<float>& inputdata, Blob<float>& outputdata)
	{

		ComputeConv_4d(inputdata, *_pKernel, outputdata);
		ComputeBias(outputdata, *_pBias);
		//outputdata.Relu();
	}

	template<typename T>
	void FullyConnectedLayer<T>::ComputeConv_4d(Blob<T>& indata, Blob<T>& kernel, Blob<T>& outputdata)
	{

		outputdata.Resize(_batchsize, _kernelnum, _output_h, _output_w);

		int index_o = 0;
		auto pKer = kernel.getData();
		auto pData = outputdata.getData();

		clock_t begin ,end;

		if(_GPU_ONLY)
		{
			begin = clock();
			matrixMul_GPU(&(indata(0, 0, 0, 0)), _batchsize, _kernel_h * _kernel_w * _channel, pKer,
				_kernel_h * _kernel_w * _channel,   _kernelnum, pData);
		}
		else
		{
			begin = clock();
			for (int batch = 0; batch < _batchsize; batch++)
			{
				matrixMul_CPU(pKer, _kernelnum, _kernel_h * _kernel_w * _channel, &(indata(batch,0,0,0)),
					_kernel_h * _kernel_w * _channel,   _output_h * _output_w, pData);
				pData += _kernelnum * _output_h * _output_w;
				//index_o += _kernelnum * _output_h * _output_w;
			}
		}
		end = clock();
		_time += end - begin;
	}
	template<typename T>
	void FullyConnectedLayer<T>::ComputeBias(Blob<T>& data2bias, Blob<T>& Bias)
	{
		Blob<T> dil_batch(_batchsize, 1, 1, 1);
		for (int index = 0; index < _batchsize; index++)
			dil_batch[index] = 1.0f;

		clock_t begin ,end;
		begin = clock();
		if(_GPU_ONLY)
			matrixAdd_GPU(&(dil_batch[0]), _batchsize, 1, &(Bias[0]), 1,
			_kernelnum * _output_h * _output_w, &(data2bias[0]));
		else 
			matrixAdd_CPU(&(dil_batch[0]), _batchsize, 1, &(Bias[0]), 1,
				_kernelnum * _output_h * _output_w, &(data2bias[0]));
		end = clock();
		_time += end - begin;
	}


	template<typename T>
	void FullyConnectedLayer<T>::Getlayerattri()
	{
		cout << "FullyConnectedLayer" << endl;
		cout << "outputchannel = " << _output_channel << "   ";
		cout << "batchsize = " << _batchsize << endl;
	}

	template<class T>
	void FullyConnectedLayer<T>::Setpath(const std::string& weightpath, const std::string& biaspath)
	{
		_weightpath = weightpath;
		_biaspath = biaspath;
	}

	template<class T>
	bool FullyConnectedLayer<T>::Load()
	{
		FILE* f;
		f = fopen(this->_biaspath.c_str(), "rb");
		if (f == nullptr)
			return false;
		int size_b = _kernelnum * sizeof(T);
		T* bias_buf = new T[size_b];
		int a = sizeof(bias_buf);
		T* pData = _pBias->getData();
		if (fread(bias_buf, 1, size_b, f) == size_b)
			std::cout << size_b / sizeof(T) << " bias loaded" << endl;

		std::copy(bias_buf, bias_buf + _kernelnum, pData);
		delete[] bias_buf;
		fclose(f);

		f = fopen(this->_weightpath.c_str(), "rb");
		if (f == nullptr)
			return false;
		int size_w = _kernelnum * _channel * _kernel_h * _kernel_w * sizeof(T);
		T* weight_buf = new T[size_w];
		pData = _pKernel->getData();
		if (fread(weight_buf, 1, size_w, f) == size_w)
			std::cout << size_w / sizeof(T) << " weight loaded" << endl;
		std::copy(weight_buf, weight_buf + _kernelnum * _channel * _kernel_h * _kernel_w, pData);
		fclose(f);
		delete[] weight_buf;

		return true;
	}
	template<class T>
	void FullyConnectedLayer<T>::Tfreshape()
	{
		Blob<T> Temp = *_pKernel;
		Temp.Resize(_kernel_h, _kernel_w, _channel, _kernelnum);
		int index = 0;
		for (int height = 0; height < _kernel_h;height++)
			for (int width = 0; width < _kernel_w; width++)
				for (int channel = 0; channel < _channel; channel++)
					for (int num = 0; num < _kernelnum; num++)
						(*_pKernel)(num, channel, height, width) = Temp(height, width, channel, num);

	}
}
