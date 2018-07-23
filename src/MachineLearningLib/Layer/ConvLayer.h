#pragma once
#include "LayerBase.h"
//#include "..\cudalib\cuda.h"
#include "..\Common\Math.h"
#include "..\Common\Matrix.h"
#include <iostream>
#include <typeinfo>
#include <cstdio>


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
	class ConvLayer : public LayerBase
	{
		template<class U>
		friend class Blob;

	public:
		ConvLayer() {}
		ConvLayer(int Kernelnum, int kernelsize,
			int stride = 1, int padding = 0) :
			_kernelnum(Kernelnum), _stride(stride), _kernelsize(kernelsize),
			_padding(padding)
		{
			_pKernel = nullptr;
			_pBias = nullptr;
			_time = 0;
		}
		~ConvLayer();

		void Forward(Blob<float>& indata, Blob<float>& outputdata);

		virtual void Initialize(int FORWARD_ONLY = 0);//初始化weight 和 bias
		void Reshape(LayerBase* prelayer);//开辟weight空间，四维， num channel height  width
		void ComputeConv_3d(T* data2mat, T* ker2mat, T* resmat,
			int channels, int height, int width, int batchsize = BATCHSIZE);

		void ComputeConv_4d(Blob<T>& indata, Blob<T>& kernel, Blob<T>& outputdata);
		void ComputeBias(Blob<T>& data2bias, Blob<T>& Bias);

		bool Load();
		void Getlayerattri();
		void Tfreshape();
		int getWidth() const
		{
			return _output_w;
		}

		int getHeight() const
		{
			return _output_h;
		}

		int getBatch() const
		{
			return _batchsize;
		}

		int getChannel() const
		{
			return _kernelnum;
		}

		int getBatchsize() const
		{
			return _batchsize;
		}
		void Setcomputeplatform(bool gpu_only)
		{
			_GPU_ONLY = gpu_only;
		}
		clock_t getTime()
		{
			return _time;
		}

		void Setpath(const std::string& weightpath, const std::string& biaspath);

	private:
		int _width;// ifmap width
		int _height;
		int _channel;
		int _batchsize;
		int _kernelsize; // output channels
		int _kernelnum; // filter size
		int _stride;//卷积步长
		int _padding;//边缘补偿
		int _output_h;
		int _output_w;

		int _GPU_ONLY;
		clock_t _time;
		std::string _weightpath;
		std::string _biaspath;
		Blob<T>* _pKernel; //卷积核 四维 装的指针 在Reshape中初始化
		Blob<T>* _pBias; //偏置
	};


	template<typename T>
	ConvLayer<T>::~ConvLayer()
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
	void ConvLayer<T>::Initialize(int FORWARD_ONLY = 0)
	{
		//初始化kernel
		if (!FORWARD_ONLY)
		{
			for (int index = 0; index < _kernelnum; index++)
			{
				for (int i = 0; i < _channel * _kernelsize * _kernelsize; i++)
				{
					(*_pKernel)[i + index * _channel * _kernelsize * _kernelsize] = index + 1;

				}
			}

			//初始化bias
			for (int index = 0; index < _kernelnum; index++)
				for(int h = 0; h < _output_h ; h ++)
					for(int w = 0; w < _output_w; w++)
						(*_pBias)(0,index,h,w) = (float)index;
		}
		else
		{
			if (Load())
				std::cout << " load from file successfully";
			Tfreshape();
		}
		Transblob(_pKernel->getData(), 1, _kernelnum, _kernelsize * _kernelsize * _channel);
	}

	template<typename T>
	void ConvLayer<T>::Reshape(LayerBase* prelayer_base)
	{

		if (ConvLayer<float>* prelayer = dynamic_cast<ConvLayer<float>*>(prelayer_base))
		{
			_width = (*prelayer)._output_w; // 这里 步长不一定为1 padding 也要重新算
			_height = (*prelayer)._output_h;
			_channel = (*prelayer)._kernelnum;
			_batchsize = (*prelayer)._batchsize;
		}
		else if (PoolingLayer<float>* prelayer = dynamic_cast<PoolingLayer<float>*>(prelayer_base))
		{
			_width = (*prelayer).getWidth(); // 这里 步长不一定为1 padding 也要重新算
			_height = (*prelayer).getHeight();
			_channel = (*prelayer).getChannel();
			_batchsize = (*prelayer).getBatch();

		}

		else if (InputLayer<float>* prelayer = dynamic_cast<InputLayer<float>*>(prelayer_base))
		{
			_width = (*prelayer).getWidth();
			_height = (*prelayer).getHeight();
			_channel = (*prelayer).getChannel();
			_batchsize = (*prelayer).getBatch();
		}
		_output_h = (_height - _kernelsize + 2 * _padding) / _stride + 1;//卷积层单通道输出图像height
		_output_w = _output_h;//卷积层单通道输出图像width;
		_pKernel = new Blob<T>(_kernelnum , _channel , _kernelsize , _kernelsize);
		_pBias = new Blob<T>(1 , _kernelnum, _output_h , _output_w);//初始化偏置
	}


	inline bool is_a_ge_zero_and_a_lt_b(int a, const int b)
	{//若a大于等于零且小于b，返回true，否则返回false
	 //这里隐含一个前提 即b 一定是大于0的 如果a小于0 最高位为1 一定大 false
		return static_cast<unsigned>(a) < static_cast<unsigned>(b);
	}

	/*
	这个函数处理的是一个四维向量，可以是一个输入的图像BLOB，
	也可以是一个卷积核（要处理多个卷积核就要调用多次这个函数）
	channels 是输入图像的通道数，height width是输入图像的二维信息，
	padding是填充大小，stride是步长，改变后的数据放在data_col中
	*/


	template<typename T>
	void ConvLayer<T>::Forward(Blob<float>& inputdata, Blob<float>& outputdata)
	{
		ComputeConv_4d(inputdata, *_pKernel, outputdata);//卷积
		ComputeBias(outputdata, *_pBias);//偏置
	}
	template<typename T>
	//bias的维度是
	void ConvLayer<T>::ComputeBias(Blob<T>& data2bias, Blob<T>& Bias)
	{
		Blob<T> dil_batch(_batchsize, 1, 1, 1);

		for (int index = 0; index <  _batchsize; index++)
			dil_batch[index] = 1.0f;
		clock_t begin ,end;
		begin = clock();
		if(_GPU_ONLY)
			matrixAdd_GPU(&(dil_batch[0]), _batchsize, 1, &(Bias[0]),	1,
			 _kernelnum * _output_h * _output_w,  &(data2bias[0]));
		else
			matrixAdd_CPU(&(dil_batch[0]), _batchsize, 1, &(Bias[0]), 1,
				_kernelnum * _output_h * _output_w, &(data2bias[0]));
		end = clock();
		_time += end - begin;
	}

	template<typename T>
	//完成一个卷积核与输入数据卷积的操作 输出的Blob batch不指定，channel不指定
	//指定只有height和width;
	void ConvLayer<T>::ComputeConv_3d(T* data2mat, T* ker2mat, T* resmat,
		int channels, int height, int width, int batchsize = BATCHSIZE)
	{

			//矩阵乘法函数；返回一维数组 再赋值
			//是kernel*data

		matrixMultiply(ker2mat, 1, _kernelsize * _kernelsize* _channel, data2mat,
				_kernelsize * _kernelsize *_channel, _output_h * _output_w, resmat);


		//col2im(&tempmat[0], channels, _output_h, _output_w, _kernelsize, _padding,
		//	_stride, resmat);
	}

	template<typename T>
	//完成所有卷积核与输入数据的卷积操作，调用输入数据与一个卷积核卷积的函数
	void ConvLayer<T>::ComputeConv_4d(Blob<T>& indata, Blob<T>& kernel, Blob<T>& outputdata)
	{
		int output_channel = _kernelnum;
		int kerneltrans_h = output_channel;
		int kerneltrans_w = _kernelsize * _kernelsize* _channel;
		int datatrans_h = _kernelsize * _kernelsize *_channel;
		int datatrans_w = _output_h * _output_w;

		//首先 卷积核和图像数据都是四维的，而卷积单位是三维，所以要降维处理
		//将四维数据分别im2col放在数组里面，然后两个for循环做卷积
		Blob<T> data2mat(_batchsize, 1, datatrans_h, datatrans_w);
		//data2mat.Resize(_batchsize, datatrans_h* datatrans_w);这样初始化会触发断点


		int index_d, index_k;
		index_d = index_k = 0;
		//转换data数据
		for (int batch = 0; batch < _batchsize; batch++)
		{
			//存放转为矩阵形式的data
			//这里面inputblob 是batch*下面数据，所以每次转换时候要增加index_d
			//转换的数据放在data2mat[batch]里
			index_d = batch * _channel * _height * _width;
			im2col(&(indata[index_d]), _channel, _height, _width, _kernelsize,
				_padding, _stride, &(data2mat(batch, 0,0,0))); //先转换data
		}
		Transblob(&(data2mat(0, 0,0,0)), _batchsize, datatrans_h, datatrans_w);

		//下面两层循环调用三维卷积
		outputdata.Resize(_batchsize, _kernelnum, _output_h, _output_w);
		data2mat.Resize(_batchsize, 1, datatrans_w, datatrans_h);
		int index_o = 0;
		auto pKer = kernel.getData();
		auto pData = outputdata.getData();		
		clock_t begin, end;
		if(this->_GPU_ONLY)
		{
			begin = clock();
			matrixMul_GPU(&(data2mat(0,0,0,0)),  _batchsize * datatrans_w, datatrans_h, pKer,
					kerneltrans_w, kerneltrans_h, pData);
			Transblob(pData, _batchsize, _output_h * _output_w, _kernelnum);
		}
		else
		{
			begin = clock();
			for (int batch = 0; batch < _batchsize; batch++)
			{
				matrixMul_CPU(pKer, kerneltrans_h, _kernelsize * _kernelsize* _channel, &(data2mat(batch, 0, 0, 0)),
					_kernelsize * _kernelsize* _channel, _output_h * _output_w, pData);
				pData += _kernelnum * _output_h * _output_w;//这里最后pData的指向会越界 但是不用这个东西就没关系
											//index_o += _kernelnum * _output_h * _output_w;
			}
		}
		end = clock();
		_time += end - begin;
	}


	template<class T>
	void ConvLayer<T>::Getlayerattri()
	{
		cout << "Convlayer" << endl;
		cout << "kernelsize = " << _kernelsize << "   ";
		cout << "kernelnum = " << _kernelnum << "   ";
		cout << "padding = " << _padding << "   ";
		cout << "batchsize = " << _batchsize << endl;

	}

	template<class T>
	void ConvLayer<T>::Setpath(const ::string& weightpath, const std::string& biaspath)
	{
		_weightpath = weightpath;
		_biaspath = biaspath;
	}

	template<class T>
	bool ConvLayer<T>::Load()
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

		Blob<T> dil(1, 1, _output_h, _output_w);
		for (int i = 0; i < _output_h * _output_w; i++)
			dil[i] = 1.0f;
		matrixMul_GPU(bias_buf, _kernelnum, 1, &(dil[0]), 1, _output_h * _output_w, pData);
		//std::copy(bias_buf, bias_buf + _kernelnum, pData);
		delete[] bias_buf;
		fclose(f);

		f = fopen(this->_weightpath.c_str(), "rb");
		if (f == nullptr)
			return false;
		int size_w = _kernelnum * _channel * _kernelsize * _kernelsize * sizeof(T);
		T* weight_buf = new T[size_w];
		pData = _pKernel->getData();
		if (fread(weight_buf, 1, size_w, f) == size_w)
			std::cout << size_w / sizeof(T) << " weight loaded" << endl;
		std::copy(weight_buf, weight_buf + _kernelnum * _channel * _kernelsize * _kernelsize, pData);
		fclose(f);
		delete[] weight_buf;

		return true;
	}

	template<class T>
	void ConvLayer<T>::Tfreshape()
	{
		Blob<T> Temp = *_pKernel;
		Temp.Resize(_kernelsize, _kernelsize, _channel, _kernelnum);
		int index = 0;
		for (int height = 0; height < _kernelsize;height++)
			for (int width = 0; width < _kernelsize; width++)
				for (int channel = 0; channel < _channel; channel++)
					for (int num = 0; num < _kernelnum; num++)
						(*_pKernel)(num, channel, height, width) = Temp(height, width, channel, num);

	}



	/*    void ConvLayer::backward()
	{
	PrintDim();
	// to do
	}*/


}
