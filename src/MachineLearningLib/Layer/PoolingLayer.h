#pragma once
#include "LayerBase.h"
#include <iostream>
#define Pooling_MAX 0
#define Pooling_AVE 1
#ifndef max(a, b)
#define max(a,b) a > b ? a :b
#endif

#ifndef min(a,b
#define min(a,b) a < b ? a :b
#endif
namespace FengML
{
	template<typename T>
    class PoolingLayer : public LayerBase
	{
	public:
		PoolingLayer(int poolingsize, int stride, int poolingtype = 0 ,int padding = 0):
		_poolingsize(poolingsize),_poolingtype(poolingtype), _stride(stride), _padding(padding)
		{
			_time = 0;
		}
		void Reshape(LayerBase* prelayer);
		void Forward(Blob<float>& indata, Blob<float>& outputdata);
		void ComputePooling(Blob<T>& inputdata, Blob<T>& outputdata);
		void Getlayerattri();
		int getWidth() const
		{
			return _output_w;
		}

		int getHeight() const
		{
			return _output_h;
		}

		int getChannel() const
		{
			return _channel;
		}

		int getBatch() const
		{
			return _batchsize;
		}

		clock_t getTime()
		{
			return _time;
		}

	protected:
		//ifmap params
		int _width;//
		int _height;
		int _channel;
		int _batchsize;

		//poolinglayer params
		int _poolingsize;
		int _stride;//
		int _padding;//
		int _poolingtype;
		//ofmaps params
		int _output_h;
		int _output_w;

		clock_t _time;
	};

	//根据上一层参数初始化pooling层参数 默认上层为Convlayer
	template<typename T>
	void PoolingLayer<T>::Reshape(LayerBase* prelayer_base)
	{
		if (ConvLayer<float>* prelayer = dynamic_cast<ConvLayer<float>*>(prelayer_base))
		{
			_width = (*prelayer).getWidth(); // 这里 步长不一定为1 padding 也要重新算
			_height = (*prelayer).getHeight();
			_channel = (*prelayer).getChannel();
			_batchsize = (*prelayer).getBatch();
		}

		_output_h = (_height - _poolingsize + 2 * _padding) / _stride + 1;
		_output_w = _output_h;

	}

	//前向计算
	template<typename T>
	void PoolingLayer<T>::Forward(Blob<float>& inputdata, Blob<float>& outputdata)
	{
		//输出BLOB的大小
		clock_t begin, end;
		begin = clock();
		ComputePooling(inputdata, outputdata);
		end = clock();
		_time += end - begin;
	}

	template<typename T>
	void PoolingLayer<T>::ComputePooling(Blob<T>& inputdata, Blob<T>& outputdata)
	{
		outputdata.Resize(_batchsize, _channel, _output_h, _output_w);
		if (_poolingtype == Pooling_MAX)
		{
			for (int batch = 0; batch < _batchsize; batch++)
			{
				for(int channel = 0; channel < _channel; channel++)
				{
					for (int ph = 0; ph < _output_h; ph++)
					{
						for (int pw = 0; pw < _output_w; pw++)
						{
							int hstart = ph * _stride - _padding;
							int wstart = pw * _stride - _padding;
							int hend = min(hstart + _poolingsize, _height);
							int wend = min(wstart + _poolingsize, _width);
							hstart = max(hstart, 0);
							wstart = max(wstart, 0);
							T flag, current;
							flag = current = 0;
							for (int h = hstart; h < hend; h++)
							{
								for (int w = wstart; w < wend; w++)
								{
									current = inputdata(batch, channel, h, w);
									flag = max(current, flag);
								}
							}
							outputdata(batch, channel, ph, pw) = flag;
						}
					}
				}
			}
		}
		else
		{
			for (int batch = 0; batch < _batchsize; batch++)
			{
				for(int channel = 0; channel < _channel; channel++)
				{
					for (int ph = 0; ph < _output_h; ph++)
					{
						for (int pw = 0; pw < _output_w; pw++)
						{
							int hstart = ph * _stride - _padding;
							int wstart = pw * _stride - _padding;
							int hend = min(hstart, _height);
							int wend = min(wstart, _width);
							hstart = min(hstart, 0);
							wstart = min(wstart, 0);
							T current = 0;
							for (int h = hstart; h < hend; h++)
							{
								for (int w = wstart; w < wend; w++)
								{
									current += inputdata(batch, channel, h, w);
								}
							}
							outputdata(batch, channel, ph, pw) =
								current / _poolingsize / _poolingsize;
						}
					}
				}
			}
		}
	}

	template<class T>
	void PoolingLayer<T>::Getlayerattri()
	{
		cout << "Poolinglayer" << endl;
		cout << "poolingsize = " << _poolingsize << "   ";
		cout << "poolingtype = " << _poolingtype<< "   ";
		cout << "padding = " << _padding << "   ";
		cout << "batchsize = " << _batchsize << endl;
	}
}
