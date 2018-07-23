#pragma once
#include "LayerBase.h"
#include <iostream>
namespace FengML
{

	template<class T>
	class InputLayer: public LayerBase
    {
    public:
        InputLayer(): _width(0),_height(0), _channel(0), _batchsize(0), _data(nullptr){}
		InputLayer(Blob<T>& imgdata);
		void Resize(const Blob<T>& imgdata);
		~InputLayer();
		void Getlayerattri();
		int getWidth() const
		{
			return _width;
		}

		int getHeight() const
		{
			return _height;
		}

		int getChannel() const
		{
			return _channel;
		}

		int getBatch() const
		{
			return _batchsize;
		}


	private:
		int _width;// ifmap width
		int _height;
		int _channel;
		int _batchsize;
		T* _data;
    };

	template<class T>
	void InputLayer<T>::Resize(const Blob<T>& imgdata)
	{
		int width = imgdata.getWidth();
		int height = imgdata.getHeight();
		int channel = imgdata.getChannel();

		_width = width;
		_height = height;
		_channel = channel;
		_batchsize = imgdata.getBatchsize();
		int imgsize = _batchsize* _channel * _height * _width;
		_data = new T[imgsize];
		std::copy(imgdata.getData(), imgdata.getData() + imgsize, _data);
	}

	template<class T>
	InputLayer<T>::InputLayer(Blob<T>& imgdata)
	{
		int width = imgdata.getWidth();
		int height = imgdata.getHeight();
		int channel = imgdata.getChannel();
		_batchsize = imgdata.getBatchsize();

		_width = width;
		_height = height;
		_channel = channel;

		int imgsize = _batchsize* _channel * _height * _width;
		_data = new T[imgsize];
		std::copy(imgdata.getData(), imgdata.getData() + imgsize, _data);
	}

	template<class T>
	InputLayer<T>::~InputLayer()
	{
		if (_data != nullptr)
		{
			delete[] _data;
			std::cout << "call Input kernel destructor"<<endl;
			_data = nullptr;
		}
	}

	template<class T>
	void InputLayer<T>::Getlayerattri()
	{
		cout << "Inputlayer" << endl;
		cout << "width = " << _width << "   ";
		cout << "height = " << _height << "   ";
		cout << "channel = " << _channel << "   ";
		cout << "batchsize = " << _batchsize << endl;
	}
}
