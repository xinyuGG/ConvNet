#pragma once
#include <vector>
#include "OneHotVector.h"
#include <ctime>
#include <cmath>
#include <fstream>
#include <cassert>
#include <cstdlib>
namespace FengML
{
    template<class T>
    class Vector;

    template<class T>
	class Matrix;

	template<class T>
    class Tensor // _channel * _height * _width
    {
    public:
        template<class U>
		friend class Matrix;

		template<class U>
        friend class Vector;

		Tensor();
		Tensor(size_t channel, size_t height, size_t width);
		Tensor(const Tensor<T>& other);
		Tensor(Tensor<T>&& other);
		virtual ~Tensor();
		Tensor& operator = (const Tensor<T>& other);
		Tensor& operator = (Tensor<T>&& other);
		T operator() (size_t channel, size_t height, size_t width) const
		//Tensor(3,4,5) 5*5*5,
		{
			assert(channel < _channel && width < _width && height < _height);
			return _data[channel * _width * _height + height * _width + width];
		}

		T& operator() (size_t channel, size_t height, size_t width)
		//Tensor(3,4,5) 5*5*5,
		{
			assert(channel < _channel && width < _width && height < _height);
			return _data[channel * _width * _height + height * _width + width];
		}
		std::tuple<size_t,size_t,size_t> GetDim(const Tensor<T> tensor);
	private:
		T* _data;
		size_t _width;
		size_t _height;
		size_t _channel;

	};

	template<class T>
	Tensor<T>::Tensor(): m_data(nullptr), _channel(0), _width(0), _height(0)
	{

	}
	template<class T>
	Tensor<T>::Tensor(size_t channel, size_t height, size_t width):
	_channel(channel), _width(width), _height(height)
	{
		_data = new T[_channel* _width* _height];
		 memset(_data, 0, _channel* _width* _height);
	}

	template<class T>
	Tensor<T>::Tensor(const Tensor<T>& other)
	{
		_channel = other._channel;
		_width = other._width;
		_height = other._height;
		_data = new T[_channel* _width* _height];

		std::copy(other._data, other._data + _channel* _width* _height, _data);
	}

	template<class T>
	Tensor<T>::Tensor(Tensor<T>&& other)
	{
		_channel = other._channel;
		_width = other._width;
		_height = other._height;
		_data = other._data;
		other._data = nullptr;
	}

	template<class T>
	Tensor<T>::~Tensor()
	{
		if (_data != nullptr)
			delete[] _data;
	}

	template<class T>
	std::tuple<size_t,size_t,size_t> Tensor<T>::GetDim(const Tensor<T> tensor)
	{
		auto res = std::make_tuple<size_t,size_t,size_t>(tensor._channel,
		tensor._height, tensor._width)
		return res;
	}

	template<class T>
	Tensor<T>& Tensor<T>:: operator = (const Tensor<T>& other)
	{
		if(this != &other)
		{
			delete[] _data;
			_channel = other._channel;
			_width = other._width;
			_height = other._height;
			_data = other._data;
			_data = new T[_channel* _width* _height];
			std::copy(other._data, other._data + _channel* _width* _height, _data);
		}
		return *this;
	}

	template<class T>
	Tensor<T>& Tensor<T>:: operator = (Tensor<T>&& other)
	{
		if(this != &other)
		{
			_channel = other._channel;
			_width = other._width;
			_height = other._height;
			_data = other._data;
			_data = other._data;
			other._data = nullptr;
		}
		return *this;
	}

	
}
