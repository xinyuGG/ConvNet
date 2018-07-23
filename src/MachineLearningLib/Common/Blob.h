#pragma once
#include <vector>
#include "OneHotVector.h"
#include "..\cudalib\cuda.h"
#include <ctime>
#include <cmath>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#define Getmax(a, b)  a > b ? a :b
using namespace std;
namespace FengML
{


    template<class T>
	class Matrix;

	template<class T>
	class Tensor;

	template<class T>
    class Blob // _channel * _height * _width
    {
    public:
        template<class U>
		friend class Matrix;


		template<class U>
		friend class Tensor;

		Blob();
		Blob(int num, int channel, int height, int width);
		Blob(const Blob<T>& other);
		Blob(Blob<T>&& other);
		virtual ~Blob();
		Blob& operator = (const Blob<T>& other);
		Blob& operator = (Blob<T>&& other);
		void Tfcout();
		Blob& Add(Blob<T>& other);
		Blob& Softmax();
		void Relu();
		int Max(int batchindex);
		Blob& Resize(int num, int channel, int height, int width);
		void Div(float i );
		T operator() (int num ,int channel, int height, int width) const
		//Blob(3,4,5) 5*5*5,
		{
			assert(num * channel * height * width <= _num * _channel * _height * _width);
			//这里使用等号，在max函数中要使用内存最后一位的下一位
			return _data[num * _channel * _height * _width
			+ channel * _width * _height
			+ height * _width
			+ width];
		}

		void Cout();
		T& operator() (int num, int channel, int height, int width)
		{
			assert(num * channel * height * width <= _num * _channel * _height * _width);
			//这里使用等号，在max函数中要使用内存最后一位的下一位
			return _data[num * _channel * _height * _width
				+ channel * _width * _height
				+ height * _width
				+ width];
		}


		void Initialize();

		T& operator []( int index) ;
		T operator [] ( int index) const;



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

		int getBatchsize() const
		{
			return _num;
		}

		T* getData() const
		{
			return _data;
		}
		std::tuple<int,int,int> GetDim(const Blob<T>& Blob);

	protected:

		T* _data;
		int _width;
		int _height;
		int _channel;
		int _num;

	};

	template<class T>
	Blob<T>::Blob(): _data(nullptr),_num(0), _channel(0), _width(0), _height(0)
	{

	}
	template<class T>
	Blob<T>::Blob(int num ,int channel, int height, int width):
	_num(num),_channel(channel), _width(width), _height(height)
	{
		_data = new T[_num * _channel * _width * _height];
		 memset(_data, 0, _num * _channel* _width* _height);
	}

	template<class T>
	Blob<T>::Blob(const Blob<T>& other)
	{
		_channel = other._channel;
		_width = other._width;
		_height = other._height;
		_num = other._num;
		_data = new T[_num * _channel* _width* _height];

		std::copy(other._data, other._data + _num * _channel* _width* _height, _data);
	}

	template<class T>
	Blob<T>::Blob(Blob<T>&& other)
	{
		_num = other._num;
		_channel = other._channel;
		_width = other._width;
		_height = other._height;
		_data = other._data;
		other._data = nullptr;
	}

	template<class T>
	Blob<T>::~Blob()
	{
		if (_data != nullptr)
		{
			delete[] _data;
			_data = nullptr;
		}
	}

	template<class T>
	std::tuple<int,int,int> Blob<T>::GetDim(const Blob<T>& Blob)
	{
		auto res = std::make_tuple<int,int,int>(Blob._channel,
		Blob._height, Blob._width)
		return res;
	}

	template<class T>
	Blob<T>& Blob<T>:: operator = (const Blob<T>& other)
	{
		if(this != &other)
		{
			if(_data != nullptr)
				delete[] _data;
			_num = other._num;
			_channel = other._channel;
			_width = other._width;
			_height = other._height;
			_data = new T[_num* _channel* _width* _height];
			std::copy(other._data, other._data +_num * _channel* _width* _height, _data);
		}
		return *this;
	}
	template<class T>
	T& Blob<T>::operator [] ( int index)
	{
		assert(index < _num * _channel * _height * _width);
		return this->_data[index];
	}

	template<class T>
	T Blob<T>::operator [](int index)const
	{
		assert(index < _num * _channel * _height * _width);
		return this->_data[index];
	}

	template<class T>
	Blob<T>& Blob<T>:: operator = (Blob<T>&& other)
	{
		if(this != &other)
		{
			_num = other._num;
			_channel = other._channel;
			_width = other._width;
			_height = other._height;
			_data = other._data;
			other._data = nullptr;
		}
		return *this;
	}
	template<class T>
	void Blob<T>::Tfcout()
	{

			//std::cout << "Batch " << batch << endl;
				for (int height = 0; height < _height; height++)
				{
					for (int width = 0; width < _width; width++)
					{
						for (int channel = 0; channel < _channel; channel++)
						{
							for (int batch = 0; batch < _num; batch++)
							{
								std::cout << setiosflags(ios::fixed) << setprecision(8) << (*this)(batch, channel, height, width) << " \t";
							}

						}
						std::cout <<endl;

					}
					std::cout << "==================================" << endl;
				}
				std::cout << endl;

	}

	template<class T>
	void Blob<T>::Cout()
	{
		for (int batch = 0; batch < _num; batch++)
		{
			std::cout << "Batch " << batch << endl;
			for (int channel = 0; channel < _channel; channel++)
			{
				std::cout << "  Channel " << channel << endl;
				for (int height = 0; height < _height; height++)
				{
					for (int width = 0; width < _width; width++)
					{
						std::cout << setiosflags(ios::fixed) << setprecision(8) << (*this)(batch, channel, height, width) << " \t";
					}
					std::cout << endl;
				}
				std::cout << endl;
			}
			std::cout << endl;
		}


	}
	template<class T>
	Blob<T>& Blob<T>::Resize(int num, int channel, int height, int width)
	{
		if (num * channel * height * width == _num * _channel * _height * _width)
		{
			_num = num;
			_channel = channel;
			_height = height;
			_width = width;
		}
		else
		{
			if (_data != nullptr)
				delete[] _data;
			_num = num;
			_channel = channel;
			_height = height;
			_width = width;
			_data = new T[_num * _channel * _height  * _width];
		}

		return *this;
	}

	template<class T>
	Blob<T>& Blob<T>::Add(Blob<T>& other)
	{
		assert(_data != nullptr && other._data != nullptr);
		assert(_num == other._num && _channel == other._channel
		&& _height == other._height && _width == other._width);

		for(int i = 0; i < _num * _channel * _height * _width; i ++)
			(*this)[i] += other[i];
		return *this;
	}

	template<class T>
	void Blob<T>::Relu()
	{
		for (int i = 0; i < _num * _channel * _height * _width; i++)
			(*this)[i] = Getmax((*this)[i], 0);
	}

	template<class T>
	Blob<T>& Blob<T>::Softmax()
	{
		if(this->_height != 1 && this->_width ==1)
		{
			std::cout << "Softmax inputdata invalid";
		}
		else
		{
			for(int batch = 0; batch <_num; batch++)
			{
				float total = 0;
				for(int channel = 0; channel < _channel; channel++)
				{
					(*this)(batch,channel,0,0) = exp((*this)(batch,channel,0,0));
					total += (*this)(batch,channel,0,0);
				}
				total = 1.0f / total;
				for(int channel = 0; channel < _channel; channel++)
				{
					(*this)(batch,channel,0,0) *= total;
				}
			}
		}
		return *this;
	}

	template<class T>
	int Blob<T>::Max(int batchindex)
	{
		T* pFirst = &((*this)(batchindex,0,0,0));
		T* pEnd = &((*this)(batchindex + 1,0,0,0));
		T* pFlag = std::max_element(pFirst,pEnd);
		return (int)(pFlag - pFirst);
	}

	template<class T>
	void Blob<T>::Div(float i)
	{
		Blob<float> div(1, 1, 1, 1);
		Blob<float> other = *this;
		div[0] = 1.0f / i;
		matrixMul_GPU(&(div[0]), 1, 1, other._data, 1, _num * _channel * _height * _width, this->_data);
	}

	template<class T>
	void Blob<T>::Initialize()
	{
		for (int batch = 0; batch < _num; batch++)
			for (int channel = 0;channel < _channel; channel++)
			{
				float index = 0;
				for (int height = 0; height < _height; height++)
					for (int width =0 ; width < _width; width++)
						(*this)(batch, channel, height, width) = index++;
			}	
	}

}
