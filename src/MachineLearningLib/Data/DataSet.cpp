#include "DataSet.h"
#include "..\Common\Utility.h"
#include <fstream>
#include <iostream>
using namespace std;
namespace FengML
{


	float* DataSet::Getdata(int index)
    {
        assert(index <= _allData.getBatchsize());
		return &(_allData(index, 0, 0, 0));
    }

	int DataSet::getSize() const
    {
        return _allData.getBatchsize();
    }

	OneHotVector DataSet::Gettarget(int index) const
    {
        return _targets[index];
    }

	bool DataSet::Load(const std::string& dataFile, const std::string& labelFile)
	{
		LoadData(dataFile);
		LoadLabels(labelFile);
		if(_allData.getBatchsize() != _targets.size())
			return false;
		return true;
	}

	bool DataSet::Dump(const std::string& filePath)
	{
		return true;
	}

	bool DataSet::LoadData(const std::string& dataFile)
	{
		std::ifstream fin(dataFile.c_str(), std::ifstream::binary);
		if (!fin)
			return false;

		int magicNumber;
		int number;
		fin.read((char *)&magicNumber, 4);
		magicNumber = Utility::EndiannessSwap(magicNumber);
		fin.read((char *)&number, 4);
		number = Utility::EndiannessSwap(number);
		fin.read((char *)&row, 4);
		fin.read((char *)&col, 4);
		row = Utility::EndiannessSwap(row);
		col = Utility::EndiannessSwap(col);

		_allData.Resize(number ,1,row,col);
		unsigned char* buffer = new unsigned char[row * col];
		for (int i = 1; i < number + 1; i++)
		{
			if (i % 1000 == 0)
			{
				std::cout << "\r" << i << " samples loaded" ;
			}
			fin.read((char *)buffer, row * col);
			std::copy(buffer, buffer+ row * col, &(_allData(i-1,0,0,0)));
		}
		_allData.Div(255.0);
		std::cout << endl;
		delete []buffer;
		return true;
	}

	bool DataSet::LoadLabels(const std::string& labelFile)
	{
		std::ifstream fin(labelFile.c_str(), std::ifstream::binary);
		if (!fin)
			return false;

		int magicNumber;
		int number;
		fin.read((char *)&magicNumber, 4);
		magicNumber = Utility::EndiannessSwap(magicNumber);
		fin.read((char *)&number, 4);
		number = Utility::EndiannessSwap(number);

		_targets.reserve(number + 1);
		unsigned char t;
		for (int i = 0; i < number; i++)
		{
			fin.read((char *)&t, 1);
			_targets.push_back(OneHotVector((size_t)t, _categoryNumber));
		}
		return true;
	}
}
