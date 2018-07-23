#pragma once
#include <string>
#include <vector>
#include "..\Common\Vector.h"
#include "..\Common\Blob.h"
#include "..\Common\OneHotVector.h"
namespace FengML
{
    class DataSet
    {
    public:
        DataSet(int categoryNumber):_categoryNumber(categoryNumber){};
        virtual bool Load(const std::string& dataFile,
            const std::string& labelFile);
		float* Getdata(int index) ;
        OneHotVector Gettarget(int index) const;
		int getSize() const;
		int size()
		{
			return _allData.getBatchsize();
		}

		bool LoadData(const std::string& dataFile);
		bool LoadLabels(const std::string& labelFile);
	 	bool Dump(const std::string& filePath);
    protected:

        Blob<float> _allData;//存放所有输入数据文件
        std::vector<OneHotVector> _targets;//存放所有输入标签文件
        int _categoryNumber;
		int row;
		int col;
    };
}
