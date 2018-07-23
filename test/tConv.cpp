#include "src\MachineLearningLib\Common\Blob.h"

#include "src\MachineLearningLib\Model\CNNModel.h"
#include "src\MachineLearningLib\Data\DataSet.h"
#include <iostream>
#ifndef BATCHSIZE
#define BATCHSIZE 50
#endif
#define FORWARD_ONLY 1
#define RANGE 10
#define CLOCKS_PER_SEC ((clock_t)1000)
#define GPU_ONLY 1
#define N 100
#define hA N
#define wA N
#define hB N
#define wB N
using namespace FengML;
using namespace std;

const string traindataPath = "F:\\Desktop\\matrixMulCUBLAS\\data\\train-images.idx3-ubyte";
const string trainlablesPath = "F:\\Desktop\\matrixMulCUBLAS\\data\\train-labels.idx1-ubyte";
const string testdataPath = "F:\\Desktop\\matrixMulCUBLAS\\data\\t10k-images.idx3-ubyte";
const string testlablesPath = "F:\\Desktop\\matrixMulCUBLAS\\data\\t10k-labels.idx1-ubyte";

int main(int argc, char const *argv[])
{

	//Blob<float> kernel(1,1,hA, hB);
	//Blob<float> input(1,1,wA, wB);
	//kernel.Initialize();
	//input.Initialize();
	//Blob<float> output(1,1,hA, wB);
	//int epoch = 2000;
	//clock_t begin, end;
	//begin = clock();
	//for (int i = 0; i < epoch; i++)
	//	matrixMul_GPU(&(kernel(0,0,0, 0)), hA, wA, &(input(0,0,0, 0)), hB, wB, &(output(0,0,0, 0)));
	//end = clock();
	//int gpu_time = end - begin;


	//begin = clock();
	//for (int i = 0; i < epoch; i++)
	//	matrixMul_CPU(&(kernel(0, 0, 0, 0)), hA, wA, &(input(0, 0, 0, 0)), hB, wB, &(output(0, 0, 0, 0)));
	//end = clock();
	//int cpu_time = end - begin;
	//double GOPS;
	//GOPS = 2.0 * hA * hB * wB * epoch;
	//cout << "CPU Compute time " << (float)cpu_time / CLOCKS_PER_SEC <<" sec" <<endl;
	//cout << "CPU Compute OPS  " << (double)GOPS * 1E-09 / (cpu_time / CLOCKS_PER_SEC) << " Gops"<< endl;
	//cout << "GPU Compute OPS  " << (double)GOPS * 1E-09 / (gpu_time / CLOCKS_PER_SEC) << " Gops" << endl;
	//system("pause");
	//return 0;



	DataSet trainingSet(10), testSet(10);
	if(trainingSet.Load(traindataPath,trainlablesPath) &&
	testSet.Load(testdataPath, testlablesPath))
	{
		cout << "load data and lables successfully." << endl;
	}

	CNNModel testModel(BATCHSIZE,GPU_ONLY);
	testModel.Initialize(FORWARD_ONLY);//初始化，包括初始化输入输出层，载入各层

	clock_t begin, end;
	begin = clock();
	testModel.Forward(testSet, RANGE);//前向测试数据
	end = clock();
	cout << "batchsize = " << BATCHSIZE << " " << "range = " << RANGE << endl;
	cout << "Total time = " << (float)(end - begin)  / CLOCKS_PER_SEC <<" sec" <<endl;


	system("pause");
	return 0;

}
