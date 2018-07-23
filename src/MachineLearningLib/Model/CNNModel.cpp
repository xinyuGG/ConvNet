#include "CNNModel.h"
#include <fstream>
#include <cstdlib>

namespace FengML
{

	CNNModel::CNNModel(const Configuration& config) //定义batchsize 学习速率等一系列东西
    {
		_layernum = 0;
		_pInputdata = new Blob<float>(_batchsize, 1, 28,28);
		_pGoal = new OneHotVector[_batchsize];
    }

	CNNModel::CNNModel(int batchsize ,bool GPU_ONLY)//定义batchsize 学习速率等一系列东西
	{
		_batchsize = batchsize;
		_layernum = 0;
		_pInputdata = new Blob<float>(_batchsize, 1, 28 ,28);
		_pGoal = new OneHotVector[_batchsize];
		_GPU_ONLY = GPU_ONLY;
	}

	CNNModel::~CNNModel()
	{

		auto it = _layerlist.begin();
		while(it != _layerlist.end())
		{
			delete *it;
			*it = nullptr;
			it++;
		}

		if(_pInputdata != nullptr)
			{
				delete _pInputdata;
				_pInputdata = nullptr;
			}
		if(_pGoal != nullptr)
			{
				delete[] _pGoal;
				_pGoal = nullptr;
			}
	}

	void CNNModel::Initialize(int for_back)
	{
		Setlayer();//将各层载入到层表中
		Layerinit(for_back);//各层 reshape以及权重初始化，这里可能要到外部载入数据
	}

	void CNNModel::Layerinit(int for_back)
	{
		int layernum = 0;
		auto it = _layerlist.begin();
		while(it != _layerlist.end() - 1)//跳过输入层的初始化
		{
			cout << endl;
			cout << "layer number " << layernum++ <<endl;
			(*it)->Getlayerattri();
			(*(it + 1))->Reshape(*it);
			(*(it + 1))->Initialize(for_back);
			(*(it + 1))->Setcomputeplatform(_GPU_ONLY);
			it++;
		}
		cout << endl;
		cout << "layer number " << layernum++ << endl;
		(*it)->Getlayerattri();
		cout << "=================All layers loaded===================" << endl;
	}

	void CNNModel::Setlayer()
	{
		//开始初始化各层， 使用上一层的地址
		_layerlist.push_back(new InputLayer<float>(*_pInputdata));//这是初始化层，大小和输入图像一致

		//Conv1 kernelsize = 5, num = 6 stride = 1
		_layerlist.push_back(new ConvLayer<float>(32, 5, 1, 2));
		_layerlist.back()->Setpath(W_conv1, b_conv1);
		_layernum++;

		//Pooling2 poolingsize = 2 stride = 2;
		_layerlist.push_back(new PoolingLayer<float>(2,2));
		_layernum++;

		//Conv3 size = 5*5 num = 16 stride = 1
		_layerlist.push_back(new ConvLayer<float>(64,5,1,2));
		_layerlist.back()->Setpath(W_conv2, b_conv2);
		_layernum++;

		//pooling4 size = 2*2 stride = 2
		_layerlist.push_back(new PoolingLayer<float>(2,2));
		_layernum++;

		//Fc5 outputlength = 512
		_layerlist.push_back(new FullyConnectedLayer<float>(512));
		_layerlist.back()->Setpath(W_fc1, b_fc1);
		_layernum++;

		//Fc5 outputlength = 10
		_layerlist.push_back(new FullyConnectedLayer<float>(10));
		_layerlist.back()->Setpath(W_fc2, b_fc2);
		_layernum++;


	}

	//template<class T>
	void CNNModel::Forward( DataSet& trainingSet, int totalpoch)
	{
		float total , correct ;
		total = trainingSet.getSize(); //样本总数
		assert(totalpoch * _batchsize <= total);
		correct = 0;//总正确数
		int* py_infer = new int[10000];
		float* py_softmax = new float[10 * 50 * 200];
		FILE *f;
		f = fopen("pythondata.bin", "rb");
		fread(py_infer, sizeof(int), 10000, f);
		fclose(f);

		f = fopen("softmax_py.bin", "rb");
		fread(py_softmax, sizeof(float), 10 * 50 * 200, f);
		fclose(f);
		int flag = 0;
		int py_index = 0;
		for(int poch = 0; poch < totalpoch; poch++)
			{
				Loaddata(trainingSet,_pInputdata, poch);//加载一次Forward的数据
				Blob<float> inputdata,outputdata;
				//第一层有些特殊
				auto it = _layerlist.begin() + 1;
				(*it)->Forward(*_pInputdata,outputdata);
				outputdata.Relu();
				inputdata = outputdata;
				it ++;
				while(it != _layerlist.end() - 1)
				{
					(*it)->Forward(inputdata,outputdata);
					outputdata.Relu();
					inputdata = outputdata;
					it++;
				}
				(*it)->Forward(inputdata, outputdata);
				outputdata.Softmax();//对每个batch分别求softmax， max函数返回最大得分的位置
				Loadlables(trainingSet, _pGoal, poch);
				int inferGoal = 0;//神经网络计算得到的结果
				int corrGoal = 0; //正确结果
				int correct_b = 0;//每个BATCH的正确数
				for(int i = 0; i < _batchsize; i++)
				{
					inferGoal = outputdata.Max(i);//求每个batch的输出目标
					corrGoal = _pGoal[i].HotIndex();
					//if(inferGoal == corrGoal)
					//	correct_b++;
					if(inferGoal == corrGoal)
						correct_b++;
				}
				if ((inferGoal == corrGoal) == py_infer[py_index])
				{
					flag += Compare(outputdata.getData(), py_softmax + poch * _batchsize * 10 , _batchsize * 10);
				}
				else
				{
					cout << " Data " << py_index << "inference data difference" << endl;
					flag += _batchsize * 10;
				}
				py_index++;
				std::cout<< "poch " << poch << std::endl;
				std::cout<< "Accuracy  " << (correct_b / (float)_batchsize) * 100 ;
				std::cout<< "%" <<std::endl;
				correct += correct_b;
				correct_b = 0;
			}

		float accu = (correct / totalpoch / _batchsize) * 100;
		std::cout<< "All data have been inferenced" <<std::endl;
		std::cout<< "Total Accuracy  " << accu << "%" << std::endl;
		std::cout << " With abseErr "  << absErr << " and relErr " << relErr<< " , err data " << (float)flag / 1000.0 << " %" << endl;
		Gettime();
		delete[] py_infer;
		delete[] py_softmax;
	}

	void CNNModel::Loaddata(DataSet& dataset, Blob<float>* modeldata, int poch)
	{
		int index = poch * _batchsize  ;
		auto pBegin = dataset.Getdata(index);
		auto pEnd = dataset.Getdata(index + _batchsize);
		int a = pEnd - pBegin;
		int b = sizeof(dataset.Getdata(index));
		std::copy(pBegin, pEnd, modeldata->getData());
	}

	void CNNModel::Gettime()
	{
		size_t conv_num, fc_num,pooling_num;
		conv_num = pooling_num = fc_num = 1;
		auto it = _layerlist.begin();
		while (it != _layerlist.end())
		{
			if (ConvLayer<float>* prelayer = dynamic_cast<ConvLayer<float>*>(*it))
			{
				cout << "ConvLayer " << conv_num++ <<":  "<<(float)(prelayer->getTime()) / CLOCKS_PER_SEC << " sec" << endl;
				it++;
			}
			else if (FullyConnectedLayer<float>* prelayer = dynamic_cast<FullyConnectedLayer<float>*>(*it))
			{
				cout << "FcLayer  " << fc_num++ << ":  "<<(float)(prelayer->getTime()) / CLOCKS_PER_SEC << " sec" << endl;
				it++;
			}
			else if (PoolingLayer<float>* prelayer = dynamic_cast<PoolingLayer<float>*>(*it))
			{
				cout << "PoolingLayer  " << pooling_num++ << ":  " << (float)(prelayer->getTime()) / CLOCKS_PER_SEC << " sec" << endl;
				it++;
			}
			else 
				it++;
		}
	}
	void CNNModel::Loadlables(DataSet& dataset, OneHotVector* modellables, int poch)
	{
		int index = poch * _batchsize ;
		for(int batch = 0; batch < _batchsize; batch++)
		{
			modellables[batch] = dataset.Gettarget(index);
			index++;
		}
	}

	template<class T>
	int& CNNModel::Compare(T* arr_A, T* arr_B, int length)
	{
		assert(length > 0);
		int errNum = 0;
		for (int index = 0; index < length; index++)
			if (!Isequal(arr_A[index], arr_B[index], absErr, relErr))
				errNum++;
		return errNum;
	}

	bool CNNModel::Isequal(float a, float b, float absError, float relError)
	{
		if (a == b) return true;
		if (fabs(a - b)<absError) return true;
		if (fabs(a)<fabs(b))   return (fabs((a - b) / b)<relError) ? true : false;
		else return (fabs((a - b) / a)<relError) ? true : false;
	}
 /*   void CNNModel::ClearGradient()
    {
        db = 0;
        dW = 0;
    }

    void CNNModel::Update()
    {
        db.Div((float)m_config.batchSize);
        dW.Div((float)m_config.batchSize);
        b.Sub(m_config.learning_rate, db);
        W.Sub(m_config.learning_rate, dW);
    }

    void CNNModel::ComputeGradient(const Vector<float>& x, const OneHotVector& y)
    {
        (y_diff = y_hat).Sub(y);
        db.Add(y_diff);
        dW.AddMul(y_diff, x);
    }

    size_t CNNModel::Eval(const Vector<float>& x)
    {
        // y_hat = softmax(Wx + b);
        y_hat.AssignMul(W, x).Add(b).SoftMax();
        return y_hat.Max().second;
    }

    void CNNModel::Setup()
    {
        db = Vector<float>(b.Size());
        dW = Matrix<float>(W.Row(), W.Col());
    }

    bool CNNModel::Load(const std::string& filePath)
    {
        std::ifstream fin(filePath.c_str());
        std::string description;
        fin >> description;
        fin >> b >> W;
        Setup();
        return true;
    }

    bool CNNModel::Save()
    {
        std::ofstream fout(m_config.modelSavePath.c_str());
        fout << "LRModel" << std::endl;
        fout << b << W;
        return true;
    }
*/
}
