#include "Model.h"
#include "LRModel.h"
#include <iostream>
namespace FengML
{
    float Model::Loss(const OneHotVector& y)
    {
        return y_hat.CrossEntropyError(y);
    }

    // Use SGD
    //
 //   void Model::Fit(const DataSet& trainingSet, const DataSet& validateSet)
 //   {

  ///  }

    float Model::Test(DataSet& dataSet, float& loss)
    {
        size_t total = dataSet.size();
        size_t correct = 0;
        loss = 0;
 /*       for (int i = 0; i < total; i++)
        {
			auto predicted = Eval(dataSet.Getdata(i));
            if (predicted == dataSet.Gettarget(i).HotIndex())
            {
                correct++;
            }

            loss += Loss(dataSet.Gettarget(i));
        }
		*/
        return correct / (float)total;
    }

	void Model::Forward(std::shared_ptr<LayerBase> layer)
	{
		std::cout << "forward" << std::endl;
		//Blob  inputdata , outputdata;
	
	}

/*	void Model::Backward(LayerBase* layer)
	{
		std::cout << "backward" << std::endl;
		while (layer != nullptr)
		{
			layer->backward();
			layer = layer->Previous();
		}
	}
*/
}
