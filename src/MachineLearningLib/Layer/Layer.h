#pragma once
#include "..\Common\Vector.h"
#include "..\Common\Matrix.h"
#include "LayerBase.h"
#include <vector>
#include <memory>
namespace FengML
{
    template<typename T>
    class Layer : public LayerBase
    {
    public:
        Layer(){}

		//virtual Blob<T> Forward(Blob<T>& inputdata);


    };


}
