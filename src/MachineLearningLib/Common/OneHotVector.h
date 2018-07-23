#pragma once
namespace FengML
{
    class OneHotVector
    {
    public:
        OneHotVector() :m_index(0),m_len(10){}
        OneHotVector(size_t index, size_t len) : m_len(len), m_index(index) {}
        virtual ~OneHotVector() {}
        size_t Size() const
        {
            return this->m_len;
        }
        size_t HotIndex() const
        {
            return m_index;
        }
		void SetIndex(int index)
		{
			m_index = index;
		}

		OneHotVector& operator = (const OneHotVector other)
		{
			m_index = other.m_index;
			m_len = other.m_len;
			return *this;
		}
		
        template<class T>
        friend class Vector;

        template<class T>
        friend class Matrix;
    private:
        size_t m_index;
        size_t m_len;
    };
}
