#pragma once
namespace FengML
{
	template<typename T>
	void im2col(const T* data_im, const int channels,
		const int height, const int width, const int kernelsize,
		const int padding, const int stride, T* data_col)
	{
		int output_h = (height - kernelsize + 2 * padding) / stride + 1;
		int output_w = output_h;
		int channel_size = height * width;//
		int index = 0;
		for (int channel = channels; channel--; data_im += channel_size)
			//最外层循环 每次递增单通道图像的大小，比如通道0循环完，进入通道1
		{
			for (int kernel_row = 0; kernel_row < kernelsize; kernel_row++)//实质上是kernel_height
			{
				for (int kernel_col = 0; kernel_col < kernelsize; kernel_col++)//kernel_width
				{
					int input_row = -padding + kernel_row;
					for (int output_row = output_h; output_row; output_row--)
					{
						if (!is_a_ge_zero_and_a_lt_b(input_row, height))
						{
							for (int output_col = output_w; output_col; output_col--)
								*(data_col++) = 0;
						}
						else
						{
							int input_col = -padding + kernel_col;
							for (int output_col = output_w; output_col; output_col--)
							{
								if (is_a_ge_zero_and_a_lt_b(input_col, width))
								{
									*(data_col++) = data_im[input_row * width + input_col];
								}
								else
									*(data_col++) = 0;
								input_col += stride;
							}
						}
						input_row += stride;
					}
				}
			}
		}
	}

	template<typename T>
	void col2im(const T* data_col, const int channels,
		const int height, const int width, const int kernelsize,
		const int padding, const int stride, T* data_im)
	{
		int output_h = (height - kernelsize + 2 * padding) / stride + 1;
		int output_w = output_h;
		const int channel_size = height * width;//
		for (int channel = channels; channel--; data_im += channel_size)
			//最外层循环 每次递增单通道图像的大小，比如通道0循环完，进入通道1
		{
			for (int kernel_row = 0; kernel_row < kernelsize; kernel_row++)//实质上是kernel_height
			{
				for (int kernel_col = 0; kernel_col < kernelsize; kernel_col++)//kernel_width
				{
					int input_row = -padding + kernel_row;
					for (int output_row = output_h; output_row; output_row--)
					{
						if (!is_a_ge_zero_and_a_lt_b(input_row, height))
						{
							for (int output_col = output_w; output_col; output_col--)
								data_col++;
						}
						else
						{
							int input_col = -padding + kernel_col;
							for (int output_col = output_w; output_col; output_col--)
							{
								if (is_a_ge_zero_and_a_lt_b(input_col, width))
									data_im[input_row * width + input_col] = *(data_col++);
								else
									data_col++;
								input_col += stride;
							}
						}
						input_row += stride;
					}
				}
			}
		}
	}

	template<class T>
	void Transmat(T* data_o, int row_o, int col_o, T* data_n)
	{
		for (int j = 0; j <col_o; j++)
			for (int i = 0; i < row_o; i++)
				*(data_n++) = data_o[i * col_o + j];
	}

	template<class T>
	void Transblob(T* data, int blocknum, int block_row, int block_col)
	{
		T* data_o = new T[blocknum * block_row * block_col];
		std::copy(data, data + blocknum * block_row * block_col, data_o);
		for (int iBatch = 0; iBatch < blocknum; iBatch++)
			Transmat(data_o + iBatch * block_row * block_col, block_row, block_col, data + iBatch * block_row * block_col);
		delete[] data_o;
	}


}
