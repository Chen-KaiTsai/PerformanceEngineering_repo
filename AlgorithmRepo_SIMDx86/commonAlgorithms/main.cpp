#include "main.h"

static void calcMinMaxU8_Iavx_test()
{
	std::unique_ptr<uint8_t[]> data = std::make_unique<uint8_t[]>(1024);

	for (int i = 0; i < 1024; ++i) {
		data[i] = (uint8_t)(i % 256);
	}

	uint8_t min, max;

	if (fsimd::calcMinMaxU8_Iavx(&min, &max, data.get(), 1024)) {
		printf("min : %u, max : %u\n\n", min, max);
	}
	else {
		exit(EXIT_FAILURE);
	}
}

static void calcMeanU8_Iavx_test()
{
	std::unique_ptr<uint8_t[]> data = std::make_unique<uint8_t[]>(1024);

	for (int i = 0; i < 1024; ++i) {
		data[i] = (uint8_t)(255);
	}
	
	double mean;
	uint64_t sum;

	if (fsimd::calcMeanU8_Iavx(&mean, &sum, data.get(), 1024)) {
		printf("mean : %lf, sum : %llu\n\n", mean, sum);
	}
	else {
		exit(EXIT_FAILURE);
	}
}

static void convertRGBtoGrayScale_Iavx2_test()
{
	const char* imageFileName = "ImageC.png";
	const float coef[4]{ 0.2126f, 0.7152f, 0.0722f, 0.0f };

	const char* GSimageFileName = "GSimage.png";

	ImageMatrix imageRGB(imageFileName, PixelType::Rgb32);

	size_t height = imageRGB.GetHeight();
	size_t width = imageRGB.GetWidth();
	size_t size = height * width;

	ImageMatrix imageGS(height, width, PixelType::Gray8);

	RGB32* bufferRGB = imageRGB.GetPixelBuffer<RGB32>();
	uint8_t* bufferGS = imageGS.GetPixelBuffer<uint8_t>();

	std::cout << "Converting RGB image " << height << " X " << width << "\n\n";

	fsimd::convertRGBtoGrayScale_Iavx2(bufferGS, bufferRGB, size, coef);

	std::cout << "Saving GrayScale image\n\n";

	imageGS.SaveImage(GSimageFileName, ImageFileType::PNG);
}

static void calcLeastSquare_Iavx2_test()
{
	const size_t size = 59;

	AlignedArray<double> X_data(size, 32);
	AlignedArray<double> Y_data(size, 32);

	double* X = X_data.Data();
	double* Y = Y_data.Data();

	MT::FillArrayFP(X, size, -25.0, 25.0, 73);
	MT::FillArrayFP(Y, size, -25.0, 25.0, 83);

	for (size_t i = 0; i < size; ++i)
		Y[i] = Y[i] * Y[i];

	double a, b;

	fsimd::calcLeastSquare_Iavx2(&a, &b, X, Y, size);

	std::cout << std::fixed << std::setprecision(8);

	std::cout << "Slope : " << std::setw(12) << a << "\n\n";
	std::cout << "Intercept : " << std::setw(12) << b << "\n\n";
}

static void ReLU_Iavx2_test()
{
	constexpr size_t size = 100;

	AlignedArray<float> inputData(size, 32);
	AlignedArray<float> outputData(size, 32);
	
	float* input = inputData.Data();
	float* output = outputData.Data();

	MT::FillArrayFP(input, size, -1.0f, 1.0f, 1024);

	fwsimd::ReLU_Iavx2(input, output, size);

	for (int i = 0; i < size; ++i) {
		if (i % 8 == 0)
			puts("");
		printf("%3.5f\t", output[i]);
	}
}

static void LeakyReLU_Iavx2_test()
{
	constexpr size_t size = 100;

	AlignedArray<float> inputData(size, 32);
	AlignedArray<float> outputData(size, 32);

	float* input = inputData.Data();
	float* output = outputData.Data();

	MT::FillArrayFP(input, size, -1.0f, 1.0f, 1024);

	fwsimd::LeakyReLU_Iavx2(input, output, 0.01f, size);

	for (int i = 0; i < size; ++i) {
		if (i % 4 == 0)
			puts("");
		printf("%3.6f\t", output[i]);
	}
}

static void Add_Iavx2_test()
{
	constexpr size_t size = 100;

	AlignedArray<float> input1Data(size, 32);
	AlignedArray<float> input2Data(size, 32);
	AlignedArray<float> outputData(size, 32);

	float* input1 = input1Data.Data();
	float* input2 = input2Data.Data();
	float* output = outputData.Data();

	MT::FillArrayFP(input1, size, -1.0f, 1.0f, 1024);
	MT::FillArrayFP(input2, size, -1.0f, 1.0f, 1024);

	fwsimd::Add_Iavx2(input1, input2, output, size);

	for (int i = 0; i < size; ++i) {
		if (i % 4 == 0)
			puts("");
		printf("%3.6f\t", input1[i]);
	}
	puts("\n");

	for (int i = 0; i < size; ++i) {
		if (i % 4 == 0)
			puts("");
		printf("%3.6f\t", input2[i]);
	}
	puts("\n");

	for (int i = 0; i < size; ++i) {
		if (i % 4 == 0)
			puts("");
		printf("%3.6f\t", output[i]);
	}
	puts("\n");
}

int main(int argc, char** argv)
{
	Add_Iavx2_test();

	return 0;
}
