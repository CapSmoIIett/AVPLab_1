//
//

#include <iostream>
#include <chrono>
#include <Windows.h>
#include <xmmintrin.h>

#define BASIC_WIDTH 1000
#define BASIC_HEIGHT 1000
#define BASIC_INNER_WIDTH 8
#define BASIC_INNER_HEIGHT 8

typedef  std::chrono::high_resolution_clock _clock;

void mult (float** ms1, float** ms2, float** result, int width, int height);
void multNotV (float** ms1, float** ms2, float** result, int width, int height);

float**** callocMM(int width, int height, int inner_width, int inner_height);
void freeMM	(float**** ms, int width, int height, int inner_width, int inner_height);
void fill	(float**** ms, int width, int height, int inner_width, int inner_height);

void show	(float**** ms, int width, int height, int inner_width, int inner_height);
void show	(float**** ms, int width, int height);

void multV	(float**** ms1, float**** ms2, float**** result, int width, int height, int inner_width, int inner_height);
void multNotV(float**** ms1, float**** ms2, float**** result, int width, int height, int inner_width, int inner_height);
void multSSE (float**** ms1, float**** ms2, float**** result, int width, int height, int inner_width, int inner_height);
bool isEqual (float**** ms1, float**** ms2, int width, int height, int inner_width, int inner_height);

void mult(float** ms1, float** ms2, float** result);



int main()
{
	float**** array1 = nullptr,
		**** array2 = nullptr,
		**** result_array1 = nullptr,
		**** result_array2 = nullptr,
		**** result_array3 = nullptr;

	array1 = callocMM(BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);
	array2 = callocMM(BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);
	result_array1 = callocMM(BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);
	result_array2 = callocMM(BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);
	result_array3 = callocMM(BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);
	
	fill(array1, BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);
	fill(array2, BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);

	/*show(array1, BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);
	show(array2, BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);*/

	_clock::time_point t_point1 = _clock::now();
	{
		multV(array1, array2, result_array1, BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);
	}
	_clock::time_point t_point2 = _clock::now();

	_clock::time_point t_point3 = _clock::now();
	{
		multNotV(array1, array2, result_array2, BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);
	}
	_clock::time_point t_point4 = _clock::now();

	_clock::time_point t_point5 = _clock::now();
	{
		multSSE(array1, array2, result_array3, BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);
	}
	_clock::time_point t_point6 = _clock::now();

	/*show(result_array1, BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);
	std::cout << "\n\n";
	show(result_array2, BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);
	std::cout << "\n\n";
	show(result_array3, BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);//*/

	std::chrono::duration<double> time_span1 = 
		std::chrono::duration_cast<std::chrono::duration<double>>(t_point2 - t_point1);

	std::chrono::duration<double> time_span2 = 
		std::chrono::duration_cast<std::chrono::duration<double>>(t_point4 - t_point3);

	std::chrono::duration<double> time_span3 = 
		std::chrono::duration_cast<std::chrono::duration<double>>(t_point6 - t_point5);
	
	std::cout << "result:" <<
		isEqual(result_array1, result_array2, BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT) << 
		isEqual(result_array1, result_array3, BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT) << "\n";
	std::cout << "time results:" << "\n";
	std::cout << "first:  " << time_span1.count() << "\n";
	std::cout << "second: " << time_span2.count() << "\n";
	std::cout << "third:  " << time_span3.count() << "\n";



	freeMM(array1, BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);
	freeMM(array2, BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);
	freeMM(result_array1, BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);
	freeMM(result_array2, BASIC_WIDTH, BASIC_HEIGHT, BASIC_INNER_WIDTH, BASIC_INNER_HEIGHT);
}


void sse_Mult00_8x8_8x8(float** ms1, float** ms2, float** result)
{
	__m128 b0, b1, b2, b3, b4, b5, b6, b7;
	__m128 row, rslt, tmp;

	b0 = _mm_loadh_pi(_mm_loadl_pi(b0, (const __m64*) & ms2[0][0]), (const __m64*) & ms2[0][2]);
	b1 = _mm_loadh_pi(_mm_loadl_pi(b1, (const __m64*) & ms2[1][0]), (const __m64*) & ms2[1][2]);
	b2 = _mm_loadh_pi(_mm_loadl_pi(b2, (const __m64*) & ms2[2][0]), (const __m64*) & ms2[2][2]);
	b3 = _mm_loadh_pi(_mm_loadl_pi(b3, (const __m64*) & ms2[3][0]), (const __m64*) & ms2[3][2]);
	b4 = _mm_loadh_pi(_mm_loadl_pi(b4, (const __m64*) & ms2[4][0]), (const __m64*) & ms2[4][2]);
	b5 = _mm_loadh_pi(_mm_loadl_pi(b5, (const __m64*) & ms2[5][0]), (const __m64*) & ms2[5][2]);
	b6 = _mm_loadh_pi(_mm_loadl_pi(b6, (const __m64*) & ms2[6][0]), (const __m64*) & ms2[6][2]);
	b7 = _mm_loadh_pi(_mm_loadl_pi(b7, (const __m64*) & ms2[7][0]), (const __m64*) & ms2[7][2]);


	row = _mm_set_ps1(ms1[0][0]);
	rslt = _mm_mul_ps(row, b0);
	row = _mm_set_ps1(ms1[0][1]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b1));
	row = _mm_set_ps1(ms1[0][2]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b2));
	row = _mm_set_ps1(ms1[0][3]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b3));
	row = _mm_set_ps1(ms1[0][4]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b4));
	row = _mm_set_ps1(ms1[0][5]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b5));
	row = _mm_set_ps1(ms1[0][6]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b6));
	row = _mm_set_ps1(ms1[0][7]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b7));

	_mm_store_ps(&result[0][0], rslt);

	row = _mm_set_ps1(ms1[1][0]);
	rslt = _mm_mul_ps(row, b0);
	row = _mm_set_ps1(ms1[1][1]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b1));
	row = _mm_set_ps1(ms1[1][2]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b2));
	row = _mm_set_ps1(ms1[1][3]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b3));
	row = _mm_set_ps1(ms1[1][4]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b4));
	row = _mm_set_ps1(ms1[1][5]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b5));
	row = _mm_set_ps1(ms1[1][6]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b6));
	row = _mm_set_ps1(ms1[1][7]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b7));

	_mm_storel_pi((__m64*) & result[1][0], rslt);
	_mm_storeh_pi((__m64*) & result[1][2], rslt);


	row = _mm_set_ps1(ms1[2][0]);
	rslt = _mm_mul_ps(row, b0);
	row = _mm_set_ps1(ms1[2][1]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b1));
	row = _mm_set_ps1(ms1[2][2]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b2));
	row = _mm_set_ps1(ms1[2][3]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b3));
	row = _mm_set_ps1(ms1[2][4]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b4));
	row = _mm_set_ps1(ms1[2][5]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b5));
	row = _mm_set_ps1(ms1[2][6]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b6));
	row = _mm_set_ps1(ms1[2][7]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b7));

	_mm_store_ps(&result[2][0], rslt);


	row = _mm_set_ps1(ms1[3][0]);
	rslt = _mm_mul_ps(row, b0);
	row = _mm_set_ps1(ms1[3][1]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b1));
	row = _mm_set_ps1(ms1[3][2]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b2));
	row = _mm_set_ps1(ms1[3][3]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b3));
	row = _mm_set_ps1(ms1[3][4]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b4));
	row = _mm_set_ps1(ms1[3][5]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b5));
	row = _mm_set_ps1(ms1[3][6]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b6));
	row = _mm_set_ps1(ms1[3][7]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b7));

	_mm_storel_pi((__m64*) & result[3][0], rslt);
	_mm_storeh_pi((__m64*) & result[3][2], rslt);


	row = _mm_set_ps1(ms1[4][0]);
	rslt = _mm_mul_ps(row, b0);
	row = _mm_set_ps1(ms1[4][1]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b1));
	row = _mm_set_ps1(ms1[4][2]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b2));
	row = _mm_set_ps1(ms1[4][3]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b3));
	row = _mm_set_ps1(ms1[4][4]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b4));
	row = _mm_set_ps1(ms1[4][5]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b5));
	row = _mm_set_ps1(ms1[4][6]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b6));
	row = _mm_set_ps1(ms1[4][7]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b7));

	_mm_store_ps(&result[4][0], rslt);


	row = _mm_set_ps1(ms1[5][0]);
	rslt = _mm_mul_ps(row, b0);
	row = _mm_set_ps1(ms1[5][1]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b1));
	row = _mm_set_ps1(ms1[5][2]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b2));
	row = _mm_set_ps1(ms1[5][3]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b3));
	row = _mm_set_ps1(ms1[5][4]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b4));
	row = _mm_set_ps1(ms1[5][5]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b5));
	row = _mm_set_ps1(ms1[5][6]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b6));
	row = _mm_set_ps1(ms1[5][7]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b7));

	_mm_storel_pi((__m64*) & result[5][0], rslt);
	_mm_storeh_pi((__m64*) & result[5][2], rslt);


	row = _mm_set_ps1(ms1[6][0]);
	rslt = _mm_mul_ps(row, b0);
	row = _mm_set_ps1(ms1[6][1]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b1));
	row = _mm_set_ps1(ms1[6][2]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b2));
	row = _mm_set_ps1(ms1[6][3]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b3));
	row = _mm_set_ps1(ms1[6][4]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b4));
	row = _mm_set_ps1(ms1[6][5]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b5));
	row = _mm_set_ps1(ms1[6][6]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b6));
	row = _mm_set_ps1(ms1[6][7]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b7));

	_mm_store_ps(&result[6][0], rslt);


	row = _mm_set_ps1(ms1[7][0]);
	rslt = _mm_mul_ps(row, b0);
	row = _mm_set_ps1(ms1[7][1]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b1));
	row = _mm_set_ps1(ms1[7][2]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b2));
	row = _mm_set_ps1(ms1[7][3]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b3));
	row = _mm_set_ps1(ms1[7][4]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b4));
	row = _mm_set_ps1(ms1[7][5]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b5));
	row = _mm_set_ps1(ms1[7][6]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b6));
	row = _mm_set_ps1(ms1[7][7]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b7));

	_mm_storel_pi((__m64*) & result[7][0], rslt);
	_mm_storeh_pi((__m64*) & result[7][2], rslt);


	b0 = _mm_loadh_pi(_mm_loadl_pi(b0, (const __m64*) & ms2[0][4]), (const __m64*) & ms2[0][6]);
	b1 = _mm_loadh_pi(_mm_loadl_pi(b1, (const __m64*) & ms2[1][4]), (const __m64*) & ms2[1][6]);
	b2 = _mm_loadh_pi(_mm_loadl_pi(b2, (const __m64*) & ms2[2][4]), (const __m64*) & ms2[2][6]);
	b3 = _mm_loadh_pi(_mm_loadl_pi(b3, (const __m64*) & ms2[3][4]), (const __m64*) & ms2[3][6]);
	b4 = _mm_loadh_pi(_mm_loadl_pi(b4, (const __m64*) & ms2[4][4]), (const __m64*) & ms2[4][6]);
	b5 = _mm_loadh_pi(_mm_loadl_pi(b5, (const __m64*) & ms2[5][4]), (const __m64*) & ms2[5][6]);
	b6 = _mm_loadh_pi(_mm_loadl_pi(b6, (const __m64*) & ms2[6][4]), (const __m64*) & ms2[6][6]);
	b7 = _mm_loadh_pi(_mm_loadl_pi(b7, (const __m64*) & ms2[7][4]), (const __m64*) & ms2[7][6]);

	row = _mm_set_ps1(ms1[0][0]);
	rslt = _mm_mul_ps(row, b0);
	row = _mm_set_ps1(ms1[0][1]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b1));
	row = _mm_set_ps1(ms1[0][2]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b2));
	row = _mm_set_ps1(ms1[0][3]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b3));
	row = _mm_set_ps1(ms1[0][4]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b4));
	row = _mm_set_ps1(ms1[0][5]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b5));
	row = _mm_set_ps1(ms1[0][6]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b6));
	row = _mm_set_ps1(ms1[0][7]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b7));

	_mm_store_ps(&result[0][4], rslt);


	row = _mm_set_ps1(ms1[1][0]);
	rslt = _mm_mul_ps(row, b0);
	row = _mm_set_ps1(ms1[1][1]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b1));
	row = _mm_set_ps1(ms1[1][2]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b2));
	row = _mm_set_ps1(ms1[1][3]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b3));
	row = _mm_set_ps1(ms1[1][4]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b4));
	row = _mm_set_ps1(ms1[1][5]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b5));
	row = _mm_set_ps1(ms1[1][6]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b6));
	row = _mm_set_ps1(ms1[1][7]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b7));

	_mm_storel_pi((__m64*) & result[1][4], rslt);
	_mm_storeh_pi((__m64*) & result[1][6], rslt);


	row = _mm_set_ps1(ms1[2][0]);
	rslt = _mm_mul_ps(row, b0);
	row = _mm_set_ps1(ms1[2][1]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b1));
	row = _mm_set_ps1(ms1[2][2]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b2));
	row = _mm_set_ps1(ms1[2][3]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b3));
	row = _mm_set_ps1(ms1[2][4]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b4));
	row = _mm_set_ps1(ms1[2][5]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b5));
	row = _mm_set_ps1(ms1[2][6]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b6));
	row = _mm_set_ps1(ms1[2][7]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b7));

	_mm_store_ps(&result[2][4], rslt);


	row = _mm_set_ps1(ms1[3][0]);
	rslt = _mm_mul_ps(row, b0);
	row = _mm_set_ps1(ms1[3][1]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b1));
	row = _mm_set_ps1(ms1[3][2]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b2));
	row = _mm_set_ps1(ms1[3][3]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b3));
	row = _mm_set_ps1(ms1[3][4]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b4));
	row = _mm_set_ps1(ms1[3][5]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b5));
	row = _mm_set_ps1(ms1[3][6]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b6));
	row = _mm_set_ps1(ms1[3][7]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b7));

	_mm_storel_pi((__m64*) & result[3][4], rslt);
	_mm_storeh_pi((__m64*) & result[3][6], rslt);


	row = _mm_set_ps1(ms1[4][0]);
	rslt = _mm_mul_ps(row, b0);
	row = _mm_set_ps1(ms1[4][1]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b1));
	row = _mm_set_ps1(ms1[4][2]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b2));
	row = _mm_set_ps1(ms1[4][3]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b3));
	row = _mm_set_ps1(ms1[4][4]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b4));
	row = _mm_set_ps1(ms1[4][5]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b5));
	row = _mm_set_ps1(ms1[4][6]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b6));
	row = _mm_set_ps1(ms1[4][7]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b7));

	_mm_store_ps(&result[4][4], rslt);


	row = _mm_set_ps1(ms1[5][0]);
	rslt = _mm_mul_ps(row, b0);
	row = _mm_set_ps1(ms1[5][1]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b1));
	row = _mm_set_ps1(ms1[5][2]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b2));
	row = _mm_set_ps1(ms1[5][3]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b3));
	row = _mm_set_ps1(ms1[5][4]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b4));
	row = _mm_set_ps1(ms1[5][5]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b5));
	row = _mm_set_ps1(ms1[5][6]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b6));
	row = _mm_set_ps1(ms1[5][7]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b7));

	_mm_storel_pi((__m64*) & result[5][4], rslt);
	_mm_storeh_pi((__m64*) & result[5][6], rslt);


	row = _mm_set_ps1(ms1[6][0]);
	rslt = _mm_mul_ps(row, b0);
	row = _mm_set_ps1(ms1[6][1]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b1));
	row = _mm_set_ps1(ms1[6][2]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b2));
	row = _mm_set_ps1(ms1[6][3]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b3));
	row = _mm_set_ps1(ms1[6][4]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b4));
	row = _mm_set_ps1(ms1[6][5]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b5));
	row = _mm_set_ps1(ms1[6][6]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b6));
	row = _mm_set_ps1(ms1[6][7]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b7));

	_mm_store_ps(&result[6][4], rslt);


	row = _mm_set_ps1(ms1[7][0]);
	rslt = _mm_mul_ps(row, b0);
	row = _mm_set_ps1(ms1[7][1]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b1));
	row = _mm_set_ps1(ms1[7][2]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b2));
	row = _mm_set_ps1(ms1[7][3]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b3));
	row = _mm_set_ps1(ms1[7][4]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b4));
	row = _mm_set_ps1(ms1[7][5]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b5));
	row = _mm_set_ps1(ms1[7][6]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b6));
	row = _mm_set_ps1(ms1[7][7]);
	rslt = _mm_add_ps(rslt, _mm_mul_ps(row, b7));

	_mm_storel_pi((__m64*) & result[7][4], rslt);
	_mm_storeh_pi((__m64*) & result[7][6], rslt);//*/
}

void mult(float** ms1, float** ms2, float** result, int width, int height)
{
	float* temp1 = nullptr;
	float* temp2 = nullptr;

#pragma loop( hint_parallel( 0 ) )
	for (int i = 0; i < width; i++)
	{
		temp1 = result[i];

#pragma loop( hint_parallel( 0 ) )
		for (int j = 0; j < height; j++)
		{
			temp2 = ms2[j];

#pragma loop( hint_parallel( 0 ) )
			for (int k = 0; k < height; k++)
			{
				temp1[k] += ms1[i][j] * temp2[k];
			}

		}
	}
}

void multNotV(float** ms1, float** ms2, float** result, int width, int height)
{
	float* temp1 = nullptr;
	float* temp2 = nullptr;

#pragma loop(no_vector)
	for (int i = 0; i < width; i++)
	{
		temp1 = result[i];

#pragma loop(no_vector)
		for (int j = 0; j < height; j++)
		{
			temp2 = ms2[j];

#pragma loop(no_vector)
			for (int k = 0; k < height; k++)
			{
				temp1[k] += ms1[i][j] * temp2[k];
			}

		}
	}
}

float**** callocMM(int width, int height, int inner_width, int inner_height)
{
	float**** ms = new float* **[height];

	for (int i = 0; i < height; i++)
	{
		ms[i] = new float**[width];

		for (int j = 0; j < width; j++)
		{
			ms[i][j] = new float * [inner_height];

			for (int k = 0; k < inner_height; k++) 
			{
				ms[i][j][k] = new float[inner_width];

				for (int m = 0; m < inner_width; m++) 
					ms[i][j][k][m] = 0;
				
			}
		}
	}

	return ms;
}

void freeMM(float**** ms, int width, int height, int inner_width, int inner_height)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int k = 0; k < inner_height; k++) 
			{
				for (int m = 0; m < inner_width; m++) 
					ms[i][j][k][m] = 0;
				
				delete[] ms[i][j][k];
			}

			delete[] ms[i][j];
		}

		delete[] ms[i];
	}

	delete[] ms;
}

void fill(float**** ms, int width, int height, int inner_width, int inner_height)
{
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			for (int k = 0; k < inner_height; k++)
				for (int m = 0; m < inner_width; m++)
					ms[i][j][k][m] = (float)(rand() % 100);
}

void show(float**** ms, int width, int height, int inner_width, int inner_height)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int k = 0; k < inner_height; k++) 
			{
				for (int m = 0; m < inner_width; m++)
					std::cout << "\t" << ms[i][j][k][m];
				
				std::cout << "\n";
			}

			std::cout << "\n";
		}

		std::cout << "\n";
	}

	std::cout << "\n";

}


void multV(float**** ms1,float**** ms2,float**** result, int width, int height, int inner_width, int inner_height)
{
#pragma loop( hint_parallel( 0 ) )
	for (int i = 0; i < width; i++) 
	{
#pragma loop( hint_parallel( 0 ) )
		for (int j = 0; j < height; j++) 
		{
			mult(ms1[i][j], ms2[i][j], result[i][j], inner_width, inner_height);
		}
	}
	
}

void multNotV(float**** ms1,float**** ms2,float**** result, int width, int height, int inner_width, int inner_height)
{
#pragma loop(no_vector)
	for (int i = 0; i < height; i++)
	{
#pragma loop(no_vector)
		for (int j = 0; j < width; j++) 
		{
			multNotV(ms1[i][j], ms2[i][j], result[i][j], inner_width, inner_height);
		}
	}
}

void multSSE(float**** ms1, float**** ms2, float**** result, int width, int height, int inner_width, int inner_height)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++) 
		{
			sse_Mult00_8x8_8x8(ms1[i][j], ms2[i][j], result[i][j]);
		}
	}

}


bool isEqual(float**** ms1, float**** ms2, int width, int height, int inner_width, int inner_height)
{	
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			for (int k = 0; k < inner_height; k++)
				for (int m = 0; m < inner_width; m++)
					if (ms1[i][j][k][m] != ms2[i][j][k][m])
						return false;

	return true;
}
