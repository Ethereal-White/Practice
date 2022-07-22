#include <iostream>
#include <time.h>

using namespace std; 
long long fastpower(long long x, long long y);
long long slowpower(long long x, long long y);

int main()
{
	clock_t start1;
	clock_t start2;
	clock_t finish1;
	clock_t finish2;
	float sum1; 
	float sum2; 

	start1 = clock();
	for (long i = 0; i < 1000000; i++) fastpower(2, 63); 
	finish1 = clock();

	start2 = clock();
	for (long i = 0; i < 1000000; i++) slowpower(2, 63);
	finish2 = clock();

	sum1 = (float)(finish1 - start1);
	sum2 = (float)(finish2 - start2);
	cout << sum1 << " " << sum2;
	return 0;
}

long long fastpower(long long x, long long y)
{
	long long res = 1;
	while (y)
	{
		if (y & 1) res = res * x;
		x = x * x;
		y = y >> 1;
	}
	return res;
}

long long slowpower(long long x, long long y)
{
	long long res = 1;
	for (long long i = 0; i < y; i++) res = res * x;
	return res; 
}
