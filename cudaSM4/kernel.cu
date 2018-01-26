//本算法一律采用小端表示
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include<fstream>
#include<random>
#include<iomanip>
#include"cpu.h"
#define CUDA_CALL(x){const cudaError_t a=(x);if(a!=cudaSuccess){printf("CUDA Error:%s(err_num=%d)\n",cudaGetErrorString(a),a);cudaDeviceReset();}}
#define COUNT_TIME(x,y){time = clock();(x);time = clock() - time;cout << (y) << time << "ms" << endl; }
using namespace std;
typedef unsigned char uc;
typedef unsigned ui;
int threads;

void printHex(uc *src, int n)//输出n个字节的16进制表示，用于调试
{
	for (int i = 0; i < n; ++i)
	{
		cout << hex << setw(2) << setfill('0') << (int)src[i] << ' ' << dec;
	}
	cout << endl;
}
void shiftEndian(ui *x, int n)//大小端转换，从x开始的n个4字节，用于检验文档中给出的示例
{
	for (int i = 0; i < n; ++i)
	{
		uc *a = (uc*)(x + i), t;
		t = a[0]; a[0] = a[3]; a[3] = t;
		t = a[1]; a[1] = a[2]; a[2] = t;
	}
}
int readFile(char*address, uc **buffer)//将address的文件读入数组并在末尾补0
{
	ifstream file;
	file.open(address, ios::binary);
	file.seekg(0, std::ios::end);
	int size = file.tellg();
	file.seekg(0, std::ios::beg);
	*buffer = new uc[size + 31];
	file.read((char*)*buffer, size);
	file.close();
	memset(*buffer + size, 0, 31);
	return size;
}
void writeFile(char*address, uc *buffer, int size)//将数组size大小的内容写入address的文件
{
	ofstream file;
	file.open(address, ios::binary);
	file.write((char*)buffer, size);
	file.close();
}

__device__ uc gpuSbox[16][16]=
{
	{ 0xd6,0x90,0xe9,0xfe,0xcc,0xe1,0x3d,0xb7,0x16,0xb6,0x14,0xc2,0x28,0xfb,0x2c,0x05 },
	{ 0x2b,0x67,0x9a,0x76,0x2a,0xbe,0x04,0xc3,0xaa,0x44,0x13,0x26,0x49,0x86,0x06,0x99 },
	{ 0x9c,0x42,0x50,0xf4,0x91,0xef,0x98,0x7a,0x33,0x54,0x0b,0x43,0xed,0xcf,0xac,0x62 },
	{ 0xe4,0xb3,0x1c,0xa9,0xc9,0x08,0xe8,0x95,0x80,0xdf,0x94,0xfa,0x75,0x8f,0x3f,0xa6 },
	{ 0x47,0x07,0xa7,0xfc,0xf3,0x73,0x17,0xba,0x83,0x59,0x3c,0x19,0xe6,0x85,0x4f,0xa8 },
	{ 0x68,0x6b,0x81,0xb2,0x71,0x64,0xda,0x8b,0xf8,0xeb,0x0f,0x4b,0x70,0x56,0x9d,0x35 },
	{ 0x1e,0x24,0x0e,0x5e,0x63,0x58,0xd1,0xa2,0x25,0x22,0x7c,0x3b,0x01,0x21,0x78,0x87 },
	{ 0xd4,0x00,0x46,0x57,0x9f,0xd3,0x27,0x52,0x4c,0x36,0x02,0xe7,0xa0,0xc4,0xc8,0x9e },
	{ 0xea,0xbf,0x8a,0xd2,0x40,0xc7,0x38,0xb5,0xa3,0xf7,0xf2,0xce,0xf9,0x61,0x15,0xa1 },
	{ 0xe0,0xae,0x5d,0xa4,0x9b,0x34,0x1a,0x55,0xad,0x93,0x32,0x30,0xf5,0x8c,0xb1,0xe3 },
	{ 0x1d,0xf6,0xe2,0x2e,0x82,0x66,0xca,0x60,0xc0,0x29,0x23,0xab,0x0d,0x53,0x4e,0x6f },
	{ 0xd5,0xdb,0x37,0x45,0xde,0xfd,0x8e,0x2f,0x03,0xff,0x6a,0x72,0x6d,0x6c,0x5b,0x51 },
	{ 0x8d,0x1b,0xaf,0x92,0xbb,0xdd,0xbc,0x7f,0x11,0xd9,0x5c,0x41,0x1f,0x10,0x5a,0xd8 },
	{ 0x0a,0xc1,0x31,0x88,0xa5,0xcd,0x7b,0xbd,0x2d,0x74,0xd0,0x12,0xb8,0xe5,0xb4,0xb0 },
	{ 0x89,0x69,0x97,0x4a,0x0c,0x96,0x77,0x7e,0x65,0xb9,0xf1,0x09,0xc5,0x6e,0xc6,0x84 },
	{ 0x18,0xf0,0x7d,0xec,0x3a,0xdc,0x4d,0x20,0x79,0xee,0x5f,0x3e,0xd7,0xcb,0x39,0x48 }
};
__device__ ui gpuFK[4]= { 0xa3b1bac6,0x56aa3350,0x677d9197,0xb27022dc };
__device__ ui gpuCK[32]=
{
	0x00070e15,0x1c232a31,0x383f464d,0x545b6269,
	0x70777e85,0x8c939aa1,0xa8afb6bd,0xc4cbd2d9,
	0xe0e7eef5,0xfc030a11,0x181f262d,0x343b4249,
	0x50575e65,0x6c737a81,0x888f969d,0xa4abb2b9,
	0xc0c7ced5,0xdce3eaf1,0xf8ff060d,0x141b2229,
	0x30373e45,0x4c535a61,0x686f767d,0x848b9299,
	0xa0a7aeb5,0xbcc3cad1,0xd8dfe6ed,0xf4fb0209,
	0x10171e25,0x2c333a41,0x484f565d,0x646b7279
};
uc Sbox[16][16] =
{
	{ 0xd6,0x90,0xe9,0xfe,0xcc,0xe1,0x3d,0xb7,0x16,0xb6,0x14,0xc2,0x28,0xfb,0x2c,0x05 },
	{ 0x2b,0x67,0x9a,0x76,0x2a,0xbe,0x04,0xc3,0xaa,0x44,0x13,0x26,0x49,0x86,0x06,0x99 },
	{ 0x9c,0x42,0x50,0xf4,0x91,0xef,0x98,0x7a,0x33,0x54,0x0b,0x43,0xed,0xcf,0xac,0x62 },
	{ 0xe4,0xb3,0x1c,0xa9,0xc9,0x08,0xe8,0x95,0x80,0xdf,0x94,0xfa,0x75,0x8f,0x3f,0xa6 },
	{ 0x47,0x07,0xa7,0xfc,0xf3,0x73,0x17,0xba,0x83,0x59,0x3c,0x19,0xe6,0x85,0x4f,0xa8 },
	{ 0x68,0x6b,0x81,0xb2,0x71,0x64,0xda,0x8b,0xf8,0xeb,0x0f,0x4b,0x70,0x56,0x9d,0x35 },
	{ 0x1e,0x24,0x0e,0x5e,0x63,0x58,0xd1,0xa2,0x25,0x22,0x7c,0x3b,0x01,0x21,0x78,0x87 },
	{ 0xd4,0x00,0x46,0x57,0x9f,0xd3,0x27,0x52,0x4c,0x36,0x02,0xe7,0xa0,0xc4,0xc8,0x9e },
	{ 0xea,0xbf,0x8a,0xd2,0x40,0xc7,0x38,0xb5,0xa3,0xf7,0xf2,0xce,0xf9,0x61,0x15,0xa1 },
	{ 0xe0,0xae,0x5d,0xa4,0x9b,0x34,0x1a,0x55,0xad,0x93,0x32,0x30,0xf5,0x8c,0xb1,0xe3 },
	{ 0x1d,0xf6,0xe2,0x2e,0x82,0x66,0xca,0x60,0xc0,0x29,0x23,0xab,0x0d,0x53,0x4e,0x6f },
	{ 0xd5,0xdb,0x37,0x45,0xde,0xfd,0x8e,0x2f,0x03,0xff,0x6a,0x72,0x6d,0x6c,0x5b,0x51 },
	{ 0x8d,0x1b,0xaf,0x92,0xbb,0xdd,0xbc,0x7f,0x11,0xd9,0x5c,0x41,0x1f,0x10,0x5a,0xd8 },
	{ 0x0a,0xc1,0x31,0x88,0xa5,0xcd,0x7b,0xbd,0x2d,0x74,0xd0,0x12,0xb8,0xe5,0xb4,0xb0 },
	{ 0x89,0x69,0x97,0x4a,0x0c,0x96,0x77,0x7e,0x65,0xb9,0xf1,0x09,0xc5,0x6e,0xc6,0x84 },
	{ 0x18,0xf0,0x7d,0xec,0x3a,0xdc,0x4d,0x20,0x79,0xee,0x5f,0x3e,0xd7,0xcb,0x39,0x48 }
};
ui FK[4] = { 0xa3b1bac6,0x56aa3350,0x677d9197,0xb27022dc };
ui CK[32] =
{
	0x00070e15,0x1c232a31,0x383f464d,0x545b6269,
	0x70777e85,0x8c939aa1,0xa8afb6bd,0xc4cbd2d9,
	0xe0e7eef5,0xfc030a11,0x181f262d,0x343b4249,
	0x50575e65,0x6c737a81,0x888f969d,0xa4abb2b9,
	0xc0c7ced5,0xdce3eaf1,0xf8ff060d,0x141b2229,
	0x30373e45,0x4c535a61,0x686f767d,0x848b9299,
	0xa0a7aeb5,0xbcc3cad1,0xd8dfe6ed,0xf4fb0209,
	0x10171e25,0x2c333a41,0x484f565d,0x646b7279
};

__device__ __host__ void inc(ui *x, ui *y, ui n)//将128位小端表示的x+32位n存在y
{
	for (int i = 1; i < 4; ++i)y[i] = x[i];
	bool overflow = x[0] > ~n;
	y[0] = x[0] + n;
	for (int i = 1; i < 4; ++i)
	{
		if (!overflow)return;
		y[i] = x[i] + 1;
		overflow = !y[i];
	}
}
__device__ __host__ ui cycle(ui x, int n)//循环左移n位
{
	return (x << n) | (x >> (32 - n));
}
__device__ __host__ ui T(ui A, bool key)
{
	uc *a = (uc*)&A;
	for (int i = 0; i < 4; ++i)
	{
		#ifdef __CUDA_ARCH__
		a[i] = gpuSbox[a[i] >> 4][a[i] & 0x0F];
		#else
		a[i] = Sbox[a[i] >> 4][a[i] & 0x0F];
		#endif
	}
	if (key)
		return A ^ cycle(A, 13) ^ cycle(A, 23);
	return A ^ cycle(A, 2) ^ cycle(A, 10) ^ cycle(A, 18) ^ cycle(A, 24);
}
__device__ __host__ ui F(ui *X, ui rk, bool key = false)
{
	return X[0] ^ T(X[1] ^ X[2] ^ X[3] ^ rk, key);
}
void generateKey(uc *key, ui *K)//密钥生成
{
	for (int i = 0; i < 4; ++i)
	{
		K[i] = *((ui*)key + i) ^ FK[i];
	}
	for (int i = 0; i < 32; ++i)
	{
		K[i + 4] = F(K + i, CK[i], true);
	}
}
void generateRandom(ui *r)
{
	default_random_engine generator(time(0));
	uniform_int_distribution<ui>distribution;
	for (int i = 0; i < 4; ++i)
	{
		r[i] = distribution(generator);
	}
}
void sm4(uc *src, ui *rk, uc *dst, bool encrypt)//128位SM4算法，CPU
{
	ui X[36];
	memcpy(X, src, 16);
	//32次迭代
	if (encrypt)
	{
		for (int i = 0; i < 32; ++i)
		{
			X[i + 4] = F(X + i, rk[i]);
		}
	}
	else
	{
		for (int i = 0; i < 32; ++i)
		{
			X[i + 4] = F(X + i, rk[31 - i]);
		}
	}
	for (int i = 0; i < 4; ++i)//反序
	{
		*((ui*)dst + i) = X[35 - i];
	}
}
void ecb(uc *src, uc *key, uc *dst, bool encrypt, int n)//n字节ECB模式，CPU
{
	int groups = n >> 4;
	ui K[36];
	generateKey(key, K);
	for (int i = 0; i < groups; ++i)
	{
		sm4(src + i * 16, K + 4, dst + i * 16, encrypt);
	}
}
void cbc(uc *src, uc *key, uc *dst, bool encrypt, int n)//n字节CBC模式，CPU
{
	int groups = n / 16 + (n % 16 ? 1 : 0);
	ui K[36];
	generateKey(key, K);
	if (encrypt)
	{
		generateRandom((ui*)dst);
		for (int i = 0; i < groups; ++i)
		{
			for (int j = 0; j < 16; ++j)
			{
				dst[(i + 1) * 16 + j] = dst[i * 16 + j] ^ src[i * 16 + j];
			}
			sm4(dst + (i + 1) * 16, K + 4, dst + (i + 1) * 16, encrypt);
		}
	}
	else
	{
		for (int i = 0; i < groups; ++i)
		{
			sm4(src + (i + 1) * 16, K + 4, dst + i * 16, encrypt);
			for (int j = 0; j < 16; ++j)
			{
				dst[i * 16 + j] ^= src[i * 16 + j];
			}
		}
	}
}
void ctr(uc *src, uc *key, uc *dst, bool encrypt, int n)//n字节CTR模式，CPU
{
	int groups = n / 16 + (n % 16 ? 1 : 0);
	ui K[36];
	generateKey(key, K);
	uc r[16];
	if (encrypt)
	{
		generateRandom((ui*)r);
		memcpy(dst, r, 16);
		for (int i = 0; i < groups; ++i)
		{
			sm4(r, K + 4, dst + (i + 1) * 16, true);
			for (int j = 0; j < 16; ++j)
			{
				dst[(i + 1) * 16 + j] ^= src[i * 16 + j];
			}
			inc((ui*)r, (ui*)r, 1);
		}
	}
	else
	{
		memcpy(r, src, 16);
		for (int i = 0; i < groups; ++i)
		{
			sm4(r, K + 4, dst + i * 16, true);
			for (int j = 0; j < 16; ++j)
			{
				dst[i * 16 + j] ^= src[(i + 1) * 16 + j];
			}
			inc((ui*)r, (ui*)r, 1);
		}
	}
}
__global__ void SM4(uc *src, ui *rk, uc *dst, bool encrypt,bool ctr)//128位SM4，GPU
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int dev = id * 16;
	ui X[36];
	if (ctr)
	{
		inc((ui*)src, X, id);
	}
	else
	{
		for (int i = 0; i < 4; ++i)
		{
			X[i] = *((ui*)(src + dev) + i);
		}
	}
	//32次迭代
	if (encrypt)
	{
		for (int i = 0; i < 32; ++i)
		{
			X[i + 4] = F(X + i, rk[i]);
		}
	}
	else
	{
		for (int i = 0; i < 32; ++i)
		{
			X[i + 4] = F(X + i, rk[31 - i]);
		}
	}
	for (int i = 0; i < 4; ++i)//反序
	{
		*((ui*)(dst + dev) + i) = X[35 - i];
	}
}
__global__ void XOR(uc *src, uc *dst)//128位异或，GPU
{
	int dev = (blockIdx.x*blockDim.x + threadIdx.x) * 16;
	for (int i = 0; i < 4; ++i)
	{
		*((ui*)(dst + dev) + i) ^= *((ui*)(src + dev) + i);
	}
}
void ECB(uc *src, uc *key, uc *dst, bool encrypt, int n)//n字节ECB模式，GPU
{
	ui K[36];
	generateKey(key, K);
	uc *gpusrc, *gpudst;
	ui *gpukey;
	//显存分配
	CUDA_CALL(cudaMalloc((void**)&gpusrc, n));
	CUDA_CALL(cudaMalloc((void**)&gpukey, 128));
	CUDA_CALL(cudaMalloc((void**)&gpudst, n));
	//主存拷贝到显存
	CUDA_CALL(cudaMemcpy(gpusrc, src, n, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(gpukey, K + 4, 128, cudaMemcpyHostToDevice));
	//调用核函数
	SM4 <<< (n >> 4) / threads + 1, threads >>> (gpusrc, gpukey, gpudst, encrypt, 0);
	//等待所有线程运行完毕
	CUDA_CALL(cudaDeviceSynchronize());
	//显存拷贝回主存
	CUDA_CALL(cudaMemcpy(dst, gpudst, n, cudaMemcpyDeviceToHost));
	//显存释放
	CUDA_CALL(cudaFree(gpusrc));
	CUDA_CALL(cudaFree(gpukey));
	CUDA_CALL(cudaFree(gpudst));
}
void CBC(uc *src, uc *key, uc *dst, bool encrypt, int n)//n字节CBC模式，GPU
{
	ui K[36];
	generateKey(key, K);
	if (encrypt)//加密无法并行
	{
		int groups = n >> 4;
		generateRandom((ui*)dst);
		for (int i = 0; i < groups; ++i)
		{
			for (int j = 0; j < 16; ++j)
			{
				dst[(i + 1) * 16 + j] = dst[i * 16 + j] ^ src[i * 16 + j];
			}
			sm4(dst + (i + 1) * 16, K + 4, dst + (i + 1) * 16, encrypt);
		}
	}
	else
	{
		uc *gpusrc, *gpudst;
		ui *gpukey;
		CUDA_CALL(cudaMalloc((void**)&gpusrc, n + 16));
		CUDA_CALL(cudaMalloc((void**)&gpukey, 128));
		CUDA_CALL(cudaMalloc((void**)&gpudst, n));
		CUDA_CALL(cudaMemcpy(gpusrc, src, n + 16, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(gpukey, K + 4, 128, cudaMemcpyHostToDevice));
		SM4 <<< (n >> 4) / threads + 1, threads >>> (gpusrc + 16, gpukey, gpudst, encrypt, 0);
		CUDA_CALL(cudaDeviceSynchronize());
		XOR <<< (n >> 4) / threads + 1, threads >>> (gpusrc, gpudst);
		CUDA_CALL(cudaDeviceSynchronize());
		CUDA_CALL(cudaMemcpy(dst, gpudst, n, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaFree(gpusrc));
		CUDA_CALL(cudaFree(gpukey));
		CUDA_CALL(cudaFree(gpudst));
	}
}
void CTR(uc *src, uc *key, uc *dst, bool encrypt, int n)
{
	ui K[36];
	generateKey(key, K);
	uc *gpusrc, *gpudst;
	ui *gpukey;
	CUDA_CALL(cudaMalloc((void**)&gpukey, 128));
	CUDA_CALL(cudaMemcpy(gpukey, K + 4, 128, cudaMemcpyHostToDevice));
	if (encrypt)
	{
		generateRandom((ui*)dst);
		CUDA_CALL(cudaMalloc((void**)&gpusrc, n));
		CUDA_CALL(cudaMalloc((void**)&gpudst, n + 16));
		CUDA_CALL(cudaMemcpy(gpusrc, src, n, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(gpudst, dst, 16, cudaMemcpyHostToDevice));
		SM4 <<< (n >> 4) / threads + 1, threads >>> (gpudst, gpukey, gpudst + 16, true, 1);
		CUDA_CALL(cudaDeviceSynchronize());
		XOR <<< (n >> 4) / threads + 1, threads >>> (gpusrc, gpudst + 16);
		CUDA_CALL(cudaDeviceSynchronize());
		CUDA_CALL(cudaMemcpy(dst + 16, gpudst + 16, n, cudaMemcpyDeviceToHost));
	}
	else
	{
		CUDA_CALL(cudaMalloc((void**)&gpusrc, n + 16));
		CUDA_CALL(cudaMalloc((void**)&gpudst, n));
		CUDA_CALL(cudaMemcpy(gpusrc, src, n + 16, cudaMemcpyHostToDevice));
		SM4 <<< (n >> 4) / threads + 1, threads >>> (gpusrc, gpukey, gpudst, true, 1);
		CUDA_CALL(cudaDeviceSynchronize());
		XOR <<< (n >> 4) / threads + 1, threads >>> (gpusrc+16, gpudst);
		CUDA_CALL(cudaDeviceSynchronize());
		CUDA_CALL(cudaMemcpy(dst, gpudst, n, cudaMemcpyDeviceToHost));
	}
	CUDA_CALL(cudaFree(gpusrc));
	CUDA_CALL(cudaFree(gpukey));
	CUDA_CALL(cudaFree(gpudst));
}
int main()
{
	uc key[16] = { 0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,0xFE,0xDC,0xBA,0x98,0x76,0x54,0x32,0x10 };
	uc *plain, *encrypted;
	char input[1000];
	cout << "Enter source file: ";
	cin.getline(input, 1000);
	int size = readFile(input, &plain);
	int fullsize = (size >> 4 << 4) + (size & 0x0F ? 16 : 0);
	cout << "Enter destination file: ";
	cin.getline(input, 1000);
	encrypted = new uc[fullsize + 16];
	cout << "Full size: " << fullsize << "B" << endl;
	int time;
	//CPU测试
	cout << endl << InstructionSet::Brand() << endl;//CPU名称
	COUNT_TIME(ecb(plain, key, encrypted, true, fullsize), "ECB encrypt: ");
	COUNT_TIME(ecb(encrypted, key, plain, false, fullsize), "ECB decrypt: ");
	COUNT_TIME(cbc(plain, key, encrypted, true, fullsize), "CBC encrypt: ");
	COUNT_TIME(cbc(encrypted, key, plain, false, fullsize), "CBC decrypt: ");
	COUNT_TIME(ctr(plain, key, encrypted, true, fullsize), "CTR encrypt: ");
	COUNT_TIME(ctr(encrypted, key, plain, false, fullsize), "CTR decrypt: ");
	//GPU测试
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);//获取第一块显卡信息
	cout << endl << deviceProp.name << endl;//显卡名称
	threads = deviceProp.maxThreadsPerBlock;//调用核函数时需要用
	COUNT_TIME(ECB(plain, key, encrypted, true, fullsize), "ECB encrypt: ");
	COUNT_TIME(ECB(encrypted, key, plain, false, fullsize), "ECB decrypt: ");
	COUNT_TIME(CBC(plain, key, encrypted, true, fullsize), "CBC encrypt: ");
	COUNT_TIME(CBC(encrypted, key, plain, false, fullsize), "CBC decrypt: ");
	COUNT_TIME(CTR(plain, key, encrypted, true, fullsize), "CTR encrypt: ");
	COUNT_TIME(CTR(encrypted, key, plain, false, fullsize), "CTR decrypt: ");
	writeFile(input, plain, size);
	system("PAUSE");
    return 0;
}