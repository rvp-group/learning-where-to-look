/*
 * Slightly changed. Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#pragma once
#include "cuda_runtime.h"
#include "math.h"

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 make_float2(float s) {
  return make_float2(s, s);
}
__forceinline__ __host__ __device__ float2 make_float2(float3 a) {
  return make_float2(a.x, a.y);
}
__forceinline__ __host__ __device__ float2 make_float2(int2 a) {
  return make_float2(float(a.x), float(a.y));
}
__forceinline__ __host__ __device__ float2 make_float2(uint2 a) {
  return make_float2(float(a.x), float(a.y));
}

__forceinline__ __host__ __device__ int2 make_int2(int s) {
  return make_int2(s, s);
}
__forceinline__ __host__ __device__ int2 make_int2(int3 a) {
  return make_int2(a.x, a.y);
}
__forceinline__ __host__ __device__ int2 make_int2(uint2 a) {
  return make_int2(int(a.x), int(a.y));
}
__forceinline__ __host__ __device__ int2 make_int2(float2 a) {
  return make_int2(int(a.x), int(a.y));
}

__forceinline__ __host__ __device__ float3 make_float3(float s) {
  return make_float3(s, s, s);
}
__forceinline__ __host__ __device__ float3 make_float3(float2 a) {
  return make_float3(a.x, a.y, 0.0f);
}
__forceinline__ __host__ __device__ float3 make_float3(float2 a, float s) {
  return make_float3(a.x, a.y, s);
}
__forceinline__ __host__ __device__ float3 make_float3(float4 a) {
  return make_float3(a.x, a.y, a.z);
}
__forceinline__ __host__ __device__ float3 make_float3(int3 a) {
  return make_float3(float(a.x), float(a.y), float(a.z));
}
__forceinline__ __host__ __device__ float3 make_float3(uint3 a) {
  return make_float3(float(a.x), float(a.y), float(a.z));
}

__forceinline__ __host__ __device__ int3 make_int3(int s) {
  return make_int3(s, s, s);
}
__forceinline__ __host__ __device__ int3 make_int3(int2 a) {
  return make_int3(a.x, a.y, 0);
}
__forceinline__ __host__ __device__ int3 make_int3(int2 a, int s) {
  return make_int3(a.x, a.y, s);
}
__forceinline__ __host__ __device__ int3 make_int3(uint3 a) {
  return make_int3(int(a.x), int(a.y), int(a.z));
}
__forceinline__ __host__ __device__ int3 make_int3(float3 a) {
  return make_int3(int(a.x), int(a.y), int(a.z));
}

__forceinline__ __host__ __device__ float4 make_float4(float s) {
  return make_float4(s, s, s, s);
}
__forceinline__ __host__ __device__ float4 make_float4(float3 a) {
  return make_float4(a.x, a.y, a.z, 0.0f);
}
__forceinline__ __host__ __device__ float4 make_float4(float3 a, float w) {
  return make_float4(a.x, a.y, a.z, w);
}
__forceinline__ __host__ __device__ float4 make_float4(int4 a) {
  return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
__forceinline__ __host__ __device__ float4 make_float4(uint4 a) {
  return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

__forceinline__ __host__ __device__ int4 make_int4(int s) {
  return make_int4(s, s, s, s);
}
__forceinline__ __host__ __device__ int4 make_int4(int3 a) {
  return make_int4(a.x, a.y, a.z, 0);
}
__forceinline__ __host__ __device__ int4 make_int4(int3 a, int w) {
  return make_int4(a.x, a.y, a.z, w);
}
__forceinline__ __host__ __device__ int4 make_int4(uint4 a) {
  return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
__forceinline__ __host__ __device__ int4 make_int4(float4 a) {
  return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}

// some basic operators defined on host too
__forceinline__ __host__ __device__ float3 operator*(float3 a, float3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__forceinline__ __host__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__forceinline__ __host__ __device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__forceinline__ __host__ __device__ float3 operator/(float3 a, float3 b) {
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
__forceinline__ __host__ __device__ int3 roundf(float3 v) {
  return make_int3(roundf(v.x), roundf(v.y), roundf(v.z));
}

#ifdef __CUDACC__

__forceinline__ __host__ __device__ uint2 make_uint2(uint3 a) {
  return make_uint2(a.x, a.y);
}
__forceinline__ __host__ __device__ uint2 make_uint2(int2 a) {
  return make_uint2(uint(a.x), uint(a.y));
}

// some other constructors
__forceinline__ __host__ __device__ uint2 make_uint2(uint s) {
  return make_uint2(s, s);
}

__forceinline__ __host__ __device__ uint3 make_uint3(uint s) {
  return make_uint3(s, s, s);
}
__forceinline__ __host__ __device__ uint3 make_uint3(uint2 a) {
  return make_uint3(a.x, a.y, 0);
}
__forceinline__ __host__ __device__ uint3 make_uint3(uint2 a, uint s) {
  return make_uint3(a.x, a.y, s);
}
__forceinline__ __host__ __device__ uint3 make_uint3(uint4 a) {
  return make_uint3(a.x, a.y, a.z);
}
__forceinline__ __host__ __device__ uint3 make_uint3(int3 a) {
  return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

__forceinline__ __host__ __device__ uint4 make_uint4(uint s) {
  return make_uint4(s, s, s, s);
}
__forceinline__ __host__ __device__ uint4 make_uint4(uint3 a) {
  return make_uint4(a.x, a.y, a.z, 0);
}
__forceinline__ __host__ __device__ uint4 make_uint4(uint3 a, uint w) {
  return make_uint4(a.x, a.y, a.z, w);
}
__forceinline__ __host__ __device__ uint4 make_uint4(int4 a) {
  return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}

////////////////////////////////////////////////////////////////////////////////
// some additional ops
////////////////////////////////////////////////////////////////////////////////
__forceinline__ __device__ float squaredEuclideanDistance(const float3& a, const float3& b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

// std::swap in CUDA device code
template <typename T>
__forceinline__ __device__ void swap(T& a, T& b) {
  T temp = a;
  a      = b;
  b      = temp;
}

__forceinline__ __host__ __device__ float norm(const float3& vec) {
  return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__forceinline__ __host__ __device__ float3 operator+(const int3& a, const float3& b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

////////////////////////////////////////////////////////////////////////////////
// missing function by matthias voxelhash
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ int sign(float val) {
  return (float(0) < val) - (val < float(0));
}

__forceinline__ __host__ __device__ int4 sign(const float4& v) {
  return make_int4(sign(v.x), sign(v.y), sign(v.z), sign(v.w));
}

__forceinline__ __host__ __device__ int3 sign(const float3& v) {
  return make_int3(sign(v.x), sign(v.y), sign(v.z));
}

__forceinline__ __host__ __device__ int2 sign(const float2& v) {
  return make_int2(sign(v.x), sign(v.y));
}

// typedef uint uint;
// typedef unsigned short ushort;

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 operator-(float2& a) {
  return make_float2(-a.x, -a.y);
}
__forceinline__ __host__ __device__ int2 operator-(int2& a) {
  return make_int2(-a.x, -a.y);
}
__forceinline__ __host__ __device__ float3 operator-(float3& a) {
  return make_float3(-a.x, -a.y, -a.z);
}
__forceinline__ __host__ __device__ int3 operator-(int3& a) {
  return make_int3(-a.x, -a.y, -a.z);
}
__forceinline__ __host__ __device__ float4 operator-(float4& a) {
  return make_float4(-a.x, -a.y, -a.z, -a.w);
}
__forceinline__ __host__ __device__ int4 operator-(int4& a) {
  return make_int4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 operator+(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}
__forceinline__ __host__ __device__ void operator+=(float2& a, float2 b) {
  a.x += b.x;
  a.y += b.y;
}
__forceinline__ __host__ __device__ float2 operator+(float2 a, float b) {
  return make_float2(a.x + b, a.y + b);
}
__forceinline__ __host__ __device__ float2 operator+(float b, float2 a) {
  return make_float2(a.x + b, a.y + b);
}
__forceinline__ __host__ __device__ void operator+=(float2& a, float b) {
  a.x += b;
  a.y += b;
}

__forceinline__ __host__ __device__ int2 operator+(int2 a, int2 b) {
  return make_int2(a.x + b.x, a.y + b.y);
}
__forceinline__ __host__ __device__ void operator+=(int2& a, int2 b) {
  a.x += b.x;
  a.y += b.y;
}
__forceinline__ __host__ __device__ int2 operator+(int2 a, int b) {
  return make_int2(a.x + b, a.y + b);
}
__forceinline__ __host__ __device__ int2 operator+(int b, int2 a) {
  return make_int2(a.x + b, a.y + b);
}
__forceinline__ __host__ __device__ void operator+=(int2& a, int b) {
  a.x += b;
  a.y += b;
}

__forceinline__ __host__ __device__ uint2 operator+(uint2 a, uint2 b) {
  return make_uint2(a.x + b.x, a.y + b.y);
}

__forceinline__ __host__ __device__ int3 operator+(int3 a, uint3 b) {
  return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __host__ __device__ void operator+=(uint2& a, uint2 b) {
  a.x += b.x;
  a.y += b.y;
}
__forceinline__ __host__ __device__ uint2 operator+(uint2 a, uint b) {
  return make_uint2(a.x + b, a.y + b);
}
__forceinline__ __host__ __device__ uint2 operator+(uint b, uint2 a) {
  return make_uint2(a.x + b, a.y + b);
}
__forceinline__ __host__ __device__ void operator+=(uint2& a, uint b) {
  a.x += b;
  a.y += b;
}

__forceinline__ __host__ __device__ void operator+=(float3& a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
__forceinline__ __host__ __device__ float3 operator+(float3 a, float b) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}
__forceinline__ __host__ __device__ void operator+=(float3& a, float b) {
  a.x += b;
  a.y += b;
  a.z += b;
}

__forceinline__ __host__ __device__ int3 operator+(int3 a, int3 b) {
  return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__forceinline__ __host__ __device__ void operator+=(int3& a, int3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
__forceinline__ __host__ __device__ int3 operator+(int3 a, int b) {
  return make_int3(a.x + b, a.y + b, a.z + b);
}
__forceinline__ __host__ __device__ void operator+=(int3& a, int b) {
  a.x += b;
  a.y += b;
  a.z += b;
}

__forceinline__ __host__ __device__ uint3 operator+(uint3 a, uint3 b) {
  return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__forceinline__ __host__ __device__ void operator+=(uint3& a, uint3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
__forceinline__ __host__ __device__ uint3 operator+(uint3 a, uint b) {
  return make_uint3(a.x + b, a.y + b, a.z + b);
}
__forceinline__ __host__ __device__ void operator+=(uint3& a, uint b) {
  a.x += b;
  a.y += b;
  a.z += b;
}

__forceinline__ __host__ __device__ int3 operator+(int b, int3 a) {
  return make_int3(a.x + b, a.y + b, a.z + b);
}
__forceinline__ __host__ __device__ uint3 operator+(uint b, uint3 a) {
  return make_uint3(a.x + b, a.y + b, a.z + b);
}
__forceinline__ __host__ __device__ float3 operator+(float b, float3 a) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}

__forceinline__ __host__ __device__ float4 operator+(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__forceinline__ __host__ __device__ void operator+=(float4& a, float4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
__forceinline__ __host__ __device__ float4 operator+(float4 a, float b) {
  return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__forceinline__ __host__ __device__ float4 operator+(float b, float4 a) {
  return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__forceinline__ __host__ __device__ void operator+=(float4& a, float b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
}

__forceinline__ __host__ __device__ int4 operator+(int4 a, int4 b) {
  return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__forceinline__ __host__ __device__ void operator+=(int4& a, int4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
__forceinline__ __host__ __device__ int4 operator+(int4 a, int b) {
  return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__forceinline__ __host__ __device__ int4 operator+(int b, int4 a) {
  return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__forceinline__ __host__ __device__ void operator+=(int4& a, int b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
}

__forceinline__ __host__ __device__ uint4 operator+(uint4 a, uint4 b) {
  return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__forceinline__ __host__ __device__ void operator+=(uint4& a, uint4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
__forceinline__ __host__ __device__ uint4 operator+(uint4 a, uint b) {
  return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__forceinline__ __host__ __device__ uint4 operator+(uint b, uint4 a) {
  return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__forceinline__ __host__ __device__ void operator+=(uint4& a, uint b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 operator-(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}
__forceinline__ __host__ __device__ void operator-=(float2& a, float2 b) {
  a.x -= b.x;
  a.y -= b.y;
}
__forceinline__ __host__ __device__ float2 operator-(float2 a, float b) {
  return make_float2(a.x - b, a.y - b);
}
__forceinline__ __host__ __device__ float2 operator-(float b, float2 a) {
  return make_float2(b - a.x, b - a.y);
}
__forceinline__ __host__ __device__ void operator-=(float2& a, float b) {
  a.x -= b;
  a.y -= b;
}

__forceinline__ __host__ __device__ int2 operator-(int2 a, int2 b) {
  return make_int2(a.x - b.x, a.y - b.y);
}
__forceinline__ __host__ __device__ void operator-=(int2& a, int2 b) {
  a.x -= b.x;
  a.y -= b.y;
}
__forceinline__ __host__ __device__ int2 operator-(int2 a, int b) {
  return make_int2(a.x - b, a.y - b);
}
__forceinline__ __host__ __device__ int2 operator-(int b, int2 a) {
  return make_int2(b - a.x, b - a.y);
}
__forceinline__ __host__ __device__ void operator-=(int2& a, int b) {
  a.x -= b;
  a.y -= b;
}

__forceinline__ __host__ __device__ uint2 operator-(uint2 a, uint2 b) {
  return make_uint2(a.x - b.x, a.y - b.y);
}
__forceinline__ __host__ __device__ void operator-=(uint2& a, uint2 b) {
  a.x -= b.x;
  a.y -= b.y;
}
__forceinline__ __host__ __device__ uint2 operator-(uint2 a, uint b) {
  return make_uint2(a.x - b, a.y - b);
}
__forceinline__ __host__ __device__ uint2 operator-(uint b, uint2 a) {
  return make_uint2(b - a.x, b - a.y);
}
__forceinline__ __host__ __device__ void operator-=(uint2& a, uint b) {
  a.x -= b;
  a.y -= b;
}

__forceinline__ __host__ __device__ void operator-=(float3& a, float3 b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}
__forceinline__ __host__ __device__ float3 operator-(float3 a, float b) {
  return make_float3(a.x - b, a.y - b, a.z - b);
}
__forceinline__ __host__ __device__ float3 operator-(float b, float3 a) {
  return make_float3(b - a.x, b - a.y, b - a.z);
}
__forceinline__ __host__ __device__ void operator-=(float3& a, float b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
}

__forceinline__ __host__ __device__ int3 operator-(int3 a, int3 b) {
  return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__forceinline__ __host__ __device__ void operator-=(int3& a, int3 b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}
__forceinline__ __host__ __device__ int3 operator-(int3 a, int b) {
  return make_int3(a.x - b, a.y - b, a.z - b);
}
__forceinline__ __host__ __device__ int3 operator-(int b, int3 a) {
  return make_int3(b - a.x, b - a.y, b - a.z);
}
__forceinline__ __host__ __device__ void operator-=(int3& a, int b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
}

__forceinline__ __host__ __device__ uint3 operator-(uint3 a, uint3 b) {
  return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__forceinline__ __host__ __device__ void operator-=(uint3& a, uint3 b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}
__forceinline__ __host__ __device__ uint3 operator-(uint3 a, uint b) {
  return make_uint3(a.x - b, a.y - b, a.z - b);
}
__forceinline__ __host__ __device__ uint3 operator-(uint b, uint3 a) {
  return make_uint3(b - a.x, b - a.y, b - a.z);
}
__forceinline__ __host__ __device__ void operator-=(uint3& a, uint b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
}

__forceinline__ __host__ __device__ float4 operator-(float4 a, float4 b) {
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__forceinline__ __host__ __device__ void operator-=(float4& a, float4 b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
}
__forceinline__ __host__ __device__ float4 operator-(float4 a, float b) {
  return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}
__forceinline__ __host__ __device__ void operator-=(float4& a, float b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
}

__forceinline__ __host__ __device__ int4 operator-(int4 a, int4 b) {
  return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__forceinline__ __host__ __device__ void operator-=(int4& a, int4 b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
}
__forceinline__ __host__ __device__ int4 operator-(int4 a, int b) {
  return make_int4(a.x - b, a.y - b, a.z - b, a.w - b);
}
__forceinline__ __host__ __device__ int4 operator-(int b, int4 a) {
  return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}
__forceinline__ __host__ __device__ void operator-=(int4& a, int b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
}

__forceinline__ __host__ __device__ uint4 operator-(uint4 a, uint4 b) {
  return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__forceinline__ __host__ __device__ void operator-=(uint4& a, uint4 b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
}
__forceinline__ __host__ __device__ uint4 operator-(uint4 a, uint b) {
  return make_uint4(a.x - b, a.y - b, a.z - b, a.w - b);
}
__forceinline__ __host__ __device__ uint4 operator-(uint b, uint4 a) {
  return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}
__forceinline__ __host__ __device__ void operator-=(uint4& a, uint b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 operator*(float2 a, float2 b) {
  return make_float2(a.x * b.x, a.y * b.y);
}
__forceinline__ __host__ __device__ void operator*=(float2& a, float2 b) {
  a.x *= b.x;
  a.y *= b.y;
}
__forceinline__ __host__ __device__ float2 operator*(float2 a, float b) {
  return make_float2(a.x * b, a.y * b);
}
__forceinline__ __host__ __device__ float2 operator*(float b, float2 a) {
  return make_float2(b * a.x, b * a.y);
}
__forceinline__ __host__ __device__ void operator*=(float2& a, float b) {
  a.x *= b;
  a.y *= b;
}

__forceinline__ __host__ __device__ int2 operator*(int2 a, int2 b) {
  return make_int2(a.x * b.x, a.y * b.y);
}
__forceinline__ __host__ __device__ void operator*=(int2& a, int2 b) {
  a.x *= b.x;
  a.y *= b.y;
}
__forceinline__ __host__ __device__ int2 operator*(int2 a, int b) {
  return make_int2(a.x * b, a.y * b);
}
__forceinline__ __host__ __device__ int2 operator*(int b, int2 a) {
  return make_int2(b * a.x, b * a.y);
}
__forceinline__ __host__ __device__ void operator*=(int2& a, int b) {
  a.x *= b;
  a.y *= b;
}

__forceinline__ __host__ __device__ uint2 operator*(uint2 a, uint2 b) {
  return make_uint2(a.x * b.x, a.y * b.y);
}
__forceinline__ __host__ __device__ void operator*=(uint2& a, uint2 b) {
  a.x *= b.x;
  a.y *= b.y;
}
__forceinline__ __host__ __device__ uint2 operator*(uint2 a, uint b) {
  return make_uint2(a.x * b, a.y * b);
}
__forceinline__ __host__ __device__ uint2 operator*(uint b, uint2 a) {
  return make_uint2(b * a.x, b * a.y);
}
__forceinline__ __host__ __device__ void operator*=(uint2& a, uint b) {
  a.x *= b;
  a.y *= b;
}

__forceinline__ __host__ __device__ void operator*=(float3& a, float3 b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
}
__forceinline__ __host__ __device__ float3 operator*(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}
__forceinline__ __host__ __device__ float3 operator*(float b, float3 a) {
  return make_float3(b * a.x, b * a.y, b * a.z);
}
__forceinline__ __host__ __device__ void operator*=(float3& a, float b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
}

__forceinline__ __host__ __device__ int3 operator*(int3 a, int3 b) {
  return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__forceinline__ __host__ __device__ void operator*=(int3& a, int3 b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
}
__forceinline__ __host__ __device__ int3 operator*(int3 a, int b) {
  return make_int3(a.x * b, a.y * b, a.z * b);
}
__forceinline__ __host__ __device__ int3 operator*(int b, int3 a) {
  return make_int3(b * a.x, b * a.y, b * a.z);
}
__forceinline__ __host__ __device__ void operator*=(int3& a, int b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
}

__forceinline__ __host__ __device__ uint3 operator*(uint3 a, uint3 b) {
  return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__forceinline__ __host__ __device__ void operator*=(uint3& a, uint3 b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
}
__forceinline__ __host__ __device__ uint3 operator*(uint3 a, uint b) {
  return make_uint3(a.x * b, a.y * b, a.z * b);
}
__forceinline__ __host__ __device__ uint3 operator*(uint b, uint3 a) {
  return make_uint3(b * a.x, b * a.y, b * a.z);
}
__forceinline__ __host__ __device__ void operator*=(uint3& a, uint b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
}

__forceinline__ __host__ __device__ float4 operator*(float4 a, float4 b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__forceinline__ __host__ __device__ void operator*=(float4& a, float4 b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
}
__forceinline__ __host__ __device__ float4 operator*(float4 a, float b) {
  return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
__forceinline__ __host__ __device__ float4 operator*(float b, float4 a) {
  return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
__forceinline__ __host__ __device__ void operator*=(float4& a, float b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
}

__forceinline__ __host__ __device__ int4 operator*(int4 a, int4 b) {
  return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__forceinline__ __host__ __device__ void operator*=(int4& a, int4 b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
}
__forceinline__ __host__ __device__ int4 operator*(int4 a, int b) {
  return make_int4(a.x * b, a.y * b, a.z * b, a.w * b);
}
__forceinline__ __host__ __device__ int4 operator*(int b, int4 a) {
  return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
__forceinline__ __host__ __device__ void operator*=(int4& a, int b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
}

__forceinline__ __host__ __device__ uint4 operator*(uint4 a, uint4 b) {
  return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__forceinline__ __host__ __device__ void operator*=(uint4& a, uint4 b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
}
__forceinline__ __host__ __device__ uint4 operator*(uint4 a, uint b) {
  return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b);
}
__forceinline__ __host__ __device__ uint4 operator*(uint b, uint4 a) {
  return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
__forceinline__ __host__ __device__ void operator*=(uint4& a, uint b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 operator/(float2 a, float2 b) {
  return make_float2(a.x / b.x, a.y / b.y);
}
__forceinline__ __host__ __device__ void operator/=(float2& a, float2 b) {
  a.x /= b.x;
  a.y /= b.y;
}
__forceinline__ __host__ __device__ float2 operator/(float2 a, float b) {
  return make_float2(a.x / b, a.y / b);
}
__forceinline__ __host__ __device__ void operator/=(float2& a, float b) {
  a.x /= b;
  a.y /= b;
}
__forceinline__ __host__ __device__ float2 operator/(float b, float2 a) {
  return make_float2(b / a.x, b / a.y);
}

__forceinline__ __host__ __device__ void operator/=(float3& a, float3 b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
}
__forceinline__ __host__ __device__ float3 operator/(float3 a, float b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}
__forceinline__ __host__ __device__ void operator/=(float3& a, float b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
}
__forceinline__ __host__ __device__ float3 operator/(float b, float3 a) {
  return make_float3(b / a.x, b / a.y, b / a.z);
}

__forceinline__ __host__ __device__ float4 operator/(float4 a, float4 b) {
  return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
__forceinline__ __host__ __device__ void operator/=(float4& a, float4 b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
}
__forceinline__ __host__ __device__ float4 operator/(float4 a, float b) {
  return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
__forceinline__ __host__ __device__ void operator/=(float4& a, float b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
}
__forceinline__ __host__ __device__ float4 operator/(float b, float4 a) {
  return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 fminf(float2 a, float2 b) {
  return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}
__forceinline__ __host__ __device__ float3 fminf(float3 a, float3 b) {
  return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
__forceinline__ __host__ __device__ float4 fminf(float4 a, float4 b) {
  return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}

__forceinline__ __host__ __device__ int2 min(int2 a, int2 b) {
  return make_int2(min(a.x, b.x), min(a.y, b.y));
}
__forceinline__ __host__ __device__ int3 min(int3 a, int3 b) {
  return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
__forceinline__ __host__ __device__ int4 min(int4 a, int4 b) {
  return make_int4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

__forceinline__ __host__ __device__ uint2 min(uint2 a, uint2 b) {
  return make_uint2(min(a.x, b.x), min(a.y, b.y));
}
__forceinline__ __host__ __device__ uint3 min(uint3 a, uint3 b) {
  return make_uint3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
__forceinline__ __host__ __device__ uint4 min(uint4 a, uint4 b) {
  return make_uint4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 fmaxf(float2 a, float2 b) {
  return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}
__forceinline__ __host__ __device__ float3 fmaxf(float3 a, float3 b) {
  return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
__forceinline__ __host__ __device__ float4 fmaxf(float4 a, float4 b) {
  return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

__forceinline__ __host__ __device__ int2 max(int2 a, int2 b) {
  return make_int2(max(a.x, b.x), max(a.y, b.y));
}
__forceinline__ __host__ __device__ int3 max(int3 a, int3 b) {
  return make_int3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
__forceinline__ __host__ __device__ int4 max(int4 a, int4 b) {
  return make_int4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

__forceinline__ __host__ __device__ uint2 max(uint2 a, uint2 b) {
  return make_uint2(max(a.x, b.x), max(a.y, b.y));
}
__forceinline__ __host__ __device__ uint3 max(uint3 a, uint3 b) {
  return make_uint3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
__forceinline__ __host__ __device__ uint4 max(uint4 a, uint4 b) {
  return make_uint4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ __host__ float lerp(float a, float b, float t) {
  return a + t * (b - a);
}
__forceinline__ __device__ __host__ float2 lerp(float2 a, float2 b, float t) {
  return a + t * (b - a);
}
__forceinline__ __device__ __host__ float3 lerp(float3 a, float3 b, float t) {
  return a + t * (b - a);
}
__forceinline__ __device__ __host__ float4 lerp(float4 a, float4 b, float t) {
  return a + t * (b - a);
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ __host__ float clamp(float f, float a, float b) {
  return fmaxf(a, fminf(f, b));
}
__forceinline__ __device__ __host__ int clamp(int f, int a, int b) {
  return max(a, min(f, b));
}
__forceinline__ __device__ __host__ uint clamp(uint f, uint a, uint b) {
  return max(a, min(f, b));
}

__forceinline__ __device__ __host__ float2 clamp(float2 v, float a, float b) {
  return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}
__forceinline__ __device__ __host__ float2 clamp(float2 v, float2 a, float2 b) {
  return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
__forceinline__ __device__ __host__ float3 clamp(float3 v, float a, float b) {
  return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
__forceinline__ __device__ __host__ float3 clamp(float3 v, float3 a, float3 b) {
  return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
__forceinline__ __device__ __host__ float4 clamp(float4 v, float a, float b) {
  return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
__forceinline__ __device__ __host__ float4 clamp(float4 v, float4 a, float4 b) {
  return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

__forceinline__ __device__ __host__ int2 clamp(int2 v, int a, int b) {
  return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}
__forceinline__ __device__ __host__ int2 clamp(int2 v, int2 a, int2 b) {
  return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
__forceinline__ __device__ __host__ int3 clamp(int3 v, int a, int b) {
  return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
__forceinline__ __device__ __host__ int3 clamp(int3 v, int3 a, int3 b) {
  return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
__forceinline__ __device__ __host__ int4 clamp(int4 v, int a, int b) {
  return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
__forceinline__ __device__ __host__ int4 clamp(int4 v, int4 a, int4 b) {
  return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

__forceinline__ __device__ __host__ uint2 clamp(uint2 v, uint a, uint b) {
  return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}
__forceinline__ __device__ __host__ uint2 clamp(uint2 v, uint2 a, uint2 b) {
  return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
__forceinline__ __device__ __host__ uint3 clamp(uint3 v, uint a, uint b) {
  return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
__forceinline__ __device__ __host__ uint3 clamp(uint3 v, uint3 a, uint3 b) {
  return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
__forceinline__ __device__ __host__ uint4 clamp(uint4 v, uint a, uint b) {
  return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
__forceinline__ __device__ __host__ uint4 clamp(uint4 v, uint4 a, uint4 b) {
  return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float dot(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
}
__forceinline__ __host__ __device__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__forceinline__ __host__ __device__ float dot(float4 a, float4 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__forceinline__ __host__ __device__ int dot(int2 a, int2 b) {
  return a.x * b.x + a.y * b.y;
}
__forceinline__ __host__ __device__ int dot(int3 a, int3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__forceinline__ __host__ __device__ int dot(int4 a, int4 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__forceinline__ __host__ __device__ uint dot(uint2 a, uint2 b) {
  return a.x * b.x + a.y * b.y;
}
__forceinline__ __host__ __device__ uint dot(uint3 a, uint3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__forceinline__ __host__ __device__ uint dot(uint4 a, uint4 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float length(float2 v) {
  return sqrtf(dot(v, v));
}
__forceinline__ __host__ __device__ float length(float3 v) {
  return sqrtf(dot(v, v));
}
__forceinline__ __host__ __device__ float length(float4 v) {
  return sqrtf(dot(v, v));
}

__forceinline__ __host__ __device__ float length(const float3& v1, const float3& v2) {
  const float3 diff = v1 - v2;
  return sqrtf(dot(diff, diff));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 normalize(float2 v) {
  float invLen = rsqrtf(dot(v, v));
  return v * invLen;
}
__forceinline__ __host__ __device__ float3 normalize(float3 v) {
  float invLen = rsqrtf(dot(v, v));
  return v * invLen;
}
__forceinline__ __host__ __device__ float4 normalize(float4 v) {
  float invLen = rsqrtf(dot(v, v));
  return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// round
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ int2 roundf(float2 v) {
  return make_int2(roundf(v.x), roundf(v.y));
}

__forceinline__ __host__ __device__ int4 roundf(float4 v) {
  return make_int4(roundf(v.x), roundf(v.y), roundf(v.z), roundf(v.w));
}

__forceinline__ __host__ __device__ uint2 lroundf(float2 v) {
  return make_uint2(lroundf(v.x), lroundf(v.y));
}
__forceinline__ __host__ __device__ uint3 lroundf(float3 v) {
  return make_uint3(lroundf(v.x), lroundf(v.y), lroundf(v.z));
}
__forceinline__ __host__ __device__ uint4 lroundf(float4 v) {
  return make_uint4(lroundf(v.x), lroundf(v.y), lroundf(v.z), lroundf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 floorf(float2 v) {
  return make_float2(floorf(v.x), floorf(v.y));
}
__forceinline__ __host__ __device__ float3 floorf(float3 v) {
  return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
__forceinline__ __host__ __device__ float4 floorf(float4 v) {
  return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float fracf(float v) {
  return v - floorf(v);
}
__forceinline__ __host__ __device__ float2 fracf(float2 v) {
  return make_float2(fracf(v.x), fracf(v.y));
}
__forceinline__ __host__ __device__ float3 fracf(float3 v) {
  return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
__forceinline__ __host__ __device__ float4 fracf(float4 v) {
  return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 fmodf(float2 a, float2 b) {
  return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}
__forceinline__ __host__ __device__ float3 fmodf(float3 a, float3 b) {
  return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}
__forceinline__ __host__ __device__ float4 fmodf(float4 a, float4 b) {
  return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 fabs(float2 v) {
  return make_float2(fabs(v.x), fabs(v.y));
}
__forceinline__ __host__ __device__ float3 fabs(float3 v) {
  return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}
__forceinline__ __host__ __device__ float4 fabs(float4 v) {
  return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

__forceinline__ __host__ __device__ int2 abs(int2 v) {
  return make_int2(abs(v.x), abs(v.y));
}
__forceinline__ __host__ __device__ int3 abs(int3 v) {
  return make_int3(abs(v.x), abs(v.y), abs(v.z));
}
__forceinline__ __host__ __device__ int4 abs(int4 v) {
  return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float3 reflect(float3 i, float3 n) {
  return i - 2.0f * n * dot(n, i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ __host__ float smoothstep(float a, float b, float x) {
  float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
  return (y * y * (3.0f - (2.0f * y)));
}
__forceinline__ __device__ __host__ float2 smoothstep(float2 a, float2 b, float2 x) {
  float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
  return (y * y * (make_float2(3.0f) - (make_float2(2.0f) * y)));
}
__forceinline__ __device__ __host__ float3 smoothstep(float3 a, float3 b, float3 x) {
  float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
  return (y * y * (make_float3(3.0f) - (make_float3(2.0f) * y)));
}
__forceinline__ __device__ __host__ float4 smoothstep(float4 a, float4 b, float4 x) {
  float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
  return (y * y * (make_float4(3.0f) - (make_float4(2.0f) * y)));
}

#endif