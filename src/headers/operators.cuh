/*---------------------------------------------------------------------------*/
// * This file contains a set of overloaded operators to deal with complex numbers.
/*---------------------------------------------------------------------------*/


#ifndef _OPERATORSCUH
#define _OPERATORSCUH

#pragma once


/////////////////////////////////////     OPERATORS     ////////////////////////////////////////
__host__ __device__ inline complex_t  operator+(const real_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a   + b.x;
	c.y =     + b.y;
	
	return c;
}


__host__ __device__ inline complex_t  operator+(const complex_t &b, const real_t &a) {
	
	complex_t c;    
	c.x = a   + b.x;
	c.y =     + b.y;
	
	return c;
}


__host__ __device__ inline complex_t  operator+(const complex_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	
	return c;
}

__host__ __device__ inline complex_t  operator-(const complex_t &a) {
	
	complex_t c;    
	c.x = -a.x;
	c.y = -a.y;
	
	return c;
}

__host__ __device__ inline complex_t  operator-(const real_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a   - b.x;
	c.y =     - b.y;
	
	return c;
}


__host__ __device__ inline complex_t  operator-(const complex_t &b, const real_t &a) {
	
	complex_t c;    
	c.x =  b.x - a ;
	c.y =  b.y ;
	
	return c;
}


__host__ __device__ inline complex_t  operator-(const complex_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a.x - b.x;
	c.y = a.y - b.y;
	
	return c;
}


__host__ __device__ inline complex_t  operator*(const real_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a * b.x ;
	c.y = a * b.y ;
	
	return c;
}


__host__ __device__ inline complex_t  operator*(const complex_t &b, const real_t &a) {
	
	complex_t c;    
	c.x = a * b.x ;
	c.y = a * b.y ;
	
	return c;
}


__host__ __device__ inline complex_t  operator*(const complex_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a.x * b.x - a.y * b.y ;
	c.y = a.x * b.y + a.y * b.x ;
	
	return c;
}


__host__ __device__ inline complex_t  operator/(const complex_t &b, const real_t &a) {
	
	complex_t c;    
	c.x = b.x / a ;
	c.y = b.y / a ;
	
	return c;
}


__host__ __device__ inline complex_t operator/(const real_t& a, const complex_t& b) {

	real_t denominator = b.x * b.x + b.y * b.y;
	complex_t c;
	c.x = (+a * b.x) / denominator;
	c.y = (-a * b.y) / denominator;

	return c;
}


__host__ __device__ inline complex_t operator/(const complex_t& a, const complex_t& b) {

	real_t denominator = b.x * b.x + b.y * b.y;
	complex_t c;
	c.x = (a.x * b.x + a.y * b.y) / denominator;
	c.y = (a.y * b.x - a.x * b.y) / denominator;

	return c;
}


/////////////////////////////////////////////////////////////////////////////////////////////////

//* Complex exponential e^(i*a) */
__host__ __device__ complex_t CpxExp (real_t a)
{
	complex_t b;
	b.x = cosf(a) ;	b.y = sinf(a) ;
	
	return b;
}


//* Complex exponential e^(a+i*b) */
__host__ __device__ complex_t CpxExp (complex_t a)
{
	complex_t b;
	b.x = expf(a.x)*cosf(a.y) ;	b.y = expf(a.x)*sinf(a.y) ;
	
	return b;
}


//* Complex conjugate */
__host__ __device__ complex_t CpxConj (complex_t a)
{
	complex_t b;
	b.x = +a.x ; b.y = -a.y ;
	
	return b;
}


//* Complex absolute value  */
__host__ __device__ real_t CpxAbs (complex_t a)
{
	real_t b;
	b = sqrtf(a.x*a.x + a.y*a.y);
	
	return b;
}


//* Complex square absolute value */
__host__ __device__ real_t CpxAbs2 (complex_t a)
{
	real_t b;
	b = a.x*a.x + a.y*a.y;
	
	return b;
}



__host__ __device__ complex_t CpxSqrt(complex_t z)
{
    real_t magnitude = sqrtf(z.x * z.x + z.y * z.y);
    real_t real = sqrtf(0.5f * (magnitude + z.x));
    real_t imag = sqrtf(0.5f * (magnitude - z.x));

    if (z.y < 0)
        imag = -imag;

    return make_cuFloatComplex(real, imag);
	
}



#endif // -> #ifdef _OPERATORSCUH
