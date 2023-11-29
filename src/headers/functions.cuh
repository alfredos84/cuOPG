/*---------------------------------------------------------------------------*/
// * This file contains a set of functions useful for the execution of the main file.
/*---------------------------------------------------------------------------*/


#ifndef _FUNCTIONSCUH
#define _FUNCTIONSCUH

#pragma once

/** Read data from external file */
void readDataFromFile (std::string nombreArchivo, real_t* inputField)
{
	// Open file
	std::ifstream file(nombreArchivo);
	
	// Verificar si se pudo abrir el file
	if (!file.is_open()) {
		std::cerr << "Open file " << nombreArchivo << std::endl;
	}
	
	// Vector to store data from file
	std::vector<real_t> aux;
	
	// Variable to store each line from file
	real_t counter;
	
	// Read each line from file and push in the vector
	while (file >> counter) {
		aux.push_back(counter);
	}
	
	// Copy vector to pointer
	for (size_t i = 0; i < aux.size(); ++i) {
		inputField[i] = aux[i];
		std::cout << inputField[i] << std::endl;
	}
	
	// Cerrar el file
	file.close();
	
	return ;
}


/** Noise generator for initial signal/idler vectors  */
void NoiseGeneratorCPU ( complex_t *A,  uint SIZE )
{
	uint seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<real_t> distribution(0.0,1.0e-15);
	
	real_t nsx, nsy;    
	for (int i=0; i<SIZE; ++i) {
		nsx = distribution(generator); A[i].x = nsx;
		nsy = distribution(generator); A[i].y = nsy;
	}
	
	return ;
}


/** Define an input field (usually the pump field) in the time
 * domain. This function is overloaded and its use depends on
 * the employed regime (cw, nanosecond or user-defined)*/
void input_field_T(complex_t *A, real_t A0, real_t *T, real_t T0, int SIZE)
{
	for (int i = 0; i < SIZE; i++){
		A[i].x = A0*exp(-powf( T[i]/T0,2)/2.0 ); // Gaussian pulse
		A[i].y = 0.0;
	}
	
	return ;
}

void input_field_T(complex_t *A, real_t A0, real_t *T, real_t T0, real_t chirp, int SIZE)
{
	complex_t Im; Im.x = 0; Im.y = 1;
	for (int i = 0; i < SIZE; i++){
		A[i]=A0*expf(-0.5*powf(T[i]/T0,2))*CpxExp(-0.5*Im*chirp*powf(T[i]/T0,2)); // Gaussian pulse
	}
	
	return ;
}


/** Define an input field (usually the pump field) in the time
 * domain. This function is overloaded and its use depends on
 * the employed regime (cw, nanosecond or user-defined)*/
void input_field_T(complex_t *A, real_t A0, int SIZE )
{
	
	for (int i = 0; i < SIZE; i++){
		A[i].x = A0; // cw field
		A[i].y = 0.0; // cw field
	}
	
	return ;
}


/** Linear spacing for time vectors */
void linspace( real_t *T, int SIZE, real_t xmin, real_t xmax)
{
	for (int i = 0; i < SIZE; i++)
		T[i] = xmin + i * (xmax - xmin)/(SIZE-1);
	
	return ;
}


/** Initializes the frequency vectors */
void inic_vector_F(real_t *F, int SIZE, real_t DF)
{
	for (int i = 0; i < SIZE; i++){
		F[i] = i * DF - SIZE* DF/2.0;
	}
	
	return ;
}

/** Flips a vector for Fourier transforms */
void fftshift( real_t *W_flip, real_t *W, int SIZE )
{
	int i, c = SIZE/2;
	for ( i = 0; i < SIZE/2; i++ ){
		W_flip[i+c] = W[i];
		W_flip[i]   = W[i+c];
	}
	
	return ;
}


/** Scales a vector after Fourier transforms (CUFFT_INVERSE mode) */
__global__ void CUFFTscale(complex_t *A, uint SIZE)
{
	uint idx = blockIdx.x*blockDim.x+threadIdx.x;
	real_t s = SIZE;
	if ( idx < SIZE){
		A[idx] = A[idx] / s;
	}
	
	return ;
}


#endif // -> #ifdef _FUNCTIONSCUH
