/* Author Alfredo Daniel Sanchez: alfredo.daniel.sanchez@gmail.com */

// Necessary headers
#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include <chrono>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cufft.h>

// Single precision Real and complex data types
using real_t = float;
using complex_t = cufftComplex;

// Define global constants
const real_t PI		= 3.14159265358979323846;	// pi
const real_t C		= 299792458*1E6/1E12;		// speed of ligth in vacuum [um/ps]
const real_t EPS0	= 8.8541878128E-12*1E12/1E6;	// vacuum pertivity [W.ps/V²μm] 
const uint SIZE		= 1 << 14;				// vector size
const uint NZ		= 1000;				// number of slices
const uint BLKX		= 16;					// block dimensions for kernels
const uint nBytesr	= sizeof(real_t)*SIZE;	 	// Mem. size for complex host vectors
const uint nBytesc	= sizeof(complex_t)*SIZE; 	// Mem. size for complex host vectors


// Package headers
#include "headers/common.cuh"
#include "headers/operators.cuh"
#ifdef PPLN // Mgo:PPLN nonlinear crystal
#include "headers/ppln.cuh"
#endif
#ifdef SPPLT // Mgo:sPPLT nonlinear crystal
#include "headers/spplt.cuh"
#endif
#include "headers/functions.cuh"
#include "headers/cwes3.cuh"
#include "headers/files.cuh"



int main(int argc, char *argv[]){
	
	std::cout << "\n\n\n#######---Welcome to OPG simulator---#######\n\n\n" << std::endl;
	
	////////////////////////////////////////////////////////////////////////////////////////
	//* Timing */
	double iStart = seconds();
	
	// Set up device (GPU)
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	std::cout << "\n\nUsing Device " << dev << ": GPU " << deviceProp.name << std::endl;
	CHECK(cudaSetDevice(dev));
	////////////////////////////////////////////////////////////////////////////////////////
	
	
	////////////////////////////////////////////////////////////////////////////////////////
	//* Define simulation parameters, physical quantities and set electric fields */
	// Define string variables for saving files
	std::string Filename, SAux, Extension = ".dat";
	
	// Grids, crystal and cavity parameters //
	real_t lp        	= atof(argv[1])*1e-3;  // pump wavelength   [μm]
	real_t Temp      	= atof(argv[2]);       // crystal temperature [ºC]
	real_t ls        	= findSignalPhaseMatching( lp, Temp, Lambda, 1.500, 1.600 );  // signal wavelength [μm]
	real_t li        	= lp*ls/(ls-lp);       // idler wavelength  [μm]
	
		
	// Define input pump parameters
	real_t FWHM		= atof(argv[4])*1e-3;			// intensity FWHM for input [ps]
	real_t sigmap	= FWHM*sqrtf(2)/(2*sqrtf(2*logf(2))); // σ of electric field gaussian pulse [ps]
	real_t waist	= 60;						// beam waist radius [um]
	real_t spot 	= PI*waist*waist;				// spot area [μm²]
	real_t repRate	= 10e6;					// repetition rate [MHz]
	real_t Energy	= atof(argv[3])*1e-9; // Enery per pulse [J]
	real_t Power	= Energy/(FWHM*1e-12);// Peak pump power in [W = J/s]
	real_t Inten	= Power/spot;				// Pump intensity in [W/um²]
	real_t Ap0		= sqrt(2*Inten/(n(lp, Temp)*EPS0*C));		// Input pump field strength [V/μm]
	
	
	// Time vector T for one round trip
	real_t *T = (real_t*) malloc(nBytesr);
	linspace( T, SIZE, -0.5*Twin, 0.5*Twin);
	
	// Frequency and angular frequency vectors f and Ω
	real_t *F = (real_t*) malloc(nBytesr);
	linspace( F, SIZE, -0.5*SIZE*dF, +0.5*SIZE*dF);
	real_t *w = (real_t*) malloc(nBytesr);
	fftshift(w,F, SIZE);
	for (uint i=0; i<SIZE; i++){
		w[i] = 2*PI*w[i]; // angular frequency [2*pi*THz]
	}
	
	
	// Define input pump vector
	complex_t *Ap = (complex_t*)malloc(nBytesc); // input pump vector
	real_t chirp = atof(argv[5]);
	bool readfile = false;
	if (readfile){
		std::string Measurement = "Measurement.dat";
		real_t* aux = (real_t*)malloc(nBytesr);
		readDataFromFile ( Measurement, aux );
		for (size_t i = 0; i < SIZE; i++){
			// Ap[i].x = sqrtf(500000/(0.5*PI*spot*EPS0*C*n(lp,Temp)))*sqrtf(aux[i]);
			Ap[i].x = 150*sqrtf(aux[i]);
			Ap[i].y = 0.0;
		}

		free(aux);
	}
	else{
		input_field_T(Ap, Ap0, T, sigmap, chirp, SIZE);	// Define string variables for saving files
	}
	
	
	// Define input signal vector (NOISE)
	complex_t *As = (complex_t*)malloc(nBytesc);
	NoiseGeneratorCPU ( As, SIZE );
	// Define input idler vector (NOISE)
	complex_t *Ai = (complex_t*)malloc(nBytesc);
	NoiseGeneratorCPU ( Ai, SIZE );
	
	
	
	bool save_input_fields = true;  // Save input fields files
	if (save_input_fields){
		Filename = "pump_input";	SaveVectorComplex (Ap, SIZE, Filename);
		Filename = "signal_input";	SaveVectorComplex (As, SIZE, Filename);
		Filename = "idler_input";	SaveVectorComplex (Ai, SIZE, Filename);	
	}
	
	
	bool print_param_on_screen = true;	// Print parameters on screen
	if ( print_param_on_screen ){
		std::cout << "\n\nSimulation parameters:\n\n " << std::endl;
		getCrystalProp ( lp, ls, li, Temp, Lambda );
		std::cout << "Time duration (FWHM)    = " << FWHM*1e3 << " fs" << std::endl;
		std::cout << "Ap0                     = " << Ap0 << " V/um" << std::endl; 
		std::cout << "waist                   = " << waist << " \u03BCm" << std::endl;
		std::cout << "spot                    = " << spot << " \u03BCm²" << std::endl;
		std::cout << "Enery per pulse         = " << atof(argv[3]) << " nJ" << std::endl;
		std::cout << "Peak pump power         = " << Power << " W" << std::endl;
		std::cout << "Repetition rate         = " << repRate*1e-6 << " MHz" << std::endl;
		std::cout << "Chirp parameter         = " << chirp << std::endl;
	}

	real_t *save_parameters = (real_t*) malloc(4*sizeof(real_t));
	save_parameters[0] = ls;
	save_parameters[1] = li;
	save_parameters[2] = 2*PI*( n(lp, Temp)/lp-n(ls, Temp)/ls-n(li, Temp)/li-1/Lambda );
	save_parameters[3] = 1/GV(li, Temp)-1/GV(lp, Temp);
	Filename = "Parameters.dat";	SaveVectorReal(save_parameters, 4, Filename);
	////////////////////////////////////////////////////////////////////////////////////////
	
	
	////////////////////////////////////////////////////////////////////////////////////////
	//* Define GPU vectors */
	
	// Define GPU vectors //
	real_t *w_gpu; // angular frequency 
	CHECK(cudaMalloc((void **)&w_gpu, nBytesr ));
	
	real_t *T_gpu;
	CHECK(cudaMalloc((void **)&T_gpu, nBytesr ));    
	CHECK(cudaMemcpy(T_gpu, T, nBytesr, cudaMemcpyHostToDevice));    
	CHECK(cudaMemcpy(w_gpu, w, nBytesr, cudaMemcpyHostToDevice));	
	
	complex_t *Ap_gpu, *Apw_gpu, *As_gpu, *Asw_gpu, *Ai_gpu, *Aiw_gpu;;
	CHECK(cudaMalloc((void **)&Ap_gpu, nBytesc ));	CHECK(cudaMalloc((void **)&Apw_gpu, nBytesc ))
	CHECK(cudaMalloc((void **)&As_gpu, nBytesc ));	CHECK(cudaMalloc((void **)&Asw_gpu, nBytesc ));
	CHECK(cudaMalloc((void **)&Ai_gpu, nBytesc ));	CHECK(cudaMalloc((void **)&Aiw_gpu, nBytesc ));
	
	CHECK(cudaMemcpy(Ap_gpu, Ap, nBytesc, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(As_gpu, As, nBytesc, cudaMemcpyHostToDevice));	
	CHECK(cudaMemcpy(Ai_gpu, Ai, nBytesc, cudaMemcpyHostToDevice));
	
	
	// RK4 (kx) and auxiliary (aux) GPU vectors 
	complex_t *k1p_gpu, *k2p_gpu, *k3p_gpu, *k4p_gpu;
	complex_t *k1s_gpu, *k2s_gpu, *k3s_gpu, *k4s_gpu;
	complex_t *k1i_gpu, *k2i_gpu, *k3i_gpu, *k4i_gpu;
	CHECK(cudaMalloc((void **)&k1p_gpu, nBytesc ));	CHECK(cudaMalloc((void **)&k2p_gpu, nBytesc ));
	CHECK(cudaMalloc((void **)&k3p_gpu, nBytesc ));	CHECK(cudaMalloc((void **)&k4p_gpu, nBytesc ));
	CHECK(cudaMalloc((void **)&k1s_gpu, nBytesc ));	CHECK(cudaMalloc((void **)&k2s_gpu, nBytesc ));
	CHECK(cudaMalloc((void **)&k3s_gpu, nBytesc ));	CHECK(cudaMalloc((void **)&k4s_gpu, nBytesc ));
	CHECK(cudaMalloc((void **)&k1i_gpu, nBytesc ));	CHECK(cudaMalloc((void **)&k2i_gpu, nBytesc ));
	CHECK(cudaMalloc((void **)&k3i_gpu, nBytesc ));	CHECK(cudaMalloc((void **)&k4i_gpu, nBytesc ));
	complex_t *auxp_gpu, *auxs_gpu, *auxi_gpu;
	CHECK(cudaMalloc((void **)&auxp_gpu, nBytesc ));
	CHECK(cudaMalloc((void **)&auxs_gpu, nBytesc ));
	CHECK(cudaMalloc((void **)&auxi_gpu, nBytesc ));	
	
	////////////////////////////////////////////////////////////////////////////////////////
	
	
	////////////////////////////////////////////////////////////////////////////////////////	
	// Single pass for coupled wave equations
	EvolutionInCrystal( w_gpu, Ap_gpu, As_gpu, Ai_gpu, Apw_gpu, Asw_gpu, Aiw_gpu,
				  k1p_gpu, k1s_gpu, k1i_gpu, k2p_gpu, k2s_gpu, k2i_gpu,
			   k3p_gpu, k3s_gpu, k3i_gpu, k4p_gpu, k4s_gpu, k4i_gpu,
			   auxp_gpu, auxs_gpu, auxi_gpu, lp, ls, li, Temp );
	
	////////////////////////////////////////////////////////////////////////////////////////
	
	
	////////////////////////////////////////////////////////////////////////////////////////	
	//* Saving results in .dat files using the function SaveVectorComplexGPU() //
	
	bool save_vectors = true; // Decide whether or not save the following vectors
	if (save_vectors){
		std::cout << "\nSaving time and frequency vectors...\n" << std::endl;
		Filename = "T"; SaveVectorReal (T, SIZE, Filename+Extension);
		Filename = "freq"; SaveVectorReal (F, SIZE, Filename+Extension);
	}
	else{ std::cout << "\nTime and frequency were previuosly save...\n" << std::endl;
	}
	
	// Save the simulation
	Filename = "signal_output";	SaveVectorComplexGPU ( As_gpu, SIZE, Filename );
	Filename = "pump_output";	SaveVectorComplexGPU ( Ap_gpu, SIZE, Filename );
	Filename = "idler_output";	SaveVectorComplexGPU ( Ai_gpu, SIZE, Filename );
	////////////////////////////////////////////////////////////////////////////////////////
	
	
	////////////////////////////////////////////////////////////////////////////////////////
	//* Deallocating memory from CPU and GPU and destroying plans */
	
	free(T);	free(w);	free(F);
	free(Ap);	free(As);	free(Ai); 
	
	CHECK(cudaFree(Ap_gpu)); 	CHECK(cudaFree(As_gpu));
	CHECK(cudaFree(Ai_gpu));	
	
	CHECK(cudaFree(T_gpu)); 	CHECK(cudaFree(w_gpu));
	CHECK(cudaFree(k1p_gpu));	CHECK(cudaFree(k2p_gpu));
	CHECK(cudaFree(k3p_gpu));     CHECK(cudaFree(k4p_gpu));
	CHECK(cudaFree(k1s_gpu));     CHECK(cudaFree(k2s_gpu));
	CHECK(cudaFree(k3s_gpu));     CHECK(cudaFree(k4s_gpu));
	CHECK(cudaFree(k1i_gpu));     CHECK(cudaFree(k2i_gpu));
	CHECK(cudaFree(k3i_gpu));     CHECK(cudaFree(k4i_gpu));	
	CHECK(cudaFree(auxs_gpu));    CHECK(cudaFree(auxp_gpu));
	CHECK(cudaFree(auxi_gpu));
	
	// Reset the GPU
	cudaDeviceReset();
	////////////////////////////////////////////////////////////////////////////////////////
	
	
	////////////////////////////////////////////////////////////////////////////////////////	
	//* Finish timing: returns the runtime simulation */
	double iElaps = seconds() - iStart;	// finish timing
	TimingCode( iElaps); // print time
	////////////////////////////////////////////////////////////////////////////////////////
	
	return 0;
}
