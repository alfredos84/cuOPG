/*---------------------------------------------------------------------------*/
// * This file contains a set of functions based on the 
// * Sellmeier equations for the MgO:sPPLT nonlinear crystal and other 
// * properties of the χ⁽²⁾ material. Sellmeier equations from reference 
// * O. Gayer: Temperature and wavelength dependent refractive index 
// * equations for MgO-doped congruent and stoichiometric LiNbO3.
/*---------------------------------------------------------------------------*/

// All the functions have two input arguments:
// *     L: wavelenght in um
// *     T: temperature in degrees



#ifndef _PPLN 
#define _PPLN 

#pragma once

// z discretization, time and frequency discretization
__constant__ const real_t Twin		= 10.0; 		// [ps]
__constant__ const real_t Lcr		= 42e3;			// crystal length [um]
__constant__ const real_t dz		= Lcr/(NZ-1);	// number of z-steps in the crystal
__constant__ const real_t dT		= Twin/SIZE;	// time step in [ps]
__constant__ const real_t dF		= 1/Twin;		// frequency step in [THz]


// Define global constants
__constant__ const real_t d33		= 25.20e-6;			// Eff. second-order susceptibility (d33) [um/V] [Ref]
__constant__ const real_t dQ		= 2.0*d33/PI;		// Eff. second-order susceptibility for QPM [um/V]
__constant__ const real_t k			= 4.5e-6;    		// thermal conductivity [W/um K]
__constant__ const real_t alpha_crp	= 0.025e-4; 		// pump linear absorption [1/μm]
__constant__ const real_t alpha_crs	= 0.025e-4;  		// signal linear absorption [1/μm]
__constant__ const real_t alpha_cri	= 0.025e-4;  		// idler linear absorption [1/μm]
// __constant__ const real_t alpha_crs	= 0.002e-4;  		// signal linear absorption [1/μm]
// __constant__ const real_t alpha_cri	= 0.002e-4;  		// idler linear absorption [1/μm]
__constant__ const real_t chi3p		= 3.0*1e-9;			// χ⁽³⁾ in [um²/V²]
__constant__ const real_t chi3s		= 1.5*1e-9;			// χ⁽³⁾ in [um²/V²]
__constant__ const real_t chi3i		= 1.0*1e-9;			// χ⁽³⁾ in [um²/V²]
__constant__ const real_t Lambda	= 30.10;	   		// grating period for QPM [um]  

__constant__ const real_t beta_crs	= 0;			// signal 2-photons absorption [1/μm]
__constant__ const real_t rho		= 0;			// walk-off angle [rad] 


/** This function returns the MgO:sPPLT extraordinary refractive index */
__host__ __device__ real_t n(real_t L,real_t T)
{
	
	real_t f = (T - 24.5) * (T + 570.82);
	real_t a1 = 5.756;
	real_t a2 = 0.0983;
	real_t a3 = 0.2020;
	real_t a4 = 189.32;
	real_t a5 = 12.52;
	real_t a6 =  1.32e-2;
	real_t b1 =  2.860e-6;
	real_t b2 =  4.700e-8;
	real_t b3 =  6.113e-8;
	real_t b4 =  1.516e-4;
	real_t G1 = a1 + b1*f;
	real_t G2 = a2 + b2*f;
	real_t G3 = a3 + b3*f;
	real_t G4 = a4 + b4*f;
	return sqrtf(G1+G2/(powf(L,2) - powf(G3,2))+G4/(powf(L,2) - powf(a5,2))-a6*L*L);
	
}


/** Returns the first-order derivative of the 
 * refractive index respect to the wavelength dn/dλ. */
__host__ __device__ real_t dndl(real_t L,real_t T)
{
	
	real_t f = (T - 24.5) * (T + 570.82);
	real_t a2 = 0.0983;
	real_t a3 = 0.2020;
	real_t a4 = 189.32;
	real_t a5 = 12.52;
	real_t a6 =  1.32e-2;
	real_t b2 =  4.700e-8;
	real_t b3 =  6.113e-8;
	real_t b4 =  1.516e-4;
	real_t G2 = a2 + b2*f;
	real_t G3 = a3 + b3*f;
	real_t G4 = a4 + b4*f;
	
	return -L*(G2/powf((pow(L,2)-powf(G3,2)),2)+G4/powf((pow(L,2)-powf(a5,2)),2) + a6)/n(L, T);
	
}


/** Returns the second-order derivative of the
 * refractive index respect to the wavelength d²n/dλ². */
__host__ __device__ real_t d2ndl2(real_t L,real_t T)
{
	
	real_t f = (T - 24.5) * (T + 570.82);
	real_t a2 = 0.0983;
	real_t a3 = 0.2020;
	real_t a4 = 189.32;
	real_t a5 = 12.52;
	real_t b2 = 4.700e-8;
	real_t b3 = 6.113e-8;
	real_t b4 = 1.516e-4;
	real_t G2 = a2+b2*f;
	real_t G3 = a3+b3*f;
	real_t G4 = a4+b4*f;
	real_t G  = G2/powf(powf(L,2)-powf(G3,2),3) + G4/powf(powf(L,2)-powf(a5,2),3);			

	real_t A  = (1./L - dndl(L,T)/n(L,T))*dndl(L,T);
	real_t B  = 4*L*L/n(L,T) * G;
	
	return A+B;
	
}


/** Returns the third-order derivative of the
 * refractive index respect to the wavelength d³n/dλ³. */
__host__ __device__ real_t d3ndl3(real_t L,real_t T)
{
	
	real_t f = (T - 24.5) * (T + 570.82);
	real_t a2 = 0.0983;
	real_t a3 = 0.2020;
	real_t a4 = 189.32;
	real_t a5 = 12.52;
	real_t b2 = 4.700e-8;
	real_t b3 = 6.113e-8;
	real_t b4 = 1.516e-4;
	real_t G2 = a2+b2*f;
	real_t G3 = a3+b3*f;
	real_t G4 = a4+b4*f;

	real_t G  = G2/powf(powf(L,2)-powf(G3,2),3) + G4/powf(powf(L,2)-powf(a5,2),3);
	real_t dG = G2/powf(powf(L,2)-powf(G3,2),4) + G4/powf(powf(L,2)-powf(a5,2),4);

	real_t AA = -(1/L + 4*L*L*G/powf(n(L,T),2))*dndl(L,T);
	real_t BB = +(1/L-2*dndl(L,T)/n(L,T))*d2ndl2(L,T);
	real_t CC = +powf(dndl(L,T),3)/powf(n(L,T),2);
	real_t DD = +8*L*G/n(L,T);
	real_t EE = -24*powf(L,3)*dG/n(L,T);

	return AA+BB+CC+DD+EE;	
}


/** Returns the group-velocity vg(λ) = c/(n(λ)-λdn/dλ). */
__host__ __device__ real_t GV(real_t L,real_t T)
{
	
	return C/(n(L,T)-L*dndl(L,T));
}


/** Returns the group-velocity β2(λ)=λ^3/(2πc²)(d²n/dλ²). */
__host__ __device__ real_t GVD(real_t L,real_t T)
{
	return powf(L,3)*d2ndl2(L, T)/(2*PI*C*C);
}


/** Returns the TOD β3(λ)=-λ^4/(4π²c³)[3.d²n/dλ² + λ.d³n/dλ³]. */
__host__ __device__ real_t TOD(real_t L,real_t T)
{
	return -powf(L,4)/(4*PI*PI*C*C*C)*(3*d2ndl2(L, T)+L*d3ndl3(L, T));
}


void getCrystalProp ( real_t lp, real_t ls, real_t li, real_t Temp, real_t Lambda )
{
	std::cout << "\n\nUsing a 5%-MgO:PPLN nonlinear crystal\n\n " << std::endl;
	std::cout << "Using N                 = " << SIZE << " points" << std::endl;
	std::cout << "Pump wavelength         = " << lp*1e3 << " nm" << std::endl;
	std::cout << "Signal wavelength       = " << ls*1e3 << " nm" << std::endl;
	std::cout << "Idler wavelength        = " << li*1e3 << " nm" << std::endl;
	std::cout << "Temp                    = " << Temp << " ºC" << std::endl;
	std::cout << "np                      = " << n(lp, Temp) << std::endl;
	std::cout << "ns                      = " << n(ls, Temp) << std::endl;
	std::cout << "ni                      = " << n(li, Temp) << std::endl;
	std::cout << "\u03BD⁻¹ pump                = " << 1.0/GV(lp, Temp) << " ps/\u03BCm" << std::endl;
	std::cout << "\u03BD⁻¹ signal              = " << 1.0/GV(ls, Temp) << " ps/\u03BCm" << std::endl;
	std::cout << "\u03BD⁻¹ idler               = " << 1.0/GV(li, Temp) << " ps/\u03BCm" << std::endl;		
	std::cout << "\u0394k                      = " << 2*PI*( n(lp, Temp)/lp-n(ls, Temp)/ls-n(li, Temp)/li-1/Lambda ) << " \u03BCm⁻¹" << std::endl;
	std::cout << "\u0394k'                     = " << 1/GV(lp, Temp)-1/GV(li, Temp) << " ps/\u03BCm" << std::endl;	
	std::cout << "GVD pump                = " << GVD(lp, Temp) << " ps²/\u03BCm" << std::endl;
	std::cout << "GVD signal              = " << GVD(ls, Temp) << " ps²/\u03BCm" << std::endl;
	std::cout << "GVD idler               = " << GVD(li, Temp) << " ps²/\u03BCm" << std::endl;		
	std::cout << "TOD pump                = " << TOD(lp, Temp) << " ps²/\u03BCm" << std::endl;
	std::cout << "TOD signal              = " << TOD(ls, Temp) << " ps²/\u03BCm" << std::endl;
	std::cout << "TOD idler               = " << TOD(li, Temp) << " ps²/\u03BCm" << std::endl;
	std::cout << "\u03A7p⁽³⁾                   = " << chi3p << " [\u03BCm²/V²]" << std::endl;
	std::cout << "dQ			= " << dQ*1e6 << " pm/V"  << std::endl;
	std::cout << "\u039B                       = " << Lambda << " \u03BCm"  << std::endl;
	std::cout << "\u03B1cp                     = " << alpha_crp << " \u03BCm⁻¹"  << std::endl;
	std::cout << "\u03B1cs                     = " << alpha_crs << " \u03BCm⁻¹" << std::endl;
	std::cout << "\u03B1ci                     = " << alpha_cri << " \u03BCm⁻¹" << std::endl;
	std::cout << "Crystal length          = " << Lcr*1e-3 << " mm"  << std::endl;
	std::cout << "\u0394z                      = " << dz << " \u03BCm"  << std::endl;
	std::cout << "dT                      = " << dT << " ps" << std::endl;
	
	return ;
	
}


/**
 * Letter   Description  Escape-Sequence
 * -------------------------------------
 * A        Alpha        \u0391
 * B        Beta         \u0392
 * Γ        Gamma        \u0393
 * Δ        Delta        \u0394
 * Ε        Epsilon      \u0395
 * Ζ        Zeta         \u0396
 * Η        Eta          \u0397
 * Θ        Theta        \u0398
 * Ι        Iota         \u0399
 * Κ        Kappa        \u039A
 * TOD idlerΛ        Lambda       \u039B
 * Μ        Mu           \u039C
 * Ν        Nu           \u039D
 * Ξ        Xi           \u039E
 * Ο        Omicron      \u039F
 * Π        Pi           \u03A0
 * Ρ        Rho          \u03A1
 * Σ        Sigma        \u03A3
 * Τ        Tau          \u03A4
 * Υ        Upsilon      \u03A5
 * Φ        Phi          \u03A6
 * Χ        Chi          \u03A7
 * Ψ        Psi          \u03A8
 * Ω        Omega        \u03A9 
 * -------------------------------------
 * Letter   Description  Escape-Sequence
 * -------------------------------------
 * α        Alpha        \u03B1
 * β        Beta         \u03B2
 * γ        Gamma        \u03B3
 * δ        Delta        \u03B4
 * ε        Epsilon      \u03B5
 * ζ        Zeta         \u03B6
 * η        Eta          \u03B7
 * θ        Theta        \u03B8
 * ι        Iota         \u03B9
 * κ        Kappa        \u03BA
 * λ        Lambda       \u03BB
 * μ        Mu           \u03BC
 * ν        Nu           \u03BD
 * ξ        Xi           \u03BE
 * ο        Omicron      \u03BF
 * π        Pi           \u03C0
 * ρ        Rho          \u03C1
 * σ        Sigma        \u03C3
 * τ        Tau          \u03C4
 * υ        Upsilon      \u03C5
 * φ        Phi          \u03C6
 * χ        Chi          \u03C7
 * ψ        Psi          \u03C8
 * ω        Omega        \u03C9
 * -------------------------------------
 */


#endif // -> #ifdef _PPLN
