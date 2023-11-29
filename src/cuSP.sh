#!/bin/bash

# This file contains a set of instructions to run the main file cuSP.cu.
# Use this file to perform simulations in bulk. This means, for example, 
# systematically varying the input power, the cavity reflectivity, etc.
# For such propose, please insert for-loops for any variable parameter.


clear 	# Clear screen
# rm !("Measurement.dat")*.dat	# This removes all .dat files in the current folder. Comment this line for safe.
rm *.txt	# This removes all .txt files in the current folder. Comment this line for safe. 
rm cuSP	# This removes a previuos executable file (if it exist)

rm -r Simulations
# rm *.avi

########################################################################################################################################

## Compilation

# To compile CUDA files NVCC compiler is used here (https://developer.nvidia.com/cuda-downloads). 
# The main file cuSP.cu is compiled by

## for three coupled wave equations
nvcc cuSP.cu -DPPLN --gpu-architecture=sm_75 -lcufftw -lcufft -o cuSP  # for three equations



# FOLDERSIM="Simulations_T_variable"
# FOLDERSIM="Simulations_SPM_off"
# FOLDERSIM="Simulations_threshold_T_50fs"

# There are three flags specific for CUDA compiler:
# --gpu-architecture=sm_75: please check your GPU card architecture (Ampere, Fermi, Tesla, Pascal, Turing, Kepler, etc) 
#                           to set the correct number sm_XX. This code was tested using a Nvidia GeForce GTX 1650 card (Turing
#							architecture). 
# Please visit https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
#                           to set the proper flag according to your GPU card.
# -lcufftw -lcufft        : flags needed to perform cuFFT (the CUDA Fourier transform)
########################################################################################################################################


########################################################################################################################################
# The variables defined below (ARGX) will be passed as arguments to the main file 
# cuSP.cu on each execution via the argv[X] instruction.


LP=(1064)		# Pump wavelength				(ARG1)
LS=(1520)		# Signal wavelength				(ARG2)
TEMP=(141)		# Phase-matching temperature	(ARG3)
E=(65)			# Energy per pulse				(ARG4)
CH=(0)			# Chirp parameter				(ARG5)

# for TEMP in $t
# do

c=$(awk 'BEGIN{for(i=-4; i<=4; i=i+1)print i}')
t=$(awk 'BEGIN{for(i=40; i<=200; i=i+4)print i}')
td=$(awk 'BEGIN{for(i=100; i<=400; i=i+100)print i}')
e=$(awk 'BEGIN{for(i=20; i<=150; i=i+1)print i}')


for PULSEDUR in $td
do
	FOLDERSIM="Simulations_PulseDur_${PULSEDUR}fs"

	for CHIRP in $c
	do 
		for E in $e
		do
			# for (( c=0; c<${#CH[@]}; c++ ))
			# do 
			# CHIRP=${CH[$c]}
			printf "\nMaking directory...\n"
			FOLDER="MgOPPLN_E_${E}_nJ_LP_${LP}nm_Temp_${TEMP}C_Chirp_${CHIRP}"
			FILE="MgOPPLN_E_${E}_nJ_LP_${LP}nm_Temp_${TEMP}C_Chirp_${CHIRP}.txt"

			printf "Bash execution and writing output file...\n\n"
			./cuSP $LP $TEMP $E $PULSEDUR $CHIRP | tee -a $FILE

			printf "Bash finished!!\n\n" 
			mkdir $FOLDER
			mv Measurement.dat Measurement
			mv *.dat "$FOLDER/"
			mv *.txt $FOLDER"/"
			mv Measurement Measurement.dat
		done
	done


	if [ -d "$FOLDERSIM" ]; then
		echo "Moving simulations in ${FOLDERSIM}..."
		mv MgOPPLN* $FOLDERSIM"/" 
	else

		mkdir $FOLDERSIM

		echo "Creating and moving simulations in ${FOLDERSIM}..."
		mv MgOPPLN* $FOLDERSIM"/" 
	fi

done

# mv -v $FOLDERSIM"/" ..