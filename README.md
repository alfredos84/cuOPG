# cuOPG
Optical parametetric generation using ultrashort pulses

This package contains an implementation of the coupled-wave equations for three-wave mixing processes in second-order nonlinear media such as nonlinear crystals.

The package includes the nonlinear coupling between the involved electric fields, the dispersion terms up to the thrid order (GVM, GVD, TOD), the effect the self-phase modulation (SPM)
and the linar absorption.

The script is written in CUDA language in order to exploit the speed up provided by a GPU.

So far, there is no metric which has been measured, but a single simulation takes less than 5 seconds, depending on the grid size.

For any questions, comments or/and suggestions, please do not hesitate to contact me.
alfredo.daniel.sanchez@gmail.com
