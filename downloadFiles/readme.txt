ReadMe file for the PPA toolbox version 0.1 Monday, Jun 6, 2005 at 13:46:01
Written by Nathaniel J. King and Neil D. Lawrence.

License Info
------------

This software is free for academic use. Please contact Neil Lawrence if you are interested in using the software for commercial purposes.

This software must not be distributed or modified without prior permission of the author.


This is the probabilistic point assimilation code that was used in the paper KL Corrected Variational Inference for Gaussian Process Models. 

The code is written by Nathaniel J. King and Neil D. Lawrence. It relies on the following toolboxes.


KERN vs 0.12
------------

This toolbox implements the different kernels. IVM interacts with this toolbox through an interface which involves files starting with kern.

NOISE vs 0.12
-------------

This toolbox implements the different noise models. IVM interacts with this toolbox through an interface which involves files starting with noise.

NDLUTIL vs 0.12
---------------

This toolbox implements some generic functions which could be used beyond the ivm toolbox, for example sigmoid.m, cumGaussian.m

OPTIMI vs 0.12
--------------

This toolbox implements functions which allow non-linear transformations between parameters to be optimised. For example it allows variances to be optimised in log space.

PRIOR vs 0.12
-------------

This toolbox allows priors to be placed over parameters, at the moment this is used so that MAP solutions can be found for the parameters rather than type-II maximum likelihood. The priors were written for the Null Category Noise Model (see NCNM toolbox) so that an exponential prior could be placed over the process variances. The rest of its funcitonality has not been tested.

File Listing
------------

demPpa1.m: A simple demonstration of the probabilistic point assimilation.
demPpa2.m: A simple overlapping data set.
demPpa3.m: A simple demonstration on the Banana data.
demPpa4.m: A simple demonstration. 
ppa.m: Set up a probabilistic point assimilation model. 
ppaCalculateLogLike.m: Compute the log likelihood of the data.
ppaCalculateLogLike2.m: Compute the log likelihood of the data.
ppaContour.m: Special contour plot showing decision boundary.
ppaCovarianceGradient.m: The gradient of the likelihood approximation wrt the covariance.
ppaDisplay.m: Display parameters of PPA model.
ppaEStep.m: Perform the expectation step in the EM optimisation.
ppaExpectf.m: Compute the expectation of f.
ppaExpectfBar.m: Expectation under q(fBar).
ppaExpectfBarfBar.m: Second moment under q(fBar).
ppaExpectff.m: Second moment of f under q(f).
ppaGunnarData.m: Script for running experiments on Gunnar data.
ppaGunnarResultsTest.m: Helper script for collating results on Gunnar's benchmarks.
ppaGunnarTest.m: Script for running tests on Gunnar data.
ppaInit.m: Initialise the probabilistic point assimilation model.
ppaKernelGradient.m: Gradient of likelihood approximation wrt kernel parameters.
ppaKernelLogLikeGrad.m: Gradient of the kernel likelihood wrt kernel parameters.
ppaKernelLogLikelihood.m: Return the approximate log-likelihood for the PPA.
ppaKernelObjective.m: Likelihood approximation.
ppaMStep.m: Perform the M-step for probabilistic point assimilation.
ppaMeshVals.m: Give the output of the PPA for contour plot display.
ppaOptimiseKernel.m: Optimise the kernel parameters.
ppaOptimisePPA.m: Optimise the probabilistic point assimilation model.
ppaOptions.m: Default options for the probabilistic point assimilation.
ppaOut.m: Evaluate the output of an ppa model.
ppaPosteriorMeanVar.m: Mean and variances of the posterior at points given by X.
ppaTwoDPlot.m: Make a 2-D plot of the PPA.
ppaUpdateB.m: Update the individual values of B.
ppaUpdateBscalar.m: Update the values of B keeping each data dimension constant.
ppaUpdateKernel.m: Update the kernel parameters.
ppaVarLikeCovarianceGradient.m: The gradient of the variational likelihood approximation wrt the covariance.
ppaVarLikeKernelGradient.m: Gradient of variational likelihood approximation wrt kernel parameters.
ppaVarLikeKernelLogLikeGrad.m: Gradient of the kernel variational likelihood wrt kernel parameters.
ppaVarLikeKernelLogLikelihood.m: Return the approximate variational log-likelihood for the PPA.
ppaVarLikeKernelObjective.m: Variational Likelihood Kernel approximation.
ppaVarLikeOptimiseKernel.m: Optimise the kernel parameters using the variational log-likelihood.
