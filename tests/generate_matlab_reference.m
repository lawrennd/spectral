% generate_matlab_reference.m
% Generate reference data for Python validation
%
% This script runs the MATLAB implementation and saves outputs
% for comparison with the Python implementation.
%
% Usage:
%   1. cd to spectral/tests directory
%   2. Run: matlab -batch "generate_matlab_reference"
%   3. Convert .mat to .npz using provided Python script

% Three circles example
cd('../matlab');
x = demoCircles();
sigma = 0.05;

% Run clustering
[labels, PcEig, Centres] = SpectralCluster(x, sigma);

% Save outputs
cd('../tests/fixtures');
save('matlab_three_circles.mat', 'x', 'sigma', 'labels', 'PcEig', 'Centres');

% Also save intermediate results for component testing
cd('../../matlab');
A = zeros(size(x,1), size(x,1));
for i=1:size(x,1)
    for j=1:size(x,1)
        A(i,j) = exp(-norm(x(i,:)-x(j,:))^2/sigma);
    end
end
D = sum(A,2);
L = inv(diag(sqrt(D)))*(A/diag(sqrt(D)));

cd('../tests/fixtures');
save('matlab_affinity.mat', 'x', 'sigma', 'A');
save('matlab_laplacian.mat', 'A', 'L');

cd('../..');
fprintf('\n=================================================================\n');
fprintf('MATLAB reference data generated in tests/fixtures/\n');
fprintf('=================================================================\n');
fprintf('\nNext step: Convert .mat files to .npz format\n');
fprintf('Run: python tests/convert_matlab_reference.py\n');
fprintf('=================================================================\n');
