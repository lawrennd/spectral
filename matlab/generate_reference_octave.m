% generate_reference_octave.m
% Generate reference data using Octave

% Set random seeds for reproducibility
randn('seed', 1);
rand('seed', 1);

% Generate three circles data (from demoCircles.m)
npts = 100;
step = 2*pi/npts;
theta = [step:step:2*pi];
radius = randn(1,npts);
r1 = [ones(1,npts)+0.1*radius];
r2 = [2*ones(1,npts)+0.1*radius];
r3 = [3*ones(1,npts)+0.1*radius];
x = [(r1.*cos(theta))', (r1.*sin(theta))';...
    (r2.*cos(theta))', (r2.*sin(theta))';...
    (r3.*cos(theta))', (r3.*sin(theta))'];

fprintf('Generated three circles: %d points\n', size(x,1));

% Run clustering with Octave-compatible version
sigma = 0.05;
fprintf('Running SpectralCluster_octave with sigma=%g...\n', sigma);
[labels, PcEig, Centres] = SpectralCluster_octave(x, sigma);

fprintf('Detected %d clusters\n', size(labels, 2));

% Save outputs
fprintf('Saving reference data...\n');
save('../tests/fixtures/octave_three_circles.mat', 'x', 'sigma', 'labels', 'PcEig', 'Centres', '-v7');

% Also save intermediate results for component testing
fprintf('Saving affinity matrix...\n');
A=zeros(size(x,1),size(x,1));
for i=1:size(x,1)
    for j=1:size(x,1)
        A(i,j) = exp(-norm(x(i,:)-x(j,:))^2/sigma);
    end
end
save('../tests/fixtures/octave_affinity.mat', 'x', 'sigma', 'A', '-v7');

fprintf('Saving Laplacian...\n');
D = sum(A,2);
L = inv(diag(sqrt(D)))*(A/diag(sqrt(D)));
save('../tests/fixtures/octave_laplacian.mat', 'A', 'L', '-v7');

fprintf('\n=================================================================\n');
fprintf('Reference data generated successfully!\n');
fprintf('=================================================================\n');
fprintf('Detected clusters: %d\n', size(labels, 2));
fprintf('Files created in ../tests/fixtures/:\n');
fprintf('  - octave_three_circles.mat\n');
fprintf('  - octave_affinity.mat\n');
fprintf('  - octave_laplacian.mat\n');
fprintf('\nNext step: Convert to NumPy format\n');
fprintf('Run: cd ../tests && python convert_matlab_reference.py\n');
fprintf('=================================================================\n');
