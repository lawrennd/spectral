% DEMOSPECTROGRAM A demonstration of spectral clustering on acoustic data (sigma=2.5).

% SPECTRAL

clear all;
close all;
clc

sigma2 = 2.5;

imageUnnorm = load('spectrogram.txt');
imageUnnorm = sum(imageUnnorm, 3);
imageNorm = imageUnnorm/max(max(imageUnnorm));
imageNorm = flipud(imageNorm);
figure
imagesc((imageNorm));
fprintf('This is the image to be segmented\nPress any key to continue\n');
pause

% image is too big, so downsample it by ratio samp
samp = 3;
rowindex = [1:samp:(floor(size(imageNorm, 1)/samp))*samp];
colindex = [1:samp:(floor(size(imageNorm, 2)/samp))*samp];
imageNorm = imageNorm(rowindex, colindex);

numrows = size(imageNorm, 1);
numcols = size(imageNorm, 2);
n = numcols*numrows;

% f is a 1 x n vector of the intensities of each pixel
f = reshape(imageNorm, 1, n)';
% x is a 2 x n matrix of the coordinates of each pixel
x1 = repmat([1:numrows], 1, numcols)';
x2 = reshape(repmat([1:numcols], numrows, 1), 1, n)';
% combine f and x, giving f the same weighting as x
x = [x1, x2, f*(numrows+numcols)];

% run the clustering algorithm
[labels, PcEig, Centres] = SpectralCluster(x, sigma2);
[npts, dim] = size(labels);

% choose a different symbol for each cluster
color = ['k', 'r', 'g', 'b', 'm'];
sym = ['>', 'o', '+', 'x', 's', 'd', 'v', '^', '<'];
symbol = zeros(dim, 2);
for i = 1:dim;
    symbol(i, :) = [color(mod(i, length(color))+1) sym(mod(i, length(sym))+1)];
end

% plot the final groupings
fprintf('I think the number of Clusters here is\n');
dim
figure
for i=1:npts
    plot(x(i,2), -x(i,1), char(symbol(find(labels(i,:)), :)));
    hold on
end
fprintf('This is the clustering we have found\nPress any key to continue\n');
pause

% plot the clusters in 3d eigenspace
if dim > 2
figure
plot3(Centres(:,1),Centres(:,2),Centres(:,3),'kd');
hold on
for i = 1:npts
    plot3(PcEig(i,1),PcEig(i,2),PcEig(i,3), char(symbol(find(labels(i,:)), :)));
end

fprintf('This is the eigenspace picture\n');
end