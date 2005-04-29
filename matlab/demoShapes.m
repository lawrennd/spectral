% DEMOSHAPES A demonstration of spectral clustering on the shapes data set (sigma=1).

% SPECTRAL

clear all;
close all;
clc

sigma2 = 1;

% read in and normalise the image
imageUnnorm = double(imread('shapes.bmp'));
imageUnnorm = sum(imageUnnorm, 3);
imageNorm = imageUnnorm/max(max(imageUnnorm));

% image may be too big, so downsample it by ratio samp
samp = 1;
rowindex = [1:samp:(floor(size(imageNorm, 1)/samp))*samp];
colindex = [1:samp:(floor(size(imageNorm, 2)/samp))*samp];
imageNorm = imageNorm(rowindex, colindex);

% extract from the image a vector f of intensity values and a matrix x of
% their corresponding coordinates
[numrows, numcols] = size(imageNorm);
n = numcols*numrows;
f = reshape(imageNorm, 1, n)';
x1 = repmat([1:numrows], 1, numcols)';
x2 = reshape(repmat([1:numcols], numrows, 1), 1, n)';

% combine f and x, giving f the same weighting as x
x = [x1, x2, f*(numrows+numcols)];
figure
imagesc(flipud(imageNorm'));
fprintf('This is the image to be segmented\nPress any key to continue\n');
pause

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
    plot(x(i,1), x(i,2), char(symbol(find(labels(i,:)), :)));
    hold on
end
fprintf('This is the clustering we have found\nPress any key to continue\n');
pause

% plot the clusters in 2d eigenspace
figure
plot(Centres(:,1),Centres(:,2),'kd');
hold on
for i = 1:npts
    plot(PcEig(i,1),PcEig(i,2), char(symbol(find(labels(i,:)), :)));
end
fprintf('This is the eigenspace picture\n');

% % plot the clusters in 3d eigenspace
% figure
% plot3(Centres(:,1),Centres(:,2),Centres(:,3),'kd');
% hold on
% for i = 1:npts
%     plot3(PcEig(i,1),PcEig(i,2),PcEig(i,3), char(symbol(find(labels(i,:)), :)));
% end
% fprintf('This is the eigenspace picture\n');
