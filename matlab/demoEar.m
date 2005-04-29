% DEMOEAR a demonstration of spectral clustering on the ear data set.

% SPECTRAL

clear all;
close all;
clc

filename = 'ear.bmp';
sigma2 = 1;

% filename = 'swirls.bmp';
% sigma2 = 1;

image = imread(filename);
image = image(:,:,1);
[x1, x2] = find(image);
x = [x1 x2];

% plot all the data points together
figure;
plot(x(:,1), x(:,2), 'kx');
xlim([0 100]);
ylim([0 100]);
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

% plot the clusters in 3d eigenspace
figure
plot3(Centres(:,1),Centres(:,2),Centres(:,3),'kd');
hold on
for i = 1:npts
    plot3(PcEig(i,1),PcEig(i,2),PcEig(i,3), char(symbol(find(labels(i,:)), :)));
end
fprintf('This is the eigenspace picture\n');
