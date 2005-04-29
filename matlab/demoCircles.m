% DEMOCIRCLES A demonstration of spectral clustering on the three circles data set.

% SPECTRAL 

close all
clear all
clc

sigma2 = 0.05;

% generate and display the three circles data set
randn('seed',1);
rand('seed',1);
npts=100;
step = 2*pi/npts;
theta = [step:step:2*pi];
radius = randn(1,npts);
r1 = [ones(1,npts)+0.1*radius];
r2=[2*ones(1,npts)+0.1*radius];
r3=[3*ones(1,npts)+0.1*radius];
x = [(r1.*cos(theta))', (r1.*sin(theta))';...
    (r2.*cos(theta))', (r2.*sin(theta))';...
    (r3.*cos(theta))', (r3.*sin(theta))'];
figure, plot(x(:,1),x(:,2),'r+')
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
hold on
for i = 1:npts
    plot(PcEig(i,1),PcEig(i,2), char(symbol(find(labels(i,:)), :)));
end
plot(Centres(:,1),Centres(:,2),'kd')
fprintf('This is the eigenspace picture\n');
