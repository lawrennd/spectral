function [labels, PcEig, Centres] = SpectralCluster_octave(x, sigma)

% SPECTRALCLUSTER_OCTAVE Octave-compatible version using eigs instead of ppca

% initialisations
npts=size(x,1);
ExtraCluster=0;
Dim=2;
lambda = 0.2;

% compute the similarity matrix A and the matrix L
A=zeros(npts,npts);
for i=1:npts
    for j=1:npts
        A(i,j) = exp(-norm(x(i,:)-x(j,:))^2/sigma);
    end
end
D = (sum(A,2));
L = inv(diag(sqrt(D)))*(A/diag(sqrt(D)));

% find the eigenvectors associated with the 10 largest eigenvalues of L
% Use eigs instead of ppca for Octave compatibility
[Y, eigvals] = eigs(L, 10, 'lm');
[eigvals_sorted, eigvalIndices] = sort(diag(eigvals), 1, 'descend');
Y = Y(:,eigvalIndices);
PcEig = Y(:,(1:Dim));

% the first Centre is initialised
norms = diag(PcEig*PcEig');
[void, index(1)] = max(norms);
Centres = PcEig(index(1), :);

% the second Centre is initialised
S = ((PcEig*Centres(1, :)').^2)./(norms);
[void, index(2)] = min(S);
Centres = [PcEig(index, :)];

while  ExtraCluster==0 & Dim <10

    % introduce a new centre at the origin and do k-means
    Centres = [Centres; zeros(1,Dim)];
    [Centres, options, labels] = mahKmeans_octave(Centres, PcEig, lambda);
    for i = 1:Dim+1
        for j = 1:npts
            CentrDist(i,j) = norm(PcEig(j,:)-Centres(i,:));
        end
    end
    % if anything is assigned to this new centre, there is an extra cluster
    if max(labels(:,Dim+1)) == 1
        ExtraCluster = 0;
        Dim = Dim+1;
        % take the next eigenvector from Y
        PcEig = Y(:,(1:Dim));
        % re-initialise the centres
        Centres=zeros(Dim,Dim);
        for i=1:Dim
            [void,point] = min(CentrDist(i,:));
            Centres(i,:)=PcEig(point,:);
        end
    else
        ExtraCluster = 1;
    end
end

% erase the last column of labels, as it is empty
labels = labels(:,1:Dim);
