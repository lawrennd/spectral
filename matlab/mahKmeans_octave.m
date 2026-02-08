function [centres, options, post, errlog] = mahKmeans_octave(centres, data, lambda, options)
% MAHKMEANS_OCTAVE Octave-compatible version with simplified options

[ndata, data_dim] = size(data);
[ncentres, dim] = size(centres);

if dim ~= data_dim
  error('Data dimension does not match dimension of centres')
end

if (ncentres > ndata)
  error('More centres than data')
end

% Simplified options handling for Octave
if nargin < 4
    options = zeros(1, 18);
    options(1) = 0;  % display
    options(2) = 1e-4;  % precision for centre updates
    options(3) = 1e-4;  % precision for error
    options(14) = 100;  % max iterations
end

niters = 100;
if length(options) >= 14 && options(14) > 0
    niters = options(14);
end

store = 0;
if (nargout > 3)
  store = 1;
  errlog = zeros(1, niters);
end

% Matrix to make unit vectors easy to construct
id = eye(ncentres);

% Main loop of algorithm
old_e = 0;
for n = 1:niters

  % Save old centres to check for termination
  old_centres = centres;
  
  % Calculate posteriors based on Mahalanobis distance
  d2 = mahDist2(data, centres, lambda);
  % Assign each point to nearest centre
  [minvals, index] = min(d2', [], 1);
  post = id(index,:);

  num_points = sum(post, 1);
  % Adjust the centres based on new posteriors
  for j = 1:ncentres
    if (num_points(j) > 0)
      centres(j,:) = sum(data(find(post(:,j)),:), 1)/num_points(j);
    end
  end

  % Error value is total squared distance from cluster centres
  e = sum(minvals);
  if store
    errlog(n) = e;
  end
  if options(1) > 0
    fprintf(1, 'Cycle %4d  Error %11.6f\n', n, e);
  end

  if n > 1
    % Test for termination
    if max(max(abs(centres - old_centres))) < options(2) && ...
        abs(old_e - e) < options(3)
      options(8) = e;
      return;
    end
  end
  old_e = e;
end

options(8) = e;
