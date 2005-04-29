function n2 = mahDist2(x, c, lambda)
% MAHDIST2 Calculates Mahalanobis distance between two sets of points (based on Netlab dist2).

% SPECTRAL



% Additional Copyright (c) Ian T Nabney (1996-2001)

[ndata, dimx] = size(x);
[ncentres, dimc] = size(c);
if dimx ~= dimc
	error('Data dimension does not match dimension of centres')
end
% Compute Mahalanobis distances from centres (GS)
for j = 1:ncentres
    d = c(j,:)*c(j,:)';
    if d > 0.0001
        Mah = (eye(dimx)-((c(j,:)'*c(j,:))/(norm(c(j,:))^2)))/lambda ...
            + lambda*c(j,:)'*c(j,:)/(norm(c(j,:))^2);
    else
        Mah=eye(dimx);
    end
    for i = 1:ndata
        n2(i,j) = (x(i,:)-c(j,:))*Mah*(x(i,:)-c(j,:))';
    end
end
% Rounding errors occasionally cause negative entries in n2
if any(any(n2<0))
  n2(n2<0) = 0;
end