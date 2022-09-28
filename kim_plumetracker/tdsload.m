function [x, y, t, u, v] = tdsload( url, s )

% TDSLOAD Load data for trajectory routines from TDS
%	Returns HF-Radar velocity data for trajectory
%	routines from the given TDS URL and subset
%	structure using nctoolbox.
%
%	nctoolbox must be setup prior to calling this
%	function.
%
%	See also ncgetdataset, geovariable, geosubset

% Get dataset and extract velocity data 
nc = ncgeodataset( url );
u = nc.geovariable('u');
v = nc.geovariable('v');
u = u.geosubset(s);
v = v.geosubset(s);

% Extract grid data and reshape all for trajectory routines
[x,y]=meshgrid(u.grid.lon, u.grid.lat);
x = x(:);
y = y(:);
t = u.grid.time';
u = shiftdim( u.data, 1);
u = reshape( u, size(u,1)*size(u,2), size(u,3) );
u = double(u).*100; %cm/s
v = shiftdim( v.data, 1);
v = reshape( v, size(v,1)*size(v,2), size(v,3) );
v = double(v).*100; %cm/s
