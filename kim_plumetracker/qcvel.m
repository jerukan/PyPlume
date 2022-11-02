function [t, x, y, u, v] = qcvel(t, x, y, u, v)

% Needs help documentation - MO

%   - Remove all-NaN velocity fields
%   - Returns most recent dataset that is temporally gap-free
%   - assumes hourly data

% Find time(s) where velocity field is all NaN
f = find( sum( isnan(u), 1 ) == size(u,1) | sum( isnan(v), 1 ) == size(v,1) );

% Remove empty velocity fields
if ~isempty(f)
    t(f) = [];
    u(:,f) = [];
    v(:,f) = [];
end

if isempty(t)
    return
end

% Look for temporal gaps
dt = diff(t);
f = find( dt.*24 > 1.01 );
if isempty(f)
    return
end

% Return the most recent time period that is gap free
t = t( f(end)+1 : end );
u = u( :, f(end)+1 : end );
v = v( :, f(end)+1 : end );
