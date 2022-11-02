function m = msort2(d);
%this is the same of uniquenum
[ns, nt] = size(d);
if nt > 1, d = d'; end

d = sort(d);
d_ = diff(d);
m = [d(1); d(find(d_ ~= 0) + 1)];

% this is very efficient comparing msort