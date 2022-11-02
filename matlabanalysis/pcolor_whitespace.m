function pcol = pcolor_whitespace(x,y,z,spacing,varargin)
% This is like pcolor, but puts white space in between the profiles so you
% don't have issues with huge gaps being shaded by one profile. 

% -------------------------------------------------------------------------
[xx,yy]=size(z);
x=x(:);
y=y(:);

if nargin>3
    approxd=spacing;
else
    approxd=min(abs(diff(x)));
end

% Get distances of profiles and create a distance file for plotting
dd=diff(x);
bad=find(abs(dd)>approxd);
dd(bad)=approxd;
x1=x(2:end)-0.499*dd;
x2=x(1:end-1)+0.499*dd;
x1=[x(1)-0.499*dd(1);x1];
x2=[x2;x(end)+0.499*dd(end)];
xt=[x1';x2'];
xt=xt(:);

maxx=xx;

z2=NaN(xx,yy*2);
z2(:,1:2:end)=z;
z2(:,2:2:end)=z;

bad=2*bad;
if ~isempty(bad)
    for i=1:length(bad)
        xt=[xt(1:bad(i));xt(bad(i))+0.001;xt(bad(i)+1)-0.001;xt(bad(i)+1:end)];
        z2=[z2(:,1:bad(i)) NaN(maxx,1) NaN(maxx,1) z2(:,bad(i)+1:end)];
        bad=bad+2;
    end
end

pcol = pcolor(xt,y,z2); shading flat;
set(gca, 'Layer', 'top')

% -------------------------------------------------------------------------
for i=1:2:numel(varargin)
    try
        set(pcol, varargin{i}, varargin{i+1});
    catch
        error([varargin{i} ' is not a valid property for pcolor function.']);
    end
end


