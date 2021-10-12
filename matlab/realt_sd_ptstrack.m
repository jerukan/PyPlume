clear all;
t0 = clock; 
%read file
% pathx = '/data/hfradar/hfrnet/trj/tjr/data/pdata/';
pathx = 'support_data/TrackerOutput/pdata/';

load glist90zj.mat;
% variables loaded:
%   glist

fname = dir(pathx);
% if length(fname) <= 3, exit; end
for j = 1: 2
	fn_ = fname(j+2).name 
	fnt_(j) = datenum([str2num(fn_(10:13)) str2num(fn_(14:15)) str2num(fn_(16:17)) str2num(fn_(19:20)) 0 0]);
end

for k = 1: 2
    d = load([pathx '/' fname(k+2).name]);  
    
    t_ = datenum([d.year1 1 0 0 0 0]);
    t(k) = t_ + d.time + 1/24/2;
    ux = full(d.U)';
    vx = full(d.V)';
    ux(ux == 0) = NaN;
    vx(vx == 0) = NaN;
    u(:,k) = ux(glist);
    v(:,k) = vx(glist);
end
fprintf('Data sorting is done .... %3.4f (s)\n', etime(clock, t0));
% save /data/hfradar/hfrnet/trj/tjr/data/sort/glist90zj_uv.mat u v t glist -V6
save glist90zj_uv.mat u v t glist -V6

%fillgaps
%clear all

load uv2yr90zj_covdd.mat;
% variables loaded:
%   covdd
%   glist

covdg = covdd; 

load uv2yr90zj_covdd_eig40.mat; 
% variables loaded:
%   covdd

% load uv2yr90zj_covdd_eig.mat;
% Q = 0.4*trace(covdd)/length(covdd);
% egvl2 = egvl + Q;
% covdd = egvt*diag(egvl2)*egvt';
% save /net/rain/syongkim/DATA/codar/z.msrd/real.time/tjriver/info/uv2yr90zj_covdd_eig40.mat covdd

load uv2yr90zj_mean.mat;
% variables loaded:
%   glist
%   u
%   v
% note that surface velocities are measured in cm/s

wm = [u v]';
clear u v;

% load /data/hfradar/hfrnet/trj/tjr/data/sort/glist90zj_uv.mat;
load glist90zj_uv.mat;
% variables loaded:
%   glist
%   t
%   u
%   v

w = [u; v]; 
[ns, nt] = size(u);
clear u v
for j = 1: 2
    k = find(isnan(w(:,j)) == 1);
    covdd_ = covdd;
    w_ = w(:,j);
    covdg_ = covdg;
    w_ = w_- wm;
    covdd_(k,:) = [];
    covdd_(:,k) = [];
    covdg_(k,:) = [];
    w_(k) = [];
    nw(:,j) = covdg_'*((covdd_)\w_) + wm;
end
u = nw(1:ns,:);
v = nw(ns+1:end,:);
% save /data/hfradar/hfrnet/trj/tjr/data/sort/glist90zj_uv.mat u v t glist -V6
save glist90zj_uv.mat u v t glist -V6
fprintf('Data filling is done .... %3.4f (s)\n', etime(clock, t0));
%clear all; 
xi = [-117.1369]; yi = [32.5556]; % TJ mouth
% load /data/hfradar/hfrnet/trj/tjr/data/sort/glist90zj_uv.mat;
load glist90zj_uv.mat;
% variables loaded:
%   glist
%   t
%   u
%   v
load coastline.mat;
% variables loaded:
%   latz0
%   lonz0
load baselistz.mat;
% variables loaded:
%   alist
%   bdeg
%   blist
%   clist
load codartotalGrid.mat;
% variables loaded:
%   totalGrid

gax = totalGrid(:,1);  % longitudes
gay = totalGrid(:,2);  % latitudes
clear totalGrid
tx = t;
nlife = 24*3;  % gets overrided in the next .mat load ???
npt = 100;  % gets overrided in the next .mat load ???
uerr = 5;
amp = 20;

% load /data/hfradar/hfrnet/trj/tjr/data/sort/glist90zj_pts_position.mat
load glist90zj_pts_position.mat
% variables loaded:
%   it
%   nlife
%   npt
%   t (override)
%   xf
%   xi
%   yf
%   yi

npos = length(xi);
cv = 1e-5*3600;  % what's this
x_ = xf;
y_ = yf;
u_b = [];
v_b = [];  % these aren't even used

xb_ = gax(glist(alist));
yb_ = gay(glist(alist)); % next to boundary
u_nb = u(alist, 1);
v_nb = v(alist, 1);
    
umgb = abs(u_nb + 1i * v_nb);
uagb = atan2(v_nb, u_nb);  % angles of specific current vectors
dxb = u_nb * cv * 10;
dyb = v_nb * cv * 10;
[x_nb, y_nb] = km2lonlat(xb_, yb_, dxb, dyb);

xpb = x_nb; ypb = y_nb; jj = 1; p = [];
for j = 1: length(alist)
    [cx_, cy_] = polyxpoly(lonz0, latz0, [xb_(j) x_nb(j)], [yb_(j) y_nb(j)], 'unique');
    if(~isempty(cx_))
        p(jj) = j;
        jj = jj + 1;
    end
end


if ~isempty(p)
    u_n1 = amp * umgb(p) .* cos(uagb(p) + 2 / 180 * pi);
    v_n1 = amp * umgb(p) .* sin(uagb(p) + 2 / 180 * pi);
    u_n2 = amp * umgb(p) .* cos(uagb(p) - 2 / 180 * pi);
    v_n2 = amp * umgb(p) .* sin(uagb(p) - 2 / 180 * pi);
    dx_n1 = u_n1 * cv;
    dy_n1 = v_n1 * cv;
    dx_n2 = u_n2 * cv;
    dy_n2 = v_n2 * cv;
    [x_n1, y_n1] = km2lonlat(xb_(p), yb_(p), dx_n1, dy_n1);
    [x_n2, y_n2] = km2lonlat(xb_(p), yb_(p), dx_n2, dy_n2);
    
    puag = [];
    for k = 1: length(p)
        [cxn1_, cyn1_] = polyxpoly(lonz0, latz0, [xb_(p(k)) x_n1(k)], [yb_(p(k)) y_n1(k)], 'unique');
        [cxn2_, cyn2_] = polyxpoly(lonz0, latz0, [xb_(p(k)) x_n2(k)], [yb_(p(k)) y_n2(k)], 'unique');
        
        if length(cxn1_) > 1
            dis = (xb_(p(k)) - cxn1_).^2 + (yb_((k)) - cyn1_).^2;
            [dis_i, dis_o] = sort(dis);
            cxn1_ = cxn1_(dis_o(1));
            cyn1_ = cyn1_(dis_o(1));
        end
        
        if length(cxn2_) > 1
            dis = (xb_(p(k)) - cxn2_).^2 + (yb_((k)) - cyn2_).^2;
            [dis_i, dis_o] = sort(dis);
            cxn2_ = cxn2_(dis_o(1));
            cyn2_ = cyn2_(dis_o(1));
        end
        
        if isempty(cxn1_) || isempty(cxn2_)
            puag(k) = pi/2;
        else
            puag(k) = atan2(cyn2_ - cyn1_, cxn2_ - cxn1_);            
        end
    end
    
    u_np = umgb(p) .* cos(uagb(p) - puag') .* cos(puag'); % cos(velocity slope - coastline slope)*cos(coastline slope)
    v_np = umgb(p) .* cos(uagb(p) - puag') .* sin(puag');     
   
    u_nb(p) = u_np;
    v_nb(p) = v_np;
end % ~isempty(p)

u_b = u_nb;
v_b = v_nb;
u(alist, 2) = u_nb;
v(alist, 2) = v_nb;



for ipos = 1: npos
    xi_ = xi(ipos);
    yi_ = yi(ipos);
    x_ = [x_; xi_*ones(npt,1)];             
    y_ = [y_; yi_*ones(npt,1)]; 
end % ipos

if it > nlife
    x_(1:npt) = [];
    y_(1:npt) = [];
end
   
u_ = griddata([gax(glist); gax(blist)], [gay(glist); gay(blist)], [u(:,2); u_b], x_, y_);
v_ = griddata([gax(glist); gax(blist)], [gay(glist); gay(blist)], [v(:,2); v_b], x_, y_);

ntotalpt = length(x_);

rand('state', sum(100 * clock));
th = 2 * pi * rand(ntotalpt, 1);
% uerr = 5 cm/s
u_n = u_ + uerr * cos(th);
v_n = v_ + uerr * sin(th);

umg = abs(u_n + 1i * v_n);
uag = atan2(v_n, u_n);
% cv = 1e-5*3600;
% converts centimeters to km, multiply by hour
dx = u_n * cv;
dy = v_n * cv;
[x_n, y_n] = km2lonlat(x_, y_, dx, dy);

xp = x_n; yp = y_n; jj = 1; p = [];
for j = 1: ntotalpt
    [cx_, cy_] = polyxpoly(lonz0, latz0, [x_(j) x_n(j)], [y_(j) y_n(j)], 'unique');
    if(~isempty(cx_))
        p(jj) = j; jj = jj +1;
    end
end


if ~isempty(p)
    u_n1 = amp * umg(p) .* cos(uag(p) + 2 / 180 * pi);
    v_n1 = amp * umg(p) .* sin(uag(p) + 2 / 180 * pi);
    u_n2 = amp * umg(p) .* cos(uag(p) - 2 / 180 * pi);
    v_n2 = amp * umg(p) .* sin(uag(p) - 2 / 180 * pi);
    dx_n1 = u_n1 * cv;
    dy_n1 = v_n1 * cv;
    dx_n2 = u_n2 * cv;
    dy_n2 = v_n2 * cv;
    [x_n1, y_n1] = km2lonlat(x_(p), y_(p), dx_n1, dy_n1);
    [x_n2, y_n2] = km2lonlat(x_(p), y_(p), dx_n2, dy_n2);
    
    puag = [];
    for k = 1: length(p)
        [cxn1_, cyn1_] = polyxpoly(lonz0, latz0, [x_(p(k)) x_n1(k)], [y_(p(k)) y_n1(k)], 'unique');
        [cxn2_, cyn2_] = polyxpoly(lonz0, latz0, [x_(p(k)) x_n2(k)], [y_(p(k)) y_n2(k)], 'unique');
        
        if length(cxn1_) > 1
            dis = (x_(p(k)) - cxn1_).^2 + (y_((k)) - cyn1_).^2;
            [dis_i, dis_o] = sort(dis);
            cxn1_ = cxn1_(dis_o(1));
            cyn1_ = cyn1_(dis_o(1));
        end
        
        if length(cxn2_) > 1
            dis = (x_(p(k)) - cxn2_).^2 + (y_((k)) - cyn2_).^2;
            [dis_i, dis_o] = sort(dis);
            cxn2_ = cxn2_(dis_o(1));
            cyn2_ = cyn2_(dis_o(1));
        end
        
        if isempty(cxn1_) || isempty(cxn2_)
            puag = pi/2;
        else
            puag = atan2(cyn2_ - cyn1_, cxn2_ - cxn1_);            
        end
        
        u_np = umg(p(k)) .* cos(uag(p(k)) - puag') .* cos(puag);
        v_np = umg(p(k)) .* cos(uag(p(k)) - puag') .* sin(puag);     

        dx_np = u_np * cv;
        dy_np = v_np * cv;
        [x_np, y_np] = km2lonlat(x_(p(k)), y_(p(k)), dx_np, dy_np);
        

        
        if ~inpolygon(x_np, y_np, lonz0, latz0)
            u_np = -0.2;
            v_np = 0;
            dx_np = u_np * cv;
            dy_np = v_np * cv;
            [x_np, y_np] = km2lonlat(x_(p(k)), y_(p(k)), dx_np, dy_np);
        end
        
        xp(p(k)) = x_np;
        yp(p(k)) = y_np;
    end
    
    
end % ~isempty(p)
x_ = xp;
y_ = yp;
it = it +1;
xf = x_;
yf = y_;
t = tx(2);
[YYYY, MM, DD, hh, mm] = datestr0(t);
% save /data/hfradar/hfrnet/trj/tjr/data/sort/glist90zj_pts_position.mat xf yf nlife it xi yi npt t -V6
save glist90zj_pts_position.mat xf yf nlife it xi yi npt t -V6
pause(1);
% save(['/data/hfradar/hfrnet/trj/tjr/data/sort/pts_' YYYY MM DD '_' hh mm '.mat'], 'xf', 'yf', 'nlife', 'it', 'xi', 'yi', 'npt', 't', '-V6');
save(['support_data/TrackerOutput/sort/pts_' YYYY MM DD '_' hh mm '.mat'], 'xf', 'yf', 'nlife', 'it', 'xi', 'yi', 'npt', 't', '-V6');
fprintf('Random Walk model is done with %d particles .... %3.4f (s)\n', ntotalpt, etime(clock, t0));
%clear all; close all
% load /data/hfradar/hfrnet/trj/tjr/data/sort/glist90zj_pts_position.mat
load glist90zj_pts_position.mat
clr  = jet(nlife);

% particle simulation random walk model end
%----------------------------------------%
% some random graphing shit below related to how close particles are to
% stations

load coastline_1km.mat;
load wq_stposition.mat
spos_x = xwq;
spos_y = ywq;
dpos = [spos_x spos_y ones(length(spos_x), 1)] * coefs;
dpos = dpos(:,1);
ii = inpolygon(xf, yf, rlonz0, rlatz0);
ii = find(ii == 1);
dr = [xf(ii) yf(ii) ones(length(ii), 1)] * coefs;
d_ = dr(:,1);

cx = 0:0.002:max(coastmap); %200m bin
[ncount, nxlabel] = hist(d_, cx);
ncf =  ncount/length(xf)*100;
if max(ncf) < 10, ylim_ = 10;
elseif 10 <= max(ncf) && max(ncf) < 25, ylim_ = 25;
elseif  25 <= max(ncf) &&  max(ncf) < 50, ylim_ = 50; 
elseif 50 <= max(ncf) && max(ncf) <= 75; ylim_ = 75;
elseif 75 <= max(ncf), ylim_ = 100;
end
dpos = dpos(1:end-1); % exclude Mexico station
for j = 1: length(dpos)
    sti(j) = find(min(abs(dpos(j)-cx)) == abs(dpos(j)-cx));
end
ii = find(ncount(sti) > 0);

figure
set( gcf, 'renderer', 'zbuffer' )
subplot(5,1,[1:4]);
SDmap3dy; axis equal; hold on
if it <= nlife, n = it; else, n  = nlife; end
plot(xi, yi, 'k+')   
for j = 1: n
    plot(xf(1+npt*(j-1):npt*j), yf(1+npt*(j-1):npt*j), '.', 'color', clr(n-j+1,:))
end
plot(xi, yi, 'k+');
plot(spos_x, spos_y, 'y.', 'markersize', 20)
plot(spos_x, spos_y, 'k.', 'markersize', 14);
plot(spos_x(ii), spos_y(ii), 'r.', 'markersize', 14);
xlim([-117.277 -117.092]); ylim([32.53 32.7]); 
set(gca, 'xtick', [-117.3:0.03:-117], 'xticklabel', [117.3:-0.03:117], 'ytick', [32.5:0.03:32.7]);
%mapax(2,0,2,0); 
grid on
%xlabel('Longitude (W)'); ylabel('Latitude (N)')

% color bar shit breaks for no reason
% cb = scaledcolorbar('vert');
% set(cb, 'xtick', [], 'ytick', [1:63/(nlife/12):64], 'yticklabel', [0:0.5:(nlife/24)]);

%[YYYY, MM, DD, hh, mm] = datestr0(t-8/24);    %GMT-> PST
%title(['TJRIVER-TRAJ : ' YYYY ' ' MM ' ' DD ' ' hh ':' mm ' (PST)'])

[YYYY, MM, DD, hh, mm] = datestr0(t-7/24);    %GMT-> PDT
title(['TJRIVER-TRAJ : ' YYYY ' ' MM ' ' DD ' ' hh ':' mm ' (PDT)'])
text(-117.0456, 32.7066, '(days)')


subplot(5,1,5)
pf = find(cx-dpos(9) < 0.04 & cx-dpos(9) > -0.16); %plotting hack - MO

hb = bar(cx(pf)-dpos(9), ncf(pf), 'y'); 
%hb = bar(cx-dpos(9), ncf, 'y'); 
set(hb, 'edgecolor', [0 0 0]); 
hold on; plot(dpos-dpos(9), zeros(length(dpos),1), 'y.', 'markersize', 18)
plot(dpos-dpos(9), zeros(length(dpos),1), 'k.', 'markersize', 12)
plot(dpos(ii)-dpos(9), zeros(length(ii)), 'r.', 'markersize', 12);
set(gca, 'xtick', [-0.16:0.02:0.04], 'xticklabel', [16:-2:2 0:2:4])
xlim([-0.16 0.04]); 
ylim([0 ylim_]);
ylabel('%'); %set(gca, 'xdir', 'reverse');
xlabel('(North)------------- distance from TJ river mouth (km) ---------------- (South)')

[YYYY, MM, DD, hh, mm] = datestr0(t);    %save with GMT

%concetration profile
% load /data/hfradar/hfrnet/trj/tjr/data/sort/glist90zj_pts_conprofile.mat
load glist90zj_pts_conprofile.mat
con = [con; ncount]; tc = [tc; t]; 
if length(tc) > 24*5
    tc(1) = [];
    con(1,:) = [];
    con( con == 0) = NaN;
end
% save /data/hfradar/hfrnet/trj/tjr/data/sort/glist90zj_pts_conprofile.mat con tc cx dpos -V6
save glist90zj_pts_conprofile.mat con tc cx dpos -V6
pause(1);
% save(['/data/hfradar/hfrnet/trj/tjr/data/sort/tjrprf_' YYYY MM DD '_' hh '30.mat'], 'tc', 'con', 'cx', 'dpos', '-V6');
save(['support_data/TrackerOutput/sort/tjrprf_' YYYY MM DD '_' hh '30.mat'], 'tc', 'con', 'cx', 'dpos', '-V6');

path_ =  'support_data/TrackerOutput/pics/';
delete([pathx '/' fname(3).name])
pngname = [path_ 'tjrpts_' YYYY MM DD '_' hh mm '.png'];

print(gcf, '-dpng', pngname, '-zbuffer');
