chdir('/home/rt/work/codar/real.time/rmw/sd/');
clear all; close all
%addpath(genpath('/net/DATA0/scripts/'));
%addpath(genpath('/net/DATA0/work/codar/archive/rmw/sd/info/'));

load /data/hfradar/hfrnet/trj/tjr/data/sort/glist90zj_pts_position.mat
clr  = jet(nlife);

load coastline_1km.mat;  load wq_stposition.mat
spos_x = xwq; spos_y = ywq;
dpos = [spos_x spos_y ones(length(spos_x),1)]*coefs;
dpos = dpos(:,1);
ii = inpolygon(xf, yf, rlonz0, rlatz0); ii = find(ii == 1);
dr = [xf(ii) yf(ii) ones(length(ii), 1)]*coefs;
d_ = dr(:,1);

cx = 0:0.002:max(coastmap); %200m bin
[ncount, nxlabel] = hist(d_, cx);
ncf =  ncount/length(xf)*100;
if max(ncf) < 10, ylim_ = 10;
elseif 10 <= max(ncf) & max(ncf) < 25, ylim_ = 25;
elseif  25 <= max(ncf) &  max(ncf) < 50, ylim_ = 50; 
elseif 50 <= max(ncf) & max(ncf) <= 75; ylim_ = 75;
else, 75 <= max(ncf), ylim_ = 100; end
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
set(gca, 'xtick', [-117.3:0.01:-117], 'xticklabel', [117.3:-0.01:117], 'ytick', [32.5:0.01:32.7]);
xlim([-117.162 -117.104]); ylim([32.5303 32.5836]);
%mapax(2,0,2,0); 
grid on
xlabel('Longitude (W)'); ylabel('Latitude (N)')
cb = scaledcolorbar('vert');
set(cb, 'xtick', [], 'ytick', [1:63/(nlife/12):64], 'yticklabel', [0:0.5:(nlife/24)]);
%[YYYY, MM, DD, hh, mm] = datestr0(t-8/24);    %GMT-> PST
%title(['TJRIVER-TRAJ : ' YYYY ' ' MM ' ' DD ' ' hh ':' mm ' (PST)'])

[YYYY, MM, DD, hh, mm] = datestr0(t-7/24);    %GMT-> PDT
title(['TJRIVER-TRAJ : ' YYYY ' ' MM ' ' DD ' ' hh ':' mm ' (PDT)'])

text(-117.0893, 32.5857, '(days)')

[YYYY, MM, DD, hh, mm] = datestr0(t);    %save with GMT
path_ =  '/data/hfradar/hfrnet/trj/tjr/data/pics/';

fname = [path_ 'tjrpti_' YYYY MM DD '_' hh mm '.txt'];
fid = fopen(fname, 'w');
fprintf(fid, 'station\t # of particles\tcontamination\n');
for j = 1: length(sti)
    if sum(j == ii), contxt = 'YES';
    else, contxt = 'NO'; end
    fprintf(fid, '%d\t%d\t%s\n', j, ncount(sti(j)), contxt);
end
fclose(fid);

% plot total map

%jpgname = [path_ 'tjrptz_' YYYY MM DD '_' hh mm '.jpg'];
%print(gcf, '-djpeg', jpgname);

pngname = [path_ 'tjrptz_' YYYY MM DD '_' hh mm '.png'];
print(gcf, '-dpng', pngname);
exit
