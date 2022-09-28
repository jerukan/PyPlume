chdir('/home/rt/work/codar/real.time/rmw/sd/');
%conprofile plot
clear all;
%addpath(genpath('/net/DATA0/scripts/'));
%addpath(genpath('/net/DATA0/work/codar/archive/rmw/sd/info/'));

load /data/hfradar/hfrnet/trj/tjr/data/sort/glist90zj_pts_conprofile.mat
tc_ = tc-1/24/2-7/24; %GMT->PDT with pcolor shift
dcx = diff(cx(1:2))/2;
xl1 = tc_(1)-1/24; xl2 = tc_(end)+1/24;
figure
set( gcf, 'renderer', 'zbuffer' )
pcolor(tc_, cx-dpos(9)-dcx, con'); shading flat; hold on
datetick('x', 6)
plot(xl1*ones(length(dpos),1), dpos-dpos(9), 'y.', 'markersize', 18)
plot(xl1*ones(length(dpos),1), dpos-dpos(9), 'k.', 'markersize', 12)
plot(xl2*ones(length(dpos),1), dpos-dpos(9), 'y.', 'markersize', 18)
plot(xl2*ones(length(dpos),1), dpos-dpos(9), 'k.', 'markersize', 12)
for j = 1: length(dpos)
    plot([xl1 xl2], [dpos(j)-dpos(9) dpos(j)-dpos(9)], 'k:', 'linewidth', 0.5);
end
set(gca, 'ytick', [-0.16:0.02:0.06], 'yticklabel', [16:-2:2 0:2:6], 'ydir', 'reverse')

xlim([xl1 xl2]); ylim([-0.16 0.04]);
gray_ = flipud(gray(11));
caxis([0 500]); colormap(gray_(2:end,:));  scaledcolorbar('horz');
xlabel(' time (PST)'); ylabel('(South) ---- distance from TJ river (km) ---- (North)')

%[YYYY, MM, DD, hh, mm] = datestr0(tc(end)-8/24);    %GMT-> PST
%title(['TJRIVER-CONCT. PROF. : ' YYYY ' ' MM ' ' DD ' ' hh ':' mm ' (PST)'])

[YYYY, MM, DD, hh, mm] = datestr0(tc(end)-7/24);    %GMT-> PDT
title(['TJRIVER-CONCT. PROF. : ' YYYY ' ' MM ' ' DD ' ' hh ':' mm ' (PDT)'])


[YYYY, MM, DD, hh, mm] = datestr0(tc(end));    %save with GMT

path_ =  '/data/hfradar/hfrnet/trj/tjr/data/pics/';
%jpgname = [path_ 'conprf_' YYYY MM DD '_' hh mm '.jpg'];
%print(gcf, '-djpeg', jpgname);

pngname = [path_ 'tjrprf_' YYYY MM DD '_' hh mm '.png'];
print(gcf, '-dpng', pngname);
exit
