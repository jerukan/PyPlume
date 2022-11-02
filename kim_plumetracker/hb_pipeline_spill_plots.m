function [fh,a1h,hc,a2h] = ocsd_plots( t, xc, yc, npts, nlife, xf, yf)

% Needs help documentation - MO

%% Load external data
% 1km offshore boundary and along-coast distance
load coastline_1km.mat % -->rlonz0, rlatz0, coastmap, coefs
% water quality stations
% load wqstations.mat % --> s.x, s.y, s.name
% coastline
load coastOR2Mex % --> OR2Mex

%% Compute along-coast plume distribution and affected water quality stations
% % Convert water quality (lon, lat) to along coast distance
% dpos = [s.x' s.y' ones(length(s.x),1)]*coefs(:,1);
% 
% % Find plume particles within the 1km boundary and convert from (lon, lat)
% % to along coast distance
% ii = inpolygon(xf{end}, yf{end}, rlonz0, rlatz0);
% ii = find(ii == 1);
% d_ = [xf{end}(ii) yf{end}(ii) ones(length(ii), 1)]*coefs(:,1);
% 
% % Bin particle count within 1km boundary into along-coast distance
% cx = 0:0.0025:max(coastmap); %500m bin(0.005), 250m bin(0.0025)
% [ncount, nxlabel] = hist(d_, cx);
% ncf =  ncount/length(xf{end})*100; %convert count to %age of total particles
% 
% % Define y-axis limits based on results
% if max(ncf) < 10, ylim_ = 10;
% elseif 10 <= max(ncf) && max(ncf) < 25, ylim_ = 25;
% elseif  25 <= max(ncf) &&  max(ncf) < 50, ylim_ = 50;
% elseif 50 <= max(ncf) && max(ncf) <= 75; ylim_ = 75;
% elseif 75 <= max(ncf), ylim_ = 100;
% end
% 
% % Find water quality stations where plume was found to be present within
% % 1km of the shore
% sti = nan( 1, numel(dpos) );
% for j = 1: numel(dpos)
%     [~, sti(j)] = min( abs( dpos(j)-cx ) );
% end
% ii = find( ncount(sti) > 0 );

%% Plot map of particle trajectories colored by age
% Define age colormap
%clr = flipud( jet(nlife+1) );

% Define figure
figure
set(gcf, 'renderer', 'painters')
%subplot(5, 1, 1:4)

% Coastline
plot(OR2Mex(:,1), OR2Mex(:,2), 'k', 'linewidth',  0.5)
daspect([1.2 1 1])
hold on;

% Observations
if t == datenum(2021,10,2,2,0,0)
    ob = load('/Users/motero/Desktop/tmp/spill/data/SENTINEL1A_10_2_2021_0158z.mat');
    if iscell(ob.b_lat)
        for I = 1:numel(ob.b_lat)
            fill(ob.b_lon{I},ob.b_lat{I},[0.9 0.9 0.9])
        end
    else
        fill(ob.b_lon,ob.b_lat,[0.9 0.9 0.9])
    end
elseif  t == datenum(2021,10,3,2,0,0)
    ob = load('/Users/motero/Desktop/tmp/spill/data/SENTINEL1B_10_3_2021_0149z.mat');
    if iscell(ob.b_lat)
        for I = 1:numel(ob.b_lat)
            fill(ob.b_lon{I},ob.b_lat{I},[0.9 0.9 0.9])
        end
    else
        fill(ob.b_lon,ob.b_lat,[0.9 0.9 0.9])
    end
elseif  t == datenum(2021,10,5,18,0,0)
    ob = load('/Users/motero/Desktop/tmp/spill/data/SENTINEL2B_10_5_2021_1831z.mat');
    if iscell(ob.b_lat)
        for I = 1:numel(ob.b_lat)
            fill(ob.b_lon{I},ob.b_lat{I},[0.9 0.9 0.9])
        end
    else
        fill(ob.b_lon,ob.b_lat,[0.9 0.9 0.9])
    end
    ob = load('/Users/motero/Desktop/tmp/spill/data/ICEYE_10_5_2021_1737z.mat');
    if iscell(ob.b_lat)
        for I = 1:numel(ob.b_lat)
            fill(ob.b_lon{I},ob.b_lat{I},[0.9 0.9 0.9])
        end
    else
        fill(ob.b_lon,ob.b_lat,[0.9 0.9 0.9])
    end
elseif  t == datenum(2021,10,6,14,0,0)
    ob = load('/Users/motero/Desktop/tmp/spill/data/RADARSAT2_10_6_2021_1342z.mat');
    if iscell(ob.b_lat)
        for I = 1:numel(ob.b_lat)
            fill(ob.b_lon{I},ob.b_lat{I},[0.9 0.9 0.9])
        end
    else
        fill(ob.b_lon,ob.b_lat,[0.9 0.9 0.9])
    end
elseif  t == datenum(2021,10,7,15,0,0)
    ob = load('/Users/motero/Desktop/tmp/spill/data/CAPELLA_10_7_2021_1515z.mat');
    if iscell(ob.b_lat)
        for I = 1:numel(ob.b_lat)
            fill(ob.b_lon{I},ob.b_lat{I},[0.9 0.9 0.9])
        end
    else
        fill(ob.b_lon,ob.b_lat,[0.9 0.9 0.9])
    end
elseif  t == datenum(2021,10,9,2,0,0)
    ob = load('/Users/motero/Desktop/tmp/spill/data/SENTINEL1A_10_9_2021_0150z.mat');
    if iscell(ob.b_lat)
        for I = 1:numel(ob.b_lat)
            fill(ob.b_lon{I},ob.b_lat{I},[0.9 0.9 0.9])
        end
    else
        fill(ob.b_lon,ob.b_lat,[0.9 0.9 0.9])
    end
elseif  t == datenum(2021,10,9,14,0,0)
    ob = load('/Users/motero/Desktop/tmp/spill/data/RADARSAT2_10_9_2021_1354z.mat');
    if iscell(ob.b_lat)
        for I = 1:numel(ob.b_lat)
            fill(ob.b_lon{I},ob.b_lat{I},[0.9 0.9 0.9])
        end
    else
        fill(ob.b_lon,ob.b_lat,[0.9 0.9 0.9])
    end
end 


% Particles colored by age
% if numel(xf) <= nlife
%     n = numel(xf);
% else
%     n  = nlife;
% end
% for j = 1: n
%     plot( xf{end}(1+npts*(j-1):npts*j), ...
%           yf{end}(1+npts*(j-1):npts*j), ...
%         '.', 'color', clr(n-j+1,:) )
% end
plot( xf{end}, yf{end}, 'r.')
%, 'color', clr(n-j+1,:) )
%end


% Source location
%plot(xc, yc, 'w+', 'markersize', 18, 'linewidth', 3);
%plot(xc, yc, 'k+', 'markersize', 12);

% Water quality stations colored by plume presence
% plot(s.x, s.y, 'y.', 'markersize', 18); 
% plot(s.x, s.y, 'k.', 'markersize', 12); 
% plot(s.x(ii), s.y(ii), 'r.', 'markersize', 12); 

% Axis limits
xlim([-118.3692 -117]);
ylim([32.4 33.8]);
a1h = gca;

% Colorbar
% colormap( clr )
% hc = colorbar('horiz');
% caxis( [0 nlife] )
% set(hc, 'xtick', [0: 12 : nlife], 'xticklabel', 0 : 0.5 : nlife/24 )
% set( get(hc, 'title'), 'string', 'Particle Age (days)' )

% Labeling
% mapax(4, 0, 2, 0)
set(gca, 'xaxislocation', 'top' );
grid
title({'HB Pipeline Leak', 'Surface Current Trajectory Estimate', ...
    datestr( t(end), 'yyyy/mm/dd HH:00 Z' ) })

%% Plot histogram of along-coast distribution
% subplot(5,1,5)
% 
% % Along-coast distribution
% hb = bar(cx-dpos(16), ncf, 'y');
% set(hb, 'edgecolor', [0 0 0]); 
% hold on;
% 
% % Axis limits
% ylim([0 ylim_]); 
% xlim([-0.12 0.12]);
% 
% % Water quality stations relative to the Santa Anna river mouth
% % colored by plume presence
% plot(dpos-dpos(16), zeros(length(dpos),1), 'y.', 'markersize', 18)
% plot(dpos-dpos(16), zeros(length(dpos),1), 'k.', 'markersize', 12)
% plot(dpos(ii)-dpos(16), zeros(length(ii)), 'r.', 'markersize', 12);
% 
% % Labeling
% grid
% title('Plume Distribution within 1km of Shoreline');
% ylabel({'Relative Plume Presence', '(% total plume particles)'});
% set(gca, 'xtick', [-0.12:0.02:0.12], 'xticklabel', [12:-2:2 0:2:12])
% xlabel('{\it\leftarrow North}        Along-coast Distance Relative to Santa Ana River (km)         {\itSouth \rightarrow}')
% a2h = gca;

%% Positioning
set(gcf,'PaperType','tabloid','PaperPositionMode','auto')
%set(hc, 'position',  [0.1300 0.3091 0.7750 0.0100] )
set(gcf, 'Position', [2863 195 762 833] )
%set(a1h, 'position', [0.1300 0.3142 0.7750 0.6423] )
%set(a2h, 'position', [0.1300 0.0751 0.7750 0.1126] )
fh = gcf;
