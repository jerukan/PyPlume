% Script to plot the particledata from HYCOM simulations specifcally.
% Requires some additional file dependencies (coastlines.mat is needed).
% If anyone knows where to find the coastline data, please tell me.

savepath = 'snapshots/';
saveopt = 'time';



%% LOAD TRAJECTORIES + HYCOM
fpath = 'particledata/';
fname = 'particle_mwb_trajectories.nc';




traj = load_any_nc([fpath fname]);
traj.timeunit = ncreadatt([fpath fname], 'time', 'units');
tmp = strsplit(traj.timeunit);
traj.tu = datenum(tmp{3}, 'yyyy-mm-ddTHH:MM:SS');


traj.t = traj.time;
traj.time =  traj.time(:,1);
traj.time = traj.time./(60*60*24) + traj.tu;
traj.t0 = traj.time(1);
traj.seeds = 1:size(traj.lat,2);

traj = rmfield(traj, 'z');

if contains(fpath, 'parcels-westcoast-master')
    traj.lon = traj.lon-360;
end

    
fpath = 'data/field_netcdfs/HYCOM_forecast/';
fname = 'hycom_mwbproj.nc';

hyc = load_any_nc([fpath fname]);
hyc.T = double(hyc.time)./24+hyc.t0;



%% calculate distance covered:

% initalpoints
initalpoints = [traj.lon(1,:); traj.lat(1,:)]';
% finalpoints
finalpoints = [traj.lon(end,:); traj.lat(end,:)]';

disttraveled = distance(initalpoints(:,2), initalpoints(:,1), finalpoints(:,2), finalpoints(:,1)); 
disttraveled = deg2km(disttraveled);
traj.disttraveled = disttraveled;

%% assessment plot
load coastlines.mat


% -------------------------------------------------------------------------
figure(997); clf; 
set(gcf, 'Position', [235   308   891   490])
set(gcf, 'Position', [237         253        1002         545]);

%%%HYCOM TI
ti = 1;
[~, ti] = min(abs(traj.t0-hyc.T));


% -------------------------------------------------------------------------
subplot(3,2,[1 3 5]); 

hold on;

y = hyc.lat; x = hyc.lon; z = sqrt(hyc.water_u(:,:,:,ti).^2+hyc.water_v(:,:,:,ti).^2);
z = squeeze(z);
x = x-360;
pcolor_whitespace(x,y,z',1); 
colormap(gca, gray); c = fixedcolorbar(gca); ylabel(c, '[m/s]'); caxis([0 1])

%%% TRAJECTORIES
plot(traj.lon, traj.lat, 'b-')

try
    cmap = buildcmap([1 0 0; 1 0.6 0.3; 1 0.9 0.3; 0 1 0; 0.23 0.5 0.19], length(traj.seeds)+5);
catch
    cmap = [0.23 0.5 0.19];
end


ccor = round((disttraveled./max(disttraveled)).*length(disttraveled));

for i=1:length(traj.seeds)
    ci = ccor(i); 
    if ~isnan(ci)
        plot(traj.lon(:,i), traj.lat(:,i), 'm-', 'Color', cmap(ci,:), 'LineWidth',1)
    end
end





grid on; box on; axis equal;

plot(coastlon, coastlat, 'k-', 'LineWidth',1); 

ylim([22 35]);
xlim([-85 -75]);
title({['trajectories']; [datestr(traj.time(1)) ' to ' datestr(traj.time(end))]})

% -------------------------------------------------------------------------
subplot(3,2,6); 
hold on;

grid on; box on; axis equal;

scatter(traj.lon(1,:), traj.lat(1,:), 70, disttraveled, 'filled');
cmap = buildcmap([1 0 0; 1 0.6 0.3; 1 0.9 0.3; 0 1 0; 0.23 0.5 0.19], length(traj.seeds));
colormap(gca, cmap); c = fixedcolorbar(gca); ylabel(c, 'Distance traveled [km]')

plot(coastlon, coastlat, 'k-'); 

ylim([22.8 25.5]);
xlim([-84.75 -79.25]);

title(['t0 = ' datestr(traj.time(1))])

% -------------------------------------------------------------------------
subplot(3,2,[2 4]); 

hold on;
grid on; box on; axis equal;

scatter(traj.lon(end,:), traj.lat(end,:), 70, disttraveled, 'filled');
cmap = buildcmap([1 0 0; 1 0.6 0.3; 1 0.9 0.3; 0 1 0; 0.23 0.5 0.19]);
colormap(gca, cmap); c = fixedcolorbar(gca); ylabel(c, 'Distance traveled [km]')

plot(coastlon, coastlat, 'k-'); 

ylim([22.8 32]);
xlim([-85 -77]);
title(['t_{end} = ' datestr(traj.time(end))])


% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------


