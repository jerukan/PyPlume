load coastlines.mat

%%

yf = [23.5:0.25:25];
xf = [-81.75:0.25:-80.25];
[xf yf] = meshgrid(xf,yf);
xf = xf+360;
save('/Users/alliho/Documents/SIO/parcels_projects/parcels-westcoast-master/spawn_points_mats/seeds_keywest.mat', 'xf','yf')

% xf = -81+360; yf = 24.25;
% save('/Users/alliho/Documents/SIO/parcels_projects/parcels-westcoast-master/spawn_points_mats/seeds.mat', 'xf','yf')

yf = [26.5:0.25:29];
xf = [-80.5:0.25:-79.25];
[xf yf] = meshgrid(xf,yf);
% xf = xf.*cos(deg2rad(rot));

rot = deg2rad(50); 
midx = nanmedian(xf(:))
midy = nanmedian(yf(:))

rotmat = [cos(rot) sin(rot); -cos(rot) sin(rot)];
dims = size(xf);
tmp = reshape([xf(:) yf(:)]*rotmat', [dims 2]);
xf = squeeze(tmp(:,:,1)); 
yf = squeeze(tmp(:,:,2)); 
xf = xf+360;
save('/Users/alliho/Documents/SIO/parcels_projects/parcels-westcoast-master/spawn_points_mats/seeds_capecan.mat', 'xf','yf')

% midx = nanmedian(xf(:))
% midy = nanmedian(yf(:))
% xf = (xf-midx).*cos(rot) - (yf-midy).*sin(rot);
% yf = (xf-midx).*sin(rot) - (yf-midy).*cos(rot);


% save('/Users/alliho/Documents/SIO/parcels_projects/parcels-westcoast-master/spawn_points_mats/seeds_capecan.mat', 'xf','yf')


%%

figure(9); clf; hold on;
plot(coastlon+360, coastlat, 'k-')
plot(xf, yf, 'r.')

% ylim([22 45])
% xlim([270 300])