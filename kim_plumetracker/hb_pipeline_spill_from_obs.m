% Needs help documentation or general summary - MO

%% Environment
% nctoolbox 
%addpath /home/motero/Documents/Computing/Software/MATLAB/Toolboxes/nctoolbox
addpath /Users/motero/Documents/Computing/MATLAB/Toolboxes/nctoolbox
setup_nctoolbox

%addpath /home/motero/Documents/Computing/Software/MATLAB/Work/General % for lonlat2km
addpath /Users/motero/Documents/Computing/MATLAB/Work/General % for lonlat2km

% mapping tools (mapstuff from sea-mat)
%addpath('/data/hfradar/hfrnet/matlab/mapstuff')


%% Definitons
%logfile = '/data/hfradar/hfrnet/logs/ocsd_trj2proc';
%url = 'http://hfrnet.ucsd.edu/thredds/dodsC/HFRNet/USWC/6km/hourly/RTV';
url = 'http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/6km/hourly/RTV/HFRADAR_US_West_Coast_6km_Resolution_Hourly_RTV_best.ncd';

% Dataset subsetting

% 10/6 radarsat to 10/7 capella
% t0 = datenum( 2021, 10, 6, 14, 0, 0 );
% tf = datenum( 2021, 10, 7, 15, 0, 0 );

% % 10/7 capella forward
% t0 = datenum( 2021, 10, 7, 15, 0, 0 );
% tf = datenum( 2021, 10, 10, 4, 0, 0 );

% % 10/9 sentenel 1A forward
% t0 = datenum( 2021, 10, 9, 2, 0, 0 );
% tf = datenum( 2021, 10, 10, 4, 0, 0 );

% 10/9 radarsat forward
t0 = datenum( 2021, 10,  9, 20, 0, 0 );
tf = datenum( 2021, 10, 13, 22, 0, 0 );

s.lat = [32 33.8];
s.lon = [-118.7 -117];
s.time = [t0 tf];

%% Get data
%log_message( logfile, 'info', ['Begin data request from ' url] )
[x, y, t, u, v] = tdsload(url, s);
%log_message( logfile, 'info', 'URL data request completed' )


%% QC velocity field for trajectory calculations
%log_message( logfile, 'info', 'Begin velocity field quality control' )
whos t
[t, x, y, u, v] = qcvel(t, x, y, u, v);
whos t
% if isempty(t)
%     log_message( logfile, 'notice', 'No data returned after quality control' )
%     exit
% end
%log_message( logfile, 'info', 'Velocity field quality control complete' )


%% Add 46025 wind drift to hfr velocity field

%Load wind data
% w = load('/Users/motero/Desktop/tmp/spill/data/ndbc_46025_wind.txt');
% th = mod( -90 - w(:,6), 360 ); %compass degrees to polar & from to toward
% [wu,wv] = pol2cart( deg2rad(th), w(:,7) );
% wt = datenum([w(:,1:5) zeros( size(w,1), 1)]);
% clear th w
% 
% % Interp to hfr time
% wui = interp1( wt, wu, t );
% wvi = interp1( wt, wv, t );
% 
% % Scale to 3%
% wuis = wui.*3; % m/s -> cm/s * 0.03
% wvis = wvi.*3; % m/s -> cm/s * 0.03
% 
% % Add scaled wind as uniform field to HFR velocity field
% um = u + repmat(wuis,size(u,1),1);
% vm = v + repmat(wvis,size(v,1),1);


%% Add Magnus wind drift to hfr velocity field

%Load wind data
w = load('/Users/motero/Desktop/tmp/spill/data/metbuoy_027-02_202110.dat');
th = mod( -90 - w(:,11), 360 ); %compass degrees to polar & from to toward
%[wu,wv] = pol2cart( deg2rad(th), w(:,15) ); % z = 10m
[wu,wv] = pol2cart( deg2rad(th), w(:,12) ); % z = ~1m
wt = datenum( w(:,2:7) );
clear th w

% Interp to hfr time
wui = interp1( wt, wu, t );
wvi = interp1( wt, wv, t );

% Scale to 3%
wuis = wui.*3; % m/s -> cm/s * 0.03
wvis = wvi.*3; % m/s -> cm/s * 0.03

% Add scaled wind as uniform field to HFR velocity field
um = u + repmat(wuis,size(u,1),1);
vm = v + repmat(wvis,size(v,1),1);


%% Compute trajectories
%log_message( logfile, 'info', 'Begin random walk particle simulation' )
[t, xc, yc, npts, nlife, xf, yf] = rmwalk_obs(t, x, y, um, vm, 5);
%log_message( logfile, 'info', 'Random walk particle simulation complete' )


%% Plot results and save
%log_message( logfile, 'info', 'Begin plotting all timesteps during spin-up' )
path = '/Users/motero/Desktop/tmp/spill/img_obs';
for I = 1:numel(xf)
    close
    hb_pipeline_spill_plots( t(I), xc, yc, npts, nlife, xf(I), yf(I) );
%    hb_pipeline_spill_plots( t(1:I), xc, yc, npts, nlife, xf(1:I), yf(1:I) );
    filename = ['hbSpill_hfrLsTrj_r5_dcf1_wmagnus_3pct_t1009radarsat2_'  datestr( t(I), 'yyyymmddHH' ) '.jpg'];
    print('-djpeg90', '-r0', fullfile( path, filename ) )
%    log_message( logfile, 'info', ['Saved image to ' fullfile( path, filename )] )
end
%log_message( logfile, 'info', 'Spin-up plotting complete' )

%% Exit
%exit
