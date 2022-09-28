function sdrs2uvlluvhourly(t)
%sdrs2uvlluvhourly: consolidates raw radial files into a single vector field
%input 
%       t: timestamp
%

% src_path  = '/imports/hfradar/hfrnet/db/files/radials/';
src_path  = 'support_data/RadialFiles/';
% out_path  = '/data/hfradar/hfrnet/trj/tjr/data/hourly/';
out_path  = 'support_data/TrackerOutput/hourly/';
%Data in the path below serves as state and gets removed by TJR particle processing
% out_path2 = '/data/hfradar/hfrnet/trj/tjr/data/pdata/';
out_path2 = 'support_data/TrackerOutput/pdata/';

load wchfradarinfo.mat;
% load sdcodargrid.mat gx gy;
load sdcodargrid_new.mat gx gy;

sts = ['SDCI'; 'SDPL'; 'SDBP'];
nr = size(sts,1);

[YYYY, MM, DD, hh, mm] = datestr0(t); 
for k = 1: nr
    j = find(strcmp(sts(k,:), stinfo.stid));
    affn = stinfo.affn{j};
    stid = stinfo.stid{j};
%     subpath = [src_path affn '/' stid '/' YYYY '-' MM '/'];
    subpath = [src_path '/' stid '/' YYYY '-' MM '/'];
    fname_meas  = ['RDL_m_' affn '_' stid '_' YYYY '_' MM '_' DD '_' hh mm '.ruv'];
    fname_ideal = ['RDL_i_' affn '_' stid '_' YYYY '_' MM '_' DD '_' hh mm '.ruv'];   
    if exist([subpath  fname_meas], 'file') 
%         fname_meas
        [d{k}.rs, d{k}.er, d{k}.ter, d{k}.r, d{k}.th, d{k}.x, d{k}.y, ~, ~, ~] = lluv2radialsall([subpath fname_meas]);
    elseif exist([subpath  fname_ideal], 'file') 
%         fname_ideal
        [d{k}.rs, d{k}.er, d{k}.ter, d{k}.r, d{k}.th, d{k}.x, d{k}.y, ~, ~, ~] = lluv2radialsall([subpath fname_ideal]);
    else
         d{k}.x = []; d{k}.y = []; d{k}.rs = []; d{k}.er = []; d{k}.th = []; d{k}.r = [];
         disp(['Neither ' fname_meas ' or ' fname_ideal ' exist']);
    end
end


sx = 2;
sy = 2;
sr = 5;  % max distance allowed from a detected current (km)
Pu = 400;
Rr = 40;
u = NaN*ones(length(gx(:)), 1);
v = NaN*ones(length(gx(:)), 1); 
nd = NaN*ones(length(gx(:)), 1);
% cycle through every single coordinate on the grid
% regardless if data exists
for k = 1: length(gx(:))
    t0 = clock;
    nsite = 0;
    x = []; y = []; dx = []; dy = []; rs = []; ang = []; ii = [];
    for l = 1: nr
        % distance km of point to coordinate of every detected current
        [de, dn] = lonlat2km(gx(k), gy(k), d{l}.x, d{l}.y);
        % find any current within 5 km of checked point
        id = find(sqrt(de.^2 + dn.^2) < sr);
        if ~isempty(id)
            % append information about the close-by current
            nsite = nsite + 1; 
            x = [x; d{l}.x(id)];
            y = [y; d{l}.y(id)];
            dx = [dx; de(id)];  % horizontal dist km
            dy = [dy; dn(id)];  % verticle dist km
            rs = [rs; d{l}.rs(id)];  % signed magnitude of current velocity
            ang = [ang; d{l}.th(id)*pi/180];
        end
    end
    
    ii = find(dx == 0 & dy == 0);
    % checked point already contains data
    % clear information of other same coordinates
    if ~isempty(ii)
        x(ii) = [];
        y(ii) = [];
        dx(ii) = [];
        dy(ii) = [];
        rs(ii) = [];
        ang(ii) = [];
    end
    
    if length(x) > 1 && nsite > 1
        % OI for u and v
        % google said this was Barne's interpolation
        P = Pu * eye(2);
        R = Rr * eye(length(rs));
        % get variance to each closeby point
        % used as the inverse weight
        f = dx.^2 / sx^2 + dy.^2 / sy^2;
        % gaussian weight value? idk
        % generate the u and v weights
        rhou = exp(-sqrt(f)) .* cos(ang);
        rhov = exp(-sqrt(f)) .* sin(ang);
        cmd = [rhou rhov];
        
        [x1, x2] = meshgrid(x, x);
        [y1, y2] = meshgrid(y, y); 
        [ang1, ang2] = meshgrid(ang, ang);
        % get the distance of every point to each other
        [drx_, dry_] = lonlat2km(x1, y1, x2, y2);
        % get variance?
        g = drx_.^2 / sx^2 + dry_.^2 / sy^2;
        % weight values
        w = exp(-sqrt(g));

        % what is happening
        cdd = w .* cos(ang1 - ang2);
%         cdd = w .* (cos(ang1).*cos(ang2) + sin(ang1).*sin(ang2));
        % Pu = 400
        cmdicdd = Pu * cmd' / (Pu * cdd + R);
%         icdd = inv(Pu * cdd + R);
%         cmdicdd = Pu * cmd' * icdd;
        % multiple weights by velocities, add it all together
        uv = cmdicdd * rs;
        % not even used either
%         ke = inv(P) * (P - cmdicdd * cmd * P);

        
        u(k) = uv(1);
        v(k) = uv(2);
        nd(k) = length(rs);
    end
end


if length(find(~isnan(u(:)))) > 100
	U = u(:)';
    V = v(:)';
	fname = ['Tot_SDLJ_' YYYY MM DD '_' hh '00.mat'];
	year1 = str2num(YYYY);
    time = t - datenum([year1 1 0 0 0 0]);
	save([out_path fname], 'U', 'V','t', 'year1', 'time', '-V6');
	save([out_path2 fname], 'U', 'V','t', 'year1', 'time', '-V6');
end
