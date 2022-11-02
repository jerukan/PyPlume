function [t, xc, yc, npts, nlife, xf, yf] = rmwalk(t, gx, gy, u, v, uerr)

%% Needs help documentation - MO
%
% Inputs
% t = datenum, (gx,gy) = (lon,lat), (u,v) = velocity in cm/s, uerr = random walk speed
% added to velocity in cm/s


%% Definitions (some to be converted to inputs)
debug = false;

% Decay fraction at the boundary
%dcf = 0.3;
dcf = 1;

% Conversion from speed to distance - in this case, cm/s -> km (note data
% timestep is 1hr so conversion expelicitly is cm/s -> km/hr * 1hr)
cv = 1e-5*3600;

% Define number of particles to release per time-step
npts = 50;

% Define the particle life
nlife = 128; %hrs

% Define point source for trajectories

% Ellen/Elly Oil Platform
% xc = -118.140833;
% yc = 33.582222;

% Ellen/Elly Pipeline - along/off elbow (p1)
% xc = -118.097793;
% yc = 33.683954;

% Ellen/Elly Pipeline - 1/2 way along offshore leg (p2)
%xc = -118.114273;
%yc = 33.641948;

% Break location (confirmed)
xc = -118.1107862;
yc = 33.64949218;

% Declare i explicitly
i = sqrt(-1);

%% Boundary setup
% This section can be done once and saved to a mat file which is loaded
% here. Maybe even make it an input field.

% Load boundary
load socal_boundary.mat

% Ginput used to find gridpoints near the boundary (South to north, if it matters...)
xi = [ -117.7412 -117.8036 -117.8660 -117.9295 -117.9919 -118.0542 -118.1166 -118.1790 -118.2414 -118.3026 -118.3661 -118.4296];
yi = [33.4869 33.5414 33.5404 33.5930 33.5930 33.6476 33.7011 33.7021 33.7021 33.7021 33.7021 33.7557];


% Define indices into grid for the gridpoints closest to xi and yi above
% (define gridpoint indices close to the boundary)
for j = 1: length(xi)
    [a, b] = lonlat2km(xi(j), yi(j), gx, gy);  
    c = abs(a + i*b);    ii = find(c == min(c));    llist(j) = ii; %grid list for gx gy
end

% Define indices into the boundary for the boundary point closest to each
% gridpoint near the boundary
for j = 1: length(llist)
    gx_ = gx(llist(j)); gy_ = gy(llist(j));    [a, b] = lonlat2km(gx_, gy_, xb, yb);    
    c = abs(a + i*b);    ii = find(c == min(c));    blist(j) = ii; %grid list for xb yb
end


% Determine the angle of coastline between every other element
[a, b] = lonlat2km(xb(1:end-2), yb(1:end-2), xb(3:end), yb(3:end));
thb = atan2(b, a); thb = [NaN thb NaN]; %add NaNs

% Define cumulative along-coast distance
[a, b] = lonlat2km(xb(2:end-1), yb(2:end-1), xb(3:end), yb(3:end));
c = abs(a + i*b);
baxis = cumsum(c); %baxis has 2 less elements than phb xb etc

%% Generate Trajectories
x_ = [];
y_ = [];
[ns, nt] = size(u);

for l = 1: length(t)
    t0 = clock;

    if debug
        hold off
        plot(xb,yb,'k-')
        hold on
        axis([ -118.0662 -117.8846   33.5657   33.6578])
        daspect([1.2 1 1])
    end
    
    % Condition velocity field near the boundary
    u_ = u(llist,l); v_ = v(llist, l);    
    mg = abs(u_ + i*v_)*dcf;
    ph = atan2(v_, u_);
    phb = thb(blist)';
    mgb = mg.*cos(ph - phb);
    ub = mgb.*cos(phb); vb = mgb.*sin(phb);
    emg = interp1(baxis(blist), mgb, baxis);
    emg = [NaN emg NaN];
    eub = emg.*cos(thb); evb = emg.*sin(thb);
    
    % Find good data in velocity field
    u_ = u(:, l);
    v_ = v(:, l);        
    ival = find(~isnan(u_));
    jval = find(~isnan(eub));
    
    % Add (append) new particles (locations) at the source
    if t(l) < datenum(2021,10,3)
        x_ = [x_; xc*ones(npts,1)]; y_ = [y_; yc*ones(npts,1)];
    end
    
    % Remove particles beyond nlife age
%     if l > nlife
%         x_(1:npts) = [];
%         y_(1:npts) = [];
%     end

    if debug
        quiver(gx(ival), gy(ival), u(ival,l)./1000, v(ival,l)./1000,0,'b')
        quiver(xb(jval), yb(jval), eub(jval)./1000, evb(jval)./1000,0,'b')
    end

    
    % Interpolate velocities at the particle locations using the combined
    % raw velocity field and border velocity field
    u_ = griddata([gx(ival); xb(jval)'], [gy(ival); yb(jval)'], [u_(ival); eub(jval)'], x_, y_);
    v_ = griddata([gx(ival); xb(jval)'], [gy(ival); yb(jval)'], [v_(ival); evb(jval)'], x_, y_);
        
    % Add constant noise (uerr) in random direction to particle locations 
    ntotalpt = length(x_);
    th = 2*pi.*rand([ntotalpt 1]);
    un_ = u_ + uerr*cos(th);
    vn_ = v_ + uerr*sin(th);
    mgn_ = abs(un_ + i*vn_);
    phn = atan2(vn_, un_);

    % Convert from velocity to distance and advect particles
    dx = un_*cv;
    dy = vn_*cv;
    [xn_, yn_] = km2lonlat(x_, y_, dx, dy); jj = 1;
    
    % Look for intersections between the particle track and the boundary
    for j = 1: ntotalpt
        [cx_, cy_] = polyxpoly(xb, yb, [x_(j) xn_(j)], [y_(j) yn_(j)], 'unique');
        if isempty(cx_)
            continue
        end
        % If an intersection is found, re-compute displacement in
        % along-shore direction
        cx_ = cx_(1);
        cy_ = cy_(1);
        [a, b] = lonlat2km(cx_, cy_, xb, yb);
        c = abs(a + i*b);
        ii = find(c == min(c));
        ii = ii(1);
        dx_ = eub(ii)*cv;
        dy_ = evb(ii)*cv;
        [xn_(j), yn_(j)] = km2lonlat(x_(j), y_(j), dx_, dy_); 
    end
    
    if debug
        plot(xn_, yn_, 'r.')
        plot(xc, yc, 'gx', 'markersize', 20)
        pause(3)
    end
    
    % Save each timestep to a cell array
    xf{l} = xn_; yf{l} = yn_;
    
    % Update the particle locations for the next iteration
    x_ = xn_; y_ = yn_;

    disp(sprintf('%d of %d, totalpt = %d : time = %4.3f', l, length(t), ntotalpt, etime(clock, t0)));
end

