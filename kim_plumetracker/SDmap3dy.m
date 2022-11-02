function SDmap3dy

load codar_nsd_bathy.mat
clon = nlon(1:121); clat = nlat(1:121:end); 
clear nlon nlat 

% czz = [-10:-10:-100 -200:-100:-2000];
% c = mycontour(clon, clat, nZ', czz);

% for i = 1:length(c)
%     if  c{i}.zdata == 0
%         plot(c{i}.xdata, c{i}.ydata, 'linewidth', 1.2, 'color', [0 0 0]); hold on
%         %    elseif czz{i}(2) == -50 | czz{i}(2) == -100
%     elseif c{i}.zdata == -50 | c{i}.zdata == -100 | mod(c{i}.zdata, -500) == 0,
%         plot(c{i}.xdata, c{i}.ydata, 'linewidth', 0.5, 'color', [0.1 0.1 0.1]);
%     else,
%         plot(c{i}.xdata, c{i}.ydata, 'linewidth', 0.2, 'color', [0.4 0.4 0.4])
%     end
% end


load coastOR2Mex.mat
plot(OR2Mex(:,1), OR2Mex(:,2), 'linewidth', 1.5, 'color', [0 0 0]);
