function ang = angntoe(ang)

% change the direction with north(cw) to direction with east(ccw)(for atan2) 
%
ii = find(ang > 270); jj = find(ang <= 270);
ang(ii) = -ang(ii) + 450; ang(jj) = -ang(jj) + 90;


% dir_ = dir;
% dir(dir_ > 270) = -dir_(dir_ > 270 ) + 450;
% dir(dir_ <= 270) = -dir_(dir_ <= 270 ) + 90;
% 
% inan = find(isnan(dir) == 1);
% ndir = zeros(1, length(dir));
% ndir(inan) = NaN;
% ndir(dir>90) = - dir(dir>90) + 450;
% ndir(dir <= 90) = - dir(dir <= 90) + 90;