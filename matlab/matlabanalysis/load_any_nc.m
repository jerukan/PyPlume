function [wb] = load_any_nc(fname)

% variable list
buoy = ncinfo(fname);
paramlist = strings(1,length(buoy.Variables)); 
for k=1:length(buoy.Variables)
    paramlist(k) = buoy.Variables(k).Name;
end

% load variables into structure
wb = loadncdf([fname], paramlist);  % function created to load netCDF into structs
