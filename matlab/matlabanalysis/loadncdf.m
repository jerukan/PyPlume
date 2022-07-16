function [ outlist ] = loadncdf( filename, paramlist)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
%   Use this to load paramlist:
%     

for n=1:length(paramlist)
    thisparam = char(paramlist(n));
    fieldname = thisparam;
    S1.(fieldname) = ncread(filename, thisparam);
    
end
outlist = S1;
end

