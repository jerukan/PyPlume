function [YYYY, MM, DD, hh, mm] = datestr0(t)
d = datevec(t); nt = length(t); 
if nt ~= 1, error ('the length of time should be one !'); end

YYYY = num2str(d(1));
if d(2)< 10,  MM = ['0',num2str(d(2))]; ,else, MM = num2str(d(2)); end
if d(3)< 10,  DD = ['0',num2str(d(3))]; ,else, DD = num2str(d(3)); end
if d(4)< 10,  hh = ['0',num2str(d(4))]; ,else, hh = num2str(d(4)); end
if d(5)< 10,  mm = ['0',num2str(d(5))]; ,else, mm = num2str(d(5)); end
