clear all;
fid = fopen('timestamp.txt');
d = fgetl(fid);
dend = fgetl(fid);
fclose(fid);
str2date = @(t) datetime(str2num(t(1:4)), str2num(t(6:7)), ...
        str2num(t(9:10)), str2num(t(12:13)), 0, 0);
t = str2date(d);
tend = str2date(dend);
% shit way of dealing with dates lol
% t = floor(datenum([str2num(d(1:4)) str2num(d(6:7)) ...
%         str2num(d(9:10)) str2num(d(12:13)) 0 0])*24-2)/24;
    %it was -1, syk change it into -2, due to sdbp late reporting 2/19/10
%syk changed this into -3, sdbp and sdpl report the data at hh:55 and hh:07. 
       %syk change it back. on 05/12/2010
       
tcurr = t;
while tcurr <= tend
    sdrs2uvlluvhourly(datenum(tcurr));
    tcurr = tcurr + hours(1);
end
