function [rs, ser, ter, rng, ang, x, y, lonc, latc, rinfo] = lluv2radialsall(file_)

%LLUV2RADIALS: read radials and bearing angle from lluv format
%input 
%       file_ : full path of file
%		  fmt : format(
%
%output
%       rs : radial component
%       er : temporal uncertainty
%       rng : range from radar
%       ang : bearing angle CCW@E
%       x, y : location of raidal grid
%       latc, lonc : location of radar

%someone can estimate the radial grid with center from r and bearing
%angle,but we use the given data at this level.
% clear all;
% file_ = 'D:\work\data\codar\z.msrd\wc.ucert\Rad_i_SIO_SDSL_08-06-02_0800.hfrss10lluv';

lluvcomplete = 0; pdr = 0; pmanu = 0;
rs = []; ang = []; ser  = []; ter = []; rng = []; x = []; y = []; maxrng = NaN; thbin = NaN; spr = NaN; dth = NaN;

fid = fopen(file_);
while 1
    ci = fgetl(fid);
    
    if ci == -1, break; end
    if length(ci) <= 9, continue; end
    if strcmp(ci(1:10), '%TableType'),
        lluvfmt = ci(18:21); %break;
    elseif strcmp(ci(1:8), '%Origin:');
        ii = find(double(ci) == 32);
        latc = str2num(ci(ii(1):ii(end)-1));
        lonc = str2num(ci(ii(end)+1:end));
    elseif strcmp(ci(1:10), '%TableColu'),
        if strcmp(ci(1:16), '%TableColumnType'),
            cs = ci(19:end);                 cn = double(cs);
            ii = find(cn == 32);
            if ii ~= length(cs), ii = [ii length(cs)+1]; end %in case of no space at the end of Table column
            for j = 1: length(ii)-1
                colt{j} = cs(ii(j)+1:ii(j+1)-1);
            end
        end
    elseif strcmp(ci(1:10), '%Manufactu'),
        ii = find(double(ci) == 32);
        if strcmpi(ci(ii(end)+1:end), 'SeaSonde'), pmanu = 1;
        else, pmanu = 2; end
    elseif strcmp(ci(1:10), '%RangeReso'),
        ii = find(double(ci) == 32);
        dr = str2num(ci(ii(1):end)); pdr = 1;
       %RangeResolutionMeters: 399.60
       %RangeResolutionKMeters: 0.9990
       if strcmp(ci(17), 'M'),                dr = dr/1000;            end
    elseif strcmp(ci(1:10), '%TableRows'),
        ii = find(double(ci) == 32);
        nrows = str2num(ci(ii(end)+1:end));
        if nrows == 0, lluvcomplete = 0; break; end
    elseif strcmp(ci(1:10), '%TableEnd:'),
        lluvcomplete = 1; break;
    end
    
end
fclose(fid);

jetm = 1; jesp = 1; jspr = 1;%just in case


if pdr == 0, dr = NaN; end

d = load(file_);
if  lluvcomplete ~= 1 || isempty(d), 
    rinfo.dr = dr; rinfo.maxrng = maxrng; rinfo.thbin = thbin; rinfo.spr = spr; rinfo.dth = dth; rinfo.pmanu = pmanu;   return; 
end

for j = 1: length(colt)
    %if strcmp(lluvfmt, 'RDL1'),
    if pmanu == 1,
        if strcmp(colt{j}, 'LOND'), jlon = j;
        elseif strcmp(colt{j}, 'LATD'), jlat = j;
        elseif strcmp(colt{j}, 'RNGE'), jrng = j;
        elseif strcmp(colt{j}, 'BEAR'), jber = j;
        elseif strcmp(colt{j}, 'VELO'), jvel = j;
        elseif strcmp(colt{j}, 'ESPC'), jesp = j;
        elseif strcmp(colt{j}, 'ETMP'), jetm = j;
        elseif strcmp(colt{j}, 'SPRC'), jspr = j;                        
        end

    elseif pmanu == 2,
        if strcmp(colt{j}, 'LOND'), jlon = j;
        elseif strcmp(colt{j}, 'LATD'), jlat = j;
        elseif strcmp(colt{j}, 'RNGE'), jrng = j;
        elseif strcmp(colt{j}, 'BEAR'), jber = j;
        elseif strcmp(colt{j}, 'VELO'), jvel = j;
        %elseif strcmp(colt{j}, 'EVAR'), jesp = j;
        elseif strcmp(colt{j}, 'EACC'), jetm = j; 
        end
    end
end

% rs = -rs(come near positive?), the base_angle = 0?
rs = -d(:,jvel);    x = d(:,jlon);      y = d(:,jlat);
ter = d(:,jetm);  rng = d(:,jrng);        
ang = angntoe(d(:,jber));%%NCW->ECCW,

%if pmanu == 1, 
minth = min(ang);
a = msort2(ang); 
if length(a) > 1,                   
    dth = min(sort(diff(a)));   
    if 0 < dth & dth <= 5,
        if minth > dth - 180,
            minth = minth - floor((minth + 180)/dth)*dth;
        end
        thbin = minth:dth:180;
    end
end
if pmanu == 1,     
    ser = d(:,jesp);     spr = d(:,jspr);        maxrng = max(spr); 
elseif pmanu == 2,
    ser = NaN;          spr = NaN; maxrng = NaN; 
end

rinfo.dr = dr; rinfo.maxrng = maxrng; rinfo.thbin = thbin; rinfo.spr = spr; rinfo.dth = dth; rinfo.pmanu = pmanu;





