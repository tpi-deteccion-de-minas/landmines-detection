function [result, coefs] = getWaveletsv2(currAScan)
    vec = vec2mat(currAScan, 48);
    coefs = abs(cwt(vec, 1:48, 'sym6'));
    result = coefs(:);
end