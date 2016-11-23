function [result, coefs] = getWaveletsv2(currAScan)
    vec = vec2mat(currAScan, 32);
    coefs = abs(cwt(vec, 1:32, 'sym6'));
    result = coefs(:);
end