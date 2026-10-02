function [R_color, L_v] = srie_wrapper(img)

    if ~isfloat(img), img = im2double(img); end

    HSV = rgb2hsv(img);
    V = HSV(:,:,3);

    evalc('[~, R_v, L_v] = srie(V);');

    HSV_R = HSV;
    HSV_R(:,:,3) = R_v;
    R_color = hsv2rgb(HSV_R);

    R_color = min(max(R_color, 0), 1);
    L_v = min(max(L_v, 0), 1);
end
