function ImageDistorter(varargin)
    % ImageDistorter(img1, img2, ...)
    % Applies transformations defined in default.csv to each input image.

    if nargin < 1
        error('Please provide at least one image base name (without .png).');
    end

    params = readmatrix("default.csv", 'Delimiter', ',');
    [numConfigs, numCols] = size(params);
    fprintf('Loaded %d parameter sets with %d columns.\n', numConfigs, numCols);

    if numCols < 8
        error('CSV must have at least 8 columns: ScaleX, ScaleY, Angle, BlurSigma, NoiseVar, ContrastFactor, ShearX, ShearY');
    end

    for i = 1:nargin
        baseName = varargin{i};
        filename = [baseName '.png'];

        if ~isfile(filename)
            warning('File not found: %s', filename);
            continue;
        end

        img = imread(filename);
        [h, w, c] = size(img);

        for row = 1:numConfigs
            scalex = params(row,1);
            scaley = params(row,2);
            angle  = params(row,3);
            blurSigma = params(row,4);
            noiseVar  = params(row,5);
            contrastF = params(row,6);
            shearX = params(row,7);
            shearY = params(row,8);

            scaled = imresize(img, [round(h*scaley), round(w*scalex)]);
            scaledCanvas = uint8(ones(h, w, c) * 255);
            startRow = max(1, floor((h - size(scaled,1))/2) + 1);
            startCol = max(1, floor((w - size(scaled,2))/2) + 1);
            endRow = min(h, startRow + size(scaled,1) - 1);
            endCol = min(w, startCol + size(scaled,2) - 1);
            rowsScaled = 1:(endRow-startRow+1);
            colsScaled = 1:(endCol-startCol+1);
            scaledCanvas(startRow:endRow, startCol:endCol, :) = scaled(rowsScaled, colsScaled, :);

            rotated = imrotate(scaledCanvas, angle, 'bilinear', 'crop');
            mask = all(rotated == 0, 3);
            rotated(repmat(mask, [1 1 c])) = 255;

            if blurSigma > 0
                blurred = imgaussfilt(rotated, blurSigma);
            else
                blurred = rotated;
            end

            if noiseVar > 0
                noisy = imnoise(blurred, 'gaussian', 0, noiseVar);
            else
                noisy = blurred;
            end

            imgDouble = im2double(noisy);
            contrastImg = imadjust(imgDouble, [], [], contrastF);
            contrastImg = im2uint8(contrastImg);

            tform = affine2d([1 shearX 0; shearY 1 0; 0 0 1]);
            shearedImg = imwarp(contrastImg, tform, ...
                'OutputView', imref2d(size(contrastImg)), 'FillValues', 255);

            outName = sprintf('%s_cfg%d.png', baseName, row);
            imwrite(shearedImg, outName);
        end
    end

    fprintf('All transformations complete.\n');
end
