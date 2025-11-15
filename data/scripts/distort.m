function distort()
    % distort()
    % Reads PNG images from ../characters/,
    % applies random transformations using CSV values as ranges (+/- percentage),
    % saves results to ../distorted.

    scriptDir = fileparts(mfilename('fullpath'));
    inputDir  = fullfile(scriptDir, '..', 'characters');
    outputDir = fullfile(scriptDir, '..', 'distorted');
    csvPath   = fullfile(scriptDir, '..', 'distortions.csv');

    if ~isfolder(inputDir)
        error('Input directory not found: %s', inputDir);
    end

    if ~isfolder(outputDir)
        mkdir(outputDir);
    end

    params = readmatrix(csvPath, 'Delimiter', ',');
    [numConfigs, numCols] = size(params);
    fprintf('Loaded %d parameter sets with %d columns.\n', numConfigs, numCols);

    if numCols < 8
        error('CSV must have at least 8 columns: ScaleX, ScaleY, Angle, BlurSigma, NoiseVar, ContrastFactor, ShearX, ShearY');
    end

    imageFiles = dir(fullfile(inputDir, '*.png'));
    if isempty(imageFiles)
        error('No PNG images found in %s', inputDir);
    end


    randomRange = 0.05;

    for i = 1:numel(imageFiles)
        filename = fullfile(inputDir, imageFiles(i).name);
        [~, baseName, ~] = fileparts(filename);
        fprintf('Processing %s...\n', imageFiles(i).name);

        [img, ~, alpha] = imread(filename);
        if ~isempty(alpha)
            img = im2double(img);
            alpha = im2double(alpha);

            if size(img,3) == 1
                img = repmat(img, 1, 1, 3);
            end

            whiteBg = ones(size(img));
            img = img .* alpha + whiteBg .* (1 - alpha);
            img = im2uint8(img);
        end

        [h, w, c] = size(img);

        for row = 1:numConfigs
            %get csv values
            scaleXRange = params(row,1);
            scaleYRange = params(row,2);
            angleRange  = params(row,3);
            blurRange = params(row,4);
            noiseRange  = params(row,5);
            contrastRange = params(row,6);
            shearXRange = params(row,7);
            shearYRange = params(row,8);

            %Apply Rand Values
            scalex = scaleXRange;
            scaley = scaleYRange;
            angle  = (2*rand()-1) * angleRange;
            blurSigma = rand() * blurRange;
            noiseVar  = rand() * noiseRange;
            contrastF = contrastRange;
            shearX = (2*rand()-1) * shearXRange;
            shearY = (2*rand()-1) * shearYRange;
            
            % Ensure valid ranges
            scalex = max(0.1, scalex);
            scaley = max(0.1, scaley);
            blurSigma = max(0, blurSigma);
            noiseVar = max(0, noiseVar);
            contrastF = max(0.1, contrastF);

            % Scale
            scaled = imresize(img, [round(h*scaley), round(w*scalex)]);
            scaledCanvas = uint8(ones(h, w, c) * 255);
            startRow = max(1, floor((h - size(scaled,1))/2) + 1);
            startCol = max(1, floor((w - size(scaled,2))/2) + 1);
            endRow = min(h, startRow + size(scaled,1) - 1);
            endCol = min(w, startCol + size(scaled,2) - 1);
            rowsScaled = 1:(endRow-startRow+1);
            colsScaled = 1:(endCol-startCol+1);
            scaledCanvas(startRow:endRow, startCol:endCol, :) = scaled(rowsScaled, colsScaled, :);

            % Rotate
            rotated = imrotate(scaledCanvas, angle, 'bilinear', 'crop');
            mask = all(rotated == 0, 3);
            rotated(repmat(mask, [1 1 c])) = 255;

            % Blur
            if blurSigma > 0
                blurred = imgaussfilt(rotated, blurSigma);
            else
                blurred = rotated;
            end

            % Noise
            if noiseVar > 0
                noisy = imnoise(blurred, 'gaussian', 0, noiseVar);
            else
                noisy = blurred;
            end

            % Contrast
            imgDouble = im2double(noisy);
            contrastImg = imadjust(imgDouble, [], [], contrastF);
            contrastImg = im2uint8(contrastImg);

            % Shear
            tform = affine2d([1 shearX 0; shearY 1 0; 0 0 1]);
            shearedImg = imwarp(contrastImg, tform, ...
                'OutputView', imref2d(size(contrastImg)), 'FillValues', 255);

            % Save Output
            outName = sprintf('%s_cfg%d.png', baseName, row);
            outPath = fullfile(outputDir, outName);
            imwrite(shearedImg, outPath);
        end
    end

    fprintf('All transformations complete.\n');
end