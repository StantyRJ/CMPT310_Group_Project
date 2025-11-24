function distort(k)
    % distort(k)
    % Generates k random distorted versions of each PNG in ../characters.
    % Distortion parameters sampled from normal distributions (mean, std).
    % Saves results in ../distorted.

    if nargin < 1
        k = 10; % default number of variants
    end

    scriptDir = fileparts(mfilename('fullpath'));
    inputDir  = fullfile(scriptDir, '..', 'characters');
    outputDir = fullfile(scriptDir, '..', 'distorted');

    if ~isfolder(inputDir)
        error('Input directory not found: %s', inputDir);
    end
    if ~isfolder(outputDir)
        mkdir(outputDir);
    end

    % ---------------------------------------------------------
    %  PARAMETER STATISTICS (adjust freely)
    % ---------------------------------------------------------
    scaleX_mu = 1.0;     scaleX_std = 0.10;
    scaleY_mu = 1.0;     scaleY_std = 0.10;

    angle_mu  = 0;       angle_std  = 7;     % degrees

    blur_mu   = 0.8;     blur_std   = 0.5;

    noise_mu  = 0.005;  noise_std  = 0.01;

    contrast_mu = 1.0;   contrast_std = 0.2;

    shearX_mu = 0;       shearX_std = 0.04;
    shearY_mu = 0;       shearY_std = 0.04;
    % ---------------------------------------------------------

    imageFiles = dir(fullfile(inputDir, '*.png'));
    if isempty(imageFiles)
        error('No PNG files found in %s', inputDir);
    end

    fprintf("Generating %d variants per image...\n", k);

    % ==============================================================
    %                      PROCESS EACH IMAGE
    % ==============================================================
    for i = 1:numel(imageFiles)
        filename = fullfile(inputDir, imageFiles(i).name);
        [~, baseName, ~] = fileparts(filename);
        fprintf("Processing %s...\n", imageFiles(i).name);

        [img, ~, alpha] = imread(filename);

        % --- Handle alpha channels (flatten over white) ---
        if ~isempty(alpha)
            img = im2double(img);
            alpha = im2double(alpha);

            if size(img,3) == 1
                img = repmat(img, 1, 1, 3);
            end

            white = ones(size(img));
            img = img .* alpha + white .* (1 - alpha);
            img = im2uint8(img);
        end

        [h, w, c] = size(img);

        % ======================================================
        %         GENERATE k RANDOM DISTORTIONS
        % ======================================================
        for sample = 1:k

            % Sample parameters from normal distributions ----------
            scalex     = max(0.1,  scaleX_mu + scaleX_std * randn());
            scaley     = max(0.1,  scaleY_mu + scaleY_std * randn());

            angle      = angle_mu      + angle_std      * randn();
            blurSigma  = max(0,  blur_mu + blur_std * randn());
            noiseVar   = max(0,  noise_mu + noise_std * randn());

            contrastF  = max(0.1, contrast_mu + contrast_std * randn());

            shearX     = shearX_mu + shearX_std * randn();
            shearY     = shearY_mu + shearY_std * randn();

            % =====================================================
            %                 APPLY EFFECTS (ORDERED)
            % =====================================================

            % 1. Scale --------------------------------------------
            scaled = imresize(img, [round(h*scaley), round(w*scalex)]);
            canvas = uint8(ones(h, w, c) * 255);

            startRow = max(1, floor((h - size(scaled,1))/2) + 1);
            startCol = max(1, floor((w - size(scaled,2))/2) + 1);
            endRow   = min(h, startRow + size(scaled,1) - 1);
            endCol   = min(w, startCol + size(scaled,2) - 1);

            rowsScaled = 1:(endRow-startRow+1);
            colsScaled = 1:(endCol-startCol+1);

            canvas(startRow:endRow, startCol:endCol, :) = scaled(rowsScaled, colsScaled, :);
            outImg = canvas;

            % 2. Rotate using imwarp (white fill, no black borders)
            theta = deg2rad(angle);
            R = [ cos(theta) -sin(theta) 0;
                  sin(theta)  cos(theta) 0;
                  0           0          1 ];
            
            tformRotate = affine2d(R);
            
            outImg = imwarp(outImg, tformRotate, ...
                'OutputView', imref2d(size(outImg)), ...
                'FillValues', 255);   % white background

            % 3. Blur ----------------------------------------------
            if blurSigma > 0
                outImg = imgaussfilt(outImg, blurSigma);
            end

            % 4. Add noise -----------------------------------------
            if noiseVar > 0
                outImg = imnoise(outImg, 'gaussian', 0, noiseVar);
            end

            % 5. Contrast (non-destructive linear contrast) --------
            imgDouble = im2double(outImg);
            meanVal = mean(imgDouble(:));

            imgDouble = imgDouble * contrastF + meanVal * (1 - contrastF);
            imgDouble = min(max(imgDouble, 0), 1);  % clip
            outImg = im2uint8(imgDouble);

            % 6. Shear ---------------------------------------------
            tform = affine2d([1 shearX 0; shearY 1 0; 0 0 1]);
            outImg = imwarp(outImg, tform, ...
                'OutputView', imref2d(size(outImg)), ...
                'FillValues', 255);

            % Save --------------------------------------------------
            outName = sprintf('%s_%03d.png', baseName, sample);
            outPath = fullfile(outputDir, outName);
            imwrite(outImg, outPath);

        end

    end

    fprintf("All distortions complete.\n");
end
