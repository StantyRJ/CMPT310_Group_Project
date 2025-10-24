function parse_raw(image_path, output_dir, n, characters)
    % parse_raw
    % Splits an image composed of rows of square character images into
    % individual files named based on the ascii character and row index.
    %
    % Parameters:
    % image_path - Full path to the input image.
    % output_dir - Directory where output images will be saved.
    % n          - Size (in pixels) of each square character (default = 64).
    % characters - Array of characters representing the order in each row
    %              (default = ['a'-'z', 'A'-'Z', '0'-'9']).
    %
    % Example:
    % parse_raw("dataset_raw.png", "output/");

    % Default values
    scriptDir = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(image_path)
        image_path  = fullfile(scriptDir, '..', 'dataset_raw_bg.png');
    end
    if nargin < 2 || isempty(output_dir)
        output_dir = fullfile(scriptDir, '..', 'characters');
    end
    if nargin < 3 || isempty(n)
        n = 64;
    end
    if nargin < 4 || isempty(characters)
        characters = [char('a':'z'), char('A':'Z'), char('0':'9')];
    end

    % Create output directory if it doesn't exist
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % Read input image
    img = imread(image_path);
    [imgHeight, imgWidth, ~] = size(img);

    % Determine number of rows and columns
    column_count = floor(imgWidth / n);
    row_count = floor(imgHeight / n);

    % Ensure that the characters are provided properly
    if length(characters) ~= column_count
        error('characters must contain exactly %d characters. It has %d.', column_count, length(characters));
    end

    % Loop through rows and columns
    for row_index = 1:row_count
        for column_index = 1:column_count
            % Extract the character image
            yStart = (row_index-1)*n + 1;
            yEnd   = row_index*n;
            xStart = (column_index-1)*n + 1;
            xEnd   = column_index*n;

            file_name = sprintf('%d_%d.png', uint8(characters(column_index)), row_index);
            file_path = fullfile(output_dir, file_name);

            imwrite(img(yStart:yEnd, xStart:xEnd, :), file_path);
            if row_index == 1
                imshow(img(yStart:yEnd, xStart:xEnd, :));
            end
        end
    end

    fprintf('Done! Saved %d rows of character images to %s\n', row_count, output_dir);
end
