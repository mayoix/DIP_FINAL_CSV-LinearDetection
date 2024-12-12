% 读取图像并转换到HSV颜色空间
image_files = {'1.png', '2.png', '3.png', '4.png', '5.png'}; % 替换为你的图像文件名

% 初始化HSV初始阈值
initial_H_min1 = 0; initial_H_max1 = 0.04;
initial_H_min2 = 0.98; initial_H_max2 = 1;
initial_S_min = 0.55; initial_S_max = 0.95;

H_min1 = initial_H_min1; H_max1 = initial_H_max1;
H_min2 = initial_H_min2; H_max2 = initial_H_max2;
S_min = initial_S_min; S_max = initial_S_max;

% 标志变量
target_detected = false;
no_target_count = 0;

% 遍历图像
for i = 1:length(image_files)
    % 读取图像并转换到HSV颜色空间
    img = imread(image_files{i});
    hsv_img = rgb2hsv(img);
    
    % 生成掩膜
    mask = ((hsv_img(:,:,1) >= H_min1 & hsv_img(:,:,1) <= H_max1) | ...
            (hsv_img(:,:,1) >= H_min2 & hsv_img(:,:,1) <= H_max2)) & ...
           (hsv_img(:,:,2) >= S_min & hsv_img(:,:,2) <= S_max);
    
    % 判断是否检测到目标
    if sum(mask(:)) > 0
        disp(['图像 ', image_files{i}, ' 检测到目标']);
        target_detected = true;
        no_target_count = 0;

        % 提取目标像素的H和S通道
        target_pixels_h = hsv_img(:,:,1);
        target_pixels_h = target_pixels_h(mask);
        target_pixels_s = hsv_img(:,:,2);
        target_pixels_s = target_pixels_s(mask);

        % 更新阈值
        if ~isempty(target_pixels_h)
            H_med = median(target_pixels_h);
            H_stdv = std(target_pixels_h);
            S_med = median(target_pixels_s);
            S_stdv = std(target_pixels_s);
            std_factor = 2;

            H_min1 = max(0, H_med - std_factor * H_stdv);
            H_max1 = min(1, H_med + std_factor * H_stdv);
            S_min = max(0, S_med - std_factor * S_stdv);
            S_max = min(1, S_med + std_factor * S_stdv);

            disp('更新后的阈值:');
            disp(['H区间1: [', num2str(H_min1), ', ', num2str(H_max1), ']']);
            disp(['S区间: [', num2str(H_min2), ', ', num2str(H_max2), ']']);
            disp(['S区间: [', num2str(S_min), ', ', num2str(S_max), ']']);
        end
    else
        disp(['图像 ', image_files{i}, ' 未检测到目标']);
        no_target_count = no_target_count + 1;

        if no_target_count >= 2
            H_min1 = initial_H_min1; H_max1 = initial_H_max1;
            H_min2 = initial_H_min2; H_max2 = initial_H_max2;
            S_min = initial_S_min; S_max = initial_S_max;
            disp('重置阈值为初始值');
        end
    end

    % 分割后的图像
    segmented_color_img = img;
    segmented_color_img(repmat(~mask, [1 1 3])) = 0; % 将非目标区域置为黑色

    % 显示分割结果
    gray_img_HS = uint8(mask) * 255;
    figure;
    imshow(gray_img_HS);
    title(['图像 ', image_files{i}, ' 分割结果']);
    
    % 添加形态学处理步骤
    % 膨胀和腐蚀（闭运算）
    se = strel('disk', 3);
    gray_img_HS_dilated = imdilate(gray_img_HS, se);  % 膨胀操作
    gray_img_HS_closed = imerode(gray_img_HS_dilated, se);  % 腐蚀操作

    % 形态学重建
    gray_img_HS_recon = imreconstruct(gray_img_HS_closed, gray_img_HS_dilated);

    % 可视化形态学处理结果
    figure;
    imshow(gray_img_HS_recon);
    title(['图像 ', image_files{i}, ' 闭运算和重建后的分割结果']);

    % 保存形态学处理后的图像
    output_filename = ['processed_segmented_', image_files{i}];
    imwrite(gray_img_HS_recon, output_filename);
    disp(['形态学处理后的分割图像已保存至: ', output_filename]);

    % 保留结果供后续处理
    % gray_img_HS_recon 已经更新





    % 参数设置
    n = 60;  % 沿x轴均匀选取的行数
    [rows, cols] = size(gray_img_HS_recon);  % 获取二值图像的尺寸
    x_indices = round(linspace(1, rows, n));  % 均匀选取的行索引
    
    left_edge_points = [];  % 存储左边缘点集
    right_edge_points = []; % 存储右边缘点集
    
    % 步骤1：逐行提取边缘点
    for i = 1:length(x_indices)
        x = x_indices(i);
        row_data = gray_img_HS_recon(x, :);  % 提取当前行的像素数据
        transitions = diff([0, row_data > 0, 0]);  % 检测像素值的突变点
        start_points = find(transitions == 1);  % 边缘开始点
        end_points = find(transitions == -1) - 1;  % 边缘结束点
        
        if ~isempty(start_points) && ~isempty(end_points)
            % 提取左边缘点（最小y值）和右边缘点（最大y值）
            left_edge_points = [left_edge_points; x, start_points(1)];
            right_edge_points = [right_edge_points; x, end_points(end)];
        end
    end
    
    % 检查是否足够的边缘点用于拟合
    if size(left_edge_points, 1) > 4 && size(right_edge_points, 1) > 4
        % 步骤2：分别对左边缘点集和右边缘点集拟合直线
        % 提取点集的x和y坐标
        x_left = left_edge_points(:, 1);
        y_left = left_edge_points(:, 2);
        x_right = right_edge_points(:, 1);
        y_right = right_edge_points(:, 2);
        
        % 剔除离群点
        %[x_left, y_left] = removeOutliers(x_left, y_left);
        %[x_right, y_right] = removeOutliers(x_right, y_right);

        % 使用最小二乘法进行直线拟合 (y = ax + b)
        % 使用 Huber 方法进行拟合
        [a_left, b_left] = huberFit(x_left, y_left);
        [a_right, b_right] = huberFit(x_right, y_right);
        
        % 显示拟合结果
        disp('左边缘直线拟合参数：');
        disp(['a_left = ', num2str(a_left), ', b_left = ', num2str(b_left)]);
        disp('右边缘直线拟合参数：');
        disp(['a_right = ', num2str(a_right), ', b_right = ', num2str(b_right)]);
        
        % 可视化边缘点和拟合直线
        figure;
        imshow(gray_img_HS_recon); hold on;
        scatter(left_edge_points(:, 2), left_edge_points(:, 1), 'r', 'filled'); % 左边缘点
        scatter(right_edge_points(:, 2), right_edge_points(:, 1), 'g', 'filled'); % 右边缘点
        
        % 绘制拟合直线
        x_line = 1:rows;
        y_left_fit = a_left * x_line + b_left;
        y_right_fit = a_right * x_line + b_right;
        plot(y_left_fit, x_line, 'r', 'LineWidth', 2); % 左边缘拟合直线
        plot(y_right_fit, x_line, 'g', 'LineWidth', 2); % 右边缘拟合直线
        title('边缘点与拟合直线');
        hold off;
    else
        disp('提取的边缘点不足，无法进行直线拟合');
    end
end



function intervals = mergeIntervals(bins, N, max_intervals)
    % 辅助函数：合并相连的单元格并限制区间数量
    bins = sort(bins);
    intervals = [];
    start = bins(1);
    for i = 2:length(bins)
        if bins(i) - bins(i-1) > 2
            intervals = [intervals; start, bins(i-1)];
            start = bins(i);
        end
    end
    intervals = [intervals; start, bins(end)]; % 最后一个区间
    if size(intervals, 1) > max_intervals
        intervals = intervals(1:max_intervals, :); % 保留最多4个区间
    end
end




function [a, b] = huberFit(x, y)
    % 使用 Matlab 的 fitlm 函数进行 Huber 鲁棒回归拟合
    if length(x) >= 2
        % 将 x 和 y 转换为表格形式
        data = table(x, y);
        % 使用 Huber 方法进行鲁棒回归
        mdl = fitlm(data, 'y ~ x', 'RobustOpts', 'huber');
        % 提取拟合的系数
        a = mdl.Coefficients.Estimate(2);  % 斜率
        b = mdl.Coefficients.Estimate(1);  % 截距
    else
        error('数据点不足，无法进行拟合');
    end
end


