% 读取图像并转换到HSV颜色空间
image_files = {'2.jpg'}; % 替换为你的图像文件名

% 初始化HSV初始阈值
initial_H_min1 = 0; initial_H_max1 = 0.04;
initial_H_min2 = 0.98; initial_H_max2 = 1;
initial_S_min = 0.65; initial_S_max = 0.95;

H_min1 = initial_H_min1; H_max1 = initial_H_max1;
H_min2 = initial_H_min2; H_max2 = initial_H_max2;
S_min = initial_S_min; S_max = initial_S_max;

% 标志变量，判断是否在上一张图像中检测到目标
target_detected = false;
% 统计连续未检测到目标的次数
no_target_count = 0;

for i = 1:length(image_files)
    % 读取图像并转换到HSV颜色空间
    img = imread(image_files{i});
    hsv_img = rgb2hsv(img);
    
    % 利用当前阈值进行初步目标检测
    mask = ((hsv_img(:,:,1) >= H_min1 & hsv_img(:,:,1) <= H_max1) | ...
            (hsv_img(:,:,1) >= H_min2 & hsv_img(:,:,1) <= H_max2)) & ...
           (hsv_img(:,:,2) >= S_min & hsv_img(:,:,2) <= S_max);
    
    % 判断是否检测到目标
    if sum(mask(:)) > 0
        % 检测到目标
        disp(['图像 ', image_files{i}, ' 检测到目标']);
        
        target_detected = true;
        no_target_count = 0; % 重置连续未检测到计数
        
        % ---- 以下为目标特征统计与阈值更新步骤 ----
        % 提取目标区域像素
        target_pixels = hsv_img(repmat(mask, [1 1 3]));
        if ~isempty(target_pixels)
            % 将目标区域的HSV值取出
            % 注意：target_pixels是mask应用后的结果，需要reshape一下才能获得对应通道数据
            target_pixels_reshaped = reshape(target_pixels, [], 3);
            H_channel = target_pixels_reshaped(:,1);
            S_channel = target_pixels_reshaped(:,2);

            N = 50; % 区间数可根据需要调整
            
            H_hist = histcounts(H_channel(:), N);
            S_hist = histcounts(S_channel(:), N);
            
            % 计算总目标像素数
            total_pixels = sum(H_hist);
            % 将像素数从大到小排序并计算98%累积和
            [sorted_H_hist, H_indices] = sort(H_hist, 'descend');
            [sorted_S_hist, S_indices] = sort(S_hist, 'descend');

            H_cumulative_sum = cumsum(sorted_H_hist);
            S_cumulative_sum = cumsum(sorted_S_hist);

            H_threshold_index = find(H_cumulative_sum >= 0.98 * total_pixels, 1);
            S_threshold_index = find(S_cumulative_sum >= 0.98 * total_pixels, 1);

            H_selected_bins = H_indices(1:H_threshold_index);
            S_selected_bins = S_indices(1:S_threshold_index);

            % 合并相邻区间，不超过4个区间
            H_intervals = mergeIntervals(H_selected_bins, N, 4);
            S_intervals = mergeIntervals(S_selected_bins, N, 4);

           % 更新阈值
            H_min1 = H_intervals(1,1) / N;
            H_max1 = H_intervals(1,2) / N;
            if size(H_intervals, 1) > 1
                H_min2 = H_intervals(2,1) / N;
                H_max2 = H_intervals(2,2) / N;
            else
                % 如果只有一个区间，则不启用第二区间
                H_min2 = 0; H_max2 = 1;
            end

            S_min = S_intervals(1,1) / N;
            S_max = S_intervals(1,2) / N;


            disp('更新后的HSV阈值：');
            disp(['H区间1: [', num2str(H_min1), ', ', num2str(H_max1), ']']);
            disp(['H区间2: [', num2str(H_min2), ', ', num2str(H_max2), ']']);
            disp(['S区间: [', num2str(S_min), ', ', num2str(S_max), ']']);
        else
            disp('没有提取到目标像素，跳过阈值更新操作');
        end

    else
        % 未检测到目标
        disp(['图像 ', image_files{i}, ' 未检测到目标']);
        
        if target_detected
            % 如果上一张图像中有目标，但这一张没有，说明目标可能消失
            no_target_count = no_target_count + 1;
            
            if no_target_count >= 2
                % 连续两帧未检测到目标，重置阈值
                H_min1 = initial_H_min1; H_max1 = initial_H_max1;
                H_min2 = initial_H_min2; H_max2 = initial_H_max2;
                S_min = initial_S_min; S_max = initial_S_max;
                
                disp('目标连续未出现，阈值已重置为初始值');
                target_detected = false;
                no_target_count = 0;
            else
                disp('目标暂时未出现，保留上一帧更新的阈值');
            end
        else
            % 若本来就未检测到过目标，则不做阈值变化
            disp('目标从未出现过，继续使用初始阈值（或现有阈值）。');
        end
    end

    % 使用H阈值分割后的图像
    hsv_img_segmented_H = hsv_img;
    hsv_img_segmented_H(:,:,1) = hsv_img_segmented_H(:,:,1) .* mask;  % 使用H通道的mask分割
    
    gray_img_H = uint8(mask) * 255;

    % 显示并保存H阈值分割后的图像
    %figure;
    %imshow(hsv_img_segmented_H);
    %imshow(gray_img_H);
    %title('H阈值分割后的图像');
    %saveas(gcf, 'H_threshold_segmented_image.jpg');


    % 使用HSV阈值分割后的图像
    hsv_img_segmented_HS = hsv_img;
    hsv_img_segmented_HS(:,:,1) = hsv_img_segmented_HS(:,:,1) .* mask;  % 使用H通道的mask分割
    hsv_img_segmented_HS(:,:,2) = hsv_img_segmented_HS(:,:,2) .* mask;  % 使用S通道的mask分割

    % 显示分割结果
    gray_img_HS = uint8(mask) * 255;
    figure;
    %imshow(hsv_img_segmented_HS);
    imshow(gray_img_HS);
    title(['图像 ', image_files{i}, ' 分割结果']);

    %比较发现H, S阈值分割后的图像效果更好，采用该分割方法
    

    %% 基于最小二乘法的直线目标检测

    % 图像预处理

    % 采用先膨胀后腐蚀的闭运算，填补小空洞
    
    se = strel('disk', 3);
    gray_img_HS_dilated = imdilate(gray_img_HS, se);  % 膨胀操作
    gray_img_HS_closed = imerode(gray_img_HS, se);  % 腐蚀操作

    % 使用形态学重建，有效填补大空洞
    gray_img_HS_recon = imreconstruct(gray_img_HS_closed, gray_img_HS_dilated);

    
    figure;
    imshow(gray_img_HS_recon);
    title(['图像 ', image_files{i}, ' 闭运算处理后的分割结果']);





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


