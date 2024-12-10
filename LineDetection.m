% 使用稳健线性拟合函数（支持迭代剔除离群点和中位数绝对偏差加权）
function final_coeffs = LineDetection(x, y)
    % 初始化
    max_iterations = 100;  % 最大迭代次数
    threshold_factor = 2;  % 离群点剔除阈值因子
    prev_inliers = false(size(x));  % 上一次的内点集
    final_coeffs = [0, 0];  % 初始拟合参数

    for iter = 1:max_iterations
        % 初始最小二乘拟合
        A = [x(:), ones(size(x(:)))];  % 设计矩阵
        beta = A \ y(:);  % 常规最小二乘解

        % 计算残差
        residuals = y(:) - A * beta;

        % 计算误差中位数和阈值
        med_residual = median(abs(residuals));
        threshold = threshold_factor * med_residual;

        % 标记内点
        inliers = abs(residuals) <= threshold;

        % 如果内点集未发生变化，终止迭代
        if isequal(prev_inliers, inliers)
            % 使用中位数绝对偏差加权对最终内点集拟合
            coeffs = RobustFit(x(inliers), y(inliers));
            final_coeffs = coeffs;
            break;
        end

        % 更新内点集
        prev_inliers = inliers;
        x = x(inliers);
        y = y(inliers);

        % 如果内点数不足，终止迭代
        if numel(x) < 2
            warning('内点不足，无法拟合');
            break;
        end
    end
end

% 使用中位数绝对偏差的加权最小二乘拟合
function coeffs = RobustFit(x, y)
    % 使用中位数绝对偏差(MAD)实现稳健加权拟合
    % 输入: x, y 为数据点
    % 输出: coeffs = [a, b] (y = ax + b)

    % 初始拟合
    A = [x(:), ones(size(x(:)))];  % 设计矩阵
    beta = A \ y(:);  % 常规最小二乘解

    % 计算残差
    residuals = y(:) - A * beta;

    % 计算MAD并更新权重
    MAD = median(abs(residuals - median(residuals)));
    weights = 1 ./ (1 + (abs(residuals) / (6 * MAD)).^2);  % Tukey's biweight

    % 带权最小二乘拟合
    W = diag(weights);  % 权重矩阵
    beta = (A' * W * A) \ (A' * W * y(:));  % 带权最小二乘解

    coeffs = beta';  % 返回拟合参数
end
