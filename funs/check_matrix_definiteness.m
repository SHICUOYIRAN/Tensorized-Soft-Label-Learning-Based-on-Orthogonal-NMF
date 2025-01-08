function matrix_type = check_matrix_definiteness(A)
    % 输入:
    % A - 待检测的矩阵
    % 输出:
    % matrix_type - 矩阵类型: 'positive_definite', 'positive_semidefinite', 'negative_definite', 'indefinite'

    % 检查矩阵是否为对称矩阵
    if ~isequal(A, A')
        error('矩阵必须为对称矩阵');
    end

    % 计算矩阵的特征值
    eigenvalues = eig(A);

    % 根据特征值判断矩阵的类型
    num_positive = sum(eigenvalues > 0);
    num_negative = sum(eigenvalues < 0);
    num_zero = sum(eigenvalues == 0);
    n = length(eigenvalues);

    if num_positive == n
        matrix_type = 'positive_definite';
    elseif num_positive + num_zero == n
        matrix_type = 'positive_semidefinite';
    elseif num_negative == n
        matrix_type = 'negative_definite';
    else
        matrix_type = 'indefinite';
    end
end
