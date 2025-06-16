## 移动最小乘法（MLS）

```
class MovingLeastSquares:
    """移动最小二乘法代理模型"""
    
    def __init__(self, weight_function='gaussian', bandwidth='auto', polynomial_degree=2):
        """
        初始化MLS模型
        
        Parameters:
        weight_function: 权重函数类型 ('gaussian', 'exponential', 'cubic', 'quartic')
        bandwidth: 带宽参数，'auto'表示自动选择
        polynomial_degree: 多项式阶数
        """
        self.weight_function = weight_function
        self.bandwidth = bandwidth
        self.polynomial_degree = polynomial_degree
        self.X_train = None
        self.y_train = None
        self.h = None  # 带宽参数
        
    def _weight_func(self, r):
        """计算权重函数值"""
        if self.weight_function == 'gaussian':
            return np.exp(-0.5 * (r / self.h) ** 2)
        elif self.weight_function == 'exponential':
            return np.exp(-r / self.h)
        elif self.weight_function == 'cubic':
            r_norm = r / self.h
            w = np.zeros_like(r_norm)
            mask = r_norm <= 1
            w[mask] = (1 - r_norm[mask]) ** 3
            return w
        elif self.weight_function == 'quartic':
            r_norm = r / self.h
            w = np.zeros_like(r_norm)
            mask = r_norm <= 1
            w[mask] = (1 - r_norm[mask]) ** 4 * (4 * r_norm[mask] + 1)
            return w
        else:
            raise ValueError(f"未知的权重函数: {self.weight_function}")
    
    def _polynomial_basis(self, x, x_center):
        """构造多项式基函数"""
        dx = x - x_center
        if dx.ndim == 1:
            dx = dx.reshape(1, -1)
        
        n_samples, n_features = dx.shape
        
        if self.polynomial_degree == 1:
            # 线性基函数: [1, dx1, dx2, ...]
            basis = np.ones((n_samples, n_features + 1))
            basis[:, 1:] = dx
        elif self.polynomial_degree == 2:
            # 二次基函数: [1, dx1, dx2, ..., dx1^2, dx2^2, ..., dx1*dx2, ...]
            n_basis = 1 + n_features + n_features + (n_features * (n_features - 1)) // 2
            basis = np.ones((n_samples, n_basis))
            
            # 线性项
            basis[:, 1:n_features+1] = dx
            
            # 二次项
            idx = n_features + 1
            for i in range(n_features):
                basis[:, idx] = dx[:, i] ** 2
                idx += 1
            
            # 交叉项
            for i in range(n_features):
                for j in range(i+1, n_features):
                    basis[:, idx] = dx[:, i] * dx[:, j]
                    idx += 1
        else:
            raise ValueError(f"不支持的多项式阶数: {self.polynomial_degree}")
        
        return basis
    
    def _estimate_bandwidth(self):
        """自动估计带宽参数"""
        if self.X_train is None:
            raise ValueError("需要先设置训练数据")
        
        # 使用k近邻距离的中位数作为带宽
        k = min(max(3, self.X_train.shape[1] + 1), self.X_train.shape[0] // 2)
        distances = cdist(self.X_train, self.X_train)
        np.fill_diagonal(distances, np.inf)  # 排除自身距离
        
        knn_distances = np.partition(distances, k-1, axis=1)[:, :k]
        median_distance = np.median(knn_distances)
        
        return median_distance * 1.5  # 稍微放大一些
    
    def fit(self, X, y):
        """训练MLS模型"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        if self.bandwidth == 'auto':
            self.h = self._estimate_bandwidth()
        else:
            self.h = self.bandwidth
            
        return self
    
    def predict(self, X_test):
        """预测单点或多点"""
        X_test = np.array(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        n_test = X_test.shape[0]
        predictions = np.zeros(n_test)
        uncertainties = np.zeros(n_test)
        
        for i in range(n_test):
            x_query = X_test[i, :]
            pred, unc = self._predict_single_point(x_query)
            predictions[i] = pred
            uncertainties[i] = unc
        
        return predictions, uncertainties
    
    def _predict_single_point(self, x_query):
        """预测单个点"""
        # 计算距离和权重
        distances = np.sqrt(np.sum((self.X_train - x_query) ** 2, axis=1))
        weights = self._weight_func(distances)
        
        # 过滤掉权重过小的点
        valid_mask = weights > 1e-10
        if np.sum(valid_mask) < 3:  # 如果有效点太少，使用更大的带宽
            self.h *= 2
            weights = self._weight_func(distances)
            valid_mask = weights > 1e-10
        
        X_valid = self.X_train[valid_mask]
        y_valid = self.y_train[valid_mask]
        w_valid = weights[valid_mask]
        
        if len(X_valid) == 0:
            return np.mean(self.y_train), np.std(self.y_train)
        
        try:
            # 构造多项式基函数矩阵
            P = self._polynomial_basis(X_valid, x_query)
            
            # 加权最小二乘求解
            W = np.diag(w_valid)
            A = P.T @ W @ P
            b = P.T @ W @ y_valid
            
            # 正则化以避免数值不稳定
            reg_param = 1e-8 * np.trace(A) / A.shape[0]
            A += reg_param * np.eye(A.shape[0])
            
            # 求解系数
            coeffs = np.linalg.solve(A, b)
            
            # 在查询点处的预测值（多项式基函数在中心点的值）
            p_query = self._polynomial_basis(x_query.reshape(1, -1), x_query)
            prediction = p_query @ coeffs
            prediction = prediction[0]
            
            # 不确定性估计：基于加权残差
            residuals = y_valid - (P @ coeffs)
            weighted_mse = np.sum(w_valid * residuals ** 2) / np.sum(w_valid)
            
            # 考虑局部数据密度的不确定性
            local_density = np.sum(w_valid) / len(self.X_train)
            uncertainty = np.sqrt(weighted_mse) * (1 + 1 / (1 + local_density))
            
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用加权平均
            prediction = np.sum(w_valid * y_valid) / np.sum(w_valid)
            uncertainty = np.sqrt(np.sum(w_valid * (y_valid - prediction) ** 2) / np.sum(w_valid))
        
        return prediction, uncertainty
```

- 主动贝叶斯推断
信息增益采样、最大化后验提升、主动控制资源采集数据等
- **变分贝叶斯推断**

- 贝叶斯神经网络
- 高斯过程
1. 生成模型
2. 使用协方差（半变异）作为数据点之间的相关关系
3. 先验的信念会使得任何采样函数都经过采样点


