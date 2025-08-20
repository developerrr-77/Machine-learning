"""
Model evaluation metrics for demand forecasting.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging


class ModelEvaluator:
    """Comprehensive model evaluation for demand forecasting."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        multioutput: str = 'uniform_average'
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            multioutput: How to handle multioutput ('uniform_average', 'raw_values')
            
        Returns:
            Dict containing various metrics
        """
        # Ensure arrays are numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            self.logger.warning("No valid predictions to evaluate")
            return {}
        
        metrics = {}
        
        try:
            # Basic regression metrics
            metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
            metrics['mse'] = mean_squared_error(y_true_clean, y_pred_clean)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true_clean, y_pred_clean)
            
            # Mean Absolute Percentage Error (MAPE)
            metrics['mape'] = self._calculate_mape(y_true_clean, y_pred_clean)
            
            # Symmetric Mean Absolute Percentage Error (SMAPE)
            metrics['smape'] = self._calculate_smape(y_true_clean, y_pred_clean)
            
            # Mean Absolute Scaled Error (MASE)
            metrics['mase'] = self._calculate_mase(y_true_clean, y_pred_clean)
            
            # Weighted Mean Absolute Percentage Error (WMAPE)
            metrics['wmape'] = self._calculate_wmape(y_true_clean, y_pred_clean)
            
            # Bias and directional accuracy
            metrics['bias'] = np.mean(y_pred_clean - y_true_clean)
            metrics['directional_accuracy'] = self._calculate_directional_accuracy(
                y_true_clean, y_pred_clean
            )
            
            # Demand-specific metrics
            metrics['forecast_accuracy'] = 100 - metrics['mape']  # Forecast accuracy %
            metrics['demand_coverage'] = self._calculate_demand_coverage(
                y_true_clean, y_pred_clean
            )
            
            # Statistical significance tests
            metrics.update(self._calculate_statistical_tests(y_true_clean, y_pred_clean))
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            
        return metrics
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        
        y_true_nonzero = y_true[mask]
        y_pred_nonzero = y_pred[mask]
        
        mape = np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100
        return float(mape)
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        # Avoid division by zero
        mask = denominator != 0
        if not np.any(mask):
            return 0.0
        
        smape = np.mean(
            np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
        ) * 100
        
        return float(smape)
    
    def _calculate_mase(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Scaled Error."""
        # Calculate naive forecast MAE (using seasonal naive)
        if len(y_true) < 2:
            return np.inf
        
        # Simple naive forecast (previous value)
        naive_forecast = np.roll(y_true, 1)[1:]
        actual_values = y_true[1:]
        
        naive_mae = np.mean(np.abs(actual_values - naive_forecast))
        
        if naive_mae == 0:
            return np.inf
        
        model_mae = np.mean(np.abs(y_true - y_pred))
        mase = model_mae / naive_mae
        
        return float(mase)
    
    def _calculate_wmape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Weighted Mean Absolute Percentage Error."""
        total_actual = np.sum(np.abs(y_true))
        
        if total_actual == 0:
            return np.inf
        
        wmape = np.sum(np.abs(y_true - y_pred)) / total_actual * 100
        return float(wmape)
    
    def _calculate_directional_accuracy(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> float:
        """Calculate directional accuracy (correct trend prediction)."""
        if len(y_true) < 2:
            return 0.0
        
        # Calculate differences (trends)
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        
        # Count correct directions
        correct_directions = np.sum(np.sign(true_diff) == np.sign(pred_diff))
        total_periods = len(true_diff)
        
        if total_periods == 0:
            return 0.0
        
        accuracy = correct_directions / total_periods * 100
        return float(accuracy)
    
    def _calculate_demand_coverage(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> float:
        """Calculate demand coverage (how well forecast covers actual demand)."""
        total_actual = np.sum(y_true)
        total_predicted = np.sum(y_pred)
        
        if total_actual == 0:
            return 0.0 if total_predicted == 0 else -100.0
        
        coverage = min(total_predicted / total_actual, 2.0) * 100  # Cap at 200%
        return float(coverage)
    
    def _calculate_statistical_tests(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate statistical significance tests."""
        metrics = {}
        
        try:
            from scipy import stats
            
            # Residual analysis
            residuals = y_true - y_pred
            
            # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for large)
            if len(residuals) <= 5000:
                stat, p_value = stats.shapiro(residuals)
                metrics['normality_test_stat'] = float(stat)
                metrics['normality_test_p_value'] = float(p_value)
            
            # Autocorrelation in residuals (Durbin-Watson test)
            if len(residuals) > 10:
                dw_stat = self._durbin_watson_stat(residuals)
                metrics['durbin_watson_stat'] = float(dw_stat)
            
            # Correlation between actual and predicted
            correlation, p_value = stats.pearsonr(y_true, y_pred)
            metrics['correlation'] = float(correlation)
            metrics['correlation_p_value'] = float(p_value)
            
        except ImportError:
            self.logger.warning("scipy not available for statistical tests")
        except Exception as e:
            self.logger.warning(f"Error in statistical tests: {str(e)}")
            
        return metrics
    
    def _durbin_watson_stat(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation."""
        diff_residuals = np.diff(residuals)
        dw = np.sum(diff_residuals**2) / np.sum(residuals**2)
        return dw
    
    def calculate_product_metrics(
        self,
        df_results: pd.DataFrame,
        group_by: str = 'product_id'
    ) -> pd.DataFrame:
        """
        Calculate metrics grouped by product or other dimensions.
        
        Args:
            df_results: DataFrame with actual and predicted values
            group_by: Column to group by
            
        Returns:
            DataFrame with metrics per group
        """
        if 'actual' not in df_results.columns or 'predicted' not in df_results.columns:
            raise ValueError("DataFrame must contain 'actual' and 'predicted' columns")
        
        metrics_list = []
        
        for group_value, group_data in df_results.groupby(group_by):
            y_true = group_data['actual'].values
            y_pred = group_data['predicted'].values
            
            metrics = self.calculate_metrics(y_true, y_pred)
            metrics[group_by] = group_value
            metrics['n_observations'] = len(y_true)
            
            metrics_list.append(metrics)
        
        return pd.DataFrame(metrics_list)
    
    def calculate_time_series_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        seasonal_period: int = 7
    ) -> Dict[str, float]:
        """
        Calculate time series specific metrics.
        
        Args:
            y_true: Actual time series values
            y_pred: Predicted time series values
            seasonal_period: Period for seasonal metrics (e.g., 7 for weekly)
            
        Returns:
            Dict containing time series metrics
        """
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Seasonal naive forecast for comparison
        if len(y_true) > seasonal_period:
            seasonal_naive = np.roll(y_true, seasonal_period)
            seasonal_naive[:seasonal_period] = y_true[:seasonal_period]
            
            seasonal_mae = mean_absolute_error(y_true, seasonal_naive)
            model_mae = metrics.get('mae', np.inf)
            
            if seasonal_mae > 0:
                metrics['seasonal_mase'] = model_mae / seasonal_mae
            else:
                metrics['seasonal_mase'] = np.inf
        
        # Trend accuracy
        if len(y_true) > 2:
            metrics['trend_accuracy'] = self._calculate_trend_accuracy(y_true, y_pred)
        
        # Volatility comparison
        true_volatility = np.std(y_true)
        pred_volatility = np.std(y_pred)
        
        if true_volatility > 0:
            metrics['volatility_ratio'] = pred_volatility / true_volatility
        else:
            metrics['volatility_ratio'] = np.inf
        
        return metrics
    
    def _calculate_trend_accuracy(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> float:
        """Calculate accuracy of trend prediction over longer periods."""
        # Use moving averages to smooth out short-term fluctuations
        window = min(7, len(y_true) // 4)  # Adaptive window size
        
        if window < 2:
            return self._calculate_directional_accuracy(y_true, y_pred)
        
        # Calculate moving averages
        true_ma = np.convolve(y_true, np.ones(window)/window, mode='valid')
        pred_ma = np.convolve(y_pred, np.ones(window)/window, mode='valid')
        
        return self._calculate_directional_accuracy(true_ma, pred_ma)
    
    def generate_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            String containing formatted evaluation report
        """
        metrics = self.calculate_metrics(y_true, y_pred)
        
        report = f"\n{'='*50}\n"
        report += f"EVALUATION REPORT: {model_name}\n"
        report += f"{'='*50}\n"
        report += f"Sample Size: {len(y_true)}\n"
        report += f"Actual Range: [{np.min(y_true):.2f}, {np.max(y_true):.2f}]\n"
        report += f"Predicted Range: [{np.min(y_pred):.2f}, {np.max(y_pred):.2f}]\n"
        report += f"\nACCURACY METRICS:\n"
        report += f"{'─'*20}\n"
        
        # Format key metrics
        key_metrics = [
            ('MAE', 'mae', '.3f'),
            ('RMSE', 'rmse', '.3f'),
            ('MAPE (%)', 'mape', '.2f'),
            ('SMAPE (%)', 'smape', '.2f'),
            ('WMAPE (%)', 'wmape', '.2f'),
            ('R² Score', 'r2', '.4f'),
            ('Forecast Accuracy (%)', 'forecast_accuracy', '.2f'),
            ('Demand Coverage (%)', 'demand_coverage', '.2f'),
            ('Directional Accuracy (%)', 'directional_accuracy', '.2f')
        ]
        
        for display_name, metric_key, fmt in key_metrics:
            if metric_key in metrics:
                value = metrics[metric_key]
                if np.isfinite(value):
                    report += f"{display_name:25s}: {value:{fmt}}\n"
                else:
                    report += f"{display_name:25s}: N/A\n"
        
        # Add interpretation
        report += f"\nINTERPRETATION:\n"
        report += f"{'─'*15}\n"
        
        mape = metrics.get('mape', np.inf)
        if mape < 10:
            accuracy_level = "Excellent"
        elif mape < 20:
            accuracy_level = "Good"
        elif mape < 30:
            accuracy_level = "Reasonable"
        else:
            accuracy_level = "Poor"
        
        report += f"Overall Accuracy: {accuracy_level}\n"
        
        # Bias analysis
        bias = metrics.get('bias', 0)
        if abs(bias) < 0.1 * np.mean(y_true):
            bias_level = "Low bias (well-calibrated)"
        elif bias > 0:
            bias_level = "Positive bias (over-forecasting)"
        else:
            bias_level = "Negative bias (under-forecasting)"
        
        report += f"Bias Assessment: {bias_level}\n"
        
        # R² interpretation
        r2 = metrics.get('r2', -np.inf)
        if r2 > 0.9:
            r2_level = "Excellent fit"
        elif r2 > 0.7:
            r2_level = "Good fit"
        elif r2 > 0.5:
            r2_level = "Moderate fit"
        else:
            r2_level = "Poor fit"
        
        report += f"Model Fit: {r2_level}\n"
        report += f"{'='*50}\n"
        
        return report
    
    def benchmark_against_baselines(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        seasonal_period: int = 7
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark model against simple baseline methods.
        
        Args:
            y_true: Actual values
            y_pred: Model predictions
            seasonal_period: Seasonal period for naive forecasts
            
        Returns:
            Dict containing metrics for model and baselines
        """
        results = {}
        
        # Model metrics
        results['model'] = self.calculate_metrics(y_true, y_pred)
        
        # Mean baseline
        mean_baseline = np.full_like(y_true, np.mean(y_true))
        results['mean_baseline'] = self.calculate_metrics(y_true, mean_baseline)
        
        # Last value baseline
        if len(y_true) > 1:
            last_value_baseline = np.full_like(y_true, y_true[-1])
            results['last_value_baseline'] = self.calculate_metrics(y_true, last_value_baseline)
        
        # Seasonal naive baseline
        if len(y_true) > seasonal_period:
            seasonal_naive = np.roll(y_true, seasonal_period)
            seasonal_naive[:seasonal_period] = y_true[:seasonal_period]
            results['seasonal_naive'] = self.calculate_metrics(y_true, seasonal_naive)
        
        # Linear trend baseline
        if len(y_true) > 2:
            x = np.arange(len(y_true))
            slope, intercept = np.polyfit(x, y_true, 1)
            trend_baseline = slope * x + intercept
            results['linear_trend'] = self.calculate_metrics(y_true, trend_baseline)
        
        return results