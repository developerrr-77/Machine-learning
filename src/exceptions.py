"""
Custom exceptions for the demand forecasting system.
"""


class DemandForecastingError(Exception):
    """Base exception for demand forecasting system."""
    pass


class DataLoadError(DemandForecastingError):
    """Raised when data loading fails."""
    pass


class ValidationError(DemandForecastingError):
    """Raised when data validation fails."""
    pass


class PreprocessingError(DemandForecastingError):
    """Raised when data preprocessing fails."""
    pass


class FeatureEngineeringError(DemandForecastingError):
    """Raised when feature engineering fails."""
    pass


class ModelTrainingError(DemandForecastingError):
    """Raised when model training fails."""
    pass


class ModelPredictionError(DemandForecastingError):
    """Raised when model prediction fails."""
    pass


class DatabaseError(DemandForecastingError):
    """Raised when database operations fail."""
    pass


class CacheError(DemandForecastingError):
    """Raised when cache operations fail."""
    pass


class ConfigurationError(DemandForecastingError):
    """Raised when configuration is invalid."""
    pass


class APIError(DemandForecastingError):
    """Raised when API operations fail."""
    pass