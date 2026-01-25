"""Great Expectations integration for advanced data validation.

This module provides enhanced data validation using Great Expectations,
building on the existing DataValidator functionality.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

try:
    import great_expectations as gx
    from great_expectations.core.expectation_suite import ExpectationSuite
    from great_expectations.checkpoint import Checkpoint
    GX_AVAILABLE = True
except ImportError:
    GX_AVAILABLE = False


@dataclass
class ExpectationResult:
    """Result from a single expectation check."""
    expectation_type: str
    success: bool
    column: str | None = None
    observed_value: Any = None
    expected_value: Any = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expectation_type": self.expectation_type,
            "success": self.success,
            "column": self.column,
            "observed_value": self.observed_value,
            "expected_value": self.expected_value,
            "details": self.details,
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    suite_name: str
    success: bool
    total_expectations: int
    successful_expectations: int
    failed_expectations: int
    results: list[ExpectationResult]
    run_time: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "success": self.success,
            "total_expectations": self.total_expectations,
            "successful_expectations": self.successful_expectations,
            "failed_expectations": self.failed_expectations,
            "success_rate": self.successful_expectations / max(self.total_expectations, 1),
            "results": [r.to_dict() for r in self.results],
            "run_time": self.run_time,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def get_failures(self) -> list[ExpectationResult]:
        """Get only failed expectations."""
        return [r for r in self.results if not r.success]


class GreatExpectationsValidator:
    """Advanced data validator using Great Expectations.

    This class provides a high-level interface for data validation
    with Great Expectations, including predefined expectation suites
    for common use cases like fraud detection.

    Attributes:
        context_root: Root directory for GX context
        use_gx: Whether to use Great Expectations (if available)
    """

    # Fraud detection columns and their expected properties
    FRAUD_SCHEMA = {
        "transaction_id": {"dtype": "object", "nullable": False, "unique": True},
        "amount": {"dtype": "float64", "nullable": False, "min": 0, "max": 100000},
        "timestamp": {"dtype": "datetime64[ns]", "nullable": False},
        "merchant_category": {"dtype": "object", "nullable": False},
        "location": {"dtype": "object", "nullable": False},
        "is_fraud": {"dtype": "int64", "nullable": False, "values": [0, 1]},
    }

    def __init__(
        self,
        context_root: str | Path = "great_expectations",
        use_gx: bool = True,
    ):
        """Initialize the validator.

        Args:
            context_root: Root directory for GX context and expectations
            use_gx: Whether to use Great Expectations (if available)
        """
        self.context_root = Path(context_root)
        self.context_root.mkdir(parents=True, exist_ok=True)

        self.use_gx = use_gx and GX_AVAILABLE
        self._expectation_suites: dict[str, list[dict]] = {}

        # Initialize GX context if available
        self._gx_context = None
        if self.use_gx:
            self._init_gx_context()

        # Load existing suites
        self._load_suites()

    def _init_gx_context(self) -> None:
        """Initialize Great Expectations context."""
        try:
            self._gx_context = gx.get_context(
                context_root_dir=str(self.context_root)
            )
        except Exception:
            self.use_gx = False

    def _load_suites(self) -> None:
        """Load expectation suites from disk."""
        suites_dir = self.context_root / "expectations"
        if suites_dir.exists():
            for suite_file in suites_dir.glob("*.json"):
                suite_name = suite_file.stem
                with open(suite_file, "r", encoding="utf-8") as f:
                    self._expectation_suites[suite_name] = json.load(f)

    def _save_suite(self, name: str) -> None:
        """Save an expectation suite to disk."""
        suites_dir = self.context_root / "expectations"
        suites_dir.mkdir(parents=True, exist_ok=True)

        suite_file = suites_dir / f"{name}.json"
        with open(suite_file, "w", encoding="utf-8") as f:
            json.dump(self._expectation_suites[name], f, indent=2)

    def create_expectation_suite(
        self,
        name: str,
        expectations: list[dict[str, Any]],
    ) -> None:
        """Create a new expectation suite.

        Args:
            name: Name of the suite
            expectations: List of expectation definitions
        """
        self._expectation_suites[name] = expectations
        self._save_suite(name)

    def create_fraud_detection_suite(self) -> None:
        """Create a predefined suite for fraud detection data."""
        expectations = [
            # Column existence
            {
                "type": "expect_column_to_exist",
                "column": "transaction_id",
            },
            {
                "type": "expect_column_to_exist",
                "column": "amount",
            },
            {
                "type": "expect_column_to_exist",
                "column": "is_fraud",
            },
            # Data types
            {
                "type": "expect_column_values_to_be_of_type",
                "column": "amount",
                "expected_type": "float64",
            },
            {
                "type": "expect_column_values_to_be_of_type",
                "column": "is_fraud",
                "expected_type": "int64",
            },
            # Null checks
            {
                "type": "expect_column_values_to_not_be_null",
                "column": "transaction_id",
            },
            {
                "type": "expect_column_values_to_not_be_null",
                "column": "amount",
            },
            {
                "type": "expect_column_values_to_not_be_null",
                "column": "is_fraud",
            },
            # Uniqueness
            {
                "type": "expect_column_values_to_be_unique",
                "column": "transaction_id",
            },
            # Value ranges
            {
                "type": "expect_column_values_to_be_between",
                "column": "amount",
                "min_value": 0,
                "max_value": 100000,
            },
            {
                "type": "expect_column_values_to_be_in_set",
                "column": "is_fraud",
                "value_set": [0, 1],
            },
            # Distribution checks
            {
                "type": "expect_column_mean_to_be_between",
                "column": "amount",
                "min_value": 10,
                "max_value": 5000,
            },
            # Class balance (fraud should be < 20%)
            {
                "type": "expect_column_proportion_of_unique_values_to_be_between",
                "column": "is_fraud",
                "min_value": 0.01,  # At least 1% fraud
                "max_value": 0.5,   # At most 50% fraud
            },
        ]

        self.create_expectation_suite("fraud_detection", expectations)

    def validate(
        self,
        df: pd.DataFrame,
        suite_name: str,
    ) -> ValidationReport:
        """Validate a DataFrame against an expectation suite.

        Args:
            df: DataFrame to validate
            suite_name: Name of the expectation suite to use

        Returns:
            ValidationReport with detailed results
        """
        if suite_name not in self._expectation_suites:
            raise ValueError(f"Expectation suite '{suite_name}' not found")

        start_time = datetime.now()
        expectations = self._expectation_suites[suite_name]
        results = []

        for exp in expectations:
            result = self._evaluate_expectation(df, exp)
            results.append(result)

        end_time = datetime.now()
        run_time = (end_time - start_time).total_seconds()

        successful = sum(1 for r in results if r.success)

        return ValidationReport(
            suite_name=suite_name,
            success=all(r.success for r in results),
            total_expectations=len(results),
            successful_expectations=successful,
            failed_expectations=len(results) - successful,
            results=results,
            run_time=run_time,
            metadata={
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
            },
        )

    def _evaluate_expectation(
        self,
        df: pd.DataFrame,
        expectation: dict[str, Any],
    ) -> ExpectationResult:
        """Evaluate a single expectation.

        Args:
            df: DataFrame to validate
            expectation: Expectation definition

        Returns:
            ExpectationResult
        """
        exp_type = expectation["type"]
        column = expectation.get("column")

        try:
            if exp_type == "expect_column_to_exist":
                success = column in df.columns
                return ExpectationResult(
                    expectation_type=exp_type,
                    success=success,
                    column=column,
                    observed_value=column in df.columns,
                    expected_value=True,
                )

            if exp_type == "expect_column_values_to_be_of_type":
                if column not in df.columns:
                    return ExpectationResult(
                        expectation_type=exp_type,
                        success=False,
                        column=column,
                        details={"error": "Column not found"},
                    )

                expected_type = expectation["expected_type"]
                actual_type = str(df[column].dtype)
                success = actual_type == expected_type or expected_type in actual_type

                return ExpectationResult(
                    expectation_type=exp_type,
                    success=success,
                    column=column,
                    observed_value=actual_type,
                    expected_value=expected_type,
                )

            if exp_type == "expect_column_values_to_not_be_null":
                if column not in df.columns:
                    return ExpectationResult(
                        expectation_type=exp_type,
                        success=False,
                        column=column,
                        details={"error": "Column not found"},
                    )

                null_count = df[column].isnull().sum()
                success = null_count == 0

                return ExpectationResult(
                    expectation_type=exp_type,
                    success=success,
                    column=column,
                    observed_value=int(null_count),
                    expected_value=0,
                    details={"null_percentage": null_count / len(df) * 100},
                )

            if exp_type == "expect_column_values_to_be_unique":
                if column not in df.columns:
                    return ExpectationResult(
                        expectation_type=exp_type,
                        success=False,
                        column=column,
                        details={"error": "Column not found"},
                    )

                duplicate_count = df[column].duplicated().sum()
                success = duplicate_count == 0

                return ExpectationResult(
                    expectation_type=exp_type,
                    success=success,
                    column=column,
                    observed_value=int(duplicate_count),
                    expected_value=0,
                    details={"unique_count": df[column].nunique()},
                )

            if exp_type == "expect_column_values_to_be_between":
                if column not in df.columns:
                    return ExpectationResult(
                        expectation_type=exp_type,
                        success=False,
                        column=column,
                        details={"error": "Column not found"},
                    )

                min_val = expectation.get("min_value", float("-inf"))
                max_val = expectation.get("max_value", float("inf"))

                col = df[column].dropna()
                below_min = (col < min_val).sum()
                above_max = (col > max_val).sum()
                success = below_min == 0 and above_max == 0

                return ExpectationResult(
                    expectation_type=exp_type,
                    success=success,
                    column=column,
                    observed_value={"min": float(col.min()), "max": float(col.max())},
                    expected_value={"min": min_val, "max": max_val},
                    details={
                        "below_min_count": int(below_min),
                        "above_max_count": int(above_max),
                    },
                )

            if exp_type == "expect_column_values_to_be_in_set":
                if column not in df.columns:
                    return ExpectationResult(
                        expectation_type=exp_type,
                        success=False,
                        column=column,
                        details={"error": "Column not found"},
                    )

                value_set = set(expectation["value_set"])
                actual_values = set(df[column].dropna().unique())
                invalid_values = actual_values - value_set
                success = len(invalid_values) == 0

                return ExpectationResult(
                    expectation_type=exp_type,
                    success=success,
                    column=column,
                    observed_value=list(actual_values),
                    expected_value=list(value_set),
                    details={"invalid_values": list(invalid_values)},
                )

            if exp_type == "expect_column_mean_to_be_between":
                if column not in df.columns:
                    return ExpectationResult(
                        expectation_type=exp_type,
                        success=False,
                        column=column,
                        details={"error": "Column not found"},
                    )

                min_val = expectation.get("min_value", float("-inf"))
                max_val = expectation.get("max_value", float("inf"))

                mean_val = df[column].mean()
                success = min_val <= mean_val <= max_val

                return ExpectationResult(
                    expectation_type=exp_type,
                    success=success,
                    column=column,
                    observed_value=float(mean_val),
                    expected_value={"min": min_val, "max": max_val},
                )

            if exp_type == "expect_column_proportion_of_unique_values_to_be_between":
                if column not in df.columns:
                    return ExpectationResult(
                        expectation_type=exp_type,
                        success=False,
                        column=column,
                        details={"error": "Column not found"},
                    )

                min_val = expectation.get("min_value", 0)
                max_val = expectation.get("max_value", 1)

                # For binary classification, check minority class proportion
                value_counts = df[column].value_counts(normalize=True)
                if len(value_counts) >= 2:
                    minority_prop = value_counts.min()
                else:
                    minority_prop = 0

                success = min_val <= minority_prop <= max_val

                return ExpectationResult(
                    expectation_type=exp_type,
                    success=success,
                    column=column,
                    observed_value=float(minority_prop),
                    expected_value={"min": min_val, "max": max_val},
                    details={"value_distribution": value_counts.to_dict()},
                )

            # Unknown expectation type
            return ExpectationResult(
                expectation_type=exp_type,
                success=False,
                column=column,
                details={"error": f"Unknown expectation type: {exp_type}"},
            )

        except Exception as e:
            return ExpectationResult(
                expectation_type=exp_type,
                success=False,
                column=column,
                details={"error": str(e)},
            )

    def validate_schema(
        self,
        df: pd.DataFrame,
        schema: dict[str, dict[str, Any]] | None = None,
    ) -> ValidationReport:
        """Validate DataFrame against a schema.

        Args:
            df: DataFrame to validate
            schema: Schema definition (uses FRAUD_SCHEMA by default)

        Returns:
            ValidationReport
        """
        if schema is None:
            schema = self.FRAUD_SCHEMA

        # Build expectations from schema
        expectations = []
        for column, props in schema.items():
            # Column existence
            expectations.append({
                "type": "expect_column_to_exist",
                "column": column,
            })

            # Data type
            if "dtype" in props:
                expectations.append({
                    "type": "expect_column_values_to_be_of_type",
                    "column": column,
                    "expected_type": props["dtype"],
                })

            # Nullability
            if not props.get("nullable", True):
                expectations.append({
                    "type": "expect_column_values_to_not_be_null",
                    "column": column,
                })

            # Uniqueness
            if props.get("unique", False):
                expectations.append({
                    "type": "expect_column_values_to_be_unique",
                    "column": column,
                })

            # Value range
            if "min" in props or "max" in props:
                expectations.append({
                    "type": "expect_column_values_to_be_between",
                    "column": column,
                    "min_value": props.get("min"),
                    "max_value": props.get("max"),
                })

            # Value set
            if "values" in props:
                expectations.append({
                    "type": "expect_column_values_to_be_in_set",
                    "column": column,
                    "value_set": props["values"],
                })

        # Create temporary suite and validate
        self._expectation_suites["_schema_validation"] = expectations
        return self.validate(df, "_schema_validation")

    def generate_data_profile(
        self,
        df: pd.DataFrame,
        name: str = "data_profile",
    ) -> dict[str, Any]:
        """Generate a data profile with suggested expectations.

        Args:
            df: DataFrame to profile
            name: Name for the profile

        Returns:
            Data profile with statistics and suggested expectations
        """
        profile = {
            "name": name,
            "generated_at": datetime.now().isoformat(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": {},
            "suggested_expectations": [],
        }

        for column in df.columns:
            col = df[column]
            col_profile = {
                "dtype": str(col.dtype),
                "null_count": int(col.isnull().sum()),
                "null_percentage": float(col.isnull().mean() * 100),
                "unique_count": int(col.nunique()),
            }

            if pd.api.types.is_numeric_dtype(col):
                col_profile.update({
                    "mean": float(col.mean()),
                    "std": float(col.std()),
                    "min": float(col.min()),
                    "max": float(col.max()),
                    "p25": float(col.quantile(0.25)),
                    "p50": float(col.quantile(0.50)),
                    "p75": float(col.quantile(0.75)),
                })

                # Suggest expectations
                profile["suggested_expectations"].extend([
                    {
                        "type": "expect_column_values_to_be_between",
                        "column": column,
                        "min_value": float(col.min()),
                        "max_value": float(col.max()),
                    },
                    {
                        "type": "expect_column_mean_to_be_between",
                        "column": column,
                        "min_value": float(col.mean() - 2 * col.std()),
                        "max_value": float(col.mean() + 2 * col.std()),
                    },
                ])
            else:
                col_profile["top_values"] = col.value_counts().head(10).to_dict()

            if col.isnull().sum() == 0:
                profile["suggested_expectations"].append({
                    "type": "expect_column_values_to_not_be_null",
                    "column": column,
                })

            profile["columns"][column] = col_profile

        return profile

    def list_suites(self) -> list[str]:
        """List all available expectation suites."""
        return list(self._expectation_suites.keys())

    def get_suite(self, name: str) -> list[dict[str, Any]] | None:
        """Get an expectation suite by name."""
        return self._expectation_suites.get(name)

    def delete_suite(self, name: str) -> bool:
        """Delete an expectation suite."""
        if name not in self._expectation_suites:
            return False

        del self._expectation_suites[name]

        suite_file = self.context_root / "expectations" / f"{name}.json"
        if suite_file.exists():
            suite_file.unlink()

        return True
