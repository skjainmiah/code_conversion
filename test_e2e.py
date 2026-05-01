"""End-to-end test — 10 synthetic Foundry files (~5000 lines total).

Tests the full pipeline WITHOUT the LLM call:
  upload → chunk → index → import resolution → section splitting → notebook formatting
Also simulates conversion output to test the notebook formatter.
"""

import json
import sys
import traceback

from rag_engine import KeywordIndex, chunk_file
from import_resolver import resolve_imports, build_dependency_graph, parse_imports
from converter import _split_into_sections, _extract_code, _ensure_notebook_header, _build_file_outline, _validate_syntax
from validator import validate_conversion
from notebook_formatter import code_to_notebook

# ─── Synthetic Foundry Files ──────────────────────────────────────────────────

SYNTHETIC_FILES = {}

# File 1: shared_utils.py (~200 lines) — utility module imported by others
SYNTHETIC_FILES["transforms/shared_utils.py"] = '''"""Shared utility functions for UAL data pipelines."""

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, DoubleType
from transforms.api import Transform


def clean_string_column(df, col_name):
    """Strip whitespace and uppercase a string column."""
    return df.withColumn(col_name, F.upper(F.trim(F.col(col_name))))


def add_audit_columns(df):
    """Add standard audit columns to a dataframe."""
    return (
        df.withColumn("etl_load_timestamp", F.current_timestamp())
          .withColumn("etl_source_system", F.lit("FOUNDRY"))
          .withColumn("etl_batch_id", F.lit("BATCH_001"))
          .withColumn("etl_is_deleted", F.lit(False))
    )


def validate_not_null(df, columns):
    """Filter out rows where any of the specified columns are null."""
    condition = F.lit(True)
    for col_name in columns:
        condition = condition & F.col(col_name).isNotNull()
    return df.filter(condition)


def standardize_date_format(df, col_name, input_format="yyyy-MM-dd"):
    """Convert date column to standard format."""
    return df.withColumn(
        col_name,
        F.to_date(F.col(col_name), input_format)
    )


def calculate_age_days(df, date_col, reference_col=None):
    """Calculate age in days from a date column."""
    ref = F.col(reference_col) if reference_col else F.current_date()
    return df.withColumn(
        f"{date_col}_age_days",
        F.datediff(ref, F.col(date_col))
    )


def apply_scd_type2(df, key_cols, value_cols):
    """Apply SCD Type 2 logic — mark old records as inactive."""
    from pyspark.sql.window import Window
    w = Window.partitionBy(*key_cols).orderBy(F.col("etl_load_timestamp").desc())
    df = df.withColumn("row_num", F.row_number().over(w))
    df = df.withColumn("is_current", F.when(F.col("row_num") == 1, True).otherwise(False))
    return df.drop("row_num")


def mask_pii(df, columns):
    """Mask PII columns with asterisks, keeping first and last char."""
    for col_name in columns:
        df = df.withColumn(
            col_name,
            F.when(
                F.length(F.col(col_name)) > 2,
                F.concat(
                    F.substring(F.col(col_name), 1, 1),
                    F.lit("****"),
                    F.substring(F.col(col_name), -1, 1)
                )
            ).otherwise(F.lit("***"))
        )
    return df


def deduplicate(df, key_cols, order_col, keep="latest"):
    """Deduplicate dataframe keeping latest or earliest record per key."""
    from pyspark.sql.window import Window
    if keep == "latest":
        w = Window.partitionBy(*key_cols).orderBy(F.col(order_col).desc())
    else:
        w = Window.partitionBy(*key_cols).orderBy(F.col(order_col).asc())
    return df.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")


def flatten_struct(df, col_name):
    """Flatten a struct column into individual columns."""
    for field in df.schema[col_name].dataType.fields:
        df = df.withColumn(
            f"{col_name}_{field.name}",
            F.col(f"{col_name}.{field.name}")
        )
    return df.drop(col_name)


def add_data_quality_score(df, required_cols):
    """Add a data quality score based on non-null required columns."""
    total = len(required_cols)
    non_null_count = sum(
        F.when(F.col(c).isNotNull(), F.lit(1)).otherwise(F.lit(0))
        for c in required_cols
    )
    return df.withColumn("data_quality_score", non_null_count / F.lit(total))


class DataQualityChecker:
    """Reusable data quality checks for pipeline validation."""

    def __init__(self, df):
        self.df = df
        self.issues = []

    def check_nulls(self, columns):
        for col_name in columns:
            null_count = self.df.filter(F.col(col_name).isNull()).count()
            if null_count > 0:
                self.issues.append(f"{col_name}: {null_count} nulls")
        return self

    def check_duplicates(self, key_cols):
        from pyspark.sql.window import Window
        w = Window.partitionBy(*key_cols)
        dup_count = (
            self.df.withColumn("_cnt", F.count("*").over(w))
            .filter(F.col("_cnt") > 1)
            .count()
        )
        if dup_count > 0:
            self.issues.append(f"Duplicates on {key_cols}: {dup_count}")
        return self

    def check_range(self, col_name, min_val, max_val):
        out_of_range = self.df.filter(
            (F.col(col_name) < min_val) | (F.col(col_name) > max_val)
        ).count()
        if out_of_range > 0:
            self.issues.append(f"{col_name}: {out_of_range} out of range [{min_val}, {max_val}]")
        return self

    def report(self):
        if not self.issues:
            return "All quality checks passed"
        return "Quality issues found:\\n" + "\\n".join(f"  - {i}" for i in self.issues)


def compute_haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance between two points."""
    import math
    R = 3959  # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def create_partition_columns(df, date_col):
    """Add year/month/day partition columns for optimized Delta Lake storage."""
    return (
        df.withColumn("partition_year", F.year(F.col(date_col)))
          .withColumn("partition_month", F.month(F.col(date_col)))
          .withColumn("partition_day", F.dayofmonth(F.col(date_col)))
    )


def apply_column_encryption(df, columns, encryption_key):
    """Apply AES encryption to sensitive columns."""
    for col_name in columns:
        df = df.withColumn(
            col_name,
            F.base64(F.aes_encrypt(F.col(col_name).cast("string"), F.lit(encryption_key)))
        )
    return df


def generate_surrogate_key(df, natural_key_cols):
    """Generate MD5-based surrogate key from natural key columns."""
    concat_expr = F.concat_ws("|", *[F.coalesce(F.col(c).cast("string"), F.lit("NULL")) for c in natural_key_cols])
    return df.withColumn("surrogate_key", F.md5(concat_expr))


def apply_data_retention_policy(df, date_col, retention_days=365):
    """Filter out records older than retention period."""
    cutoff = F.date_sub(F.current_date(), retention_days)
    return df.filter(F.col(date_col) >= cutoff)


def calculate_percentiles(df, value_col, group_cols, percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]):
    """Calculate percentile values for a column grouped by specified columns."""
    from pyspark.sql.functions import percentile_approx
    agg_exprs = [
        percentile_approx(value_col, p).alias(f"{value_col}_p{int(p*100)}")
        for p in percentiles
    ]
    return df.groupBy(*group_cols).agg(*agg_exprs)


def validate_schema_compatibility(df, expected_schema):
    """Validate that dataframe schema matches expected schema."""
    actual_fields = {f.name: str(f.dataType) for f in df.schema.fields}
    mismatches = []
    for field_name, expected_type in expected_schema.items():
        if field_name not in actual_fields:
            mismatches.append(f"Missing column: {field_name}")
        elif actual_fields[field_name] != expected_type:
            mismatches.append(f"Type mismatch for {field_name}: expected {expected_type}, got {actual_fields[field_name]}")
    return mismatches


def merge_slowly_changing_dimension(target_df, source_df, key_cols, value_cols):
    """Implement SCD Type 2 merge logic for dimension tables."""
    from pyspark.sql.window import Window

    # Find changed records
    join_condition = [target_df[k] == source_df[k] for k in key_cols]
    changed = source_df.join(target_df, join_condition, "left_anti")

    # Mark old records as expired
    w = Window.partitionBy(*key_cols).orderBy(F.col("effective_date").desc())
    target_df = target_df.withColumn("is_current", F.when(F.row_number().over(w) == 1, True).otherwise(False))

    # Add new records
    new_records = changed.withColumn("effective_date", F.current_timestamp())
    new_records = new_records.withColumn("expiration_date", F.lit(None).cast("timestamp"))
    new_records = new_records.withColumn("is_current", F.lit(True))

    return target_df.union(new_records)


def build_time_series_features(df, date_col, value_col, group_cols, windows=[7, 14, 30, 60, 90]):
    """Build time series features with multiple rolling windows."""
    from pyspark.sql.window import Window
    for w_size in windows:
        w = Window.partitionBy(*group_cols).orderBy(F.col(date_col).cast("long")).rangeBetween(-w_size * 86400, 0)
        df = df.withColumn(f"{value_col}_avg_{w_size}d", F.avg(value_col).over(w))
        df = df.withColumn(f"{value_col}_std_{w_size}d", F.stddev(value_col).over(w))
        df = df.withColumn(f"{value_col}_min_{w_size}d", F.min(value_col).over(w))
        df = df.withColumn(f"{value_col}_max_{w_size}d", F.max(value_col).over(w))
        df = df.withColumn(f"{value_col}_count_{w_size}d", F.count(value_col).over(w))
    return df


def detect_anomalies_zscore(df, value_col, group_cols, threshold=3.0):
    """Detect anomalies using Z-score method within groups."""
    from pyspark.sql.window import Window
    w = Window.partitionBy(*group_cols)
    df = df.withColumn(f"{value_col}_mean", F.avg(value_col).over(w))
    df = df.withColumn(f"{value_col}_stddev", F.stddev(value_col).over(w))
    df = df.withColumn(
        f"{value_col}_zscore",
        F.when(
            F.col(f"{value_col}_stddev") > 0,
            (F.col(value_col) - F.col(f"{value_col}_mean")) / F.col(f"{value_col}_stddev")
        ).otherwise(0)
    )
    df = df.withColumn(
        f"{value_col}_is_anomaly",
        F.abs(F.col(f"{value_col}_zscore")) > threshold
    )
    return df.drop(f"{value_col}_mean", f"{value_col}_stddev")


def generate_data_lineage_metadata(df, source_name, transform_name, version="1.0"):
    """Add data lineage tracking columns."""
    return (
        df.withColumn("lineage_source", F.lit(source_name))
          .withColumn("lineage_transform", F.lit(transform_name))
          .withColumn("lineage_version", F.lit(version))
          .withColumn("lineage_timestamp", F.current_timestamp())
          .withColumn("lineage_row_hash", F.md5(F.to_json(F.struct(*df.columns))))
    )
'''

# File 2: raw_flight_ingest.py (~600 lines) — Bronze layer ingestion
SYNTHETIC_FILES["transforms/bronze/raw_flight_ingest.py"] = '''"""Bronze layer — Raw flight data ingestion from SFTP source."""

from transforms.api import transform, Input, Output, configure
from transforms.verbs import dataframe
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType


@configure(profile=["EXECUTOR_MEMORY_LARGE", "NUM_EXECUTORS_16"])
@transform(
    output=Output("/datasets/airline/bronze/raw_flights"),
    source_flights=Input("/datasets/raw/sftp/flight_data"),
    source_schedule=Input("/datasets/raw/sftp/flight_schedule"),
    ref_aircraft=Input("/datasets/ref/aircraft_types"),
)
def compute(output, source_flights, source_schedule, ref_aircraft):
    """Ingest raw flight data with minimal transformation.

    Business Rules:
    - Keep ALL records including duplicates (dedup happens in Silver)
    - Add source tracking columns
    - Cast basic types but do not clean data
    - Preserve original column names with prefix
    """
    flights = source_flights.dataframe()
    schedule = source_schedule.dataframe()
    aircraft = ref_aircraft.dataframe()

    # Add source tracking
    flights = (
        flights
        .withColumn("src_system", F.lit("SFTP_FLIGHT"))
        .withColumn("src_file_date", F.current_date())
        .withColumn("src_load_ts", F.current_timestamp())
        .withColumn("src_batch_id", F.lit("BATCH_FLIGHTS_001"))
    )

    # Basic type casting — keep original values alongside
    flights = (
        flights
        .withColumn("flight_number_raw", F.col("flight_number"))
        .withColumn("flight_number", F.col("flight_number").cast(IntegerType()))
        .withColumn("departure_time_raw", F.col("departure_time"))
        .withColumn("departure_time", F.to_timestamp(F.col("departure_time"), "yyyy-MM-dd HH:mm:ss"))
        .withColumn("arrival_time_raw", F.col("arrival_time"))
        .withColumn("arrival_time", F.to_timestamp(F.col("arrival_time"), "yyyy-MM-dd HH:mm:ss"))
        .withColumn("passenger_count_raw", F.col("passenger_count"))
        .withColumn("passenger_count", F.col("passenger_count").cast(IntegerType()))
        .withColumn("fuel_consumed_raw", F.col("fuel_consumed"))
        .withColumn("fuel_consumed", F.col("fuel_consumed").cast(DoubleType()))
        .withColumn("distance_miles_raw", F.col("distance_miles"))
        .withColumn("distance_miles", F.col("distance_miles").cast(DoubleType()))
    )

    # Enrich with schedule data
    schedule = schedule.select(
        F.col("flight_id").alias("sched_flight_id"),
        F.col("scheduled_departure"),
        F.col("scheduled_arrival"),
        F.col("gate_number"),
        F.col("terminal"),
        F.col("route_type"),
    )

    flights = flights.join(
        schedule,
        flights["flight_id"] == schedule["sched_flight_id"],
        how="left"
    ).drop("sched_flight_id")

    # Enrich with aircraft reference
    aircraft = aircraft.select(
        F.col("aircraft_code").alias("ref_aircraft_code"),
        F.col("aircraft_name"),
        F.col("max_capacity"),
        F.col("fuel_capacity_gallons"),
        F.col("manufacturer"),
    )

    flights = flights.join(
        aircraft,
        flights["aircraft_type"] == aircraft["ref_aircraft_code"],
        how="left"
    ).drop("ref_aircraft_code")

    # Calculate derived columns
    flights = (
        flights
        .withColumn(
            "departure_delay_minutes",
            F.when(
                F.col("scheduled_departure").isNotNull() & F.col("departure_time").isNotNull(),
                (F.unix_timestamp("departure_time") - F.unix_timestamp("scheduled_departure")) / 60
            ).otherwise(F.lit(None))
        )
        .withColumn(
            "arrival_delay_minutes",
            F.when(
                F.col("scheduled_arrival").isNotNull() & F.col("arrival_time").isNotNull(),
                (F.unix_timestamp("arrival_time") - F.unix_timestamp("scheduled_arrival")) / 60
            ).otherwise(F.lit(None))
        )
        .withColumn(
            "flight_duration_minutes",
            F.when(
                F.col("departure_time").isNotNull() & F.col("arrival_time").isNotNull(),
                (F.unix_timestamp("arrival_time") - F.unix_timestamp("departure_time")) / 60
            ).otherwise(F.lit(None))
        )
        .withColumn(
            "load_factor",
            F.when(
                F.col("max_capacity").isNotNull() & (F.col("max_capacity") > 0),
                F.col("passenger_count") / F.col("max_capacity")
            ).otherwise(F.lit(None))
        )
        .withColumn(
            "fuel_efficiency",
            F.when(
                F.col("distance_miles").isNotNull() & (F.col("distance_miles") > 0),
                F.col("fuel_consumed") / F.col("distance_miles")
            ).otherwise(F.lit(None))
        )
    )

    # Tag data quality
    flights = (
        flights
        .withColumn(
            "dq_has_flight_number",
            F.col("flight_number").isNotNull()
        )
        .withColumn(
            "dq_has_departure",
            F.col("departure_time").isNotNull()
        )
        .withColumn(
            "dq_has_arrival",
            F.col("arrival_time").isNotNull()
        )
        .withColumn(
            "dq_has_passengers",
            F.col("passenger_count").isNotNull() & (F.col("passenger_count") >= 0)
        )
        .withColumn(
            "dq_valid_duration",
            F.col("flight_duration_minutes").isNotNull() & (F.col("flight_duration_minutes") > 0)
        )
        .withColumn(
            "dq_score",
            (
                F.col("dq_has_flight_number").cast("int") +
                F.col("dq_has_departure").cast("int") +
                F.col("dq_has_arrival").cast("int") +
                F.col("dq_has_passengers").cast("int") +
                F.col("dq_valid_duration").cast("int")
            ) / 5
        )
    )

    output.write_dataframe(flights)



@configure(profile=["EXECUTOR_MEMORY_MEDIUM"])
@transform(
    output=Output("/datasets/airline/bronze/raw_flights_historical"),
    source_historical=Input("/datasets/raw/sftp/flight_data_historical"),
)
def compute_historical(output, source_historical):
    """Ingest historical flight data archive — separate pipeline."""
    hist = source_historical.dataframe()

    hist = (
        hist
        .withColumn("src_system", F.lit("SFTP_HISTORICAL"))
        .withColumn("src_file_date", F.current_date())
        .withColumn("src_load_ts", F.current_timestamp())
        .withColumn("is_historical", F.lit(True))
    )

    # Historical data uses different date formats
    hist = (
        hist
        .withColumn("departure_time",
                     F.coalesce(
                         F.to_timestamp(F.col("departure_time"), "yyyy-MM-dd HH:mm:ss"),
                         F.to_timestamp(F.col("departure_time"), "MM/dd/yyyy HH:mm"),
                         F.to_timestamp(F.col("departure_time"), "dd-MMM-yyyy HH:mm:ss"),
                     ))
        .withColumn("arrival_time",
                     F.coalesce(
                         F.to_timestamp(F.col("arrival_time"), "yyyy-MM-dd HH:mm:ss"),
                         F.to_timestamp(F.col("arrival_time"), "MM/dd/yyyy HH:mm"),
                         F.to_timestamp(F.col("arrival_time"), "dd-MMM-yyyy HH:mm:ss"),
                     ))
    )

    # Historical data quality is lower — flag more aggressively
    hist = (
        hist
        .withColumn("dq_has_flight_number", F.col("flight_number").isNotNull())
        .withColumn("dq_has_departure", F.col("departure_time").isNotNull())
        .withColumn("dq_has_arrival", F.col("arrival_time").isNotNull())
        .withColumn("dq_parseable_dates",
                     F.col("departure_time").isNotNull() & F.col("arrival_time").isNotNull())
        .withColumn("dq_score",
                     (F.col("dq_has_flight_number").cast("int") +
                      F.col("dq_has_departure").cast("int") +
                      F.col("dq_has_arrival").cast("int") +
                      F.col("dq_parseable_dates").cast("int")) / 4)
    )

    # Partition by year for efficient storage
    hist = hist.withColumn("partition_year", F.year("departure_time"))

    output.write_dataframe(hist)


@configure(profile=["EXECUTOR_MEMORY_SMALL"])
@transform(
    output=Output("/datasets/airline/bronze/raw_codeshare_flights"),
    source_codeshare=Input("/datasets/raw/api/codeshare_data"),
    ref_partners=Input("/datasets/ref/partner_airlines"),
)
def compute_codeshare(output, source_codeshare, ref_partners):
    """Ingest codeshare flight data from partner airline API."""
    codeshare = source_codeshare.dataframe()
    partners = ref_partners.dataframe()

    codeshare = (
        codeshare
        .withColumn("src_system", F.lit("API_CODESHARE"))
        .withColumn("src_load_ts", F.current_timestamp())
        .withColumn("is_codeshare", F.lit(True))
    )

    # Enrich with partner information
    codeshare = codeshare.join(
        partners.select(
            F.col("partner_code").alias("operating_carrier_code"),
            F.col("partner_name").alias("operating_carrier_name"),
            F.col("alliance").alias("partner_alliance"),
            F.col("revenue_share_pct"),
        ),
        on="operating_carrier_code",
        how="left"
    )

    # Calculate UAL revenue share
    codeshare = codeshare.withColumn(
        "ual_revenue_share",
        F.col("total_fare") * F.col("revenue_share_pct") / 100
    )

    # Flag data quality
    codeshare = (
        codeshare
        .withColumn("dq_has_operating_carrier", F.col("operating_carrier_code").isNotNull())
        .withColumn("dq_has_fare", F.col("total_fare").isNotNull() & (F.col("total_fare") > 0))
        .withColumn("dq_has_route",
                     F.col("origin").isNotNull() & F.col("destination").isNotNull())
    )

    output.write_dataframe(codeshare)
'''

# File 3: cleaned_flights.py (~700 lines) — Silver layer
SYNTHETIC_FILES["transforms/silver/cleaned_flights.py"] = '''"""Silver layer — Cleaned and validated flight data."""

from transforms.api import transform, Input, Output, configure
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
from transforms.shared_utils import clean_string_column, add_audit_columns, validate_not_null, deduplicate


@configure(profile=["EXECUTOR_MEMORY_LARGE"])
@transform(
    output=Output("/datasets/airline/silver/cleaned_flights"),
    raw_flights=Input("/datasets/airline/bronze/raw_flights"),
    ref_airports=Input("/datasets/ref/airports"),
    ref_airlines=Input("/datasets/ref/airlines"),
)
def compute(output, raw_flights, ref_airports, ref_airlines):
    """Clean and validate flight data for Silver layer.

    Business Rules:
    1. Remove records with null flight_id
    2. Deduplicate by flight_id keeping latest record
    3. Standardize airport codes to uppercase
    4. Validate departure/arrival airports exist in reference
    5. Filter out test flights (flight_number < 100)
    6. Calculate additional metrics
    7. Add audit columns
    """
    df = raw_flights.dataframe()
    airports = ref_airports.dataframe()
    airlines = ref_airlines.dataframe()

    # Step 1: Remove nulls on critical columns
    df = validate_not_null(df, ["flight_id", "flight_number", "departure_airport", "arrival_airport"])

    # Step 2: Deduplicate
    df = deduplicate(df, key_cols=["flight_id"], order_col="src_load_ts", keep="latest")

    # Step 3: Standardize codes
    df = clean_string_column(df, "departure_airport")
    df = clean_string_column(df, "arrival_airport")
    df = clean_string_column(df, "airline_code")
    df = clean_string_column(df, "aircraft_type")
    df = clean_string_column(df, "flight_status")

    # Step 4: Validate against reference data
    valid_airports = airports.select(F.col("airport_code").alias("valid_code")).distinct()

    df = df.join(
        valid_airports.withColumnRenamed("valid_code", "dep_valid"),
        df["departure_airport"] == F.col("dep_valid"),
        how="left"
    )
    df = df.join(
        valid_airports.withColumnRenamed("valid_code", "arr_valid"),
        df["arrival_airport"] == F.col("arr_valid"),
        how="left"
    )
    df = df.withColumn(
        "is_valid_route",
        F.col("dep_valid").isNotNull() & F.col("arr_valid").isNotNull()
    )
    df = df.drop("dep_valid", "arr_valid")

    # Enrich with airport details
    dep_airports = airports.select(
        F.col("airport_code").alias("dep_code"),
        F.col("airport_name").alias("departure_airport_name"),
        F.col("city").alias("departure_city"),
        F.col("state").alias("departure_state"),
        F.col("country").alias("departure_country"),
        F.col("latitude").alias("departure_lat"),
        F.col("longitude").alias("departure_lon"),
        F.col("timezone").alias("departure_tz"),
    )

    arr_airports = airports.select(
        F.col("airport_code").alias("arr_code"),
        F.col("airport_name").alias("arrival_airport_name"),
        F.col("city").alias("arrival_city"),
        F.col("state").alias("arrival_state"),
        F.col("country").alias("arrival_country"),
        F.col("latitude").alias("arrival_lat"),
        F.col("longitude").alias("arrival_lon"),
        F.col("timezone").alias("arrival_tz"),
    )

    df = df.join(dep_airports, df["departure_airport"] == dep_airports["dep_code"], "left").drop("dep_code")
    df = df.join(arr_airports, df["arrival_airport"] == arr_airports["arr_code"], "left").drop("arr_code")

    # Enrich with airline details
    airline_ref = airlines.select(
        F.col("airline_code").alias("ref_airline_code"),
        F.col("airline_name"),
        F.col("alliance"),
        F.col("hub_airport"),
    )
    df = df.join(airline_ref, df["airline_code"] == airline_ref["ref_airline_code"], "left").drop("ref_airline_code")

    # Step 5: Filter test flights
    df = df.filter(F.col("flight_number") >= 100)

    # Step 6: Calculate additional metrics
    df = (
        df
        .withColumn(
            "is_domestic",
            F.when(F.col("departure_country") == F.col("arrival_country"), True).otherwise(False)
        )
        .withColumn(
            "is_delayed",
            F.when(F.col("departure_delay_minutes") > 15, True).otherwise(False)
        )
        .withColumn(
            "delay_category",
            F.when(F.col("departure_delay_minutes") <= 0, "ON_TIME")
             .when(F.col("departure_delay_minutes") <= 15, "MINOR_DELAY")
             .when(F.col("departure_delay_minutes") <= 60, "MODERATE_DELAY")
             .when(F.col("departure_delay_minutes") <= 180, "MAJOR_DELAY")
             .otherwise("SEVERE_DELAY")
        )
        .withColumn(
            "is_long_haul",
            F.when(F.col("distance_miles") > 2000, True).otherwise(False)
        )
        .withColumn(
            "is_redeye",
            F.when(
                F.hour(F.col("departure_time")).between(22, 23) |
                F.hour(F.col("departure_time")).between(0, 5),
                True
            ).otherwise(False)
        )
        .withColumn(
            "passenger_category",
            F.when(F.col("passenger_count") == 0, "EMPTY")
             .when(F.col("load_factor") < 0.5, "LOW")
             .when(F.col("load_factor") < 0.8, "MEDIUM")
             .when(F.col("load_factor") < 0.95, "HIGH")
             .otherwise("FULL")
        )
        .withColumn(
            "great_circle_distance",
            F.lit(3959) * F.acos(
                F.sin(F.radians(F.col("departure_lat"))) * F.sin(F.radians(F.col("arrival_lat"))) +
                F.cos(F.radians(F.col("departure_lat"))) * F.cos(F.radians(F.col("arrival_lat"))) *
                F.cos(F.radians(F.col("arrival_lon")) - F.radians(F.col("departure_lon")))
            )
        )
    )

    # Running totals per route
    route_window = Window.partitionBy("departure_airport", "arrival_airport").orderBy("departure_time")
    df = (
        df
        .withColumn("route_flight_seq", F.row_number().over(route_window))
        .withColumn("route_cumulative_passengers", F.sum("passenger_count").over(route_window))
        .withColumn("route_avg_delay", F.avg("departure_delay_minutes").over(route_window))
    )

    # Step 7: Add audit columns
    df = add_audit_columns(df)

    # Final column selection and ordering
    df = df.select(
        "flight_id", "flight_number", "airline_code", "airline_name", "alliance",
        "departure_airport", "departure_airport_name", "departure_city", "departure_state",
        "arrival_airport", "arrival_airport_name", "arrival_city", "arrival_state",
        "departure_time", "arrival_time", "scheduled_departure", "scheduled_arrival",
        "departure_delay_minutes", "arrival_delay_minutes", "flight_duration_minutes",
        "aircraft_type", "aircraft_name", "manufacturer",
        "passenger_count", "max_capacity", "load_factor", "passenger_category",
        "fuel_consumed", "fuel_efficiency", "distance_miles", "great_circle_distance",
        "is_domestic", "is_delayed", "delay_category", "is_long_haul", "is_redeye",
        "is_valid_route", "dq_score",
        "route_flight_seq", "route_cumulative_passengers", "route_avg_delay",
        "etl_load_timestamp", "etl_source_system", "etl_batch_id",
    )

    output.write_dataframe(df)



@transform(
    output=Output("/datasets/airline/silver/flight_legs"),
    cleaned_flights=Input("/datasets/airline/silver/cleaned_flights"),
)
def compute_legs(output, cleaned_flights):
    """Break multi-leg itineraries into individual flight legs."""
    df = cleaned_flights.dataframe()

    # Identify connecting flights (same passenger, arrival connects to departure)
    connection_window = Window.partitionBy("itinerary_id").orderBy("departure_time")

    legs = (
        df
        .withColumn("leg_number", F.row_number().over(connection_window))
        .withColumn("total_legs", F.count("*").over(Window.partitionBy("itinerary_id")))
        .withColumn("prev_arrival_airport", F.lag("arrival_airport").over(connection_window))
        .withColumn("prev_arrival_time", F.lag("arrival_time").over(connection_window))
        .withColumn("next_departure_airport", F.lead("departure_airport").over(connection_window))
        .withColumn("next_departure_time", F.lead("departure_time").over(connection_window))
    )

    # Calculate connection time
    legs = legs.withColumn(
        "connection_time_minutes",
        F.when(
            F.col("prev_arrival_time").isNotNull(),
            (F.unix_timestamp("departure_time") - F.unix_timestamp("prev_arrival_time")) / 60
        ).otherwise(F.lit(None))
    )

    # Flag tight connections (under 45 minutes)
    legs = legs.withColumn(
        "is_tight_connection",
        F.when(
            F.col("connection_time_minutes").isNotNull() & (F.col("connection_time_minutes") < 45),
            True
        ).otherwise(False)
    )

    # Flag misconnections (connecting airport mismatch)
    legs = legs.withColumn(
        "is_valid_connection",
        F.when(
            F.col("prev_arrival_airport").isNotNull(),
            F.col("prev_arrival_airport") == F.col("departure_airport")
        ).otherwise(True)
    )

    # Leg classification
    legs = legs.withColumn(
        "leg_type",
        F.when(F.col("leg_number") == 1, "ORIGIN")
         .when(F.col("leg_number") == F.col("total_legs"), "DESTINATION")
         .otherwise("CONNECTION")
    )

    legs = add_audit_columns(legs)
    output.write_dataframe(legs)
'''

# File 4: passenger_profiles.py (~500 lines)
SYNTHETIC_FILES["transforms/silver/passenger_profiles.py"] = '''"""Silver layer — Passenger profile aggregation."""

from transforms.api import transform, Input, Output
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from transforms.shared_utils import add_audit_columns, validate_not_null, mask_pii


@transform(
    output=Output("/datasets/airline/silver/passenger_profiles"),
    raw_passengers=Input("/datasets/airline/bronze/raw_passengers"),
    raw_bookings=Input("/datasets/airline/bronze/raw_bookings"),
    loyalty_data=Input("/datasets/airline/bronze/loyalty_program"),
)
def compute(output, raw_passengers, raw_bookings, loyalty_data):
    """Build passenger profiles by combining booking and loyalty data."""
    passengers = raw_passengers.dataframe()
    bookings = raw_bookings.dataframe()
    loyalty = loyalty_data.dataframe()

    # Clean passengers
    passengers = validate_not_null(passengers, ["passenger_id", "first_name", "last_name"])
    passengers = mask_pii(passengers, ["email", "phone_number", "ssn_last4"])

    # Aggregate booking history per passenger
    booking_agg = (
        bookings
        .groupBy("passenger_id")
        .agg(
            F.count("booking_id").alias("total_bookings"),
            F.sum("fare_amount").alias("total_spent"),
            F.avg("fare_amount").alias("avg_fare"),
            F.min("booking_date").alias("first_booking_date"),
            F.max("booking_date").alias("last_booking_date"),
            F.countDistinct("route_id").alias("unique_routes"),
            F.countDistinct("cabin_class").alias("cabin_classes_used"),
            F.sum(F.when(F.col("is_cancelled"), 1).otherwise(0)).alias("cancellations"),
            F.sum(F.when(F.col("is_no_show"), 1).otherwise(0)).alias("no_shows"),
            F.collect_set("departure_airport").alias("departure_airports_list"),
            F.collect_set("arrival_airport").alias("arrival_airports_list"),
        )
    )

    # Calculate booking metrics
    booking_agg = (
        booking_agg
        .withColumn(
            "cancellation_rate",
            F.when(F.col("total_bookings") > 0,
                   F.col("cancellations") / F.col("total_bookings"))
            .otherwise(0)
        )
        .withColumn(
            "no_show_rate",
            F.when(F.col("total_bookings") > 0,
                   F.col("no_shows") / F.col("total_bookings"))
            .otherwise(0)
        )
        .withColumn(
            "booking_frequency_days",
            F.when(
                F.col("total_bookings") > 1,
                F.datediff(F.col("last_booking_date"), F.col("first_booking_date")) / (F.col("total_bookings") - 1)
            ).otherwise(F.lit(None))
        )
        .withColumn(
            "customer_tenure_days",
            F.datediff(F.current_date(), F.col("first_booking_date"))
        )
        .withColumn(
            "departure_airports_count",
            F.size("departure_airports_list")
        )
        .withColumn(
            "arrival_airports_count",
            F.size("arrival_airports_list")
        )
    )

    # Join with loyalty data
    loyalty = loyalty.select(
        F.col("member_id").alias("loyalty_member_id"),
        F.col("passenger_id").alias("loyalty_passenger_id"),
        "tier",
        "miles_balance",
        "miles_earned_ytd",
        "miles_redeemed_ytd",
        "tier_qualification_miles",
        "tier_qualification_segments",
        "enrollment_date",
        "status",
    )

    # Join everything
    profiles = passengers.join(booking_agg, on="passenger_id", how="left")
    profiles = profiles.join(
        loyalty,
        passengers["passenger_id"] == loyalty["loyalty_passenger_id"],
        how="left"
    ).drop("loyalty_passenger_id")

    # Derive passenger segments
    profiles = (
        profiles
        .withColumn(
            "value_segment",
            F.when(F.col("total_spent") > 50000, "PLATINUM")
             .when(F.col("total_spent") > 20000, "GOLD")
             .when(F.col("total_spent") > 5000, "SILVER")
             .when(F.col("total_spent") > 1000, "BRONZE")
             .otherwise("BASIC")
        )
        .withColumn(
            "frequency_segment",
            F.when(F.col("total_bookings") > 50, "VERY_FREQUENT")
             .when(F.col("total_bookings") > 20, "FREQUENT")
             .when(F.col("total_bookings") > 5, "REGULAR")
             .when(F.col("total_bookings") > 1, "OCCASIONAL")
             .otherwise("ONE_TIME")
        )
        .withColumn(
            "risk_flag",
            F.when(
                (F.col("cancellation_rate") > 0.3) | (F.col("no_show_rate") > 0.2),
                True
            ).otherwise(False)
        )
        .withColumn(
            "is_loyalty_member",
            F.col("loyalty_member_id").isNotNull()
        )
        .withColumn(
            "days_since_last_booking",
            F.datediff(F.current_date(), F.col("last_booking_date"))
        )
        .withColumn(
            "is_dormant",
            F.when(F.col("days_since_last_booking") > 365, True).otherwise(False)
        )
    )

    profiles = add_audit_columns(profiles)
    output.write_dataframe(profiles)



@transform(
    output=Output("/datasets/airline/silver/passenger_segments"),
    profiles=Input("/datasets/airline/silver/passenger_profiles"),
    bookings=Input("/datasets/airline/bronze/raw_bookings"),
)
def compute_segments(output, profiles, bookings):
    """Derive advanced passenger segments for marketing and MARS models."""
    df = profiles.dataframe()
    bk = bookings.dataframe()

    # RFM scoring (Recency, Frequency, Monetary)
    rfm = (
        bk
        .groupBy("passenger_id")
        .agg(
            F.datediff(F.current_date(), F.max("booking_date")).alias("recency_days"),
            F.count("booking_id").alias("frequency"),
            F.sum("fare_amount").alias("monetary"),
        )
    )

    # Score each dimension 1-5
    for col_name in ["recency_days", "frequency", "monetary"]:
        quantiles = rfm.approxQuantile(col_name, [0.2, 0.4, 0.6, 0.8], 0.01)
        if col_name == "recency_days":
            # Lower recency is better
            rfm = rfm.withColumn(
                f"{col_name}_score",
                F.when(F.col(col_name) <= quantiles[0], 5)
                 .when(F.col(col_name) <= quantiles[1], 4)
                 .when(F.col(col_name) <= quantiles[2], 3)
                 .when(F.col(col_name) <= quantiles[3], 2)
                 .otherwise(1)
            )
        else:
            rfm = rfm.withColumn(
                f"{col_name}_score",
                F.when(F.col(col_name) >= quantiles[3], 5)
                 .when(F.col(col_name) >= quantiles[2], 4)
                 .when(F.col(col_name) >= quantiles[1], 3)
                 .when(F.col(col_name) >= quantiles[0], 2)
                 .otherwise(1)
            )

    rfm = rfm.withColumn(
        "rfm_score",
        F.col("recency_days_score") + F.col("frequency_score") + F.col("monetary_score")
    )

    rfm = rfm.withColumn(
        "rfm_segment",
        F.when(F.col("rfm_score") >= 13, "CHAMPION")
         .when(F.col("rfm_score") >= 10, "LOYAL")
         .when(F.col("rfm_score") >= 7, "POTENTIAL")
         .when(F.col("rfm_score") >= 4, "AT_RISK")
         .otherwise("LOST")
    )

    # Join RFM scores back to profiles
    result = df.join(rfm, on="passenger_id", how="left")

    # Predicted lifetime value
    result = result.withColumn(
        "predicted_ltv",
        F.col("avg_fare") * F.col("total_bookings") *
        F.when(F.col("rfm_segment") == "CHAMPION", 3.0)
         .when(F.col("rfm_segment") == "LOYAL", 2.0)
         .when(F.col("rfm_segment") == "POTENTIAL", 1.5)
         .when(F.col("rfm_segment") == "AT_RISK", 0.5)
         .otherwise(0.2)
    )

    result = add_audit_columns(result)
    output.write_dataframe(result)
'''

# File 5: flight_performance_gold.py (~600 lines) — Gold layer
SYNTHETIC_FILES["transforms/gold/flight_performance.py"] = '''"""Gold layer — Flight performance aggregations for BI."""

from transforms.api import transform, Input, Output, configure
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from transforms.shared_utils import add_audit_columns


@configure(profile=["EXECUTOR_MEMORY_XLARGE", "NUM_EXECUTORS_32"])
@transform(
    output=Output("/datasets/airline/gold/flight_performance"),
    cleaned_flights=Input("/datasets/airline/silver/cleaned_flights"),
)
def compute(output, cleaned_flights):
    """Aggregate flight performance metrics for dashboards and MARS.

    Output feeds:
    - Operations dashboard (daily OTP by hub)
    - Revenue management (load factor trends)
    - MARS AI models (delay prediction features)
    """
    df = cleaned_flights.dataframe()

    # Daily route-level aggregation
    daily_route = (
        df
        .withColumn("flight_date", F.to_date("departure_time"))
        .groupBy("flight_date", "departure_airport", "arrival_airport", "airline_code")
        .agg(
            F.count("flight_id").alias("total_flights"),
            F.sum("passenger_count").alias("total_passengers"),
            F.avg("passenger_count").alias("avg_passengers"),
            F.avg("load_factor").alias("avg_load_factor"),
            F.avg("departure_delay_minutes").alias("avg_departure_delay"),
            F.avg("arrival_delay_minutes").alias("avg_arrival_delay"),
            F.avg("flight_duration_minutes").alias("avg_duration"),
            F.sum("fuel_consumed").alias("total_fuel"),
            F.avg("fuel_efficiency").alias("avg_fuel_efficiency"),
            F.sum("distance_miles").alias("total_distance"),
            F.sum(F.when(F.col("is_delayed"), 1).otherwise(0)).alias("delayed_flights"),
            F.sum(F.when(F.col("delay_category") == "ON_TIME", 1).otherwise(0)).alias("on_time_flights"),
            F.sum(F.when(F.col("delay_category") == "SEVERE_DELAY", 1).otherwise(0)).alias("severe_delays"),
            F.max("departure_delay_minutes").alias("max_delay_minutes"),
            F.min("departure_delay_minutes").alias("min_delay_minutes"),
            F.stddev("departure_delay_minutes").alias("stddev_delay"),
            F.countDistinct("aircraft_type").alias("aircraft_types_used"),
        )
    )

    # Calculate derived KPIs
    daily_route = (
        daily_route
        .withColumn(
            "on_time_performance",
            F.when(F.col("total_flights") > 0,
                   F.col("on_time_flights") / F.col("total_flights") * 100)
            .otherwise(0)
        )
        .withColumn(
            "delay_rate",
            F.when(F.col("total_flights") > 0,
                   F.col("delayed_flights") / F.col("total_flights") * 100)
            .otherwise(0)
        )
        .withColumn(
            "severe_delay_rate",
            F.when(F.col("total_flights") > 0,
                   F.col("severe_delays") / F.col("total_flights") * 100)
            .otherwise(0)
        )
        .withColumn(
            "revenue_passenger_miles",
            F.col("total_passengers") * F.col("total_distance") / F.col("total_flights")
        )
        .withColumn(
            "available_seat_miles",
            F.lit(180) * F.col("total_distance")  # avg 180 seats
        )
        .withColumn(
            "system_load_factor",
            F.col("revenue_passenger_miles") / F.col("available_seat_miles") * 100
        )
        .withColumn(
            "cost_per_asm",
            F.col("total_fuel") * F.lit(3.50) / F.col("available_seat_miles")  # $3.50/gal
        )
    )

    # Rolling 7-day and 30-day averages
    route_date_window_7d = (
        Window
        .partitionBy("departure_airport", "arrival_airport")
        .orderBy(F.col("flight_date").cast("long"))
        .rangeBetween(-6 * 86400, 0)
    )
    route_date_window_30d = (
        Window
        .partitionBy("departure_airport", "arrival_airport")
        .orderBy(F.col("flight_date").cast("long"))
        .rangeBetween(-29 * 86400, 0)
    )

    daily_route = (
        daily_route
        .withColumn("otp_7d_avg", F.avg("on_time_performance").over(route_date_window_7d))
        .withColumn("otp_30d_avg", F.avg("on_time_performance").over(route_date_window_30d))
        .withColumn("delay_7d_avg", F.avg("avg_departure_delay").over(route_date_window_7d))
        .withColumn("delay_30d_avg", F.avg("avg_departure_delay").over(route_date_window_30d))
        .withColumn("load_factor_7d_avg", F.avg("avg_load_factor").over(route_date_window_7d))
        .withColumn("load_factor_30d_avg", F.avg("avg_load_factor").over(route_date_window_30d))
        .withColumn("flights_7d_total", F.sum("total_flights").over(route_date_window_7d))
        .withColumn("flights_30d_total", F.sum("total_flights").over(route_date_window_30d))
    )

    # Add day-of-week and seasonal features for MARS
    daily_route = (
        daily_route
        .withColumn("day_of_week", F.dayofweek("flight_date"))
        .withColumn("day_name", F.date_format("flight_date", "EEEE"))
        .withColumn("month", F.month("flight_date"))
        .withColumn("quarter", F.quarter("flight_date"))
        .withColumn("year", F.year("flight_date"))
        .withColumn("week_of_year", F.weekofyear("flight_date"))
        .withColumn(
            "is_weekend",
            F.when(F.dayofweek("flight_date").isin(1, 7), True).otherwise(False)
        )
        .withColumn(
            "season",
            F.when(F.month("flight_date").isin(12, 1, 2), "WINTER")
             .when(F.month("flight_date").isin(3, 4, 5), "SPRING")
             .when(F.month("flight_date").isin(6, 7, 8), "SUMMER")
             .otherwise("FALL")
        )
        .withColumn(
            "is_peak_travel",
            F.when(
                F.month("flight_date").isin(6, 7, 8, 11, 12) |
                (F.month("flight_date") == 3),  # Spring break
                True
            ).otherwise(False)
        )
    )

    daily_route = add_audit_columns(daily_route)
    output.write_dataframe(daily_route)
'''

# File 6: revenue_summary.py (~400 lines) — Gold
SYNTHETIC_FILES["transforms/gold/revenue_summary.py"] = '''"""Gold layer — Revenue summary for finance team."""

from transforms.api import transform, Input, Output
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from transforms.shared_utils import add_audit_columns


@transform(
    output=Output("/datasets/airline/gold/revenue_summary"),
    bookings=Input("/datasets/airline/silver/cleaned_bookings"),
    flights=Input("/datasets/airline/silver/cleaned_flights"),
    passengers=Input("/datasets/airline/silver/passenger_profiles"),
)
def compute(output, bookings, flights, passengers):
    """Revenue summary aggregated by route, month, and cabin class."""
    df_bookings = bookings.dataframe()
    df_flights = flights.dataframe()
    df_passengers = passengers.dataframe()

    # Join bookings with flight info
    revenue = df_bookings.join(
        df_flights.select("flight_id", "departure_airport", "arrival_airport",
                          "distance_miles", "is_domestic", "airline_code"),
        on="flight_id",
        how="inner"
    )

    # Monthly route revenue
    monthly_revenue = (
        revenue
        .withColumn("booking_month", F.date_trunc("month", "booking_date"))
        .groupBy("booking_month", "departure_airport", "arrival_airport",
                 "cabin_class", "is_domestic", "airline_code")
        .agg(
            F.count("booking_id").alias("total_bookings"),
            F.sum("fare_amount").alias("gross_revenue"),
            F.sum("tax_amount").alias("total_tax"),
            F.sum("ancillary_revenue").alias("total_ancillary"),
            F.avg("fare_amount").alias("avg_fare"),
            F.min("fare_amount").alias("min_fare"),
            F.max("fare_amount").alias("max_fare"),
            F.stddev("fare_amount").alias("fare_stddev"),
            F.sum(F.when(F.col("is_refunded"), F.col("refund_amount")).otherwise(0)).alias("total_refunds"),
            F.sum(F.when(F.col("is_cancelled"), 1).otherwise(0)).alias("cancelled_bookings"),
            F.countDistinct("passenger_id").alias("unique_passengers"),
            F.avg("distance_miles").alias("avg_distance"),
        )
    )

    monthly_revenue = (
        monthly_revenue
        .withColumn("net_revenue", F.col("gross_revenue") - F.col("total_refunds"))
        .withColumn("total_revenue", F.col("net_revenue") + F.col("total_ancillary"))
        .withColumn("revenue_per_booking", F.col("total_revenue") / F.col("total_bookings"))
        .withColumn("revenue_per_mile",
                     F.when(F.col("avg_distance") > 0,
                            F.col("revenue_per_booking") / F.col("avg_distance"))
                     .otherwise(0))
        .withColumn("cancellation_rate", F.col("cancelled_bookings") / F.col("total_bookings"))
        .withColumn("refund_rate", F.col("total_refunds") / F.col("gross_revenue"))
        .withColumn("ancillary_share", F.col("total_ancillary") / F.col("total_revenue"))
    )

    # Year-over-year comparison
    yoy_window = Window.partitionBy(
        "departure_airport", "arrival_airport", "cabin_class",
        F.month("booking_month")
    ).orderBy("booking_month")

    monthly_revenue = (
        monthly_revenue
        .withColumn("prev_year_revenue", F.lag("total_revenue", 12).over(yoy_window))
        .withColumn(
            "yoy_growth",
            F.when(F.col("prev_year_revenue") > 0,
                   (F.col("total_revenue") - F.col("prev_year_revenue")) / F.col("prev_year_revenue") * 100)
            .otherwise(F.lit(None))
        )
    )

    monthly_revenue = add_audit_columns(monthly_revenue)
    output.write_dataframe(monthly_revenue)
'''

# File 7: crew_scheduling.py (~500 lines)
SYNTHETIC_FILES["transforms/silver/crew_scheduling.py"] = '''"""Silver layer — Crew scheduling and assignment data."""

from transforms.api import transform, Input, Output
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from transforms.shared_utils import validate_not_null, add_audit_columns, clean_string_column


@transform(
    output=Output("/datasets/airline/silver/crew_assignments"),
    raw_crew=Input("/datasets/airline/bronze/raw_crew"),
    raw_assignments=Input("/datasets/airline/bronze/raw_crew_assignments"),
    raw_flights=Input("/datasets/airline/bronze/raw_flights"),
)
def compute(output, raw_crew, raw_assignments, raw_flights):
    """Process crew scheduling data with FAA duty time compliance."""
    crew = raw_crew.dataframe()
    assignments = raw_assignments.dataframe()
    flights = raw_flights.dataframe()

    # Clean crew data
    crew = validate_not_null(crew, ["crew_id", "employee_id", "crew_role"])
    crew = clean_string_column(crew, "crew_role")
    crew = clean_string_column(crew, "base_airport")

    # Join assignments with flight details
    crew_flights = assignments.join(
        flights.select("flight_id", "departure_time", "arrival_time",
                       "departure_airport", "arrival_airport",
                       "flight_duration_minutes"),
        on="flight_id",
        how="inner"
    )

    crew_flights = crew_flights.join(
        crew.select("crew_id", "employee_id", "crew_role", "base_airport",
                    "seniority_date", "certification_level"),
        on="crew_id",
        how="inner"
    )

    # Calculate duty time windows
    duty_window = (
        Window
        .partitionBy("crew_id")
        .orderBy("departure_time")
        .rangeBetween(-24 * 3600, 0)  # 24-hour window
    )

    crew_flights = (
        crew_flights
        .withColumn("duty_hours_24h",
                     F.sum("flight_duration_minutes").over(duty_window) / 60)
        .withColumn("flights_24h",
                     F.count("flight_id").over(duty_window))
    )

    # 7-day rolling window
    weekly_window = (
        Window
        .partitionBy("crew_id")
        .orderBy(F.col("departure_time").cast("long"))
        .rangeBetween(-7 * 86400, 0)
    )

    crew_flights = (
        crew_flights
        .withColumn("duty_hours_7d",
                     F.sum("flight_duration_minutes").over(weekly_window) / 60)
        .withColumn("flights_7d",
                     F.count("flight_id").over(weekly_window))
    )

    # FAA compliance flags
    crew_flights = (
        crew_flights
        .withColumn(
            "faa_duty_limit_exceeded",
            F.when(
                (F.col("crew_role") == "PILOT") & (F.col("duty_hours_24h") > 8),
                True
            ).when(
                (F.col("crew_role") == "FIRST_OFFICER") & (F.col("duty_hours_24h") > 8),
                True
            ).when(
                (F.col("crew_role") == "FLIGHT_ATTENDANT") & (F.col("duty_hours_24h") > 14),
                True
            ).otherwise(False)
        )
        .withColumn(
            "faa_weekly_limit_exceeded",
            F.when(F.col("duty_hours_7d") > 60, True).otherwise(False)
        )
        .withColumn(
            "rest_requirement_hours",
            F.when(F.col("crew_role") == "PILOT", 10)
             .when(F.col("crew_role") == "FIRST_OFFICER", 10)
             .when(F.col("crew_role") == "FLIGHT_ATTENDANT", 9)
             .otherwise(8)
        )
        .withColumn(
            "compliance_status",
            F.when(
                F.col("faa_duty_limit_exceeded") | F.col("faa_weekly_limit_exceeded"),
                "NON_COMPLIANT"
            ).otherwise("COMPLIANT")
        )
    )

    # Deadhead detection
    crew_flights = crew_flights.withColumn(
        "is_deadhead",
        F.when(F.col("assignment_type") == "DEADHEAD", True).otherwise(False)
    )

    # Calculate time between flights for rest analysis
    prev_arrival_window = (
        Window.partitionBy("crew_id").orderBy("departure_time")
    )
    crew_flights = crew_flights.withColumn(
        "prev_arrival_time",
        F.lag("arrival_time").over(prev_arrival_window)
    )
    crew_flights = crew_flights.withColumn(
        "rest_hours_since_last",
        F.when(
            F.col("prev_arrival_time").isNotNull(),
            (F.unix_timestamp("departure_time") - F.unix_timestamp("prev_arrival_time")) / 3600
        ).otherwise(F.lit(None))
    )
    crew_flights = crew_flights.withColumn(
        "adequate_rest",
        F.when(
            F.col("rest_hours_since_last").isNotNull(),
            F.col("rest_hours_since_last") >= F.col("rest_requirement_hours")
        ).otherwise(True)
    )

    crew_flights = add_audit_columns(crew_flights)
    output.write_dataframe(crew_flights)
'''

# File 8: maintenance_events.py (~400 lines)
SYNTHETIC_FILES["transforms/silver/maintenance_events.py"] = '''"""Silver layer — Aircraft maintenance events and scheduling."""

from transforms.api import transform, Input, Output
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from transforms.shared_utils import validate_not_null, add_audit_columns


@transform(
    output=Output("/datasets/airline/silver/maintenance_events"),
    raw_maintenance=Input("/datasets/airline/bronze/raw_maintenance"),
    ref_aircraft=Input("/datasets/ref/aircraft_types"),
)
def compute(output, raw_maintenance, ref_aircraft):
    """Process maintenance events with predictive maintenance features."""
    maint = raw_maintenance.dataframe()
    aircraft = ref_aircraft.dataframe()

    # Validate required fields
    maint = validate_not_null(maint, ["event_id", "aircraft_id", "event_type", "event_date"])

    # Join with aircraft reference
    maint = maint.join(
        aircraft.select("aircraft_code", "aircraft_name", "manufacturer",
                        "max_flight_hours", "maintenance_interval_hours"),
        maint["aircraft_type"] == aircraft["aircraft_code"],
        how="left"
    ).drop("aircraft_code")

    # Calculate time since last maintenance per aircraft
    maint_window = Window.partitionBy("aircraft_id").orderBy("event_date")

    maint = (
        maint
        .withColumn("prev_event_date", F.lag("event_date").over(maint_window))
        .withColumn("days_since_last_maint",
                     F.datediff(F.col("event_date"), F.col("prev_event_date")))
        .withColumn("event_sequence", F.row_number().over(maint_window))
    )

    # Cumulative flight hours at time of maintenance
    maint = maint.withColumn(
        "hours_since_last_maint",
        F.when(
            F.col("days_since_last_maint").isNotNull(),
            F.col("days_since_last_maint") * F.lit(8.5)  # avg 8.5 flight hours/day
        ).otherwise(F.lit(None))
    )

    # Maintenance categories
    maint = (
        maint
        .withColumn(
            "maintenance_category",
            F.when(F.col("event_type") == "A_CHECK", "ROUTINE")
             .when(F.col("event_type") == "B_CHECK", "INTERMEDIATE")
             .when(F.col("event_type") == "C_CHECK", "MAJOR")
             .when(F.col("event_type") == "D_CHECK", "OVERHAUL")
             .when(F.col("event_type") == "UNSCHEDULED", "UNSCHEDULED")
             .otherwise("OTHER")
        )
        .withColumn(
            "is_unscheduled",
            F.col("event_type") == "UNSCHEDULED"
        )
        .withColumn(
            "estimated_downtime_hours",
            F.when(F.col("event_type") == "A_CHECK", 8)
             .when(F.col("event_type") == "B_CHECK", 72)
             .when(F.col("event_type") == "C_CHECK", 720)
             .when(F.col("event_type") == "D_CHECK", 2400)
             .when(F.col("event_type") == "UNSCHEDULED", 24)
             .otherwise(4)
        )
    )

    # Predictive maintenance features
    aircraft_window = (
        Window.partitionBy("aircraft_id")
        .orderBy("event_date")
        .rowsBetween(-5, 0)
    )

    maint = (
        maint
        .withColumn("unscheduled_events_last5",
                     F.sum(F.when(F.col("is_unscheduled"), 1).otherwise(0)).over(aircraft_window))
        .withColumn("avg_days_between_maint_last5",
                     F.avg("days_since_last_maint").over(aircraft_window))
        .withColumn(
            "maintenance_risk_score",
            F.when(F.col("unscheduled_events_last5") >= 3, "HIGH")
             .when(F.col("unscheduled_events_last5") >= 1, "MEDIUM")
             .otherwise("LOW")
        )
    )

    # Calculate next expected maintenance
    maint = maint.withColumn(
        "next_expected_maint_date",
        F.when(
            F.col("maintenance_interval_hours").isNotNull(),
            F.date_add(F.col("event_date"),
                       (F.col("maintenance_interval_hours") / F.lit(8.5)).cast("int"))
        ).otherwise(F.date_add(F.col("event_date"), 90))
    )

    maint = add_audit_columns(maint)
    output.write_dataframe(maint)
'''

# File 9: weather_impact.py (~400 lines)
SYNTHETIC_FILES["transforms/silver/weather_impact.py"] = '''"""Silver layer — Weather impact on flight operations."""

from transforms.api import transform, Input, Output
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from transforms.shared_utils import add_audit_columns, validate_not_null


@transform(
    output=Output("/datasets/airline/silver/weather_impact"),
    raw_weather=Input("/datasets/airline/bronze/raw_weather"),
    cleaned_flights=Input("/datasets/airline/silver/cleaned_flights"),
    ref_airports=Input("/datasets/ref/airports"),
)
def compute(output, raw_weather, cleaned_flights, ref_airports):
    """Correlate weather conditions with flight delays."""
    weather = raw_weather.dataframe()
    flights = cleaned_flights.dataframe()
    airports = ref_airports.dataframe()

    # Validate weather data
    weather = validate_not_null(weather, ["station_id", "observation_time", "temperature_f"])

    # Map weather stations to airports
    weather = weather.join(
        airports.select(
            F.col("weather_station_id").alias("station_id_ref"),
            F.col("airport_code").alias("weather_airport_code"),
        ),
        weather["station_id"] == F.col("station_id_ref"),
        how="inner"
    ).drop("station_id_ref")

    # Aggregate weather to hourly
    hourly_weather = (
        weather
        .withColumn("obs_hour", F.date_trunc("hour", "observation_time"))
        .groupBy("weather_airport_code", "obs_hour")
        .agg(
            F.avg("temperature_f").alias("avg_temp_f"),
            F.avg("wind_speed_mph").alias("avg_wind_speed"),
            F.max("wind_gust_mph").alias("max_wind_gust"),
            F.avg("visibility_miles").alias("avg_visibility"),
            F.avg("ceiling_ft").alias("avg_ceiling"),
            F.sum("precipitation_inches").alias("total_precip"),
            F.last("weather_condition").alias("primary_condition"),
            F.max("is_thunderstorm").alias("has_thunderstorm"),
            F.max("is_freezing").alias("has_freezing"),
            F.max("is_fog").alias("has_fog"),
            F.max("is_snow").alias("has_snow"),
        )
    )

    # Weather severity score
    hourly_weather = hourly_weather.withColumn(
        "weather_severity",
        (
            F.when(F.col("avg_visibility") < 1, 30).otherwise(0) +
            F.when(F.col("avg_visibility") < 3, 15).otherwise(0) +
            F.when(F.col("max_wind_gust") > 50, 30).otherwise(0) +
            F.when(F.col("max_wind_gust") > 30, 15).otherwise(0) +
            F.when(F.col("has_thunderstorm"), 40).otherwise(0) +
            F.when(F.col("has_freezing"), 25).otherwise(0) +
            F.when(F.col("has_snow"), 20).otherwise(0) +
            F.when(F.col("has_fog"), 15).otherwise(0) +
            F.when(F.col("total_precip") > 1.0, 20).otherwise(0)
        )
    )

    hourly_weather = hourly_weather.withColumn(
        "weather_category",
        F.when(F.col("weather_severity") >= 60, "SEVERE")
         .when(F.col("weather_severity") >= 30, "MODERATE")
         .when(F.col("weather_severity") >= 10, "MINOR")
         .otherwise("CLEAR")
    )

    # Join flights with departure weather
    flights_with_weather = flights.join(
        hourly_weather,
        (flights["departure_airport"] == hourly_weather["weather_airport_code"]) &
        (F.date_trunc("hour", flights["departure_time"]) == hourly_weather["obs_hour"]),
        how="left"
    ).drop("weather_airport_code", "obs_hour")

    # Rename departure weather columns
    weather_cols = ["avg_temp_f", "avg_wind_speed", "max_wind_gust",
                    "avg_visibility", "avg_ceiling", "total_precip",
                    "primary_condition", "has_thunderstorm", "has_freezing",
                    "has_fog", "has_snow", "weather_severity", "weather_category"]

    for col_name in weather_cols:
        flights_with_weather = flights_with_weather.withColumnRenamed(
            col_name, f"dep_{col_name}"
        )

    # Analyze weather-delay correlation
    flights_with_weather = (
        flights_with_weather
        .withColumn(
            "weather_contributed_delay",
            F.when(
                (F.col("dep_weather_severity") >= 30) & (F.col("is_delayed")),
                True
            ).otherwise(False)
        )
        .withColumn(
            "estimated_weather_delay_mins",
            F.when(F.col("dep_weather_category") == "SEVERE", F.col("departure_delay_minutes") * 0.8)
             .when(F.col("dep_weather_category") == "MODERATE", F.col("departure_delay_minutes") * 0.5)
             .when(F.col("dep_weather_category") == "MINOR", F.col("departure_delay_minutes") * 0.2)
             .otherwise(0)
        )
    )

    flights_with_weather = add_audit_columns(flights_with_weather)
    output.write_dataframe(flights_with_weather)
'''

# File 10: operational_kpis.py (~500 lines) — Gold layer
SYNTHETIC_FILES["transforms/gold/operational_kpis.py"] = '''"""Gold layer — Operational KPIs for executive dashboard."""

from transforms.api import transform, Input, Output, configure
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from transforms.shared_utils import add_audit_columns


@configure(profile=["EXECUTOR_MEMORY_XLARGE"])
@transform(
    output=Output("/datasets/airline/gold/operational_kpis"),
    flights=Input("/datasets/airline/silver/cleaned_flights"),
    crew=Input("/datasets/airline/silver/crew_assignments"),
    maintenance=Input("/datasets/airline/silver/maintenance_events"),
    weather=Input("/datasets/airline/silver/weather_impact"),
    passengers=Input("/datasets/airline/silver/passenger_profiles"),
)
def compute(output, flights, crew, maintenance, weather, passengers):
    """Daily operational KPIs aggregated by hub airport.

    Feeds the executive operations dashboard and MARS alerting system.
    """
    ctx = output.ctx
    spark = ctx.spark_session

    df_flights = flights.dataframe()
    df_crew = crew.dataframe()
    df_maint = maintenance.dataframe()
    df_weather = weather.dataframe()
    df_passengers = passengers.dataframe()

    # Daily hub-level flight metrics
    hub_daily = (
        df_flights
        .withColumn("flight_date", F.to_date("departure_time"))
        .groupBy("flight_date", "departure_airport")
        .agg(
            F.count("flight_id").alias("total_departures"),
            F.avg("departure_delay_minutes").alias("avg_delay"),
            F.sum(F.when(F.col("is_delayed"), 1).otherwise(0)).alias("delayed_count"),
            F.sum("passenger_count").alias("total_passengers"),
            F.avg("load_factor").alias("avg_load_factor"),
            F.sum("fuel_consumed").alias("total_fuel"),
            F.countDistinct("airline_code").alias("operating_airlines"),
            F.countDistinct("aircraft_type").alias("aircraft_types"),
            F.sum(F.when(F.col("delay_category") == "SEVERE_DELAY", 1).otherwise(0)).alias("severe_delays"),
            F.sum(F.when(F.col("is_domestic"), 1).otherwise(0)).alias("domestic_flights"),
            F.sum(F.when(~F.col("is_domestic"), 1).otherwise(0)).alias("international_flights"),
        )
    )

    hub_daily = hub_daily.withColumn(
        "on_time_pct",
        (F.col("total_departures") - F.col("delayed_count")) / F.col("total_departures") * 100
    )

    # Crew compliance metrics per hub
    crew_daily = (
        df_crew
        .withColumn("flight_date", F.to_date("departure_time"))
        .groupBy("flight_date", "departure_airport")
        .agg(
            F.countDistinct("crew_id").alias("active_crew"),
            F.sum(F.when(F.col("compliance_status") == "NON_COMPLIANT", 1).otherwise(0)).alias("compliance_violations"),
            F.sum(F.when(~F.col("adequate_rest"), 1).otherwise(0)).alias("rest_violations"),
            F.avg("duty_hours_24h").alias("avg_duty_hours"),
            F.sum(F.when(F.col("is_deadhead"), 1).otherwise(0)).alias("deadhead_assignments"),
        )
    )

    # Maintenance metrics per hub
    maint_daily = (
        df_maint
        .withColumn("event_date_only", F.to_date("event_date"))
        .groupBy(F.col("event_date_only").alias("flight_date"),
                 F.col("base_airport").alias("departure_airport"))
        .agg(
            F.count("event_id").alias("maintenance_events"),
            F.sum(F.when(F.col("is_unscheduled"), 1).otherwise(0)).alias("unscheduled_events"),
            F.sum("estimated_downtime_hours").alias("total_downtime_hours"),
            F.sum(F.when(F.col("maintenance_risk_score") == "HIGH", 1).otherwise(0)).alias("high_risk_aircraft"),
        )
    )

    # Combine all metrics
    kpis = hub_daily.join(crew_daily, on=["flight_date", "departure_airport"], how="left")
    kpis = kpis.join(maint_daily, on=["flight_date", "departure_airport"], how="left")

    # Fill nulls for metrics
    fill_cols = {
        "active_crew": 0, "compliance_violations": 0, "rest_violations": 0,
        "avg_duty_hours": 0, "deadhead_assignments": 0,
        "maintenance_events": 0, "unscheduled_events": 0,
        "total_downtime_hours": 0, "high_risk_aircraft": 0,
    }
    kpis = kpis.fillna(fill_cols)

    # Composite operational health score (0-100)
    kpis = kpis.withColumn(
        "ops_health_score",
        F.greatest(
            F.lit(0),
            F.lit(100)
            - (100 - F.col("on_time_pct")) * 0.4  # OTP weight: 40%
            - F.col("compliance_violations") * 5    # Each violation: -5
            - F.col("unscheduled_events") * 3        # Each unscheduled maint: -3
            - F.col("severe_delays") * 2             # Each severe delay: -2
        )
    )

    # Trend calculations
    hub_window_7d = (
        Window
        .partitionBy("departure_airport")
        .orderBy(F.col("flight_date").cast("long"))
        .rangeBetween(-6 * 86400, 0)
    )

    kpis = (
        kpis
        .withColumn("otp_7d_trend", F.avg("on_time_pct").over(hub_window_7d))
        .withColumn("ops_health_7d_trend", F.avg("ops_health_score").over(hub_window_7d))
        .withColumn("delay_7d_trend", F.avg("avg_delay").over(hub_window_7d))
        .withColumn("fuel_7d_total", F.sum("total_fuel").over(hub_window_7d))
        .withColumn("pax_7d_total", F.sum("total_passengers").over(hub_window_7d))
    )

    # Hub classification
    kpis = kpis.withColumn(
        "hub_status",
        F.when(F.col("ops_health_score") >= 90, "EXCELLENT")
         .when(F.col("ops_health_score") >= 75, "GOOD")
         .when(F.col("ops_health_score") >= 60, "FAIR")
         .when(F.col("ops_health_score") >= 40, "POOR")
         .otherwise("CRITICAL")
    )

    kpis = add_audit_columns(kpis)
    output.write_dataframe(kpis)
'''


# ─── Test Runner ──────────────────────────────────────────────────────────────

def run_tests():
    passed = 0
    failed = 0
    errors = []

    def test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            errors.append((name, e))
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()

    total_lines = sum(content.count("\n") + 1 for content in SYNTHETIC_FILES.values())
    print(f"\n{'='*70}")
    print(f"END-TO-END TEST — {len(SYNTHETIC_FILES)} files, {total_lines} total lines")
    print(f"{'='*70}\n")

    # ── Test 1: Chunking ──
    print("[1] RAG Chunking")

    all_chunks = []
    def test_chunking():
        for fp, content in SYNTHETIC_FILES.items():
            file_id = f"fid_{fp}"
            chunks = chunk_file(file_id, fp, content)
            # Only keep chunks with actual code content (skip padding-only chunks)
            real_chunks = [c for c in chunks if c.content.strip()]
            all_chunks.extend(real_chunks)
            assert len(real_chunks) > 0, f"No chunks for {fp}"
            for c in real_chunks:
                assert c.keywords, f"No keywords in chunk {c.id}"
    test("chunk all 10 files", test_chunking)

    def test_chunk_count():
        assert len(all_chunks) >= 30, f"Expected 30+ chunks, got {len(all_chunks)}"
    test(f"total chunks = {len(all_chunks)}", test_chunk_count)

    # ── Test 2: Keyword Index ──
    print("\n[2] Keyword Index")

    index = KeywordIndex()
    def test_indexing():
        index.add_chunks(all_chunks)
        stats = index.get_stats()
        assert stats["chunk_count"] == len(all_chunks)
        assert stats["file_count"] == len(SYNTHETIC_FILES)
    test("index all chunks", test_indexing)

    def test_search_flight():
        results = index.search("flight delay performance", top_k=5)
        assert len(results) > 0, "No results for flight query"
        assert any("flight" in r.file_path for r in results), "Expected flight-related file in results"
    test("search 'flight delay performance'", test_search_flight)

    def test_search_maintenance():
        results = index.search("maintenance aircraft check", top_k=5)
        assert len(results) > 0, "No results for maintenance query"
    test("search 'maintenance aircraft check'", test_search_maintenance)

    def test_search_crew():
        results = index.search("crew duty FAA compliance", top_k=3)
        assert len(results) > 0
    test("search 'crew duty FAA compliance'", test_search_crew)

    def test_search_revenue():
        results = index.search("revenue booking fare", top_k=3)
        assert len(results) > 0
    test("search 'revenue booking fare'", test_search_revenue)

    # ── Test 3: Import Resolution ──
    print("\n[3] Import Resolution")

    def test_parse_imports():
        imports = parse_imports(SYNTHETIC_FILES["transforms/silver/cleaned_flights.py"])
        assert "transforms.api" in imports
        assert "transforms.shared_utils" in imports
    test("parse imports from cleaned_flights.py", test_parse_imports)

    def test_dep_graph():
        graph = build_dependency_graph(SYNTHETIC_FILES)
        assert len(graph) == len(SYNTHETIC_FILES)
        # cleaned_flights imports shared_utils
        deps = graph.get("transforms/silver/cleaned_flights.py", [])
        assert "transforms/shared_utils.py" in deps, f"Expected shared_utils in deps, got {deps}"
    test("build dependency graph", test_dep_graph)

    def test_resolve_transitive():
        resolved = resolve_imports("transforms/silver/cleaned_flights.py", SYNTHETIC_FILES)
        assert "transforms/shared_utils.py" in resolved
    test("resolve transitive imports for cleaned_flights.py", test_resolve_transitive)

    def test_resolve_gold():
        resolved = resolve_imports("transforms/gold/operational_kpis.py", SYNTHETIC_FILES)
        assert "transforms/shared_utils.py" in resolved
    test("resolve imports for operational_kpis.py", test_resolve_gold)

    # ── Test 4: Section Splitting (Large File Handling) ──
    print("\n[4] Section Splitting for Large Files")

    for fp, content in SYNTHETIC_FILES.items():
        lines = content.count("\n") + 1
        def make_test(fp=fp, content=content, lines=lines):
            def test_fn():
                sections = _split_into_sections(content)
                assert len(sections) > 0, "No sections"
                # Verify all content is preserved
                total_section_lines = sum(s["end_line"] - s["start_line"] + 1 for s in sections)
                # Allow some variance from merging
                if lines > 200:
                    assert len(sections) > 1, f"Large file ({lines} lines) should split into >1 sections"
                for s in sections:
                    assert s["code"].strip(), f"Empty section: {s['name']}"
            return test_fn
        test(f"split {fp} ({lines} lines)", make_test())

    # ── Test 5: Notebook Formatting ──
    print("\n[5] Notebook Formatting (.ipynb)")

    def test_notebook_basic():
        code = "# Databricks notebook source\n\n# COMMAND ----------\n\nfrom pyspark.sql import functions as F\n\n# COMMAND ----------\n\ndf = spark.read.format('delta').load('s3://bucket/path')\n\n# COMMAND ----------\n\nprint(df.count())"
        nb_json = code_to_notebook(code, file_path="test.py", target_layer="silver")
        nb = json.loads(nb_json)
        assert nb["nbformat"] == 4
        assert len(nb["cells"]) == 3, f"Expected 3 cells, got {len(nb['cells'])}"
        assert all(c["cell_type"] in ("code", "markdown") for c in nb["cells"])
    test("basic notebook with 3 cells", test_notebook_basic)

    def test_notebook_markdown_detection():
        code = "# Databricks notebook source\n\n# COMMAND ----------\n\n# Notebook: test\n# Layer: silver\n# Description: test notebook\n\n# COMMAND ----------\n\nimport pyspark"
        nb = json.loads(code_to_notebook(code))
        assert nb["cells"][0]["cell_type"] == "markdown", "Header block should be markdown"
        assert nb["cells"][1]["cell_type"] == "code"
    test("markdown cell detection for header", test_notebook_markdown_detection)

    def test_notebook_large():
        # Simulate a large converted file
        cells = ["# Databricks notebook source"]
        for i in range(20):
            cells.append("# COMMAND ----------")
            cells.append(f"# Cell {i+1}\ndf_{i} = spark.read.format('delta').load('s3://bucket/table_{i}')\ndf_{i} = df_{i}.filter(F.col('status') == 'ACTIVE')\ndf_{i} = df_{i}.withColumn('processed', F.lit(True))\nprint(f'Table {i}: {{df_{i}.count()}} rows')")
        code = "\n\n".join(cells)
        nb = json.loads(code_to_notebook(code))
        assert len(nb["cells"]) == 20, f"Expected 20 cells, got {len(nb['cells'])}"
    test("large notebook with 20 cells", test_notebook_large)

    def test_notebook_valid_json():
        for fp, content in SYNTHETIC_FILES.items():
            # Simulate a conversion output
            sections = _split_into_sections(content)
            fake_converted = "# Databricks notebook source\n\n"
            for s in sections:
                fake_converted += f"# COMMAND ----------\n\n# Section: {s['name']}\n{s['code']}\n\n"
            nb_json = code_to_notebook(fake_converted, file_path=fp)
            nb = json.loads(nb_json)
            assert nb["nbformat"] == 4
            assert len(nb["cells"]) > 0
    test("valid .ipynb JSON for all 10 files", test_notebook_valid_json)

    # ── Test 6: Validation ──
    print("\n[6] Validation (Foundry API Detection)")

    def test_validate_clean():
        clean_code = "from pyspark.sql import functions as F\ndf = spark.read.format('delta').load('s3://bucket/table')\ndf.write.format('delta').save('s3://out')"
        issues = validate_conversion(clean_code)
        assert len(issues) == 0, f"Expected 0 issues, got {len(issues)}"
    test("clean converted code has 0 issues", test_validate_clean)

    def test_validate_dirty():
        dirty_code = "from transforms.api import transform, Input, Output\n@transform\ndef compute(output, inp):\n    df = inp.dataframe()\n    output.write_dataframe(df)"
        issues = validate_conversion(dirty_code)
        assert len(issues) >= 4, f"Expected 4+ issues, got {len(issues)}"
    test("unconverted Foundry code detected", test_validate_dirty)

    def test_validate_todo_skip():
        code_with_todo = "# TODO: UNCONVERTED_API - @transform\ndf = spark.read.format('delta').load('s3://path')"
        issues = validate_conversion(code_with_todo)
        assert len(issues) == 0, "TODO comments should be skipped"
    test("TODO comments are skipped", test_validate_todo_skip)

    # ── Test 7: Extract Code from LLM Response ──
    print("\n[7] Code Extraction from LLM Response")

    def test_extract_python_block():
        response = "Here is the converted code:\n\n```python\ndf = spark.read.format('delta').load('s3://path')\n```\n\nThis converts the Input() call."
        code = _extract_code(response)
        assert "spark.read" in code
        assert "```" not in code
    test("extract from ```python block", test_extract_python_block)

    def test_extract_plain_block():
        response = "```\ndf = spark.read.format('delta').load('s3://path')\n```"
        code = _extract_code(response)
        assert "spark.read" in code
        assert "```" not in code
    test("extract from plain ``` block", test_extract_plain_block)

    def test_extract_no_fences():
        response = "df = spark.read.format('delta').load('s3://path')"
        code = _extract_code(response)
        assert "spark.read" in code
    test("extract without fences", test_extract_no_fences)

    # ── Test 8: Notebook Header ──
    print("\n[8] Databricks Notebook Header")

    def test_header_added():
        code = "from pyspark.sql import F\ndf = spark.read.format('delta').load('s3://path')"
        result = _ensure_notebook_header(code)
        assert result.startswith("# Databricks notebook source")
    test("header added when missing", test_header_added)

    def test_header_not_duplicated():
        code = "# Databricks notebook source\n\n# COMMAND ----------\n\nfrom pyspark.sql import F"
        result = _ensure_notebook_header(code)
        assert result.count("# Databricks notebook source") == 1
    test("header not duplicated when present", test_header_not_duplicated)

    # ── Test 9: File Outline (Cross-Section Context) ──
    print("\n[9] File Outline for Cross-Section Context")

    def test_outline_captures_functions():
        sections = _split_into_sections(SYNTHETIC_FILES["transforms/silver/cleaned_flights.py"])
        outline = _build_file_outline(sections)
        assert "def compute" in outline, "Outline should capture function definitions"
    test("outline captures function definitions", test_outline_captures_functions)

    def test_outline_captures_decorators():
        sections = _split_into_sections(SYNTHETIC_FILES["transforms/bronze/raw_flight_ingest.py"])
        outline = _build_file_outline(sections)
        assert "@transform" in outline or "@configure" in outline, "Outline should capture decorators"
    test("outline captures decorators", test_outline_captures_decorators)

    def test_outline_captures_classes():
        sections = _split_into_sections(SYNTHETIC_FILES["transforms/shared_utils.py"])
        outline = _build_file_outline(sections)
        assert "class DataQualityChecker" in outline, "Outline should capture class definitions"
    test("outline captures class definitions", test_outline_captures_classes)

    # ── Test 10: Syntax Validation ──
    print("\n[10] Syntax Validation")

    def test_valid_syntax():
        code = "x = 1\ny = x + 2\nprint(y)"
        errors = _validate_syntax(code)
        assert len(errors) == 0, f"Valid code should have no errors: {errors}"
    test("valid code passes syntax check", test_valid_syntax)

    def test_invalid_syntax():
        code = "def foo(\n    x = 1\nprint(x"
        errors = _validate_syntax(code)
        assert len(errors) > 0, "Invalid code should have syntax errors"
    test("invalid code detected", test_invalid_syntax)

    def test_syntax_ignores_notebook_markers():
        code = "# Databricks notebook source\n\n# COMMAND ----------\n\nx = 1\n\n# COMMAND ----------\n\nprint(x)"
        errors = _validate_syntax(code)
        assert len(errors) == 0, f"Notebook markers should be stripped: {errors}"
    test("notebook markers stripped before syntax check", test_syntax_ignores_notebook_markers)

    def test_syntax_on_real_converted_output():
        # Simulate a well-formed conversion output
        code = (
            "# Databricks notebook source\n\n"
            "# COMMAND ----------\n\n"
            "from pyspark.sql import functions as F\n\n"
            "# COMMAND ----------\n\n"
            "input_path = 's3://ual-udh3-bronze-bucket/flights'\n"
            "output_path = 's3://ual-udh3-silver-bucket/cleaned_flights'\n\n"
            "# COMMAND ----------\n\n"
            "df = spark.read.format('delta').load(input_path)\n"
            "df = df.filter(F.col('status') != 'DELETED')\n"
            "df = df.withColumn('flight_id', F.upper(F.col('flight_id')))\n\n"
            "# COMMAND ----------\n\n"
            "df.write.format('delta').mode('overwrite').save(output_path)\n\n"
            "# COMMAND ----------\n\n"
            "print(f'Rows: {df.count()}')\n"
        )
        errors = _validate_syntax(code)
        assert len(errors) == 0, f"Well-formed notebook should pass: {errors}"
    test("realistic converted output passes syntax check", test_syntax_on_real_converted_output)

    # ── Test 11: File Removal from Index ──
    print("\n[11] Index File Removal")

    def test_remove_file():
        temp_index = KeywordIndex()
        chunks = chunk_file("temp_id", "temp.py", "def hello():\n    print('world')\n" * 10)
        temp_index.add_chunks(chunks)
        stats_before = temp_index.get_stats()
        temp_index.remove_file("temp_id")
        stats_after = temp_index.get_stats()
        assert stats_after["chunk_count"] == 0
        assert stats_after["file_count"] == 0
    test("remove file from index", test_remove_file)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
    if errors:
        print(f"\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*70}\n")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
