#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_Delta.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Analyze historical and current HERMES_Delta scalability timing logs."""

import argparse
import glob
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd

TIMING_STAT_COLUMNS = ["min", "mean", "max", "median"]
TIMING_SCHEMAS = ["auto", "legacy", "current"]

GROUPED_COLORS = {
    "Initialization": "#8EDDE8",
    "Calculation": "#FFE27A",
    "Writing": "#F3A0CF",
    "Other": "#BAB0AC",
}
TOTAL_LINE_STYLES = {
    "Initialization": {"marker": "s", "linestyle": "-"},
    "Calculation": {"marker": "^", "linestyle": "-"},
    "HERMES": {"marker": "o", "linestyle": "-"},
}
DETAIL_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]


class ScalabilityAnalysis:
    """Group timers on each MPI rank before rank statistics and run aggregation.

    Outlier filtering uses HERMES/TOTAL to keep or discard whole replications,
    preserving the same sample set across all categories of each process count.
    """

    def __init__(self, log_pattern, out_dir, replications=None, outlier_method=None,
                 outlier_threshold=1.5, aggregation_method="mean", min_replications_for_filtering=4,
                 timing_stat="max", timing_schema="auto"):
        if timing_stat not in TIMING_STAT_COLUMNS or timing_schema not in TIMING_SCHEMAS:
            raise ValueError("Unsupported timing statistic or schema")
        if aggregation_method not in {"mean", "median"}:
            raise ValueError("Unsupported aggregation method")
        if outlier_method not in {None, "none", "iqr", "mad", "zscore"}:
            raise ValueError("Unsupported outlier method")
        if outlier_threshold <= 0 or min_replications_for_filtering < 1:
            raise ValueError("Filtering threshold and minimum replication count must be positive")
        if replications is not None and replications < 1:
            raise ValueError("Replication count must be positive")
        self.out_dir = os.fspath(out_dir)
        self.logs = self.discover_logs(os.fspath(log_pattern), replications)
        self.timing_stat = timing_stat
        self.timing_schema = timing_schema
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.aggregation_method = aggregation_method
        self.min_replications_for_filtering = min_replications_for_filtering
        self.frames = None
        self.outlier_summary = None
        self.total_times = None

    @staticmethod
    def _compile_log_pattern(log_pattern):
        """Build a regular expression from a log pattern containing placeholders."""
        if "<processes>" not in log_pattern:
            raise ValueError("The log pattern must contain the <processes> placeholder.")

        escaped_pattern = re.escape(os.path.abspath(log_pattern))

        for placeholder, group_name in [
                ("<replication>", "replication"),
                ("<processes>", "processes")]:
            placeholder_pattern = re.escape(placeholder)
            replacement_count = 0

            def replace_placeholder(_match):
                nonlocal replacement_count
                replacement_count += 1
                if replacement_count == 1:
                    return r"(?P<{}>[^/]+)".format(group_name)
                return r"(?P={})".format(group_name)

            escaped_pattern = re.sub(
                placeholder_pattern,
                replace_placeholder,
                escaped_pattern,
            )
        return re.compile(r"^{}$".format(escaped_pattern))

    @classmethod
    def discover_logs(cls, log_pattern, replications=None):
        """
        Discover timing logs and extract their replication and process metadata.

        Parameters
        ----------
        log_pattern : str
            File path pattern. It must contain ``<processes>`` and can contain
            ``<replication>``.
        replications : int or None
            Optional number of expected integer replication folders.

        Returns
        -------
        list of dict
            One record per discovered file.
        """
        glob_pattern = log_pattern.replace("<replication>", "*").replace("<processes>", "*")
        log_paths = sorted(glob.glob(glob_pattern))
        pattern_regex = cls._compile_log_pattern(log_pattern)
        log_records = []

        for log_path in log_paths:
            abs_log_path = os.path.abspath(log_path)
            match = pattern_regex.match(abs_log_path)
            if match is None:
                continue

            process_value = match.group("processes")
            try:
                process_id = int(process_value)
            except ValueError:
                continue

            if process_id < 1:
                raise ValueError(f"Invalid process count in {log_path}")

            if "<replication>" in log_pattern:
                replication_value = match.group("replication")
                try:
                    replication_id = int(replication_value)
                except ValueError:
                    replication_id = replication_value
            else:
                replication_id = 1

            log_records.append({
                "path": abs_log_path,
                "replication": replication_id,
                "process": process_id,
            })

        if replications is not None:
            integer_replications = [
                log_record["replication"] for log_record in log_records
                if isinstance(log_record["replication"], int)
            ]
            first_replication = 0 if 0 in integer_replications else 1
            expected_replications = set(range(first_replication, first_replication + replications))
            log_records = [
                log_record for log_record in log_records
                if log_record["replication"] in expected_replications
            ]

        if not log_records:
            raise FileNotFoundError("No timing logs were found with the provided pattern.")

        keys = [(record["replication"], record["process"]) for record in log_records]
        if len(set(keys)) != len(keys):
            raise ValueError("More than one timing log for the same replication/process pair")
        return sorted(log_records, key=lambda item: (str(item["replication"]), item["process"], item["path"]))

    @staticmethod
    def _get_inlier_mask(values, outlier_method=None, outlier_threshold=1.5, min_replications_for_filtering=4):
        """Return a boolean mask indicating which replication values are kept."""
        valid_values = values.dropna()
        keep_mask = pd.Series(True, index=values.index, dtype=bool)

        if outlier_method in (None, "none") or valid_values.size < min_replications_for_filtering:
            return keep_mask

        if outlier_method not in {"iqr", "mad", "zscore"}:
            raise ValueError("Unsupported outlier method: {}".format(outlier_method))

        if outlier_method == "iqr":
            quantile_25 = valid_values.quantile(0.25)
            quantile_75 = valid_values.quantile(0.75)
            iqr_value = quantile_75 - quantile_25
            if iqr_value == 0:
                return keep_mask
            lower_bound = quantile_25 - outlier_threshold * iqr_value
            upper_bound = quantile_75 + outlier_threshold * iqr_value
            inlier_mask = (valid_values >= lower_bound) & (valid_values <= upper_bound)
        elif outlier_method == "mad":
            median_value = valid_values.median()
            mad_value = (valid_values - median_value).abs().median()
            if mad_value == 0:
                return keep_mask
            modified_zscore = 0.6745 * (valid_values - median_value).abs() / mad_value
            inlier_mask = modified_zscore <= outlier_threshold
        else:
            std_value = valid_values.std(ddof=0)
            if std_value == 0:
                return keep_mask
            zscore = ((valid_values - valid_values.mean()) / std_value).abs()
            inlier_mask = zscore <= outlier_threshold

        keep_mask.loc[valid_values.index] = inlier_mask
        return keep_mask

    def read_timing_log(self, record):
        """Read rank measurements, recomputing statistics rather than trusting old summaries."""
        frame = pd.read_csv(record['path'], index_col=['section', 'subsection'])
        if frame.empty or frame.index.has_duplicates:
            raise ValueError(f"Empty log or duplicate timers: {record['path']}")
        ranks = [str(rank) for rank in range(record['process'])]
        actual_ranks = [column for column in frame.columns if str(column).isdigit()]
        if set(actual_ranks) != set(ranks):
            raise ValueError(f"Expected ranks 0..{record['process'] - 1} in {record['path']}")
        frame = frame[ranks].apply(pd.to_numeric, errors='raise')
        if np.isinf(frame.to_numpy()).any() or (frame < 0).any().any() or frame.isna().all(axis=1).any():
            raise ValueError(f"Invalid rank timing measurements: {record['path']}")
        for section in ('Init', 'Calcul', 'Write', 'HERMES'):
            if (section, 'TOTAL') not in frame.index or frame.loc[(section, 'TOTAL')].isna().any():
                raise ValueError(f"Missing {section}/TOTAL rank measurements: {record['path']}")
        if (frame.loc[('HERMES', 'TOTAL')] <= 0).any():
            raise ValueError(f"Nonpositive execution total: {record['path']}")
        current = ('Write', 'write_time_step_emissions') in frame.index
        if self.timing_schema != 'auto' and (self.timing_schema == 'current') != current:
            raise ValueError(f"Log does not match schema {self.timing_schema}: {record['path']}")
        return frame

    def analyze(self):
        """Discover rank data and select whole-run inliers using the total execution time."""
        self.frames = [self.read_timing_log(record) for record in self.logs]
        self.keep = np.ones(len(self.logs), dtype=bool)
        reports = []
        for process in sorted({record['process'] for record in self.logs}):
            indices = [index for index, record in enumerate(self.logs) if record['process'] == process]
            totals = pd.Series({index: self._stat(self.frames[index].loc[('HERMES', 'TOTAL')])
                                for index in indices})
            mask = self._get_inlier_mask(totals, self.outlier_method, self.outlier_threshold,
                                         self.min_replications_for_filtering)
            if not mask.any():
                mask[:] = True
            for index in indices:
                self.keep[index] = mask[index]
                reports.append({'Process': process, 'Replication': self.logs[index]['replication'],
                                'path': self.logs[index]['path'], 'total': totals[index], 'kept': bool(mask[index]),
                                'outlier_method': self.outlier_method or 'none',
                                'aggregation_method': self.aggregation_method})
        self.outlier_summary = pd.DataFrame(reports)
        self.total_times = self._aggregate(self.frames)
        return self.total_times

    def _stat(self, series):
        return getattr(series, self.timing_stat)()

    def _aggregate(self, frames):
        columns = []
        keys = []
        for index, (record, frame) in enumerate(zip(self.logs, frames)):
            if self.keep[index]:
                columns.append(frame.apply(self._stat, axis=1))
                keys.append(record['process'])
        table = pd.concat(columns, axis=1)
        table.columns = keys
        return table.T.groupby(level=0).agg(self.aggregation_method).T.sort_index(axis=1)

    @staticmethod
    def _sum(frame, rows):
        rows = [row for row in rows if row in frame.index]
        return frame.loc[rows].sum(axis=0, min_count=1) if rows else pd.Series(0., index=frame.columns)

    @classmethod
    def _residual(cls, total, components, label):
        """Expose unmeasured time; do not invent a negative residual for overlapping timers."""
        residual = total - components
        if (residual < -1e-6).any():
            raise ValueError(f"Overlapping or inconsistent timers in {label}; use --summary-style raw to inspect")
        return residual.clip(lower=0)

    def _summary_frame(self, frame, style):
        """Build categories before applying any statistic across MPI ranks."""
        if style == 'raw':
            result = frame.copy()
            result.index = [f'{section}/{name}' for section, name in frame.index]
            return result
        totals = {label: frame.loc[(section, 'TOTAL')]
                  for label, section in [('Initialization', 'Init'), ('Calculation', 'Calcul'), ('Writing', 'Write')]}
        if style == 'grouped':
            rows = dict(totals)
        else:
            rows = {}
            sectors = list(frame.index.get_level_values('section').unique())
            reserved = {'Init', 'Calcul', 'Write', 'HERMES', 'EmissionInventory'}
            sectors = [section for section in sectors if section not in reserved]
            if style == 'simple':
                rows['Initialization'] = totals['Initialization']
            else:
                init_rows = [('Init', name) for name in ('Grid', 'Clip') if ('Init', name) in frame.index]
                for row in init_rows:
                    rows[f'Initialization/{row[1]}'] = frame.loc[row]
                for name in ('Read', 'Speciation', 'Vertical', 'Temporal', 'Horizontal'):
                    matching = [(sector, f'Init {name}') for sector in sectors
                                if (sector, f'Init {name}') in frame.index]
                    if matching:
                        rows[f'Initialization/{name}'] = self._sum(frame, matching)
                measured = sum(rows.values(), pd.Series(0., index=frame.columns))
                rows['Initialization/Other'] = self._residual(totals['Initialization'], measured, 'initialization')
            calc = {}
            for name in ('PointSourcesAllocation', 'Speciation', 'Vertical', 'Temporal', 'Horizontal'):
                matching = [(sector, name) for sector in sectors if (sector, name) in frame.index]
                if matching:
                    calc[name] = self._sum(frame, matching)
            if style == 'detailed':
                for name in ('UpdateSectorInfo', 'OutputUnits'):
                    matching = [(sector, name) for sector in sectors if (sector, name) in frame.index]
                    if matching:
                        calc[name] = self._sum(frame, matching)
                if ('Calcul', 'combine_sectors') in frame.index:
                    calc['CombineSectors'] = frame.loc[('Calcul', 'combine_sectors')]
            measured = sum(calc.values(), pd.Series(0., index=frame.columns))
            calc['Other'] = self._residual(totals['Calculation'], measured, 'calculation')
            rows.update({name if style == 'simple' and name != 'Other' else f'Calculation/{name}': value
                         for name, value in calc.items()})
            if style == 'simple':
                rows['Write'] = totals['Writing']
            else:
                # Prefer NES leaf timers; never stack append together with its children.
                names = ['prepare_output', 'unit_conversion', 'communication', 'netcdf_write',
                         'netcdf_write_unaccounted', 'close_output']
                if ('Write', 'unit_conversion') not in frame.index:
                    names = ['prepare_output', 'write_time_step_emissions', 'close_output']
                measured = pd.Series(0., index=frame.columns)
                for name in names:
                    if ('Write', name) in frame.index:
                        rows[f'Writing/{name}'] = frame.loc[('Write', name)]
                        measured += frame.loc[('Write', name)].fillna(0)
                rows['Writing/Other'] = self._residual(totals['Writing'], measured, 'writing')
        phase_sum = sum(totals.values(), pd.Series(0., index=frame.columns))
        rows['Other'] = self._residual(frame.loc[('HERMES', 'TOTAL')], phase_sum, 'HERMES total')
        rows['TOTAL'] = frame.loc[('HERMES', 'TOTAL')]
        return pd.DataFrame.from_dict(rows, orient='index')

    def summarize(self, style='auto'):
        """Simple uses historical Delta categories; auto/grouped uses three phases.

        Raw preserves inclusive parent and child timers and must be plotted as
        separate lines rather than stacked. All other styles use nonoverlapping
        rows on each rank, with residual time explicitly labeled Other.
        """
        aliases = {'auto': 'grouped', 'phase': 'grouped', 'detail': 'detailed'}
        style = aliases.get(style, style)
        if style not in {'simple', 'grouped', 'detailed', 'raw'}:
            raise ValueError(f'Unsupported summary style: {style}')
        if self.frames is None:
            self.analyze()
        return self._aggregate([self._summary_frame(frame, style) for frame in self.frames])

    def get_ordered_times(self):
        if self.total_times is None:
            self.analyze()
        return self.total_times

    @staticmethod
    def get_speedup(times):
        """Compute strong-scaling speedup as T(1) / T(p)."""
        if 1 not in times.columns:
            raise KeyError("Process count 1 is required to compute speedup and efficiency.")
        positive = times.where(times > 0)
        return positive.div(positive[1], axis=0).rdiv(1)

    @staticmethod
    def get_efficiency(times):
        """Compute parallel efficiency as speedup / p."""
        speedup = ScalabilityAnalysis.get_speedup(times)
        return speedup.div(speedup.columns, axis=1)

    @staticmethod
    def _load_pyplot():
        if "MPLCONFIGDIR" not in os.environ:
            os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="hermes_matplotlib_")
        if "XDG_CACHE_HOME" not in os.environ:
            os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="hermes_cache_")
        try:
            import matplotlib as mpl
        except ImportError as exc:
            raise ImportError("Plotting requires matplotlib; install it or use --skip-plots") from exc
        mpl.use("Agg")
        import matplotlib.pyplot as plt
        return plt

    @staticmethod
    def _is_grouped_categories(categories):
        return set(map(str, categories)).issubset(set(GROUPED_COLORS.keys()))

    @staticmethod
    def _get_phase_boundary_series(times, categories):
        boundaries = {}
        if "Initialization/TOTAL" in times.index:
            boundaries["Initialization"] = times.loc["Initialization/TOTAL"]
        if "Calculation/TOTAL" in times.index:
            calculation_total = times.loc["Calculation/TOTAL"]
            if "Initialization" in boundaries:
                calculation_total = boundaries["Initialization"] + calculation_total
            boundaries["Calculation"] = calculation_total
        return boundaries

    @classmethod
    def _get_category_colors(cls, categories):
        categories = list(categories)
        if cls._is_grouped_categories(categories):
            return {
                category: GROUPED_COLORS[str(category)]
                for category in categories
            }
        return {
            category: DETAIL_COLORS[idx % len(DETAIL_COLORS)]
            for idx, category in enumerate(categories)
        }

    @staticmethod
    def _blend_hex_color(hex_color, blend_with="#FFFFFF", factor=0.35):
        """Blend a hex color with another color."""
        color = hex_color.lstrip("#")
        other = blend_with.lstrip("#")
        rgb = [int(color[idx:idx + 2], 16) for idx in range(0, 6, 2)]
        other_rgb = [int(other[idx:idx + 2], 16) for idx in range(0, 6, 2)]
        blended = [
            int(round(component * (1.0 - factor) + other_component * factor))
            for component, other_component in zip(rgb, other_rgb)
        ]
        return "#{:02X}{:02X}{:02X}".format(*blended)

    @classmethod
    def _get_fill_and_edge_colors(cls, color):
        return (
            cls._blend_hex_color(color, blend_with="#FFFFFF", factor=0.25),
            cls._blend_hex_color(color, blend_with="#000000", factor=0.45),
        )

    @classmethod
    def plot_times(cls, times, output_path, title=None):
        """Generate a stacked execution-time figure."""
        plt = cls._load_pyplot()
        fig, ax = plt.subplots(figsize=(13.5, 8.5))
        categories = [
            category for category in times.index
            if category != "TOTAL" and not str(category).endswith("/TOTAL")
        ]
        x_values = list(times.columns)
        accumulated = pd.Series(0.0, index=times.columns)
        colors = cls._get_category_colors(categories)

        for category in categories:
            upper = accumulated + times.loc[category]
            fill_color, edge_color = cls._get_fill_and_edge_colors(colors[category])
            ax.fill_between(
                x_values, accumulated.values, upper.values,
                color=fill_color, alpha=0.82, edgecolor=edge_color, linewidth=0.7, label=category)
            accumulated = upper

        if not cls._is_grouped_categories(categories):
            for phase, phase_boundary in cls._get_phase_boundary_series(times, categories).items():
                line_style = TOTAL_LINE_STYLES[phase]
                ax.plot(
                    x_values,
                    phase_boundary,
                    marker=line_style["marker"],
                    color="black",
                    linestyle=line_style["linestyle"],
                    linewidth=1.4,
                    label="{} TOTAL".format(phase),
                )

        total_label = None
        for candidate in ["TOTAL", "HERMES/TOTAL"]:
            if candidate in times.index:
                total_label = candidate
                break

        if total_label is not None:
            line_style = TOTAL_LINE_STYLES["HERMES"]
            ax.plot(
                x_values,
                times.loc[total_label],
                marker=line_style["marker"],
                color="black",
                linestyle=line_style["linestyle"],
                linewidth=1.7,
                label="HERMES TOTAL",
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(value) for value in x_values])
        ax.set_xlabel("MPI processes")
        ax.set_ylabel("Execution time (s)")
        ax.set_title(title or "HERMES scalability timing analysis")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    @classmethod
    def plot_section_times(cls, times, output_path, section, title_suffix=""):
        """Generate an execution-time plot for the detailed timers of one section."""
        if times.empty:
            return False

        plot_times = times.drop(index="TOTAL", errors="ignore")
        if plot_times.empty:
            return False

        cls.plot_times(
            plot_times,
            output_path,
            title="HERMES {} timing detail{}".format(section.lower(), title_suffix),
        )
        return True

    @classmethod
    def plot_metric(cls, metric, output_path, ylabel, title=None, ideal=None):
        """Generate a line plot for a scalability metric."""
        plt = cls._load_pyplot()
        fig, ax = plt.subplots(figsize=(13.5, 8.5))
        x_values = list(metric.columns)
        categories = [
            category for category in metric.index
            if category != "TOTAL" and not str(category).endswith("/TOTAL")
        ]
        colors = cls._get_category_colors(categories)

        for category in categories:
            ax.plot(x_values, metric.loc[category], marker="o", color=colors[category], label=category)

        total_label = None
        for candidate in ["TOTAL", "HERMES/TOTAL"]:
            if candidate in metric.index:
                total_label = candidate
                break

        if total_label is not None:
            line_style = TOTAL_LINE_STYLES["HERMES"]
            ax.plot(
                x_values,
                metric.loc[total_label],
                marker=line_style["marker"],
                color="black",
                linestyle=line_style["linestyle"],
                linewidth=1.7,
                label="HERMES TOTAL",
            )

        if ideal == "speedup":
            ax.plot(x_values, x_values, linestyle="--", color="red", label="Ideal")
        elif ideal == "efficiency":
            ax.axhline(1.0, linestyle="--", color="red", label="Ideal")

        ax.set_xscale("log", base=2)
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(value) for value in x_values])
        ax.set_xlabel("MPI processes")
        ax.set_ylabel(ylabel)
        ax.set_title(title or ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    def write_outputs(self, summary_style='auto', env=None, write_plots=True):
        """Write GR-compatible CSV outputs and figures with Delta categories."""
        times = self.summarize(summary_style)
        os.makedirs(self.out_dir, exist_ok=True)
        outputs = []
        tables = {'Times_Summary.csv': times, 'Times_Detail.csv': self.total_times,
                  'Outlier_Summary.csv': self.outlier_summary}
        try:
            speedup = self.get_speedup(times)
            efficiency = self.get_efficiency(times)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            speedup = efficiency = None
        else:
            tables.update({'Speedup_Summary.csv': speedup, 'Efficiency_Summary.csv': efficiency})
        for name, table in tables.items():
            path = os.path.join(self.out_dir, name)
            table.to_csv(path, index=name != 'Outlier_Summary.csv')
            outputs.append(path)
        if write_plots:
            suffix = f' ({env})' if env else ''
            path = os.path.join(self.out_dir, 'HERMES_Times.png')
            if summary_style == 'raw':
                self.plot_metric(times, path, 'Execution time (s)', title=f'HERMES_Delta raw timers{suffix}')
            else:
                self.plot_times(times, path, title=f'HERMES_Delta execution times{suffix}')
            outputs.append(path)
            if speedup is not None:
                for metric, label, ideal in [(speedup, 'Speedup', 'speedup'),
                                             (efficiency, 'Efficiency', 'efficiency')]:
                    path = os.path.join(self.out_dir, f'HERMES_{label}.png')
                    self.plot_metric(metric, path, label, title=f'HERMES_Delta {label.lower()}{suffix}', ideal=ideal)
                    outputs.append(path)
        return outputs


def get_arguments(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze HERMES_Delta scalability timing logs.")
    parser.add_argument(
        "--log-pattern", required=True,
        help="Timing log pattern with <processes> and optionally <replication> placeholders.")
    parser.add_argument(
        "--out-dir", required=True,
        help="Directory where CSV summaries and plots will be written.")
    parser.add_argument(
        "--replications", type=int, default=None,
        help="Optional number of expected integer replications. Uses 0-based filtering when replication 0 exists.")
    parser.add_argument(
        "--outlier-method", choices=["none", "iqr", "mad", "zscore"], default="none",
        help="Optional outlier filtering method.")
    parser.add_argument(
        "--outlier-threshold", type=float, default=1.5,
        help="Threshold used by the selected outlier method.")
    parser.add_argument(
        "--aggregation-method", choices=["mean", "median"], default="mean",
        help="Method used to aggregate replications.")
    parser.add_argument(
        "--timing-stat", choices=TIMING_STAT_COLUMNS, default="max",
        help="Timing statistic read from each HERMES time log.")
    parser.add_argument(
        "--timing-schema", choices=TIMING_SCHEMAS, default="auto",
        help="Delta log schema: auto, legacy (v1.0.1) or current (detailed writing timers).")
    parser.add_argument(
        "--min-replications-for-filtering", type=int, default=4,
        help="Minimum valid replications required before filtering outliers.")
    parser.add_argument(
        "--summary-style", choices=["auto", "simple", "phase", "grouped", "detailed", "detail", "raw"],
        default="auto",
        help="Timing summary style used for output tables and plots.")
    parser.add_argument(
        "--env", default=None,
        help="Optional environment name to include in plot titles.")
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Only write CSV outputs.")
    return parser.parse_args(args)


def run(args=None):
    """Execute the scalability-analysis CLI."""
    parsed_args = get_arguments(args=args)
    outlier_method = None if parsed_args.outlier_method == "none" else parsed_args.outlier_method
    analysis = ScalabilityAnalysis(
        parsed_args.log_pattern,
        parsed_args.out_dir,
        replications=parsed_args.replications,
        outlier_method=outlier_method,
        outlier_threshold=parsed_args.outlier_threshold,
        aggregation_method=parsed_args.aggregation_method,
        min_replications_for_filtering=parsed_args.min_replications_for_filtering,
        timing_stat=parsed_args.timing_stat,
        timing_schema=parsed_args.timing_schema,
    )
    output_files = analysis.write_outputs(
        summary_style=parsed_args.summary_style,
        env=parsed_args.env,
        write_plots=not parsed_args.skip_plots,
    )

    for output_file in output_files:
        print(output_file)


if __name__ == "__main__":
    run()
