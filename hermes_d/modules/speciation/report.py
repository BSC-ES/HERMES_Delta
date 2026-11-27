"""Persistent, master-only diagnostics for one speciation run."""
import csv
from collections import Counter
from os import makedirs, path, remove

from pandas import isna

from hermes_d.config import log_message


class SpeciationReport:
    """Collect issues across sectors and append each issue immediately to a UTF-8 CSV.

    Create one instance per run and pass it to all sectors. Only the master may
    write or remove files; inventories are replicated, so no MPI gather is needed.
    Existing reports are removed at run start, preventing stale diagnostics when
    a new run is clean. A clean run creates no CSV. Validation remains fail-fast.
    """

    fields = ('severity', 'issue_type', 'sector', 'source_file', 'related_file',
              'profile_id', 'inventory_type', 'inventory_file', 'original_pollutant', 'component_id',
              'component_name', 'destination_species', 'total_weight_percent',
              'excluded_weight_percent', 'emission_amount_t_year', 'affected_records',
              'action', 'impact', 'message')

    def __init__(self, output_path, master=True):
        self.output_path = str(output_path)
        self.master = master
        self.issues = []
        if master and path.exists(self.output_path):
            remove(self.output_path)

    def add(self, **values):
        """Save one structured issue; irrelevant or missing values become blank cells."""
        if not self.master:
            return
        row = {field: '' if isna(values.get(field)) else values[field] for field in self.fields}
        self.issues.append(row)
        makedirs(path.dirname(path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, 'a', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            if len(self.issues) == 1:
                writer.writeheader()
            writer.writerow(row)

    def summary(self):
        """Log cumulative warning/error counts and the actual report location."""
        if self.master and self.issues:
            counts = Counter(row['severity'] for row in self.issues)
            log_message(f"Speciation report: {counts['warning']} warning issues, {counts['error']} error issues. "
                        f"Report: {self.output_path}", level=1)
